#!/usr/bin/env python3
# Entry point for GRPO training of the Cosmos Predict 2 Point Adapter.
#
# Usage example
# -------------
#   cd /path/to/workspace   # repo root (cosmos-predict2.5)
#
#   python ../cosmos_grpo/train.py \
#       --config cosmos_predict2._src.predict2.configs.video2world.config \
#       +experiment=predict2_point_adapter_training_2b_cosmos_so100_point \
#       --base-ckpt checkpoints/Cosmos-Predict2-2B-Video2World/model.pt \
#       [--adapter-ckpt cosmos_grpo/outputs/point_adapter_iter000500.pt] \
#       [--max-iter 5000] [--group-size 4] [--lr 3e-5] \
#       [--dryrun]
#
# The script reuses the Hydra / Imaginaire config infrastructure to instantiate
# the model and dataloader exactly as in supervised fine-tuning.  After loading
# the model, it hands control to GRPOTrainer which runs the RL loop.

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from importlib.util import find_spec

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

# Ensure the repo root (cosmos-predict2.5) is importable when running the script
# directly from the cosmos_grpo directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.isdir(_REPO_ROOT) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import inspect
import torch.distributed as dist

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.utils.model_loader import (
    load_model_state_dict_from_checkpoint,
)

from cosmos_grpo.config.grpo_so100_point_adapter import CosmosGRPOConfig
from cosmos_grpo.cosmos_grpo import GRPOTrainer

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s %(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cosmos_grpo.train")


def _check_online_reward_dependencies() -> None:
    """Validate imports needed by the online SAM+Depth+ICP reward path."""
    required_modules = [
        "sam2.build_sam",
        "depth_anything_3.api",
        "open3d",
    ]
    missing = [m for m in required_modules if find_spec(m) is None]
    if missing:
        raise RuntimeError(
            "Missing required modules for online reward pipeline: "
            + ", ".join(missing)
        )

    if os.system("which ffmpeg >/dev/null 2>&1") != 0:
        raise RuntimeError("ffmpeg is required for online reward pipeline but was not found in PATH.")


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_model(config, base_ckpt: str | None, grpo_cfg=None):
    """Instantiate the model and load weights from a consolidated checkpoint.

    Args:
        config: fully resolved Imaginaire / Hydra config object.
        base_ckpt: path to the ``.pt`` consolidated checkpoint; overrides
            ``config.checkpoint.load_path`` when provided.
        grpo_cfg: optional CosmosGRPOConfig; when provided, components that
            will be offloaded (text_encoder, tokenizer) are kept on CPU instead
            of being moved to CUDA during initialisation.

    Returns:
        model: ``Text2WorldModelRectifiedFlow`` on CUDA, Point Adapter unfrozen.
    """
    # Force fsdp_shard_size=1 for single-GPU RL training
    config.model.config.fsdp_shard_size = 1

    logger.info("Instantiating model …")
    model = instantiate(config.model)

    # ── Selective CUDA placement ──────────────────────────────────────────────
    # model.net is already on CUDA (build_net() uses to_empty("cuda")).
    # model.net_ema is on CPU by design (keep_on_cpu=True for fsdp_shard_size=1)
    # and must stay there for GRPO (apply_fsdp is never called).
    # model.text_encoder is on CUDA (TextEncoder.__init__ calls to_empty("cuda")).
    # model.tokenizer / conditioner are on CPU initially.
    #
    # GPU budget on a 32 GB card:
    #   net (2B, bf16)           ≈  4 GB  (already on CUDA)
    #   text_encoder (7B VLM)   ≈ 16 GB  (already on CUDA)
    #   net_ema copy             ≈  4 GB  (on CPU, .cuda() would move it)
    #   tokenizer VAE            ≈  0.5 GB (on CPU, .cuda() would move it)
    # Total without offloads    ≈ 24.5 GB + CUDA overhead → OOM during .cuda()
    #
    # Strategy:
    #  1. Move text_encoder to CPU before .cuda() (if offload_text_encoder flag).
    #  2. Stash net_ema + optionally text_encoder/tokenizer as None so .cuda()
    #     skips them.
    #  3. Call model.cuda() on the remaining (small) components.
    #  4. Re-attach stashed modules → they remain on CPU.
    _cpu_stash: dict = {}

    # text_encoder: move from CUDA → CPU first (it was placed on CUDA during init).
    if grpo_cfg is not None and grpo_cfg.offload_text_encoder:
        if getattr(model, "text_encoder", None) is not None:
            logger.info("Offloading text encoder to CPU to free CUDA memory …")
            model.text_encoder.model = model.text_encoder.model.to("cpu")
            torch.cuda.empty_cache()
            _cpu_stash["text_encoder"] = model.text_encoder
            model.text_encoder = None

    # tokenizer: on CPU already; just stash it if offloading.
    if grpo_cfg is not None and grpo_cfg.offload_tokenizer:
        if getattr(model, "tokenizer", None) is not None:
            _cpu_stash["tokenizer"] = model.tokenizer
            model.tokenizer = None

    # Always keep net_ema on CPU for GRPO (gradient updates use model.net only).
    if getattr(model, "net_ema", None) is not None:
        _cpu_stash["net_ema"] = model.net_ema
        model.net_ema = None

    # Move the remaining (small) sub-modules to CUDA.
    model.cuda()

    # Restore stashed CPU modules.
    for _attr, _mod in _cpu_stash.items():
        setattr(model, _attr, _mod)
    # ─────────────────────────────────────────────────────────────────────────

    ckpt_path = base_ckpt or config.checkpoint.load_path
    if ckpt_path:
        if not os.path.exists(str(ckpt_path)):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Pass --base-ckpt <path> or set config.checkpoint.load_path."
            )
        logger.info(f"Loading weights from: {ckpt_path}")
        # Override config checkpoint path so load_model_state_dict_from_checkpoint
        # uses the right path.
        config.checkpoint.load_path = str(ckpt_path)
        model = load_model_state_dict_from_checkpoint(
            model=model,
            config=config,
            s3_checkpoint_dir=str(ckpt_path),
        )
    else:
        logger.warning(
            "No checkpoint path provided. Model weights are randomly initialised!"
        )

    # Verify Point Adapter freeze status (set_up_model already does this, but
    # double-check here in case the ckpt loading changed requires_grad flags).
    n_trainable = 0
    n_total = 0
    for name, param in model.net.named_parameters():
        n_total += param.numel()
        if param.requires_grad:
            if "point_adapter" not in name:
                # Safety: freeze any stray non-adapter parameters
                param.requires_grad_(False)
                logger.warning(f"Freezing non-adapter param: {name}")
            else:
                n_trainable += param.numel()

    logger.info(
        f"Trainable (Point Adapter) params: {n_trainable/1e6:.2f}M / "
        f"{n_total/1e6:.1f}M total ({100*n_trainable/n_total:.2f}%)"
    )
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO training for Cosmos Predict 2 Point Adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Hydra / Imaginaire config -----------------------------------------
    parser.add_argument(
        "--config",
        default="cosmos_predict2._src.predict2.configs.video2world.config",
        help="Dotted path to the Python config module (make_config() is called).",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help=(
            "Config overrides in key=value format, e.g.: "
            "+experiment=predict2_point_adapter_training_2b_cosmos_so100_point"
        ),
    )

    # ---- Checkpoints --------------------------------------------------------
    parser.add_argument("--base-ckpt",    default=None,
                        help="Path to the consolidated base model checkpoint (.pt).")
    parser.add_argument("--adapter-ckpt", default=None,
                        help="Optional Point Adapter .pt to resume from.")

    # ---- GRPO hyperparameters (override CosmosGRPOConfig defaults) ----------
    parser.add_argument("--max-iter",    type=int,   default=None,
                        help="Total RL iterations.")
    parser.add_argument("--group-size",  type=int,   default=None,
                        help="Number of rollout samples per condition (G).")
    parser.add_argument("--lr",          type=float, default=None,
                        help="Learning rate for Point Adapter AdamW.")
    parser.add_argument("--num-steps",   type=int,   default=None,
                        help="Diffusion denoising steps during rollout.")
    parser.add_argument("--save-every",  type=int,   default=None,
                        help="Save checkpoint every N iterations.")
    parser.add_argument("--log-every",   type=int,   default=None,
                        help="Log training metrics every N iterations.")
    parser.add_argument("--output-dir",  default=None,
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--debug-save-intermediates", action="store_true",
                        help="Save rollout/reward intermediate debug artifacts.")
    parser.add_argument("--debug-save-every", type=int, default=None,
                        help="Save debug artifacts every N iterations.")
    parser.add_argument("--debug-max-videos", type=int, default=None,
                        help="Max rollout videos dumped per iter for debugging.")
    parser.add_argument("--debug-dir", default=None,
                        help="Directory to store debug artifacts.")

    # ---- Memory offloading --------------------------------------------------
    parser.add_argument("--offload-diffusion-model", action="store_true",
                        help="Offload diffusion net to CPU between encode and decode steps.")
    parser.add_argument("--offload-text-encoder", action="store_true",
                        help="Offload T5 text encoder to CPU after computing embeddings.")
    parser.add_argument("--offload-tokenizer", action="store_true",
                        help="Offload tokenizer encoder/decoder to CPU when not in use.")
    parser.add_argument("--offload-model-for-reward", action="store_true",
                        help="Move entire Cosmos model to CPU before SAM+Depth+ICP reward to free VRAM.")

    # ---- Misc ---------------------------------------------------------------
    parser.add_argument("--dryrun", action="store_true",
                        help="Load model and dataloader, then exit without training.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _init_distributed() -> None:
    """Initialize torch.distributed + Megatron parallel_state when launched via torchrun.

    torchrun sets LOCAL_RANK / RANK / WORLD_SIZE env vars.  When running with plain
    ``python -m`` those vars are absent (LOCAL_RANK defaults to 0 but WORLD_SIZE to 1),
    so we only enter the full init path when WORLD_SIZE > 1 OR when MASTER_ADDR is set
    (torchrun always sets MASTER_ADDR).
    """
    if not os.getenv("MASTER_ADDR"):
        # Plain python launch — no distributed init needed.
        return

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    logger.info(
        f"torch.distributed initialized: rank={dist.get_rank()}, "
        f"world_size={dist.get_world_size()}, local_rank={local_rank}"
    )

    # Megatron parallel_state (needed by DistributedSampler + model sharding)
    try:
        from megatron.core import parallel_state
        USE_MEGATRON = True
    except ImportError:
        USE_MEGATRON = False

    if USE_MEGATRON and not parallel_state.model_parallel_is_initialized():
        kwargs = dict(
            pipeline_model_parallel_size=1,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        )
        if "create_gloo_process_groups" in inspect.signature(
            parallel_state.initialize_model_parallel
        ).parameters:
            kwargs["create_gloo_process_groups"] = False
        parallel_state.initialize_model_parallel(**kwargs)
        logger.info("Megatron parallel_state initialized (1×1×1).")


def main():
    args = parse_args()
    _init_distributed()
    _check_online_reward_dependencies()

    # ---- 1. Build GRPO config -----------------------------------------------
    grpo_cfg = CosmosGRPOConfig()
    if args.max_iter    is not None: grpo_cfg.max_iter             = args.max_iter
    if args.group_size  is not None: grpo_cfg.group_size           = args.group_size
    if args.lr          is not None: grpo_cfg.lr                   = args.lr
    if args.num_steps   is not None: grpo_cfg.num_diffusion_steps  = args.num_steps
    if args.save_every  is not None: grpo_cfg.save_every           = args.save_every
    if args.log_every   is not None: grpo_cfg.log_every            = args.log_every
    if args.output_dir  is not None: grpo_cfg.output_dir           = args.output_dir
    if args.base_ckpt   is not None: grpo_cfg.base_model_checkpoint = args.base_ckpt
    if args.adapter_ckpt is not None: grpo_cfg.point_adapter_checkpoint = args.adapter_ckpt
    if args.debug_save_intermediates: grpo_cfg.debug_save_intermediates = True
    if args.debug_save_every is not None: grpo_cfg.debug_save_every = args.debug_save_every
    if args.debug_max_videos is not None: grpo_cfg.debug_max_videos_per_iter = args.debug_max_videos
    if args.debug_dir is not None: grpo_cfg.debug_dir = args.debug_dir
    if args.offload_diffusion_model: grpo_cfg.offload_diffusion_model = True
    if args.offload_text_encoder: grpo_cfg.offload_text_encoder = True
    if args.offload_tokenizer: grpo_cfg.offload_tokenizer = True
    if args.offload_model_for_reward: grpo_cfg.offload_model_for_reward = True

    logger.info(f"CosmosGRPOConfig: {grpo_cfg}")

    # ---- 2. Build Imaginaire / Hydra config ---------------------------------
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, list(args.opts))

    # ---- 3. Load model ------------------------------------------------------
    resolved_base_ckpt = args.base_ckpt or grpo_cfg.base_model_checkpoint
    model = load_model(config, base_ckpt=resolved_base_ckpt, grpo_cfg=grpo_cfg)

    # ---- 4. Optionally load a previously-saved Point Adapter ----------------
    start_iter = 0
    if args.adapter_ckpt:
        start_iter = GRPOTrainer.load_point_adapter(model, args.adapter_ckpt)

    # ---- 5. Build dataloader ------------------------------------------------
    logger.info("Building dataloader …")
    dataloader_train = instantiate(config.dataloader_train)

    # ---- 6. Dry-run check ---------------------------------------------------
    if args.dryrun:
        logger.info(
            "Dry-run: model and dataloader are ready. Exiting without training."
        )
        # Print a quick summary
        logger.info(f"  Model type : {type(model).__name__}")
        logger.info(f"  Dataloader : {type(dataloader_train).__name__}")
        logger.info(f"  GRPO cfg   : group_size={grpo_cfg.group_size}, "
                    f"max_iter={grpo_cfg.max_iter}, lr={grpo_cfg.lr:.2e}")
        return

    # ---- 7. Launch GRPO training --------------------------------------------
    trainer = GRPOTrainer(grpo_cfg)
    trainer.train(model, dataloader_train, start_iter=start_iter)


if __name__ == "__main__":
    main()
