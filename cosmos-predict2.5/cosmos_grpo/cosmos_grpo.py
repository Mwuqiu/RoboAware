# GRPOTrainer for Cosmos Predict 2 – Point Adapter fine-tuning via RL.
#
# Algorithm (GRPO / reward-weighted regression variant)
# --------------------------------------------------------
# For each training iteration:
#
#   1. ROLLOUT  – generate G videos by running generate_samples_from_batch
#                 G times with different seeds (model.eval, no_grad).
#   2. DECODE   – decode all G latents to pixel-space via model.tokenizer.
#   3. REWARD   – call reward.compute_rewards(videos, data_batch) → [G] floats.
#   4. ADVANTAGE – group-normalise rewards:
#                   A_i = (R_i - mean(R)) / (std(R) + eps), then clip.
#   5. UPDATE   – for each group member i (repeat num_update_epochs times):
#                    a. sample random noise timestep t
#                    b. add noise to x0_latent_i → x_t
#                    c. run model.denoise(eps, x_t, t, condition)   → v_pred
#                    d. compute flow-matching target v_target = eps - x0
#                    e. GRPO loss = mean(-A_i * ||v_pred - v_target||²)
#                    f. backward + optimizer step (Point Adapter only)
#   6. CHECKPOINT & LOG every N iterations.
#
# Only Point Adapter parameters receive gradients; the base Cosmos model is
# kept frozen throughout.  FSDP and multi-GPU support are intentionally
# excluded here to keep the code readable – use the standard ImaginaireTrainer
# for large-scale distributed training.

from __future__ import annotations

import copy
import logging
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldModelRectifiedFlow,
)

from .config.grpo_so100_point_adapter import CosmosGRPOConfig
from .reward import (
    compute_rewards,
    configure_reward_engine,
    get_last_reward_diagnostics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device | str
) -> Dict[str, torch.Tensor]:
    """Move all tensor values in *batch* to *device* (in-place clone)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Shallow-clone a data-batch dict (detach tensors so gradients don't leak)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone().detach()
        else:
            out[k] = copy.copy(v) if not isinstance(v, (str, int, float, bool)) else v
    return out


def _ensure_text_conditioning(
    model: Text2WorldModelRectifiedFlow,
    data_batch: Dict[str, torch.Tensor],
) -> None:
    """Populate online text conditioning fields when needed.

    Sampling/update paths in GRPO do not go through model.forward(), so we
    explicitly mirror forward()'s online text embedding step here.
    """
    if not (
        model.config.text_encoder_config is not None
        and model.config.text_encoder_config.compute_online
    ):
        return

    # Match cross-attention projection parameter dtype/device to avoid
    # linear matmul dtype mismatch (e.g. bf16 input vs fp32 weights).
    try:
        ref_param = next(model.net.crossattn_proj.parameters())
    except Exception:
        ref_param = next(model.net.parameters())
    target_device = ref_param.device
    target_dtype = ref_param.dtype

    emb = data_batch.get("t5_text_embeddings")    
    if emb is None:
        emb = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
    data_batch["t5_text_embeddings"] = emb.to(device=target_device, dtype=target_dtype)

    mask = data_batch.get("t5_text_mask")
    if mask is None:
        mask = torch.ones(emb.shape[0], emb.shape[1], device=target_device, dtype=torch.bool)
    else:
        mask = mask.to(device=target_device)
    data_batch["t5_text_mask"] = mask


def _ensure_point_conditioning_dtype(
    model: Text2WorldModelRectifiedFlow,
    data_batch: Dict[str, torch.Tensor],
) -> None:
    """Align point-cloud latent dtype/device with Point Adapter weights."""
    pc = data_batch.get("pc_latent_x0")
    if pc is None:
        return
    try:
        ref_param = next(model.net.point_adapter.pc_encoder.parameters())
    except Exception:
        ref_param = next(model.net.parameters())
    data_batch["pc_latent_x0"] = pc.to(device=ref_param.device, dtype=ref_param.dtype)


def _validate_conditioning_keys(data_batch: Dict[str, torch.Tensor]) -> None:
    """Fail fast with a clear error when required conditioning tensors are missing."""
    required = ["pc_latent_x0", "pc_latent_mask", "t5_text_embeddings", "t5_text_mask"]
    missing = [k for k in required if data_batch.get(k) is None]
    if missing:
        raise RuntimeError(
            "Missing conditioning fields in batch: "
            f"{missing}. Available keys: {sorted(list(data_batch.keys()))}"
        )


# ---------------------------------------------------------------------------
# GRPOTrainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """Lightweight GRPO trainer for the Cosmos Point Adapter.

    Does NOT depend on ImaginaireTrainer or Hydra – it receives an already-
    initialised :class:`Text2WorldModelRectifiedFlow` and a plain PyTorch
    dataloader.
    """

    def __init__(self, config: CosmosGRPOConfig) -> None:
        self.cfg = config
        os.makedirs(config.output_dir, exist_ok=True)
        configure_reward_engine(config)
        self._wandb_run = None
        self._init_wandb()

    # -----------------------------------------------------------------------
    # W&B helpers
    # -----------------------------------------------------------------------

    def _init_wandb(self) -> None:
        cfg = self.cfg
        if not cfg.wandb_enabled:
            return
        if not _WANDB_AVAILABLE:
            logger.warning(
                "[GRPOTrainer] wandb_enabled=True but 'wandb' package is not installed. "
                "Install it with: pip install wandb"
            )
            return
        import dataclasses
        run_kwargs: dict = dict(
            project=cfg.wandb_project,
            config=dataclasses.asdict(cfg),
            tags=cfg.wandb_tags or None,
            notes=cfg.wandb_notes or None,
            reinit=True,
        )
        if cfg.wandb_run_name:
            run_kwargs["name"] = cfg.wandb_run_name
        self._wandb_run = _wandb.init(**run_kwargs)
        logger.info(f"[GRPOTrainer] W&B run initialised: {self._wandb_run.url}")

    def _watch_wandb(self, model) -> None:
        """Call wandb.watch on Point Adapter parameters (once, after model is ready)."""
        if self._wandb_run is None or not self.cfg.wandb_watch_model:
            return
        adapter_modules = [
            m for n, m in model.net.named_modules()
            if "point_adapter" in n
        ]
        if adapter_modules:
            _wandb.watch(adapter_modules[0], log="all", log_freq=self.cfg.log_every)
            logger.info("[GRPOTrainer] wandb.watch() registered on point_adapter")

    def _finish_wandb(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    # -----------------------------------------------------------------------
    # Phase 1 – Rollout
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Offload helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _offload_text_encoder(model: Text2WorldModelRectifiedFlow) -> None:
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to("cpu")
            else:
                model.text_encoder = model.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    @staticmethod
    def _offload_tokenizer(model: Text2WorldModelRectifiedFlow) -> None:
        if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
            model.tokenizer.encoder = model.tokenizer.encoder.to("cpu")
        if hasattr(model.tokenizer, "decoder") and model.tokenizer.decoder is not None:
            model.tokenizer.decoder = model.tokenizer.decoder.to("cpu")
        torch.cuda.empty_cache()

    @staticmethod
    def _load_tokenizer_encoder(model: Text2WorldModelRectifiedFlow) -> None:
        if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
            model.tokenizer.encoder = model.tokenizer.encoder.to("cuda")
        torch.cuda.empty_cache()

    @staticmethod
    def _load_tokenizer_decoder(model: Text2WorldModelRectifiedFlow) -> None:
        if hasattr(model.tokenizer, "decoder") and model.tokenizer.decoder is not None:
            model.tokenizer.decoder = model.tokenizer.decoder.to("cuda")
        torch.cuda.empty_cache()

    @staticmethod
    def _offload_diffusion_net(model: Text2WorldModelRectifiedFlow) -> None:
        model.net = model.net.to("cpu")
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to("cpu")
        torch.cuda.empty_cache()

    @staticmethod
    def _load_diffusion_net(model: Text2WorldModelRectifiedFlow) -> None:
        model.net = model.net.to("cuda")
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to("cuda")
        torch.cuda.empty_cache()

    def offload_model_for_reward(self, model: Text2WorldModelRectifiedFlow) -> None:
        """Free CPU/GPU memory of components no longer needed before reward.

        Instead of offloading to CPU (which fills system RAM and risks OOM-kill),
        we permanently DELETE the large components whose job is already done:
          - text_encoder  (~14-16 GB on CPU)  – embeddings computed during rollout
          - net_ema       (~4 GB on CPU)       – not used in GRPO updates
          - tokenizer     (~0.5 GB on CPU)     – encode/decode already done
          - conditioner   (GPU, small)          – conditioning cached in data_batch
        model.net stays on GPU so the update step can still backprop through
        point_adapter without reloading the checkpoint.
        """
        if not self.cfg.offload_model_for_reward:
            return
        import gc
        logger.info("[GRPOTrainer] Destroying completed model components to free memory for reward")
        if getattr(model, "text_encoder", None) is not None:
            del model.text_encoder
            model.text_encoder = None
        if getattr(model, "net_ema", None) is not None:
            del model.net_ema
            model.net_ema = None
        if getattr(model, "tokenizer", None) is not None:
            del model.tokenizer
            model.tokenizer = None
        if getattr(model, "conditioner", None) is not None:
            model.conditioner.cpu()
            del model.conditioner
            model.conditioner = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("[GRPOTrainer] Memory freed. model.net remains on GPU for update step.")

    def reload_model_after_reward(self, model: Text2WorldModelRectifiedFlow) -> None:
        """No-op: components were deleted (not offloaded), model.net is still on GPU."""
        pass

    @torch.no_grad()
    def rollout_phase(
        self,
        model: Text2WorldModelRectifiedFlow,
        data_batch: Dict[str, torch.Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate G video latents from *data_batch*.

        Returns:
            latents: list of G tensors, each ``[B, C, T, H, W]`` float32.
            videos:  list of G decoded pixel-space tensors, same spatial shape,
                     values in ``[-1, 1]``, for reward computation.  Kept on CPU
                     to save GPU memory.
        """
        model.eval()
        G = self.cfg.group_size
        use_amp = torch.cuda.is_available()
        amp_dtype = torch.bfloat16

        latents: List[Tensor] = []
        videos:  List[Tensor] = []

        # Each call to generate_samples_from_batch modifies data_batch in-place
        # (normalisation flags), so we work with a single batch but differ seeds.
        for i in range(G):
            seed = int(torch.randint(0, 2**31, (1,)).item()) + i

            # ---- Step 1: ensure text embeddings (text encoder may be offloaded) ----
            # Reload text encoder to GPU if offloading is enabled so that online
            # T5 embedding computation succeeds.
            if self.cfg.offload_text_encoder and model.text_encoder is not None:
                if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                    model.text_encoder.model = model.text_encoder.model.to("cuda")

            _ensure_text_conditioning(model, data_batch)
            _ensure_point_conditioning_dtype(model, data_batch)
            _validate_conditioning_keys(data_batch)

            # ---- Step 2: offload text encoder after embeddings are ready ----
            if self.cfg.offload_text_encoder:
                self._offload_text_encoder(model)

            # ---- Step 3: load tokenizer encoder to GPU (encode conditioning frames) ----
            if self.cfg.offload_tokenizer:
                self._load_tokenizer_encoder(model)

            # ---- Step 4: load diffusion net to GPU ----
            if self.cfg.offload_diffusion_model:
                self._load_diffusion_net(model)

            # ---- Step 5: run diffusion sampling ----
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                x0_latent = model.generate_samples_from_batch(
                    data_batch=data_batch,          # normalised in-place on first call
                    guidance=self.cfg.guidance,
                    seed=seed,
                    num_steps=self.cfg.num_diffusion_steps,
                    shift=self.cfg.shift,
                )
            latents.append(x0_latent.detach())

            # ---- Step 6: offload diffusion net; swap tokenizer for decoder ----
            if self.cfg.offload_diffusion_model:
                self._offload_diffusion_net(model)
            if self.cfg.offload_tokenizer:
                if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
                    model.tokenizer.encoder = model.tokenizer.encoder.to("cpu")
                self._load_tokenizer_decoder(model)

            # ---- Step 7: decode latent → pixel-space video ----
            # model.decode returns an unnormalised video; keep on CPU to save VRAM.
            decoded = model.decode(x0_latent)   # [B, C, T, H, W]
            videos.append(decoded.cpu())

            # ---- Step 8: offload tokenizer decoder after decode ----
            if self.cfg.offload_tokenizer:
                if hasattr(model.tokenizer, "decoder") and model.tokenizer.decoder is not None:
                    model.tokenizer.decoder = model.tokenizer.decoder.to("cpu")

            del decoded
            torch.cuda.empty_cache()

        return latents, videos

    # -----------------------------------------------------------------------
    # Phase 2 – Advantages
    # -----------------------------------------------------------------------

    def compute_advantages(self, rewards: List[float]) -> Tensor:
        """Group-normalise *rewards* and return advantage tensor.

        Args:
            rewards: list of G scalar floats (one per group member).

        Returns:
            advantages: float32 tensor of shape ``[G]``.
        """
        r = torch.tensor(rewards, dtype=torch.float32)
        mean = r.mean()
        std  = r.std(unbiased=False) + 1e-8
        adv = (r - mean) / std
        adv = adv.clamp(-self.cfg.adv_clip, self.cfg.adv_clip)
        return adv  # shape [G]

    # -----------------------------------------------------------------------
    # Phase 3 – GRPO update
    # -----------------------------------------------------------------------

    def grpo_update_step(
        self,
        model: Text2WorldModelRectifiedFlow,
        optimizer: torch.optim.Optimizer,
        sampled_latents: List[Tensor],
        advantages: Tensor,
        data_batch: Dict[str, torch.Tensor],
    ) -> Tuple[float, float]:
        """One gradient-accumulation update over all G group members.

        Returns:
            Tuple of (loss, grad_norm).

        Args:
            model: the Cosmos model (Point Adapter unfrozen, base frozen).
            optimizer: optimises only Point Adapter parameters.
            sampled_latents: list of G latents ``[B, C, T, H, W]`` from rollout.
            advantages: ``[G]`` float32 advantage values.
            data_batch: original batch dict (normalised in-place by rollout).

        Returns:
            Scalar mean GRPO loss (Python float) for logging.
        """
        model.train()
        G = len(sampled_latents)
        device = model.tensor_kwargs["device"]
        use_amp = torch.cuda.is_available()
        amp_dtype = torch.bfloat16

        # --- Extract condition from data_batch -----------------------------------
        # Build a full Video2WorldCondition (including gt_frames/mask) via
        # model.get_data_and_condition(), mirroring the native training path.
        _ensure_text_conditioning(model, data_batch)
        _ensure_point_conditioning_dtype(model, data_batch)
        _validate_conditioning_keys(data_batch)
        _, _, condition = model.get_data_and_condition(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)
        # Disable context-parallel for single-GPU training
        model.net.disable_context_parallel()

        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for i in range(G):
            x0: Tensor = sampled_latents[i].to(device=device, dtype=torch.float32)
            adv_i: float = advantages[i].item()

            # ---- Add noise (rectified-flow interpolation, same as training_step) ----
            B = x0.shape[0]
            epsilon = torch.randn_like(x0)                                  # noise

            t_B   = model.rectified_flow.sample_train_time(B).to(**model.tensor_kwargs_fp32)
            t_B   = rearrange(t_B, "b -> b 1")                              # [B, 1]

            timesteps = model.rectified_flow.get_discrete_timestamp(t_B, model.tensor_kwargs_fp32)  # [B, 1]
            sigmas    = model.rectified_flow.get_sigmas(timesteps, model.tensor_kwargs_fp32)         # [B, 1]
            sigmas    = rearrange(sigmas, "b -> b 1")                                                # [B, 1]

            # xt = epsilon * sigma + x0 * (1 - sigma)  ;  v_target = epsilon - x0
            xt, v_target = model.rectified_flow.get_interpolation(epsilon, x0, sigmas)

            # ---- Denoise (with gradient through Point Adapter only) ----------------
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                v_pred = model.denoise(
                    noise=epsilon,
                    xt_B_C_T_H_W=xt.to(**model.tensor_kwargs),
                    timesteps_B_T=timesteps,
                    condition=condition,
                )  # [B, C, T, H, W], model dtype (bfloat16)

            # ---- Per-instance flow-matching loss -----------------------------------
            v_pred_f32 = v_pred.float()
            time_weights = model.rectified_flow.train_time_weight(
                timesteps, model.tensor_kwargs_fp32
            )  # [B, 1] – currently uniform (all 1s)

            per_instance_loss = torch.mean(
                (v_pred_f32 - v_target) ** 2,
                dim=list(range(1, v_pred_f32.dim())),
            )  # [B]

            # ---- GRPO loss: advantage-weighted, negative (gradient ascent on R) ----
            # time_weights may be [B, 1]; squeeze to [B] for broadcast
            tw = time_weights.squeeze(-1)   # [B]
            loss_i = (-adv_i * tw * per_instance_loss).mean() / G

            loss_i.backward()
            total_loss = total_loss + loss_i.detach()

        # ---- Gradient clip + step ----------------------------------------------
        grad_norm: float = 0.0
        if self.cfg.grad_clip > 0:
            point_adapter_params = [
                p for n, p in model.net.named_parameters()
                if "point_adapter" in n and p.requires_grad
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(point_adapter_params, self.cfg.grad_clip).item()

        optimizer.step()

        return total_loss.item(), grad_norm

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(
        self,
        model: Text2WorldModelRectifiedFlow,
        dataloader: Iterator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        start_iter: int = 0,
    ) -> None:
        """Run the GRPO RL training loop.

        Args:
            model: Cosmos model with Point Adapter loaded; base weights frozen.
            dataloader: yields data-batch dicts (tensors on CPU or GPU).
            optimizer: if None, creates AdamW over Point Adapter params.
            start_iter: resume iteration counter (e.g. when loading a checkpoint).
        """
        cfg = self.cfg
        device = model.tensor_kwargs["device"]

        # Build optimizer over Point Adapter parameters only
        if optimizer is None:
            adapter_params = [
                p for n, p in model.net.named_parameters()
                if "point_adapter" in n and p.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                adapter_params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            n_adapter = sum(p.numel() for p in adapter_params)
            logger.info(f"[GRPOTrainer] AdamW over {n_adapter/1e6:.2f}M Point Adapter params")

        # Register wandb.watch once
        self._watch_wandb(model)

        data_iter = iter(dataloader)
        iteration = start_iter
        t0 = time.time()

        while iteration < cfg.max_iter:
            t_iter_start = time.time()

            # ---- Fetch next batch --------------------------------------------------
            try:
                raw_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                raw_batch = next(data_iter)

            data_batch = _move_batch_to_device(raw_batch, device)

            # ================================================================
            # ROLLOUT
            # ================================================================
            t_rollout_start = time.time()
            latents, videos = self.rollout_phase(model, data_batch)
            t_rollout = time.time() - t_rollout_start
            # data_batch is now normalised in-place (safe to reuse in update)

            # ================================================================
            # REWARD  (optionally offload Cosmos model to free VRAM for SAM/Depth)
            # ================================================================
            t_reward_start = time.time()
            self.offload_model_for_reward(model)
            rewards: List[float] = compute_rewards(videos, data_batch)
            reward_diag = get_last_reward_diagnostics()
            self.reload_model_after_reward(model)
            t_reward = time.time() - t_reward_start
            del videos  # free pixel buffers

            # ================================================================
            # ADVANTAGE
            # ================================================================
            advantages: Tensor = self.compute_advantages(rewards)   # [G]

            # ================================================================
            # UPDATE  (optionally multiple epochs over the same rollout)
            # ================================================================
            t_update_start = time.time()
            epoch_losses: List[float] = []
            epoch_grad_norms: List[float] = []
            for _epoch in range(cfg.num_update_epochs):
                # Work with a fresh clone each epoch so in-place ops
                # in the conditioner don't corrupt the batch.
                batch_for_update = _clone_batch(data_batch)
                loss_val, grad_norm = self.grpo_update_step(
                    model=model,
                    optimizer=optimizer,
                    sampled_latents=latents,
                    advantages=advantages,
                    data_batch=batch_for_update,
                )
                epoch_losses.append(loss_val)
                epoch_grad_norms.append(grad_norm)
            t_update = time.time() - t_update_start

            del latents  # free rollout buffers
            iteration += 1
            t_iter = time.time() - t_iter_start

            # ================================================================
            # LOG
            # ================================================================
            mean_reward = float(sum(rewards) / len(rewards))
            mean_adv    = float(advantages.abs().mean().item())
            mean_loss   = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
            mean_grad_norm = float(sum(epoch_grad_norms) / len(epoch_grad_norms)) if epoch_grad_norms else 0.0

            if iteration % cfg.log_every == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"[GRPO] iter={iteration:6d} | "
                    f"reward={mean_reward:.4f} | "
                    f"|adv|={mean_adv:.4f} | "
                    f"loss={mean_loss:.6f} | "
                    f"grad_norm={mean_grad_norm:.4f} | "
                    f"elapsed={elapsed:.1f}s | "
                    f"rollout={t_rollout:.1f}s | "
                    f"reward_t={t_reward:.1f}s ({reward_diag.elapsed_sec:.2f}s icp) | "
                    f"update={t_update:.1f}s | "
                    f"fit={reward_diag.mean_fitness:.4f} | "
                    f"rmse={reward_diag.mean_rmse:.4f} | "
                    f"align={reward_diag.mean_alignment_score:.4f} | "
                    f"fallback={reward_diag.fallback_count}/{reward_diag.sample_count}"
                )

            if self._wandb_run is not None and iteration % cfg.wandb_log_every == 0:
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                adv_tensor = advantages.float()
                lr_now = optimizer.param_groups[0]["lr"]

                wandb_metrics: dict = {
                    # ---- training core ----
                    "train/loss": mean_loss,
                    "train/loss_epoch_std": float(torch.tensor(epoch_losses).std().item()) if len(epoch_losses) > 1 else 0.0,
                    "train/grad_norm": mean_grad_norm,
                    "train/lr": lr_now,
                    # ---- rewards ----
                    "reward/mean": mean_reward,
                    "reward/max": float(rewards_tensor.max().item()),
                    "reward/min": float(rewards_tensor.min().item()),
                    "reward/std": float(rewards_tensor.std().item()),
                    # ---- advantages ----
                    "advantage/mean": float(adv_tensor.mean().item()),
                    "advantage/std": float(adv_tensor.std().item()),
                    "advantage/abs_mean": mean_adv,
                    "advantage/max": float(adv_tensor.max().item()),
                    "advantage/min": float(adv_tensor.min().item()),
                    # ---- ICP / reward diagnostics ----
                    "reward_diag/mean_fitness": reward_diag.mean_fitness,
                    "reward_diag/mean_rmse": reward_diag.mean_rmse,
                    "reward_diag/mean_alignment_score": reward_diag.mean_alignment_score,
                    "reward_diag/fallback_count": reward_diag.fallback_count,
                    "reward_diag/fallback_rate": (
                        reward_diag.fallback_count / reward_diag.sample_count
                        if reward_diag.sample_count > 0 else 0.0
                    ),
                    "reward_diag/sample_count": reward_diag.sample_count,
                    "reward_diag/icp_elapsed_sec": reward_diag.elapsed_sec,
                    # ---- per-sample rewards (individual group members) ----
                    **{f"reward/sample_{i}": float(r) for i, r in enumerate(rewards)},
                    # ---- timing ----
                    "perf/rollout_sec": t_rollout,
                    "perf/reward_sec": t_reward,
                    "perf/update_sec": t_update,
                    "perf/iter_sec": t_iter,
                    # ---- GPU memory ----
                    "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                    "gpu/memory_reserved_gb": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0,
                    "gpu/memory_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                    # ---- misc ----
                    "train/iteration": iteration,
                    "train/elapsed_total_sec": time.time() - t0,
                }
                _wandb.log(wandb_metrics, step=iteration)

                # ---- optionally log alignment images ----
                if cfg.wandb_log_media_every > 0 and iteration % cfg.wandb_log_media_every == 0:
                    debug_dir = cfg.debug_dir or os.path.join(cfg.output_dir, "debug")
                    iter_dir = os.path.join(debug_dir, f"iter_{iteration:06d}")
                    media: dict = {}
                    if os.path.isdir(iter_dir):
                        import glob
                        for png in sorted(glob.glob(os.path.join(iter_dir, "*.png")))[:8]:
                            key = "debug/" + os.path.basename(png).replace(".png", "")
                            media[key] = _wandb.Image(png)
                    if media:
                        _wandb.log(media, step=iteration)

            # ================================================================
            # CHECKPOINT
            # ================================================================
            if iteration % cfg.save_every == 0:
                self._save_point_adapter(model, iteration)

        # Final save
        self._save_point_adapter(model, iteration, tag="final")
        self._finish_wandb()
        logger.info("[GRPOTrainer] Training complete.")

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    def _save_point_adapter(
        self,
        model: Text2WorldModelRectifiedFlow,
        iteration: int,
        tag: str = "",
    ) -> None:
        """Save the Point Adapter state dict to *output_dir*."""
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(
            self.cfg.output_dir, f"point_adapter_iter{iteration:06d}{suffix}.pt"
        )
        state = {
            name: param.detach().cpu()
            for name, param in model.net.named_parameters()
            if "point_adapter" in name
        }
        torch.save({"iteration": iteration, "point_adapter": state}, path)
        logger.info(f"[GRPOTrainer] Saved Point Adapter checkpoint → {path}")

    @staticmethod
    def load_point_adapter(
        model: Text2WorldModelRectifiedFlow,
        checkpoint_path: str,
        strict: bool = True,
    ) -> int:
        """Load a previously-saved Point Adapter checkpoint into *model*.

        Returns:
            iteration at which the checkpoint was saved.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        adapter_state = ckpt["point_adapter"]
        # Load into model.net (key prefix must match)
        missing, unexpected = model.net.load_state_dict(adapter_state, strict=False)
        if strict:
            # In strict mode, only accept unknown keys that are NOT in the adapter
            adapter_missing = [k for k in missing    if "point_adapter" in k]
            non_adapter_unexp = [k for k in unexpected if "point_adapter" not in k]
            if adapter_missing:
                raise RuntimeError(f"Missing Point Adapter keys: {adapter_missing}")
            if non_adapter_unexp:
                raise RuntimeError(f"Unexpected non-adapter keys: {non_adapter_unexp}")
        iteration = ckpt.get("iteration", 0)
        logger.info(
            f"[GRPOTrainer] Loaded Point Adapter from {checkpoint_path} (iter={iteration})"
        )
        return iteration
