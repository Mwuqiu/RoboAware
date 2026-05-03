#!/usr/bin/env python3
"""One-shot sampling script for Cosmos Point Adapter with online pointcloud encoding.

This script generates one video sample for a target episode using:
- a specified checkpoint (iter dir or .pt file),
- text/video conditions from dataset files,
- online pointcloud encoder (Pointcept) from `dataset_video.py`.

It intentionally forces `pc_latent_source='online'` so it does not use precomputed
`pc_latent/*.pt` files.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader

# Ensure repo root is importable when running from anywhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset
from cosmos_predict2._src.predict2.utils.model_loader import load_model_state_dict_from_checkpoint


def _resolve_checkpoint(path_str: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = [
        path / "model_ema_bf16.pt",
        path / "model.pt",
        path / "model_ema_fp32.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Could not resolve a consolidated checkpoint file under: "
        f"{path}. Expected one of {[c.name for c in candidates]}"
    )


def _ensure_pointcept_root() -> str:
    existing = os.environ.get("POINTCEPT_ROOT")
    if existing and (Path(existing).expanduser() / "pointflow" / "pf_encoder.py").is_file():
        return str(Path(existing).expanduser().resolve())

    candidates = [
        Path(_REPO_ROOT) / "Pointcept",
        Path("/root/autodl-tmp/Pointcept"),
    ]
    for candidate in candidates:
        if (candidate / "pointflow" / "pf_encoder.py").is_file():
            os.environ["POINTCEPT_ROOT"] = str(candidate.resolve())
            return os.environ["POINTCEPT_ROOT"]

    raise FileNotFoundError(
        "POINTCEPT_ROOT is not set and no Pointcept checkout was found. "
        "Please set POINTCEPT_ROOT to a directory containing pointflow/pf_encoder.py"
    )


def _build_config(config_module_path: str, experiment_name: str):
    config_module = get_config_module(config_module_path)
    config = importlib.import_module(config_module).make_config()
    # Keep the same override style as existing train/inference flow.
    config = override(config, ["--", f"experiment={experiment_name}"])
    return config


def _load_model_for_sampling(config, checkpoint_path: str):
    """Instantiate model and load checkpoint, avoiding package import conflicts."""
    config.model.config.fsdp_shard_size = 1
    model = instantiate(config.model)

    # Keep EMA on CPU to reduce unnecessary VRAM usage.
    cpu_stash: dict[str, Any] = {}
    if getattr(model, "net_ema", None) is not None:
        cpu_stash["net_ema"] = model.net_ema
        model.net_ema = None

    model.cuda()
    for attr, module in cpu_stash.items():
        setattr(model, attr, module)

    config.checkpoint.load_path = str(checkpoint_path)
    model = load_model_state_dict_from_checkpoint(
        model=model,
        config=config,
        s3_checkpoint_dir=str(checkpoint_path),
    )
    return model


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _ensure_text_conditioning(model, data_batch: dict[str, Any]) -> None:
    """Mirror GRPO path: explicitly compute online text embeddings when needed."""
    if not (
        model.config.text_encoder_config is not None
        and model.config.text_encoder_config.compute_online
    ):
        return

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


def _ensure_point_conditioning_dtype(model, data_batch: dict[str, Any]) -> None:
    pc = data_batch.get("pc_latent_x0")
    if pc is None:
        return
    try:
        ref_param = next(model.net.point_adapter.pc_encoder.parameters())
    except Exception:
        ref_param = next(model.net.parameters())
    data_batch["pc_latent_x0"] = pc.to(device=ref_param.device, dtype=ref_param.dtype)


def _validate_conditioning_keys(data_batch: dict[str, Any]) -> None:
    required = ["pc_latent_x0", "pc_latent_mask", "t5_text_embeddings", "t5_text_mask"]
    missing = [k for k in required if data_batch.get(k) is None]
    if missing:
        raise RuntimeError(
            "Missing conditioning fields in batch: "
            f"{missing}. Available keys: {sorted(list(data_batch.keys()))}"
        )


def _get_video_chunk_at(
    video_path: str,
    preprocess,
    start_frame: int,
    sequence_length: int,
) -> tuple[torch.Tensor, float]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    if total_frames < sequence_length:
        raise ValueError(
            f"Video {video_path} has only {total_frames} frames, requires {sequence_length}."
        )
    max_start = total_frames - sequence_length
    if start_frame < 0 or start_frame > max_start:
        raise ValueError(
            f"start_frame={start_frame} out of valid range [0, {max_start}] for {video_path}"
        )

    frame_ids = np.arange(start_frame, start_frame + sequence_length).tolist()
    batch = vr.get_batch(frame_ids)
    frame_data = batch.numpy() if hasattr(batch, "numpy") else batch.asnumpy()
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 16.0
    vr.seek(0)
    del vr

    frames = torch.from_numpy(frame_data.astype(np.uint8)).permute(0, 3, 1, 2)
    frames = preprocess(frames)
    frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
    video = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
    return video, fps


def _build_single_batch(
    dataset: VideoDataset,
    video_path: str,
    episode_id: str,
    start_frame: int | None,
) -> dict[str, Any]:
    if start_frame is None:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        return next(iter(loader))

    # Build a deterministic sample at fixed start_frame using dataset internals.
    video, fps = _get_video_chunk_at(
        video_path=video_path,
        preprocess=dataset.preprocess,
        start_frame=start_frame,
        sequence_length=dataset.sequence_length,
    )

    data: dict[str, Any] = {}
    pc_latent = dataset._load_pc_latent_window(episode_id, start_frame)
    if pc_latent is not None:
        pc_x0, pc_mask = pc_latent
        data["pc_latent_x0"] = pc_x0.unsqueeze(0)
        data["pc_latent_mask"] = pc_mask.unsqueeze(0)

    if dataset.caption_format == "json":
        caption = dataset._load_json_caption(Path(dataset.caption_dir) / f"{episode_id}.json")
    else:
        caption = dataset._load_text(Path(dataset.caption_dir) / f"{episode_id}.txt")

    _, _, h, w = video.shape
    data["start_frame"] = torch.tensor([start_frame], dtype=torch.int64)
    data["episode_id"] = [episode_id]
    data["video_basename"] = [episode_id]
    data["video_path"] = [video_path]
    data["video"] = video.unsqueeze(0)
    data["ai_caption"] = [caption]
    data["fps"] = torch.tensor([fps], dtype=torch.float32)
    data["image_size"] = torch.tensor([[h, w, h, w]], dtype=torch.int64)
    data["num_frames"] = torch.tensor([dataset.sequence_length], dtype=torch.int64)
    data["padding_mask"] = torch.zeros((1, 1, h, w), dtype=torch.float32)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one sample using online pointcloud encoding from a Point Adapter checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path (.pt) or iter dir containing model_ema_bf16.pt/model.pt.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="/root/autodl-tmp/cosmos-predict2.5/datasets/cosmos_so100_point",
        help="Dataset root that contains videos/, metas/ and pointclouds/.",
    )
    parser.add_argument(
        "--episode-id",
        default="Beyond_Success_1031_Cleaned__ep_000000",
        help="Episode basename without extension.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/cosmos-output/samples_online_pc",
        help="Output directory for generated mp4 and metadata.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Optional fixed start frame. If omitted, sample one random chunk from the episode.",
    )
    parser.add_argument(
        "--config",
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        help="Hydra/Imaginaire config module.",
    )
    parser.add_argument(
        "--experiment",
        default="predict2_point_adapter_training_2b_cosmos_so100_point",
        help="Experiment override to instantiate the Point Adapter architecture.",
    )
    parser.add_argument("--pointcept-k", type=int, default=30)
    parser.add_argument("--pointcept-sample", default="first")
    parser.add_argument("--pointcept-pad-value", type=float, default=0.0)
    parser.add_argument("--pointcept-amp", action="store_true")
    parser.add_argument("--pointcept-grid-size", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pointcept_root = _ensure_pointcept_root()
    ckpt_path = _resolve_checkpoint(args.checkpoint)

    dataset_dir = Path(args.dataset_dir).resolve()
    video_path = dataset_dir / "videos" / f"{args.episode_id}.mp4"
    pointcloud_path = dataset_dir / "pointclouds" / f"{args.episode_id}.npy"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not pointcloud_path.is_file():
        raise FileNotFoundError(f"Pointcloud not found: {pointcloud_path}")

    config = _build_config(args.config, args.experiment)
    model = _load_model_for_sampling(config, ckpt_path)
    model.eval()

    dataset = VideoDataset(
        dataset_dir=str(dataset_dir),
        num_frames=args.num_frames,
        video_size=(args.height, args.width),
        caption_format="auto",
        video_paths=[str(video_path)],
        pc_latent_source="online",
        pc_latent_k=args.pointcept_k,
        pc_latent_sample=args.pointcept_sample,
        pc_latent_pad_value=args.pointcept_pad_value,
        pc_latent_amp=args.pointcept_amp,
        pc_latent_grid_size=args.pointcept_grid_size,
    )
    batch = _build_single_batch(
        dataset=dataset,
        video_path=str(video_path),
        episode_id=args.episode_id,
        start_frame=args.start_frame,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _move_batch_to_device(batch, device)

    _ensure_text_conditioning(model, batch)
    _ensure_point_conditioning_dtype(model, batch)
    _validate_conditioning_keys(batch)

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16

    t0 = time.time()
    with torch.no_grad():
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with amp_ctx:
            x0 = model.generate_samples_from_batch(
                data_batch=batch,
                guidance=args.guidance,
                seed=args.seed,
                num_steps=args.num_steps,
                shift=args.shift,
            )
        if isinstance(x0, list):
            video = torch.cat([model.decode(chunk) for chunk in x0], dim=3)
        else:
            video = model.decode(x0)

    # model.decode output is typically in [-1, 1].
    video_in01 = ((video[0].detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_stem = out_dir / f"{args.episode_id}_seed{args.seed}_{stamp}"
    save_img_or_video(video_in01, str(out_stem), fps=args.fps)

    start_frame_value: int | None
    if isinstance(batch.get("start_frame"), torch.Tensor):
        start_frame_value = int(batch["start_frame"].flatten()[0].item())
    else:
        start_frame_value = args.start_frame

    meta = {
        "checkpoint": ckpt_path,
        "pointcept_root": pointcept_root,
        "dataset_dir": str(dataset_dir),
        "episode_id": args.episode_id,
        "video_path": str(video_path),
        "pointcloud_path": str(pointcloud_path),
        "pc_latent_source": "online",
        "seed": args.seed,
        "guidance": args.guidance,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "resolution": [args.height, args.width],
        "start_frame": start_frame_value,
        "output_mp4": f"{out_stem}.mp4",
        "elapsed_sec": time.time() - t0,
    }
    with open(f"{out_stem}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[sample_online_pointcloud] saved video: {out_stem}.mp4")
    print(f"[sample_online_pointcloud] saved meta : {out_stem}.json")


if __name__ == "__main__":
    main()
