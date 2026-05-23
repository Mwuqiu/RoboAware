#!/usr/bin/env python3
"""D4-CN baseline sampling: 8 episodes (same as LoRA baseline).

Each episode uses its own PC + first frame (cond_frames=1 == image2world).
Output mp4 files matching LoRA baseline naming for side-by-side comparison.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

_REPO = "/root/autodl-tmp/cosmos-predict2.5"
if _REPO not in sys.path: sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "cosmos_grpo"))

from sample_online_pointcloud import (
    _build_config, _load_model_for_sampling,
    _build_single_batch, _move_batch_to_device,
    _ensure_text_conditioning, _ensure_point_conditioning_dtype, _validate_conditioning_keys,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video


def run_one(model, batch, seed=0, num_steps=35, guidance=1.5, shift=5.0):
    torch.manual_seed(seed); np.random.seed(seed)
    ncond = int(batch["num_conditional_frames"].flatten()[0].item())
    cond01 = batch["video"][0, :, :ncond].detach().float().cpu() / 255.0
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x0 = model.generate_samples_from_batch(
                data_batch=batch, guidance=guidance, seed=seed,
                num_steps=num_steps, shift=shift,
            )
        v = torch.cat([model.decode(c) for c in x0], dim=3) if isinstance(x0, list) else model.decode(x0)
    v = ((v[0].detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
    n = min(cond01.shape[1], v.shape[1])
    v[:, :n] = cond01[:, :n].to(dtype=v.dtype)
    return v


EPISODES = [
    ("adjust_bottle", 40, "R"),
    ("adjust_bottle", 48, "L"),
    ("dump_bin_bigbin", 40, "U"),
    ("dump_bin_bigbin", 48, "U"),
    ("handover_mic", 40, "D"),
    ("handover_mic", 47, "D"),
    ("pick_dual_bottles", 43, "D"),
    ("pick_dual_bottles", 47, "D"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/root/autodl-tmp/v5_d4cn_bjb1_iter5000_pt/model_ema_bf16.pt")
    p.add_argument("--experiment", default="predict2_point_adapter_v5_controlnet_4tasks")
    p.add_argument("--config", default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    p.add_argument("--dataset-dir", default="/root/autodl-tmp/cosmos_training_data_world_arena_ablation_4tasks/test")
    p.add_argument("--output-dir", default="/root/autodl-tmp/d4cn_iter5000_baseline_8ep")
    p.add_argument("--num-conditional-frames", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ds_dir = Path(args.dataset_dir).resolve()

    print(f"=== load ckpt {args.ckpt} ===")
    config = _build_config(args.config, args.experiment)
    model = _load_model_for_sampling(config, args.ckpt)
    model.eval()
    device = torch.device("cuda")

    pc_encoder_config = dict(
        DATASET="robotwin", CONFIG="semseg-pt-v3m1-0-base",
        EXP_NAME="semseg-pt-v3m1-0-base-cosmos-pcenc",
        WEIGHT_NAME="model_last", v5_layer="dec_0", v5_feat_input="coord",
    )
    dataset = VideoDataset(
        dataset_dir=str(ds_dir), num_frames=93, video_size=(480, 640),
        pc_latent_source="precomputed", pc_latent_k=104, pc_latent_amp=True,
        pc_encoder_config=pc_encoder_config,
        pc_conditioning_mode_probs=dict(full=1.0),
        pc_conditioning_prefix_frames=[1, 2],
    )

    for task, ep, arm in EPISODES:
        ep_id = f"TianxingChen_RoboTwin2.0_{task}_aloha-agilex_ep_{ep:06d}"
        out_name = f"{task}_ep{ep:03d}_{arm}.mp4"
        out_path = out / out_name
        if out_path.exists():
            print(f"  skip {out_name} (exists)")
            continue
        print(f"\n=== {task} ep {ep} ===")
        batch = _build_single_batch(
            dataset=dataset,
            video_path=str(ds_dir / "videos" / f"{ep_id}.mp4"),
            episode_id=ep_id, start_frame=0,
            num_conditional_frames=args.num_conditional_frames,
        )
        batch = _move_batch_to_device(batch, device)
        _ensure_text_conditioning(model, batch)
        _ensure_point_conditioning_dtype(model, batch)
        _validate_conditioning_keys(batch)

        t0 = time.time()
        video = run_one(model, batch, args.seed, args.num_steps, args.guidance, args.shift)
        print(f"  elapsed {time.time() - t0:.1f}s; saving {out_name}")
        save_img_or_video(video, str(out_path), fps=args.fps)

    print(f"\nDONE. 8 videos in {out}")


if __name__ == "__main__":
    main()
