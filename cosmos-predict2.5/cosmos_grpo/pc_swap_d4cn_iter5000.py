#!/usr/bin/env python3
"""PC swap inference for D4-CN iter_5000 — does PC alone guide arm-side?

4 runs:
  A1 frame_R + text_R + PC_R (baseline)
  A2 frame_R + text_R + PC_L (swap)
  B1 frame_L + text_L + PC_L (baseline)
  B2 frame_L + text_L + PC_R (swap)

ep_R = adjust_bottle ep_000040 (right arm)
ep_L = adjust_bottle ep_000048 (left arm)
ckpt = D4-CN iter_5000 (the breakthrough ckpt)
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
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "cosmos_grpo"))

from sample_online_pointcloud import (
    _build_config,
    _load_model_for_sampling,
    _build_single_batch,
    _move_batch_to_device,
    _ensure_text_conditioning,
    _ensure_point_conditioning_dtype,
    _validate_conditioning_keys,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video


def run_one(model, batch, seed, num_steps, guidance, shift, force_init=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cond_frames01 = None
    ncond = int(batch["num_conditional_frames"].flatten()[0].item())
    if force_init and ncond > 0:
        cond_frames01 = batch["video"][0, :, :ncond].detach().float().cpu() / 255.0
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x0 = model.generate_samples_from_batch(
                data_batch=batch, guidance=guidance, seed=seed,
                num_steps=num_steps, shift=shift,
            )
        if isinstance(x0, list):
            video_pix = torch.cat([model.decode(chunk) for chunk in x0], dim=3)
        else:
            video_pix = model.decode(x0)
    video_pix = ((video_pix[0].detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
    if cond_frames01 is not None:
        n = min(cond_frames01.shape[1], video_pix.shape[1])
        video_pix[:, :n] = cond_frames01[:, :n].to(dtype=video_pix.dtype)
    return video_pix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/root/autodl-tmp/v5_d4cn_bjb1_iter5000_pt/model_ema_bf16.pt")
    p.add_argument("--experiment", default="predict2_point_adapter_v5_controlnet_4tasks")
    p.add_argument("--config", default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    p.add_argument("--dataset-dir", default="/root/autodl-tmp/cosmos_training_data_world_arena_ablation_4tasks/test")
    p.add_argument("--num-conditional-frames", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output-dir", default="/root/autodl-tmp/pc_swap_d4cn_iter5000")
    p.add_argument("--ep-r", default="TianxingChen_RoboTwin2.0_adjust_bottle_aloha-agilex_ep_000040")
    p.add_argument("--ep-l", default="TianxingChen_RoboTwin2.0_adjust_bottle_aloha-agilex_ep_000048")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = Path(args.dataset_dir).resolve()

    print(f"=== load ckpt {args.ckpt} exp={args.experiment} ===")
    config = _build_config(args.config, args.experiment)
    model = _load_model_for_sampling(config, args.ckpt)
    model.eval()
    device = torch.device("cuda")

    pc_encoder_config = dict(
        DATASET="robotwin",
        CONFIG="semseg-pt-v3m1-0-base",
        EXP_NAME="semseg-pt-v3m1-0-base-cosmos-pcenc",
        WEIGHT_NAME="model_last",
        v5_layer="dec_0",
        v5_feat_input="coord",
    )
    dataset = VideoDataset(
        dataset_dir=str(ds_dir),
        num_frames=93,
        video_size=(480, 640),
        pc_latent_source="precomputed",
        pc_latent_k=104,
        pc_latent_amp=True,
        pc_encoder_config=pc_encoder_config,
        pc_conditioning_mode_probs=dict(full=1.0),
        pc_conditioning_prefix_frames=[1, 2],
    )

    def build(ep_id):
        b = _build_single_batch(
            dataset=dataset, video_path=str(ds_dir / "videos" / f"{ep_id}.mp4"),
            episode_id=ep_id, start_frame=0,
            num_conditional_frames=args.num_conditional_frames,
        )
        b = _move_batch_to_device(b, device)
        _ensure_text_conditioning(model, b)
        _ensure_point_conditioning_dtype(model, b)
        _validate_conditioning_keys(b)
        return b

    print(f"=== build batch_R from {args.ep_r} ===")
    batch_R = build(args.ep_r)
    print(f"=== build batch_L from {args.ep_l} ===")
    batch_L = build(args.ep_l)

    PC_R_x0 = batch_R["pc_latent_x0"].clone()
    PC_R_msk = batch_R["pc_latent_mask"].clone()
    PC_L_x0 = batch_L["pc_latent_x0"].clone()
    PC_L_msk = batch_L["pc_latent_mask"].clone()
    print(f"  PC_R shape={tuple(PC_R_x0.shape)}  PC_L shape={tuple(PC_L_x0.shape)}")

    def set_pc(batch, pc_x0, pc_msk):
        cur_T = batch["pc_latent_x0"].shape[1]
        new_T = pc_x0.shape[1]
        if new_T > cur_T:
            pc_x0_use = pc_x0[:, :cur_T]; pc_msk_use = pc_msk[:, :cur_T]
        elif new_T < cur_T:
            pad = cur_T - new_T
            pc_x0_use = torch.cat([pc_x0, pc_x0[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            pc_msk_use = torch.cat([pc_msk, pc_msk[:, -1:].expand(-1, pad, -1)], dim=1)
        else:
            pc_x0_use, pc_msk_use = pc_x0, pc_msk
        batch["pc_latent_x0"] = pc_x0_use.to(device=device, dtype=batch["pc_latent_x0"].dtype)
        batch["pc_latent_mask"] = pc_msk_use.to(device=device, dtype=batch["pc_latent_mask"].dtype)

    runs = [
        ("A1_frameR_pcR_baseline", batch_R, PC_R_x0, PC_R_msk),
        ("A2_frameR_pcL_swap",     batch_R, PC_L_x0, PC_L_msk),
        ("B1_frameL_pcL_baseline", batch_L, PC_L_x0, PC_L_msk),
        ("B2_frameL_pcR_swap",     batch_L, PC_R_x0, PC_R_msk),
    ]
    for tag, batch, pc_x0, pc_msk in runs:
        set_pc(batch, pc_x0, pc_msk)
        print(f"\n=== run {tag}  PC.shape={tuple(batch['pc_latent_x0'].shape)} ===")
        t0 = time.time()
        video = run_one(model, batch, args.seed, args.num_steps, args.guidance, args.shift)
        print(f"   elapsed {time.time() - t0:.1f}s, video.shape={tuple(video.shape)}")
        out_path = out_dir / f"swap_{tag}.mp4"
        save_img_or_video(video, str(out_path), fps=args.fps)
        print(f"   saved → {out_path}")

    print(f"\n=== DONE. 4 videos in {out_dir}")


if __name__ == "__main__":
    main()
