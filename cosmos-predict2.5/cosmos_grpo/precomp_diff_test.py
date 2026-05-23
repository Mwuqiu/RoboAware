#!/usr/bin/env python3
"""Control: same as online_encode_diff_test.py but uses precomputed pc_latent.

If output mp4s are bit-exact → all 5% pixel diff in online test came from PC encoding shuffle.
If they also differ → diffusion itself has nondeterminism.
"""
from __future__ import annotations
import os, sys, time, argparse
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


def build_batch(model, dataset, ds_dir, ep_id, ncond, device):
    b = _build_single_batch(
        dataset=dataset,
        video_path=str(ds_dir / "videos" / f"{ep_id}.mp4"),
        episode_id=ep_id, start_frame=0,
        num_conditional_frames=ncond,
    )
    b = _move_batch_to_device(b, device)
    _ensure_text_conditioning(model, b)
    _ensure_point_conditioning_dtype(model, b)
    _validate_conditioning_keys(b)
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/root/autodl-tmp/v5_d4cn_bjb1_iter5000_pt/model_ema_bf16.pt")
    p.add_argument("--experiment", default="predict2_point_adapter_v5_controlnet_4tasks")
    p.add_argument("--config", default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    p.add_argument("--dataset-dir", default="/root/autodl-tmp/cosmos_training_data_world_arena_ablation_4tasks/test")
    p.add_argument("--ep-id", default="TianxingChen_RoboTwin2.0_adjust_bottle_aloha-agilex_ep_000040")
    p.add_argument("--output-dir", default="/root/autodl-tmp/precomp_diff_test")
    p.add_argument("--num-conditional-frames", type=int, default=1)
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
        pc_latent_source="precomputed",
        pc_latent_k=104, pc_latent_amp=True,
        pc_encoder_config=pc_encoder_config,
        pc_conditioning_mode_probs=dict(full=1.0),
        pc_conditioning_prefix_frames=[1, 2],
    )

    print(f"\n=== RUN 1: precomputed PC + diffusion (seed=0) ===")
    b1 = build_batch(model, dataset, ds_dir, args.ep_id, args.num_conditional_frames, device)
    pc1 = b1["pc_latent_x0"].detach().clone()
    t0 = time.time(); v1 = run_one(model, b1, seed=0)
    print(f"  elapsed {time.time()-t0:.1f}s")
    save_img_or_video(v1, str(out / "run1.mp4"), fps=30)

    print(f"\n=== RUN 2: precomputed PC + diffusion (seed=0) ===")
    b2 = build_batch(model, dataset, ds_dir, args.ep_id, args.num_conditional_frames, device)
    pc2 = b2["pc_latent_x0"].detach().clone()
    t0 = time.time(); v2 = run_one(model, b2, seed=0)
    print(f"  elapsed {time.time()-t0:.1f}s")
    save_img_or_video(v2, str(out / "run2.mp4"), fps=30)

    print(f"\n=== PC feature diff (precomputed → should be 0) ===")
    pc1c, pc2c = pc1.float().cpu(), pc2.float().cpu()
    print(f"  pc diff max: {(pc1c-pc2c).abs().max().item():.4e}  (should be 0)")
    print(f"  pc exact equal: {torch.equal(pc1c, pc2c)}")

    print(f"\n=== Video diff (pixel level, skip cond frame) ===")
    d = (v1[:, 1:] - v2[:, 1:]).abs()
    print(f"  pixel diff mean: {d.mean().item()*255:.3f}/255 = {d.mean().item()*100:.3f}%")
    print(f"  pixel diff max:  {d.max().item()*255:.3f}/255")
    print(f"  pixel exact equal: {torch.equal(v1, v2)}")

    print(f"\nDONE. 2 videos in {out}")


if __name__ == "__main__":
    main()
