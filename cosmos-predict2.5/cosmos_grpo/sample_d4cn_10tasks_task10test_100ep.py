#!/usr/bin/env python3
"""D4-CN 10tasks iter_5000 sampling on Task 10 test (all 100 episodes).

Same protocol as LoRA 100ep eval: meta text prompt + GT first frame + precomputed PC.
Output: 100 mp4 in /root/autodl-tmp/d4cn_10tasks_iter5k_task10test_100ep/
"""
from __future__ import annotations
import argparse, os, sys, time, glob
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/root/autodl-tmp/v5_d4cn_10tasks_iter5000_pt/model_ema_bf16.pt")
    p.add_argument("--experiment", default="predict2_point_adapter_v5_controlnet_10tasks")
    p.add_argument("--config", default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    p.add_argument("--dataset-dir", default="/root/autodl-tmp/cosmos_training_data_world_arena_ablation_10tasks/test")
    p.add_argument("--output-dir", default="/root/autodl-tmp/d4cn_10tasks_iter5k_task10test_100ep")
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

    videos = sorted(glob.glob(str(ds_dir / "videos" / "*.mp4")))
    print(f"Found {len(videos)} test videos in Task 10")
    for i, vp in enumerate(videos):
        ep_id = Path(vp).stem
        out_path = out / f"{ep_id}.mp4"
        if out_path.exists():
            print(f"[{i+1}/{len(videos)}] skip {ep_id} (exists)"); continue
        print(f"\n[{i+1}/{len(videos)}] === {ep_id} ===")
        try:
            batch = _build_single_batch(
                dataset=dataset, video_path=vp, episode_id=ep_id,
                start_frame=0, num_conditional_frames=args.num_conditional_frames,
            )
            batch = _move_batch_to_device(batch, device)
            _ensure_text_conditioning(model, batch)
            _ensure_point_conditioning_dtype(model, batch)
            _validate_conditioning_keys(batch)
            t0 = time.time()
            video = run_one(model, batch, args.seed, args.num_steps, args.guidance, args.shift)
            print(f"  elapsed {time.time()-t0:.1f}s; saving {ep_id}.mp4")
            # save_img_or_video adds .mp4; pass path without ext
            save_img_or_video(video, str(out / ep_id), fps=args.fps)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nDONE. {len(list(out.glob('*.mp4')))} mp4 in {out}")


if __name__ == "__main__":
    main()
