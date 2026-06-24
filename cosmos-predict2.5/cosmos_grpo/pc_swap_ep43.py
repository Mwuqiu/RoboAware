#!/usr/bin/env python3
"""PC swap using move_can_pot ep_43 as clean R donor.

Group C:
  C1 frame=mcp_43 (R) + PC=mcp_41 (L)  -> R->L flip
  C2 frame=mcp_41 (L) + PC=mcp_43 (R)  -> L->R flip
  C3 frame=pick_dual_40 + PC=mcp_43 (R) -> dual + good R-PC, kill L arm?

Baselines:
  C0_R = mcp_43 baseline (already in 100ep eval)
  C0_L = mcp_41 baseline (already in pc_swap_v3)
  A0 dualinit_dualPC (already in pc_swap_v3)
"""
from __future__ import annotations
import argparse, os, sys, time
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


def run_one(model, batch, seed, num_steps, guidance, shift):
    torch.manual_seed(seed); np.random.seed(seed)
    ncond = int(batch["num_conditional_frames"].flatten()[0].item())
    cond01 = batch["video"][0, :, :ncond].detach().float().cpu() / 255.0 if ncond > 0 else None
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x0 = model.generate_samples_from_batch(
                data_batch=batch, guidance=guidance, seed=seed,
                num_steps=num_steps, shift=shift,
            )
        v = torch.cat([model.decode(c) for c in x0], dim=3) if isinstance(x0, list) else model.decode(x0)
    v = ((v[0].detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
    if cond01 is not None:
        n = min(cond01.shape[1], v.shape[1])
        v[:, :n] = cond01[:, :n].to(dtype=v.dtype)
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/root/autodl-tmp/v5_d4cn_10tasks_iter5000_pt/model_ema_bf16.pt")
    p.add_argument("--experiment", default="predict2_point_adapter_v5_controlnet_10tasks")
    p.add_argument("--config", default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    p.add_argument("--dataset-dir", default="/root/autodl-tmp/cosmos_training_data_world_arena_ablation_10tasks/test")
    p.add_argument("--num-conditional-frames", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output-dir", default="/root/autodl-tmp/pc_swap_d4cn_10tasks_iter5000_v3")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
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

    PREF = "TianxingChen_RoboTwin2.0_"
    SUF = "_aloha-agilex_ep_"
    ep = lambda task, num: f"{PREF}{task}{SUF}{num}"

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

    EP_MCP_43_R = ep("move_can_pot", "000043")    # clean R
    EP_MCP_41_L = ep("move_can_pot", "000041")    # L
    EP_PICK_DUAL = ep("pick_dual_bottles", "000040")  # DUAL

    print("=== build base batches ===")
    batches = {tag: build(tag) for tag in [EP_MCP_43_R, EP_MCP_41_L, EP_PICK_DUAL]}
    PCs = {tag: (b["pc_latent_x0"].clone(), b["pc_latent_mask"].clone()) for tag, b in batches.items()}

    def set_pc(batch, pc_x0, pc_msk):
        cur_T = batch["pc_latent_x0"].shape[1]; new_T = pc_x0.shape[1]
        if new_T > cur_T: x, m = pc_x0[:, :cur_T], pc_msk[:, :cur_T]
        elif new_T < cur_T:
            pad = cur_T - new_T
            x = torch.cat([pc_x0, pc_x0[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            m = torch.cat([pc_msk, pc_msk[:, -1:].expand(-1, pad, -1)], dim=1)
        else:
            x, m = pc_x0, pc_msk
        batch["pc_latent_x0"] = x.to(device=device, dtype=batch["pc_latent_x0"].dtype)
        batch["pc_latent_mask"] = m.to(device=device, dtype=batch["pc_latent_mask"].dtype)

    runs = [
        ("C0R_mcp43_baseline",          EP_MCP_43_R,  EP_MCP_43_R,  "mcp_43 R-init + R-PC (clean R baseline)"),
        ("C1_mcp43Rinit_LPC_flip",      EP_MCP_43_R,  EP_MCP_41_L,  "mcp_43 R-init + L-PC (R->L flip)"),
        ("C2_mcp41Linit_clean43RPC_flip", EP_MCP_41_L, EP_MCP_43_R, "mcp_41 L-init + clean-43 R-PC (L->R flip)"),
        ("C3_dualinit_clean43RPC",      EP_PICK_DUAL, EP_MCP_43_R,  "dual init + clean-43 R-PC (kill L arm?)"),
    ]
    for tag, init_ep, pc_ep, desc in runs:
        out_path = out_dir / f"{tag}.mp4"
        if out_path.with_suffix(".mp4.mp4").exists() or out_path.exists():
            print(f"  skip {tag} (exists)"); continue
        b = batches[init_ep]
        pc_x0, pc_msk = PCs[pc_ep]
        set_pc(b, pc_x0, pc_msk)
        print(f"\n=== {tag} | {desc} ===")
        t0 = time.time()
        video = run_one(model, b, args.seed, args.num_steps, args.guidance, args.shift)
        print(f"  elapsed {time.time()-t0:.1f}s, video.shape={tuple(video.shape)}")
        save_img_or_video(video, str(out_path), fps=args.fps)
        print(f"  saved → {out_path}.mp4")

    print(f"\n=== DONE. videos in {out_dir}/")


if __name__ == "__main__":
    main()
