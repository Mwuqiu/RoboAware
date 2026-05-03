"""End-to-end sampling + decoding + Open3D visualization.

This script:
1) Loads a trained DiT checkpoint (from `load_encoder.py` training).
2) Samples a latent sequence x0 ~ p(x0) by iterative reverse diffusion (DDPM-style).
3) Decodes sampled latents to point cloud sequence with a separately trained decoder.
4) Visualizes a chosen frame (or plays the sequence) using Open3D, and optionally exports PLYs.

Expected artifacts:
- DiT ckpt: exp/.../dit_ckpt/dit_last.pt (or dit_epochXXX.pt)
- Decoder ckpt: exp/.../decoder_ckpt/decoder_mse_T25_K30.pt

Notes:
- This implementation uses a simple DDPM reverse sampler for v-pred.
- It assumes the same latent shape (T, L/K, C) used during training.

python pointflow/viz_tool.py --mode gt --gt_npy path/to/episode.npy --play
python pointflow/viz_tool.py --mode cache --idx 0 --play
python pointflow/viz_tool.py --mode sample --play


"""

from __future__ import annotations

import os
import math
import argparse
from typing import Optional, Tuple

import torch

from pf_dit import PointCloudDiT, DiffusionSchedule, x0_from_v
from pf_decoder import LatentToPointDecoder


@torch.no_grad()
def predict_x0_from_v(xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    """Given v-pred, reconstruct x0."""
    return x0_from_v(xt=xt, t=t, schedule=schedule, v=v)


@torch.no_grad()
def predict_eps_from_v(xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    """Given v-pred, reconstruct eps.

    eps = sqrt(1-ab)*xt + sqrt(ab)*v
    """
    B = xt.shape[0]
    s_ab = schedule.sqrt_alpha_bar.index_select(0, t).view(B, 1, 1, 1)
    s_om = schedule.sqrt_one_minus_alpha_bar.index_select(0, t).view(B, 1, 1, 1)
    eps = s_om * xt + s_ab * v
    return eps


@torch.no_grad()
def ddpm_sample_v(
    dit: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: Tuple[int, int, int, int],
    mask: Optional[torch.Tensor],
    device: torch.device,
    clip_x0: Optional[float] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """DDPM ancestral sampling for v-pred model.

    shape: (B, T, L, C)
    mask:  [B, T, L] bool or None. If None, uses all-valid mask.

    Returns:
      x0_hat: [B,T,L,C] (final predicted x0 at t=0)
    """
    B, T, L, C = shape
    if mask is None:
        mask = torch.ones((B, T, L), dtype=torch.bool, device=device)

    x = torch.randn((B, T, L, C), device=device)

    for ti in reversed(range(schedule.num_steps)):
        t = torch.full((B,), ti, device=device, dtype=torch.long)

        v_pred = dit(x, t, mask=mask)
        x0 = predict_x0_from_v(x, t, v_pred, schedule)
        eps = predict_eps_from_v(x, t, v_pred, schedule)

        if clip_x0 is not None and math.isfinite(float(clip_x0)):
            x0 = x0.clamp(-float(clip_x0), float(clip_x0))

        if ti == 0:
            x = x0
            break

        beta_t = schedule.betas[ti]
        alpha_t = schedule.alphas[ti]
        abar_t = schedule.alpha_bar[ti]
        abar_prev = schedule.alpha_bar[ti - 1]

        # posterior mean coefficients
        # mu = (sqrt(abar_prev)*beta/(1-abar))*x0 + (sqrt(alpha)*(1-abar_prev)/(1-abar))*x_t
        coef_x0 = torch.sqrt(abar_prev) * beta_t / (1.0 - abar_t)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - abar_prev) / (1.0 - abar_t)

        mu = coef_x0 * x0 + coef_xt * x

        # posterior variance
        var = beta_t * (1.0 - abar_prev) / (1.0 - abar_t)
        noise = torch.randn_like(x)
        x = mu + torch.sqrt(var) * noise

        if verbose and (ti % 100 == 0 or ti == schedule.num_steps - 1):
            x0_std = float(x0.std(unbiased=False).item())
            x_std = float(x.std(unbiased=False).item())
            print(f"[sample] t={ti:04d} x_std={x_std:.4f} x0_std={x0_std:.4f}")

    return x


def _build_decoder_from_ckpt(ckpt: dict, device: torch.device, in_channels: int) -> LatentToPointDecoder:
    # Train script may save args inside ckpt, but don’t assume; fallback to common defaults.
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # common keys used in our train_decoder.py
    num_points_target = int(args.get("num_points_target", 2000))
    d_model = int(args.get("d_model", 512))
    depth = int(args.get("depth", 6))
    heads = int(args.get("heads", 8))

    dec = LatentToPointDecoder(
        in_channels=int(in_channels),
        d_model=d_model,
        depth=depth,
        num_heads=heads,
        num_points=num_points_target,
    ).to(device)

    # decoder ckpt formats we support:
    # - {"state_dict": <weights>, "args": {...}}
    # - {"model": <weights>, ...}
    # - <bare state_dict>
    state = ckpt
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]

    missing, unexpected = dec.load_state_dict(state, strict=True)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[decoder] missing={len(missing)} unexpected={len(unexpected)}")

    dec.eval()
    return dec


def _build_dit_from_ckpt(ckpt: dict, device: torch.device) -> PointCloudDiT:
    cfg = ckpt.get("config", {})

    # Prefer explicit shapes saved by training script.
    input_size = int(ckpt.get("shape_L", cfg.get("input_size", 30)))
    in_channels = int(ckpt.get("shape_C", cfg.get("in_channels", 512)))
    num_frames = int(ckpt.get("shape_T", cfg.get("num_frames", 25)))

    hidden_size = int(cfg.get("hidden_size", 512))
    depth = int(cfg.get("depth", 12))
    num_heads = int(cfg.get("num_heads", 8))

    def _make_dit(L: int) -> PointCloudDiT:
        return PointCloudDiT(
            input_size=int(L),
            in_channels=int(in_channels),
            hidden_size=int(hidden_size),
            depth=int(depth),
            num_heads=int(num_heads),
            num_frames=int(num_frames),
        ).to(device)

    state = ckpt.get("model", ckpt)

    # First try with the shape_L recorded in checkpoint.
    dit = _make_dit(input_size)
    try:
        dit.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # Common failure: L (num tokens) mismatch -> positional embedding mismatch.
        if "pos_embed_spatial" in str(e) and isinstance(state, dict) and ("pos_embed_spatial" in state):
            pe = state["pos_embed_spatial"]
            if isinstance(pe, torch.Tensor) and pe.ndim >= 3:
                L_ckpt = int(pe.shape[-2])  # [1,1,L,H]
                print(
                    f"[dit] pos_embed_spatial mismatch; rebuilding DiT with input_size={L_ckpt} to match checkpoint (was {input_size})."
                )
                dit = _make_dit(L_ckpt)
                dit.load_state_dict(state, strict=True)
            else:
                raise RuntimeError(
                    f"[dit] Failed to infer L from pos_embed_spatial; original error: {e}"
                )
        else:
            raise

    dit.eval()
    return dit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        choices=["gt", "cache", "sample"],
        default="cache",
        help="Which mode to run: gt=visualize an episode npy; cache=decode from latent cache; sample=DiT sampling.",
    )
    ap.add_argument(
        "--dit_ckpt",
        type=str,
        default="exp/so100/semseg-pt-v3m1-0-base-only-grid/dit_ckpt/dit_last.pt",
        help="DiT checkpoint (used in --mode sample).",
    )
    ap.add_argument(
        "--decoder_ckpt",
        type=str,
        default="exp/so100/semseg-pt-v3m1-0-base-only-grid/decoder_ckpt/decoder_mse_T25_K30.pt",
        help="Decoder checkpoint (used in --mode cache/sample).",
    )
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--num_steps", type=int, default=None, help="override num_steps; by default read from dit_ckpt")
    ap.add_argument("--clip_x0", type=float, default=None)

    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--mask_all_valid", action="store_true", help="use all-valid mask for sampling")

    # ---- visualize ground-truth episode ----
    ap.add_argument(
        "--gt_npy",
        type=str,
        default=None,
        help="Path to an exported episode .npy dict (expects key 'coord' as [T,N,3]).",
    )
    ap.add_argument("--gt_length", type=int, default=250, help="window length for GT visualization (default: 25)")
    ap.add_argument(
        "--gt_start",
        type=int,
        default=None,
        help="start frame for GT window (0-based). If omitted and --gt_random_window is set, choose randomly.",
    )
    ap.add_argument("--gt_random_window", action="store_true", help="randomly pick a GT window of length --gt_length")
    ap.add_argument("--seed", type=int, default=None, help="random seed for --gt_random_window")

    # ---- decode-from-cache mode ----
    ap.add_argument(
        "--cache_dir",
        type=str,
        default="exp/so100/semseg-pt-v3m1-0-base-only-grid/latent_cache/T25_K30",
        help="Latent cache directory containing latents.dat/masks.dat/meta.json (used in --mode cache).",
    )
    ap.add_argument("--idx", type=int, default=0, help="Sample index for --cache_dir mode")

    # i/o
    ap.add_argument("--out", type=str, default=None, help="Optional output .pt file to save decoded xyz (for cache/sampling modes)")
    ap.add_argument("--export_ply", type=str, default=None, help="Optional output directory to export pred/gt ply for a single frame")
    ap.add_argument("--export_ply_seq", type=str, default=None, help="Optional output directory to export pred/gt ply for all frames")

    # visualization / export
    ap.add_argument("--frame", type=int, default=0, help="frame index to show/export")
    ap.add_argument("--play", action="store_true", help="play the whole sequence in Open3D (interactive controls)")
    ap.add_argument("--fps", type=float, default=10.0)

    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Ensure open3d is available if any visualization/export is requested
    need_o3d = bool(args.play or args.export_ply or args.export_ply_seq)

    # ------------------------------
    # 1) GT-only path
    # ------------------------------
    if args.mode == "gt":
        if args.gt_npy is None:
            raise SystemExit("--mode gt requires --gt_npy")

        import numpy as np

        npy_path = os.path.abspath(args.gt_npy)
        ep = np.load(npy_path, allow_pickle=True)
        ep = ep.item() if hasattr(ep, "item") else ep
        if not isinstance(ep, dict):
            raise SystemExit(f"--gt_npy must be a dict saved by np.save(...). Got type={type(ep)}")
        if "coord" not in ep:
            raise SystemExit(f"--gt_npy dict missing key 'coord'. Available keys: {list(ep.keys())}")

        coord = ep["coord"]
        if not (hasattr(coord, "shape") and len(coord.shape) == 3 and coord.shape[-1] == 3):
            raise SystemExit(f"coord must be [T,N,3], got shape={getattr(coord, 'shape', None)}")

        T_full, N, _ = coord.shape
        print(f"[gt] loaded: {npy_path}")
        print(f"[gt] coord shape: T={T_full} N={N}")

        win_len = int(args.gt_length)
        win_len = max(1, min(win_len, int(T_full)))

        if args.gt_start is not None:
            start = int(args.gt_start)
        elif args.gt_random_window:
            rng = np.random.default_rng(int(args.seed) if args.seed is not None else None)
            start = int(rng.integers(0, max(1, T_full - win_len + 1)))
        else:
            start = 0

        start = max(0, min(start, max(0, T_full - win_len)))
        end = start + win_len
        coord_win = coord[start:end]
        print(f"[gt] window: start={start} end={end} len={win_len}")

        xyz_t = torch.from_numpy(coord_win).to(torch.float32)  # [T,N,3]

        if need_o3d:
            try:
                import open3d as o3d  # noqa: F401
            except Exception as e:
                raise SystemExit(f"open3d import failed: {e}")

        if args.export_ply_seq:
            out_dir = os.path.abspath(args.export_ply_seq)
            os.makedirs(out_dir, exist_ok=True)
            import open3d as o3d

            for t in range(int(xyz_t.shape[0])):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_t[t].numpy())
                out = os.path.join(out_dir, f"gt_s{start:06d}_t{t:03d}.ply")
                o3d.io.write_point_cloud(out, pcd)
            print(f"[export] wrote {int(xyz_t.shape[0])} gt ply files to: {out_dir}")

        if args.play:
            _play_sequence_o3d(
                xyz_pred=xyz_t,
                xyz_gt=None,
                fps=float(args.fps),
                window_name="GT episode (coord)",
                pred_color=(0.9, 0.2, 0.2),
            )
        else:
            if need_o3d:
                import open3d as o3d

                t = int(args.frame)
                t = max(0, min(int(xyz_t.shape[0]) - 1, t))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_t[t].numpy())
                pcd.paint_uniform_color([0.9, 0.2, 0.2])
                o3d.visualization.draw_geometries([pcd], window_name=f"GT episode (coord) start={start} frame={t}")

        return

    # ------------------------------
    # 2) Decode-from-cache path (requires --cache_dir + --decoder_ckpt)
    # ------------------------------
    if args.mode == "cache":
        if args.cache_dir is None:
            raise SystemExit("--mode cache requires --cache_dir")
        if args.decoder_ckpt is None:
            raise SystemExit("--mode cache requires --decoder_ckpt")

        from pf_latent_cache import CachedLatentDataset

        cache_dir = os.path.abspath(args.cache_dir)
        ds = CachedLatentDataset(cache_dir)
        sample = ds[int(args.idx)]
        z = sample["x0"].unsqueeze(0).to(device)  # [1,T,L,C]
        m = sample["mask"].unsqueeze(0).to(device)  # [1,T,L]

        dec_ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
        decoder = _build_decoder_from_ckpt(dec_ckpt, device=device, in_channels=int(z.shape[-1]))

        with torch.no_grad():
            xyz = decoder(z, mask=m)  # [1,T,N,3]

        xyz_t = xyz[0].detach().cpu()
        print(f"[cache] decoded xyz shape: {tuple(xyz_t.shape)}")
        print(
            f"[cache] xyz stats: mean={xyz_t.mean().item():.4e} std={xyz_t.std(unbiased=False).item():.4e} absmax={xyz_t.abs().max().item():.4e}"
        )

        # optional GT overlay
        xyz_gt = None
        if args.gt_npy is not None:
            import numpy as np

            ep = np.load(args.gt_npy, allow_pickle=True)
            if isinstance(ep, np.ndarray) and ep.dtype == object:
                ep = ep.item()
            if isinstance(ep, dict) and "coord" in ep:
                gt_seq = ep["coord"]
            else:
                gt_seq = ep
            if gt_seq.ndim == 2:
                gt_seq = gt_seq[None, ...]
            if gt_seq.ndim != 3 or gt_seq.shape[-1] != 3:
                raise SystemExit(f"gt_npy must be [T,N,3] (or dict with 'coord'), got {getattr(gt_seq, 'shape', None)}")
            xyz_gt = torch.from_numpy(gt_seq).to(torch.float32)

        if args.out:
            out_path = os.path.abspath(args.out)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            torch.save({"xyz": xyz_t, "idx": int(args.idx), "cache_dir": str(cache_dir)}, out_path)
            print(f"[cache] saved decoded xyz to: {out_path}")

        if need_o3d:
            try:
                import open3d as o3d  # noqa: F401
            except Exception as e:
                raise SystemExit(f"open3d import failed: {e}")

        if args.export_ply:
            out_dir = os.path.abspath(args.export_ply)
            os.makedirs(out_dir, exist_ok=True)
            import open3d as o3d

            t = int(args.frame)
            t = max(0, min(int(xyz_t.shape[0]) - 1, t))

            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(xyz_t[t].numpy())
            o3d.io.write_point_cloud(os.path.join(out_dir, f"pred_idx{int(args.idx)}_t{t}.ply"), pcd_pred)

            if xyz_gt is not None and t < int(xyz_gt.shape[0]):
                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(xyz_gt[t].numpy())
                o3d.io.write_point_cloud(os.path.join(out_dir, f"gt_idx{int(args.idx)}_t{t}.ply"), pcd_gt)

            print(f"[export] wrote ply(s) to: {out_dir}")

        if args.export_ply_seq:
            out_dir = os.path.abspath(args.export_ply_seq)
            os.makedirs(out_dir, exist_ok=True)
            import open3d as o3d

            for t in range(int(xyz_t.shape[0])):
                pcd_pred = o3d.geometry.PointCloud()
                pcd_pred.points = o3d.utility.Vector3dVector(xyz_t[t].numpy())
                o3d.io.write_point_cloud(os.path.join(out_dir, f"pred_idx{int(args.idx)}_t{t:03d}.ply"), pcd_pred)

                if xyz_gt is not None and t < int(xyz_gt.shape[0]):
                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(xyz_gt[t].numpy())
                    o3d.io.write_point_cloud(os.path.join(out_dir, f"gt_idx{int(args.idx)}_t{t:03d}.ply"), pcd_gt)

            print(f"[export] wrote sequence ply(s) to: {out_dir}")

        if args.play:
            _play_sequence_o3d(
                xyz_pred=xyz_t,
                xyz_gt=xyz_gt,
                fps=float(args.fps),
                window_name=f"decode cache idx={int(args.idx)}",
                pred_color=(0.2, 0.8, 0.2),
                gt_color=(0.9, 0.2, 0.2),
            )
        else:
            if need_o3d:
                import open3d as o3d

                t = int(args.frame)
                t = max(0, min(int(xyz_t.shape[0]) - 1, t))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_t[t].numpy())
                pcd.paint_uniform_color([0.2, 0.8, 0.2])
                o3d.visualization.draw_geometries([pcd], window_name=f"cache decode idx={int(args.idx)} frame={t}")

        return

    # ------------------------------
    # 3) Sampling path (requires --dit_ckpt + --decoder_ckpt)
    # ------------------------------
    if args.mode == "sample":
        if args.dit_ckpt is None:
            raise SystemExit("--mode sample requires --dit_ckpt")
        if args.decoder_ckpt is None:
            raise SystemExit("--mode sample requires --decoder_ckpt")

        dit_ckpt = torch.load(args.dit_ckpt, map_location="cpu")
        dec_ckpt = torch.load(args.decoder_ckpt, map_location="cpu")

        dit = _build_dit_from_ckpt(dit_ckpt, device)

        # shapes from dit ckpt
        T = int(dit_ckpt.get("shape_T", 25))
        L = int(dit_ckpt.get("shape_L", 30))
        C = int(dit_ckpt.get("shape_C", 512))
        B = int(args.B)

        decoder = _build_decoder_from_ckpt(dec_ckpt, device, in_channels=C)

        num_steps = int(args.num_steps) if args.num_steps is not None else int(dit_ckpt.get("num_steps", 1000))
        schedule = DiffusionSchedule(num_steps=num_steps, schedule="linear", device=device)

        if args.mask_all_valid:
            mask = torch.ones((B, T, L), dtype=torch.bool, device=device)
        else:
            # default: use all-valid too. Keeping this branch in case you later want to load real masks.
            mask = torch.ones((B, T, L), dtype=torch.bool, device=device)

        print(f"[info] sampling latents: B={B} T={T} L={L} C={C} steps={num_steps} device={device}")
        x0 = ddpm_sample_v(
            dit=dit,
            schedule=schedule,
            shape=(B, T, L, C),
            mask=mask,
            device=device,
            clip_x0=args.clip_x0,
            verbose=True,
        )

        print("[info] decoding to point clouds...")
        xyz = decoder(x0, mask=mask)  # [B,T,N,3]
        xyz = xyz.detach().cpu()

        if args.out:
            out_path = os.path.abspath(args.out)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            torch.save(
                {
                    "xyz": xyz[0],
                    "mode": "sample",
                    "dit_ckpt": str(args.dit_ckpt),
                    "decoder_ckpt": str(args.decoder_ckpt),
                    "num_steps": int(num_steps),
                },
                out_path,
            )
            print(f"[sample] saved decoded xyz to: {out_path}")

        if args.export_ply_seq is not None:
            os.makedirs(args.export_ply_seq, exist_ok=True)
            try:
                import open3d as o3d
            except Exception as e:
                raise SystemExit(f"open3d import failed: {e}")

            for t in range(T):
                pts = xyz[0, t].numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                out = os.path.join(args.export_ply_seq, f"sample_t{t:03d}.ply")
                o3d.io.write_point_cloud(out, pcd)
            print(f"[export] wrote {T} ply files to: {args.export_ply_seq}")

        # visualization
        try:
            import open3d as o3d
        except Exception as e:
            raise SystemExit(f"open3d import failed: {e}")

        xyz_t = xyz[0]  # [T,N,3]

        if args.play:
            _play_sequence_o3d(
                xyz_pred=xyz_t,
                xyz_gt=None,
                fps=float(args.fps),
                window_name="DiT sample (decoded)",
                pred_color=(0.1, 0.9, 0.1),
            )
        else:
            t = int(args.frame)
            t = max(0, min(int(xyz_t.shape[0]) - 1, t))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_t[t].numpy())
            pcd.paint_uniform_color([0.1, 0.9, 0.1])
            o3d.visualization.draw_geometries([pcd], window_name=f"DiT sample (decoded) frame={t}")

        return

    raise SystemExit(f"Unknown --mode {args.mode}")


# ------------------------------
# Open3D interactive player (copied from test_decoder and lightly generalized)
# ------------------------------


def _pcd_from_xyz(xyz: torch.Tensor, color=(0.2, 0.8, 0.2)):
    import numpy as np
    import open3d as o3d

    pts = xyz.detach().cpu().numpy().astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    col = np.array(color, dtype=np.float64).reshape(1, 3)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(col, pts.shape[0], axis=0))
    return pcd


def _play_sequence_o3d(
    xyz_pred: torch.Tensor,
    xyz_gt: Optional[torch.Tensor] = None,
    fps: float = 10.0,
    window_name: str = "sequence",
    pred_color=(0.2, 0.8, 0.2),
    gt_color=(0.9, 0.2, 0.2),
    pred_shift=(0.6, 0.0, 0.0),
):
    """Interactive viewer.

    Args:
      xyz_pred: [T,N,3] tensor
      xyz_gt:   [T,N,3] tensor or None
    """

    import time
    import open3d as o3d

    assert xyz_pred.ndim == 3 and int(xyz_pred.shape[-1]) == 3
    T = int(xyz_pred.shape[0])

    state = {"t": 0, "paused": False, "quit": False, "dirty": True}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)

    pcd_pred = _pcd_from_xyz(xyz_pred[0], color=pred_color)
    vis.add_geometry(pcd_pred)

    pcd_gt = None
    if xyz_gt is not None:
        assert xyz_gt.ndim == 3 and int(xyz_gt.shape[-1]) == 3
        pcd_gt = _pcd_from_xyz(xyz_gt[0], color=gt_color)
        # shift pred for clearer comparison
        pcd_pred.translate(tuple(float(x) for x in pred_shift), relative=True)
        vis.add_geometry(pcd_gt)

    def _render_frame(t: int):
        t = int(max(0, min(T - 1, t)))
        pred_pts = xyz_pred[t].detach().cpu().numpy()
        pcd_pred.points = o3d.utility.Vector3dVector(pred_pts)
        vis.update_geometry(pcd_pred)

        if pcd_gt is not None and xyz_gt is not None and t < int(xyz_gt.shape[0]):
            gt_pts = xyz_gt[t].detach().cpu().numpy()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            vis.update_geometry(pcd_gt)

        vis.poll_events()
        vis.update_renderer()

    def _toggle_pause(v):
        state["paused"] = not state["paused"]
        return False

    def _prev(v):
        state["t"] = int(max(0, state["t"] - 1))
        state["dirty"] = True
        return False

    def _next(v):
        state["t"] = int(min(T - 1, state["t"] + 1))
        state["dirty"] = True
        return False

    def _reset(v):
        state["t"] = 0
        state["dirty"] = True
        return False

    def _quit(v):
        state["quit"] = True
        return False

    # controls
    vis.register_key_callback(ord(" "), _toggle_pause)
    vis.register_key_callback(ord("A"), _prev)
    vis.register_key_callback(ord("D"), _next)
    vis.register_key_callback(ord("R"), _reset)
    vis.register_key_callback(ord("Q"), _quit)
    vis.register_key_callback(256, _quit)  # ESC
    try:
        vis.register_key_callback(263, _prev)  # left
        vis.register_key_callback(262, _next)  # right
    except Exception:
        pass

    dt = 1.0 / max(float(fps), 1e-6)
    print("[play] controls: Space=pause/resume, A/Left=prev, D/Right=next, R=reset, Q/Esc=quit")

    while True:
        if state["quit"]:
            break

        if (not state["paused"]) and (not state["dirty"]):
            state["t"] += 1
            if state["t"] >= T:
                state["t"] = 0

        if state["dirty"] or (not state["paused"]):
            _render_frame(state["t"])
            state["dirty"] = False
        else:
            vis.poll_events()
            vis.update_renderer()

        time.sleep(dt)

    vis.destroy_window()


if __name__ == "__main__":
    main()
