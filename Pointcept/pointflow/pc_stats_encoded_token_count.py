# pc_stats_encoded_token_count.py
from __future__ import annotations

import os, glob, argparse, random
from typing import Dict, Any, List, Optional

import numpy as np
import torch

from pf_encoder import load_ptv3_model, PTV3Encoder


def load_episode_npy(path: str) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True).item()
    if "coord" not in d:
        raise KeyError(f"Missing 'coord' in {path}")
    d["__path__"] = path
    return d


def infer_grid_size_from_encoder(encoder) -> Optional[float]:
    """
    Try to find a reasonable default grid_size from common Pointcept/PTv3 configs.
    Returns float or None if cannot infer.
    """
    # direct attribute
    for name in ["grid_size", "voxel_size"]:
        v = getattr(encoder, name, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass

    # sometimes stored in encoder.model / encoder.backbone
    for obj_name in ["model", "backbone", "net"]:
        obj = getattr(encoder, obj_name, None)
        if obj is None:
            continue
        for name in ["grid_size", "voxel_size"]:
            v = getattr(obj, name, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass

    # look for sparsify/voxelize modules
    # (best-effort; safe to ignore if not found)
    try:
        for m in encoder.modules():
            for name in ["grid_size", "voxel_size"]:
                if hasattr(m, name):
                    v = getattr(m, name)
                    try:
                        return float(v)
                    except Exception:
                        continue
    except Exception:
        pass

    return None


@torch.no_grad()
def count_tokens_per_frame(
    encoder: PTV3Encoder,
    episode: Dict[str, Any],
    frames: List[int],
    grid_size_fallback: Optional[float],
    amp: bool = False,
) -> List[int]:
    coord = episode["coord"]
    T = len(coord)
    device = next(iter(encoder.parameters())).device
    in_ch = int(getattr(encoder.embedding, "in_channels", 0))

    # grid_size is REQUIRED by Point.serialization()
    grid_size = episode.get("grid_size", None)
    if grid_size is None:
        grid_size = grid_size_fallback
    if grid_size is None:
        raise RuntimeError(
            "grid_size missing in episode and cannot infer from encoder. "
            "Please pass --grid_size to this script (must match training/encoding)."
        )

    out_Ni: List[int] = []
    for t in frames:
        if t < 0 or t >= T:
            continue
        fr = coord[t]

        if isinstance(fr, np.ndarray):
            coord_t = torch.from_numpy(fr.astype(np.float32))
        elif isinstance(fr, torch.Tensor):
            coord_t = fr.to(torch.float32)
        else:
            coord_t = torch.tensor(np.array(fr).astype(np.float32))

        N = int(coord_t.shape[0])
        if N == 0:
            out_Ni.append(0)
            continue

        coord_t = coord_t.to(device)
        batch_idx = torch.zeros((N,), dtype=torch.long, device=device)
        offset = torch.tensor([N], dtype=torch.long, device=device)
        feat_init = torch.zeros((N, in_ch), dtype=torch.float32, device=device)

        single = {
            "coord": coord_t,
            "batch": batch_idx,
            "offset": offset,
            "feat": feat_init,
            "grid_size": float(grid_size),
        }

        # debug checks
        if not torch.isfinite(coord_t).all():
            raise RuntimeError("coord contains NaN/Inf")

        # mn = coord_t.min(dim=0).values.cpu().numpy()
        # mx = coord_t.max(dim=0).values.cpu().numpy()
        # print(f"[debug] coord range: min={mn}, max={mx}, N={coord_t.shape[0]}, grid_size={grid_size}")

        if amp and torch.cuda.is_available():
            from torch.cuda.amp import autocast
            with autocast():
                p = encoder.forward(single)
        else:
            p = encoder.forward(single)

        pts_feat = p.get("feat") if hasattr(p, "get") else getattr(p, "feat", None)
        out_Ni.append(int(pts_feat.shape[0]) if pts_feat is not None else 0)

    return out_Ni


def summarize(values: List[int]) -> Dict[str, Any]:
    if len(values) == 0:
        return {"count": 0}
    a = np.array(values, dtype=np.int64)
    return {
        "count": int(a.size),
        "min": int(a.min()),
        "p50": int(np.percentile(a, 50)),
        "p90": int(np.percentile(a, 90)),
        "p95": int(np.percentile(a, 95)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
        "mean": float(a.mean()),
    }


def find_dataset_dirs(root_dir: str) -> List[str]:
    root_dir = os.path.abspath(root_dir)
    ds = []
    for d in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(d):
            continue
        pc_dir = os.path.join(d, "pointclouds")
        if os.path.isdir(pc_dir) and len(glob.glob(os.path.join(pc_dir, "*.npy"))) > 0:
            ds.append(d)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--max_datasets", type=int, default=None)
    ap.add_argument("--episodes_per_dataset", type=int, default=30)
    ap.add_argument("--frames_per_episode", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grid_size", type=float, default=None, help="optional override; use if cannot infer from encoder")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PTV3Encoder(load_ptv3_model()).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    inferred = infer_grid_size_from_encoder(encoder)
    grid_size_fallback = args.grid_size if args.grid_size is not None else inferred
    print(f"grid_size fallback = {grid_size_fallback} (inferred={inferred}, cli={args.grid_size})")

    dataset_dirs = find_dataset_dirs(args.root_dir)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: int(args.max_datasets)]
    if len(dataset_dirs) == 0:
        raise RuntimeError(f"No datasets found under {os.path.abspath(args.root_dir)}")

    all_Ni: List[int] = []

    for ds_i, dataset_dir in enumerate(dataset_dirs):
        pc_dir = os.path.join(dataset_dir, "pointclouds")
        files = sorted(glob.glob(os.path.join(pc_dir, "*.npy")))
        if len(files) == 0:
            continue

        n_pick = min(len(files), int(args.episodes_per_dataset))
        pick_files = random.sample(files, n_pick) if n_pick < len(files) else files

        ds_Ni: List[int] = []
        oom = 0

        print(f"\n[{ds_i+1}/{len(dataset_dirs)}] {os.path.basename(dataset_dir)} episodes={len(files)} sample={n_pick}")

        for path in pick_files:
            ep = load_episode_npy(path)
            T = len(ep["coord"])
            if T <= 0:
                continue
            f_pick = min(int(args.frames_per_episode), T)
            frames = random.sample(range(T), f_pick) if f_pick < T else list(range(T))

            try:
                nis = count_tokens_per_frame(
                    encoder, ep, frames=frames, grid_size_fallback=grid_size_fallback, amp=args.amp
                )
                ds_Ni.extend(nis)
                all_Ni.extend(nis)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        print("  Ni stats:", summarize(ds_Ni), f"oom_episodes={oom}")

    print("\n=== Overall Ni stats ===")
    print(summarize(all_Ni))
    print("Suggestion: pick k around p90/p95 depending on padding tolerance.")


if __name__ == "__main__":
    main()