#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np

from cosmos_grpo.reward_online.icp_core import ICPConfig, quality_check, run_icp_multiscale


def _load_sim_frames(path: str, key: str) -> List[np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    arr = np.asarray(obj[key]) if isinstance(obj, dict) else np.asarray(obj)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return [arr[i].astype(np.float64) for i in range(arr.shape[0])]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return [arr.astype(np.float64)]
    raise ValueError(f"Unsupported sim coord shape: {arr.shape}")


def _load_depth_frames(path: str) -> Dict[int, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    points = np.asarray(obj["points"], dtype=np.float64)
    frame_idx = np.asarray(obj["frame_idx"], dtype=np.int32)
    out: Dict[int, np.ndarray] = {}
    for fid in np.unique(frame_idx):
        out[int(fid)] = points[frame_idx == fid]
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sequence ICP in cosmos_grpo.")
    ap.add_argument("--sim_npy", required=True)
    ap.add_argument("--depth_npy", required=True)
    ap.add_argument("--sim_key", default="coord")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--out_json", default="icp_sequence_result.json")
    ap.add_argument("--method", default="p2p", choices=["p2p", "p2l"])
    ap.add_argument("--multiscale", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    sim_frames = _load_sim_frames(args.sim_npy, args.sim_key)
    depth_frames = _load_depth_frames(args.depth_npy)
    depth_ids = sorted(depth_frames.keys())

    if not depth_ids:
        raise RuntimeError("No depth frames found.")

    end = depth_ids[-1] if args.end < 0 else min(args.end, depth_ids[-1])
    selected = [fid for fid in depth_ids if args.start <= fid <= end][:: max(1, args.stride)]

    cfg = ICPConfig(method=args.method, multiscale=bool(args.multiscale))
    T_prev = np.eye(4, dtype=np.float64)

    frame_results: List[Dict[str, Any]] = []
    for depth_fid in selected:
        sim_idx = int(round(args.k * depth_fid + args.offset))
        if not (0 <= sim_idx < len(sim_frames)):
            continue
        src = sim_frames[sim_idx]
        tgt = depth_frames[depth_fid]
        T, fit, rmse = run_icp_multiscale(src, tgt, T_prev, cfg)
        good = quality_check(fit, rmse, cfg)
        frame_results.append(
            {
                "depth_frame_idx": int(depth_fid),
                "sim_idx": int(sim_idx),
                "fitness": float(fit),
                "inlier_rmse": float(rmse),
                "good": bool(good),
                "T_sim_to_depth": T.tolist(),
            }
        )
        T_prev = T

    result = {
        "meta": {
            "sim_npy": args.sim_npy,
            "depth_npy": args.depth_npy,
            "method": args.method,
            "multiscale": bool(args.multiscale),
            "frames": len(frame_results),
        },
        "frames": frame_results,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
