#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np

from cosmos_grpo.reward_online.alignment_metrics import (
    aggregate_metric_dicts,
    compute_alignment_metrics,
    compute_nn_distances,
)


def _load_sim_frames(path: str, key: str) -> list[np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    arr = np.asarray(obj[key]) if isinstance(obj, dict) else np.asarray(obj)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return [arr[i].astype(np.float64) for i in range(arr.shape[0])]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return [arr.astype(np.float64)]
    raise ValueError(f"Unsupported sim coord shape: {arr.shape}")


def _load_depth_frames(path: str) -> dict[int, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    points = np.asarray(obj["points"], dtype=np.float64)
    frame_idx = np.asarray(obj["frame_idx"], dtype=np.int32)
    out: dict[int, np.ndarray] = {}
    for fid in np.unique(frame_idx):
        out[int(fid)] = points[frame_idx == fid]
    return out


def _apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
    return (T @ pts_h.T).T[:, :3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate alignment in cosmos_grpo.")
    ap.add_argument("--sim_npy", required=True)
    ap.add_argument("--depth_npy", required=True)
    ap.add_argument("--icp_json", required=True)
    ap.add_argument("--sim_key", default="coord")
    ap.add_argument("--outlier_thresh", type=float, default=0.05)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    sim_frames = _load_sim_frames(args.sim_npy, args.sim_key)
    depth_frames = _load_depth_frames(args.depth_npy)
    with open(args.icp_json, "r", encoding="utf-8") as f:
        icp = json.load(f)

    per_frame = []
    for item in icp.get("frames", []):
        d = int(item["depth_frame_idx"])
        s = int(item["sim_idx"])
        if d not in depth_frames or not (0 <= s < len(sim_frames)):
            continue
        T = np.asarray(item["T_sim_to_depth"], dtype=np.float64)
        src = _apply_transform(sim_frames[s], T)
        tgt = depth_frames[d]
        dists = compute_nn_distances(src, tgt)
        m = compute_alignment_metrics(dists, args.outlier_thresh)
        m["depth_frame_idx"] = d
        m["sim_idx"] = s
        per_frame.append(m)

    summary = aggregate_metric_dicts(per_frame)
    print(json.dumps(summary, indent=2))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)


if __name__ == "__main__":
    main()
