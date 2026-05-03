#!/usr/bin/env python3
"""Standalone reward replay from saved debug intermediates.

Loads pred/gt depth pointcloud npy files and the corresponding dataset sim
pointcloud, then re-runs the EXACT same ICP + reward computation as reward.py
(_score_single_video), without needing the SAM/Depth pipeline.

All default parameter values match config/grpo_so100_point_adapter.py defaults.

Usage examples (run as module from project root to avoid import collision)
---------------------------------------------------------------------------
# Replay a single sample (pred only)
python -m cosmos_grpo.replay_reward \
    --pred_npy cosmos-output/grpo_debug_v0/iter_000001/sample_00/sample_00_pred_pointcloud.npy \
    --sim_npy  datasets/cosmos_so100_point/pointclouds/Beyond_Success_1031_Cleaned__ep_000000.npy \
    --icp_init datasets/cosmos_so100_point/icp_init/Beyond_Success_1031_Cleaned__ep_000000.json \
    --out_json /tmp/replay_result.json --verbose

# With GT comparison
python -m cosmos_grpo.replay_reward \
    --pred_npy ... --gt_npy ... --sim_npy ... --icp_init ... \
    --out_json /tmp/replay_result.json --verbose

# Tune ICP thresholds
python -m cosmos_grpo.replay_reward ... --corr_coarse 0.3 --no_global_refine
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from cosmos_grpo.reward_online.icp_core import (
    CoarseSearchConfig,
    ICPConfig,
    coarse_global_search,
    compute_alignment_quality,
    quality_check,
    refine_global_T,
    run_icp_multiscale,
    score_from_icp,
    weighted_average_transforms,
)
from cosmos_grpo.reward_online.alignment_metrics import (
    aggregate_metric_dicts,
    compute_alignment_metrics,
    compute_nn_distances,
)


# ---------------------------------------------------------------------------
# Replay config (mirrors CosmosGRPOConfig reward fields)
# ---------------------------------------------------------------------------

@dataclass
class ReplayConfig:
    # ICP
    icp_method: str = "p2p"
    icp_multiscale: bool = True
    icp_voxel_coarse: float = 0.02
    icp_voxel_medium: float = 0.015
    icp_voxel_fine: float = 0.01
    icp_corr_coarse: float = 0.20
    icp_corr_medium: float = 0.15
    icp_corr_fine: float = 0.10
    icp_iters_coarse: int = 60
    icp_iters_medium: int = 40
    icp_iters_fine: int = 30
    icp_voxel: float = 0.01
    icp_max_corr: float = 0.10
    icp_iters: int = 50
    icp_fitness_thresh: float = 0.3
    icp_rmse_thresh: float = 0.2
    # Global fusion
    icp_global_alpha: float = 10.0
    icp_global_top_k: Optional[int] = None
    icp_global_refine: bool = True
    icp_global_refine_max_frames: int = 30
    # Reward formula
    reward_weight_fitness: float = 1.0
    reward_weight_rmse: float = 1.0
    reward_bias: float = 0.0
    reward_weight_alignment: float = 1.0
    reward_weight_local_icp: float = 0.0
    frame_agg: str = "mean"
    outlier_thresh: float = 0.05
    # Frame pairing (k * global_frame + offset = sim_idx)
    frame_pair_k: float = 1.0
    frame_pair_offset: float = 0.0
    max_icp_pairs: int = 16
    failure_fallback_reward: float = 0.0
    # If True, each frame restarts ICP from T_init rather than T_prev (sliding window).
    # Useful when T_sim_to_depth is a fixed extrinsic and T_init quality is poor.
    per_frame_init: bool = False


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_depth_npy(path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (points N×3, frame_idx N, episode_id)."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"depth npy must be a dict, got {type(obj)}")
    pts = np.asarray(obj["points"], dtype=np.float64)
    fid = np.asarray(obj["frame_idx"], dtype=np.int32)
    ep = str(obj.get("episode_id", ""))
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid], fid[valid], ep


def load_sim_npy(path: str, key: str = "coord") -> Dict[int, np.ndarray]:
    """Return dict {global_frame_idx -> (N,3) float64}."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    arr = np.asarray(obj[key]) if isinstance(obj, dict) else np.asarray(obj)
    out: Dict[int, np.ndarray] = {}
    if arr.ndim == 3 and arr.shape[-1] == 3:
        for i in range(arr.shape[0]):
            pts = arr[i].astype(np.float64)
            if pts.shape[0] > 0:
                out[i] = pts
    elif arr.dtype == object and arr.ndim == 1:
        for i, x in enumerate(arr):
            pts = np.asarray(x, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
                out[i] = pts
    else:
        raise ValueError(f"Unsupported sim array shape: {arr.shape}")
    return out


def load_icp_init(path: str) -> Tuple[np.ndarray, float, float]:
    """Return (T_sim_to_depth 4×4, sim_scale, depth_scale)."""
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    T_raw = j.get("T_sim_to_depth")
    if T_raw is None:
        raise KeyError(f"'T_sim_to_depth' not found in {path}")
    T = np.array(T_raw, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_sim_to_depth must be 4×4, got {T.shape}")
    meta = j.get("meta", {}) or {}
    sim_scale = float(meta.get("sim_scale", 1.0)) or 1.0
    depth_scale = float(meta.get("depth_scale", 1.0)) or 1.0
    return T, sim_scale, depth_scale


# ---------------------------------------------------------------------------
# Scale helper (identical to reward.py _scale_points_centroid)
# ---------------------------------------------------------------------------

def _scale_pts(pts: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0 or pts.shape[0] == 0:
        return pts
    c = pts.mean(0)
    return (pts - c) * scale + c


# ---------------------------------------------------------------------------
# Build frame pairs (mirrors reward.py pred_sim_pairs logic)
# ---------------------------------------------------------------------------
def build_frame_pairs(
    depth_pts: np.ndarray,
    depth_fids: np.ndarray,
    sim_seq: Dict[int, np.ndarray],
    cfg: ReplayConfig,
) -> List[Tuple[int, int]]:
    """Return list of (depth_fid, sim_global_idx) in frame order, capped at max_icp_pairs."""
    unique_fids = sorted(np.unique(depth_fids).tolist())
    pairs: List[Tuple[int, int]] = []
    for depth_fid in unique_fids:
        sim_global = int(round(cfg.frame_pair_k * depth_fid + cfg.frame_pair_offset))
        if sim_global in sim_seq and sim_seq[sim_global].shape[0] > 0:
            pairs.append((int(depth_fid), sim_global))
        if len(pairs) >= cfg.max_icp_pairs:
            break
    return pairs


# ---------------------------------------------------------------------------
# Core replay — exact mirror of reward.py _score_single_video ICP block
# ---------------------------------------------------------------------------

def replay_icp(
    depth_pts: np.ndarray,
    depth_fids: np.ndarray,
    sim_seq: Dict[int, np.ndarray],
    T_init: np.ndarray,
    sim_scale: float,
    depth_scale: float,
    icp_cfg: ICPConfig,
    cfg: ReplayConfig,
    verbose: bool = False,
    label: str = "pred",
    per_frame_init: bool = False,
    coarse_cfg: Optional[CoarseSearchConfig] = None,
) -> Dict:
    """Exact reproduction of reward.py _score_single_video ICP + reward logic.

    per_frame_init=True : each frame restarts ICP from T_init (independent).
    per_frame_init=False: sliding window — each frame starts from previous T.
    coarse_cfg         : if enabled, run grid search before each frame's ICP.

    Returns a result dict with the same structure as icp.json written by reward.py.
    """
    pairs = build_sim_pairs(depth_pts, depth_fids, sim_seq, cfg)
    if not pairs:
        return {"error": "no valid frame pairs", "reward": cfg.failure_fallback_reward}

    T_prev = T_init.copy()
    frame_records: List[Dict] = []
    frame_rewards: List[float] = []
    local_icp_rewards: List[float] = []
    frame_metric_list: List[Dict] = []
    good_pairs_for_refine: List[Tuple[np.ndarray, np.ndarray]] = []

    for depth_fid, sim_global_idx in pairs:
        src = _scale_pts(sim_seq[sim_global_idx], sim_scale)
        tgt_raw = depth_pts[depth_fids == depth_fid]
        tgt = _scale_pts(tgt_raw, depth_scale)

        # ── Per-frame ICP ────────────────────────────────────────────────────
        # per_frame_init=True : each frame is independent (restarts from T_init)
        # per_frame_init=False: sliding window (each frame updates T_prev)
        if per_frame_init or cfg.per_frame_init:
            T_prev = T_init.copy()
        # Coarse search: SE(3) grid search to escape bad T_init
        if coarse_cfg is not None and coarse_cfg.enabled:
            if verbose:
                from scipy.spatial import cKDTree as _cKDTree
                _s = src[np.random.default_rng(0).choice(src.shape[0], min(src.shape[0], 500), replace=False)]
                _t = tgt[np.random.default_rng(0).choice(tgt.shape[0], min(tgt.shape[0], 500), replace=False)]
                _kdt = _cKDTree(_t)
                _h = np.hstack([_s, np.ones((_s.shape[0], 1))])
                _d0, _ = _kdt.query((T_prev @ _h.T).T[:, :3])
                T_prev = coarse_global_search(src, tgt, T_prev, coarse_cfg, icp_cfg)
                _d1, _ = _kdt.query((T_prev @ _h.T).T[:, :3])
                print(f"    [coarse] df={depth_fid} init_nn={_d0.mean():.4f}m → best_nn={_d1.mean():.4f}m")
            else:
                T_prev = coarse_global_search(src, tgt, T_prev, coarse_cfg, icp_cfg)
        T_prev, fitness, rmse = run_icp_multiscale(src, tgt, T_prev, icp_cfg)

        local_icp_reward = score_from_icp(
            fitness, rmse,
            cfg.reward_weight_fitness, cfg.reward_weight_rmse, cfg.reward_bias,
        )
        local_icp_rewards.append(local_icp_reward)
        frame_rewards.append(local_icp_reward)

        # ── Per-frame alignment metrics ──────────────────────────────────────
        metric_this_frame: Dict = {}
        if src.shape[0] > 0 and tgt.shape[0] > 0:
            src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
            src_aligned = (T_prev @ src_h.T).T[:, :3]
            dists = compute_nn_distances(src_aligned, tgt)
            metric_this_frame = compute_alignment_metrics(dists, cfg.outlier_thresh)
            frame_metric_list.append(metric_this_frame)
            frame_rewards[-1] = float(np.clip(
                cfg.reward_weight_alignment * float(metric_this_frame.get("alignment_score", 0.0))
                + cfg.reward_weight_local_icp * local_icp_reward,
                0.0, 1.0,
            ))

        is_good = quality_check(fitness, rmse, icp_cfg)

        if metric_this_frame.get("alignment_score", 0.0) > 0.0:
            pair_score = float(metric_this_frame["alignment_score"])
        else:
            pair_score = float(fitness - cfg.icp_global_alpha * rmse)

        if src.shape[0] > 0 and tgt.shape[0] > 0:
            if is_good:
                good_pairs_for_refine.append((src, tgt))

        rec = {
            "depth_fid": int(depth_fid),
            "sim_global_idx": int(sim_global_idx),
            "fitness": float(fitness),
            "rmse": float(rmse),
            "reward": float(frame_rewards[-1]),
            "local_icp_reward": float(local_icp_reward),
            "alignment_score": float(metric_this_frame.get("alignment_score", 0.0)),
            "mean_dist": float(metric_this_frame.get("mean_dist", 1.0)),
            "p90_dist": float(metric_this_frame.get("p90_dist", 1.0)),
            "outlier_ratio": float(metric_this_frame.get("outlier_ratio", 1.0)),
            "good": bool(is_good),
            "pair_score": float(pair_score),
            "T_refined": T_prev.tolist(),
        }
        frame_records.append(rec)

        if verbose:
            print(
                f"  [{label}] df={depth_fid} sf={sim_global_idx} "
                f"fit={fitness:.4f} rmse={rmse:.4f} good={is_good} "
                f"score={metric_this_frame.get('alignment_score', 0.0):.4f} "
                f"frame_reward={frame_rewards[-1]:.4f}"
            )

    # ── Global T fusion (weighted average of good-frame T candidates) ────────
    candidates = [r for r in frame_records if r["good"]]
    if not candidates:
        candidates = frame_records
    candidates = sorted(candidates, key=lambda r: r["pair_score"], reverse=True)
    if cfg.icp_global_top_k is not None:
        candidates = candidates[: max(1, int(cfg.icp_global_top_k))]

    T_candidates = [np.array(r["T_refined"], dtype=np.float64) for r in candidates]
    w_candidates = np.array([max(r["pair_score"], 1e-6) for r in candidates], dtype=np.float64)
    T_global = weighted_average_transforms(T_candidates, w_candidates)

    global_refine_fitness = 0.0
    global_refine_rmse = 1.0
    if cfg.icp_global_refine:
        T_global, global_refine_fitness, global_refine_rmse = refine_global_T(
            T_global, good_pairs_for_refine, icp_cfg,
            max_frames=cfg.icp_global_refine_max_frames,
        )

    # ── Evaluate T_global on all frames ─────────────────────────────────────
    global_metric_list: List[Dict] = []
    for depth_fid, sim_global_idx in pairs:
        src = _scale_pts(sim_seq[sim_global_idx], sim_scale)
        tgt = _scale_pts(depth_pts[depth_fids == depth_fid], depth_scale)
        if src.shape[0] == 0 or tgt.shape[0] == 0:
            continue
        src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
        src_aligned = (T_global @ src_h.T).T[:, :3]
        dists = compute_nn_distances(src_aligned, tgt)
        global_metric_list.append(compute_alignment_metrics(dists, cfg.outlier_thresh))

    global_alignment_summary = aggregate_metric_dicts(global_metric_list)
    global_alignment_reward = float(global_alignment_summary.get("avg_alignment_score", 0.0))

    # ── Final reward (exact formula from reward.py) ─────────────────────────
    if cfg.frame_agg == "median":
        local_reward = float(np.median(frame_rewards))
    else:
        local_reward = float(np.mean(frame_rewards))

    reward = float(np.clip(
        cfg.reward_weight_alignment * global_alignment_reward
        + cfg.reward_weight_local_icp * local_reward,
        0.0, 1.0,
    ))

    alignment_summary = aggregate_metric_dicts(frame_metric_list)

    if verbose:
        print(
            f"  [{label}] global_alignment={global_alignment_reward:.4f}  "
            f"local={local_reward:.4f}  REWARD={reward:.4f}"
        )

    return {
        "label": label,
        "reward": reward,
        "reward_local": local_reward,
        "reward_global_alignment": global_alignment_reward,
        "frame_agg": cfg.frame_agg,
        "global": {
            "num_candidates": len(candidates),
            "global_refine": cfg.icp_global_refine,
            "global_refine_fitness": global_refine_fitness,
            "global_refine_rmse": global_refine_rmse,
            "T_global": T_global.tolist(),
            "good_pairs_for_refine": len(good_pairs_for_refine),
        },
        "frame_records": frame_records,
        "alignment_summary": alignment_summary,
        "global_alignment_summary": global_alignment_summary,
    }


# ---------------------------------------------------------------------------
# T_init diagnostics
# ---------------------------------------------------------------------------

def diagnose_init(
    depth_pts: np.ndarray,
    depth_fids: np.ndarray,
    sim_seq: Dict[int, np.ndarray],
    T_init: np.ndarray,
    sim_scale: float,
    depth_scale: float,
    cfg: ReplayConfig,
) -> None:
    print("\n[Diagnostics] T_init centroid distance check")
    unique = sorted(np.unique(depth_fids).tolist())[:5]
    dists_before, dists_after = [], []
    for fid in unique:
        sim_idx = int(round(cfg.frame_pair_k * fid + cfg.frame_pair_offset))
        if sim_idx not in sim_seq:
            continue
        src = _scale_pts(sim_seq[sim_idx], sim_scale)
        tgt = _scale_pts(depth_pts[depth_fids == fid], depth_scale)
        if src.shape[0] == 0 or tgt.shape[0] == 0:
            continue
        dists_before.append(float(np.linalg.norm(src.mean(0) - tgt.mean(0))))
        src_h = np.hstack([src, np.ones((src.shape[0], 1))])
        aligned = (T_init @ src_h.T).T[:, :3]
        dists_after.append(float(np.linalg.norm(aligned.mean(0) - tgt.mean(0))))
        print(
            f"  frame {fid}: before={dists_before[-1]:.4f}  "
            f"after_T_init={dists_after[-1]:.4f}  "
            f"depth_pts={depth_pts[depth_fids==fid].shape[0]}  sim_pts={src.shape[0]}"
        )
    if dists_before:
        avg_after = float(np.mean(dists_after))
        print(f"  avg: before={np.mean(dists_before):.4f}  after_T_init={avg_after:.4f}")
        if avg_after > 0.15:
            print(f"  [WARNING] centroid dist after T_init={avg_after:.3f}m is large.")
        else:
            print("  [OK] centroid distance after T_init looks reasonable.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_icp_cfg(args) -> ICPConfig:
    return ICPConfig(
        method=args.icp_method,
        multiscale=args.icp_multiscale,
        voxel_coarse=args.voxel_coarse,
        voxel_medium=args.voxel_medium,
        voxel_fine=args.voxel_fine,
        corr_coarse=args.corr_coarse,
        corr_medium=args.corr_medium,
        corr_fine=args.corr_fine,
        iters_coarse=args.iters_coarse,
        iters_medium=args.iters_medium,
        iters_fine=args.iters_fine,
        voxel=args.icp_voxel,
        max_corr=args.icp_max_corr,
        iters=args.icp_iters,
        fitness_thresh=args.fitness_thresh,
        rmse_thresh=args.rmse_thresh,
    )


def build_replay_cfg(args) -> ReplayConfig:
    return ReplayConfig(
        icp_global_alpha=args.global_alpha,
        icp_global_top_k=args.global_top_k,
        icp_global_refine=args.global_refine,
        icp_global_refine_max_frames=args.global_refine_max_frames,
        reward_weight_fitness=args.reward_weight_fitness,
        reward_weight_rmse=args.reward_weight_rmse,
        reward_bias=args.reward_bias,
        reward_weight_alignment=args.reward_weight_alignment,
        reward_weight_local_icp=args.reward_weight_local_icp,
        frame_agg=args.frame_agg,
        outlier_thresh=args.outlier_thresh,
        frame_pair_k=args.frame_pair_k,
        frame_pair_offset=args.frame_pair_offset,
        max_icp_pairs=args.max_icp_pairs,
        per_frame_init=args.per_frame_init,
    )


def build_coarse_cfg(args) -> CoarseSearchConfig:
    return CoarseSearchConfig(
        enabled=args.coarse_search,
        trans_range=args.coarse_trans_range,
        trans_step=args.coarse_trans_step,
        rot_range_deg=args.coarse_rot_range_deg,
        rot_step_deg=args.coarse_rot_step_deg,
        rot_axes=args.coarse_rot_axes,
        top_k_icp=args.coarse_top_k_icp,
        fast_voxel=args.coarse_fast_voxel,
        fast_corr=args.coarse_fast_corr,
        fast_iters=args.coarse_fast_iters,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay ICP+Reward (identical to reward.py) from saved depth pointcloud npy files."
    )
    ap.add_argument("--pred_npy", required=True)
    ap.add_argument("--gt_npy", default="")
    ap.add_argument("--sim_npy", required=True)
    ap.add_argument("--icp_init", required=True)
    ap.add_argument("--sim_key", default="coord")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--verbose", action="store_true")

    # ICP params — defaults match grpo_so100_point_adapter.py
    ap.add_argument("--icp_method", default="p2p", choices=["p2p", "p2l"])
    ap.add_argument("--icp_multiscale", action="store_true", default=True)
    ap.add_argument("--no_multiscale", dest="icp_multiscale", action="store_false")
    ap.add_argument("--voxel_coarse", type=float, default=0.02)
    ap.add_argument("--voxel_medium", type=float, default=0.015)
    ap.add_argument("--voxel_fine", type=float, default=0.01)
    ap.add_argument("--corr_coarse", type=float, default=0.20)
    ap.add_argument("--corr_medium", type=float, default=0.15)
    ap.add_argument("--corr_fine", type=float, default=0.10)
    ap.add_argument("--iters_coarse", type=int, default=60)
    ap.add_argument("--iters_medium", type=int, default=40)
    ap.add_argument("--iters_fine", type=int, default=30)
    ap.add_argument("--icp_voxel", type=float, default=0.01)
    ap.add_argument("--icp_max_corr", type=float, default=0.10)
    ap.add_argument("--icp_iters", type=int, default=50)
    ap.add_argument("--fitness_thresh", type=float, default=0.3)
    ap.add_argument("--rmse_thresh", type=float, default=0.2)

    # Global fusion
    ap.add_argument("--global_alpha", type=float, default=10.0)
    ap.add_argument("--global_top_k", type=int, default=None)
    ap.add_argument("--global_refine", action="store_true", default=True)
    ap.add_argument("--no_global_refine", dest="global_refine", action="store_false")
    ap.add_argument("--global_refine_max_frames", type=int, default=30)

    # Reward formula
    ap.add_argument("--reward_weight_fitness", type=float, default=1.0)
    ap.add_argument("--reward_weight_rmse", type=float, default=1.0)
    ap.add_argument("--reward_bias", type=float, default=0.0)
    ap.add_argument("--reward_weight_alignment", type=float, default=1.0)
    ap.add_argument("--reward_weight_local_icp", type=float, default=0.0)
    ap.add_argument("--frame_agg", default="mean", choices=["mean", "median"])
    ap.add_argument("--outlier_thresh", type=float, default=0.05)

    # Frame pairing
    ap.add_argument("--frame_pair_k", type=float, default=1.0)
    ap.add_argument("--frame_pair_offset", type=float, default=0.0)
    ap.add_argument("--max_icp_pairs", type=int, default=16)
    ap.add_argument(
        "--per_frame_init", action="store_true", default=False,
        help="Each frame independently restarts ICP from T_init instead of sliding window T_prev. "
             "Better when T_sim_to_depth is fixed extrinsic and T_init quality is poor.",
    )

    # Coarse search (secondary initialization before ICP)
    ap.add_argument(
        "--coarse_search", action="store_true", default=False,
        help="Enable SE(3) grid search before each frame's ICP to escape bad T_init.",
    )
    ap.add_argument("--coarse_trans_range", type=float, default=0.10,
                    help="±translation search range in meters (default: 0.10)")
    ap.add_argument("--coarse_trans_step", type=float, default=0.05,
                    help="Translation step in meters (default: 0.05 → 5 vals/axis → 125 trans)")
    ap.add_argument("--coarse_rot_range_deg", type=float, default=30.0,
                    help="±rotation search range in degrees (default: 30)")
    ap.add_argument("--coarse_rot_step_deg", type=float, default=15.0,
                    help="Rotation step in degrees (default: 15 → 5 values)")
    ap.add_argument("--coarse_rot_axes", default="z", choices=["none", "z", "xyz"],
                    help="Which rotation axes to search (default: z only)")
    ap.add_argument("--coarse_top_k_icp", type=int, default=10,
                    help="Quick ICP on top-K NN-scored candidates (default: 10)")
    ap.add_argument("--coarse_fast_voxel", type=float, default=0.03)
    ap.add_argument("--coarse_fast_corr", type=float, default=0.25)
    ap.add_argument("--coarse_fast_iters", type=int, default=10)

    # Scale overrides (0 = use value from icp_init json)
    ap.add_argument("--sim_scale", type=float, default=0.0)
    ap.add_argument("--depth_scale", type=float, default=0.0)

    args = ap.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading pred:  {args.pred_npy}")
    pred_pts, pred_fids, ep_id = load_depth_npy(args.pred_npy)
    print(f"  episode_id={ep_id}  points={pred_pts.shape[0]}  "
          f"frames={np.unique(pred_fids).shape[0]} [{pred_fids.min()}..{pred_fids.max()}]")

    gt_pts = gt_fids = None
    if args.gt_npy:
        print(f"Loading gt:    {args.gt_npy}")
        gt_pts, gt_fids, _ = load_depth_npy(args.gt_npy)
        print(f"  points={gt_pts.shape[0]}  frames={np.unique(gt_fids).shape[0]} "
              f"[{gt_fids.min()}..{gt_fids.max()}]")

    print(f"Loading sim:   {args.sim_npy}")
    sim_seq = load_sim_npy(args.sim_npy, key=args.sim_key)
    print(f"  total sim frames: {len(sim_seq)}")

    print(f"Loading init:  {args.icp_init}")
    T_init, sim_scale_json, depth_scale_json = load_icp_init(args.icp_init)
    sim_scale = args.sim_scale if args.sim_scale > 0 else sim_scale_json
    depth_scale = args.depth_scale if args.depth_scale > 0 else depth_scale_json
    print(f"  sim_scale={sim_scale}  depth_scale={depth_scale}")

    # ── Sanity check ───────────────────────────────────────────────────────
    overlap = set(np.unique(pred_fids).tolist()) & set(sim_seq.keys())
    print(f"\n[Frame alignment] pred [{pred_fids.min()}..{pred_fids.max()}]  "
          f"sim [0..{max(sim_seq.keys())}]  overlap={len(overlap)}/{np.unique(pred_fids).shape[0]}")
    if not overlap:
        print("  [ERROR] no overlap between pred frames and sim frames")
        sys.exit(1)

    icp_cfg = build_icp_cfg(args)
    cfg = build_replay_cfg(args)
    coarse_cfg = build_coarse_cfg(args)
    if coarse_cfg.enabled:
        import math
        n_rot = {"none": 1, "z": int(round(2 * args.coarse_rot_range_deg / args.coarse_rot_step_deg)) + 1}
        n_rot_val = n_rot.get(args.coarse_rot_axes,
                              (int(round(2 * args.coarse_rot_range_deg / args.coarse_rot_step_deg)) + 1) ** 3)
        n_trans = (int(round(2 * args.coarse_trans_range / args.coarse_trans_step)) + 1) ** 3
        print(f"[Coarse search] enabled: ~{n_trans * n_rot_val} candidates/frame, "
              f"top_k={coarse_cfg.top_k_icp}, rot_axes={coarse_cfg.rot_axes}")

    diagnose_init(pred_pts, pred_fids, sim_seq, T_init, sim_scale, depth_scale, cfg)

    # ── Run pred ───────────────────────────────────────────────────────────
    print(f"\n[Replay] pred vs sim")
    pred_result = replay_icp(
        pred_pts, pred_fids, sim_seq, T_init, sim_scale, depth_scale,
        icp_cfg, cfg, verbose=args.verbose, label="pred", coarse_cfg=coarse_cfg,
    )

    # ── Run gt (optional) ──────────────────────────────────────────────────
    gt_result: Optional[Dict] = None
    if gt_pts is not None and gt_fids is not None:
        print(f"\n[Replay] gt vs sim")
        gt_result = replay_icp(
            gt_pts, gt_fids, sim_seq, T_init, sim_scale, depth_scale,
            icp_cfg, cfg, verbose=args.verbose, label="gt", coarse_cfg=coarse_cfg,
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for rec in pred_result.get("frame_records", []):
        print(
            f"  pred df={rec['depth_fid']} sf={rec['sim_global_idx']} "
            f"fit={rec['fitness']:.4f} good={rec['good']} "
            f"score={rec['alignment_score']:.4f} reward={rec['reward']:.4f}"
        )
    print(f"\n  pred  REWARD={pred_result['reward']:.4f}  "
          f"(global_align={pred_result['reward_global_alignment']:.4f}  "
          f"local={pred_result['reward_local']:.4f})")
    if gt_result:
        print(f"  gt    REWARD={gt_result['reward']:.4f}  "
              f"(global_align={gt_result['reward_global_alignment']:.4f}  "
              f"local={gt_result['reward_local']:.4f})")
        ratio = pred_result["reward"] / max(gt_result["reward"], 1e-6)
        print(f"  pred/gt ratio = {ratio:.3f}  (ideal for GRPO: > 1 after training)")

    # ── Save ───────────────────────────────────────────────────────────────
    if args.out_json:
        out = {
            "pred_npy": args.pred_npy,
            "sim_npy": args.sim_npy,
            "icp_init": args.icp_init,
            "sim_scale": sim_scale,
            "depth_scale": depth_scale,
            "icp_params": vars(args),
            "pred": pred_result,
            "gt": gt_result,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to: {args.out_json}")


if __name__ == "__main__":
    main()
