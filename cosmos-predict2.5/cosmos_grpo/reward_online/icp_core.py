"""Core ICP utilities for Cosmos-GRPO reward computation.

Public API (matches what reward.py / icp_sequence.py / __init__.py import):
  ICPConfig                   – parameter dataclass
  CoarseSearchConfig          – SE(3) grid search config for secondary init
  run_icp_multiscale          – (src_pts, tgt_pts, T_init, cfg) → (T, fitness, rmse)
  coarse_global_search        – SE(3) grid search before ICP to escape bad T_init
  quality_check               – (fitness, rmse, cfg) → bool
  score_from_icp              – (fitness, rmse, a, b, c) → float
  weighted_average_transforms – T-list fusion via quaternion averaging
  refine_global_T             – merged-cloud final ICP pass
  compute_alignment_quality   – NN-distance statistics dict
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Coarse search config
# ---------------------------------------------------------------------------

@dataclass
class CoarseSearchConfig:
    """SE(3) grid search around T_prev to escape a bad hand-tuned T_init.

    Two-pass per frame:
    1. Score all (translation × rotation) candidates by mean NN distance on
       downsampled clouds — fast, no ICP.
    2. Run quick single-scale ICP on the top-K best candidates.
    3. Best T found replaces T_prev as init for the main multiscale ICP.

    Default grid (z-rotation only): 5³ trans × 5 rot = 625 candidates.
    At score_max_pts=500 this takes ~50 ms per frame on CPU.
    """
    enabled: bool = False
    # Translation grid (meters) in depth/camera frame
    trans_range: float = 0.10    # ±range around current translation
    trans_step: float = 0.05     # step → 5 values/axis → 125 trans candidates
    # Rotation grid around sim-cloud centroid in depth frame (degrees)
    rot_range_deg: float = 30.0
    rot_step_deg: float = 15.0
    rot_axes: str = "z"          # "none" | "z" | "xyz"
    # Quick ICP refinement of top-K NN-scored candidates
    top_k_icp: int = 10
    fast_voxel: float = 0.03
    fast_corr: float = 0.25
    fast_iters: int = 10
    score_max_pts: int = 500     # max pts sampled for NN scoring


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ICPConfig:
    method: str = "p2p"         # "p2p" | "p2l"
    multiscale: bool = True

    # multiscale voxel sizes
    voxel_coarse: float = 0.02
    voxel_medium: float = 0.015
    voxel_fine: float = 0.01

    # multiscale correspondence distances
    corr_coarse: float = 0.20
    corr_medium: float = 0.15
    corr_fine: float = 0.10

    # multiscale iteration counts
    iters_coarse: int = 60
    iters_medium: int = 40
    iters_fine: int = 30

    # single-scale fallback params (used when multiscale=False)
    voxel: float = 0.01
    max_corr: float = 0.10
    iters: int = 50

    # point-to-plane normal estimation
    normal_radius: float = 0.05
    normal_maxnn: int = 30

    # quality thresholds (used by quality_check)
    fitness_thresh: float = 0.8
    rmse_thresh: float = 0.07


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return p


def _ensure_normals(pcd: o3d.geometry.PointCloud, cfg: ICPConfig) -> None:
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.normal_radius, max_nn=cfg.normal_maxnn)
    )


# ---------------------------------------------------------------------------
# Core ICP
# ---------------------------------------------------------------------------

def run_icp_multiscale(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    T_init: np.ndarray,
    cfg: ICPConfig,
) -> Tuple[np.ndarray, float, float]:
    """Run multiscale (or single-scale) ICP.

    src = sim point cloud (already scaled),
    tgt = depth point cloud (already scaled).

    Returns (T_refined 4x4, fitness, inlier_rmse).
    """
    if source_pts.shape[0] < 20 or target_pts.shape[0] < 20:
        return T_init.copy(), 0.0, 1.0

    src_pcd = _to_pcd(source_pts)
    tgt_pcd = _to_pcd(target_pts)

    if cfg.multiscale:
        T = T_init.copy()
        fitness = rmse = 0.0
        for vox, corr, n_it in [
            (cfg.voxel_coarse, cfg.corr_coarse, cfg.iters_coarse),
            (cfg.voxel_medium, cfg.corr_medium, cfg.iters_medium),
            (cfg.voxel_fine,   cfg.corr_fine,   cfg.iters_fine),
        ]:
            src_d = src_pcd.voxel_down_sample(vox)
            tgt_d = tgt_pcd.voxel_down_sample(vox)
            if cfg.method == "p2l":
                est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
                _ensure_normals(src_d, cfg)
                _ensure_normals(tgt_d, cfg)
            else:
                est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            reg = o3d.pipelines.registration.registration_icp(
                src_d, tgt_d, corr, T,
                estimation_method=est,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=n_it),
            )
            T, fitness, rmse = reg.transformation, float(reg.fitness), float(reg.inlier_rmse)
        return T, fitness, rmse

    # single-scale fallback
    src_d = src_pcd.voxel_down_sample(cfg.voxel)
    tgt_d = tgt_pcd.voxel_down_sample(cfg.voxel)
    if cfg.method == "p2l":
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        _ensure_normals(src_d, cfg)
        _ensure_normals(tgt_d, cfg)
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    reg = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d, cfg.max_corr, T_init,
        estimation_method=est,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.iters),
    )
    return reg.transformation, float(reg.fitness), float(reg.inlier_rmse)


# ---------------------------------------------------------------------------
# Coarse SE(3) grid search (secondary initialization)
# ---------------------------------------------------------------------------

def coarse_global_search(
    src: np.ndarray,
    tgt: np.ndarray,
    T_prev: np.ndarray,
    coarse_cfg: CoarseSearchConfig,
    icp_cfg: ICPConfig,
) -> np.ndarray:
    """SE(3) grid search in the neighbourhood of T_prev, then refine top candidates
    with quick ICP.  Returns the best T found as a new init for run_icp_multiscale.

    Translation candidates shift the transformed src centroid in depth frame.
    Rotation candidates rotate around the src cloud centroid in depth frame so
    the object stays centred while orientations are scanned.

    coarse_cfg.enabled must be True before calling (caller's responsibility).
    """
    if src.shape[0] < 20 or tgt.shape[0] < 20:
        return T_prev.copy()

    rng = np.random.default_rng(0)

    def _down(pts: np.ndarray, n: int) -> np.ndarray:
        if pts.shape[0] <= n:
            return pts
        return pts[rng.choice(pts.shape[0], n, replace=False)]

    src_s = _down(src, coarse_cfg.score_max_pts)
    tgt_s = _down(tgt, coarse_cfg.score_max_pts)
    tgt_kdt = cKDTree(tgt_s)

    src_mean = src.mean(0)                               # sim-frame centroid
    c0 = T_prev[:3, :3] @ src_mean + T_prev[:3, 3]      # centroid in depth frame

    # Build rotation candidates
    r_vals = np.arange(
        -coarse_cfg.rot_range_deg, coarse_cfg.rot_range_deg + 1e-9, coarse_cfg.rot_step_deg
    )
    if coarse_cfg.rot_axes == "none" or r_vals.size == 0:
        rot_mats: List[np.ndarray] = [np.eye(3)]
    elif coarse_cfg.rot_axes == "z":
        rot_mats = [
            Rotation.from_euler("z", float(a), degrees=True).as_matrix() for a in r_vals
        ]
    else:  # "xyz" — full 3-axis search
        rot_mats = [
            Rotation.from_euler("xyz", [float(rx), float(ry), float(rz)], degrees=True).as_matrix()
            for rx, ry, rz in itertools.product(
                r_vals.tolist(), r_vals.tolist(), r_vals.tolist()
            )
        ]

    t_vals = np.arange(
        -coarse_cfg.trans_range, coarse_cfg.trans_range + 1e-9, coarse_cfg.trans_step
    )
    src_s_h = np.hstack([src_s, np.ones((src_s.shape[0], 1), dtype=np.float64)])

    # Score every (rotation × translation) candidate by mean NN distance
    scored: List[Tuple[float, np.ndarray]] = []
    for R_d in rot_mats:
        R_new = (R_d @ T_prev[:3, :3]).copy()
        # Rotate around depth-frame centroid c0 so object stays approximately centred
        t_base = R_d @ (T_prev[:3, 3] - c0) + c0
        for dx, dy, dz in itertools.product(
            t_vals.tolist(), t_vals.tolist(), t_vals.tolist()
        ):
            t_new = t_base + np.array([dx, dy, dz])
            T_cand = np.eye(4, dtype=np.float64)
            T_cand[:3, :3] = R_new
            T_cand[:3, 3] = t_new
            src_t = (T_cand @ src_s_h.T).T[:, :3]
            dists, _ = tgt_kdt.query(src_t)
            scored.append((float(dists.mean()), T_cand))

    scored.sort(key=lambda x: x[0])

    # Quick single-scale ICP on top-K candidates
    fast_cfg = ICPConfig(
        multiscale=False,
        method=icp_cfg.method,
        voxel=coarse_cfg.fast_voxel,
        max_corr=coarse_cfg.fast_corr,
        iters=coarse_cfg.fast_iters,
        normal_radius=icp_cfg.normal_radius,
        normal_maxnn=icp_cfg.normal_maxnn,
        fitness_thresh=icp_cfg.fitness_thresh,
        rmse_thresh=icp_cfg.rmse_thresh,
    )
    best_T = T_prev.copy()
    best_fitness = -1.0
    for _, T_cand in scored[: max(1, coarse_cfg.top_k_icp)]:
        T_r, fit, _ = run_icp_multiscale(src, tgt, T_cand, fast_cfg)
        if fit > best_fitness:
            best_fitness = fit
            best_T = T_r

    return best_T


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------

def quality_check(fitness: float, rmse: float, cfg: ICPConfig) -> bool:
    """Return True if ICP result passes quality thresholds in cfg."""
    return bool(fitness >= cfg.fitness_thresh and rmse <= cfg.rmse_thresh)


def score_from_icp(fitness: float, rmse: float, a: float, b: float, c: float) -> float:
    """Linear reward: clip(a * fitness - b * rmse + c, 0, 1)."""
    return float(np.clip(a * fitness - b * rmse + c, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Global T fusion
# ---------------------------------------------------------------------------

def weighted_average_transforms(
    T_list: List[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted average of transforms: translation averaged linearly,
    rotation averaged via quaternion SLERP (flip-consistent).
    """
    if not T_list:
        return np.eye(4, dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()

    translations = np.stack([T[:3, 3] for T in T_list], axis=0)
    t_avg = (weights[:, None] * translations).sum(axis=0)

    quats = np.stack([Rotation.from_matrix(T[:3, :3]).as_quat() for T in T_list], axis=0)
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    q_avg = (weights[:, None] * quats).sum(axis=0)
    q_avg /= np.linalg.norm(q_avg)
    R_avg = Rotation.from_quat(q_avg).as_matrix()

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    return T_avg


def refine_global_T(
    T_global_init: np.ndarray,
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    cfg: ICPConfig,
    max_frames: int = 30,
) -> Tuple[np.ndarray, float, float]:
    """Merge all (src, tgt) pairs into single clouds and do one final ICP pass.

    This is the global refinement step from icp_sequence.py.
    """
    if not pairs:
        return T_global_init, 0.0, 1.0

    if len(pairs) > max(1, int(max_frames)):
        step = max(1, len(pairs) // int(max_frames))
        pairs = pairs[::step][: int(max_frames)]

    valid = [(s, t) for s, t in pairs if s.shape[0] > 0 and t.shape[0] > 0]
    if not valid:
        return T_global_init, 0.0, 1.0

    src_merged = _to_pcd(np.vstack([p[0] for p in valid])).voxel_down_sample(cfg.voxel_fine)
    tgt_merged = _to_pcd(np.vstack([p[1] for p in valid])).voxel_down_sample(cfg.voxel_fine)

    if cfg.method == "p2l":
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        _ensure_normals(src_merged, cfg)
        _ensure_normals(tgt_merged, cfg)
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    reg = o3d.pipelines.registration.registration_icp(
        src_merged, tgt_merged,
        cfg.corr_fine,
        T_global_init,
        estimation_method=est,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max(1, cfg.iters_fine * 3)
        ),
    )
    return reg.transformation, float(reg.fitness), float(reg.inlier_rmse)


# ---------------------------------------------------------------------------
# Alignment quality statistics
# ---------------------------------------------------------------------------

def compute_alignment_quality(
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
    T: np.ndarray,
    *,
    max_eval_points: int = 50000,
    outlier_thresh: float = 0.05,
) -> Dict:
    """Transform src by T, compute NN-distances to tgt, return detailed stats.

    Returns dict with keys: valid, num_points, mean_dist, median_dist, std_dist,
    p75/p90/p95/p99_dist, max_dist, outlier_ratio, outlier_thresh, alignment_score.
    alignment_score = max(0, 1 - mean_d/thresh) * (1 - outlier_ratio)
    """
    src = np.asarray(src_xyz, dtype=np.float64)
    tgt = np.asarray(tgt_xyz, dtype=np.float64)

    ones = np.ones((src.shape[0], 1), dtype=np.float64)
    src_t = (T @ np.hstack([src, ones]).T).T[:, :3]

    if src_t.shape[0] > max_eval_points:
        sel = np.random.choice(src_t.shape[0], max_eval_points, replace=False)
        src_t = src_t[sel]

    tgt_pcd = _to_pcd(tgt)
    kdt = o3d.geometry.KDTreeFlann(tgt_pcd)

    dists = np.empty(src_t.shape[0], dtype=np.float64)
    for i in range(src_t.shape[0]):
        _, _, d2 = kdt.search_knn_vector_3d(src_t[i], 1)
        dists[i] = float(np.sqrt(d2[0])) if d2 else np.nan

    dists = dists[np.isfinite(dists)]
    if dists.size == 0:
        return {"valid": False}

    outlier_ratio = float(np.mean(dists > outlier_thresh))
    mean_d = float(np.mean(dists))
    score = max(0.0, 1.0 - mean_d / outlier_thresh) * (1.0 - outlier_ratio)

    return {
        "valid": True,
        "num_points": int(dists.size),
        "mean_dist": mean_d,
        "median_dist": float(np.median(dists)),
        "std_dist": float(np.std(dists)),
        "p75_dist": float(np.percentile(dists, 75)),
        "p90_dist": float(np.percentile(dists, 90)),
        "p95_dist": float(np.percentile(dists, 95)),
        "p99_dist": float(np.percentile(dists, 99)),
        "max_dist": float(np.max(dists)),
        "outlier_ratio": outlier_ratio,
        "outlier_thresh": float(outlier_thresh),
        "alignment_score": float(score),
    }
