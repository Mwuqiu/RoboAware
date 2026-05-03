from __future__ import annotations

from typing import Dict, List

import numpy as np
import open3d as o3d


def compute_nn_distances(source_pts: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor distances from source to target points."""
    if source_pts.size == 0 or target_pts.size == 0:
        return np.zeros((0,), dtype=np.float64)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    dists = np.zeros((source_pts.shape[0],), dtype=np.float64)
    for i, pt in enumerate(source_pts.astype(np.float64)):
        _, _, dist_sq = tree.search_knn_vector_3d(pt, 1)
        dists[i] = float(np.sqrt(dist_sq[0]))
    return dists


def compute_alignment_metrics(distances: np.ndarray, outlier_thresh: float) -> Dict[str, float]:
    """Return robust alignment metrics from NN distances."""
    if distances.size == 0:
        return {
            "num_points": 0,
            "mean_dist": 1.0,
            "median_dist": 1.0,
            "p90_dist": 1.0,
            "p95_dist": 1.0,
            "max_dist": 1.0,
            "outlier_ratio": 1.0,
            "alignment_score": 0.0,
        }

    metrics = {
        "num_points": int(distances.size),
        "mean_dist": float(np.mean(distances)),
        "median_dist": float(np.median(distances)),
        "p90_dist": float(np.percentile(distances, 90)),
        "p95_dist": float(np.percentile(distances, 95)),
        "max_dist": float(np.max(distances)),
        "outlier_ratio": float(np.mean(distances > outlier_thresh)),
    }
    score = 1.0 / (
        1.0
        + 10.0 * metrics["mean_dist"]
        + 5.0 * metrics["p90_dist"]
        + 20.0 * metrics["outlier_ratio"]
    )
    metrics["alignment_score"] = float(score)
    return metrics


def aggregate_metric_dicts(per_frame: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate frame-level metrics into a single dictionary."""
    if not per_frame:
        return {
            "frames": 0,
            "avg_mean_dist": 1.0,
            "avg_p90_dist": 1.0,
            "avg_outlier_ratio": 1.0,
            "avg_alignment_score": 0.0,
        }

    mean_dist = float(np.mean([m["mean_dist"] for m in per_frame]))
    p90_dist = float(np.mean([m["p90_dist"] for m in per_frame]))
    outlier = float(np.mean([m["outlier_ratio"] for m in per_frame]))
    score = float(np.mean([m["alignment_score"] for m in per_frame]))
    return {
        "frames": len(per_frame),
        "avg_mean_dist": mean_dist,
        "avg_p90_dist": p90_dist,
        "avg_outlier_ratio": outlier,
        "avg_alignment_score": score,
    }
