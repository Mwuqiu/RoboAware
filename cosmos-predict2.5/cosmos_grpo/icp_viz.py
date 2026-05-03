"""
icp_viz.py — Multi-view ICP alignment visualisation (no GUI required).

Creates a 2×3 subplot PNG from two point clouds (depth / sim-aligned):
  Row 0 : three orthographic projections  XY | XZ | YZ
  Row 1 : 3-D scatter (oblique)  |  per-frame alignment-score bar chart  |  summary text

Usage (standalone) ::

    from cosmos_grpo.icp_viz import save_alignment_viz
    save_alignment_viz(
        out_path="grpo_debug/iter_000001/sample_00/sample_00_alignment.png",
        depth_pts_list=[...],   # list of N×3 arrays (depth, one per ICP frame)
        sim_pts_list=[...],     # list of N×3 arrays (sim AFTER T_global applied)
        frame_scores=[0.31, 0.69, ...],
        frame_ids=[137, 141, ...],
        reward=0.4136,
    )
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEPTH_COLOR  = (0.95, 0.55, 0.05)   # orange  — depth / prediction cloud
_SIM_COLOR    = (0.15, 0.45, 0.95)   # blue    — sim cloud aligned with T_global
_ALPHA_PROJ   = 0.35                  # opacity for projection scatter
_ALPHA_3D     = 0.25                  # opacity for 3-D scatter (smaller = faster render)
_PT_SIZE_PROJ = 0.8                   # scatter marker size for projection plots
_PT_SIZE_3D   = 0.5                   # marker size for 3-D plot


def _subsample(pts: np.ndarray, max_n: int) -> np.ndarray:
    if pts.shape[0] <= max_n:
        return pts
    idx = np.random.choice(pts.shape[0], size=max_n, replace=False)
    return pts[idx]


def _aggregate(pts_list: Sequence[np.ndarray], max_total: int) -> np.ndarray:
    """Stack all per-frame arrays and subsample to at most max_total points."""
    parts = [p for p in pts_list if p is not None and p.shape[0] > 0]
    if not parts:
        return np.empty((0, 3), dtype=np.float64)
    cat = np.concatenate(parts, axis=0).astype(np.float64)
    return _subsample(cat, max_total)


def _score_bar_colors(scores: Sequence[float]) -> List[tuple]:
    """Map scores 0‒1 → colors from red (low) through yellow to green (high)."""
    colors = []
    for s in scores:
        s = float(np.clip(s, 0.0, 1.0))
        r = float(np.clip(2.0 * (1.0 - s), 0.0, 1.0))
        g = float(np.clip(2.0 * s,          0.0, 1.0))
        b = 0.1
        colors.append((r, g, b))
    return colors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_alignment_viz(
    out_path: str,
    depth_pts_list: List[np.ndarray],
    sim_pts_list: List[np.ndarray],
    frame_scores: List[float],
    frame_ids: Optional[List[int]] = None,
    reward: float = 0.0,
    max_pts_proj: int = 8000,
    max_pts_3d: int = 1500,
    dpi: int = 130,
) -> None:
    """Render and save a 2×3 alignment diagnostic figure.

    Parameters
    ----------
    out_path        : output PNG path (directory created automatically)
    depth_pts_list  : depth point clouds per ICP frame (each N×3, camera coords)
    sim_pts_list    : sim point clouds per ICP frame, already transformed with T_global (each N×3)
    frame_scores    : per-frame alignment scores (same length as the two lists)
    frame_ids       : depth frame indices for x-axis labels in the bar chart
    reward          : final scalar reward to annotate
    max_pts_proj    : max total points drawn in each projection plot
    max_pts_3d      : max total points in the 3-D scatter
    dpi             : figure resolution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ── Aggregate point clouds ──────────────────────────────────────────────
    depth_all  = _aggregate(depth_pts_list, max_pts_proj)
    sim_all    = _aggregate(sim_pts_list,   max_pts_proj)
    depth_3d   = _subsample(depth_all, max_pts_3d)
    sim_3d     = _subsample(sim_all,   max_pts_3d)

    n_frames   = len(frame_scores)
    x_labels   = [str(fid) for fid in frame_ids] if frame_ids else [str(i) for i in range(n_frames)]

    reward_clr = (
        (0.1, 0.7, 0.1) if reward >= 0.40 else          # green
        (0.85, 0.65, 0.0) if reward >= 0.20 else          # yellow
        (0.85, 0.15, 0.15)                                 # red
    )

    # ── Figure layout: 2 rows × 3 cols ─────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), dpi=dpi)
    fig.patch.set_facecolor("#111111")

    # GridSpec: bottom row has different heights
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.25,
                  left=0.05, right=0.97, top=0.90, bottom=0.07)

    ax_xy  = fig.add_subplot(gs[0, 0])
    ax_xz  = fig.add_subplot(gs[0, 1])
    ax_yz  = fig.add_subplot(gs[0, 2])
    ax_3d  = fig.add_subplot(gs[1, 0], projection="3d")
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[1, 2])

    _style_ax(ax_xy)
    _style_ax(ax_xz)
    _style_ax(ax_yz)
    _style_ax(ax_bar)
    ax_txt.axis("off")
    ax_txt.set_facecolor("#111111")

    # ── Projection plots ────────────────────────────────────────────────────
    for ax, cols, xlabel, ylabel, title in [
        (ax_xy, (0, 1), "X", "Y", "Top-down  XY"),
        (ax_xz, (0, 2), "X", "Z", "Front     XZ"),
        (ax_yz, (1, 2), "Y", "Z", "Side      YZ"),
    ]:
        c0, c1 = cols
        if depth_all.shape[0] > 0:
            ax.scatter(depth_all[:, c0], depth_all[:, c1],
                       s=_PT_SIZE_PROJ, color=_DEPTH_COLOR, alpha=_ALPHA_PROJ, rasterized=True)
        if sim_all.shape[0] > 0:
            ax.scatter(sim_all[:, c0], sim_all[:, c1],
                       s=_PT_SIZE_PROJ, color=_SIM_COLOR, alpha=_ALPHA_PROJ, rasterized=True)
        ax.set_title(title, color="white", fontsize=9, pad=3)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=7)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    # ── 3-D scatter ─────────────────────────────────────────────────────────
    ax_3d.set_facecolor("#111111")
    for spine in ("left", "right", "bottom", "top"):
        pass  # 3d axes don't have regular spines
    ax_3d.tick_params(colors="#888888", labelsize=6)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor("#333333")
    ax_3d.yaxis.pane.set_edgecolor("#333333")
    ax_3d.zaxis.pane.set_edgecolor("#333333")
    ax_3d.set_xlabel("X", color="#aaaaaa", fontsize=7)
    ax_3d.set_ylabel("Y", color="#aaaaaa", fontsize=7)
    ax_3d.set_zlabel("Z", color="#aaaaaa", fontsize=7)

    if depth_3d.shape[0] > 0:
        ax_3d.scatter(depth_3d[:, 0], depth_3d[:, 1], depth_3d[:, 2],
                      s=_PT_SIZE_3D, color=_DEPTH_COLOR, alpha=_ALPHA_3D,
                      rasterized=True, depthshade=False)
    if sim_3d.shape[0] > 0:
        ax_3d.scatter(sim_3d[:, 0], sim_3d[:, 1], sim_3d[:, 2],
                      s=_PT_SIZE_3D, color=_SIM_COLOR, alpha=_ALPHA_3D,
                      rasterized=True, depthshade=False)

    # Match 3-D axis limits to projection plots
    all_xyz = np.concatenate([a for a in [depth_3d, sim_3d] if a.shape[0] > 0], axis=0) \
        if (depth_3d.shape[0] + sim_3d.shape[0]) > 0 else np.zeros((1, 3))
    lo, hi = all_xyz.min(0) - 0.05, all_xyz.max(0) + 0.05
    mid = (lo + hi) / 2
    half = float(np.max(hi - lo)) / 2 + 0.02
    ax_3d.set_xlim(mid[0] - half, mid[0] + half)
    ax_3d.set_ylim(mid[1] - half, mid[1] + half)
    ax_3d.set_zlim(mid[2] - half, mid[2] + half)
    ax_3d.view_init(elev=25, azim=45)
    ax_3d.set_title("3-D view (iso)", color="white", fontsize=9, pad=4)

    # ── Per-frame bar chart ─────────────────────────────────────────────────
    bar_clrs = _score_bar_colors(frame_scores)
    x_pos = np.arange(n_frames)
    bars = ax_bar.bar(x_pos, frame_scores, color=bar_clrs, edgecolor="#333333", linewidth=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7, color="#aaaaaa")
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.axhline(0.40, color="#44ff44", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_bar.axhline(0.20, color="#ffaa00", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_bar.set_title("Per-frame alignment score", color="white", fontsize=9, pad=3)
    ax_bar.set_ylabel("score", color="#aaaaaa", fontsize=7)
    ax_bar.set_xlabel("frame id", color="#aaaaaa", fontsize=7)
    # Annotate each bar with its value
    for bar, val in zip(bars, frame_scores):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            float(val) + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=6, color="white",
        )

    # ── Summary text panel ──────────────────────────────────────────────────
    mean_score = float(np.mean(frame_scores)) if frame_scores else 0.0
    n_good = sum(1 for s in frame_scores if s >= 0.20)
    n_bad  = sum(1 for s in frame_scores if s < 0.15)
    depth_total = sum(p.shape[0] for p in depth_pts_list if p is not None)
    sim_total   = sum(p.shape[0] for p in sim_pts_list   if p is not None)

    lines = [
        ("REWARD",            f"{reward:.4f}",      reward_clr),
        ("mean align score",  f"{mean_score:.4f}",  None),
        ("frames",            str(n_frames),         None),
        ("good  (≥0.20)",     str(n_good),           (0.2, 0.8, 0.2)),
        ("bad   (<0.15)",     str(n_bad),            (0.85, 0.3, 0.3) if n_bad else None),
        ("depth pts (total)", f"{depth_total:,}",    None),
        ("sim pts  (total)",  f"{sim_total:,}",      None),
    ]
    y = 0.93
    ax_txt.text(0.05, y, "Summary", transform=ax_txt.transAxes,
                fontsize=10, color="white", fontweight="bold")
    y -= 0.08
    for label, value, clr in lines:
        ax_txt.text(0.05, y, label + ":", transform=ax_txt.transAxes,
                    fontsize=8, color="#aaaaaa")
        ax_txt.text(0.62, y, value, transform=ax_txt.transAxes,
                    fontsize=8, color=clr if clr else "white", fontweight="bold")
        y -= 0.09

    # Legend
    y -= 0.04
    from matplotlib.patches import Patch
    leg_handles = [
        Patch(facecolor=_DEPTH_COLOR, label="Depth (pred/gt)"),
        Patch(facecolor=_SIM_COLOR,   label="Sim (T_global)"),
    ]
    ax_txt.legend(handles=leg_handles, loc="lower left",
                  bbox_to_anchor=(0.0, 0.0),
                  fontsize=8, framealpha=0.3,
                  facecolor="#222222", edgecolor="#555555",
                  labelcolor="white")

    # ── Global title ────────────────────────────────────────────────────────
    fig.suptitle(
        f"ICP Alignment  ·  REWARD = {reward:.4f}  ·  {n_frames} frames",
        fontsize=13,
        color=reward_clr,
        fontweight="bold",
        y=0.96,
    )

    # ── Save ────────────────────────────────────────────────────────────────
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _style_ax(ax) -> None:
    """Apply dark-theme styling to a 2-D Axes."""
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#888888", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
