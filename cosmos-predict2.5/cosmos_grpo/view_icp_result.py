#!/usr/bin/env python3

import argparse
import json
import os
import time

import numpy as np
import open3d as o3d


def load_sim_npy(path, key="coord"):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    arr = np.asarray(obj[key]) if isinstance(obj, dict) else np.asarray(obj)

    if arr.ndim == 3 and arr.shape[-1] == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return [arr]
    if arr.dtype == object and arr.ndim == 1:
        return [np.asarray(x) for x in arr]
    raise TypeError(f"Unsupported sim array shape/dtype: shape={arr.shape}, dtype={arr.dtype}")


def load_depth_episode_npy(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()

    if not isinstance(obj, dict):
        raise TypeError(f"depth npy root must be dict, got {type(obj)}")

    for k in ("points", "conf", "frame_idx"):
        if k not in obj:
            raise KeyError(f"depth npy missing '{k}', keys={list(obj.keys())}")

    points = np.asarray(obj["points"], dtype=np.float64)
    conf = np.asarray(obj["conf"], dtype=np.float64)
    frame_idx = np.asarray(obj["frame_idx"], dtype=np.int32)
    episode_id = str(obj.get("episode_id", "unknown_episode"))

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    points = points[valid]
    conf = conf[valid]
    frame_idx = frame_idx[valid]
    return episode_id, points, conf, frame_idx


def build_depth_frames(points, conf, frame_idx, max_points_per_frame):
    unique_frames = np.unique(frame_idx)
    frame_points = {}
    frame_conf = {}

    for f in unique_frames:
        mask = frame_idx == f
        pts = points[mask]
        cf = conf[mask]

        if pts.shape[0] > max_points_per_frame:
            sel = np.random.choice(pts.shape[0], size=max_points_per_frame, replace=False)
            pts = pts[sel]
            cf = cf[sel]

        frame_points[int(f)] = pts
        frame_conf[int(f)] = cf

    return [int(f) for f in unique_frames.tolist()], frame_points, frame_conf


def scale_points(points, scale_factor):
    c = np.mean(points, axis=0)
    return (points - c) * scale_factor + c


def conf_to_color(conf: np.ndarray, lo_q=2.0, hi_q=98.0):
    if conf.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    lo = float(np.percentile(conf, lo_q))
    hi = float(np.percentile(conf, hi_q))
    if hi <= lo:
        hi = lo + 1e-6

    norm = np.clip((conf - lo) / (hi - lo), 0.0, 1.0).astype(np.float64)
    r = np.clip(2.0 * norm - 0.2, 0.0, 1.0)
    g = np.clip(2.0 * norm, 0.0, 1.0)
    b = np.clip(1.2 - 2.0 * norm, 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def colormap_blue_to_red(t):
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(2.0 * t - 0.0, 0, 1)
    g = np.clip(2.0 * (1 - np.abs(t - 0.5)), 0, 1)
    b = np.clip(1.0 - 2.0 * t + 0.0, 0, 1)
    return np.stack([r, g, b], axis=-1)


def to_pcd(xyz):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return p


def nn_distance(src_xyz, tgt_pcd):
    if len(tgt_pcd.points) == 0 or src_xyz.shape[0] == 0:
        return np.zeros((src_xyz.shape[0],), dtype=np.float64)

    kdt = o3d.geometry.KDTreeFlann(tgt_pcd)
    d2 = np.empty((src_xyz.shape[0],), dtype=np.float64)

    for i, p in enumerate(src_xyz):
        _, idx, dist2 = kdt.search_knn_vector_3d(p, 1)
        d2[i] = dist2[0] if len(idx) else np.nan

    return np.sqrt(np.nan_to_num(d2, nan=0.0))


def load_T_from_result_json(path, label="pred"):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    # 1. replay_result_v2.json: j[label]["global"]["T_global"] — highest priority
    for lbl in (label, "pred", "gt"):
        section = j.get(lbl)
        if isinstance(section, dict):
            global_block = section.get("global", {})
            if isinstance(global_block, dict) and "T_global" in global_block:
                T = np.array(global_block["T_global"], dtype=np.float64)
                if T.shape == (4, 4):
                    print(f"[T source] {lbl}.global.T_global")
                    return T, j

    # 2. top-level T_sim_to_depth (original reward icp.json)
    for k in ("T_sim_to_depth", "T_sim_to_ply", "T"):
        if k in j:
            return np.array(j[k], dtype=np.float64), j
    if "icp_init" in j and isinstance(j["icp_init"], dict):
        for k in ("T_sim_to_depth", "T_sim_to_ply", "T"):
            if k in j["icp_init"] and j["icp_init"][k] is not None:
                return np.array(j["icp_init"][k], dtype=np.float64), j
    # 3. icp_init string path (fallback — raw hand-tuned T, NOT T_global)
    if "icp_init" in j and isinstance(j["icp_init"], str) and os.path.exists(j["icp_init"]):
        with open(j["icp_init"], "r", encoding="utf-8") as f2:
            ji = json.load(f2)
        for k in ("T_sim_to_depth", "T_sim_to_ply", "T"):
            if k in ji and ji[k] is not None:
                print("[T source] icp_init file (raw hand-tuned T, not T_global!)")
                return np.array(ji[k], dtype=np.float64), j
    if "meta" in j and isinstance(j["meta"], dict):
        for k in ("T_sim_to_depth", "T_sim_to_ply", "T"):
            if k in j["meta"] and j["meta"][k] is not None:
                return np.array(j["meta"][k], dtype=np.float64), j
    raise KeyError("result json missing T_sim_to_depth/T_sim_to_ply/T")


def print_reward_icp_summary(j):
    if not isinstance(j, dict):
        return

    icp_init = j.get("icp_init", {}) if isinstance(j.get("icp_init", {}), dict) else {}
    frame_records = j.get("frame_records", []) if isinstance(j.get("frame_records", []), list) else []
    alignment = j.get("alignment_summary", {}) if isinstance(j.get("alignment_summary", {}), dict) else {}

    if icp_init:
        print("[ICP init]")
        print(f"  path={icp_init.get('path', '')}")
        if "sim_scale" in icp_init or "depth_scale" in icp_init:
            print(
                "  scales: "
                f"sim={icp_init.get('sim_scale', 'n/a')} "
                f"depth={icp_init.get('depth_scale', 'n/a')}"
            )

    if frame_records:
        fitness = np.array([float(r.get("fitness", 0.0)) for r in frame_records], dtype=np.float64)
        rmse = np.array([float(r.get("rmse", 0.0)) for r in frame_records], dtype=np.float64)
        reward = np.array([float(r.get("reward", 0.0)) for r in frame_records], dtype=np.float64)
        print("[ICP per-frame summary]")
        print(
            "  fitness mean/p50/p90="
            f"{fitness.mean():.4f}/{np.percentile(fitness, 50):.4f}/{np.percentile(fitness, 90):.4f}"
        )
        print(
            "  rmse    mean/p50/p90="
            f"{rmse.mean():.4f}/{np.percentile(rmse, 50):.4f}/{np.percentile(rmse, 90):.4f}"
        )
        print(
            "  reward  mean/p50/p90="
            f"{reward.mean():.4f}/{np.percentile(reward, 50):.4f}/{np.percentile(reward, 90):.4f}"
        )

    if alignment:
        print("[Alignment summary]")
        print(
            "  frames={frames} avg_alignment_score={score:.4f} avg_mean_dist={md:.4f} avg_p90_dist={p90:.4f}".format(
                frames=int(alignment.get("frames", 0)),
                score=float(alignment.get("avg_alignment_score", 0.0)),
                md=float(alignment.get("avg_mean_dist", 0.0)),
                p90=float(alignment.get("avg_p90_dist", 0.0)),
            )
        )

    global_info = j.get("global", {}) if isinstance(j.get("global", {}), dict) else {}
    global_align = (
        j.get("global_alignment_summary", {})
        if isinstance(j.get("global_alignment_summary", {}), dict)
        else {}
    )
    if global_info:
        print("[Global ICP]")
        print(
            "  candidates={n} top_k={k} alpha={a:.3f} refine={rf} refine_fitness={fit:.4f} refine_rmse={rmse:.4f}".format(
                n=int(global_info.get("num_candidates", 0)),
                k=global_info.get("global_top_k", None),
                a=float(global_info.get("global_alpha", 0.0)),
                rf=bool(global_info.get("global_refine", False)),
                fit=float(global_info.get("global_refine_fitness", 0.0)),
                rmse=float(global_info.get("global_refine_rmse", 0.0)),
            )
        )
    if global_align:
        print("[Global Alignment summary]")
        print(
            "  frames={frames} avg_alignment_score={score:.4f} avg_mean_dist={md:.4f} avg_p90_dist={p90:.4f}".format(
                frames=int(global_align.get("frames", 0)),
                score=float(global_align.get("avg_alignment_score", 0.0)),
                md=float(global_align.get("avg_mean_dist", 0.0)),
                p90=float(global_align.get("avg_p90_dist", 0.0)),
            )
        )


def build_refined_T_map(j, label="pred"):
    """Return dict {frame_id -> T_refined 4x4}.

    Supports two formats:
    - Original reward icp.json: frame_records at root, key=src_idx
    - replay_result.json: frame_records under pred/gt, key=depth_fid
    """
    refined = {}
    if not isinstance(j, dict):
        return refined

    # root-level frame_records (original icp.json format)
    frame_records = j.get("frame_records", [])
    if not frame_records:
        # replay_result.json: nested under label ('pred' or 'gt')
        for lbl in (label, "pred", "gt"):
            section = j.get(lbl, {})
            if isinstance(section, dict):
                frame_records = section.get("frame_records", [])
                if frame_records:
                    break
    if not isinstance(frame_records, list):
        return refined

    for r in frame_records:
        if not isinstance(r, dict) or "T_refined" not in r:
            continue
        try:
            # prefer depth_fid (replay format), fallback to src_idx (old format)
            if "depth_fid" in r:
                key = int(r["depth_fid"])
            elif "src_idx" in r:
                key = int(r["src_idx"])
            else:
                continue
            T = np.array(r["T_refined"], dtype=np.float64)
            if T.shape == (4, 4):
                refined[key] = T
        except Exception:
            continue
    return refined


def apply_view_if_exists(vis, view_path):
    if not view_path or not os.path.exists(view_path):
        return False
    try:
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(view_path)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        print(f"applied view: {view_path}")
        return True
    except Exception:
        return False


def save_view(vis, view_path):
    if not view_path:
        return
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(view_path, param)
    print(f"saved view: {view_path}")


def render_headless_sequence(
    out_dir,
    frames,
    frame_points,
    frame_conf,
    sim_frames,
    T_global,
    T_refined_by_src_idx,
    args,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    max_frames = len(frames)
    if args.headless_max_frames is not None:
        max_frames = min(max_frames, max(1, int(args.headless_max_frames)))

    for cursor in range(max_frames):
        fid = frames[cursor]
        dxyz = np.asarray(frame_points[fid], dtype=np.float64)
        dconf = np.asarray(frame_conf[fid], dtype=np.float64)
        if args.depth_scale != 1.0 and dxyz.shape[0] > 0:
            dxyz = scale_points(dxyz, args.depth_scale)

        dcol = (
            conf_to_color(dconf, lo_q=args.conf_q_low, hi_q=args.conf_q_high)
            if args.color_mode == "conf"
            else np.repeat(np.array(args.solid_rgb, dtype=np.float64).reshape(1, 3), dxyz.shape[0], axis=0)
        )

        sim_xyz = np.zeros((0, 3), dtype=np.float64)
        sim_idx = None
        title_mode = "play"
        transform_name = "none"
        if args.mode == "icp":
            sim_idx = int(round(args.k * fid + args.offset))
            if 0 <= sim_idx < len(sim_frames):
                sxyz = np.asarray(sim_frames[sim_idx], dtype=np.float64)
                if args.sim_scale != 1.0 and sxyz.shape[0] > 0:
                    sxyz = scale_points(sxyz, args.sim_scale)
                T_use = T_global
                transform_name = "init"
                if args.use_refined_t and fid in T_refined_by_src_idx:
                    T_use = T_refined_by_src_idx[fid]
                    transform_name = "refined"
                sxyz_h = np.hstack([sxyz, np.ones((sxyz.shape[0], 1), dtype=np.float64)])
                sim_xyz = (T_use @ sxyz_h.T).T[:, :3]
                title_mode = f"icp sim={sim_idx} T={transform_name}"

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
        for ax in axes:
            ax.set_facecolor("white" if args.bg == "white" else "black")
            if args.bg == "black":
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_color("white")

        if dxyz.shape[0] > 0:
            axes[0].scatter(dxyz[:, 0], dxyz[:, 1], s=1, c=dcol)
            axes[1].scatter(dxyz[:, 0], dxyz[:, 2], s=1, c=dcol)
        if sim_xyz.shape[0] > 0:
            axes[0].scatter(sim_xyz[:, 0], sim_xyz[:, 1], s=1, c="royalblue", alpha=0.7)
            axes[1].scatter(sim_xyz[:, 0], sim_xyz[:, 2], s=1, c="royalblue", alpha=0.7)

        axes[0].set_title(f"XY | frame={fid} | {title_mode}")
        axes[1].set_title(f"XZ | frame={fid} | {title_mode}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        axes[0].axis("equal")
        axes[1].axis("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{cursor:05d}.png"))
        plt.close(fig)

    print(f"headless frames saved to: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Unified depth episode viewer (play / icp).")
    ap.add_argument("--mode", choices=["play", "icp"], default="play")
    ap.add_argument("--depth_npy", required=True)

    ap.add_argument("--sim_npy", default="")
    ap.add_argument("--sim_key", default="coord")
    ap.add_argument("--result_json", default="")
    ap.add_argument("--use_refined_t", action="store_true",
                    help="If result_json includes per-frame T_refined, use it for per-frame transform.")
    ap.add_argument("--result_label", default="pred", choices=["pred", "gt"],
                    help="Which label section to load T_global/frame_records from (default: pred).")

    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--offset", type=float, default=0.0)

    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--max_points_per_frame", type=int, default=120000)
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--bg", choices=["white", "black"], default="white")

    ap.add_argument("--sim_scale", type=float, default=1.0)
    ap.add_argument("--depth_scale", type=float, default=0.8)

    ap.add_argument("--color_mode", choices=["conf", "solid"], default="conf")
    ap.add_argument("--solid_rgb", type=float, nargs=3, default=[0.1, 0.8, 0.2])
    ap.add_argument("--conf_q_low", type=float, default=2.0)
    ap.add_argument("--conf_q_high", type=float, default=98.0)

    ap.add_argument("--err_clip", type=float, default=0.05)
    ap.add_argument("--show_err_color", action="store_true")

    ap.add_argument("--view", type=str, default="view.json")
    ap.add_argument("--headless_dir", type=str, default="",
                    help="If set, export PNG frames instead of opening an interactive window.")
    ap.add_argument("--headless_max_frames", type=int, default=None,
                    help="Optional cap on exported frame count in headless mode.")
    args = ap.parse_args()

    episode_id, points, conf, frame_idx = load_depth_episode_npy(args.depth_npy)
    frames, frame_points, frame_conf = build_depth_frames(
        points,
        conf,
        frame_idx,
        max_points_per_frame=max(1, int(args.max_points_per_frame)),
    )
    if not frames:
        raise RuntimeError("No valid points after filtering.")

    T_global = None
    T_refined_by_src_idx = {}
    sim_frames = None
    if args.mode == "icp":
        if not args.sim_npy:
            raise RuntimeError("--sim_npy is required in --mode icp")
        if not args.result_json:
            raise RuntimeError("--result_json is required in --mode icp")
        sim_frames = load_sim_npy(args.sim_npy, key=args.sim_key)
        T_global, result_json = load_T_from_result_json(args.result_json, label=args.result_label)
        print(f"loaded result json: {args.result_json} (label={args.result_label})")
        print_reward_icp_summary(result_json)
        if args.use_refined_t:
            T_refined_by_src_idx = build_refined_T_map(result_json, label=args.result_label)
            print(f"[ICP refined T] loaded per-frame transforms: {len(T_refined_by_src_idx)}")

    print(f"mode={args.mode} episode={episode_id}")
    print(f"frames={len(frames)}, frame_range=[{frames[0]}, {frames[-1]}]")
    print(f"scales: sim={args.sim_scale}, depth={args.depth_scale}")

    if args.headless_dir:
        render_headless_sequence(
            out_dir=args.headless_dir,
            frames=frames,
            frame_points=frame_points,
            frame_conf=frame_conf,
            sim_frames=sim_frames,
            T_global=T_global,
            T_refined_by_src_idx=T_refined_by_src_idx,
            args=args,
        )
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    ok = vis.create_window(window_name=f"viewer ({args.mode}) - {episode_id}", width=args.width, height=args.height)

    opt = vis.get_render_option()
    if not ok or opt is None:
        fallback_dir = os.path.join(os.getcwd(), f"view_icp_headless_{episode_id}")
        print(f"interactive window unavailable, falling back to headless export: {fallback_dir}")
        render_headless_sequence(
            out_dir=fallback_dir,
            frames=frames,
            frame_points=frame_points,
            frame_conf=frame_conf,
            sim_frames=sim_frames,
            T_global=T_global,
            T_refined_by_src_idx=T_refined_by_src_idx,
            args=args,
        )
        try:
            vis.destroy_window()
        except Exception:
            pass
        return
    opt.point_size = float(args.point_size)
    opt.background_color = np.asarray([1.0, 1.0, 1.0] if args.bg == "white" else [0.0, 0.0, 0.0])

    depth_pcd = o3d.geometry.PointCloud()
    sim_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(depth_pcd)
    vis.add_geometry(sim_pcd)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(axes)

    state = {
        "paused": False,
        "cursor": 0,
        "last_time": time.time(),
        "show_mode": 1,  # 1 both, 2 depth only, 3 sim only
    }

    def refit_camera(reset_bounding_box=True):
        # Open3D may keep an old camera frustum while geometry changes over time.
        # Re-fitting the camera helps avoid clipping and "incomplete cloud" view.
        vis.reset_view_point(reset_bounding_box)
        vis.poll_events()
        vis.update_renderer()

    def redraw():
        fid = frames[state["cursor"]]
        dxyz = np.asarray(frame_points[fid], dtype=np.float64)
        dconf = np.asarray(frame_conf[fid], dtype=np.float64)

        if args.depth_scale != 1.0 and dxyz.shape[0] > 0:
            dxyz = scale_points(dxyz, args.depth_scale)

        depth_pcd.points = o3d.utility.Vector3dVector(dxyz)
        if args.color_mode == "conf":
            dcol = conf_to_color(dconf, lo_q=args.conf_q_low, hi_q=args.conf_q_high)
        else:
            rgb = np.array(args.solid_rgb, dtype=np.float64).reshape(1, 3)
            dcol = np.repeat(rgb, dxyz.shape[0], axis=0)
        depth_pcd.colors = o3d.utility.Vector3dVector(dcol)

        if args.mode == "play":
            sim_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            sim_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            vis.update_geometry(depth_pcd)
            vis.update_geometry(sim_pcd)
            print(f"frame={fid}, depth_pts={dxyz.shape[0]}")
            return

        sim_idx = int(round(args.k * fid + args.offset))
        if not (0 <= sim_idx < len(sim_frames)):
            sim_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            sim_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            vis.update_geometry(depth_pcd)
            vis.update_geometry(sim_pcd)
            print(f"frame={fid}, sim_idx={sim_idx} out of range")
            return

        sxyz = np.asarray(sim_frames[sim_idx], dtype=np.float64)
        if args.sim_scale != 1.0 and sxyz.shape[0] > 0:
            sxyz = scale_points(sxyz, args.sim_scale)

        T_use = T_global
        using_refined = False
        if args.use_refined_t and fid in T_refined_by_src_idx:
            T_use = T_refined_by_src_idx[fid]
            using_refined = True

        sim_now = to_pcd(sxyz)
        sim_now.transform(T_use)
        sim_xyz = np.asarray(sim_now.points)

        sim_pcd.points = o3d.utility.Vector3dVector(sim_xyz)

        if args.show_err_color and sim_xyz.shape[0] > 0 and dxyz.shape[0] > 0:
            dist = nn_distance(sim_xyz, depth_pcd)
            t = np.clip(dist / max(args.err_clip, 1e-9), 0.0, 1.0)
            scol = colormap_blue_to_red(t)
            p50 = float(np.percentile(dist, 50))
            p90 = float(np.percentile(dist, 90))
        else:
            scol = np.repeat(np.array([[0.1, 0.4, 1.0]], dtype=np.float64), sim_xyz.shape[0], axis=0)
            p50 = 0.0
            p90 = 0.0

        sim_pcd.colors = o3d.utility.Vector3dVector(scol)

        if state["show_mode"] == 2:
            sim_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            sim_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        elif state["show_mode"] == 3:
            depth_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            depth_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        vis.update_geometry(depth_pcd)
        vis.update_geometry(sim_pcd)
        print(
            f"frame={fid} sim={sim_idx} depth_pts={dxyz.shape[0]} sim_pts={sim_xyz.shape[0]} "
            f"NN p50={p50:.4f} p90={p90:.4f} transform={'refined' if using_refined else 'init'}"
        )

    def mark_redraw():
        redraw()
        vis.poll_events()
        vis.update_renderer()

    def cb_pause(_):
        state["paused"] = not state["paused"]
        print("paused" if state["paused"] else "playing")
        return False

    def cb_next(_):
        state["cursor"] = min(state["cursor"] + 1, len(frames) - 1)
        mark_redraw()
        return False

    def cb_prev(_):
        state["cursor"] = max(state["cursor"] - 1, 0)
        mark_redraw()
        return False

    def cb_restart(_):
        state["cursor"] = 0
        mark_redraw()
        return False

    def cb_mode1(_):
        state["show_mode"] = 1
        mark_redraw()
        return False

    def cb_mode2(_):
        state["show_mode"] = 2
        mark_redraw()
        return False

    def cb_mode3(_):
        state["show_mode"] = 3
        mark_redraw()
        return False

    def cb_save_view(_):
        save_view(vis, args.view)
        return False

    def cb_refit(_):
        refit_camera(True)
        print("camera refit to current scene")
        return False

    def cb_quit(_):
        vis.close()
        return False

    vis.register_key_callback(32, cb_pause)
    vis.register_key_callback(ord("N"), cb_next)
    vis.register_key_callback(ord("."), cb_next)
    vis.register_key_callback(ord(","), cb_prev)
    vis.register_key_callback(ord("R"), cb_restart)
    vis.register_key_callback(ord("1"), cb_mode1)
    vis.register_key_callback(ord("2"), cb_mode2)
    vis.register_key_callback(ord("3"), cb_mode3)
    vis.register_key_callback(ord("V"), cb_save_view)
    vis.register_key_callback(ord("C"), cb_refit)
    vis.register_key_callback(ord("c"), cb_refit)
    vis.register_key_callback(ord("Q"), cb_quit)

    print("Controls:")
    print("  Space: play/pause")
    print("  N/.: next frame, ,: prev frame")
    print("  R: restart")
    print("  1/2/3: both/depth-only/sim-only")
    print("  C: camera refit (fix clipping/incomplete view)")
    print("  V: save view, Q: quit")

    mark_redraw()
    has_view = apply_view_if_exists(vis, args.view)
    if not has_view:
        refit_camera(True)

    dt = 1.0 / max(args.fps, 1e-6)
    try:
        while vis.poll_events():
            if not state["paused"]:
                now = time.time()
                if now - state["last_time"] >= dt:
                    state["last_time"] = now
                    state["cursor"] += 1
                    if state["cursor"] >= len(frames):
                        if args.loop:
                            state["cursor"] = 0
                        else:
                            state["cursor"] = len(frames) - 1
                            state["paused"] = True
                            print("Reached end; paused.")
                    redraw()
            vis.update_renderer()
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
