# Reward function placeholder for Cosmos GRPO training.
#
# During RL training, the GRPOTrainer calls compute_rewards() after decoding
# rollout latents into pixel-space videos. Replace the stub below with your
# actual reward logic (e.g. trajectory tracking error, VQA score, classifier
# confidence, etc.).
#
# Contract
# --------
# Input:
#   videos    – list of G decoded video tensors, each with shape [B, C, T, H, W],
#               dtype float32, values in [-1, 1] (same range as model outputs).
#   data_batch – the original data batch dict from the dataloader, containing
#               ground-truth video, point-cloud latents, text embeddings, etc.
#
# Output:
#   rewards   – list of G floats (one scalar reward per group member).
#               Higher is better.

from __future__ import annotations

import time
import json
import os
import traceback
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config.grpo_so100_point_adapter import CosmosGRPOConfig
from .icp_viz import save_alignment_viz
from .reward_online import (
    CoarseSearchConfig,
    ICPConfig,
    OnlinePointCloudExtractor,
    OnlineVisionConfig,
    aggregate_metric_dicts,
    build_frame_pairs,
    coarse_global_search,
    compute_alignment_metrics,
    compute_nn_distances,
    quality_check,
    refine_global_T,
    run_icp_multiscale,
    score_from_icp,
    weighted_average_transforms,
)


@dataclass
class RewardDiagnostics:
    fallback_count: int = 0
    sample_count: int = 0
    elapsed_sec: float = 0.0
    mean_reward: float = 0.0
    mean_fitness: float = 0.0
    mean_rmse: float = 0.0
    mean_alignment_score: float = 0.0


_ENGINE: "OnlineRewardEngine | None" = None
_LAST_DIAGNOSTICS: RewardDiagnostics = RewardDiagnostics()
logger = logging.getLogger(__name__)


class OnlineRewardEngine:
    def __init__(self, cfg: CosmosGRPOConfig) -> None:
        self.cfg = cfg
        self._reward_step = 0
        self._debug_dir = cfg.debug_dir or os.path.join(cfg.output_dir, "debug")
        self._annotations_dir = os.path.join(cfg.dataset_dir, "annotations")
        self._sam_point_index = self._load_sam_point_index(self._annotations_dir)
        self._icp_init_dir = os.path.join(cfg.dataset_dir, "icp_init")
        self._icp_init_index = self._load_icp_init_index(self._icp_init_dir)
        self._gt_mask_cache_dir = os.path.join(cfg.dataset_dir, "sam_mask_cache")
        self._sim_pointcloud_dir = os.path.join(cfg.dataset_dir, "pointclouds")
        self.extractor = OnlinePointCloudExtractor(
            OnlineVisionConfig(
                sam2_checkpoint=cfg.sam2_checkpoint,
                sam2_model_cfg=cfg.sam2_model_cfg,
                sam_obj_id=cfg.sam_obj_id,
                depth_model_id=cfg.depth_model_id,
                conf_percentile=cfg.depth_conf_percentile,
                max_frames=cfg.online_max_frames,
                frame_stride=cfg.online_frame_stride,
                max_points_per_frame=cfg.max_points_per_frame,
            )
        )
        self.icp_cfg = ICPConfig(
            method=cfg.icp_method,
            multiscale=cfg.icp_multiscale,
            voxel_coarse=cfg.icp_voxel_coarse,
            voxel_medium=cfg.icp_voxel_medium,
            voxel_fine=cfg.icp_voxel_fine,
            corr_coarse=cfg.icp_corr_coarse,
            corr_medium=cfg.icp_corr_medium,
            corr_fine=cfg.icp_corr_fine,
            iters_coarse=cfg.icp_iters_coarse,
            iters_medium=cfg.icp_iters_medium,
            iters_fine=cfg.icp_iters_fine,
            voxel=cfg.icp_voxel,
            max_corr=cfg.icp_max_corr,
            iters=cfg.icp_iters,
            fitness_thresh=cfg.icp_fitness_thresh,
            rmse_thresh=cfg.icp_rmse_thresh,
        )
        self.coarse_cfg = CoarseSearchConfig(
            enabled=cfg.icp_coarse_search,
            trans_range=cfg.icp_coarse_trans_range,
            trans_step=cfg.icp_coarse_trans_step,
            rot_range_deg=cfg.icp_coarse_rot_range_deg,
            rot_step_deg=cfg.icp_coarse_rot_step_deg,
            rot_axes=cfg.icp_coarse_rot_axes,
            top_k_icp=cfg.icp_coarse_top_k_icp,
            fast_voxel=cfg.icp_coarse_fast_voxel,
            fast_corr=cfg.icp_coarse_fast_corr,
            fast_iters=cfg.icp_coarse_fast_iters,
        )

    @staticmethod
    def _norm_episode_key(key: str) -> str:
        return key.strip().replace("__", "_")

    def _load_sam_point_index(self, annotations_dir: str) -> Dict[str, List[Dict[str, object]]]:
        index: Dict[str, List[Dict[str, object]]] = {}
        if not os.path.isdir(annotations_dir):
            logger.warning("[Reward] annotations dir not found: %s", annotations_dir)
            return index

        for fn in os.listdir(annotations_dir):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(annotations_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                ep = str(ann.get("episode_id", "")).strip()
                point = ann.get("point", None)
                points = ann.get("points", None)
                labels = ann.get("labels", None)
                ann_frame_idx = int(ann.get("ann_frame_idx", 0))
                if not ep:
                    continue

                parsed_points: List[Tuple[float, float]] = []
                if isinstance(points, list):
                    for p in points:
                        if isinstance(p, (list, tuple)) and len(p) == 2:
                            parsed_points.append((float(p[0]), float(p[1])))
                if not parsed_points and point is not None and len(point) == 2:
                    parsed_points = [(float(point[0]), float(point[1]))]
                if not parsed_points:
                    continue

                parsed_labels: List[int] = []
                if isinstance(labels, list) and len(labels) == len(parsed_points):
                    parsed_labels = [1 if int(x) > 0 else 0 for x in labels]
                else:
                    parsed_labels = [1 for _ in parsed_points]

                record = {
                    "points": parsed_points,
                    "labels": parsed_labels,
                    "ann_frame_idx": ann_frame_idx,
                    "path": path,
                }
                # 支持通过episode_id和文件名双键检索
                index.setdefault(self._norm_episode_key(ep), []).append(record)
                stem = os.path.splitext(fn)[0]
                index.setdefault(self._norm_episode_key(stem), []).append(record)
            except Exception:
                logger.exception("[Reward] failed parsing annotation: %s", path)

        logger.info("[Reward] loaded %d annotation keys from %s", len(index), annotations_dir)
        return index

    def _extract_prompt_from_batch(
        self,
        data_batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[int]], int, str, int]:
        episode_raw = data_batch.get("episode_id", None)
        if isinstance(episode_raw, list) and len(episode_raw) > 0:
            episode = str(episode_raw[0])
        elif isinstance(episode_raw, str):
            episode = episode_raw
        else:
            episode = ""

        start_raw = data_batch.get("start_frame", 0)
        if isinstance(start_raw, torch.Tensor):
            start_frame = int(start_raw.flatten()[0].item())
        elif isinstance(start_raw, list) and len(start_raw) > 0:
            v0 = start_raw[0]
            start_frame = int(v0.item()) if isinstance(v0, torch.Tensor) else int(v0)
        else:
            start_frame = int(start_raw) if start_raw is not None else 0

        key = self._norm_episode_key(episode)
        ann_list = self._sam_point_index.get(key, None)
        if not ann_list:
            return None, None, 0, episode, start_frame

        # 当一个episode有多份标注时，选最接近当前chunk起始帧的一份。
        ann = min(ann_list, key=lambda x: abs(int(x["ann_frame_idx"]) - start_frame))

        points = ann["points"]
        labels = ann["labels"]
        ann_frame_idx_global = int(ann["ann_frame_idx"])
        ann_frame_idx_local = ann_frame_idx_global - start_frame
        return points, labels, ann_frame_idx_local, episode, start_frame

    def _load_icp_init_index(self, icp_init_dir: str) -> Dict[str, Dict[str, object]]:
        index: Dict[str, Dict[str, object]] = {}
        if not os.path.isdir(icp_init_dir):
            logger.warning("[Reward] icp_init dir not found: %s", icp_init_dir)
            return index

        for fn in os.listdir(icp_init_dir):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(icp_init_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)

                T_raw = obj.get("T_sim_to_depth", None)
                meta = obj.get("meta", {}) or {}
                ep = str(meta.get("episode_id", "")).strip()
                if T_raw is None:
                    continue
                T = np.asarray(T_raw, dtype=np.float64)
                if T.shape != (4, 4):
                    continue

                sim_scale = float(meta.get("sim_scale", 1.0))
                depth_scale = float(meta.get("depth_scale", 1.0))
                if sim_scale <= 0:
                    sim_scale = 1.0
                if depth_scale <= 0:
                    depth_scale = 1.0

                rec = {
                    "T_sim_to_depth": T,
                    "sim_scale": sim_scale,
                    "depth_scale": depth_scale,
                    "path": path,
                }
                stem = os.path.splitext(fn)[0]
                index[self._norm_episode_key(stem)] = rec
                if ep:
                    index[self._norm_episode_key(ep)] = rec
            except Exception:
                logger.exception("[Reward] failed parsing icp_init json: %s", path)

        logger.info("[Reward] loaded %d icp_init keys from %s", len(index), icp_init_dir)
        return index

    def _get_icp_init_for_episode(self, episode: str) -> Optional[Dict[str, object]]:
        if not episode:
            return None
        return self._icp_init_index.get(self._norm_episode_key(episode), None)

    def _load_sim_sequence(self, episode: str) -> Dict[int, np.ndarray]:
        """Load sim point cloud sequence from dataset pointclouds dir.

        Returns dict mapping global frame index -> (N, 3) float64 array in sim coords.
        """
        path = os.path.join(self._sim_pointcloud_dir, f"{episode}.npy")
        if not os.path.exists(path):
            logger.warning("[Reward] sim pointcloud not found: %s", path)
            return {}
        try:
            obj = np.load(path, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
                obj = obj.item()
            arr = np.asarray(obj["coord"]) if isinstance(obj, dict) else np.asarray(obj)
            out: Dict[int, np.ndarray] = {}
            if arr.ndim == 3 and arr.shape[-1] == 3:
                for i in range(arr.shape[0]):
                    out[i] = arr[i].astype(np.float64)
            elif arr.dtype == object and arr.ndim == 1:
                for i, x in enumerate(arr):
                    pts = np.asarray(x, dtype=np.float64)
                    if pts.ndim == 2 and pts.shape[1] == 3:
                        out[i] = pts
            logger.info("[Reward] loaded sim sequence episode=%s frames=%d path=%s", episode, len(out), path)
            return out
        except Exception:
            logger.exception("[Reward] failed to load sim sequence: %s", path)
            return {}

    @staticmethod
    def _extract_video_path_from_batch(data_batch: Dict[str, torch.Tensor]) -> str:
        video_path_raw = data_batch.get("video_path", "")
        if isinstance(video_path_raw, list) and len(video_path_raw) > 0:
            return str(video_path_raw[0])
        if isinstance(video_path_raw, str):
            return video_path_raw
        return ""

    def _get_episode_cache_dir(self, episode: str) -> str:
        return os.path.join(self._gt_mask_cache_dir, self._norm_episode_key(episode))

    def _ensure_gt_mask_cache(
        self,
        episode: str,
        video_path: str,
        sam_points_xy: Optional[List[Tuple[float, float]]],
        sam_point_labels: Optional[List[int]],
        ann_frame_idx_global: int,
    ) -> Optional[str]:
        if not episode or not video_path or not sam_points_xy:
            return None

        cache_dir = self._get_episode_cache_dir(episode)
        meta_path = os.path.join(cache_dir, "meta.json")
        if os.path.exists(meta_path):
            return cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        logger.info("[Reward] building GT mask cache for episode=%s video=%s", episode, video_path)
        self.extractor.build_video_mask_cache(
            video_path=video_path,
            cache_dir=cache_dir,
            prompt_points_xy=sam_points_xy,
            prompt_labels=sam_point_labels,
            ann_frame_idx=ann_frame_idx_global,
        )
        return cache_dir

    def _get_sampled_global_frame_ids(self, video_tensor: torch.Tensor, start_frame: int) -> List[int]:
        if video_tensor.ndim == 5:
            t = int(video_tensor.shape[2])
        elif video_tensor.ndim == 4:
            t = int(video_tensor.shape[1])
        else:
            return []
        local_ids = list(range(0, t, max(1, int(self.cfg.online_frame_stride))))[: int(self.cfg.online_max_frames)]
        return [int(start_frame + idx) for idx in local_ids]

    def _load_cached_masks_for_frames(self, cache_dir: Optional[str], frame_ids: List[int]) -> Optional[List[np.ndarray]]:
        if not cache_dir:
            return None
        masks: List[np.ndarray] = []
        for frame_id in frame_ids:
            mask = self.extractor.load_mask_from_cache(cache_dir, frame_id)
            if mask is None:
                return None
            masks.append(mask)
        return masks

    @staticmethod
    def _save_video_frames(video_tensor: torch.Tensor, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        if video_tensor.ndim == 5:
            video_tensor = video_tensor[0]
        v = video_tensor.detach().float().cpu()
        if float(v.min()) >= -1.1 and float(v.max()) <= 1.1:
            v = (v + 1.0) * 127.5
        v = v.clamp(0, 255).to(torch.uint8)
        _, t, _, _ = v.shape
        from PIL import Image
        for i in range(t):
            frame = v[:, i].permute(1, 2, 0).numpy()
            Image.fromarray(frame, mode="RGB").save(os.path.join(out_dir, f"{i:05d}.jpg"), quality=95)

    @staticmethod
    def _save_video_mp4(video_tensor: torch.Tensor, out_path: str, fps: int = 8) -> None:
        """Save a video tensor as an mp4 file using ffmpeg."""
        import subprocess, tempfile
        from PIL import Image
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if video_tensor.ndim == 5:
            video_tensor = video_tensor[0]
        v = video_tensor.detach().float().cpu()
        if float(v.min()) >= -1.1 and float(v.max()) <= 1.1:
            v = (v + 1.0) * 127.5
        v = v.clamp(0, 255).to(torch.uint8)
        _, t, h, w = v.shape
        with tempfile.TemporaryDirectory(prefix="grpo_mp4_") as tmp:
            for i in range(t):
                frame = v[:, i].permute(1, 2, 0).numpy()
                Image.fromarray(frame, mode="RGB").save(os.path.join(tmp, f"{i:05d}.jpg"), quality=95)
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", os.path.join(tmp, "%05d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_path,
            ]
            subprocess.run(cmd, check=True)

    @staticmethod
    def _compile_frames_to_mp4(frames_dir: str, out_path: str, fps: int = 8, pattern: str = "%05d.jpg") -> bool:
        """Compile frames from a directory into an mp4 using ffmpeg. Returns True on success."""
        import subprocess
        if not os.path.isdir(frames_dir):
            return False
        try:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, pattern),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_path,
            ]
            subprocess.run(cmd, check=True)
            return True
        except Exception as exc:
            logger.warning("[Reward] compile_frames_to_mp4 failed: %s", exc)
            return False

    @staticmethod
    def _compile_masks_to_mp4(
        masks_dir: str,
        frames_dir: str,
        out_path: str,
        fps: int = 8,
    ) -> bool:
        """Overlay mask PNGs (grayscale) on frame JPGs (RGB) and save as mp4.

        The mask is composited in green (alpha=0.5) over the original frame.
        """
        import subprocess, tempfile, glob
        from PIL import Image

        mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not mask_paths:
            return False
        n = min(len(mask_paths), len(frame_paths)) if frame_paths else len(mask_paths)
        try:
            with tempfile.TemporaryDirectory(prefix="grpo_mask_mp4_") as tmp:
                for i in range(n):
                    mp = mask_paths[i]
                    mask = np.array(Image.open(mp).convert("L"), dtype=np.float32) / 255.0  # [H,W] in [0,1]
                    if i < len(frame_paths):
                        frame = np.array(Image.open(frame_paths[i]).convert("RGB"), dtype=np.float32)
                    else:
                        h, w = mask.shape
                        frame = np.zeros((h, w, 3), dtype=np.float32)
                    # Resize mask to match frame if needed
                    if mask.shape != (frame.shape[0], frame.shape[1]):
                        m_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                        m_img = m_img.resize((frame.shape[1], frame.shape[0]), Image.NEAREST)
                        mask = np.array(m_img, dtype=np.float32) / 255.0
                    # Overlay: green tint where mask=1
                    green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
                    alpha = 0.5
                    m3 = mask[:, :, None]
                    overlay = (1 - alpha * m3) * frame + alpha * m3 * green
                    out_frame = np.clip(overlay, 0, 255).astype(np.uint8)
                    Image.fromarray(out_frame, mode="RGB").save(
                        os.path.join(tmp, f"{i:05d}.jpg"), quality=92
                    )
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", str(fps),
                    "-i", os.path.join(tmp, "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    out_path,
                ]
                subprocess.run(cmd, check=True)
            return True
        except Exception as exc:
            logger.warning("[Reward] compile_masks_to_mp4 failed: %s", exc)
            return False

    @staticmethod
    def _save_pointcloud_sim_npy(seq, out_path: str) -> None:
        """Save a PointCloudSequence in the sim (ragged object array) format.

        Compatible with view_icp_result.py load_sim_npy(path, key='coord').
        Layout: {"coord": np.object_[T], each element is (N_i, 3) float64}
        """
        n = len(seq.points_per_frame)
        arr = np.empty(n, dtype=object)
        for i, pts in enumerate(seq.points_per_frame):
            arr[i] = np.asarray(pts, dtype=np.float64)
        np.save(out_path, {"coord": arr})

    @staticmethod
    def _save_pointcloud_depth_npy(seq, out_path: str, episode_id: str = "", start_frame: int = 0) -> None:
        """Save a PointCloudSequence in the depth-episode format.

        Compatible with view_icp_result.py load_depth_episode_npy(path).
        Layout: {"points": (N,3), "conf": (N,), "frame_idx": (N,), "episode_id": str}
        frame_idx values are GLOBAL (local + start_frame) so they align with dataset sim npy.
        """
        all_pts: List[np.ndarray] = []
        all_frame_idx = []
        for fid, pts in zip(seq.frame_ids, seq.points_per_frame):
            if pts.shape[0] == 0:
                continue
            all_pts.append(np.asarray(pts, dtype=np.float64))
            all_frame_idx.append(np.full(pts.shape[0], fid + start_frame, dtype=np.int32))
        if all_pts:
            points_cat = np.vstack(all_pts)
            frame_idx_cat = np.concatenate(all_frame_idx)
            conf_cat = np.ones(points_cat.shape[0], dtype=np.float64)
        else:
            points_cat = np.zeros((0, 3), dtype=np.float64)
            frame_idx_cat = np.zeros((0,), dtype=np.int32)
            conf_cat = np.zeros((0,), dtype=np.float64)
        obj = {
            "points": points_cat,
            "conf": conf_cat,
            "frame_idx": frame_idx_cat,
            "episode_id": str(episode_id),
        }
        np.save(out_path, obj)

    @staticmethod
    def _scale_points_centroid(pts: np.ndarray, scale: float) -> np.ndarray:
        """Scale point cloud around its centroid (matches reference icp_sequence.py)."""
        if scale == 1.0 or pts.shape[0] == 0:
            return pts
        c = np.mean(pts, axis=0)
        return (pts - c) * scale + c

    def _should_dump_debug(self) -> bool:
        if not self.cfg.debug_save_intermediates:
            return False
        every = max(1, int(self.cfg.debug_save_every))
        return (self._reward_step % every) == 0

    def _iter_debug_dir(self) -> str:
        return os.path.join(self._debug_dir, f"iter_{self._reward_step:06d}")

    def _score_single_video(
        self,
        pred_video: torch.Tensor,
        gt_video: torch.Tensor,
        sim_seq: Dict[int, np.ndarray],
        debug_dir: str | None = None,
        sample_idx: int = 0,
        sam_points_xy: Optional[List[Tuple[float, float]]] = None,
        sam_point_labels: Optional[List[int]] = None,
        ann_frame_idx: int = 0,
        gt_external_masks: Optional[List[np.ndarray]] = None,
        pred_init_mask: Optional[np.ndarray] = None,
        gt_mask_cache_dir: str = "",
        icp_init_T: Optional[np.ndarray] = None,
        icp_sim_scale: float = 1.0,
        icp_depth_scale: float = 1.0,
        icp_init_path: str = "",
        episode_id: str = "",
        start_frame: int = 0,
    ) -> tuple[float, List[float], List[float], Dict[str, float]]:
        # ------------------------------------------------------------------ #
        # Save original videos as mp4 for debugging                          #
        # ------------------------------------------------------------------ #
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            try:
                self._save_video_mp4(
                    pred_video, os.path.join(debug_dir, f"sample_{sample_idx:02d}_pred.mp4")
                )
            except Exception:
                logger.warning("[Reward] failed to save pred video mp4", exc_info=True)
            try:
                self._save_video_mp4(
                    gt_video, os.path.join(debug_dir, f"sample_{sample_idx:02d}_gt.mp4")
                )
            except Exception:
                logger.warning("[Reward] failed to save gt video mp4", exc_info=True)

        pred_seq = self.extractor.extract_pointcloud_sequence(
            pred_video,
            debug_dir=debug_dir,
            save_prefix=f"sample_{sample_idx:02d}_pred",
            prompt_points_xy=sam_points_xy,
            prompt_labels=sam_point_labels,
            ann_frame_idx=ann_frame_idx,
            init_mask=pred_init_mask,
            mask_source=(f"gt_cache:{gt_mask_cache_dir}" if pred_init_mask is not None else None),
        )
        gt_seq = self.extractor.extract_pointcloud_sequence(
            gt_video,
            debug_dir=debug_dir,
            save_prefix=f"sample_{sample_idx:02d}_gt",
            prompt_points_xy=sam_points_xy,
            prompt_labels=sam_point_labels,
            ann_frame_idx=ann_frame_idx,
            external_masks=gt_external_masks,
            mask_source=(f"gt_cache:{gt_mask_cache_dir}" if gt_external_masks is not None else None),
        )

        # ------------------------------------------------------------------ #
        # Save SAM mask mp4 and point cloud npy files for debugging          #
        # ------------------------------------------------------------------ #
        if debug_dir is not None:
            for tag in (f"sample_{sample_idx:02d}_pred", f"sample_{sample_idx:02d}_gt"):
                base = os.path.join(debug_dir, tag)
                masks_dir = os.path.join(base, "masks")
                frames_dir = os.path.join(base, "frames")
                if os.path.isdir(masks_dir):
                    try:
                        self._compile_masks_to_mp4(
                            masks_dir,
                            frames_dir,
                            os.path.join(debug_dir, f"{tag}_sam.mp4"),
                        )
                    except Exception:
                        logger.warning("[Reward] failed to compile SAM mask mp4 for %s", tag, exc_info=True)
            try:
                self._save_pointcloud_depth_npy(
                    pred_seq,
                    os.path.join(debug_dir, f"sample_{sample_idx:02d}_pred_pointcloud.npy"),
                    episode_id=episode_id,
                    start_frame=start_frame,
                )
                self._save_pointcloud_depth_npy(
                    gt_seq,
                    os.path.join(debug_dir, f"sample_{sample_idx:02d}_gt_pointcloud.npy"),
                    episode_id=episode_id,
                    start_frame=start_frame,
                )
            except Exception:
                logger.warning("[Reward] failed to save pointcloud npy files", exc_info=True)

        # Build (pred_local_idx, sim_global_idx) pairs:
        # pred_seq.frame_ids are LOCAL indices (0, stride, 2*stride, ...)
        # global depth frame = start_frame + local_fid
        # sim is indexed by the same global frame number (1:1 fps match by default)
        sim_k = float(self.cfg.frame_pair_k)
        sim_offset = float(self.cfg.frame_pair_offset)
        pred_sim_pairs: List[Tuple[int, int]] = []
        for local_i, local_fid in enumerate(pred_seq.frame_ids):
            global_fid = int(start_frame) + int(local_fid)
            sim_global = int(round(sim_k * global_fid + sim_offset))
            if sim_global in sim_seq and sim_seq[sim_global].shape[0] > 0:
                pred_sim_pairs.append((local_i, sim_global))
            if len(pred_sim_pairs) >= self.cfg.max_icp_pairs:
                break

        if not pred_sim_pairs:
            logger.warning("[Reward] no valid pred-sim frame pairs for episode=%s start_frame=%d", episode_id, start_frame)
            return self.cfg.failure_fallback_reward, [], [], {}

        # T_sim_to_depth: initial transform mapping sim coords → depth camera coords.
        # src = sim point cloud (sim coords), tgt = pred/gt depth point cloud (depth coords).
        T_init_used = icp_init_T.copy() if isinstance(icp_init_T, np.ndarray) and icp_init_T.shape == (4, 4) else np.eye(4, dtype=np.float64)
        T_prev = T_init_used.copy()
        src_scale = float(icp_sim_scale) if float(icp_sim_scale) > 0 else 1.0    # sim point scale
        tgt_scale = float(icp_depth_scale) if float(icp_depth_scale) > 0 else 1.0  # depth point scale
        frame_rewards: List[float] = []
        local_icp_rewards: List[float] = []
        fitness_list: List[float] = []
        rmse_list: List[float] = []
        frame_metric_list: List[Dict[str, float]] = []
        frame_records: List[Dict[str, float]] = []
        all_pairs_for_global: List[Tuple[np.ndarray, np.ndarray]] = []
        # Only good-quality pairs are used for global refinement (matches reference icp_sequence.py).
        good_pairs_for_refine: List[Tuple[np.ndarray, np.ndarray]] = []

        for pred_local_idx, sim_global_idx in pred_sim_pairs:
            # src = sim point cloud in sim coords (will be transformed by T toward depth coords)
            # tgt = pred depth point cloud in depth camera coords
            src = self._scale_points_centroid(sim_seq[sim_global_idx], src_scale)
            tgt = self._scale_points_centroid(pred_seq.points_per_frame[pred_local_idx], tgt_scale)

            # ── Per-frame ICP: first frame uses T_init, subsequent frames use T_prev ──
            # Optional coarse SE(3) search to escape a bad T_init.
            if self.coarse_cfg.enabled:
                T_prev = coarse_global_search(src, tgt, T_prev, self.coarse_cfg, self.icp_cfg)
            T_prev, fitness, rmse = run_icp_multiscale(src, tgt, T_prev, self.icp_cfg)
            fitness_list.append(fitness)
            rmse_list.append(rmse)
            local_icp_reward = score_from_icp(
                fitness,
                rmse,
                self.cfg.reward_weight_fitness,
                self.cfg.reward_weight_rmse,
                self.cfg.reward_bias,
            )
            local_icp_rewards.append(local_icp_reward)
            frame_rewards.append(local_icp_reward)

            metric_this_frame: Dict[str, float] = {}
            if src.shape[0] > 0 and tgt.shape[0] > 0:
                src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
                src_aligned = (T_prev @ src_h.T).T[:, :3]
                dists = compute_nn_distances(src_aligned, tgt)
                metric_this_frame = compute_alignment_metrics(dists, self.cfg.outlier_thresh)
                frame_metric_list.append(metric_this_frame)
                frame_rewards[-1] = float(
                    np.clip(
                        self.cfg.reward_weight_alignment * float(metric_this_frame.get("alignment_score", 0.0))
                        + self.cfg.reward_weight_local_icp * local_icp_reward,
                        0.0,
                        1.0,
                    )
                )

            is_good = quality_check(fitness, rmse, self.icp_cfg)

            # ── pair_score prefers alignment_score when available (matches reference) ──
            if metric_this_frame.get("alignment_score", 0.0) > 0.0:
                pair_score = float(metric_this_frame["alignment_score"])
            else:
                pair_score = float(fitness - self.cfg.icp_global_alpha * rmse)

            pair_buf_idx = -1
            if src.shape[0] > 0 and tgt.shape[0] > 0:
                pair_buf_idx = len(all_pairs_for_global)
                all_pairs_for_global.append((src, tgt))
                # ── Only good pairs go into refinement pool (matches reference) ──
                if is_good:
                    good_pairs_for_refine.append((src, tgt))

            frame_records.append(
                {
                    "src_idx": int(pred_local_idx),   # viewer uses this for T_refined_by_src_idx
                    "sim_global_idx": int(sim_global_idx),
                    "global_depth_frame": int(start_frame) + int(pred_seq.frame_ids[pred_local_idx]),
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
                    "pair_buf_idx": int(pair_buf_idx),
                    "T_refined": T_prev.tolist(),
                }
            )

        if not frame_rewards:
            return self.cfg.failure_fallback_reward, fitness_list, rmse_list, {}

        candidates = [r for r in frame_records if bool(r.get("good", False))]
        if not candidates:
            candidates = frame_records
        candidates = sorted(candidates, key=lambda r: float(r.get("pair_score", 0.0)), reverse=True)
        if self.cfg.icp_global_top_k is not None:
            candidates = candidates[: max(1, int(self.cfg.icp_global_top_k))]

        T_candidates = [np.asarray(r["T_refined"], dtype=np.float64) for r in candidates]
        w_candidates = np.asarray([max(float(r.get("pair_score", 0.0)), 1e-6) for r in candidates], dtype=np.float64)
        T_global = weighted_average_transforms(T_candidates, w_candidates)

        global_refine_fitness = 0.0
        global_refine_rmse = 1.0
        if self.cfg.icp_global_refine:
            # Use only good-quality pairs for global refinement (matches reference icp_sequence.py).
            refine_pairs = good_pairs_for_refine
            T_global, global_refine_fitness, global_refine_rmse = refine_global_T(
                T_global,
                refine_pairs,
                self.icp_cfg,
                max_frames=self.cfg.icp_global_refine_max_frames,
            )

        global_metric_list: List[Dict[str, float]] = []
        for pred_local_idx, sim_global_idx in pred_sim_pairs:
            src = self._scale_points_centroid(sim_seq[sim_global_idx], src_scale)
            tgt = self._scale_points_centroid(pred_seq.points_per_frame[pred_local_idx], tgt_scale)
            if src.shape[0] == 0 or tgt.shape[0] == 0:
                continue
            src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
            src_aligned = (T_global @ src_h.T).T[:, :3]
            dists = compute_nn_distances(src_aligned, tgt)
            global_metric_list.append(compute_alignment_metrics(dists, self.cfg.outlier_thresh))

        global_alignment_summary = aggregate_metric_dicts(global_metric_list)
        global_alignment_reward = float(global_alignment_summary.get("avg_alignment_score", 0.0))

        # ── GT vs sim ICP (debug / logging only) ──────────────────────────────
        gt_alignment_scores: List[float] = []
        gt_depth_pts_viz:    List[np.ndarray] = []
        gt_sim_pts_viz:      List[np.ndarray] = []
        gt_frame_scores_viz: List[float] = []
        gt_frame_ids_viz:    List[int]   = []
        if gt_seq.points_per_frame:
            gt_sim_pairs: List[Tuple[int, int]] = []
            for local_i, local_fid in enumerate(gt_seq.frame_ids):
                global_fid = int(start_frame) + int(local_fid)
                sim_global = int(round(sim_k * global_fid + sim_offset))
                if sim_global in sim_seq and sim_seq[sim_global].shape[0] > 0:
                    gt_sim_pairs.append((local_i, sim_global))
                if len(gt_sim_pairs) >= self.cfg.max_icp_pairs:
                    break
            T_gt = T_init_used.copy()
            for gt_local_idx, sim_global_idx in gt_sim_pairs:
                src = self._scale_points_centroid(sim_seq[sim_global_idx], src_scale)
                tgt = self._scale_points_centroid(gt_seq.points_per_frame[gt_local_idx], tgt_scale)
                T_gt, _, _ = run_icp_multiscale(src, tgt, T_gt, self.icp_cfg)
                if src.shape[0] > 0 and tgt.shape[0] > 0:
                    src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
                    src_aligned = (T_gt @ src_h.T).T[:, :3]
                    dists = compute_nn_distances(src_aligned, tgt)
                    m = compute_alignment_metrics(dists, self.cfg.outlier_thresh)
                    score = float(m.get("alignment_score", 0.0))
                    gt_alignment_scores.append(score)
                    if debug_dir is not None:
                        gt_depth_pts_viz.append(tgt)
                        gt_sim_pts_viz.append(src_aligned)
                        gt_frame_scores_viz.append(score)
                        gt_frame_ids_viz.append(
                            int(start_frame) + int(gt_seq.frame_ids[gt_local_idx])
                        )
        gt_avg_alignment = float(np.mean(gt_alignment_scores)) if gt_alignment_scores else 0.0
        logger.info(
            "[Reward] episode=%s pred_global_alignment=%.4f gt_alignment=%.4f",
            episode_id, global_alignment_reward, gt_avg_alignment,
        )

        if self.cfg.frame_agg == "median":
            local_reward = float(np.median(frame_rewards))
        else:
            local_reward = float(np.mean(frame_rewards))
        reward = float(
            np.clip(
                self.cfg.reward_weight_alignment * global_alignment_reward
                + self.cfg.reward_weight_local_icp * local_reward,
                0.0,
                1.0,
            )
        )

        icp_result = {
            "T_sim_to_depth": T_global.tolist(),
            "start_frame": int(start_frame),
            "reward": float(reward),
            "reward_local": float(local_reward),
            "reward_global_alignment": float(global_alignment_reward),
            "gt_avg_alignment": float(gt_avg_alignment),
            "frame_agg": self.cfg.frame_agg,
            "icp_init": {
                "path": icp_init_path,
                "sim_scale": float(src_scale),
                "depth_scale": float(tgt_scale),
                "T_sim_to_depth": T_init_used.tolist(),
            },
            "global": {
                "num_candidates": int(len(candidates)),
                "global_refine": bool(self.cfg.icp_global_refine),
                "global_refine_fitness": float(global_refine_fitness),
                "global_refine_rmse": float(global_refine_rmse),
                "T_global": T_global.tolist(),
                "global_top_k": self.cfg.icp_global_top_k,
                "global_alpha": float(self.cfg.icp_global_alpha),
                "good_pairs_for_refine": int(len(good_pairs_for_refine)),
            },
            "frame_records": frame_records,
            "alignment_summary": aggregate_metric_dicts(frame_metric_list),
            "global_alignment_summary": global_alignment_summary,
        }

        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            icp_json_path = os.path.join(debug_dir, f"sample_{sample_idx:02d}_icp.json")
            with open(icp_json_path, "w", encoding="utf-8") as f:
                json.dump(icp_result, f, indent=2)

            # ── Multi-view alignment visualisation ──────────────────────────
            try:
                depth_pts_viz  = [
                    self._scale_points_centroid(pred_seq.points_per_frame[li], tgt_scale)
                    for li, _ in pred_sim_pairs
                    # if pred_seq.points_per_frame.get(li) is not None
                    if li < len(pred_seq.points_per_frame) and pred_seq.points_per_frame[li] is not None
                ]
                src_h_all = [
                    np.hstack([self._scale_points_centroid(sim_seq[si], src_scale),
                               np.ones((sim_seq[si].shape[0], 1), dtype=np.float64)])
                    for _, si in pred_sim_pairs
                    if sim_seq.get(si) is not None and sim_seq[si].shape[0] > 0
                ]
                sim_pts_viz = [
                    (T_global @ h.T).T[:, :3] for h in src_h_all
                ]
                frame_scores_viz = [
                    float(r.get("alignment_score", 0.0)) for r in frame_records
                ]
                frame_ids_viz = [
                    int(r.get("global_depth_frame", r.get("src_idx", i)))
                    for i, r in enumerate(frame_records)
                ]
                viz_path = os.path.join(debug_dir, f"sample_{sample_idx:02d}_alignment.png")
                save_alignment_viz(
                    out_path=viz_path,
                    depth_pts_list=depth_pts_viz,
                    sim_pts_list=sim_pts_viz,
                    frame_scores=frame_scores_viz,
                    frame_ids=frame_ids_viz,
                    reward=reward,
                )
            except Exception:
                logger.warning("[Reward] failed to save alignment viz", exc_info=True)

            # ── GT multi-view alignment visualisation ────────────────────────
            if gt_depth_pts_viz:
                try:
                    gt_viz_path = os.path.join(debug_dir, f"sample_{sample_idx:02d}_alignment_gt.png")
                    save_alignment_viz(
                        out_path=gt_viz_path,
                        depth_pts_list=gt_depth_pts_viz,
                        sim_pts_list=gt_sim_pts_viz,
                        frame_scores=gt_frame_scores_viz,
                        frame_ids=gt_frame_ids_viz,
                        reward=gt_avg_alignment,
                    )
                except Exception:
                    logger.warning("[Reward] failed to save GT alignment viz", exc_info=True)

        return reward, fitness_list, rmse_list, global_alignment_summary

    def compute(self, videos: List[torch.Tensor], data_batch: Dict[str, torch.Tensor]) -> tuple[List[float], RewardDiagnostics]:
        t0 = time.time()
        self._reward_step += 1
        gt_video = data_batch.get("video")
        if not isinstance(gt_video, torch.Tensor):
            # Keep training alive even when batch misses ground-truth video.
            rewards = [self.cfg.failure_fallback_reward for _ in videos]
            diag = RewardDiagnostics(
                fallback_count=len(videos),
                sample_count=len(videos),
                elapsed_sec=time.time() - t0,
                mean_reward=float(np.mean(rewards) if rewards else 0.0),
            )
            return rewards, diag

        rewards: List[float] = []
        all_fitness: List[float] = []
        all_rmse: List[float] = []
        all_alignment: List[float] = []
        fallback_count = 0
        debug_on = self._should_dump_debug()
        iter_debug_dir = self._iter_debug_dir() if debug_on else None
        if debug_on:
            os.makedirs(iter_debug_dir, exist_ok=True)
            # Save ground-truth video as mp4 (once per iteration, not per sample)
            try:
                self._save_video_mp4(gt_video, os.path.join(iter_debug_dir, "gt_video.mp4"))
            except Exception:
                logger.warning("[Reward] failed to save gt video mp4", exc_info=True)
                self._save_video_frames(gt_video, os.path.join(iter_debug_dir, "gt_video"))

        sam_points_xy, sam_point_labels, ann_frame_idx_local_raw, episode, start_frame = self._extract_prompt_from_batch(data_batch)
        gt_len = int(gt_video.shape[2]) if gt_video.ndim == 5 else int(gt_video.shape[1])
        ann_frame_idx = int(np.clip(ann_frame_idx_local_raw, 0, max(0, gt_len - 1)))
        ann_frame_idx_global = int(start_frame + ann_frame_idx_local_raw)

        if sam_points_xy is not None:
            if ann_frame_idx_local_raw < 0 or ann_frame_idx_local_raw >= gt_len:
                logger.warning(
                    "[Reward] annotation frame out of current chunk for episode=%s ann_frame_local_raw=%d start_frame=%d chunk_len=%d; clamped_prompt_frame=%d",
                    episode,
                    ann_frame_idx_local_raw,
                    start_frame,
                    gt_len,
                    ann_frame_idx,
                )
            logger.info(
                "[Reward] using annotation points for episode=%s n_points=%d ann_frame_local=%d",
                episode,
                len(sam_points_xy),
                ann_frame_idx,
            )
        else:
            logger.warning(
                "[Reward] annotation points not found for episode=%s; fallback to center prompt",
                episode,
            )

        video_path = self._extract_video_path_from_batch(data_batch)
        gt_mask_cache_dir = self._ensure_gt_mask_cache(
            episode=episode,
            video_path=video_path,
            sam_points_xy=sam_points_xy,
            sam_point_labels=sam_point_labels,
            ann_frame_idx_global=ann_frame_idx_global,
        )
        sampled_global_frame_ids = self._get_sampled_global_frame_ids(gt_video, start_frame)
        gt_external_masks = self._load_cached_masks_for_frames(gt_mask_cache_dir, sampled_global_frame_ids)
        pred_init_mask = None
        if gt_mask_cache_dir:
            pred_init_mask = self.extractor.load_mask_from_cache(gt_mask_cache_dir, start_frame)

        icp_init = self._get_icp_init_for_episode(episode)
        if icp_init is None:
            raise RuntimeError(
                f"icp_init is required but missing for episode={episode}. "
                f"Expected json under: {self._icp_init_dir}"
            )

        icp_init_T = np.asarray(icp_init["T_sim_to_depth"], dtype=np.float64)
        icp_sim_scale = float(icp_init["sim_scale"])
        icp_depth_scale = float(icp_init["depth_scale"])
        icp_init_path = str(icp_init.get("path", ""))
        logger.info(
            "[Reward] using icp_init episode=%s sim_scale=%.6f depth_scale=%.6f path=%s",
            episode,
            icp_sim_scale,
            icp_depth_scale,
            icp_init_path,
        )

        sim_seq = self._load_sim_sequence(episode)
        if not sim_seq:
            raise RuntimeError(
                f"sim pointcloud required but not found for episode={episode}. "
                f"Expected npy under: {self._sim_pointcloud_dir}"
            )

        for i, pred_video in enumerate(videos):
            try:
                if debug_on and i < max(1, int(self.cfg.debug_max_videos_per_iter)):
                    sample_debug_dir = os.path.join(iter_debug_dir, f"sample_{i:02d}")
                    os.makedirs(sample_debug_dir, exist_ok=True)
                else:
                    sample_debug_dir = None

                reward, fitness_list, rmse_list, metric_summary = self._score_single_video(
                    pred_video,
                    gt_video,
                    sim_seq=sim_seq,
                    debug_dir=sample_debug_dir,
                    sample_idx=i,
                    sam_points_xy=sam_points_xy,
                    sam_point_labels=sam_point_labels,
                    ann_frame_idx=ann_frame_idx,
                    gt_external_masks=gt_external_masks,
                    pred_init_mask=pred_init_mask,
                    gt_mask_cache_dir=gt_mask_cache_dir or "",
                    icp_init_T=icp_init_T,
                    icp_sim_scale=icp_sim_scale,
                    icp_depth_scale=icp_depth_scale,
                    icp_init_path=icp_init_path,
                    episode_id=episode,
                    start_frame=start_frame,
                )
                rewards.append(reward)
                all_fitness.extend(fitness_list)
                all_rmse.extend(rmse_list)
                if metric_summary:
                    all_alignment.append(metric_summary.get("avg_alignment_score", 0.0))
            except Exception:
                fallback_count += 1
                rewards.append(self.cfg.failure_fallback_reward)
                if sample_debug_dir is not None:
                    os.makedirs(sample_debug_dir, exist_ok=True)
                    with open(os.path.join(sample_debug_dir, "error.txt"), "w", encoding="utf-8") as f:
                        f.write(traceback.format_exc())
                logger.exception("[Reward] sample %d failed; fallback reward used.", i)

        diag = RewardDiagnostics(
            fallback_count=fallback_count,
            sample_count=len(videos),
            elapsed_sec=time.time() - t0,
            mean_reward=float(np.mean(rewards) if rewards else 0.0),
            mean_fitness=float(np.mean(all_fitness) if all_fitness else 0.0),
            mean_rmse=float(np.mean(all_rmse) if all_rmse else 0.0),
            mean_alignment_score=float(np.mean(all_alignment) if all_alignment else 0.0),
        )
        if debug_on:
            with open(os.path.join(iter_debug_dir, "reward_summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "iter_reward_step": int(self._reward_step),
                        "rewards": [float(r) for r in rewards],
                        "fallback_count": int(fallback_count),
                        "sample_count": int(len(videos)),
                        "elapsed_sec": float(diag.elapsed_sec),
                        "mean_reward": float(diag.mean_reward),
                        "mean_fitness": float(diag.mean_fitness),
                        "mean_rmse": float(diag.mean_rmse),
                        "mean_alignment_score": float(diag.mean_alignment_score),
                    },
                    f,
                    indent=2,
                )
        return rewards, diag


def configure_reward_engine(cfg: CosmosGRPOConfig) -> None:
    global _ENGINE
    _ENGINE = OnlineRewardEngine(cfg)


def get_last_reward_diagnostics() -> RewardDiagnostics:
    return _LAST_DIAGNOSTICS


def compute_rewards(
    videos: List[torch.Tensor],
    data_batch: Dict[str, torch.Tensor],
) -> List[float]:
    """Compute online geometry rewards for rollout videos.

    This is the runtime training entry-point used by GRPOTrainer.
    """
    global _LAST_DIAGNOSTICS
    if _ENGINE is None:
        # Safe fallback when trainer has not configured the engine yet.
        rewards = [0.0 for _ in videos]
        _LAST_DIAGNOSTICS = RewardDiagnostics(
            fallback_count=len(videos),
            sample_count=len(videos),
            mean_reward=0.0,
        )
        return rewards

    rewards, diag = _ENGINE.compute(videos, data_batch)
    _LAST_DIAGNOSTICS = diag
    return rewards
