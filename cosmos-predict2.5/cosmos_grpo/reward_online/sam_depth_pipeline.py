from __future__ import annotations

import gc
import os
import json
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class OnlineVisionConfig:
    sam2_checkpoint: str = "/home/wuqiu/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_obj_id: int = 1
    depth_model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    conf_percentile: float = 20.0
    max_frames: int = 16
    frame_stride: int = 4
    max_points_per_frame: int = 120000


@dataclass
class PointCloudSequence:
    frame_ids: List[int]
    points_per_frame: List[np.ndarray]


class OnlinePointCloudExtractor:
    """Online video->(SAM mask + depth)->point cloud sequence.

    This class intentionally reimplements the glue logic inside cosmos_grpo,
    without importing helper functions from external scripts.
    """

    def __init__(self, cfg: OnlineVisionConfig) -> None:
        self.cfg = cfg
        self._sam_predictor = None
        self._depth_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_sam_predictor(self) -> None:
        """Build SAM2 predictor fresh on GPU (always destroyed after use)."""
        if self._sam_predictor is None:
            from sam2.build_sam import build_sam2_video_predictor
            self._sam_predictor = build_sam2_video_predictor(
                self.cfg.sam2_model_cfg,
                self.cfg.sam2_checkpoint,
                device=self._device,
            )

    def _ensure_depth_model(self) -> None:
        """Build depth model fresh on GPU (always destroyed after use)."""
        if self._depth_model is None:
            from depth_anything_3.api import DepthAnything3
            self._depth_model = DepthAnything3.from_pretrained(self.cfg.depth_model_id).to(device=self._device)

    def _destroy_sam(self) -> None:
        """Permanently delete SAM2 from GPU memory — its job is done."""
        if self._sam_predictor is not None:
            del self._sam_predictor
            self._sam_predictor = None
            gc.collect()
            torch.cuda.empty_cache()

    def _destroy_depth_model(self) -> None:
        """Permanently delete depth model from GPU memory — its job is done."""
        if self._depth_model is not None:
            del self._depth_model
            self._depth_model = None
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _video_to_uint8_frames(
        video_tensor: torch.Tensor, max_frames: int, stride: int
    ) -> tuple[List[np.ndarray], List[int], int]:
        # Accept [B,C,T,H,W] or [C,T,H,W], use the first sample in batch mode.
        if video_tensor.ndim == 5:
            video_tensor = video_tensor[0]
        if video_tensor.ndim != 4:
            raise ValueError(f"Expected video tensor [C,T,H,W], got shape={tuple(video_tensor.shape)}")

        c, t, _, _ = video_tensor.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels, got C={c}")

        stride = max(1, int(stride))
        frame_ids = list(range(0, t, stride))[: max_frames]
        frames: List[np.ndarray] = []

        # Handle both [0,255] uint8 and [-1,1] float model outputs.
        v = video_tensor.detach().float().cpu()
        v_min = float(v.min().item())
        v_max = float(v.max().item())
        if v_min >= -1.1 and v_max <= 1.1:
            v = (v + 1.0) * 127.5
        v = v.clamp(0, 255).to(torch.uint8)

        for idx in frame_ids:
            frame = v[:, idx].permute(1, 2, 0).numpy()  # [H,W,C]
            frames.append(frame)
        return frames, frame_ids, int(t)

    @staticmethod
    def _write_frames(frames: List[np.ndarray], out_dir: str) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        paths: List[str] = []
        for i, frame in enumerate(frames):
            p = str(Path(out_dir) / f"{i:05d}.jpg")
            Image.fromarray(frame, mode="RGB").save(p, quality=95)
            paths.append(p)
        return paths

    @staticmethod
    def _save_mask(mask: np.ndarray, path: str) -> None:
        m = (mask.astype(np.uint8) * 255)
        Image.fromarray(m, mode="L").save(path)

    @staticmethod
    def _save_depth(depth_hw: np.ndarray, path: str) -> None:
        d = depth_hw.astype(np.float32)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        dmin = float(np.min(d))
        dmax = float(np.max(d))
        if dmax > dmin:
            x = ((d - dmin) / (dmax - dmin) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x = np.zeros_like(d, dtype=np.uint8)
        Image.fromarray(x, mode="L").save(path)

    def _segment_masks(
        self,
        frames: List[np.ndarray],
        prompt_points_xy: Optional[List[Tuple[float, float]]] = None,
        prompt_labels: Optional[List[int]] = None,
        prompt_frame_idx: int = 0,
        prompt_mask: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        self._ensure_sam_predictor()

        tmp_dir = tempfile.mkdtemp(prefix="grpo_sam_")
        try:
            self._write_frames(frames, tmp_dir)
            state = self._sam_predictor.init_state(video_path=tmp_dir)

            h, w, _ = frames[0].shape
            prompt_points: List[Tuple[float, float]] = []
            if prompt_points_xy is None or len(prompt_points_xy) == 0:
                prompt_points = [(w / 2.0, h / 2.0)]
            else:
                for px, py in prompt_points_xy:
                    pxc = float(np.clip(float(px), 0.0, max(0.0, w - 1.0)))
                    pyc = float(np.clip(float(py), 0.0, max(0.0, h - 1.0)))
                    prompt_points.append((pxc, pyc))

            if prompt_labels is not None and len(prompt_labels) == len(prompt_points):
                labels = np.array([1 if int(v) > 0 else 0 for v in prompt_labels], dtype=np.int32)
            else:
                labels = np.ones((len(prompt_points),), dtype=np.int32)

            prompt_points_np = np.array(prompt_points, dtype=np.float32)
            prompt_frame_idx = int(np.clip(prompt_frame_idx, 0, max(0, len(frames) - 1)))

            with torch.inference_mode():
                if prompt_mask is not None:
                    pm = np.asarray(prompt_mask).astype(bool)
                    if pm.shape != (h, w):
                        pm = np.array(
                            Image.fromarray(pm.astype(np.uint8) * 255, mode="L").resize((w, h), Image.NEAREST)
                        ) > 0
                    self._sam_predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=prompt_frame_idx,
                        obj_id=self.cfg.sam_obj_id,
                        mask=pm,
                    )
                else:
                    self._sam_predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=prompt_frame_idx,
                        obj_id=self.cfg.sam_obj_id,
                        points=prompt_points_np,
                        labels=labels,
                    )

                masks_map: Dict[int, np.ndarray] = {}
                for frame_idx, out_obj_ids, out_mask_logits in self._sam_predictor.propagate_in_video(state):
                    for i, out_obj_id in enumerate(out_obj_ids):
                        if int(out_obj_id) == self.cfg.sam_obj_id:
                            masks_map[int(frame_idx)] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

            masks: List[np.ndarray] = []
            for i, f in enumerate(frames):
                if i in masks_map:
                    m = masks_map[i]
                    if m.dtype != np.bool_:
                        m = m.astype(bool)
                    masks.append(m)
                else:
                    masks.append(np.ones((f.shape[0], f.shape[1]), dtype=bool))
            return masks
        except Exception:
            # Robust fallback for online training stability.
            return [np.ones((f.shape[0], f.shape[1]), dtype=bool) for f in frames]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _depth_predict(self, frames: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_depth_model()
        pred = self._depth_model.inference(
            frames,
            export_dir=None,
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        depth = pred.depth.astype(np.float32)
        conf = pred.conf.astype(np.float32)
        intrinsics = pred.intrinsics.astype(np.float32)
        extrinsics = pred.extrinsics.astype(np.float32)
        return depth, conf, intrinsics, extrinsics

    @staticmethod
    def _backproject_masked_points(
        depth_hw: np.ndarray,
        conf_hw: np.ndarray,
        mask_hw: np.ndarray,
        ixt: np.ndarray,
        ext: np.ndarray,
        conf_thr: float,
    ) -> np.ndarray:
        h, w = depth_hw.shape
        if mask_hw.shape != (h, w):
            mask_hw = np.array(
                Image.fromarray(mask_hw.astype(np.uint8) * 255, mode="L").resize((w, h), Image.NEAREST)
            ) > 0
        valid = np.isfinite(depth_hw) & (depth_hw > 0)
        valid &= np.isfinite(conf_hw)
        valid &= conf_hw >= conf_thr
        valid &= mask_hw
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float64)

        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3).astype(np.float32)

        d_flat = depth_hw.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        k_inv = np.linalg.inv(ixt).astype(np.float32)
        rays = k_inv @ pix[vidx].T
        x_c = rays * d_flat[vidx][None, :]

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :4] = ext[:3, :4]
        c2w = np.linalg.inv(w2c)

        x_c_h = np.vstack([x_c, np.ones((1, x_c.shape[1]), dtype=np.float32)])
        x_w = (c2w @ x_c_h)[:3].T.astype(np.float64)
        return x_w

    def extract_pointcloud_sequence(
        self,
        video_tensor: torch.Tensor,
        debug_dir: Optional[str] = None,
        save_prefix: str = "sample",
        prompt_points_xy: Optional[List[Tuple[float, float]]] = None,
        prompt_labels: Optional[List[int]] = None,
        ann_frame_idx: int = 0,
        external_masks: Optional[List[np.ndarray]] = None,
        init_mask: Optional[np.ndarray] = None,
        mask_source: Optional[str] = None,
    ) -> PointCloudSequence:
        frame_debug_dir = None
        try:
            frames, sampled_frame_ids, total_frames = self._video_to_uint8_frames(
                video_tensor,
                max_frames=self.cfg.max_frames,
                stride=self.cfg.frame_stride,
            )
            if not frames:
                return PointCloudSequence(frame_ids=[], points_per_frame=[])

            if debug_dir is not None:
                frame_debug_dir = Path(debug_dir) / save_prefix
                os.makedirs(frame_debug_dir, exist_ok=True)
                self._write_frames(frames, str(frame_debug_dir / "frames"))
                os.makedirs(frame_debug_dir / "masks", exist_ok=True)
                os.makedirs(frame_debug_dir / "depth", exist_ok=True)
                os.makedirs(frame_debug_dir / "conf", exist_ok=True)
                os.makedirs(frame_debug_dir / "points", exist_ok=True)
                with open(frame_debug_dir / "frame_ids.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "total_frames_in_video": int(total_frames),
                            "sampled_frame_ids": [int(x) for x in sampled_frame_ids],
                            "sample_count": int(len(sampled_frame_ids)),
                            "frame_stride": int(self.cfg.frame_stride),
                            "max_frames": int(self.cfg.max_frames),
                            "prompt_points_xy": (
                                [[float(x), float(y)] for (x, y) in prompt_points_xy]
                                if prompt_points_xy is not None
                                else None
                            ),
                            "prompt_labels": (
                                [int(v) for v in prompt_labels]
                                if prompt_labels is not None
                                else None
                            ),
                            "prompt_point_count": int(len(prompt_points_xy)) if prompt_points_xy is not None else 0,
                            "prompt_frame_idx": int(ann_frame_idx),
                            "mask_source": mask_source,
                            "uses_external_masks": bool(external_masks is not None),
                            "uses_init_mask": bool(init_mask is not None),
                        },
                        f,
                        indent=2,
                    )

            if external_masks is not None:
                masks = external_masks
            else:
                masks = self._segment_masks(
                    frames,
                    prompt_points_xy=prompt_points_xy,
                    prompt_labels=prompt_labels,
                    prompt_frame_idx=ann_frame_idx,
                    prompt_mask=init_mask,
                )
            # SAM2 has done its job — destroy it to free GPU VRAM for depth model.
            self._destroy_sam()
            depth, conf, ixt, ext = self._depth_predict(frames)
            # Depth model has done its job — destroy it to free GPU VRAM.
            self._destroy_depth_model()
            conf_thr = float(np.percentile(conf.reshape(-1), self.cfg.conf_percentile))

            n = min(len(frames), len(masks), depth.shape[0], conf.shape[0], ixt.shape[0], ext.shape[0])
            points_per_frame: List[np.ndarray] = []
            frame_ids: List[int] = []
            for i in range(n):
                if frame_debug_dir is not None:
                    self._save_mask(masks[i], str(frame_debug_dir / "masks" / f"{i:05d}.png"))
                    self._save_depth(depth[i], str(frame_debug_dir / "depth" / f"{i:05d}.png"))
                    self._save_depth(conf[i], str(frame_debug_dir / "conf" / f"{i:05d}.png"))

                pts = self._backproject_masked_points(
                    depth[i],
                    conf[i],
                    masks[i],
                    ixt[i],
                    ext[i],
                    conf_thr,
                )
                if pts.shape[0] > self.cfg.max_points_per_frame:
                    sel = np.random.choice(pts.shape[0], self.cfg.max_points_per_frame, replace=False)
                    pts = pts[sel]
                frame_ids.append(int(sampled_frame_ids[i]))
                points_per_frame.append(pts)
                if frame_debug_dir is not None:
                    np.save(str(frame_debug_dir / "points" / f"{i:05d}.npy"), pts)

            return PointCloudSequence(frame_ids=frame_ids, points_per_frame=points_per_frame)
        except Exception:
            if frame_debug_dir is not None:
                with open(frame_debug_dir / "error.txt", "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            return PointCloudSequence(frame_ids=[], points_per_frame=[])

    def build_video_mask_cache(
        self,
        video_path: str,
        cache_dir: str,
        prompt_points_xy: List[Tuple[float, float]],
        prompt_labels: Optional[List[int]],
        ann_frame_idx: int,
    ) -> None:
        self._ensure_sam_predictor()
        os.makedirs(cache_dir, exist_ok=True)

        state = self._sam_predictor.init_state(video_path=video_path)
        video_h = int(state["video_height"])
        video_w = int(state["video_width"])
        prompt_frame_idx = int(np.clip(ann_frame_idx, 0, max(0, int(state["num_frames"]) - 1)))

        prompt_points: List[Tuple[float, float]] = []
        for px, py in prompt_points_xy:
            prompt_points.append(
                (
                    float(np.clip(float(px), 0.0, max(0.0, video_w - 1.0))),
                    float(np.clip(float(py), 0.0, max(0.0, video_h - 1.0))),
                )
            )
        labels = (
            np.array([1 if int(v) > 0 else 0 for v in prompt_labels], dtype=np.int32)
            if prompt_labels is not None and len(prompt_labels) == len(prompt_points)
            else np.ones((len(prompt_points),), dtype=np.int32)
        )
        points_np = np.array(prompt_points, dtype=np.float32)

        meta = {
            "video_path": video_path,
            "prompt_points_xy": [[float(x), float(y)] for x, y in prompt_points],
            "prompt_labels": [int(v) for v in labels.tolist()],
            "ann_frame_idx": int(prompt_frame_idx),
            "video_height": video_h,
            "video_width": video_w,
            "num_frames": int(state["num_frames"]),
        }
        with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with torch.inference_mode():
            self._sam_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt_frame_idx,
                obj_id=self.cfg.sam_obj_id,
                points=points_np,
                labels=labels,
            )
            for frame_idx, out_obj_ids, out_mask_logits in self._sam_predictor.propagate_in_video(state):
                for i, out_obj_id in enumerate(out_obj_ids):
                    if int(out_obj_id) != self.cfg.sam_obj_id:
                        continue
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(bool)
                    self._save_mask(mask, os.path.join(cache_dir, f"{int(frame_idx):05d}.png"))

    @staticmethod
    def load_mask_from_cache(cache_dir: str, frame_idx: int) -> Optional[np.ndarray]:
        path = os.path.join(cache_dir, f"{int(frame_idx):05d}.png")
        if not os.path.exists(path):
            return None
        return np.array(Image.open(path).convert("L")) > 0
