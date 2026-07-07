# RoboTwin action-conditioned dataset: VideoDataset + 14-dim dual-arm relative joint action.
#
# Reads actions/<same_basename>.npy = absolute [T,14] arm+gripper joint state, frame-aligned
# with the video (extracted from the pointcloud q). Samples the SAME random window as the
# video and returns per-frame relative increments (np.diff), key "action" [num_frames-1, 14].
import os
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu

from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset
from cosmos_predict2._src.imaginaire.utils import log


class ActionVideoDataset(VideoDataset):
    def __init__(self, *args, action_dim: int = 14, action_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.actions_dir = os.path.join(self.dataset_dir, "actions")
        if not os.path.isdir(self.actions_dir):
            raise ValueError(f"actions dir not found: {self.actions_dir}")

    def __getitem__(self, index: int):
        try:
            video_path = self.video_paths[index]
            base = os.path.basename(video_path).replace(".mp4", "")

            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            total_frames = len(vr)
            if total_frames < self.sequence_length:
                raise ValueError(f"{video_path}: {total_frames} < {self.sequence_length}")
            max_start = total_frames - self.sequence_length
            start = int(np.random.randint(0, max_start + 1))
            end = start + self.sequence_length
            frame_ids = list(range(start, end))
            frames = vr.get_batch(frame_ids).asnumpy()
            try:
                fps = vr.get_avg_fps()
            except Exception:
                fps = 16
            del vr

            frames = torch.from_numpy(frames.astype(np.uint8)).permute(0, 3, 1, 2)  # [T,C,H,W]
            frames = self.preprocess(frames)
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
            video = frames.permute(1, 0, 2, 3)  # [C,T,H,W]

            act_path = os.path.join(self.actions_dir, f"{base}.npy")
            act_abs = np.load(act_path).astype(np.float32)  # [Tfull, 14]
            if act_abs.shape[0] != total_frames:
                raise ValueError(f"{base}: action rows {act_abs.shape[0]} != video frames {total_frames}")
            win = act_abs[start:end]                         # [num_frames, 14]
            action = np.diff(win, axis=0) * self.action_scale  # [num_frames-1, 14] relative delta
            action = torch.from_numpy(np.ascontiguousarray(action)).float()

            _, _, h, w = video.shape
            if self.caption_format == "json":
                caption = self._load_json_caption(Path(os.path.join(self.caption_dir, f"{base}.json")))
            else:
                caption = self._load_text(Path(os.path.join(self.caption_dir, f"{base}.txt")))

            return dict(
                video=video,
                ai_caption=caption,
                action=action,
                # action conditioner keys off t5_text_embeddings; null text for now (v1 smoke)
                t5_text_embeddings=torch.zeros(512, 1024, dtype=torch.bfloat16),
                t5_text_mask=torch.ones(512, dtype=torch.int64),
                fps=fps,
                image_size=torch.tensor([h, w, h, w]),
                num_frames=self.sequence_length,
                padding_mask=torch.zeros(1, h, w),
            )
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(f"ActionVideoDataset failed on {self.video_paths[index]}: {e}", rank0_only=False)
            return self[int(np.random.randint(len(self.video_paths)))]
