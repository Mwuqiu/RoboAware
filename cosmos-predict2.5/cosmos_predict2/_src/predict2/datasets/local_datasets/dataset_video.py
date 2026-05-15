# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic video dataset loader for Cosmos Predict2."""

import json
import os
import random
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


class PointcloudEncodingConfigurationError(RuntimeError):
    """Raised for non-retryable online pointcloud encoding setup issues."""


_POINTCEPT_ENCODERS: dict[str, torch.nn.Module] = {}
_POINTCEPT_ENCODER_LOCK = threading.Lock()


def _make_absolute_pointcept_path(pointcept_root: Path, path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((pointcept_root / path).resolve())


def _find_pointcept_root() -> Optional[Path]:
    candidates: list[Path] = []

    pointcept_root = os.environ.get("POINTCEPT_ROOT")
    if pointcept_root:
        candidates.append(Path(pointcept_root).expanduser())

    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "Pointcept")

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if (candidate / "pointflow" / "pf_encoder.py").is_file():
            return candidate
    return None


# ── Pointcept pf_encoder import ──────────────────────────────────────────────
# pf_encoder.py internally does sys.path.insert(0, ...) to add the Pointcept
# root, so the only thing cosmos needs is to make the pointflow directory
# importable.
_pointcept_root = _find_pointcept_root()
if _pointcept_root is None:
    raise PointcloudEncodingConfigurationError(
        "Could not locate the Pointcept repository. Set POINTCEPT_ROOT to your Pointcept checkout."
    )
_pointflow_dir = str(_pointcept_root / "pointflow")
if _pointflow_dir not in sys.path:
    sys.path.insert(0, _pointflow_dir)
import pf_encoder  # noqa: E402


def _get_online_pointcloud_encoder(pc_encoder_config: Optional[dict] = None) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise PointcloudEncodingConfigurationError(
            "Online pointcloud encoding requires CUDA because Pointcept loads its checkpoint onto GPU."
        )

    device = torch.device("cuda", torch.cuda.current_device())
    device_key = str(device)

    with _POINTCEPT_ENCODER_LOCK:
        encoder = _POINTCEPT_ENCODERS.get(device_key)
        if encoder is None:
            pf_encoder.apply_encoder_config(pc_encoder_config or {})
            pf_encoder.CONFIG_FILE = _make_absolute_pointcept_path(_pointcept_root, pf_encoder.CONFIG_FILE)
            pf_encoder.EXP_DIR = _make_absolute_pointcept_path(_pointcept_root, pf_encoder.EXP_DIR)
            pf_encoder.WEIGHT_PATH = _make_absolute_pointcept_path(_pointcept_root, pf_encoder.WEIGHT_PATH)
            encoder = pf_encoder.PTV3Encoder(pf_encoder.load_ptv3_model()).to(device)
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            _POINTCEPT_ENCODERS[device_key] = encoder

    return encoder


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
        pc_latent_source: str = "auto",  # "precomputed", "online", or "auto"
        pc_latent_k: int = 30,
        pc_latent_sample: str = "first",
        pc_latent_pad_value: float = 0.0,
        pc_latent_amp: bool = False,
        pc_latent_grid_size: float = 0.005,
        pc_encoder_config: Optional[dict] = None,
        pc_conditioning_mode_probs: Optional[dict[str, float]] = None,
        pc_conditioning_prefix_frames: int | list[int] | tuple[int, ...] = 2,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            prompt_type (str | None): Which prompt to use from JSON ("long", "short", "medium").
                                     If None, uses the first available prompt type.
                                     Only applicable when using JSON format.
            caption_format (str): Caption format - "text", "json", or "auto" to detect automatically

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format
        self.pc_latent_source = pc_latent_source
        self.pc_latent_k = int(pc_latent_k)
        self.pc_latent_sample = pc_latent_sample
        self.pc_latent_pad_value = float(pc_latent_pad_value)
        self.pc_latent_amp = bool(pc_latent_amp)
        self.pc_latent_grid_size = float(pc_latent_grid_size)
        self.pc_encoder_config = pc_encoder_config
        self.pc_conditioning_mode_probs = self._normalize_pc_conditioning_mode_probs(pc_conditioning_mode_probs)
        self.pc_conditioning_prefix_frames = self._normalize_pc_conditioning_prefix_frames(
            pc_conditioning_prefix_frames
        )

        if self.pc_latent_source not in {"precomputed", "online", "auto"}:
            raise ValueError(
                f"Invalid pc_latent_source: {self.pc_latent_source}. Must be 'precomputed', 'online', or 'auto'"
            )

        # Determine caption format and directory
        self._setup_caption_format()

        self.pc_latent_dir = os.path.join(self.dataset_dir, "pc_latent")
        self.pointcloud_dir = os.path.join(self.dataset_dir, "pointclouds")
        video_dir = os.path.join(self.dataset_dir, "videos")
        
        if video_paths is None:
            self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            self.video_paths = sorted(self.video_paths)
        else:
            self.video_paths = video_paths
        log.info(f"{len(self.video_paths)} videos in total")

        self._all_precomputed_pc_latents_available = self._check_all_precomputed_pc_latents_available()
        if self.pc_latent_source == "precomputed" and not self._all_precomputed_pc_latents_available:
            raise ValueError(
                f"pc_latent_source='precomputed' but not every video has a matching latent in {self.pc_latent_dir}"
            )
        if self.pc_latent_source == "online" and not os.path.isdir(self.pointcloud_dir):
            raise ValueError(
                f"pc_latent_source='online' but pointclouds directory is missing: {self.pointcloud_dir}"
            )

        self.requires_main_process_data_loading = self.pc_latent_source == "online" or (
            self.pc_latent_source == "auto"
            and os.path.isdir(self.pointcloud_dir)
            and not self._all_precomputed_pc_latents_available
        )

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

        if self.requires_main_process_data_loading:
            log.info(
                "VideoDataset will compute pc_latent online from pointclouds; DataLoader workers should stay at 0."
            )

    @staticmethod
    def _normalize_pc_conditioning_mode_probs(
        mode_probs: Optional[dict[str, float]],
    ) -> dict[str, float]:
        if mode_probs is None:
            mode_probs = {"full": 1.0}

        valid_modes = {"full", "prefix", "none"}
        unknown_modes = set(mode_probs) - valid_modes
        if unknown_modes:
            raise ValueError(
                f"Unknown pc_conditioning_mode_probs keys: {sorted(unknown_modes)}. "
                f"Valid keys are {sorted(valid_modes)}"
            )

        normalized = {mode: max(float(mode_probs.get(mode, 0.0)), 0.0) for mode in valid_modes}
        total = sum(normalized.values())
        if total <= 0.0:
            raise ValueError("pc_conditioning_mode_probs must contain at least one positive probability")
        return {mode: prob / total for mode, prob in normalized.items()}

    @staticmethod
    def _normalize_pc_conditioning_prefix_frames(
        prefix_frames: int | list[int] | tuple[int, ...],
    ) -> tuple[int, ...]:
        if isinstance(prefix_frames, int):
            values = (prefix_frames,)
        elif isinstance(prefix_frames, (list, tuple)) or (
            hasattr(prefix_frames, "__iter__") and not isinstance(prefix_frames, (str, bytes))
        ):
            values = tuple(int(value) for value in prefix_frames)
        else:
            raise TypeError(
                "pc_conditioning_prefix_frames must be an int or an iterable of ints, "
                f"got {type(prefix_frames)!r}"
            )

        if not values or any(value <= 0 for value in values):
            raise ValueError(
                f"pc_conditioning_prefix_frames must contain positive integers, got {values}"
            )
        return values

    def _sample_pc_conditioning_mode(self) -> str:
        sample = random.random()
        cumulative = 0.0
        for mode in ("full", "prefix", "none"):
            cumulative += self.pc_conditioning_mode_probs[mode]
            if sample <= cumulative:
                return mode
        return "none"

    def _apply_pc_conditioning_policy(
        self, pc_x0: torch.Tensor, pc_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self._sample_pc_conditioning_mode()
        if mode == "full":
            return pc_x0, pc_mask.bool()

        pc_x0 = pc_x0.clone()
        pc_mask = pc_mask.clone().bool()
        if mode == "none":
            pc_mask.zero_()
        elif mode == "prefix":
            keep_frames = min(random.choice(self.pc_conditioning_prefix_frames), pc_mask.shape[0])
            if keep_frames < pc_mask.shape[0]:
                pc_mask[keep_frames:] = False
        else:
            raise ValueError(f"Unsupported pc conditioning mode: {mode}")

        pc_x0 = pc_x0.masked_fill(~pc_mask[..., None], 0)
        return pc_x0, pc_mask

    def _check_all_precomputed_pc_latents_available(self) -> bool:
        if not os.path.isdir(self.pc_latent_dir):
            return False

        available_stems = {path.stem for path in Path(self.pc_latent_dir).glob("*.pt")}
        return all(Path(video_path).stem in available_stems for video_path in self.video_paths)

    def _slice_precomputed_pc_latent(
        self, pc_x0: torch.Tensor, pc_mask: torch.Tensor, start_frame: int, video_basename: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end_frame = start_frame + self.sequence_length
        pc_x0 = pc_x0[start_frame:end_frame]
        pc_mask = pc_mask[start_frame:end_frame]
        if pc_x0.shape[0] != self.sequence_length or pc_mask.shape[0] != self.sequence_length:
            raise PointcloudEncodingConfigurationError(
                f"Precomputed pc_latent for {video_basename} does not cover frames [{start_frame}, {end_frame})"
            )
        return pc_x0, pc_mask.bool()

    def _load_precomputed_pc_latent(
        self, video_basename: str, start_frame: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        pc_path = os.path.join(self.pc_latent_dir, f"{video_basename}.pt")
        if not os.path.exists(pc_path):
            return None

        pc = torch.load(pc_path, map_location="cpu")
        return self._slice_precomputed_pc_latent(pc["x0"], pc["mask"], start_frame, video_basename)

    def _encode_pointcloud_window(self, video_basename: str, start_frame: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        pc_path = os.path.join(self.pointcloud_dir, f"{video_basename}.npy")
        if not os.path.exists(pc_path):
            if self.pc_latent_source == "auto":
                return None
            raise PointcloudEncodingConfigurationError(f"Pointcloud episode not found: {pc_path}")

        episode = np.load(pc_path, allow_pickle=True).item()
        coords = episode.get("coord")
        if coords is None:
            raise PointcloudEncodingConfigurationError(f"Pointcloud episode is missing 'coord': {pc_path}")

        end_frame = start_frame + self.sequence_length
        if len(coords) < end_frame:
            raise PointcloudEncodingConfigurationError(
                f"Pointcloud episode {pc_path} has {len(coords)} frames but needs at least {end_frame}"
            )

        encoder_input = {
            "coord": list(coords[start_frame:end_frame]),
            "grid_size": float(episode.get("grid_size") or self.pc_latent_grid_size),
        }

        encoder = _get_online_pointcloud_encoder(self.pc_encoder_config)
        feats, mask = encoder.encode_batch(
            [encoder_input],
            k=self.pc_latent_k,
            sample=self.pc_latent_sample,
            pad_value=self.pc_latent_pad_value,
            return_mask=True,
            amp=self.pc_latent_amp,
        )
        return feats[0].detach().cpu(), mask[0].detach().cpu().bool()

    def _load_pc_latent_window(self, video_basename: str, start_frame: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self.pc_latent_source != "online":
            precomputed = self._load_precomputed_pc_latent(video_basename, start_frame)
            if precomputed is not None:
                return precomputed
            if self.pc_latent_source == "precomputed":
                raise PointcloudEncodingConfigurationError(
                    f"Missing precomputed pc_latent for {video_basename} in {self.pc_latent_dir}"
                )

        if self.pc_latent_source == "precomputed":
            return None

        return self._encode_pointcloud_window(video_basename, start_frame)

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        start_frame = np.random.randint(0, max_start_idx)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        _batch = vr.get_batch(frame_ids)
        frame_data = _batch.numpy() if hasattr(_batch, "numpy") else _batch.asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps, start_frame

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        else:
            raise ValueError(f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', or 'auto'")

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            log.warning(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                content = f.read()
                # Handle JSON that might not have top-level object
                if not content.strip().startswith("{"):
                    # Wrap in object if needed
                    data = json.loads("{" + content + "}")
                else:
                    data = json.loads(content)

            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    log.warning(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            log.warning(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
        frames, fps, start_frame = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps, start_frame

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video_path = self.video_paths[index]
            video, fps, start_frame = self._get_frames(video_path)
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Load caption based on format
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            pc_latent = self._load_pc_latent_window(video_basename, start_frame)
            if pc_latent is not None:
                pc_x0, pc_mask = self._apply_pc_conditioning_policy(*pc_latent)
                data["pc_latent_x0"] = pc_x0
                data["pc_latent_mask"] = pc_mask

            data["start_frame"] = start_frame
            data["episode_id"] = video_basename
            data["video_basename"] = video_basename
            data["video_path"] = video_path

            if self.caption_format == "json":
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.json")
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
                caption = self._load_text(Path(caption_path))

            data["video"] = video
            data["ai_caption"] = caption

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except PointcloudEncodingConfigurationError:
            raise
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs,  # Ignore extra arguments
) -> DataLoader:
    """Create DataLoader with commonly used parameters.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        sampler: Optional sampler for data loading
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA transfer
        drop_last: Drop incomplete last batch
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        collate_fn: Custom collate function
        **kwargs: Extra arguments (ignored)

    Returns:
        Configured DataLoader
    """
    if getattr(dataset, "requires_main_process_data_loading", False) and num_workers != 0:
        log.warning(
            f"Overriding DataLoader num_workers from {num_workers} to 0 because VideoDataset is doing online pointcloud encoding."
        )
        num_workers = 0
        prefetch_factor = None
        persistent_workers = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # False when using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_train_val_dataloaders(
    dataset_path: str, val_percentage: float, seed: int, video_size: tuple[int, int] = (704, 1280)
):
    video_dir = os.path.join(dataset_path, "videos")
    if not os.path.exists(video_dir):
        log.debug(f"Dataset path {dataset_path} does not exist, returning empty dataloaders")
        return dict(), dict()
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    random.seed(seed)
    random.shuffle(video_paths)

    cutoff = int(len(video_paths) * val_percentage)
    val_video_paths = video_paths[:cutoff]
    train_video_paths = video_paths[cutoff:]

    def get_dataset(video_paths):
        return L(VideoDataset)(
            video_paths=video_paths,
            num_frames=93,
            video_size=video_size,
            dataset_dir=dataset_path,
        )

    ipn_hand_train_dataset = get_dataset(train_video_paths)
    ipn_hand_val_dataset = get_dataset(val_video_paths)

    def get_dataloader(dataset):
        return L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

    return get_dataloader(ipn_hand_train_dataset), get_dataloader(ipn_hand_val_dataset)
