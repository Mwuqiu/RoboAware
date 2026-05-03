import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import math
import torch
import torch.nn as nn
import numpy as np
import glob
import json
from typing import Sequence
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from pointcept.engines.defaults import (
    default_config_parser,
    default_setup,
)
from pointcept.models import build_model
from pointcept.utils import comm
from pointcept.models.utils.structure import Point
from pointcept.datasets.defaults import DefaultDataset
from pointcept.datasets.transform import Compose, TRANSFORMS
from timm.models.vision_transformer import Mlp, Attention

DATASET = "so100"
CONFIG = "semseg-pt-v3m1-0-base"
EXP_NAME = "semseg-pt-v3m1-0-base-only-grid"
WEIGHT_NAME = "model_last"

CONFIG_FILE = os.path.join("configs", DATASET, f"{CONFIG}.py")
EXP_DIR = os.path.join("exp", DATASET, EXP_NAME)
WEIGHT_PATH = os.path.join(EXP_DIR, "model", f"{WEIGHT_NAME}.pth")

class TemporalPointDataset(DefaultDataset):
    
    def __init__(
        self,
        split="train",
        data_root="pointflow/generated_pointclouds_dataset",
        window_size: int = 25,
        stride: int = 1,
        precompute_index: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(split=split, data_root=data_root, *args, **kwargs)

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.precompute_index = bool(precompute_index)
        self.data_list = self.get_data_list()

        self.windows = []
        if self.precompute_index:
            for ei, ep_path in enumerate(self.data_list):
                try:
                    T = self._get_episode_length(ep_path)
                except Exception:
                    T = 0
                if T >= self.window_size:
                    for s in range(0, T - self.window_size + 1, self.stride):
                        self.windows.append((ei, s))
        
        transform_cfg = [
            dict(
                type="GridSample",
                grid_size=0.005,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
        ]
        self.transform = Compose(transform_cfg)
        self.transform = None
        # store default grid_size so callers (encode_sequence) can rely on it
        self.grid_size = transform_cfg[0].get("grid_size", 0.005)

    def _get_episode_length(self, ep_path):
        raw = np.load(ep_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            try:
                data_src = raw.item()
            except Exception:
                data_src = {"coord": raw}
        elif isinstance(raw, dict):
            data_src = raw
        else:
            data_src = {"coord": raw}

        if "frames" in data_src:
            return len(data_src["frames"])
        if "coord" in data_src:
            coords = data_src["coord"]
            if hasattr(coords, "__len__"):
                return len(coords)
            if isinstance(coords, np.ndarray) and coords.ndim == 2:
                return 1
        return 0

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            path = os.path.join(self.data_root, split)
            if os.path.isfile(path):
                with open(path) as f:
                    data_list += [os.path.join(self.data_root, d) for d in json.load(f)]
            elif os.path.isdir(path):
                data_list += glob.glob(os.path.join(path, "*.npy"))
            else:
                data_list += glob.glob(os.path.join(self.data_root, split))
        return data_list

    def get_data_name(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        return os.path.splitext(os.path.basename(path))[0]

    def get_split_name(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        return os.path.basename(os.path.dirname(path))

    def __len__(self):
        if self.precompute_index:
            return len(self.windows)
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.precompute_index:
            ep_idx, start = self.windows[idx]
        else:
            ep_idx = idx % len(self.data_list)
            T = self._get_episode_length(self.data_list[ep_idx])
            max_start = max(0, T - self.window_size)
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        ep_path = self.data_list[ep_idx]
        raw = np.load(ep_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            try:
                data_src = raw.item()
            except Exception:
                data_src = {"coord": raw}
        elif isinstance(raw, dict):
            data_src = raw
        else:
            data_src = {"coord": raw}

        if "frames" in data_src:
            frames = list(data_src["frames"])
        else:
            frames = list(data_src.get("coord", []))

        end = start + self.window_size
        window_frames = frames[start:end]

        # base fields (non-temporal)
        data_dict = {k: data_src[k] for k in ["color", "normal", "strength", "instance", "pose"] if k in data_src}

        # segment can be stored either per-point [N] or per-frame [T,N]
        if "segment" in data_src:
            seg = data_src["segment"]
            seg = np.asarray(seg)
            if seg.ndim == 2:
                # [T,N] -> take the first frame window (should be constant anyway)
                seg = seg[start]
            elif seg.ndim == 3 and seg.shape[-1] == 1:
                # legacy [T,N,1]
                seg = seg[start, :, 0]
            data_dict["segment"] = seg.astype(np.int32)

        # temporal extra supervision fields
        for tk in ["body_xpos", "body_xmat", "q"]:
            if tk in data_src:
                arr = np.asarray(data_src[tk])
                if arr.ndim >= 1 and arr.shape[0] >= end:
                    data_dict[tk] = arr[start:end]
                else:
                    data_dict[tk] = arr

        data_dict["coord"] = [np.array(f).astype(np.float32) for f in window_frames]
        data_dict["name"] = os.path.splitext(os.path.basename(ep_path))[0]
        data_dict["split"] = os.path.basename(os.path.dirname(ep_path))
        data_dict["ep_path"] = ep_path
        data_dict["start"] = start

        if hasattr(self, "transform") and self.transform is not None:
            proc_coords = []
            for fr in data_dict["coord"]:
                single = {"coord": np.array(fr).astype(np.float32)}
                out = self.transform(single)
                out_coord = out.get("coord") if isinstance(out, dict) else None
                if out_coord is None:
                    out_coord = np.array(fr).astype(np.float32)
                proc_coords.append(out_coord)
            data_dict["coord"] = proc_coords

        # ensure grid_size present (Point.serialization requires 'grid_size' when no 'grid_coord')
        if "grid_size" not in data_dict:
            data_dict["grid_size"] = getattr(self, "grid_size", 0.005)
        return data_dict
