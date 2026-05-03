import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import math
import torch
import torch.nn as nn
import numpy as np
import glob
import json
import spconv.pytorch as spconv
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
# EXP_NAME = "semseg-pt-v3m1-0-base-only-grid"
EXP_NAME = "semseg-pt-v3m1-0-base-cosmos-pcenc"
WEIGHT_NAME = "model_last"

CONFIG_FILE = os.path.join("configs", DATASET, f"{CONFIG}.py")
EXP_DIR = os.path.join("exp", DATASET, EXP_NAME)
WEIGHT_PATH = os.path.join(EXP_DIR, "model", f"{WEIGHT_NAME}.pth")


def _force_spconv_native_algo(module: torch.nn.Module) -> int:
    changed = 0
    if not hasattr(spconv, "ConvAlgo"):
        return changed

    for submodule in module.modules():
        if spconv.modules.is_spconv_module(submodule) and hasattr(submodule, "algo"):
            submodule.algo = spconv.ConvAlgo.Native
            changed += 1
    return changed

def load_ptv3_model():
    cfg = default_config_parser(CONFIG_FILE, {})

    cfg.save_path = EXP_DIR
    cfg.weight = WEIGHT_PATH

    default_setup(cfg)
    model = build_model(cfg.model).cuda()

    if not os.path.isfile(cfg.weight):
        raise RuntimeError(f"=> No checkpoint found at '{cfg.weight}'")

    checkpoint = torch.load(cfg.weight, map_location="cuda", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    new_state_dict = OrderedDict()
    world_size = comm.get_world_size() if hasattr(comm, "get_world_size") else 1

    for key, value in state_dict.items():
        if key.startswith("module."):
            if world_size == 1:
                key = key[7:]  # 去掉 module.
        else:
            if world_size > 1:
                key = "module." + key  # 加上 module.
        new_state_dict[key] = value

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    native_algo_modules = _force_spconv_native_algo(model)
    print(f"Loaded checkpoint from {cfg.weight}")
    if native_algo_modules:
        print(f"Configured {native_algo_modules} spconv modules to use ConvAlgo.Native")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    return model


class PTV3Encoder(torch.nn.Module):
    def __init__(self, ptv3_model: torch.nn.Module):
        super().__init__()
        self.order = ptv3_model.backbone.order if hasattr(ptv3_model, "backbone") else ptv3_model.order
        self.shuffle_orders = (
            ptv3_model.backbone.shuffle_orders
            if hasattr(ptv3_model, "backbone")
            else ptv3_model.shuffle_orders
        )
        self.embedding = (
            ptv3_model.backbone.embedding
            if hasattr(ptv3_model, "backbone")
            else ptv3_model.embedding
        )
        self.enc = (
            ptv3_model.backbone.enc if hasattr(ptv3_model, "backbone") else ptv3_model.enc
        )

    # This encoder is used in the dataloader as an auxiliary preprocessing path.
    # Keeping it eager avoids long first-step TorchInductor compilation stalls.
    @torch.compiler.disable
    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        return point

    @torch.compiler.disable
    def encode_sequence(self, data_dict):
        """
        Encode a sequence of frames stored in data_dict['coord'] (list/tuple).

        - data_dict['coord'] should be a list/tuple of frames (each frame: ndarray or Tensor of shape [N,3]).
        - For keys that are per-frame lists (same length as coord list), the i-th element will be used for frame i.
        """
        coords = data_dict.get("coord", None)
        if coords is None or not isinstance(coords, (list, tuple)):
            raise ValueError("data_dict['coord'] must be a list/tuple of frames")

        points = []
        device = torch.device("cpu")
        # try to get device from model params (correct iterator use)
        try:
            device = next(iter(self.parameters())).device
        except Exception:
            # keep cpu if no parameters found
            device = torch.device("cpu")

        for i, fr in enumerate(coords):
            single = {}
            for k, v in data_dict.items():
                if k == "coord":
                    fr_arr = fr
                    coord_t = torch.from_numpy(fr_arr.astype(np.float32))
                    single["coord"] = coord_t
                elif isinstance(v, (list, tuple)) and len(v) == len(coords):
                    val = v[i]
                    single[k] = torch.from_numpy(val)
                else:
                    single[k] = v

            coord_t = single.get("coord")
            if coord_t is None:
                raise ValueError(f"frame {i} has no coord")

            if not isinstance(coord_t, torch.Tensor):
                coord_t = torch.tensor(np.array(coord_t).astype(np.float32))
                single["coord"] = coord_t

            if "batch" not in single:
                single["batch"] = torch.zeros(coord_t.shape[0], dtype=torch.long)
            if "offset" not in single:
                single["offset"] = torch.tensor([coord_t.shape[0]], dtype=torch.long)
            if "feat" not in single:
                in_ch = getattr(self.embedding, "in_channels", None)
                single["feat"] = torch.zeros((coord_t.shape[0], int(in_ch)), dtype=torch.float32)

            if "grid_size" not in single and "grid_size" in data_dict:
                single["grid_size"] = data_dict["grid_size"]

            # move tensors to model device
            for kk in list(single.keys()):
                if isinstance(single[kk], torch.Tensor):
                    single[kk] = single[kk].to(device)

            with torch.no_grad():
                p = self.forward(single)
            points.append(p)
        return points

    @torch.compiler.disable
    def encode_sequence_stacked(self, data_dict, k: int = 30, sample: str = "first", pad_value: float = 0.0, return_mask: bool = True):
        """
        Encode sequence and return per-frame per-point latents stacked into a tensor of shape (T, k, C).

        Returns:
          - feature: (T, k, C)
          - mask:    (T, k) bool, True=valid token, False=padding (only if return_mask=True)
        """
        points = self.encode_sequence(data_dict)
        if not isinstance(points, (list, tuple)):
            points = [points]
        if len(points) == 0:
            feat_empty = torch.empty((0, k, 0))
            if return_mask:
                return feat_empty, torch.empty((0, k), dtype=torch.bool)
            return feat_empty

        # infer channel, device and dtype from first point
        first = points[0]
        feat0 = first.get("feat") if hasattr(first, "get") else getattr(first, "feat", None)
        if feat0 is None:
            raise RuntimeError("encoded Point has no 'feat' field")
        C = int(feat0.shape[1])
        device = feat0.device
        dtype = feat0.dtype

        out = torch.full((len(points), k, C), float(pad_value), dtype=dtype, device=device)
        mask = torch.zeros((len(points), k), dtype=torch.bool, device=device)

        for i, p in enumerate(points):
            feat = p.get("feat") if hasattr(p, "get") else getattr(p, "feat", None)
            if feat is None:
                continue
            N = int(feat.shape[0])
            if N == 0:
                continue

            if N >= k:
                if sample == "first":
                    sel = torch.arange(k, device=device)
                elif sample == "random":
                    sel = torch.randperm(N, device=device)[:k]
                else:
                    raise NotImplementedError(f"Unknown sample mode: {sample}")
                out[i] = feat[sel]
                mask[i, :k] = True
            else:
                out[i, :N] = feat
                mask[i, :N] = True

        if return_mask:
            return out, mask
        return out

    @torch.compiler.disable
    def encode_batch(
        self,
        batch_data,
        k: int = 30,
        sample: str = "first",
        pad_value: float = 0.0,
        return_mask: bool = True,
        amp: bool = False,
        progress_callback=None,
    ):
        """Batch encode a list of sequence data_dicts.

        batch_data: list/sequence of data_dict, each like used by encode_sequence (must have 'coord' list with same length T)
        Returns:
          - feats: Tensor (B, T, k, C)
          - masks: Tensor (B, T, k) bool (if return_mask)
        Notes: Uses per-time-step concatenation to run one forward per time-step for the whole batch.

        Important: After serialization + sparsify + spconv/transformer, the output token count is NOT equal to total input points.
        So we must split the output by `p.batch` (token ownership) instead of slicing by original `Ns`.
        """
        if not isinstance(batch_data, (list, tuple)):
            batch_data = list(batch_data)
        B = len(batch_data)
        if B == 0:
            raise ValueError("Empty batch_data")

        Ts = [len(d.get("coord", [])) for d in batch_data]
        if len(set(Ts)) != 1:
            raise ValueError(f"All sequences in batch must have same length T, got {Ts}")
        T = Ts[0]

        # lazy init output once we know C/device/dtype from the first non-empty forward output
        out = None
        mask = None
        C = None
        device = None
        dtype = None

        in_ch = getattr(self.embedding, "in_channels", 0)

        for t in range(T):
            coords_list = []
            Ns = []
            grid_size = None
            for i, d in enumerate(batch_data):
                coord = d.get("coord", None)
                if coord is None:
                    raise ValueError("data_dict missing 'coord'")
                fr = coord[t]
                if isinstance(fr, np.ndarray):
                    coord_t = torch.from_numpy(fr.astype(np.float32))
                elif isinstance(fr, torch.Tensor):
                    coord_t = fr.to(torch.float32)
                else:
                    coord_t = torch.tensor(np.array(fr).astype(np.float32))
                Ns.append(int(coord_t.shape[0]))
                coords_list.append(coord_t)
                if grid_size is None and "grid_size" in d:
                    grid_size = d["grid_size"]

            total_N = int(sum(Ns))
            if total_N == 0:
                continue

            # infer device from model (do not assume cuda)
            if device is None:
                try:
                    device = next(iter(self.parameters())).device
                except Exception:
                    device = torch.device("cpu")

            coords_cat = torch.cat([c.to(device) for c in coords_list if c.numel() > 0], dim=0)
            # batch ids for input points (used by Point.sparsify to build sparse indices)
            batch_chunks = [torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(Ns) if n > 0]
            batch_idx = torch.cat(batch_chunks, dim=0) if len(batch_chunks) > 0 else torch.empty((0,), dtype=torch.long, device=device)

            # offset should be prefix sums (cumulative counts)
            offset = torch.tensor(np.cumsum(Ns), dtype=torch.long, device=device)
            feat_init = torch.zeros((total_N, int(in_ch)), dtype=torch.float32, device=device)

            single = {"coord": coords_cat, "batch": batch_idx, "offset": offset, "feat": feat_init}
            if grid_size is not None:
                single["grid_size"] = grid_size

            with torch.no_grad():
                if amp and torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        p = self.forward(single)
                else:
                    p = self.forward(single)

            pts_feat = p.get("feat") if hasattr(p, "get") else getattr(p, "feat", None)
            pts_batch = p.get("batch") if hasattr(p, "get") else getattr(p, "batch", None)
            if pts_feat is None or pts_batch is None:
                # cannot split without batch ownership
                continue

            if progress_callback is not None and (t == 0 or (t + 1) % 10 == 0 or t == T - 1):
                progress_callback(t + 1, T, total_N, int(pts_feat.shape[0]))

            if pts_feat.numel() == 0:
                continue

            if out is None:
                C = int(pts_feat.shape[1])
                dtype = pts_feat.dtype
                # device already set above
                out = torch.full((B, T, k, C), float(pad_value), dtype=dtype, device=device)
                mask = torch.zeros((B, T, k), dtype=torch.bool, device=device)

            # group tokens by output batch id
            # note: pts_batch is per-output-token ownership; values are sample indices in [0, B)
            for i in range(B):
                sel_i = (pts_batch == i).nonzero(as_tuple=False).squeeze(1)
                Ni = int(sel_i.numel())
                if Ni == 0:
                    continue
                seg = pts_feat.index_select(0, sel_i)
                
                if Ni >= k:
                    if sample == "first":
                        pick = torch.arange(k, device=device)
                    elif sample == "random":
                        pick = torch.randperm(Ni, device=device)[:k]
                    else:
                        raise NotImplementedError(f"Unknown sample mode: {sample}")
                    out[i, t] = seg[pick]
                    mask[i, t, :k] = True
                else:
                    out[i, t, :Ni] = seg
                    mask[i, t, :Ni] = True

        if out is None:
            # never got a valid output (all empty)
            out = torch.empty((B, T, k, 0))
            if return_mask:
                return out, torch.zeros((B, T, k), dtype=torch.bool)
            return out

        if return_mask:
            return out, mask
        return out


