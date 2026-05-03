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
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models import build_model
from pointcept.utils import comm
from pointcept.models.utils.structure import Point
from pointcept.datasets.defaults import DefaultDataset
from pointcept.datasets.transform import Compose, TRANSFORMS
from timm.models.vision_transformer import Mlp, Attention


def _linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device=None,
):
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
    return betas


class DiffusionSchedule:
    """Precomputed diffusion schedule buffers."""

    def __init__(self, num_steps: int = 1000, schedule: str = "linear", device=None):
        self.num_steps = int(num_steps)
        self.schedule = str(schedule)

        if self.schedule != "linear":
            raise NotImplementedError(f"schedule={schedule} not implemented")

        betas = _linear_beta_schedule(self.num_steps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def to(self, device):
        for k in [
            "betas",
            "alphas",
            "alpha_bar",
            "sqrt_alpha_bar",
            "sqrt_one_minus_alpha_bar",
        ]:
            setattr(self, k, getattr(self, k).to(device))
        return self


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule, noise: torch.Tensor):
    """Sample x_t from x_0 at timestep t.

    Args:
      x0:   [B,T,L,C]
      t:    [B] (0..num_steps-1)
      noise:[B,T,L,C]
    """
    B = x0.shape[0]
    s1 = schedule.sqrt_alpha_bar.index_select(0, t).view(B, 1, 1, 1)
    s2 = schedule.sqrt_one_minus_alpha_bar.index_select(0, t).view(B, 1, 1, 1)
    return s1 * x0 + s2 * noise


def v_target(x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule, noise: torch.Tensor):
    """Compute v target.

    v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x0
    """
    B = x0.shape[0]
    s1 = schedule.sqrt_alpha_bar.index_select(0, t).view(B, 1, 1, 1)
    s2 = schedule.sqrt_one_minus_alpha_bar.index_select(0, t).view(B, 1, 1, 1)
    return s1 * noise - s2 * x0


def x0_from_v(xt: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule, v: torch.Tensor):
    """Reconstruct x0 given v-pred and xt.

    Using:
      xt = a * x0 + b * eps
      v  = a * eps - b * x0
    => x0 = a * xt - b * v

    Shapes:
      xt,v: [B,T,L,C]
      t:    [B]
    """
    B = xt.shape[0]
    a = schedule.sqrt_alpha_bar.index_select(0, t).view(B, 1, 1, 1)
    b = schedule.sqrt_one_minus_alpha_bar.index_select(0, t).view(B, 1, 1, 1)
    return a * xt - b * v


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Masked MSE where mask indicates valid tokens.

    Args:
      pred/target: [B,T,L,C]
      mask:        [B,T,L] bool
    """
    m = mask.unsqueeze(-1).float()
    diff2 = (pred - target) ** 2
    diff2 = diff2 * m
    denom = m.sum() * pred.shape[-1]
    if denom.item() == 0:
        return diff2.mean()
    return diff2.sum() / denom


class PointCloudDiT(nn.Module):    
    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    class TimestepEmbedder(nn.Module):
        def __init__(self, hidden_size: int, freq_dim: int = None):
            super().__init__()
            self.hidden_size = hidden_size
            self.freq_dim = freq_dim or hidden_size
            self.mlp = nn.Sequential(
                nn.Linear(self.freq_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        def forward(self, t: torch.Tensor):
            emb = self.timestep_embedding(t, self.freq_dim)
            return self.mlp(emb)

        @staticmethod
        def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
                / half
            )
            args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
            if dim % 2 == 1:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
            return emb

    class DiTBlock(nn.Module):
        def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **kwargs)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=nn.GELU,
                drop=0,
            )
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

        def forward(self, x, t_emb):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(t_emb).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attn(
                PointCloudDiT.modulate(self.norm1(x), shift_msa, scale_msa)
            )
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                PointCloudDiT.modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
            return x

    def __init__(
        self,
        input_size=128,      # L (tokens per frame)
        in_channels=512,     # C
        hidden_size=512,     # D
        depth=12,
        num_heads=8,
        num_frames=50,       # T (window size)
        learn_sigma=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.x_embed = nn.Linear(in_channels, hidden_size)
        self.t_embedder = PointCloudDiT.TimestepEmbedder(hidden_size)

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1, input_size, hidden_size))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, num_frames, 1, hidden_size))

        self.blocks = nn.ModuleList([
            PointCloudDiT.DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, self.out_channels, bias=True)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed_spatial, std=0.02)
        nn.init.normal_(self.pos_embed_temporal, std=0.02)

        nn.init.xavier_uniform_(self.x_embed.weight)
        nn.init.constant_(self.x_embed.bias, 0)

        # adaLN-Zero init: make blocks start as (near) identity
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

    def forward(self, x, t, mask=None):
        """
        x:    [B, T, L, C]
        t:    [B]
        mask: [B, T, L] bool (True=valid)
        """
        B, T, L, C = x.shape

        x = self.x_embed(x)  # [B,T,L,D]
        x = x + self.pos_embed_spatial[:, :, :L, :] + self.pos_embed_temporal[:, :T, :, :]

        t_emb = self.t_embedder(t)  # [B,D]

        x = x.view(B, T * L, -1)  # [B,N,D]

        mask_flat = None
        if mask is not None:
            mask_flat = mask.view(B, T * L).to(x.device)

        for block in self.blocks:
            x = block(x, t_emb)
            if mask_flat is not None:
                x = x * mask_flat.unsqueeze(-1).float()

        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=1)
        x = PointCloudDiT.modulate(self.final_norm(x), shift, scale)
        x = self.final_linear(x)  # [B,N,out]

        if mask_flat is not None:
            x = x * mask_flat.unsqueeze(-1).float()

        return x.view(B, T, L, -1)