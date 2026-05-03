import math
import torch
import torch.nn as nn


def normalize_sequence_xyz(
    coords: torch.Tensor,
    eps: float = 1e-6,
    mode: str = "max_norm",
):
    """Normalize a point cloud sequence.

    Args:
        coords: [B,T,N,3] float
        mode:
          - "max_norm": divide by max(||x||) over (T,N)
          - "std": divide by std over (T,N,3)

    Returns:
        coords_n: [B,T,N,3]
        center:   [B,1,1,3]
        scale:    [B,1,1,1]
    """
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be [B,T,N,3], got {tuple(coords.shape)}")

    center = coords.mean(dim=(1, 2), keepdim=True)  # [B,1,1,3]
    x = coords - center

    if mode == "max_norm":
        # max radius
        scale = x.norm(dim=-1, keepdim=True).amax(dim=(1, 2), keepdim=True).clamp(min=eps)
    elif mode == "std":
        scale = x.reshape(x.shape[0], -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=eps)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    return x / scale, center, scale


def denormalize_sequence_xyz(coords_n: torch.Tensor, center: torch.Tensor, scale: torch.Tensor):
    return coords_n * scale + center


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_mlp = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_key_padding_mask: torch.Tensor | None = None):
        """Cross-attention.

        Args:
            q:  [B,Nq,D]
            kv: [B,Nk,D]
            kv_key_padding_mask: [B,Nk] where True means "ignore" (padded)
        """
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn, key_padding_mask=kv_key_padding_mask, need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q


class LatentToPointDecoder(nn.Module):
    """Decode per-frame latent tokens (K,C) into a fixed-size point set (N,3).

    Input:
      - z:    [B,T,K,C]
      - mask: [B,T,K] bool, True=valid token

    Output:
      - xyz:  [B,T,N,3]

    Notes:
      - N is fixed (e.g., 2000)
      - Uses learnable point queries + several cross-attn blocks.
    """

    def __init__(
        self,
        in_channels: int,
        num_points: int = 2000,
        d_model: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_points = int(num_points)
        self.d_model = int(d_model)

        self.token_proj = nn.Linear(self.in_channels, self.d_model)

        # learnable queries represent output points
        self.point_queries = nn.Parameter(torch.randn(self.num_points, self.d_model) * 0.02)

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=self.d_model,
                    n_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )

        self.norm_out = nn.LayerNorm(self.d_model)
        self.xyz_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 3),
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None):
        if z.ndim != 4:
            raise ValueError(f"z must be [B,T,K,C], got {tuple(z.shape)}")
        B, T, K, C = z.shape
        if C != self.in_channels:
            raise ValueError(f"z last dim C={C} != in_channels={self.in_channels}")

        # flatten frames as batch
        z = z.reshape(B * T, K, C)
        kv = self.token_proj(z)  # [B*T,K,D]

        kv_pad_mask = None
        if mask is not None:
            if mask.shape != (B, T, K):
                raise ValueError(f"mask must be [B,T,K], got {tuple(mask.shape)}")
            # key_padding_mask: True = ignore
            kv_pad_mask = (~mask).reshape(B * T, K)

        q = self.point_queries.unsqueeze(0).expand(B * T, -1, -1).contiguous()  # [B*T,N,D]

        for blk in self.blocks:
            q = blk(q, kv, kv_key_padding_mask=kv_pad_mask)

        out = self.xyz_head(self.norm_out(q))  # [B*T,N,3]
        out = out.reshape(B, T, self.num_points, 3)
        return out


def chamfer_l2(x: torch.Tensor, y: torch.Tensor):
    """Naive Chamfer-L2 (squared) distance.

    Args:
        x: [B,N,3]
        y: [B,M,3]

    Returns:
        scalar loss

    Warning:
        O(N*M) memory/time. With N=M=2000 this is heavy.
        Prefer installing a CUDA chamfer implementation for real training.
    """
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("x,y must be [B,N,3] and [B,M,3]")
    # [B,N,M]
    dist2 = torch.cdist(x, y, p=2) ** 2
    # [B,N]
    x2y = dist2.min(dim=2).values
    # [B,M]
    y2x = dist2.min(dim=1).values
    return (x2y.mean() + y2x.mean())
