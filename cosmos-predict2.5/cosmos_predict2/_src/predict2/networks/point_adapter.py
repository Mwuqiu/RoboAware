from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PCEncoder(nn.Module):

    def __init__(self, d_pc: int, d_a: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_pc, d_a, bias=True),
            nn.SiLU(),
            nn.Linear(d_a, d_a, bias=True),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        return self.mlp(pc)


class PointAdapter(nn.Module):
    """
    PointAdapter：将点云信息以 zero-init 残差方式注入 Cosmos 主干。
    内部仅使用与主干同构的 Cosmos Block，不再包含轻量自定义 Block 分支。

    架构：
        PC latent [B, T_pc, K, D_pc]
              │
        [PCEncoder]  →  [B, T, K, d_a]          (时间维度对齐到主干 T)
              │
        ┌─────────────────────────────────┐
        │  Cosmos Block Stage 0           │  ←  接收主干 block_{inj[0]-1} 的输出投影
        └──────────────┬──────────────────┘
                   │ after_proj(d_a → D_main), zero-init
                       ▼
          主干 Block 3 output  +=  residual_0
                       │
        ┌──────────────▼──────────────────┐
        │  Cosmos Block Stage 1           │  ←  接收主干 block_{inj[1]-1} 的输出投影
        └──────────────┬──────────────────┘
                   │ after_proj
                       ▼
          主干 Block 7 output  +=  residual_1
                       │
                   ... (共 N 个注入点) ...

    参数：
        d_pc        (int):  点云 latent 原始维度（= crossattn_emb_channels）
        d_main      (int):  Cosmos 主干维度（= model_channels = 8192）
        d_a         (int):  Adapter 内部维度，默认 512
        num_adapter_blocks (int): 注入点数量，默认 7
        adapter_block_depth (int): 每个注入点内部串联的 Cosmos Block 数量，默认 1
        inject_every_k (int): 每隔 k 个主干 block 注入一次，默认 4
        num_main_blocks (int): 主干总 block 数，默认 28
        block_factory: Cosmos Block 构造器（必填）
        block_factory_kwargs: 传给 block_factory 的参数
    """

    def __init__(
        self,
        d_pc: int,
        d_main: int,
        d_a: int = 512,
        num_adapter_blocks: int = 7,
        adapter_block_depth: int = 1,
        num_heads: int = 8,
        inject_block_ids: Optional[List[int]] = None,
        inject_every_k: int = 4,
        num_main_blocks: int = 28,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        block_factory: Optional[Callable[..., nn.Module]] = None,
        block_factory_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        del num_heads, mlp_ratio, dropout
        self.d_a = d_a
        self.d_main = d_main
        self.num_adapter_blocks = int(num_adapter_blocks)
        self.adapter_block_depth = int(adapter_block_depth)
        if block_factory is None:
            raise ValueError("PointAdapter now requires block_factory (Cosmos Block) and no longer supports legacy blocks")

        if self.adapter_block_depth < 1:
            raise ValueError(
                f"adapter_block_depth must be >= 1, got {self.adapter_block_depth}"
            )

        # ── 注入点计算 ──────────────────────────────────────────────────────
        if inject_block_ids is not None:
            normalized_ids = sorted(set(int(i) for i in inject_block_ids))
            if not normalized_ids:
                raise ValueError("inject_block_ids is empty")
            if normalized_ids[0] < 0 or normalized_ids[-1] >= num_main_blocks:
                raise ValueError(
                    f"inject_block_ids must be in [0, {num_main_blocks - 1}], got {normalized_ids}"
                )
            self.inject_block_ids = normalized_ids
            self.num_adapter_blocks = len(self.inject_block_ids)
        else:
            # 注入点为每组最后一个 block 的 idx（0-indexed）
            # k=4: [3, 7, 11, 15, 19, 23, 27]
            self.inject_block_ids = [
                inject_every_k * (i + 1) - 1
                for i in range(self.num_adapter_blocks)
                if inject_every_k * (i + 1) - 1 < num_main_blocks
            ]
            if len(self.inject_block_ids) != self.num_adapter_blocks:
                raise ValueError(
                    f"注入点数量 {len(self.inject_block_ids)} 与 num_adapter_blocks "
                    f"{self.num_adapter_blocks} 不匹配，请检查 inject_every_k / num_main_blocks 设置。"
                )

        # ── 子模块 ──────────────────────────────────────────────────────────
        self.pc_encoder = PCEncoder(d_pc=d_pc, d_a=d_a)
        self.x_proj = nn.Linear(d_main, d_a, bias=False) if d_main != d_a else nn.Identity()
        self.t_proj = nn.Linear(d_main, d_a, bias=False) if d_main != d_a else nn.Identity()

        self.before_proj = nn.Linear(d_a, d_a, bias=True)
        self.after_projs = nn.ModuleList([nn.Linear(d_a, d_main, bias=True) for _ in range(self.num_adapter_blocks)])

        block_factory_kwargs = block_factory_kwargs or {}
        self.adapter_blocks = nn.ModuleList([
            nn.ModuleList(
                [block_factory(x_dim=d_a, **block_factory_kwargs) for _ in range(self.adapter_block_depth)]
            )
            for _ in range(self.num_adapter_blocks)
        ])
        self._init_weights()

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def _init_weights(self):
        # PC Encoder：标准初始化
        for m in self.pc_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # x_proj：标准初始化
        if isinstance(self.x_proj, nn.Linear):
            nn.init.xavier_uniform_(self.x_proj.weight)

        if isinstance(self.t_proj, nn.Linear):
            nn.init.xavier_uniform_(self.t_proj.weight)

        # Adapter stage 内部：复用 Cosmos Block 初始化。
        for stage in self.adapter_blocks:
            for block in stage:
                if hasattr(block, "init_weights"):
                    block.init_weights()

        # GeoAdapter 风格 zero-init：before_proj + after_proj 全零
        nn.init.zeros_(self.before_proj.weight)
        nn.init.zeros_(self.before_proj.bias)
        for after_proj in self.after_projs:
            nn.init.zeros_(after_proj.weight)
            nn.init.zeros_(after_proj.bias)

    # ── 时间维度对齐 ──────────────────────────────────────────────────────────

    @staticmethod
    def _align_temporal(
        pc: torch.Tensor,       # [B, T_pc, K, d_a]
        T_target: int,
    ) -> torch.Tensor:
        """
        将点云时间维度 T_pc 对齐到主干时间维度 T_target。
        使用自适应平均池化（沿时间轴），支持 T_pc != T_target 的任意情况。

        策略：
          - T_pc == T_target：直接返回
          - T_pc >  T_target：沿 T 做 adaptive_avg_pool1d 下采样
          - T_pc <  T_target：线性插值上采样
        """
        B, T_pc, K, d_a = pc.shape
        if T_pc == T_target:
            return pc

        # reshape 为 [B*K, d_a, T_pc]，沿时间做 1D 池化/插值
        pc_BK_da_T = rearrange(pc, "b t k d -> (b k) d t")

        if T_pc > T_target:
            # 下采样：adaptive average pooling
            pc_BK_da_T = F.adaptive_avg_pool1d(pc_BK_da_T, T_target)
        else:
            # 上采样：线性插值
            pc_BK_da_T = F.interpolate(
                pc_BK_da_T, size=T_target, mode="linear", align_corners=False
            )

        pc_aligned = rearrange(pc_BK_da_T, "(b k) d t -> b t k d", b=B, k=K)
        return pc_aligned

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pc_latent: torch.Tensor,                # [B, T_pc, K, D_pc]
        pc_mask: Optional[torch.Tensor],        # [B, T_pc, K] bool 或 None
        main_block_outputs: List[torch.Tensor], # 每个注入点前一个 block 的输出
                                                # 每个元素: [B, T, H, W, D_main]
        t_embedding_B_T_D: Optional[torch.Tensor] = None,
        crossattn_emb: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            pc_latent:          点云 latent，[B, T_pc, K, D_pc]
            pc_mask:            点云有效掩码，[B, T_pc, K]，True 表示有效点；可为 None
            main_block_outputs: 长度为 num_adapter_blocks 的列表，
                                第 i 个元素是主干第 inject_block_ids[i] 个 block
                                执行完毕后的特征 [B, T, H, W, D_main]

        Returns:
            residuals: 长度为 num_adapter_blocks 的列表，
                       第 i 个元素是要加到主干第 inject_block_ids[i] 输出上的残差
                       shape: [B, T, H, W, D_main]
        """
        assert len(main_block_outputs) == self.num_adapter_blocks, (
            f"期望 {self.num_adapter_blocks} 个主干输出，实际收到 {len(main_block_outputs)} 个"
        )

        B, T_pc, K, D_pc = pc_latent.shape
        # 取第一个主干输出推断 T, H, W
        B_, T, H, W, D_main = main_block_outputs[0].shape
        assert B_ == B

        # ── Step 1: PC Encoder：[B, T_pc, K, D_pc] → [B, T_pc, K, d_a] ──────
        pc_feat = self.pc_encoder(pc_latent)                    # [B, T_pc, K, d_a]

        # ── Step 2: 时间对齐：T_pc → T ─────────────────────────────────────
        pc_feat = self._align_temporal(pc_feat, T)              # [B, T, K, d_a]

        # 同步对齐 mask
        if pc_mask is not None:
            # mask: [B, T_pc, K] → [B, T, K]
            # 用最近邻插值保持 bool 语义
            pc_mask_float = pc_mask.float()                     # [B, T_pc, K]
            pc_mask_BK_T = rearrange(pc_mask_float, "b t k -> (b k) 1 t")
            if T_pc != T:
                pc_mask_BK_T = F.interpolate(
                    pc_mask_BK_T, size=T, mode="nearest"
                )
            pc_mask_aligned = rearrange(
                pc_mask_BK_T, "(b k) 1 t -> b t k", b=B, k=K
            ).bool()                                            # [B, T, K]
        else:
            pc_mask_aligned = None

        # ── Step 3: 逐注入点处理 ────────────────────────────────────────────
        residuals = []

        # pc_feat 在各 Adapter Block 之间串行传递（类似 ControlNet 的特征链）
        pc_feat_BT_K_da = rearrange(pc_feat, "b t k d -> (b t) k d")   # [B*T, K, d_a]
        if pc_mask_aligned is not None:
            pc_mask_BT_K = rearrange(pc_mask_aligned, "b t k -> (b t) k")  # [B*T, K]
        else:
            pc_mask_BT_K = None

        for i, x_main in enumerate(main_block_outputs):
            pc_feat_BT_K_da, residual = self.apply_stage(
                adapter_idx=i,
                pc_feat_BT_K_da=pc_feat_BT_K_da,
                pc_mask_BT_K=pc_mask_BT_K,
                x_main=x_main,
                t_embedding_B_T_D=t_embedding_B_T_D,
                crossattn_emb=crossattn_emb,
            )
            residuals.append(residual)
        return residuals

    def apply_stage(
        self,
        adapter_idx: int,
        pc_feat_BT_K_da: torch.Tensor,
        pc_mask_BT_K: Optional[torch.Tensor],
        x_main: torch.Tensor,
        t_embedding_B_T_D: Optional[torch.Tensor] = None,
        crossattn_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one injection stage and return updated pc features plus residual."""
        B, T, H, W, _ = x_main.shape
        
        if t_embedding_B_T_D is None or crossattn_emb is None:
            raise ValueError("Cosmos Block adapter requires t_embedding_B_T_D and crossattn_emb")
        
        stage_blocks = self.adapter_blocks[adapter_idx]
        c_B_T_1_K_da = rearrange(pc_feat_BT_K_da, "(b t) k d -> b t 1 k d", b=B, t=T)
        # 与 GeoAdapter 一致：仅首个 stage 用 before_proj 将主干信息注入 adapter 分支。
        if adapter_idx == 0:
            x_global_B_T_da = self.x_proj(x_main).mean(dim=(2, 3))
            x_global_B_T_1_1_da = rearrange(x_global_B_T_da, "b t d -> b t 1 1 d")
            c_B_T_1_K_da = self.before_proj(c_B_T_1_K_da) + x_global_B_T_1_1_da

        emb_B_T_da = self.t_proj(t_embedding_B_T_D)

        adapter_context = crossattn_emb[0] if isinstance(crossattn_emb, tuple) else crossattn_emb

        for block in stage_blocks:
            c_B_T_1_K_da = block(
                c_B_T_1_K_da,
                emb_B_T_da,
                adapter_context,
                rope_emb_L_1_1_D=None,
                adaln_lora_B_T_3D=None,
                extra_per_block_pos_emb=None,
            )

        if pc_mask_BT_K is not None:
            pc_mask_B_T_K = rearrange(pc_mask_BT_K, "(b t) k -> b t k", b=B, t=T)
            c_B_T_1_K_da = c_B_T_1_K_da * pc_mask_B_T_K[:, :, None, :, None].to(c_B_T_1_K_da.dtype)
        else:
            pc_mask_B_T_K = None

        c_skip_B_T_1_K_D = self.after_projs[adapter_idx](c_B_T_1_K_da)
        c_skip_B_T_K_D = c_skip_B_T_1_K_D.squeeze(2)

        if pc_mask_B_T_K is not None:
            mask_B_T_K_1 = pc_mask_B_T_K[:, :, :, None].to(c_skip_B_T_K_D.dtype)
            valid_count = mask_B_T_K_1.sum(dim=2).clamp(min=1.0)
            pc_global_B_T_D = (c_skip_B_T_K_D * mask_B_T_K_1).sum(dim=2) / valid_count
        else:
            pc_global_B_T_D = c_skip_B_T_K_D.mean(dim=2)

        residual = pc_global_B_T_D[:, :, None, None, :].expand(-1, -1, H, W, -1)
        pc_feat_next = rearrange(c_B_T_1_K_da.squeeze(2), "b t k d -> (b t) k d")
        return pc_feat_next, residual