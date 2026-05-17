"""PointAdapter A.v4

V4 (差异 vs V3):
─────────────────
* 新增 learnable null PC tokens (per K-position). 在 cross-attn 之前, 用 pc_mask
  做 per-token 替换: True (valid voxel) 保留 PC encoder 输出; False (padding) 用
  learnable null token 替代. 这样 cross-attn 看到的 30 个 token 永远都有内容,
  模型可以学习 "real 还是 null" 的结构化 attention, 不再被迫退化为 uniform pool.
* 删除 V3 末尾的 frame-level mask (delta * frame_visible) — 那段在实际中是 dead
  code (每 frame 至少 1 valid voxel, frame_visible 永远 True), 而且引入了诱导
  uniform attention 的训练约束.
* none mode (PC 全 0) 下, 所有 token 都被 null replace, adapter 仍可贡献一个
  default residual, 不再 freeze.

设计要点 (沿用 V3)
────────────────
1. PC 通过 Cosmos Block 的 cross-attn (K/V) 注入主干, 不再 mean-pool + spatial broadcast.
2. Adapter Block 与 backbone Cosmos Block 同构. x_dim=d_main, context_dim=d_main;
   init/adaLN-zero 复用 backbone 已验证的 scheme.
3. Block 在 (B*T) 维度上 per-frame 工作: cross-attn K/V 是该 latent t 的 PC token.
4. 不用 d_a 中间瓶颈 / x_proj / before_proj / after_projs.
5. PC 在 stage 间 passthrough, 不更新.

外部契约 (vs V3)
────────────────
PointAdapter.__init__ 多了 `pc_latent_k=30` 参数 (用来 pre-allocate null tokens).
其他接口完全不变 — minimal_v4_dit.py 不需要再加 K 参数 (默认 30 即可, 跟数据集对齐).

注意
────
- d_a 必须等于 d_main.
- adapter_block_depth 必须为 1.
- V3 ckpt 缺少 null_pc_tokens 参数, V4 不能直接 resume V3 ckpt. 需要从 backbone
  base checkpoint 重训 (null tokens 从 trunc_normal_(std=0.02) 起步).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PCEncoder(nn.Module):
    """把 Pointcept 输出的 PC latent 投到主干维度.

    backbone 在 forward 入口直接 call self.pc_encoder(pc_latent_x0).
    输入 [B, T_pc, K, D_pc], 输出 [B, T_pc, K, d_a] (d_a == d_main).
    """

    def __init__(self, d_pc: int, d_a: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_pc, d_a, bias=True),
            nn.SiLU(),
            nn.Linear(d_a, d_a, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        return self.mlp(pc)


class PointAdapter(nn.Module):
    """A.v4: 同构 Cosmos Block + cross-attn + learnable null PC tokens for padded positions."""

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
        pc_latent_k: int = 30,    # NEW (V4): K (PC token count per frame); used to pre-allocate null tokens.
    ):
        super().__init__()
        del num_heads, mlp_ratio, dropout  # 通过 block_factory_kwargs 传给 Cosmos Block

        if block_factory is None:
            raise ValueError("PointAdapter requires block_factory (Cosmos Block)")
        if d_a != d_main:
            raise ValueError(
                f"A.v3 要求 d_a == d_main (取消 d_a 中间瓶颈). "
                f"got d_a={d_a}, d_main={d_main}. 在 experiment 里把 point_adapter_d_a=None 即可."
            )
        if int(adapter_block_depth) != 1:
            raise ValueError(
                f"A.v3 不支持 adapter_block_depth>1 (每个 inject 点单 Block). "
                f"got {adapter_block_depth}"
            )

        self.d_a = d_a
        self.d_main = d_main
        self.adapter_block_depth = 1
        self.pc_latent_k = int(pc_latent_k)

        # ── 注入点解析 (与原版一致) ─────────────────────────────────────────
        if inject_block_ids is not None:
            normalized_ids = sorted({int(i) for i in inject_block_ids})
            if not normalized_ids:
                raise ValueError("inject_block_ids is empty")
            if normalized_ids[0] < 0 or normalized_ids[-1] >= num_main_blocks:
                raise ValueError(
                    f"inject_block_ids must be in [0, {num_main_blocks - 1}], got {normalized_ids}"
                )
            self.inject_block_ids = normalized_ids
            self.num_adapter_blocks = len(self.inject_block_ids)
        else:
            self.num_adapter_blocks = int(num_adapter_blocks)
            self.inject_block_ids = [
                inject_every_k * (i + 1) - 1
                for i in range(self.num_adapter_blocks)
                if inject_every_k * (i + 1) - 1 < num_main_blocks
            ]
            if len(self.inject_block_ids) != self.num_adapter_blocks:
                raise ValueError(
                    f"注入点数量 {len(self.inject_block_ids)} 与 num_adapter_blocks "
                    f"{self.num_adapter_blocks} 不匹配, 请检查 inject_every_k / num_main_blocks."
                )

        # ── PCEncoder: D_pc → d_main, xavier init ──────────────────────────
        self.pc_encoder = PCEncoder(d_pc=d_pc, d_a=d_a)

        # ── (V4 NEW) Learnable null PC tokens. Shape (1, 1, K, d_main).
        # 当 pc_mask 中某个 (frame, token) 是 False (padding 或 zero-coord 的 fallback voxel),
        # apply_stage 用 null_pc_tokens 在该位置替换 PC encoder 输出, 让 cross-attn 永远
        # 看到 "real-or-null" 的有内容 token. (1, 1, ...) shape 方便 expand 到 (B*T, K, D).
        self.null_pc_tokens = nn.Parameter(
            torch.empty(1, 1, self.pc_latent_k, d_main)
        )

        # ── Adapter Blocks: 每个 inject 点一个 Cosmos Block, 与 backbone 同构 ──
        # 关键: x_dim=d_main (输入 video tokens), context_dim=d_main (PC 已投到 d_main).
        block_factory_kwargs = dict(block_factory_kwargs or {})
        block_factory_kwargs["context_dim"] = d_main
        block_factory_kwargs["image_context_dim"] = None  # 不用 I2V cross-attn

        self.adapter_blocks = nn.ModuleList(
            [block_factory(x_dim=d_main, **block_factory_kwargs) for _ in range(self.num_adapter_blocks)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # PCEncoder: re-init explicitly here (xavier_uniform_). We don't rely on
        # PCEncoder's own __init__-time init, because cosmos instantiates the
        # whole model on `device='meta'` first and only later does
        #   net.to_empty(device=...); net.init_weights()
        # to materialize and re-init. So *this* method is the canonical entry
        # point for adapter init and must reset every trainable parameter.
        for m in self.pc_encoder.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # (V4 NEW) null PC tokens: small random init so gradients can flow.
        # Zero-init would block learning entirely (delta would be deterministic).
        nn.init.trunc_normal_(self.null_pc_tokens, std=0.02)

        # Each adapter Block uses the backbone Cosmos Block's own init_weights
        # (includes adaLN-zero for the modulation last layer, trunc_normal for
        # q/k/v in self/cross attention).
        for block in self.adapter_blocks:
            if hasattr(block, "init_weights"):
                block.init_weights()

    def init_weights(self) -> None:
        """Public init hook called by MinimalV4DiT.init_weights() after the
        framework's meta -> cpu / meta -> cuda materialization. See `_init_weights`."""
        self._init_weights()

    # ───────────────────────── 时间维度对齐 ────────────────────────────────
    @staticmethod
    def _align_temporal(pc: torch.Tensor, T_target: int) -> torch.Tensor:
        """[B, T_pc, K, d] → [B, T, K, d].

        backbone 在 forward 入口直接 call (minimal_v4_dit.py:2176), 接口必须保留.
        当前用 adaptive_avg_pool, 简单稳定; 已知缺点是会抹平相邻帧的运动差异.
        若以后要换 learnable 1D conv, 改这里即可, 不影响 backbone.
        """
        B, T_pc, K, d = pc.shape
        if T_pc == T_target:
            return pc
        pc_BK_d_T = rearrange(pc, "b t k d -> (b k) d t")
        if T_pc > T_target:
            pc_BK_d_T = F.adaptive_avg_pool1d(pc_BK_d_T, T_target)
        else:
            pc_BK_d_T = F.interpolate(
                pc_BK_d_T, size=T_target, mode="linear", align_corners=False
            )
        return rearrange(pc_BK_d_T, "(b k) d t -> b t k d", b=B, k=K)

    # ───────────────────────── 单 stage 注入 ───────────────────────────────
    def apply_stage(
        self,
        adapter_idx: int,
        pc_feat_BT_K_da: torch.Tensor,         # [B*T, K, d_main], 来自 PCEncoder + _align_temporal
        pc_mask_BT_K: Optional[torch.Tensor],  # [B*T, K] bool, True=有效 voxel, False=padded/null slot
        x_main: torch.Tensor,                  # [B, T, H, W, d_main], backbone block 输出
        t_embedding_B_T_D: Optional[torch.Tensor] = None,
        crossattn_emb: Optional[torch.Tensor] = None,  # 传 backbone 的 text emb, 本设计不用
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A.v4 注入: video tokens 作为 query, PC tokens (real OR null) 作为 cross-attn K/V.

        Returns:
            pc_feat_next: [B*T, K, d_main]  (passthrough; PC 不在 stage 间更新)
            residual:     [B, T, H, W, d_main]  直接 add 到 backbone block output
        """
        del crossattn_emb  # adapter 不用 text; PC 自带空间结构

        if t_embedding_B_T_D is None:
            raise ValueError("Cosmos Block adapter requires t_embedding_B_T_D")

        B, T, H, W, D = x_main.shape
        BT, K, _ = pc_feat_BT_K_da.shape

        # ── (V4 NEW) per-token null replacement ─────────────────────────────
        # 把 padding 位置 (mask=False) 替换为 learnable null_pc_tokens. 这样 cross-attn
        # 看到的 K=30 个 token 永远有内容, 模型可以学 "real vs null" 的结构化 attention,
        # 避免被迫退化为 uniform pool (V3 的失败模式).
        if pc_mask_BT_K is not None:
            null_BT_K_D = self.null_pc_tokens.to(pc_feat_BT_K_da.dtype).expand(BT, K, D)
            pc_feat_BT_K_da = torch.where(
                pc_mask_BT_K.unsqueeze(-1),    # [B*T, K, 1] broadcast over D
                pc_feat_BT_K_da,                # real voxel encoding
                null_BT_K_D,                    # learnable null prior
            )

        # ── reshape video 到 per-frame batch ──
        # Block 看到的 input shape 是 (B*T, 1, H, W, D), 内部 self-attn over (1*H*W) per frame
        x_BT_1_H_W_D = rearrange(x_main, "b t h w d -> (b t) 1 h w d")
        # t_embedding 形状通常是 (B, 1, D) — 整段 video 共用一个 diffusion timestep emb,
        # 在 backbone 内部依靠 broadcast 作用到所有 latent frame.
        # 我们把 video 拆到 per-frame batch 后, 每个 (b, t) sample 也都用对应 b 的同一个 t_emb,
        # 所以把 t_emb 显式 expand 到 (B, T, D) 再合并 batch 维.
        t_emb = t_embedding_B_T_D
        if t_emb.shape[1] != T:
            if t_emb.shape[1] == 1:
                t_emb = t_emb.expand(B, T, t_emb.shape[-1])
            else:
                raise ValueError(
                    f"t_embedding_B_T_D shape mismatch: expected dim1 == 1 or {T}, got {tuple(t_emb.shape)}"
                )
        emb_BT_1_D = rearrange(t_emb, "b t d -> (b t) 1 d")

        # ── 调用同构 Cosmos Block, cross-attn K/V = 当前 latent t 的 (real + null) PC token ──
        block = self.adapter_blocks[adapter_idx]
        out_BT_1_H_W_D = block(
            x_BT_1_H_W_D,
            emb_BT_1_D,
            pc_feat_BT_K_da,            # crossattn_emb 替换为 PC tokens [B*T, K, D]
            rope_emb_L_1_1_D=None,      # adapter 内部 self-attn 不需要 RoPE (per-frame spatial)
            adaln_lora_B_T_3D=None,     # use_adaln_lora=False (跟 backbone 配置一致)
            extra_per_block_pos_emb=None,
        )

        # delta = Block(x) - x; adaLN-zero 保证初始 delta ≈ 0
        delta_BT_1_H_W_D = out_BT_1_H_W_D - x_BT_1_H_W_D
        delta_B_T_H_W_D = rearrange(delta_BT_1_H_W_D, "(b t) 1 h w d -> b t h w d", b=B, t=T)

        # (V4) 去掉 V3 末尾的 frame-level mask multiply — 在 V3 中是 dead code
        # (每 frame 至少 1 valid voxel, frame_visible 永远 True), 而且诱导了 uniform attention
        # 的训练约束 (模型必须对 0 内容和真 PC 输出"等价的小 delta"). null token 接管这个职责.

        # PC tokens 在各 stage 之间不更新, 直接 passthrough.
        return pc_feat_BT_K_da, delta_B_T_H_W_D

    # forward 接口保留, 但 backbone 实际只 call apply_stage. 留作 standalone debugging.
    def forward(
        self,
        pc_latent: torch.Tensor,                # [B, T_pc, K, D_pc]
        pc_mask: Optional[torch.Tensor],        # [B, T_pc, K] 或 None
        main_block_outputs: List[torch.Tensor], # 每个元素 [B, T, H, W, D_main]
        t_embedding_B_T_D: Optional[torch.Tensor] = None,
        crossattn_emb: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        assert len(main_block_outputs) == self.num_adapter_blocks
        B, T_pc, K, _ = pc_latent.shape
        B_, T, H, W, _ = main_block_outputs[0].shape
        assert B_ == B

        # PC encode + temporal align
        pc_feat = self.pc_encoder(pc_latent)
        pc_feat = self._align_temporal(pc_feat, T)
        pc_feat_BT_K_da = rearrange(pc_feat, "b t k d -> (b t) k d")

        # mask 时间对齐
        pc_mask_BT_K: Optional[torch.Tensor] = None
        if pc_mask is not None:
            pc_mask_float = pc_mask.float()
            pc_mask_BK_T = rearrange(pc_mask_float, "b t k -> (b k) 1 t")
            if T_pc != T:
                pc_mask_BK_T = F.interpolate(pc_mask_BK_T, size=T, mode="nearest")
            pc_mask_aligned = rearrange(
                pc_mask_BK_T, "(b k) 1 t -> b t k", b=B, k=K
            ).bool()
            pc_mask_BT_K = rearrange(pc_mask_aligned, "b t k -> (b t) k")

        residuals: List[torch.Tensor] = []
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
