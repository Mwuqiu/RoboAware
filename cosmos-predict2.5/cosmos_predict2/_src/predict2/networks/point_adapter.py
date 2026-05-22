"""PointAdapter A.v3

设计要点（与原版的差异）
────────────────────────
1. PC 通过 Cosmos Block 的 cross-attn (K/V) 注入主干, 不再 mean-pool + spatial broadcast.
   每个 video patch (h, w) 在 K=30 个 PC token 中自己挑 attention 权重.
2. Adapter Block 与 backbone Cosmos Block 同构 (同一个 block_factory). x_dim=d_main,
   context_dim=d_main (PC 已投到 d_main); init/adaLN-zero 复用 backbone 已验证的 scheme.
3. Block 在 (B*T) 维度上 per-frame 工作: 输入 [B*T, 1, H, W, D], cross-attn K/V
   是该 latent t 的 PC token [B*T, K, D]. 满足"PC 与 video 逐帧对齐, 不跨时间 attend".
4. 不再使用 d_a 中间瓶颈 / x_proj / t_proj / before_proj / after_projs.
   唯一的 zero-init 由 Block 自带的 adaLN-zero 提供, 不会堵 PC → adapter 的梯度通路.
5. Frame-level mask: prefix mode 后段 / none mode 等 PC 不可见的 frame, 输出强制为 0,
   完全交给 backbone 通过 temporal layers 自己 propagate.

外部契约
────────
PointAdapter 暴露给 backbone (minimal_v4_dit.py) 的接口完全保持不变:
  - .pc_encoder(pc_latent)            backbone 在 forward 入口直接 call
  - ._align_temporal(pc_feat, T)      backbone 在 forward 入口直接 call
  - .inject_block_ids                 list[int]
  - .apply_stage(adapter_idx=, pc_feat_BT_K_da=, pc_mask_BT_K=,
                 x_main=, t_embedding_B_T_D=, crossattn_emb=)
                                       返回 (pc_feat_next, residual);
                                       residual shape == x_main.shape, 直接 add.
所以 minimal_v4_dit.py 不需要任何改动.

注意
────
- d_a 必须等于 d_main. exp 配置里 point_adapter_d_a=None (= d_main) 已经满足.
- adapter_block_depth 必须为 1.
- 旧 checkpoint 的 PointAdapter 子模块 shape 与新版完全不同, 不能 resume,
  需要从 backbone+text 已有的 base checkpoint 起重新训.
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

    def __init__(self, d_pc: int, d_a: int, use_layernorm: bool = False):
        super().__init__()
        layers = []
        if use_layernorm:
            # V5: normalize PC features before projection. dec_0 outputs have
            # much smaller magnitude than enc_out (norm ~5k vs ~80k), so
            # without LayerNorm the cross-attn K/V would be tiny and get
            # softmax-suppressed by text tokens.
            layers.append(nn.LayerNorm(d_pc))
        layers.extend([
            nn.Linear(d_pc, d_a, bias=True),
            nn.SiLU(),
            nn.Linear(d_a, d_a, bias=True),
        ])
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # V5 fix: previously LN was silently skipped here, causing
                # PCEncoder.LayerNorm to land at γ=1, β=1 (uninit memory after
                # cosmos meta→to_empty materialization). β=1 shifted PC input
                # off-distribution and the model learned to ignore PC entirely.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        return self.mlp(pc)


class PointAdapter(nn.Module):
    """A.v3: 同构 Cosmos Block + cross-attn (video Q ↔ PC K/V) 注入."""

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
        pc_encoder_use_layernorm: bool = False,  # V5: True for dec_0 features
        cross_attn_only: bool = False,  # D4-A: legacy bool, equivalent to adapter_mode="cross_attn_only"
        adapter_mode: str = "full",     # "full" | "cross_attn_only" | "cross_attn_plus_mlp"
        controlnet_copy_from_backbone: bool = False,  # ControlNet-style: after backbone ckpt loaded,
                                                       # copy backbone[inject_id] sublayer weights into adapter[i].
                                                       # Sets adaln_modulation to zero (ControlNet zero-conv style).
                                                       # k_proj/v_proj of cross_attn stay random (modality mismatch).
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
        # Adapter mode (which sublayers of the Cosmos Block to run in apply_stage):
        #   "full"                — D3/D4 baseline: self-attn + cross-attn + MLP all run.
        #                            ❌ self-attn / MLP form a PC-independent downhill path
        #                            so optimizer never learns to use cross-attn properly.
        #   "cross_attn_only"     — D4-A: only run cross-attn sublayer. Forces the only
        #                            PC channel to be the sole residual. ✅ trains content
        #                            sensitivity (zero=0.59 rel_L2) but capacity-limited.
        #   "cross_attn_plus_mlp" — D4-B (FAILED): cross-attn + MLP. MLP eats LN(x_after_ca)
        #                            = LN(x_main + delta_ca). When delta_ca << x_main (which
        #                            it always is at init), MLP effectively eats LN(x_main).
        #                            Verified empirically at iter_500→750: delta_mlp/delta_ca
        #                            grew 82× → 148×, PC sensitivity collapsed 0.066 → 0.008.
        #                            MLP became PC-independent x_main bypass = same failure
        #                            as full Block's self-attn. Kept for reproducibility.
        #   "cross_attn_then_mlp" — D4-B' (NEW): MLP eats LN(delta_ca) only. If delta_ca → 0
        #                            then LN(0) = 0 and MLP outputs constant bias (no spatial
        #                            structure). MLP can only AMPLIFY / DECODE cross-attn
        #                            output, not bypass it. gate_mlp zero-init for gradual
        #                            wake-up; iter 0 behaves like D4-A.
        valid_modes = {"full", "cross_attn_only", "cross_attn_plus_mlp", "cross_attn_then_mlp"}
        if bool(cross_attn_only) and adapter_mode == "full":
            # Backward compat: legacy bool flag mapped to "cross_attn_only".
            adapter_mode = "cross_attn_only"
        if adapter_mode not in valid_modes:
            raise ValueError(f"adapter_mode must be one of {valid_modes}, got {adapter_mode!r}")
        self.adapter_mode = adapter_mode
        # Keep the legacy attribute name for any external probe / ckpt diag code.
        self.cross_attn_only = adapter_mode == "cross_attn_only"
        self.controlnet_copy = bool(controlnet_copy_from_backbone)
        self._controlnet_copy_done = False

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
        self.pc_encoder = PCEncoder(d_pc=d_pc, d_a=d_a, use_layernorm=pc_encoder_use_layernorm)

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
            elif isinstance(m, nn.LayerNorm):
                # V5 fix: LN must be explicitly init'd here too because this is
                # the canonical entry under cosmos meta→to_empty→init_weights().
                # Without this, LN.weight ended up γ=1 (lucky) but LN.bias=β=1
                # (uninit memory), corrupting PCEncoder input distribution and
                # making the model learn to ignore PC content. Bug discovered
                # via PC-zero ablation on V5 iter_3k showing identical output
                # with full vs zero PC. Fix: ones_ for γ, zeros_ for β.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Each adapter Block uses the backbone Cosmos Block's own init_weights
        # (includes adaLN-zero for the modulation last layer, trunc_normal for
        # q/k/v in self/cross attention).
        for block in self.adapter_blocks:
            if hasattr(block, "init_weights"):
                block.init_weights()
            # V5 fix: undo adaLN-zero on adapter blocks. Backbone modulation
            # came from pretrained Cosmos (non-zero, good). Adapter modulation
            # was zero-init by Cosmos Block.reset_parameters() → gates the
            # entire adapter contribution → 5k iter fine-tune not enough to
            # bootstrap modulation from 0. Re-init each modulation Linear with
            # xavier_uniform_ so adapter starts contributing from iter 1.
            #
            # D4-B exception: when adapter_mode == "cross_attn_plus_mlp", the
            # MLP modulation is zero-init so MLP gate = 0 → MLP contributes 0
            # at iter 0, behaving like D4-A initially. MLP wakes up gradually
            # only if its capacity helps lower the loss. This lets us *add*
            # MLP without disturbing the trained-cross-attn equilibrium.
            mlp_zero_init = self.adapter_mode in ("cross_attn_plus_mlp", "cross_attn_then_mlp")
            for mod_name in (
                "adaln_modulation_self_attn",
                "adaln_modulation_cross_attn",
                "adaln_modulation_mlp",
            ):
                mod = getattr(block, mod_name, None)
                if mod is None:
                    continue
                use_zero = mlp_zero_init and mod_name == "adaln_modulation_mlp"
                for layer in mod:
                    if isinstance(layer, nn.Linear):
                        if use_zero:
                            nn.init.zeros_(layer.weight)
                        else:
                            nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def init_weights(self) -> None:
        """Public init hook called by MinimalV4DiT.init_weights() after the
        framework's meta -> cpu / meta -> cuda materialization. See `_init_weights`."""
        self._init_weights()

    @torch.no_grad()
    def controlnet_copy_from_backbone(self, backbone_blocks: nn.ModuleList) -> None:
        """ControlNet-style adapter init: copy modality-agnostic weights from backbone
        blocks at inject points into adapter blocks. Must be called AFTER backbone
        pretrained ckpt is loaded (e.g. from on_train_start).

        Copied (matching shapes verified):
          - self_attn.{q,k,v,output}_proj.weight, self_attn.{q,k}_norm.weight
          - cross_attn.q_proj.weight, cross_attn.output_proj.weight,
            cross_attn.{q,k}_norm.weight
          - mlp.layer{1,2}.weight

        NOT copied (modality / shape mismatch):
          - cross_attn.k_proj.weight, cross_attn.v_proj.weight  (text emb is 1024-d,
            our PC tokens are 2048-d; also semantics differ). Kept at random init.
          - adaln_modulation_{self_attn,cross_attn,mlp}.{1,2}.weight  (backbone uses
            AdaLN-LoRA: 2048→256→6144; adapter uses single 2048→6144). Zero-init
            as ControlNet "zero conv" — adapter contributes 0 at iter 0.

        Idempotent: subsequent calls are no-ops.
        """
        if self._controlnet_copy_done:
            return
        if not self.controlnet_copy:
            return

        # Resume safety: if adapter has been trained (resuming from ckpt), skip
        # the copy so we don't overwrite trained weights with backbone init.
        #
        # We probe adaln_modulation_MLP gate rows (NOT cross_attn). Reason:
        # _init_weights() zero-inits ALL Linears of mlp modulation (only mlp in
        # mlp_zero_init mode), so fresh init has mlp gate rows = 0 exactly.
        # After any training, gradient grows them. Cross/self modulation rows
        # have xavier-init values ~3e-2 even at fresh init (verified 2026-05-22),
        # which made the previous threshold check trigger incorrectly on fresh
        # launch. MLP gate is a clean 0→trained signal.
        try:
            sample_adaln = self.adapter_blocks[0].adaln_modulation_mlp
            lins = [m for m in sample_adaln if isinstance(m, nn.Linear)]
            if len(lins) >= 2:
                layer_B = lins[-1]
                out_dim = layer_B.weight.shape[0]
                D = out_dim // 3
                gate_max = layer_B.weight[2 * D:].detach().abs().max().item()
                if gate_max > 1e-6:
                    print(
                        f"[PointAdapter] ControlNet copy SKIPPED — mlp gate_max={gate_max:.3e} > 1e-6, "
                        f"adapter appears already trained (resume from ckpt). "
                        f"Not overwriting trained weights.",
                        flush=True,
                    )
                    self._controlnet_copy_done = True
                    return
        except Exception as e:
            print(f"[PointAdapter] gate-state check raised {e!r}, proceeding with copy.", flush=True)

        # sublayers with leaf nn.Linear named "<module>.<attr>"
        LINEAR_PATHS = [
            ("self_attn", "q_proj"), ("self_attn", "k_proj"),
            ("self_attn", "v_proj"), ("self_attn", "output_proj"),
            ("cross_attn", "q_proj"), ("cross_attn", "output_proj"),
            ("mlp", "layer1"), ("mlp", "layer2"),
        ]
        # tensor leaves (RMSNorm with just .weight)
        TENSOR_PATHS = [
            ("self_attn", "q_norm"), ("self_attn", "k_norm"),
            ("cross_attn", "q_norm"), ("cross_attn", "k_norm"),
        ]

        copied = 0
        skipped = 0
        for i, blk_id in enumerate(self.inject_block_ids):
            src_blk = backbone_blocks[blk_id]
            dst_blk = self.adapter_blocks[i]

            for mod_name, attr in LINEAR_PATHS:
                src_mod = getattr(src_blk, mod_name, None)
                dst_mod = getattr(dst_blk, mod_name, None)
                if src_mod is None or dst_mod is None:
                    skipped += 1
                    continue
                src_lin = getattr(src_mod, attr, None)
                dst_lin = getattr(dst_mod, attr, None)
                if src_lin is None or dst_lin is None:
                    skipped += 1
                    continue
                if src_lin.weight.shape != dst_lin.weight.shape:
                    skipped += 1
                    continue
                dst_lin.weight.copy_(src_lin.weight)
                if (getattr(src_lin, "bias", None) is not None
                        and getattr(dst_lin, "bias", None) is not None):
                    dst_lin.bias.copy_(src_lin.bias)
                copied += 1

            for mod_name, attr in TENSOR_PATHS:
                src_mod = getattr(src_blk, mod_name, None)
                dst_mod = getattr(dst_blk, mod_name, None)
                if src_mod is None or dst_mod is None:
                    continue
                src_norm = getattr(src_mod, attr, None)
                dst_norm = getattr(dst_mod, attr, None)
                if src_norm is None or dst_norm is None:
                    continue
                if getattr(src_norm, "weight", None) is None or getattr(dst_norm, "weight", None) is None:
                    continue
                if src_norm.weight.shape == dst_norm.weight.shape:
                    dst_norm.weight.copy_(src_norm.weight)
                    copied += 1

            # adaLN modulation: timestep-modulation inheritance + gate-zero.
            # Backbone uses AdaLN-LoRA: nn.Sequential(SiLU, Linear(D, d_lora), Linear(d_lora, 3D)).
            # Adapter now also uses LoRA (use_adaln_lora=True). Both have len([Linear])==2.
            # - Linear A (D, d_lora): copy from backbone — modality-agnostic timestep encoding.
            # - Linear B (d_lora, 3D): output is chunked into [shift, scale, gate]:
            #     shift rows [0   .. D-1]   ← copy from backbone (inherit timestep curve)
            #     scale rows [D   .. 2D-1]  ← copy from backbone (inherit timestep curve)
            #     gate  rows [2D  .. 3D-1]  ← ZERO (ControlNet "zero conv": adapter silent at iter 0)
            # When gate gradient turns gate non-zero, adapter immediately produces
            # well-shaped timestep-aware shift/scale residuals (vs learning t→shift/scale from scratch).
            for adaln_name in (
                "adaln_modulation_self_attn",
                "adaln_modulation_cross_attn",
                "adaln_modulation_mlp",
            ):
                src_mod = getattr(src_blk, adaln_name, None)
                dst_mod = getattr(dst_blk, adaln_name, None)
                if src_mod is None or dst_mod is None:
                    continue
                src_lins = [m for m in src_mod if isinstance(m, nn.Linear)]
                dst_lins = [m for m in dst_mod if isinstance(m, nn.Linear)]
                if len(src_lins) != 2 or len(dst_lins) != 2:
                    # Structural mismatch (e.g. one is LoRA, the other isn't) — fall back
                    # to pure zero-init so adapter stays silent.
                    for layer in dst_mod:
                        if isinstance(layer, nn.Linear):
                            nn.init.zeros_(layer.weight)
                            if layer.bias is not None:
                                nn.init.zeros_(layer.bias)
                    skipped += 1
                    continue
                src_A, dst_A = src_lins[0], dst_lins[0]
                src_B, dst_B = src_lins[1], dst_lins[1]
                # Linear A: full copy (modality-agnostic timestep encoder)
                if src_A.weight.shape == dst_A.weight.shape:
                    dst_A.weight.copy_(src_A.weight)
                    if src_A.bias is not None and dst_A.bias is not None:
                        dst_A.bias.copy_(src_A.bias)
                else:
                    nn.init.zeros_(dst_A.weight)
                    if dst_A.bias is not None:
                        nn.init.zeros_(dst_A.bias)
                # Linear B: build [backbone_shift, backbone_scale, zeros_gate] via
                # torch.cat — avoids any in-place slice-fill (DTensor doesn't
                # support `aten.fill_.Tensor` on shards; that's exactly why the
                # previous `weight[2*D:] = 0` was a silent no-op under FSDP).
                if src_B.weight.shape == dst_B.weight.shape:
                    out_dim = dst_B.weight.shape[0]
                    D = out_dim // 3
                    keep = src_B.weight[: 2 * D]                            # slicing view = OK
                    zero_gate = torch.zeros_like(src_B.weight[2 * D :])     # new tensor = OK
                    new_W = torch.cat([keep, zero_gate], dim=0)             # cat returns full tensor
                    dst_B.weight.copy_(new_W)                                # full-Parameter copy = works
                    if src_B.bias is not None and dst_B.bias is not None:
                        keep_b = src_B.bias[: 2 * D]
                        zero_gate_b = torch.zeros_like(src_B.bias[2 * D :])
                        new_b = torch.cat([keep_b, zero_gate_b], dim=0)
                        dst_B.bias.copy_(new_b)
                else:
                    nn.init.zeros_(dst_B.weight)
                    if dst_B.bias is not None:
                        nn.init.zeros_(dst_B.bias)
                copied += 2  # A + B (gate-zeroed)

        self._controlnet_copy_done = True
        print(
            f"[PointAdapter] ControlNet copy DONE: copied={copied} skipped={skipped} "
            f"blocks={self.inject_block_ids}",
            flush=True,
        )

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
        pc_mask_BT_K: Optional[torch.Tensor],  # [B*T, K] bool, True=有效
        x_main: torch.Tensor,                  # [B, T, H, W, d_main], backbone block 输出
        t_embedding_B_T_D: Optional[torch.Tensor] = None,
        crossattn_emb: Optional[torch.Tensor] = None,  # 传 backbone 的 text emb, 本设计不用
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A.v3 注入: video tokens 作为 query, PC tokens 作为 cross-attn K/V.

        Returns:
            pc_feat_next: [B*T, K, d_main]  (本版本不更新 PC, passthrough; 保留接口)
            residual:     [B, T, H, W, d_main]  直接 add 到 backbone block output
        """
        del crossattn_emb  # adapter 不用 text; PC 自带空间结构

        if t_embedding_B_T_D is None:
            raise ValueError("Cosmos Block adapter requires t_embedding_B_T_D")

        B, T, H, W, D = x_main.shape

        # V5: zero out padded K positions of pc_feat before cross-attn.
        # Without this, V5's LayerNorm in PCEncoder turns pad_value=0 into β
        # (LN bias, may drift during training) → cross-attn sees non-zero K/V at
        # padded positions and injects noise. Zeroing here makes K_proj(0)=K_bias
        # (constant across padded K), so attention weight on padded ≈ uniform-low
        # rather than random noise. Mimics V3's effective behavior with pad_value=0.
        if pc_mask_BT_K is not None:
            pc_feat_BT_K_da = pc_feat_BT_K_da * pc_mask_BT_K.unsqueeze(-1).to(pc_feat_BT_K_da.dtype)

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

        # ── 调用同构 Cosmos Block, cross-attn K/V = 当前 latent t 的 PC token ──
        block = self.adapter_blocks[adapter_idx]

        if self.adapter_mode in ("cross_attn_only", "cross_attn_plus_mlp", "cross_attn_then_mlp"):
            # D4-A / D4-B: bypass self-attn inside Block. Run only:
            #   D4-A: cross-attn path (LN + adaln + cross-attn + gate)
            #   D4-B: cross-attn path + MLP path (LN + adaln + MLP + gate, zero-init MLP gate)
            # Self-attn (PC-independent token mixing) is skipped → closes the
            # "self-attn refinement of x_main" bypass that lets adapter lower
            # loss without learning PC content.
            #
            # Reference (Block.forward, minimal_v4_dit.py:1338-1444):
            #   x ← x + gate_self  * self_attn(LN(x))           ← SKIPPED in D4-A/B
            #   x ← x + gate_cross * cross_attn(LN(x), pc_feat) ← RUN both
            #   x ← x + gate_mlp   * mlp(LN(x))                 ← D4-B only (zero-init gate)
            #
            # Dtype: adapter is cast to bf16 (text2world_model_rectified_flow.py:258),
            # but emb / pc_feat / x may arrive in fp32 (e.g. t_embedder under
            # autocast(enabled=False)). Cast inputs to adapter weight dtype to
            # avoid mat1/mat2 dtype mismatch in the Linear ops.
            ada_dtype = block.adaln_modulation_cross_attn[1].weight.dtype
            emb_for_mod = emb_BT_1_D.to(ada_dtype)
            x_for_mod = x_BT_1_H_W_D.to(ada_dtype)
            pc_for_attn = pc_feat_BT_K_da.to(ada_dtype)

            # ── Cross-attn portion (always on for D4-A/B) ────────────────────
            mod_ca = block.adaln_modulation_cross_attn(emb_for_mod)
            shift_ca_BT_1_D, scale_ca_BT_1_D, gate_ca_BT_1_D = mod_ca.chunk(3, dim=-1)
            shift_ca = rearrange(shift_ca_BT_1_D, "bt one d -> bt one 1 1 d").type_as(x_for_mod)
            scale_ca = rearrange(scale_ca_BT_1_D, "bt one d -> bt one 1 1 d").type_as(x_for_mod)
            gate_ca = rearrange(gate_ca_BT_1_D, "bt one d -> bt one 1 1 d").type_as(x_for_mod)
            # D4-C gate-offset REMOVED (2026-05-22 audit). Original premise
            # ("bf16 quantization makes per-step Adam update below quantum →
            # cross_attn weights stuck at init") was disproved by weight_diff —
            # actual drift is 0.3-1.5% over 750 iter, slow but not stuck.
            # Adding +1.0 also violated ControlNet "zero conv" principle: with
            # gate_ca + 1.0 at iter 0, adapter contributes random ca_out from
            # step 0 (random because k_proj/v_proj are random init for PC).
            # With sliced-copy bug fixed (gate rows now properly zero-initialized),
            # gate_ca starts at 0 naturally → adapter silent at iter 0 → standard
            # ControlNet behavior. Gradient still flows to gate rows via chain
            # rule even when gate=0 (delta_ca = ca_out * gate, ∂loss/∂gate is
            # non-zero as long as ∂loss/∂delta_ca is non-zero — which it is
            # because backbone has many other paths contributing to loss).
            # If training stalls or PC swap effect doesn't emerge with this fix,
            # the original line was:
            #     gate_ca = gate_ca + 1.0

            normed_ca = block.layer_norm_cross_attn(x_for_mod) * (1 + scale_ca) + shift_ca
            ca_out_BT_HW_D = block.cross_attn(
                rearrange(normed_ca, "bt one h w d -> bt (one h w) d"),
                pc_for_attn,
                rope_emb=None,
            )
            ca_out_BT_1_H_W_D = rearrange(
                ca_out_BT_HW_D, "bt (one h w) d -> bt one h w d", one=1, h=H, w=W
            )
            delta_ca = ca_out_BT_1_H_W_D * gate_ca

            if self.adapter_mode in ("cross_attn_plus_mlp", "cross_attn_then_mlp"):
                # ── MLP portion ──────────────────────────────────────────────
                # cross_attn_plus_mlp (D4-B): MLP eats x + delta_ca = mostly x_main.
                #   Empirically lets MLP bypass to x_main and ignore PC.
                # cross_attn_then_mlp (D4-B'): MLP eats delta_ca only. If delta_ca=0
                #   then LN(0)=0, MLP outputs only constant bias (no spatial freedom).
                #   MLP can only amplify cross-attn output; cannot bypass.
                # Both zero-init gate_mlp for gradual wake-up.
                if self.adapter_mode == "cross_attn_plus_mlp":
                    mlp_input = x_for_mod + delta_ca         # ❌ x_main bypass risk
                else:
                    mlp_input = delta_ca                     # ✅ pure PC-derived

                mod_mlp = block.adaln_modulation_mlp(emb_for_mod)
                shift_mlp_BT_1_D, scale_mlp_BT_1_D, gate_mlp_BT_1_D = mod_mlp.chunk(3, dim=-1)
                shift_mlp = rearrange(shift_mlp_BT_1_D, "bt one d -> bt one 1 1 d").type_as(x_for_mod)
                scale_mlp = rearrange(scale_mlp_BT_1_D, "bt one d -> bt one 1 1 d").type_as(x_for_mod)
                gate_mlp = rearrange(gate_mlp_BT_1_D, "bt one d -> bt one 1 1 d").type_as(x_for_mod)

                normed_mlp = block.layer_norm_mlp(mlp_input) * (1 + scale_mlp) + shift_mlp
                mlp_out = block.mlp(normed_mlp)
                delta_mlp = mlp_out * gate_mlp

                delta_BT_1_H_W_D = (delta_ca + delta_mlp).to(x_BT_1_H_W_D.dtype)
            else:
                # D4-A: cross-attn only
                delta_BT_1_H_W_D = delta_ca.to(x_BT_1_H_W_D.dtype)
        else:
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

        # ── frame-level mask: PC 不可见的帧强制 0 (prefix 后段 / none mode) ──
        # cross-attn 内部 K/V padding 全 False 时 softmax 可能数值不稳, 这里用乘法显式截断.
        if pc_mask_BT_K is not None:
            frame_visible_BT = pc_mask_BT_K.any(dim=-1).to(delta_B_T_H_W_D.dtype)  # [B*T]
            frame_visible_B_T = rearrange(frame_visible_BT, "(b t) -> b t", b=B, t=T)
            delta_B_T_H_W_D = delta_B_T_H_W_D * frame_visible_B_T[:, :, None, None, None]

        # PC tokens 在各 stage 之间不更新, 直接 passthrough.
        # (与原版"PC 通过 adapter chain 串行更新"不同; PC 视作固定 prior.)
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
