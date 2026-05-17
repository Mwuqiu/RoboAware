# SPDX-License-Identifier: Apache-2.0
"""PointAdapter health monitor — logs V4 design signals to wandb each every_n steps.

This callback is FSDP-correct: parameter norms are computed by all-reduce SUM
of squared values across the param shards, so the reported number is the norm
of the FULL (un-sharded) parameter, not just rank 0's slice.

Tracked metrics
───────────────
adapter/null_tokens/{norm, std, abs_mean, grad_norm}
    V4 null PC tokens — primary "is V4 working?" signal.
    * norm should grow from ~0.02·sqrt(60K) ≈ 4.9 at init.
    * std should grow as the model learns to encode useful priors.
    * grad_norm > 0 confirms gradients are actually reaching this parameter.

adapter/pc_encoder/mlp_{0,2}/{weight_norm, grad_norm}
    PCEncoder D_pc → d_main MLP weights. V2 had this stuck at zero (dead path);
    V3/V4 should show monotonically growing weight_norm.

adapter/block_{0..N-1}/cross_attn/{q_proj,k_proj,v_proj,output_proj}_norm
    Per-block cross-attn projection magnitudes.

adapter/block_{0..N-1}/cross_attn_adaln_gate_norm
    The adaLN-zero "gate" — last Linear of adaln_modulation_cross_attn. Zero-init
    at start; growth measures how much the adapter is allowed to influence the
    backbone. Persistently small ≈ adapter contribution is suppressed.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb

from cosmos_predict2._src.imaginaire.callbacks.every_n import EveryN
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.trainer import ImaginaireTrainer
from cosmos_predict2._src.imaginaire.utils import distributed, log


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Get the local shard of a (possibly DTensor / FSDP-sharded) tensor as a plain tensor.

    Necessary because under fully_shard FSDP each param is a DTensor; dispatching
    ops on it goes through DTensor.__torch_dispatch__, which then can't handle
    `dist.all_reduce` (no DeviceMesh). Converting to local first avoids that path.
    """
    if hasattr(t, "to_local"):
        return t.to_local()
    return t


def _full_sq_sum(param: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Squared sum of a (possibly FSDP-sharded) param, all-reduced to global value.

    Returns None if param is None. Returns the squared sum on the local device,
    so caller can sqrt() or combine further.
    """
    if param is None:
        return None
    local = _to_local(param.detach()).float()
    local_sq = local.pow(2).sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    return local_sq


def _full_norm(param: Optional[torch.Tensor]) -> float:
    sq = _full_sq_sum(param)
    if sq is None:
        return 0.0
    return float(sq.sqrt().item())


def _full_std(param: Optional[torch.Tensor]) -> float:
    """Population std of a (possibly FSDP-sharded) param via all-reduce."""
    if param is None:
        return 0.0
    p = _to_local(param.detach()).float()
    local_sum = p.sum()
    local_sq = p.pow(2).sum()
    local_n = torch.tensor(float(p.numel()), device=p.device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
    n = float(local_n.item())
    if n < 2:
        return 0.0
    mean = float(local_sum.item()) / n
    var = float(local_sq.item()) / n - mean * mean
    return float(max(var, 0.0) ** 0.5)


def _full_abs_mean(param: Optional[torch.Tensor]) -> float:
    if param is None:
        return 0.0
    p = _to_local(param.detach()).float()
    local_sum = p.abs().sum()
    local_n = torch.tensor(float(p.numel()), device=p.device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
    n = float(local_n.item())
    if n < 1:
        return 0.0
    return float(local_sum.item()) / n


def _get_point_adapter(model: ImaginaireModel):
    net = getattr(model, "net", None) or getattr(model, "model", None)
    if net is None:
        return None
    return getattr(net, "point_adapter", None)


class PointAdapterMonitor(EveryN):
    """Log V4 PointAdapter weight/grad health stats to wandb."""

    def __init__(self, *args, save_s3: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # save_s3 accepted for config-uniformity (other callbacks support it);
        # we don't write to s3 here — wandb is enough for these scalars.
        del save_s3

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch,
        output_batch,
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        adapter = _get_point_adapter(model)
        if adapter is None:
            return

        metrics: dict[str, float] = {}

        # ── null_pc_tokens (V4 critical) ────────────────────────────────────
        if hasattr(adapter, "null_pc_tokens"):
            p = adapter.null_pc_tokens
            metrics["adapter/null_tokens/norm"] = _full_norm(p)
            metrics["adapter/null_tokens/std"] = _full_std(p)
            metrics["adapter/null_tokens/abs_mean"] = _full_abs_mean(p)
            metrics["adapter/null_tokens/grad_norm"] = _full_norm(getattr(p, "grad", None))

        # ── PCEncoder MLP (D_pc -> d_main projection) ──────────────────────
        for i, m in enumerate(adapter.pc_encoder.mlp):
            if isinstance(m, nn.Linear):
                metrics[f"adapter/pc_encoder/mlp_{i}/weight_norm"] = _full_norm(m.weight)
                metrics[f"adapter/pc_encoder/mlp_{i}/grad_norm"] = _full_norm(
                    getattr(m.weight, "grad", None)
                )

        # ── adapter blocks ─────────────────────────────────────────────────
        for bidx, block in enumerate(adapter.adapter_blocks):
            cross_attn = getattr(block, "cross_attn", None)
            if cross_attn is not None:
                for pname in ("q_proj", "k_proj", "v_proj", "output_proj"):
                    proj = getattr(cross_attn, pname, None)
                    if proj is not None and hasattr(proj, "weight"):
                        metrics[f"adapter/block_{bidx}/cross_attn/{pname}_norm"] = _full_norm(
                            proj.weight
                        )
            # adaLN gate — the LAST Linear of adaln_modulation_cross_attn (zero-init
            # at start; growth measures adapter influence on backbone)
            adaln_mod = getattr(block, "adaln_modulation_cross_attn", None)
            if adaln_mod is not None and hasattr(adaln_mod, "__getitem__"):
                last = adaln_mod[-1]
                if isinstance(last, nn.Linear):
                    metrics[f"adapter/block_{bidx}/cross_attn_adaln_gate_norm"] = _full_norm(
                        last.weight
                    )

        # Write to wandb on rank 0 (all-reduces already happened on all ranks)
        rank = distributed.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0 and wandb.run:
            wandb.log(metrics, step=iteration)
            log.info(
                f"[PointAdapterMonitor iter={iteration}] "
                f"null_tokens.norm={metrics.get('adapter/null_tokens/norm', 0):.4f} "
                f"std={metrics.get('adapter/null_tokens/std', 0):.4f} "
                f"grad={metrics.get('adapter/null_tokens/grad_norm', 0):.4f} "
                f"| pc_enc.mlp_0.norm={metrics.get('adapter/pc_encoder/mlp_0/weight_norm', 0):.4f}"
            )
