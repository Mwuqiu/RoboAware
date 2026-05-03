import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import math
import wandb
import torch
import torch.nn as nn
import numpy as np
import glob
import json
from typing import Sequence, Any, Dict
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

from pf_encoder import load_ptv3_model, PTV3Encoder
from pf_dataset import TemporalPointDataset
from pf_dit import PointCloudDiT, DiffusionSchedule, masked_mse, q_sample, v_target, x0_from_v
from pf_decoder import LatentToPointDecoder, denormalize_sequence_xyz, normalize_sequence_xyz  # type: ignore
from pf_losses import fk_warp_consistency_loss

from pf_latent_cache import (
    CachedLatentDataset,
    EncodedLatentDataset,
    MixedCachedLatentSupervisionDataset,
    inspect_cache,
    preencode_latents,
    preencode_only,
)

class TrainDitCfg:
    def to_dict(self) -> Dict[str, Any]:
        return {
            "DATASET": "so100",
            "CONFIG": "semseg-pt-v3m1-0-base",
            "EXP_NAME": "semseg-pt-v3m1-0-base-only-grid",
            "WEIGHT_NAME": "model_last",

            # training
            "num_steps": 1000,
            "batch_size": 4,
            "epochs": 10,
            "log_every": 10,
            "lr": 1e-4,
            "min_lr": 1e-5,
            "weight_decay": 1e-2,

            # model
            "k_tokens": 30,
            "dit": {
                "hidden_size": 512,
                "depth": 12,
                "num_heads": 8,
            },

            # dataset
            "data": {
                "data_root": "pointflow/generated_pointclouds_dataset",
                "split": "training",
                "window_size": 25,
                "stride": 1,
                "precompute_index": True,
            },

            # latent cache
            "cache": {
                "use_cache": True,
                "cache_overwrite": False,
                "cache_dir": "exp/so100/semseg-pt-v3m1-0-base-only-grid/latent_cache/T25_K30",
            },

            # FK loss
            "fk": {
                "lambda_fk": 100.0,
                "huber_delta": 0.001,
            },

            # decoder (for FK loss)
            "decoder": {
                "ckpt_path": "exp/so100/semseg-pt-v3m1-0-base-only-grid/decoder_ckpt/decoder_mse_T25_K30.pt",
                "strict": True,
                "num_points": 2000,
            },

            # checkpointing
            "ckpt": {
                "ckpt_dir": None,
                "save_every_epochs": 1,
            },

            # wandb
            "wandb": {
                "enabled": True,
                "project": "pointflow-dit",
                "name": None,
            },
        }


if __name__ == "__main__":
    # ---- load config (internal defaults + env overrides) ----
    cfg = TrainDitCfg().to_dict()

    DATASET = str(cfg.get("DATASET", None))
    CONFIG = str(cfg.get("CONFIG", None))
    EXP_NAME = str(cfg.get("EXP_NAME", None))
    WEIGHT_NAME = str(cfg.get("WEIGHT_NAME", None))

    CONFIG_FILE = os.path.join("configs", DATASET, f"{CONFIG}.py")
    EXP_DIR = os.path.join("exp", DATASET, EXP_NAME)
    WEIGHT_PATH = os.path.join(EXP_DIR, "model", f"{WEIGHT_NAME}.pth")

    # ---- hyperparams ----
    num_steps = int(cfg["num_steps"])
    batch_size = int(cfg["batch_size"])
    k_tokens = int(cfg["k_tokens"])

    epochs = int(cfg["epochs"])
    log_every = int(cfg["log_every"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])

    # kin loss weights
    lambda_fk = float(cfg.get("fk", {}).get("lambda_fk", 0.0))
    huber_delta = float(cfg.get("fk", {}).get("huber_delta", 0.01))

    # decoder config (for FK loss)
    decoder_cfg = dict(cfg.get("decoder", {}))
    decoder_num_points = int(decoder_cfg.get("num_points", 2000))
    decoder_ckpt_path = decoder_cfg.get("ckpt_path", None)
    decoder_strict = bool(decoder_cfg.get("strict", True))

    # checkpoint config
    ckpt_cfg = dict(cfg.get("ckpt", {}))
    ckpt_dir = ckpt_cfg.get("ckpt_dir", None) or os.path.join(EXP_DIR, "dit_ckpt")
    save_every_epochs = int(ckpt_cfg.get("save_every_epochs", 1))

    # wandb config
    wb_cfg = dict(cfg.get("wandb", {}))
    wandb_project = wb_cfg.get("project", "pointflow-dit")
    wandb_run_name = wb_cfg.get("name", None)
    wandb_enabled = bool(wb_cfg.get("enabled", True))

    # Minimum LR floor
    min_lr = float(cfg.get("min_lr", 1e-5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = dict(cfg.get("data", {}))
    ds = TemporalPointDataset(
        split=str(data_cfg.get("split", "training")),
        data_root=str(data_cfg.get("data_root", "pointflow/generated_pointclouds_dataset")),
        window_size=int(data_cfg.get("window_size", 25)),
        stride=int(data_cfg.get("stride", 1)),
        precompute_index=bool(data_cfg.get("precompute_index", True)),
    )

    cache_cfg = dict(cfg.get("cache", {}))
    use_cache = bool(cache_cfg.get("use_cache", True))
    cache_overwrite = bool(cache_cfg.get("cache_overwrite", False))
    cache_dir = cache_cfg.get("cache_dir", None)

    encoder = None
    if (not use_cache) or cache_overwrite or (not os.path.exists(os.path.join(cache_dir, "meta.json"))):
        encoder = PTV3Encoder(load_ptv3_model()).to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

    if use_cache:
        if (not os.path.exists(os.path.join(cache_dir, "meta.json"))) or cache_overwrite:
            print(f"[cache] cache missing or overwrite requested, preencoding to: {cache_dir}")
            assert encoder is not None
            preencode_latents(ds, encoder, cache_dir, k=k_tokens, sample="first", pad_value=0.0, overwrite=True)

        cached_latent_ds = CachedLatentDataset(cache_dir)
        latent_ds = MixedCachedLatentSupervisionDataset(
            latent_ds=cached_latent_ds,
            base_ds=ds,
            supervision_keys=("segment", "body_xpos", "body_xmat", "q"),
            strict=False,
        )
        print(f"Loaded cached latents + supervision mix from {cache_dir}")
    else:
        assert encoder is not None
        latent_ds = EncodedLatentDataset(ds, encoder, k=k_tokens, sample="first", pad_value=0.0)
        print("Using on-the-fly encoding (cache disabled)")
        preencode_latents(ds, encoder, cache_dir, k=k_tokens, sample="first", pad_value=0.0, overwrite=cache_overwrite)

    loader = DataLoader(
        latent_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    # infer T,L,C from one batch
    batch0 = next(iter(loader))
    x0 = batch0["x0"].to(device)  # [B,T,L,C]
    mask = batch0["mask"].to(device)
    B, T, L, C = x0.shape

    # ---- diffusion schedule + DiT ----
    schedule = DiffusionSchedule(num_steps=num_steps, schedule="linear", device=device).to(device)

    dit_cfg = dict(cfg.get("dit", {}))

    # Build DiT denoiser (predict v)
    dit = PointCloudDiT(
        in_channels=int(C),
        hidden_size=int(dit_cfg.get("hidden_size", 512)),
        depth=int(dit_cfg.get("depth", 12)),
        num_heads=int(dit_cfg.get("num_heads", 8)),
        num_frames=int(T),
    ).to(device)

    opt = torch.optim.AdamW(dit.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    print(f"[train] dataset_len={len(ds)} batches_per_epoch={len(loader)} T={T} L={L} C={C} device={device}")

    use_fk_loss = (
        LatentToPointDecoder is not None
        and ("segment" in batch0)
        and ("body_xpos" in batch0)
        and ("body_xmat" in batch0)
        and (lambda_fk > 0)
    )

    decoder = None
    if use_fk_loss:
        decoder = LatentToPointDecoder(in_channels=C, num_points=decoder_num_points).to(device)

        if decoder_ckpt_path and os.path.exists(decoder_ckpt_path):
            ckpt = torch.load(decoder_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "model" in ckpt and isinstance(ckpt["model"], dict):
                    state = ckpt["model"]
                elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                    state = ckpt["state_dict"]
                else:
                    state = ckpt
            else:
                state = ckpt

            missing, unexpected = decoder.load_state_dict(state, strict=decoder_strict)
            if (not decoder_strict) and (missing or unexpected):
                print(f"[decoder] loaded with strict=False. missing={len(missing)} unexpected={len(unexpected)}")
            print(f"[decoder] weights loaded: {decoder_ckpt_path}")
        else:
            print(f"[decoder] WARNING: ckpt not found, FK loss may be meaningless: {decoder_ckpt_path}")

        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad_(False)

    use_wandb = (
        wandb is not None
        and bool(wb_cfg.get("enabled", True))
        and os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes")
    )
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "num_steps": int(num_steps),
                "batch_size": int(batch_size),
                "k_tokens": int(k_tokens),
                "epochs": int(epochs),
                "log_every": int(log_every),
                "lr": float(lr),
                "min_lr": float(min_lr),
                "weight_decay": float(weight_decay),
                "use_cache": bool(use_cache),
                "cache_dir": str(cache_dir),
                "device": str(device),
                "shape_T": int(T),
                "shape_L": int(L),
                "shape_C": int(C),
                "lambda_fk": float(lambda_fk),
                "fk_huber_delta": float(huber_delta),
                "use_fk_loss": bool(use_fk_loss),
                "decoder_ckpt": str(decoder_ckpt_path),
                "decoder_strict": bool(decoder_strict),
                "log_fk_contrib": True,
                "log_fk_ratio": True,
            },
        )

    global_step = 0
    empty_batch_steps = 0
    dit.train()
    for epoch in range(0, epochs):
        for batch in loader:
            x0 = batch["x0"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # skip batches with no valid tokens
            with torch.no_grad():
                mask_sum_now = int(mask.sum().item())
            if mask_sum_now == 0:
                empty_batch_steps += 1
                if global_step % log_every == 0:
                    print(
                        f"epoch={epoch} step={global_step} lr={opt.param_groups[0]['lr']:.2e} "
                        f"SKIP(empty mask) empty_batch_steps={empty_batch_steps}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/skip_empty_batch": 1,
                                "train/empty_batch_steps": int(empty_batch_steps),
                                "train/epoch": int(epoch),
                            },
                            step=int(global_step),
                        )
                global_step += 1
                continue

            # sample t and noise
            t = torch.randint(0, num_steps, (x0.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)

            xt = q_sample(x0, t, schedule, noise)
            v = v_target(x0, t, schedule, noise)

            v_pred = dit(xt, t, mask=mask)
            loss_diff = masked_mse(v_pred, v, mask)

            loss_fk = x0.new_tensor(0.0)
            if use_fk_loss and decoder is not None:
                # reconstruct x0_pred -> decode to world xyz -> apply FK warp loss
                x0_pred = x0_from_v(xt, t, schedule, v_pred)

                with torch.no_grad():
                    # cached segment/body poses are supervision only
                    segment = batch["segment"].to(device, non_blocking=True)
                    body_xpos = batch["body_xpos"].to(device, non_blocking=True)
                    body_xmat = batch["body_xmat"].to(device, non_blocking=True)

                # decode predicted latent tokens into point clouds
                xyz_pred = decoder(x0_pred, mask=mask)  # [B,T,2000,3]

                loss_fk = fk_warp_consistency_loss(
                    xyz_pred=xyz_pred,
                    segment=segment,
                    body_xpos=body_xpos,
                    body_xmat=body_xmat,
                    robust_delta=huber_delta,
                )

            loss = loss_diff + (lambda_fk * loss_fk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if global_step % log_every == 0:
                cur_lr = opt.param_groups[0]["lr"]

                # diagnostics (masked)
                with torch.no_grad():
                    mask_ratio = mask.float().mean().item()
                    mask_sum = int(mask.sum().item())
                    mask_ratio_per_t = mask.float().mean(dim=(0, 2)).detach().cpu().tolist()

                    loss_zero = masked_mse(torch.zeros_like(v), v, mask).item()

                    denom = int(mask.sum().item())
                    if denom > 0:
                        v_flat = v.float().reshape(-1, v.shape[-1])
                        x0_flat = x0.float().reshape(-1, x0.shape[-1])
                        mask_flat = mask.reshape(-1)
                        v_valid = v_flat[mask_flat]
                        x0_valid = x0_flat[mask_flat]
                        v_mse = float((v_valid ** 2).mean().item())
                        v_std = float(v_valid.std(unbiased=False).item())
                        x0_mse = float((x0_valid ** 2).mean().item())
                        x0_std = float(x0_valid.std(unbiased=False).item())
                    else:
                        v_mse = float("nan")
                        v_std = float("nan")
                        x0_mse = float("nan")
                        x0_std = float("nan")

                    loss_now = float(loss.item())
                    improve_vs_zero = (1.0 - loss_now / max(loss_zero, 1e-12)) if math.isfinite(loss_zero) else float("nan")
                    loss_fk_now = float(loss_fk.item()) if isinstance(loss_fk, torch.Tensor) else float(loss_fk)
                    loss_diff_now = float(loss_diff.item())

                    fk_contrib = float(lambda_fk) * float(loss_fk_now)
                    fk_ratio = (fk_contrib / max(loss_now, 1e-12)) if math.isfinite(loss_now) else float("nan")

                print(
                    f"epoch={epoch} step={global_step} lr={cur_lr:.2e} "
                    f"loss={loss_now:.3e} diff={loss_diff_now:.3e} fk={loss_fk_now:.3e} "
                    f"fk_contrib={fk_contrib:.3e} fk_ratio={fk_ratio:.3f} "
                    f"loss0={loss_zero:.3e} imp={improve_vs_zero:.3f} "
                    f"v_mse={v_mse:.3e} v_std={v_std:.3e} x0_mse={x0_mse:.3e} x0_std={x0_std:.3e} "
                    f"mask_ratio={mask_ratio:.4f} mask_sum={mask_sum}"
                )

                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": float(loss_now),
                            "train/loss_diff": float(loss_diff_now),
                            "train/loss_fk": float(loss_fk_now),
                            "train/fk_contrib": float(fk_contrib),
                            "train/fk_ratio": float(fk_ratio),
                            "train/loss_zero": float(loss_zero),
                            "train/improve_vs_zero": float(improve_vs_zero),
                            "train/v_mse": float(v_mse),
                            "train/v_std": float(v_std),
                            "train/x0_mse": float(x0_mse),
                            "train/x0_std": float(x0_std),
                            "train/lr": float(cur_lr),
                            "train/epoch": int(epoch),
                            "train/mask_ratio": float(mask_ratio),
                            "train/mask_sum": int(mask_sum),
                            "train/mask_ratio_per_t": mask_ratio_per_t,
                            "train/skip_empty_batch": 0,
                            "train/empty_batch_steps": int(empty_batch_steps),
                        },
                        step=int(global_step),
                    )

            global_step += 1

        # --- save checkpoint at epoch end ---
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            "model": dit.state_dict(),
            "opt": opt.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "k_tokens": int(k_tokens),
            "num_steps": int(num_steps),
            "shape_T": int(T),
            "shape_L": int(L),
            "shape_C": int(C),
            "config": {
                "hidden_size": int(dit_cfg.get("hidden_size", 512)),
                "depth": int(dit_cfg.get("depth", 12)),
                "num_heads": int(dit_cfg.get("num_heads", 8)),
                "num_frames": int(T),
            },
        }
        last_path = os.path.join(ckpt_dir, "dit_last.pt")
        torch.save(ckpt, last_path)
        print(f"[ckpt] saved: {last_path}")

        if save_every_epochs > 0 and ((epoch + 1) % save_every_epochs == 0):
            snap_path = os.path.join(ckpt_dir, f"dit_epoch{epoch+1:03d}.pt")
            torch.save(ckpt, snap_path)
            print(f"[ckpt] saved: {snap_path}")

        if use_wandb:
            wandb.log(
                {
                    "train/epoch_end": int(epoch + 1),
                    "train/empty_batch_steps": int(empty_batch_steps),
                },
                step=int(global_step),
            )

    if use_wandb:
        wandb.finish()
