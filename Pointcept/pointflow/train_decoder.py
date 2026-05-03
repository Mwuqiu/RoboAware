import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pf_dataset import TemporalPointDataset
from pf_latent_cache import CachedLatentDataset
from pf_decoder import LatentToPointDecoder, normalize_sequence_xyz
from pf_losses import fk_warp_consistency_loss, rigid_pairwise_distance_consistency_loss

DATASET = "so100"
CONFIG = "semseg-pt-v3m1-0-base"
EXP_NAME = "semseg-pt-v3m1-0-base-only-grid"
WEIGHT_NAME = "model_last"
EXP_DIR = os.path.join("exp", DATASET, EXP_NAME)
cache_dir = os.path.join(EXP_DIR, "latent_cache", f"T{25}_K{30}")

class LatentWithCoordDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds: TemporalPointDataset, latent_ds: CachedLatentDataset):
        self.base_ds = base_ds
        self.latent_ds = latent_ds
        if len(self.base_ds) < len(self.latent_ds):
            raise ValueError(f"base_ds({len(self.base_ds)}) shorter than latent_ds({len(self.latent_ds)})")

    def __len__(self):
        return len(self.latent_ds)

    def __getitem__(self, idx):
        lat = self.latent_ds[idx]  # {x0, mask}
        data = self.base_ds[idx]
        np = __import__("numpy")

        frames = [np.asarray(fr, dtype=np.float32) for fr in data["coord"]]
        coords_np = np.stack(frames, axis=0)  # [T,N,3]
        coords = torch.from_numpy(coords_np)

        # FK supervision (assumed aligned with coord)
        segment = data.get("segment", None)
        body_xpos = data.get("body_xpos", None)
        body_xmat = data.get("body_xmat", None)

        out = {"x0": lat["x0"], "mask": lat["mask"], "coord": coords}
        if segment is not None:
            out["segment"] = torch.as_tensor(segment, dtype=torch.long)
        if body_xpos is not None:
            out["body_xpos"] = torch.as_tensor(body_xpos, dtype=torch.float32)
        if body_xmat is not None:
            out["body_xmat"] = torch.as_tensor(body_xmat, dtype=torch.float32)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="pointflow/generated_pointclouds_dataset")
    ap.add_argument("--split", type=str, default="training")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num_points", type=int, default=2000)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "max_norm", "std"],
        help="Optional per-sequence normalization for coords; set to none to keep original coord scale.",
    )
    # --- Loss-only normalization (scheme A) ---
    ap.add_argument(
        "--loss_norm",
        type=str,
        default="none",
        choices=["none", "coord_std", "coord_absmax"],
        help=(
            "Normalize losses by a scale computed from coord (does NOT change coord/pred). "
            "All of rec/FK/rigid losses will be divided by (scale^2 + eps)."
        ),
    )
    ap.add_argument(
        "--loss_norm_eps",
        type=float,
        default=1e-12,
        help="Epsilon used in loss-only normalization: loss /= (scale^2 + eps).",
    )

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_path", type=str, default="")
    ap.add_argument("--num_points_target", type=int, default=2000, help="Resample each frame to this many points for reconstruction.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="pointflow-decoder")
    ap.add_argument("--wandb_name", type=str, default="")
    ap.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "chamfer_proxy"],
        help="Loss for decoder training. Use mse if point order is consistent; use chamfer_proxy otherwise.",
    )
    ap.add_argument(
        "--eval_shuffle",
        action="store_true",
        help="Evaluate a shuffled-latent baseline (should be much worse if decoder depends on z).",
    )
    ap.add_argument("--eval_every", type=int, default=200, help="Evaluation frequency in steps.")

    # FK loss for decoder pretraining
    ap.add_argument("--lambda_fk", type=float, default=0.0, help="Weight of FK warp consistency loss during decoder pretraining.")
    ap.add_argument("--fk_huber_delta", type=float, default=0.01, help="Huber delta used in fk_warp_consistency_loss.")

    # Rigid pairwise-distance consistency loss (no pose needed)
    ap.add_argument(
        "--lambda_rigid",
        type=float,
        default=0.0,
        help="Weight of rigid pairwise-distance consistency loss (within each body) during decoder pretraining.",
    )
    ap.add_argument(
        "--rigid_pairs",
        type=int,
        default=1024,
        help="Number of point pairs sampled per body for rigid consistency loss.",
    )
    ap.add_argument(
        "--rigid_delta",
        type=float,
        default=0.01,
        help="Huber delta (in meters) for rigid pairwise-distance consistency loss.",
    )

    args = ap.parse_args()

    # default save_path if not provided
    if not args.save_path:
        args.save_path = os.path.join(EXP_DIR, "decoder_ckpt", f"decoder_{args.loss_type}_T{args.window}_K30.pt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb was set")
        wandb.init(project=str(args.wandb_project), name=(str(args.wandb_name) if args.wandb_name else None), config=vars(args))

    base_ds = TemporalPointDataset(
        split=args.split,
        data_root=args.data_root,
        window_size=args.window,
        stride=args.stride,
        precompute_index=True,
    )
    latent_ds = CachedLatentDataset(cache_dir)
    ds = LatentWithCoordDataset(base_ds, latent_ds)

    loader = DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    b0 = next(iter(loader))
    x0 = b0["x0"]
    B, T, K, C = x0.shape

    dec = LatentToPointDecoder(
        in_channels=C,
        num_points=int(args.num_points_target),
        d_model=int(args.d_model),
        depth=int(args.depth),
        num_heads=int(args.heads),
    ).to(device)

    opt = torch.optim.AdamW(dec.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    # NOTE: naive Chamfer via torch.cdist is O(N^2) and heavy for N=2000.
    # We train with a proxy (randomly subsample points) to keep it runnable.
    subsample = min(512, int(args.num_points_target))

    def chamfer_proxy(x, y):
        # x,y: [B,N,3]
        if x.shape[1] > subsample:
            ix = torch.randperm(x.shape[1], device=x.device)[:subsample]
            x = x.index_select(1, ix)
        if y.shape[1] > subsample:
            iy = torch.randperm(y.shape[1], device=y.device)[:subsample]
            y = y.index_select(1, iy)
        dist2 = torch.cdist(x, y, p=2) ** 2
        return dist2.min(dim=2).values.mean() + dist2.min(dim=1).values.mean()

    def mse_loss(x, y):
        return ((x - y) ** 2).mean()

    def compute_loss_norm_scale(coord_bt_n3: torch.Tensor) -> torch.Tensor:
        """Return a scalar scale used for loss-only normalization (scheme A)."""
        if args.loss_norm == "coord_std":
            # global std over batch/time/points/xyz
            return coord_bt_n3.std().clamp(min=0.0)
        if args.loss_norm == "coord_absmax":
            return coord_bt_n3.abs().max().clamp(min=0.0)
        # none
        return coord_bt_n3.new_tensor(1.0)

    global_step = 0
    dec.train()
    for epoch in range(int(args.epochs)):
        for batch in loader:
            z = batch["x0"].to(device, non_blocking=True)        # [B,T,K,C]
            m = batch["mask"].to(device, non_blocking=True)      # [B,T,K]
            coord = batch["coord"].to(device, non_blocking=True) # [B,T,N,3]

            if coord.shape[2] != int(args.num_points_target):
                raise ValueError(
                    f"coord N={coord.shape[2]} does not match --num_points_target={int(args.num_points_target)}. "
                    "If you unified point counts elsewhere, make sure this matches the dataset output."
                )

            if args.normalize != "none":
                coord, _, _ = normalize_sequence_xyz(coord, mode=args.normalize)

            pred = dec(z, mask=m)  # [B,T,N,3]

            # loss-only normalization scale (does not change coords)
            loss_scale = compute_loss_norm_scale(coord)
            loss_den = (loss_scale * loss_scale + float(args.loss_norm_eps))

            # --- reconstruction loss ---
            loss_rec = 0.0
            loss_zero = 0.0
            for t in range(pred.shape[1]):
                if args.loss_type == "mse":
                    loss_rec = loss_rec + mse_loss(pred[:, t], coord[:, t])
                    loss_zero = loss_zero + mse_loss(torch.zeros_like(coord[:, t]), coord[:, t])
                else:
                    loss_rec = loss_rec + chamfer_proxy(pred[:, t], coord[:, t])
                    loss_zero = loss_zero + chamfer_proxy(torch.zeros_like(coord[:, t]), coord[:, t])

            loss_rec = loss_rec / float(pred.shape[1])
            loss_zero = loss_zero / float(pred.shape[1])

            # Apply loss-only normalization to reconstruction losses
            loss_rec = loss_rec / loss_den
            loss_zero = loss_zero / loss_den

            improve_vs_zero = (1.0 - (loss_rec / (loss_zero + 1e-12))).clamp(min=-10.0, max=10.0)

            # --- FK loss (uses body pose) ---
            loss_fk = pred.new_tensor(0.0)
            use_fk = (
                float(args.lambda_fk) > 0
                and ("segment" in batch)
                and ("body_xpos" in batch)
                and ("body_xmat" in batch)
            )
            if use_fk:
                segment = batch["segment"].to(device, non_blocking=True)     # [B,N] or [N]
                body_xpos = batch["body_xpos"].to(device, non_blocking=True) # [B,T,nb,3]
                body_xmat = batch["body_xmat"].to(device, non_blocking=True) # [B,T,nb,9]

                loss_fk = fk_warp_consistency_loss(
                    xyz_pred=pred,
                    segment=segment,
                    body_xpos=body_xpos,
                    body_xmat=body_xmat,
                    robust_delta=float(args.fk_huber_delta),
                )
                # Apply same loss-only normalization
                loss_fk = loss_fk / loss_den

            # --- rigid pairwise-distance consistency loss (no pose) ---
            loss_rigid = pred.new_tensor(0.0)
            use_rigid = float(args.lambda_rigid) > 0 and ("segment" in batch)
            if use_rigid:
                segment = batch["segment"].to(device, non_blocking=True)
                loss_rigid = rigid_pairwise_distance_consistency_loss(
                    xyz_pred=pred,
                    segment=segment,
                    num_pairs_per_body=int(args.rigid_pairs),
                    robust_delta=float(args.rigid_delta),
                )
                # Apply same loss-only normalization
                loss_rigid = loss_rigid / loss_den

            # total
            loss = loss_rec + float(args.lambda_fk) * loss_fk + float(args.lambda_rigid) * loss_rigid

            # optional: shuffled-latent eval (no grad)
            shuffle_loss = None
            if args.eval_shuffle and (global_step % int(args.eval_every) == 0):
                with torch.no_grad():
                    Bcur = z.shape[0]
                    perm = torch.randperm(Bcur, device=z.device)
                    z_shuf = z.index_select(0, perm)
                    m_shuf = m.index_select(0, perm)
                    pred_shuf = dec(z_shuf, mask=m_shuf)

                    sl = 0.0
                    for t in range(pred_shuf.shape[1]):
                        if args.loss_type == "mse":
                            sl = sl + mse_loss(pred_shuf[:, t], coord[:, t])
                        else:
                            sl = sl + chamfer_proxy(pred_shuf[:, t], coord[:, t])
                    shuffle_loss = (sl / float(pred_shuf.shape[1])).detach()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if global_step % 20 == 0:
                coord_std = coord.std().item()
                coord_absmax = coord.abs().max().item()
                pred_std = pred.std().item()
                pred_absmax = pred.abs().max().item()

                loss_fk_now = float(loss_fk.item())
                fk_contrib = float(args.lambda_fk) * loss_fk_now

                loss_rigid_now = float(loss_rigid.item())
                rigid_contrib = float(args.lambda_rigid) * loss_rigid_now

                loss_now = float(loss.item())
                fk_ratio = fk_contrib / max(loss_now, 1e-12)
                rigid_ratio = rigid_contrib / max(loss_now, 1e-12)

                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss_now:.4e} rec={float(loss_rec.item()):.4e} "
                    f"fk={loss_fk_now:.4e} fk_contrib={fk_contrib:.4e} fk_ratio={fk_ratio:.3f} "
                    f"rigid={loss_rigid_now:.4e} rigid_contrib={rigid_contrib:.4e} rigid_ratio={rigid_ratio:.3f} "
                    f"loss_zero={loss_zero.item():.4e} improve={improve_vs_zero.item():.3f} "
                    f"coord_std={coord_std:.3e} coord_absmax={coord_absmax:.3e} "
                    f"pred_std={pred_std:.3e} pred_absmax={pred_absmax:.3e} "
                    f"loss_norm={args.loss_norm} loss_scale={float(loss_scale.item()):.3e} "
                    f"loss_type={args.loss_type}"
                )

                if args.wandb:
                    payload = {
                        "train/loss": float(loss_now),
                        "train/loss_rec": float(loss_rec.item()),
                        "train/loss_fk": float(loss_fk_now),
                        "train/fk_contrib": float(fk_contrib),
                        "train/fk_ratio": float(fk_ratio),
                        "train/loss_rigid": float(loss_rigid_now),
                        "train/rigid_contrib": float(rigid_contrib),
                        "train/rigid_ratio": float(rigid_ratio),
                        "train/loss_zero": float(loss_zero.item()),
                        "train/improve_vs_zero": float(improve_vs_zero.item()),
                        "train/loss_type": str(args.loss_type),
                        "fk/lambda_fk": float(args.lambda_fk),
                        "fk/huber_delta": float(args.fk_huber_delta),
                        "rigid/lambda_rigid": float(args.lambda_rigid),
                        "rigid/pairs": int(args.rigid_pairs),
                        "rigid/huber_delta": float(args.rigid_delta),
                        "stats/coord_std": coord_std,
                        "stats/coord_absmax": coord_absmax,
                        "stats/pred_std": pred_std,
                        "stats/pred_absmax": pred_absmax,
                        "global_step": global_step,
                        "epoch": epoch,
                        "loss_norm/mode": str(args.loss_norm),
                        "loss_norm/scale": float(loss_scale.item()),
                        "loss_norm/den": float(loss_den.item()),
                    }
                    if shuffle_loss is not None:
                        payload["eval/shuffle_loss"] = float(shuffle_loss.item())
                        payload["eval/shuffle_gap"] = float((shuffle_loss - loss_rec).item())
                    wandb.log(payload, step=global_step)

            global_step += 1

    # save checkpoint (always)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": dec.state_dict(),
            "args": vars(args),
        },
        args.save_path,
    )
    print(f"Saved decoder checkpoint to {args.save_path}")

    if args.wandb:
        wandb.finish()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({"state_dict": dec.state_dict(), "args": vars(args)}, args.save_path)
        print(f"Saved decoder to {args.save_path}")


if __name__ == "__main__":
    main()
