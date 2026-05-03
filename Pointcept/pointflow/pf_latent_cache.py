from __future__ import annotations

import os
import json
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from pf_dataset import TemporalPointDataset
from pf_encoder import load_ptv3_model, PTV3Encoder


class EncodedLatentDataset(torch.utils.data.Dataset):
    """On-the-fly encoding dataset wrapper."""

    def __init__(
        self,
        base_ds: TemporalPointDataset,
        encoder: PTV3Encoder,
        k: int = 30,
        sample: str = "first",
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.base_ds = base_ds
        self.encoder = encoder
        self.k = int(k)
        self.sample = str(sample)
        self.pad_value = float(pad_value)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        data = self.base_ds[idx]
        feat, mask = self.encoder.encode_sequence_stacked(
            data,
            k=self.k,
            sample=self.sample,
            pad_value=self.pad_value,
            return_mask=True,
        )
        return {"x0": feat.detach().cpu(), "mask": mask.detach().cpu()}


class CachedLatentDataset(torch.utils.data.Dataset):
    """Read cached latents from memmap files."""

    def __init__(self, cache_dir: str):
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise RuntimeError(f"Cache meta not found: {meta_path}")
        meta = json.load(open(meta_path, "r"))
        N = int(meta["N"])
        T = int(meta["T"])
        L = int(meta["L"])
        C = int(meta["C"])

        self.cache_dir = str(cache_dir)
        self.meta = meta
        self.latents = np.memmap(
            os.path.join(cache_dir, "latents.dat"),
            dtype="float32",
            mode="r",
            shape=(N, T, L, C),
        )
        self.masks = np.memmap(
            os.path.join(cache_dir, "masks.dat"),
            dtype="uint8",
            mode="r",
            shape=(N, T, L),
        )

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        x0 = torch.from_numpy(np.array(self.latents[idx]))
        mask = torch.from_numpy(np.array(self.masks[idx]).astype(bool))
        return {"x0": x0, "mask": mask}


class MixedCachedLatentSupervisionDataset(torch.utils.data.Dataset):
    """Latent from cache + supervision fields from base_ds."""

    def __init__(
        self,
        latent_ds: CachedLatentDataset,
        base_ds: TemporalPointDataset,
        supervision_keys: Sequence[str] = ("segment", "body_xpos", "body_xmat", "q"),
        strict: bool = True,
    ):
        super().__init__()
        self.latent_ds = latent_ds
        self.base_ds = base_ds
        self.supervision_keys = tuple(supervision_keys)
        self.strict = bool(strict)

        n_lat = len(self.latent_ds)
        n_base = len(self.base_ds)
        if n_lat > n_base:
            raise RuntimeError(
                f"Latent cache has N={n_lat} samples but base_ds has len={n_base}. "
                "This suggests cache was built from a different dataset/indexing."
            )

    def __len__(self):
        return len(self.latent_ds)

    def __getitem__(self, idx):
        out = dict(self.latent_ds[idx])
        sup = self.base_ds[idx]
        for k in self.supervision_keys:
            if k in sup:
                out[k] = sup[k]
            else:
                if self.strict:
                    raise KeyError(
                        f"Supervision key '{k}' missing in base_ds sample. Available keys: {list(sup.keys())}"
                    )
        return out


def preencode_latents(
    base_ds: TemporalPointDataset,
    encoder: PTV3Encoder,
    cache_dir: str,
    k: int = 30,
    sample: str = "first",
    pad_value: float = 0.0,
    overwrite: bool = False,
    verbose: bool = True,
    enc_batch: int = 100,
    num_workers: int = 16,
    amp: bool = False,
    write_every: int = 10,
    max_samples: int | None = None,
):
    """Pre-encode latents for the dataset into memmap files."""

    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.json")
    latents_path = os.path.join(cache_dir, "latents.dat")
    masks_path = os.path.join(cache_dir, "masks.dat")

    if os.path.exists(meta_path) and not overwrite:
        if verbose:
            print(f"Cache exists at {cache_dir}, skipping preencode.")
        return

    N_full = len(base_ds)
    if N_full == 0:
        raise RuntimeError("Empty base dataset")

    N = int(min(N_full, int(max_samples))) if max_samples is not None else int(N_full)
    if N <= 0:
        raise RuntimeError("max_samples must be > 0")

    subset_ds = torch.utils.data.Subset(base_ds, list(range(N)))

    loader = DataLoader(
        subset_ds,
        batch_size=int(enc_batch),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    first_batch = next(iter(loader))
    with torch.no_grad():
        feats_b, masks_b = encoder.encode_batch(
            first_batch,
            k=k,
            sample=sample,
            pad_value=pad_value,
            return_mask=True,
            amp=amp,
        )
    _, T, L, C = feats_b.shape

    latents = np.memmap(latents_path, dtype="float32", mode="w+", shape=(N, T, L, C))
    masks = np.memmap(masks_path, dtype="uint8", mode="w+", shape=(N, T, L))

    start = 0
    end = min(N, start + feats_b.shape[0])
    latents[start:end] = feats_b.detach().cpu().numpy().astype("float32")[: end - start]
    masks[start:end] = masks_b.detach().cpu().numpy().astype("uint8")[: end - start]

    if verbose:
        extra = f" (subset {N}/{N_full})" if N != N_full else ""
        print(
            f"Preencoding{extra} samples to {cache_dir} (T={T}, L={L}, C={C}), "
            f"enc_batch={enc_batch}, num_workers={num_workers}"
        )

    written = end
    batch_idx = 1

    with torch.no_grad():
        for batch in loader:
            if batch_idx == 1:
                batch_idx += 1
                continue

            feats, masks_t = encoder.encode_batch(
                batch,
                k=k,
                sample=sample,
                pad_value=pad_value,
                return_mask=True,
                amp=amp,
            )
            bsz = feats.shape[0]
            s = written
            e = min(N, s + bsz)
            latents[s:e] = feats.detach().cpu().numpy().astype("float32")[: e - s]
            masks[s:e] = masks_t.detach().cpu().numpy().astype("uint8")[: e - s]
            written = e

            if verbose and (written % (enc_batch * 10) == 0 or written == N):
                print(f"  preencoded {written}/{N}")

            if batch_idx % int(write_every) == 0:
                latents.flush()
                masks.flush()
            batch_idx += 1

            if written >= N:
                break

    latents.flush()
    masks.flush()

    meta = {
        "N": int(N),
        "N_full": int(N_full),
        "T": int(T),
        "L": int(L),
        "C": int(C),
        "k": int(k),
        "sample": str(sample),
        "pad_value": float(pad_value),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    if verbose:
        print(f"Preencode finished, cache saved to {cache_dir}")


def inspect_cache(
    cache_dir: str,
    max_samples: int = 512,
    batch_size: int = 64,
    num_workers: int = 4,
    verbose: bool = True,
):
    """Inspect cache statistics (mask ratios, empty samples)."""

    ds = CachedLatentDataset(cache_dir)
    n = min(len(ds), int(max_samples))
    if n <= 0:
        raise RuntimeError("No samples to inspect")

    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
    )

    total_tokens = 0
    total_valid = 0
    empty_samples = 0
    per_t_valid = None
    per_t_total = None

    seen = 0
    for batch in loader:
        mask = batch["mask"]  # [B,T,L] bool
        B, T, L = mask.shape
        if per_t_valid is None:
            per_t_valid = torch.zeros((T,), dtype=torch.long)
            per_t_total = torch.zeros((T,), dtype=torch.long)

        if seen + B > n:
            mask = mask[: n - seen]
            B = mask.shape[0]

        valid = mask.sum().item()
        total = mask.numel()
        total_valid += int(valid)
        total_tokens += int(total)

        empty_samples += int((mask.view(B, -1).sum(dim=1) == 0).sum().item())

        per_t_valid += mask.sum(dim=(0, 2)).to(torch.long)
        per_t_total += torch.full((T,), B * L, dtype=torch.long)

        seen += B
        if seen >= n:
            break

    overall_ratio = (total_valid / total_tokens) if total_tokens > 0 else 0.0
    empty_ratio = (empty_samples / n) if n > 0 else 0.0
    per_t_ratio = (
        (per_t_valid.float() / per_t_total.clamp(min=1).float()).tolist() if per_t_valid is not None else []
    )

    if verbose:
        print(f"[inspect_cache] cache_dir={cache_dir}")
        print(
            f"[inspect_cache] inspected={n} overall_mask_ratio={overall_ratio:.6f} "
            f"empty_sample_ratio={empty_ratio:.6f}"
        )
        if per_t_valid is not None:
            print(f"[inspect_cache] per_t_mask_ratio (len={len(per_t_ratio)}):")
            print("  " + ", ".join([f"{r:.4f}" for r in per_t_ratio]))

    return {
        "inspected": int(n),
        "overall_mask_ratio": float(overall_ratio),
        "empty_sample_ratio": float(empty_ratio),
        "per_t_mask_ratio": per_t_ratio,
        "total_valid": int(total_valid),
        "total_tokens": int(total_tokens),
    }


def preencode_only(
    cache_dir: str | None = None,
    k: int = 30,
    data_root: str = "pointflow/generated_pointclouds_dataset",
    window_size: int = 25,
    stride: int = 1,
    precompute_index: bool = True,
    overwrite: bool = False,
    device: torch.device | None = None,
    verbose: bool = True,
):
    """CLI-friendly helper to just precompute cache."""

    from pf_encoder import load_ptv3_model

    # lazy import to avoid circular imports
    if cache_dir is None:
        # Use EXP_DIR if available in caller; else fallback to local cache folder.
        cache_dir = os.path.join("exp", "latent_cache", f"T{window_size}_K{k}")

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if verbose:
        print(f"Building dataset from {data_root} (window={window_size}, stride={stride})")
    ds = TemporalPointDataset(
        split="training",
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        precompute_index=precompute_index,
    )

    if len(ds) == 0:
        data_root_abs = os.path.abspath(data_root)
        raise RuntimeError(
            "TemporalPointDataset is empty (len=0). "
            "This usually means no episode .npy files were found under the split folder. "
            f"Please check: {data_root_abs}/training (and contains *.npy)."
        )

    if verbose:
        print("Loading encoder model...")
    encoder = PTV3Encoder(load_ptv3_model()).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    preencode_latents(
        ds,
        encoder,
        cache_dir,
        k=k,
        sample="first",
        pad_value=0.0,
        overwrite=overwrite,
        verbose=verbose,
    )
    return cache_dir


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = PTV3Encoder(load_ptv3_model()).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    
    
    preencode_latents(ds, encoder, cache_dir, k=k_tokens, sample="first", pad_value=0.0, overwrite=True)
