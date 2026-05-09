# preencode_all_cosmos_pc_latent.py
from __future__ import annotations

import os
import glob
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pf_encoder
from pf_encoder import PTV3Encoder


class CosmosPointcloudEpisodeDataset(Dataset):
    """Read episodes from <dataset_dir>/pointclouds/*.npy"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.pc_dir = os.path.join(self.dataset_dir, "pointclouds")
        if not os.path.isdir(self.pc_dir):
            raise RuntimeError(f"pointclouds dir not found: {self.pc_dir}")

        self.files = sorted(glob.glob(os.path.join(self.pc_dir, "*.npy")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy found in: {self.pc_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        d = np.load(path, allow_pickle=True).item()
        if "coord" not in d:
            raise KeyError(f"Missing 'coord' in {path}")
        d["__path__"] = path
        return d



POINTCEPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pointcept_abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(POINTCEPT_ROOT, path))


def configure_point_encoder(args) -> dict:
    encoder_config = {
        "DATASET": args.encoder_dataset,
        "CONFIG": args.encoder_config,
        "EXP_NAME": args.encoder_exp_name,
        "WEIGHT_NAME": args.encoder_weight_name,
    }
    if args.config_file:
        encoder_config["CONFIG_FILE"] = args.config_file
    if args.exp_dir:
        encoder_config["EXP_DIR"] = args.exp_dir
    if args.weight_path:
        encoder_config["WEIGHT_PATH"] = args.weight_path

    pf_encoder.apply_encoder_config(encoder_config)
    pf_encoder.CONFIG_FILE = _pointcept_abs(pf_encoder.CONFIG_FILE)
    pf_encoder.EXP_DIR = _pointcept_abs(pf_encoder.EXP_DIR)
    pf_encoder.WEIGHT_PATH = _pointcept_abs(pf_encoder.WEIGHT_PATH)

    return {
        "CONFIG_FILE": pf_encoder.CONFIG_FILE,
        "EXP_DIR": pf_encoder.EXP_DIR,
        "WEIGHT_PATH": pf_encoder.WEIGHT_PATH,
        "DATASET": pf_encoder.DATASET,
        "CONFIG": pf_encoder.CONFIG,
        "EXP_NAME": pf_encoder.EXP_NAME,
        "WEIGHT_NAME": pf_encoder.WEIGHT_NAME,
    }


def collate_as_list(batch: List[Dict[str, Any]]):
    return batch


def list_dataset_dirs(root_dir: str) -> List[str]:
    """Return subdirs that look like a dataset: contains pointclouds/*.npy"""
    root_dir = os.path.abspath(root_dir)
    direct_pc_dir = os.path.join(root_dir, "pointclouds")
    if os.path.isdir(direct_pc_dir) and len(glob.glob(os.path.join(direct_pc_dir, "*.npy"))) > 0:
        return [root_dir]

    subdirs = sorted([p for p in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(p)])
    out = []
    for d in subdirs:
        pc_dir = os.path.join(d, "pointclouds")
        if os.path.isdir(pc_dir) and len(glob.glob(os.path.join(pc_dir, "*.npy"))) > 0:
            out.append(d)
    return out


def get_shard_indices(total_count: int, shard_id: int, num_shards: int, max_episodes: Optional[int]) -> List[int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

    limit = total_count if max_episodes is None else min(total_count, int(max_episodes))
    return [idx for idx in range(limit) if (idx % num_shards) == shard_id]


@torch.no_grad()
def preencode_one_dataset(
    dataset_dir: str,
    encoder: PTV3Encoder,
    k: int,
    sample: str,
    pad_value: float,
    amp: bool,
    num_workers: int,
    overwrite: bool,
    max_episodes: Optional[int],
    verbose: bool,
    grid_size: float,
    shard_id: int = 0,
    num_shards: int = 1,
):
    ds = CosmosPointcloudEpisodeDataset(dataset_dir)
    out_dir = os.path.join(os.path.abspath(dataset_dir), "pc_latent")
    os.makedirs(out_dir, exist_ok=True)

    shard_indices = get_shard_indices(
        total_count=len(ds),
        shard_id=int(shard_id),
        num_shards=int(num_shards),
        max_episodes=max_episodes,
    )
    n = len(shard_indices)

    if verbose:
        full_n = len(ds) if max_episodes is None else min(len(ds), int(max_episodes))
        print(
            f"  shard {shard_id + 1}/{num_shards}: processing {n} episodes "
            f"from {full_n} eligible episodes"
        )

    if n == 0:
        if verbose:
            print(f"  shard {shard_id + 1}/{num_shards}: nothing to do")
        return {
            "dataset_dir": dataset_dir,
            "wrote": 0,
            "skipped": 0,
            "total": 0,
            "out_dir": out_dir,
        }

    # Variable-length T: must use batch_size=1 (your encode_batch requires same T within a batch)
    loader = DataLoader(
        torch.utils.data.Subset(ds, shard_indices),
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=collate_as_list,
    )

    written = 0
    skipped = 0

    for batch in loader:
        ep = batch[0]
        if "grid_size" not in ep or ep["grid_size"] is None:
            ep["grid_size"] = float(grid_size)

        src_path = ep.get("__path__", "")
        base = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(out_dir, f"{base}.pt")

        if (not overwrite) and os.path.exists(out_path):
            skipped += 1
            continue

        feats, mask = encoder.encode_batch(
            batch,
            k=int(k),
            sample=str(sample),
            pad_value=float(pad_value),
            return_mask=True,
            amp=bool(amp),
        )

        payload = {
            "x0": feats[0].detach().cpu(),          # [T, k, C] (your code calls this k, but it's L in other parts)
            "mask": mask[0].detach().cpu().bool(),  # [T, k]
            "src_path": src_path,
            "k": int(k),
            "sample": str(sample),
            "pad_value": float(pad_value),
            "encoder": {
                "DATASET": pf_encoder.DATASET,
                "CONFIG": pf_encoder.CONFIG,
                "EXP_NAME": pf_encoder.EXP_NAME,
                "WEIGHT_NAME": pf_encoder.WEIGHT_NAME,
                "CONFIG_FILE": pf_encoder.CONFIG_FILE,
                "EXP_DIR": pf_encoder.EXP_DIR,
                "WEIGHT_PATH": pf_encoder.WEIGHT_PATH,
            },
        }
        torch.save(payload, out_path)
        written += 1

        if verbose and (written % 50 == 0):
            print(f"    wrote {written}/{n} (skipped {skipped}) -> {out_dir}")

    if verbose:
        print(f"  done: wrote={written}, skipped={skipped}, total={n}, out={out_dir}")

    return {"dataset_dir": dataset_dir, "wrote": written, "skipped": skipped, "total": n, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="e.g. ~/CollectDataset/cosmos_training_data")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--sample", default="first")
    ap.add_argument("--pad_value", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_datasets", type=int, default=None)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--grid_size", type=float, required=True)
    ap.add_argument("--encoder_dataset", default="robotwin")
    ap.add_argument("--encoder_config", default="semseg-pt-v3m1-0-base")
    ap.add_argument("--encoder_exp_name", default="semseg-pt-v3m1-0-base-cosmos-pcenc")
    ap.add_argument("--encoder_weight_name", default="model_last")
    ap.add_argument("--config_file", default=None, help="Optional direct Pointcept config path")
    ap.add_argument("--exp_dir", default=None, help="Optional direct Pointcept experiment directory")
    ap.add_argument("--weight_path", default=None, help="Optional direct Pointcept checkpoint path")
    ap.add_argument(
        "--num_shards",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
        help="Number of independent shards/processes. Defaults to WORLD_SIZE when launched with torchrun.",
    )
    ap.add_argument(
        "--shard_id",
        type=int,
        default=int(os.environ.get("RANK", "0")),
        help="This process shard id in [0, num_shards). Defaults to RANK when launched with torchrun.",
    )
    ap.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
        help="Local CUDA device index. Defaults to LOCAL_RANK when launched with torchrun.",
    )
    args = ap.parse_args()

    if args.num_shards <= 0:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"--shard_id must be in [0, {args.num_shards}), got {args.shard_id}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if args.local_rank < 0 or args.local_rank >= device_count:
            raise ValueError(f"--local_rank must be in [0, {device_count}), got {args.local_rank}")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cpu")

    print(
        f"Launching shard {args.shard_id + 1}/{args.num_shards} on device={device} "
        f"(local_rank={args.local_rank})"
    )

    encoder_config = configure_point_encoder(args)
    print("Point encoder config:")
    for key, value in encoder_config.items():
        print(f"  {key}: {value}")

    encoder = pf_encoder.PTV3Encoder(pf_encoder.load_ptv3_model()).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    dataset_dirs = list_dataset_dirs(args.root_dir)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: int(args.max_datasets)]

    if len(dataset_dirs) == 0:
        raise RuntimeError(f"No dataset dirs found under: {os.path.abspath(args.root_dir)}")

    print(f"Found {len(dataset_dirs)} dataset dirs under {os.path.abspath(args.root_dir)}")
    for d in dataset_dirs:
        print(f"  - {d}")

    total_wrote = 0
    total_skipped = 0
    total_eps = 0

    for i, dataset_dir in enumerate(dataset_dirs):
        print(f"\n[{i+1}/{len(dataset_dirs)}] encoding dataset: {dataset_dir}")
        stats = preencode_one_dataset(
            dataset_dir=dataset_dir,
            encoder=encoder,
            k=args.k,
            sample=args.sample,
            pad_value=args.pad_value,
            amp=args.amp,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            max_episodes=args.max_episodes,
            verbose=True,
            grid_size=float(args.grid_size),
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
        total_wrote += stats["wrote"]
        total_skipped += stats["skipped"]
        total_eps += stats["total"]

    print("\nAll done.")
    print(
        f"shard={args.shard_id}/{args.num_shards} episodes_total={total_eps} "
        f"wrote={total_wrote} skipped={total_skipped}"
    )


if __name__ == "__main__":
    main()