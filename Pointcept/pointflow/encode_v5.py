#!/usr/bin/env python3
"""V5 PC-latent encoder: feat=coord + dec_0 extraction.

This is the canonical V5 encoder script. It bypasses the legacy
encode_batch path (feat=zeros + enc_out) and runs PTV3 with:
  - feat = coord.clone()        (gives the network meaningful input features)
  - extract dec_0 layer         (D=512 voxel features, ~half token count of enc_out)

Output .pt format matches legacy pc_latent_gen.py:
  {"x0": (T,K,512), "mask": (T,K) bool, "src_path", "k", "sample", "pad_value", "encoder": {...}}

K is chosen as ceil(max_voxel_count * 1.1 / 8) * 8. Probed values:
  - 4tasks (160 train + 40 test): max=88, K=104
  - 10tasks (400 train + 100 test): max=88, K=104

Multi-shard via env vars WORLD_SIZE / RANK / LOCAL_RANK (set by torchrun).

Example:
  cd /root/autodl-tmp/Pointcept
  export PYTHONPATH=.:pointflow
  torchrun --nproc_per_node=4 pointflow/encode_v5.py \\
      --data-root /root/autodl-tmp/cosmos_training_data_world_arena_ablation_10tasks \\
      --K 104
"""
import argparse
import os
import sys
import time
import glob

import numpy as np
import torch

POINTCEPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, POINTCEPT_ROOT)
sys.path.insert(0, os.path.join(POINTCEPT_ROOT, "pointflow"))

from pf_encoder import load_ptv3_model, apply_encoder_config
from pointcept.models.utils.structure import Point


def encode_frame(bb, dec0, device, coord_np):
    coord_t = torch.from_numpy(coord_np.astype(np.float32)).to(device)
    feat_t = coord_t.clone()
    data = {
        "coord": coord_t,
        "feat": feat_t,
        "batch": torch.zeros(coord_t.shape[0], dtype=torch.long, device=device),
        "offset": torch.tensor([coord_t.shape[0]], dtype=torch.long, device=device),
        "grid_size": 0.005,
    }
    with torch.no_grad():
        point = Point(data)
        point.serialization(order=bb.order, shuffle_orders=bb.shuffle_orders)
        point.sparsify()
        point = bb.embedding(point)
        point = bb.enc(point)
        point = dec0(point)
    return point.feat.cpu().float()


def probe_voxel_count(bb, dec0, device, files, n_frames_per_file=3):
    """Sample n frames per file to estimate dec_0 voxel count distribution."""
    counts = []
    for fp in files:
        d = np.load(fp, allow_pickle=True).item()
        T = d["coord"].shape[0]
        if n_frames_per_file == 1:
            idxs = [T // 2]
        else:
            idxs = sorted({0, T // 2, T - 1})[:n_frames_per_file]
        for t in idxs:
            counts.append(encode_frame(bb, dec0, device, d["coord"][t]).shape[0])
    return np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Dataset root containing {training,test}/pointclouds/*.npy")
    ap.add_argument("--K", type=int, default=None,
                    help="Token count K. If omitted, probes voxel count and picks ceil(max*1.1/8)*8.")
    ap.add_argument("--pad-value", type=float, default=0.0)
    ap.add_argument("--encoder-dataset", default="robotwin")
    ap.add_argument("--encoder-config", default="semseg-pt-v3m1-0-base")
    ap.add_argument("--encoder-exp-name", default="semseg-pt-v3m1-0-base-cosmos-pcenc")
    ap.add_argument("--encoder-weight-name", default="model_last")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    apply_encoder_config({
        "DATASET": args.encoder_dataset,
        "CONFIG": args.encoder_config,
        "EXP_NAME": args.encoder_exp_name,
        "WEIGHT_NAME": args.encoder_weight_name,
    })

    num_shards = int(os.environ.get("WORLD_SIZE", "1"))
    shard_id = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(shard_id)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    print(f"Shard {shard_id + 1}/{num_shards} on cuda:{local_rank}")

    print("Loading PTV3...")
    ptv3 = load_ptv3_model()
    ptv3.eval()
    bb = ptv3.backbone
    device = next(ptv3.parameters()).device
    dec0 = list(bb.dec.children())[0]
    print(f"  dec_0 type: {type(dec0).__name__}")

    all_files = sorted(glob.glob(os.path.join(args.data_root, "*/pointclouds/*.npy")))
    if not all_files:
        raise RuntimeError(f"No .npy under {args.data_root}/*/pointclouds/")

    K = args.K
    if K is None:
        if shard_id == 0:
            print(f"\n=== probing voxel count over {len(all_files)} files (rank 0 only) ===")
            t0 = time.time()
            counts = probe_voxel_count(bb, dec0, device, all_files, n_frames_per_file=3)
            print(f"  probed in {time.time() - t0:.1f}s: "
                  f"min={counts.min()} p50={int(np.percentile(counts, 50))} "
                  f"p99={int(np.percentile(counts, 99))} max={counts.max()}")
            K = int(np.ceil(counts.max() * 1.1 / 8) * 8)
            print(f"  → K = ceil({counts.max()} * 1.1 / 8) * 8 = {K}")
            # broadcast to other shards via a sentinel file
            sentinel = os.path.join(args.data_root, ".v5_K.txt")
            with open(sentinel, "w") as f:
                f.write(str(K))
        else:
            sentinel = os.path.join(args.data_root, ".v5_K.txt")
            while not os.path.exists(sentinel):
                time.sleep(1)
            with open(sentinel) as f:
                K = int(f.read().strip())
            print(f"  shard {shard_id} read K={K} from sentinel")

    my_files = [fp for i, fp in enumerate(all_files) if i % num_shards == shard_id]
    print(f"\nShard {shard_id} owns {len(my_files)}/{len(all_files)} files; K={K}")

    encoder_meta = {
        "DATASET": args.encoder_dataset,
        "CONFIG": args.encoder_config,
        "EXP_NAME": args.encoder_exp_name,
        "WEIGHT_NAME": args.encoder_weight_name,
        "feat_input": "coord",
        "extract_layer": "dec_0",
        "v5_diff_vs_v3": f"feat=coord + dec_0 extraction (dim 512, K={K})",
    }

    t0 = time.time()
    encoded, skipped = 0, 0
    for idx, fp in enumerate(my_files):
        rel = os.path.relpath(fp, args.data_root)
        out_rel = rel.replace("/pointclouds/", "/pc_latent/").replace(".npy", ".pt")
        out_path = os.path.join(args.data_root, out_rel)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        d = np.load(fp, allow_pickle=True).item()
        coord_all = d["coord"]
        T = coord_all.shape[0]
        out_feat = torch.full((T, K, 512), args.pad_value, dtype=torch.float32)
        out_mask = torch.zeros((T, K), dtype=torch.bool)
        for t in range(T):
            feat = encode_frame(bb, dec0, device, coord_all[t])
            nk = min(feat.shape[0], K)
            out_feat[t, :nk] = feat[:nk]
            out_mask[t, :nk] = True

        torch.save({
            "x0": out_feat,
            "mask": out_mask,
            "src_path": fp,
            "k": K,
            "sample": "first",
            "pad_value": args.pad_value,
            "encoder": encoder_meta,
        }, out_path)
        encoded += 1

        if (idx + 1) % 10 == 0:
            el = time.time() - t0
            rate = encoded / el if encoded > 0 else 0
            eta = (len(my_files) - idx - 1) / rate if rate > 0 else 0
            print(f"  [shard {shard_id}] {idx + 1}/{len(my_files)} encoded={encoded} skipped={skipped} "
                  f"{el:.1f}s rate={rate:.2f} f/s ETA {eta:.0f}s")

    print(f"\nShard {shard_id} DONE. encoded={encoded} skipped={skipped} total={len(my_files)}")


if __name__ == "__main__":
    main()
