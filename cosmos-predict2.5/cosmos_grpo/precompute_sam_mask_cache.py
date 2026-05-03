#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from cosmos_grpo.config.grpo_so100_point_adapter import CosmosGRPOConfig
from cosmos_grpo.reward_online.sam_depth_pipeline import OnlinePointCloudExtractor, OnlineVisionConfig


def _norm_episode_key(value: str) -> str:
    return value.strip().replace("__", "_")


def _parse_annotation(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[skip] failed to parse {path}: {exc}")
        return None

    episode_id = str(data.get("episode_id", "")).strip()
    ann_frame_idx = int(data.get("ann_frame_idx", 0))
    points_raw = data.get("points", None)
    labels_raw = data.get("labels", None)

    if not episode_id or not isinstance(points_raw, list) or len(points_raw) == 0:
        print(f"[skip] invalid annotation content: {path}")
        return None

    points: List[Tuple[float, float]] = []
    for point in points_raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        points.append((float(point[0]), float(point[1])))
    if not points:
        print(f"[skip] no valid points in {path}")
        return None

    if isinstance(labels_raw, list) and len(labels_raw) == len(points):
        labels = [1 if int(v) > 0 else 0 for v in labels_raw]
    else:
        labels = [1 for _ in points]

    return {
        "episode_id": episode_id,
        "episode_key": _norm_episode_key(episode_id),
        "stem_key": _norm_episode_key(path.stem),
        "ann_frame_idx": ann_frame_idx,
        "points": points,
        "labels": labels,
        "path": path,
    }


def _resolve_video_path(videos_dir: Path, ann: dict) -> Optional[Path]:
    candidates = [
        videos_dir / f"{ann['path'].stem}.mp4",
        videos_dir / f"{ann['episode_id']}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    cfg = CosmosGRPOConfig()
    parser = argparse.ArgumentParser(
        description="Precompute full-video SAM mask caches from annotation points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=cfg.dataset_dir)
    parser.add_argument("--annotations-dir", default=None)
    parser.add_argument("--videos-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--sam2-checkpoint", default=cfg.sam2_checkpoint)
    parser.add_argument("--sam2-model-cfg", default=cfg.sam2_model_cfg)
    parser.add_argument("--sam-obj-id", type=int, default=cfg.sam_obj_id)
    parser.add_argument("--episode-id", action="append", default=None,
                        help="Only process matching episode_id/file stem. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retry count for a failed episode build, excluding the first attempt.")
    parser.add_argument("--summary-json", default=None,
                        help="Optional path to write a run summary json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else dataset_dir / "annotations"
    videos_dir = Path(args.videos_dir) if args.videos_dir else dataset_dir / "videos"
    cache_dir = Path(args.cache_dir) if args.cache_dir else dataset_dir / "sam_mask_cache"

    episode_filters = None
    if args.episode_id:
        episode_filters = {_norm_episode_key(x) for x in args.episode_id}

    extractor = OnlinePointCloudExtractor(
        OnlineVisionConfig(
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
            sam_obj_id=args.sam_obj_id,
        )
    )

    ann_paths = sorted(annotations_dir.glob("*.json"))
    work_items: List[tuple[Path, dict, Path, Path]] = []
    skipped = 0
    for ann_path in ann_paths:
        ann = _parse_annotation(ann_path)
        if ann is None:
            skipped += 1
            continue

        if episode_filters and ann["episode_key"] not in episode_filters and ann["stem_key"] not in episode_filters:
            continue

        video_path = _resolve_video_path(videos_dir, ann)
        if video_path is None:
            print(f"[skip] video not found for {ann_path.name}")
            skipped += 1
            continue

        episode_cache_dir = cache_dir / ann["episode_key"]
        meta_path = episode_cache_dir / "meta.json"
        if meta_path.exists() and not args.overwrite:
            print(f"[skip] cache exists for {ann['episode_id']}: {episode_cache_dir}")
            skipped += 1
            continue

        work_items.append((ann_path, ann, video_path, episode_cache_dir))

    if args.limit is not None:
        work_items = work_items[: args.limit]

    processed = 0
    failed = 0
    retried_success = 0
    failed_items: List[dict] = []
    started_at = time.time()

    progress = tqdm(work_items, desc="precompute_sam_mask_cache", unit="episode")
    for index, (_, ann, video_path, episode_cache_dir) in enumerate(progress, start=1):
        attempts = 0
        last_error = None
        success = False
        max_attempts = max(1, int(args.retries) + 1)

        while attempts < max_attempts and not success:
            attempts += 1
            try:
                episode_cache_dir.mkdir(parents=True, exist_ok=True)
                progress.set_postfix_str(
                    f"episode={ann['episode_id']} attempt={attempts}/{max_attempts}"
                )
                extractor.build_video_mask_cache(
                    video_path=str(video_path),
                    cache_dir=str(episode_cache_dir),
                    prompt_points_xy=ann["points"],
                    prompt_labels=ann["labels"],
                    ann_frame_idx=int(ann["ann_frame_idx"]),
                )
                success = True
            except Exception as exc:
                last_error = exc
                print(
                    f"[fail] episode={ann['episode_id']} attempt={attempts}/{max_attempts} error={exc}"
                )
                if attempts >= max_attempts:
                    break
                time.sleep(1.0)

        if success:
            processed += 1
            if attempts > 1:
                retried_success += 1
            print(
                f"[done] {index}/{len(work_items)} episode={ann['episode_id']} "
                f"points={len(ann['points'])} attempts={attempts} cache={episode_cache_dir}"
            )
        else:
            failed += 1
            failed_items.append(
                {
                    "episode_id": ann["episode_id"],
                    "annotation_path": str(ann["path"]),
                    "video_path": str(video_path),
                    "cache_dir": str(episode_cache_dir),
                    "attempts": attempts,
                    "error": str(last_error),
                }
            )

    elapsed_sec = time.time() - started_at
    summary = {
        "dataset_dir": str(dataset_dir),
        "annotations_dir": str(annotations_dir),
        "videos_dir": str(videos_dir),
        "cache_dir": str(cache_dir),
        "requested_limit": args.limit,
        "eligible_items": len(work_items),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "retried_success": retried_success,
        "retries": int(args.retries),
        "elapsed_sec": elapsed_sec,
        "failed_items": failed_items,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        summary_path = cache_dir / "precompute_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(
        f"[done] processed={processed} skipped={skipped} failed={failed} "
        f"retried_success={retried_success} elapsed_sec={elapsed_sec:.1f} summary={summary_path}"
    )


if __name__ == "__main__":
    main()