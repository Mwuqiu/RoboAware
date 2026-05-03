from __future__ import annotations

from typing import List, Tuple


def build_frame_pairs(
    num_src_frames: int,
    num_tgt_frames: int,
    *,
    stride: int = 1,
    k: float = 1.0,
    offset: float = 0.0,
    max_pairs: int = 16,
) -> List[Tuple[int, int]]:
    """Build frame index pairs with linear mapping tgt ~= k*src + offset."""
    if num_src_frames <= 0 or num_tgt_frames <= 0:
        return []

    stride = max(1, int(stride))
    pairs: List[Tuple[int, int]] = []
    for src_idx in range(0, num_src_frames, stride):
        tgt_idx = int(round(k * src_idx + offset))
        if 0 <= tgt_idx < num_tgt_frames:
            pairs.append((src_idx, tgt_idx))
        if len(pairs) >= max_pairs:
            break
    return pairs
