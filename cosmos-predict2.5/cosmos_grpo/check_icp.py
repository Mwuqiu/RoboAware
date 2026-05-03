"""
check_icp.py — Text-based diagnostic for ICP alignment results.

Supports two JSON formats:
  • cosmos_output format: top-level keys (reward, frame_records, global, …)
  • replay format: wrapped under 'pred'/'gt' sub-keys

Usage:
  python cosmos_grpo/check_icp.py --icp_json <path_or_glob> [options]

Options:
  --icp_json PATH        Path to icp.json (can be glob; first match used if multiple)
  --icp_json2 PATH       Second icp.json to compare (pred vs gt side-by-side)
  --label LABEL          Section label to read from replay format (default: pred)
  --no_warnings           Suppress auto-warning section
  --full                 Print full T_global matrix (default: print summary only)
  --frames               Print per-frame detail table (default: always on)
  --good_only            Only print "good" frames

Examples:
  python cosmos_grpo/check_icp.py \\
      --icp_json cosmos-output/grpo_debug_v0/iter_000001/sample_00/sample_00_icp.json

  python cosmos_grpo/check_icp.py \\
      --icp_json /tmp/replay_result_pred.json \\
      --icp_json2 /tmp/replay_result_gt.json

  python cosmos_grpo/check_icp.py \\
      --icp_json 'cosmos-output/**/*_icp.json' --label pred
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Formatting helpers ──────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"


def _color(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"{code}{text}{RESET}"
    return text


def _ok(text: str) -> str:   return _color(text, GREEN)
def _warn(text: str) -> str: return _color(text, YELLOW)
def _bad(text: str) -> str:  return _color(text, RED)
def _hdr(text: str) -> str:  return _color(text, BOLD + CYAN)


def _bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """ASCII progress bar 0‥1."""
    n = max(0, min(width, int(round(value * width))))
    return fill * n + empty * (width - n)


# ── JSON loading ────────────────────────────────────────────────────────────

def _resolve_path(path: str) -> str:
    """Expand glob; return first match or the literal path."""
    if "*" in path or "?" in path:
        matches = sorted(glob.glob(path, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No files matched glob: {path!r}")
        return matches[0]
    return path


def load_result(path: str, label: str = "pred") -> Tuple[Dict, str, str]:
    """Load an icp.json and return (result_dict, detected_format, label_used).

    Detected formats: 'cosmos' | 'replay'
    """
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    # Replay format has 'pred'/'gt' sub-keys
    if "pred" in j or "gt" in j:
        fmt = "replay"
        r = j.get(label) or j.get("pred") or next(iter(j.values()))
        used_label = label
    else:
        fmt = "cosmos"
        r = j
        used_label = "pred"

    return r, fmt, used_label


def _frame_key(rec: Dict, fmt: str) -> int:
    """Return the depth frame index for a frame_record."""
    if fmt == "replay":
        return int(rec.get("depth_fid", rec.get("src_idx", 0)))
    return int(rec.get("global_depth_frame", rec.get("src_idx", 0)))


def _frame_sim(rec: Dict, fmt: str) -> int:
    if fmt == "replay":
        return int(rec.get("sim_fid", rec.get("sim_global_idx", 0)))
    return int(rec.get("sim_global_idx", 0))


# ── Section printers ────────────────────────────────────────────────────────

def print_header(path: str, r: Dict, fmt: str, label: str) -> None:
    print()
    print(_hdr("=" * 72))
    print(_hdr(f"  ICP Diagnostic  —  {os.path.basename(path)}"))
    print(_hdr(f"  Format: {fmt}   Label: {label}   Path: {path}"))
    print(_hdr("=" * 72))


def print_rewards(r: Dict) -> None:
    reward   = float(r.get("reward", float("nan")))
    r_local  = float(r.get("reward_local", float("nan")))
    r_global = float(r.get("reward_global_alignment", float("nan")))

    print()
    print(_hdr("── Rewards ─────────────────────────────────────────────────"))
    bar = _bar(reward)
    clr = _ok if reward >= 0.40 else (_warn if reward >= 0.20 else _bad)
    print(f"  REWARD        : {clr(f'{reward:.4f}')}  [{bar}]")
    if not np.isnan(r_local):
        print(f"  local (ICP)   : {r_local:.4f}")
    if not np.isnan(r_global):
        print(f"  global (align): {r_global:.4f}")


def print_global_info(r: Dict) -> List[str]:
    """Print global T summary; return list of warning strings."""
    g = r.get("global", {})
    warnings: List[str] = []

    print()
    print(_hdr("── Global T ────────────────────────────────────────────────"))
    if g:
        print(f"  num_candidates : {g.get('num_candidates', '?')}")
        print(f"  good_pairs     : {g.get('good_pairs_for_refine', '?')}")
        print(f"  global_refine  : {g.get('global_refine', '?')}")
        rf = g.get("global_refine_fitness", float("nan"))
        rr = g.get("global_refine_rmse",    float("nan"))
        print(f"  refine fitness : {rf:.4f}")
        print(f"  refine rmse    : {rr:.4f}  m")
        if not np.isnan(rr) and rr > 0.05:
            warnings.append(f"global_refine_rmse={rr:.4f} is high (>0.05 m)")
    else:
        print("  (no global section)")

    # T_global centroid shift magnitude
    T_global = g.get("T_global")
    if T_global:
        T = np.array(T_global, dtype=np.float64)
        t = T[:3, 3]
        R = T[:3, :3]
        angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1, 1))))
        print(f"  T_global translation : [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]  |t|={np.linalg.norm(t):.4f} m")
        print(f"  T_global rotation    : {angle_deg:.2f}°")

    # Global alignment summary
    gs = r.get("global_alignment_summary", {})
    if gs:
        score = gs.get("avg_alignment_score", float("nan"))
        mean  = gs.get("avg_mean_dist",       float("nan"))
        p90   = gs.get("avg_p90_dist",        float("nan"))
        outr  = gs.get("avg_outlier_ratio",   float("nan"))
        print()
        print(_hdr("── Global-T Alignment Summary ──────────────────────────────"))
        print(f"  frames            : {gs.get('frames', '?')}")
        clr = _ok if score >= 0.4 else (_warn if score >= 0.2 else _bad)
        print(f"  avg_alignment_score: {clr(f'{score:.4f}')}")
        print(f"  avg_mean_dist      : {mean:.4f} m")
        print(f"  avg_p90_dist       : {p90:.4f} m")
        print(f"  avg_outlier_ratio  : {outr:.4f}")
        if score < 0.2:
            warnings.append(f"global avg_alignment_score={score:.4f} is very low (<0.2)")
        if outr > 0.3:
            warnings.append(f"global avg_outlier_ratio={outr:.4f} is high (>0.3)")

    return warnings


def print_frame_table(r: Dict, fmt: str, good_only: bool = False) -> List[str]:
    """Print per-frame table; return list of warning strings."""
    records = r.get("frame_records", [])
    if not records:
        print("\n  (no frame_records)")
        return []

    warnings: List[str] = []
    bad_frames: List[int] = []

    COL_W = 76
    print()
    print(_hdr("── Per-Frame ICP Results ───────────────────────────────────"))
    header = f"  {'fid':>5}  {'sim':>5}  {'fit':>6}  {'rmse':>7}  {'mean':>7}  {'p90':>7}  {'out':>6}  {'score':>7}  {'good':>4}"
    print(header)
    print("  " + "-" * (COL_W - 2))

    for rec in records:
        fid   = _frame_key(rec, fmt)
        sim   = _frame_sim(rec, fmt)
        fit   = float(rec.get("fitness", float("nan")))
        rmse  = float(rec.get("rmse",    float("nan")))
        mean  = float(rec.get("mean_dist",       float("nan")))
        p90   = float(rec.get("p90_dist",        float("nan")))
        outr  = float(rec.get("outlier_ratio",   float("nan")))
        score = float(rec.get("alignment_score", float("nan")))
        good  = bool(rec.get("good", True))

        if good_only and not good:
            continue

        # Per-field color
        score_s = (_ok if score >= 0.40 else (_warn if score >= 0.20 else _bad))(f"{score:.4f}")
        good_s  = _ok("yes") if good else _bad("no ")

        row = (
            f"  {fid:>5}  {sim:>5}  {fit:>6.3f}  {rmse:>7.4f}  "
            f"{mean:>7.4f}  {p90:>7.4f}  {outr:>6.3f}  {score_s}  {good_s}"
        )
        print(row)

        if score < 0.15:
            bad_frames.append(fid)

    print("  " + "-" * (COL_W - 2))

    # Per-frame summary
    ls = r.get("alignment_summary", {})
    if ls:
        score = ls.get("avg_alignment_score", float("nan"))
        clr_s = _ok if score >= 0.40 else (_warn if score >= 0.20 else _bad)
        print(
            f"  {'AVG':>5}  {'':>5}  {'':>6}  {'':>7}  "
            f"{ls.get('avg_mean_dist',float('nan')):>7.4f}  "
            f"{ls.get('avg_p90_dist', float('nan')):>7.4f}  "
            f"{ls.get('avg_outlier_ratio', float('nan')):>6.3f}  "
            f"{clr_s(f'{score:.4f}')}"
        )

    if bad_frames:
        warnings.append(f"Frames with alignment_score<0.15: {bad_frames}")

    n_bad = sum(1 for r2 in records if not r2.get("good", True))
    if n_bad > 0:
        warnings.append(f"{n_bad}/{len(records)} frames marked good=False")

    return warnings


def print_compare(r1: Dict, fmt1: str, l1: str, r2: Dict, fmt2: str, l2: str) -> None:
    """Side-by-side compare of two results (e.g. pred vs gt)."""
    print()
    print(_hdr("── Comparison ──────────────────────────────────────────────"))
    keys = ["reward", "reward_local", "reward_global_alignment"]
    for k in keys:
        v1 = float(r1.get(k, float("nan")))
        v2 = float(r2.get(k, float("nan")))
        diff = v1 - v2
        sign = "+" if diff >= 0 else ""
        print(f"  {k:<30}: {l1}={v1:.4f}  {l2}={v2:.4f}  Δ={sign}{diff:.4f}")

    # Frame-by-frame score diff
    recs1 = {_frame_key(r, fmt1): r for r in r1.get("frame_records", [])}
    recs2 = {_frame_key(r, fmt2): r for r in r2.get("frame_records", [])}
    common = sorted(set(recs1) & set(recs2))
    if common:
        print()
        print(f"  {'fid':>5}  {l1+' score':>10}  {l2+' score':>10}  {'Δ':>8}")
        print("  " + "-" * 42)
        for fid in common:
            s1 = float(recs1[fid].get("alignment_score", float("nan")))
            s2 = float(recs2[fid].get("alignment_score", float("nan")))
            diff = s1 - s2
            sign = "+" if diff >= 0 else ""
            clr = _ok if abs(diff) < 0.05 else (_warn if abs(diff) < 0.15 else _bad)
            print(f"  {fid:>5}  {s1:>10.4f}  {s2:>10.4f}  {clr(f'{sign}{diff:.4f}')}")


def print_warnings(warnings: List[str]) -> None:
    if not warnings:
        print()
        print(_ok("  [OK] No significant issues detected."))
        return
    print()
    print(_hdr("── Warnings ────────────────────────────────────────────────"))
    for w in warnings:
        print(_warn(f"  [WARN] {w}"))


def print_t_global(r: Dict) -> None:
    g = r.get("global", {})
    T_global = g.get("T_global")
    if T_global is None:
        return
    T = np.array(T_global, dtype=np.float64)
    print()
    print(_hdr("── T_global (full 4×4) ─────────────────────────────────────"))
    for row in T:
        print("  " + "  ".join(f"{x:+.6f}" for x in row))


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Text-based diagnostic for ICP alignment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    p.add_argument("--icp_json",  required=True,  help="Path (or glob) to icp.json")
    p.add_argument("--icp_json2", default=None,   help="Second icp.json for comparison")
    p.add_argument("--label",  default="pred",    help="Label to read from replay format (default: pred)")
    p.add_argument("--label2", default="gt",      help="Label for second json (default: gt)")
    p.add_argument("--no_warnings", action="store_true", help="Suppress warning section")
    p.add_argument("--full",        action="store_true", help="Print full T_global matrix")
    p.add_argument("--good_only",   action="store_true", help="Only show good frames in table")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    path1 = _resolve_path(args.icp_json)
    r1, fmt1, label1 = load_result(path1, label=args.label)

    print_header(path1, r1, fmt1, label1)
    print_rewards(r1)
    warnings = []
    warnings += print_global_info(r1)
    warnings += print_frame_table(r1, fmt1, good_only=args.good_only)

    if args.full:
        print_t_global(r1)

    # Optional second file
    if args.icp_json2:
        path2 = _resolve_path(args.icp_json2)
        r2, fmt2, label2 = load_result(path2, label=args.label2)
        print()
        print(_hdr(f"-- Second file: {os.path.basename(path2)}  Label: {label2} --"))
        print_rewards(r2)
        warnings += print_global_info(r2)
        warnings += print_frame_table(r2, fmt2, good_only=args.good_only)
        if args.full:
            print_t_global(r2)
        print_compare(r1, fmt1, label1, r2, fmt2, label2)

    if not args.no_warnings:
        print_warnings(warnings)

    print()


if __name__ == "__main__":
    main()
