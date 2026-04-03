#!/usr/bin/env python3
"""
Post-run analysis: extract BPB from seed logs, compute p-value vs SOTA,
check artifact size, and generate submission.json + PR folder.

Usage:
    python analyze_runs.py log1.log log2.log log3.log [--baseline 1.1147]

After RunPod runs, download the logs and run this script. It will:
  1. Parse final_slot_exact val_bpb from each log
  2. Run paired one-sided t-test vs baseline SOTA
  3. Check artifact size (code bytes)
  4. Print submission.json template with real values
  5. Optionally copy files into records/track_10min_16mb/<folder>/
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUR_SUBMISSION = Path(__file__).resolve().parent


def extract_bpb(log_path: str) -> dict:
    """Extract key metrics from a training log file."""
    result = {
        "slot_bpb": None,
        "slot_loss": None,
        "sliding_bpb": None,
        "sliding_loss": None,
        "bytes_total": None,
        "bytes_model": None,
        "bytes_code": None,
        "seed": None,
    }
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            # Primary: SLOT BPB (exact)
            m = re.search(r"final_slot_exact val_loss:([\d.]+) val_bpb:([\d.]+)", line)
            if m:
                result["slot_loss"] = float(m.group(1))
                result["slot_bpb"] = float(m.group(2))
            # SLOT timing: "final_slot val_loss:X val_bpb:X steps:64 lr:0.010 time:XXXXXms"
            m = re.search(r"final_slot.*time:([\d.]+)ms", line)
            if m:
                result["slot_time_ms"] = float(m.group(1))
            # Fallback: sliding window BPB (exact)
            m = re.search(r"final_int6_sliding_window_exact val_loss:([\d.]+) val_bpb:([\d.]+)", line)
            if m:
                result["sliding_loss"] = float(m.group(1))
                result["sliding_bpb"] = float(m.group(2))
            # Artifact size — matches log lines like:
            #   "Total submission size int6+lzma: 15400000 bytes"
            #   "Serialized model int6+lzma: 15350000 bytes"
            #   "Code size: 50000 bytes"
            m = re.search(r"Total submission size[^:]*:\s*([\d]+)\s*bytes", line)
            if m:
                result["bytes_total"] = int(m.group(1))
            m = re.search(r"Serialized model[^:]*:\s*([\d]+)\s*bytes", line)
            if m:
                result["bytes_model"] = int(m.group(1))
            m = re.search(r"Code size:\s*([\d]+)\s*bytes", line)
            if m:
                result["bytes_code"] = int(m.group(1))
            # Seed
            m = re.search(r"seed[=:\s]+(\d+)", line, re.IGNORECASE)
            if m and result["seed"] is None:
                result["seed"] = int(m.group(1))
    return result


def ttest_one_sided(values: list[float], baseline: float) -> tuple[float, float]:
    """One-sided paired t-test: H1 = mean < baseline. Returns (t_stat, p_value)."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    diffs = [baseline - v for v in values]  # positive = improvement
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d)
    if std_d == 0:
        return float("inf"), 0.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    # Approximate p-value using t-distribution CDF (one-sided)
    # Using scipy if available, else simple approximation
    try:
        from scipy import stats
        p_value = stats.t.sf(t_stat, df=n - 1)  # one-sided upper tail (improvement)
    except ImportError:
        # Simple normal approximation for large n
        import math
        # Use regularized incomplete beta function approximation
        # For df degrees of freedom
        df = n - 1
        x = df / (df + t_stat ** 2)
        # Very rough approximation: use complementary normal CDF
        z = t_stat
        p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return t_stat, p_value


def check_artifact_size(script_path: Path) -> int:
    """Return code size in bytes (model bytes embedded in script)."""
    return script_path.stat().st_size


def main():
    parser = argparse.ArgumentParser(description="Analyze Parameter Golf run logs and prepare submission")
    parser.add_argument("logs", nargs="+", help="Log files from training runs")
    parser.add_argument("--baseline", type=float, default=1.1147, help="SOTA BPB to beat (default: 1.1147)")
    parser.add_argument("--threshold", type=float, default=0.005, help="Minimum improvement in nats (default: 0.005)")
    parser.add_argument("--p-threshold", type=float, default=0.01, help="P-value threshold (default: 0.01)")
    parser.add_argument("--make-pr-folder", action="store_true", help="Copy files to records/track_10min_16mb/")
    parser.add_argument("--author", default="Danilo Canivel", help="Your name")
    parser.add_argument("--github-id", default="canivel", help="Your GitHub ID")
    args = parser.parse_args()

    print("=" * 60)
    print("  Parameter Golf — Post-Run Analysis")
    print("=" * 60)

    # Parse all logs
    results = []
    for log_path in args.logs:
        if not os.path.exists(log_path):
            print(f"WARNING: Log not found: {log_path}")
            continue
        r = extract_bpb(log_path)
        r["log"] = log_path
        results.append(r)
        bpb = r["slot_bpb"] or r["sliding_bpb"]
        print(f"\nLog: {log_path}")
        print(f"  SLOT BPB:     {r['slot_bpb']}")
        print(f"  Sliding BPB:  {r['sliding_bpb']}")
        print(f"  Seed:         {r['seed']}")
        if r.get("slot_time_ms"):
            slot_min = r["slot_time_ms"] / 60_000
            limit_ok = r["slot_time_ms"] <= 600_000
            print(f"  SLOT time:    {slot_min:.1f} min [{'OK' if limit_ok else 'OVER 10min LIMIT!'}]")
        if r["bytes_total"]:
            print(f"  Bytes total:  {r['bytes_total']:,} / 16,000,000")
        if bpb:
            delta = args.baseline - bpb
            print(f"  Δ vs SOTA:    {delta:+.6f} nats ({'IMPROVEMENT' if delta > 0 else 'regression'})")

    if not results:
        print("\nERROR: No valid logs parsed.")
        sys.exit(1)

    # Use SLOT BPB preferentially
    bpb_values = [r["slot_bpb"] or r["sliding_bpb"] for r in results]
    bpb_values = [v for v in bpb_values if v is not None]

    if not bpb_values:
        print("\nERROR: Could not extract any BPB values from logs.")
        print("  Expected lines like: 'final_slot_exact val_loss:X val_bpb:Y'")
        sys.exit(1)

    mean_bpb = sum(bpb_values) / len(bpb_values)
    best_bpb = min(bpb_values)
    mean_improvement = args.baseline - mean_bpb

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Seeds run:        {len(bpb_values)}")
    print(f"  BPB values:       {[f'{v:.6f}' for v in bpb_values]}")
    print(f"  Mean BPB:         {mean_bpb:.6f}")
    print(f"  Best BPB:         {best_bpb:.6f}")
    print(f"  Baseline SOTA:    {args.baseline:.6f}")
    print(f"  Mean improvement: {mean_improvement:+.6f} nats")

    # Statistical test
    t_stat, p_value = ttest_one_sided(bpb_values, args.baseline)
    print(f"\n  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.6f} (one-sided)")

    # Timing check
    slot_times = [r.get("slot_time_ms") for r in results if r.get("slot_time_ms")]
    if slot_times:
        max_slot_ms = max(slot_times)
        slot_time_ok = max_slot_ms <= 600_000
        print(f"\n  Max SLOT eval time: {max_slot_ms/60000:.1f} min (limit: 10 min)")
        if not slot_time_ok:
            print(f"  WARNING: SLOT eval exceeded 10-min limit! Reduce SLOT_STEPS before submitting.")
    else:
        slot_time_ok = True  # unknown, assume ok
        print("\n  SLOT timing: not found in logs (will be known after first run)")

    # Checks
    improvement_ok = mean_improvement >= args.threshold
    p_ok = p_value <= args.p_threshold
    seeds_ok = len(bpb_values) >= 3

    print(f"\n  [{'PASS' if improvement_ok else 'FAIL'}] Improvement >= {args.threshold} nats: {mean_improvement:.6f}")
    print(f"  [{'PASS' if p_ok else 'FAIL'}] p < {args.p_threshold}: {p_value:.6f}")
    print(f"  [{'PASS' if seeds_ok else 'FAIL'}] >= 3 seed runs: {len(bpb_values)}")
    print(f"  [{'PASS' if slot_time_ok else 'FAIL'}] SLOT eval <= 10 min")

    # Artifact size check
    train_script = OUR_SUBMISSION / "train_gpt.py"
    if train_script.exists():
        code_bytes = train_script.stat().st_size
        print(f"\n  Code size: {code_bytes:,} bytes")
        # Model bytes from log (if available)
        bytes_total = next((r["bytes_total"] for r in results if r["bytes_total"]), None)
        if bytes_total:
            print(f"  Total artifact: {bytes_total:,} / 16,000,000 bytes")
            size_ok = bytes_total <= 16_000_000
            print(f"  [{'PASS' if size_ok else 'FAIL'}] Artifact <= 16MB")
        else:
            print("  NOTE: Artifact total bytes not found in logs — check manually after run.")

    # Submission readiness
    all_pass = improvement_ok and p_ok and seeds_ok and slot_time_ok
    print(f"\n  {'READY TO SUBMIT!' if all_pass else 'NOT YET READY — see failures above'}")

    # Generate submission.json
    print("\n" + "=" * 60)
    print("  submission.json (fill in after runs)")
    print("=" * 60)

    best_result = min(results, key=lambda r: r["slot_bpb"] or r["sliding_bpb"] or 99.9)
    submission = {
        "author": args.author,
        "github_id": args.github_id,
        "name": "Record: 11L LeakyReLU² + XSA-all + Full GPTQ + SLOT64 + AR-calib + BigramHash(3072,112)",
        "blurb": (
            "11L 512d GQA(8h/4kv). LeakyReLU(0.5)² MLP 3x. XSA on all 11 layers. Full Hessian GPTQ "
            "(Cholesky+column-reorder) with AR self-generated calibration (64×2048 tokens, temp=0.8). "
            "Int6 MLP+attn, Int8 embed, lzma preset=9. SLOT eval: 64 AdamW steps, cosine LR 0.010→0.001, "
            "warmstart=0.85. BigramHash(3072,112). Focal loss γ=1.0. Sqrt warmdown."
        ),
        "date": "2026-04-03T00:00:00Z",
        "val_bpb": round(mean_bpb, 8),
        "best_seed_val_bpb": round(best_bpb, 8),
        "val_loss": round(best_result.get("slot_loss") or 0.0, 8),
        "pre_quant_val_bpb": None,  # fill from log
        "bytes_total": best_result.get("bytes_total"),
        "bytes_model": best_result.get("bytes_model"),
        "bytes_code": best_result.get("bytes_code"),
        "seeds": bpb_values,
        "p_value": round(p_value, 6),
        "improvement_nats": round(mean_improvement, 6),
    }
    print(json.dumps(submission, indent=2))

    # Optionally create PR folder
    if args.make_pr_folder and all_pass:
        folder_name = f"2026-04-03_11L_LeakyReLU2_XSA-all_GPTQ-AR_SLOT64_BigramHash3072_{best_bpb:.4f}"
        pr_folder = REPO_ROOT / "records" / "track_10min_16mb" / folder_name
        pr_folder.mkdir(parents=True, exist_ok=True)
        # Copy train_gpt.py
        shutil.copy(OUR_SUBMISSION / "train_gpt.py", pr_folder / "train_gpt.py")
        # Write submission.json
        (pr_folder / "submission.json").write_text(json.dumps(submission, indent=2))
        # Copy logs
        for i, log_path in enumerate(args.logs[:3]):
            seed = results[i].get("seed") or (1337 + i)
            shutil.copy(log_path, pr_folder / f"train_seed{seed}.log")
        print(f"\nPR folder created: {pr_folder}")
        print("Next: update README.md in that folder, then open PR to openai/parameter-golf")
    elif args.make_pr_folder and not all_pass:
        print("\nSkipping PR folder creation — not all checks passed.")


if __name__ == "__main__":
    main()
