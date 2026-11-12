#!/usr/bin/env python3
"""Compare lite vs NeMo result files: per-utterance agreement and WER delta.

Pairs compared (when both files exist in RESULTS_DIR):
  ref_offline_r{R}_{N} vs lite_offline_r{R}_{N}
  ref_stream_r{R}_{N}  vs lite_stream_r{R}_{N}
  lite_offline_r{R}_{N} vs lite_stream_r{R}_{N}   (cache-aware streaming must equal offline)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from common import RESULTS_DIR

MIN_MATCH = 0.99
MAX_WER_DELTA = 0.1  # percentage points


def load(stem):
    summary = json.loads((RESULTS_DIR / f"{stem}.summary.json").read_text())
    records = [json.loads(l) for l in (RESULTS_DIR / f"{stem}.jsonl").read_text().splitlines() if l.strip()]
    return summary, records


def compare(a_stem, b_stem):
    a_sum, a = load(a_stem)
    b_sum, b = load(b_stem)
    assert [r["audio"] for r in a] == [r["audio"] for r in b], f"{a_stem} and {b_stem} cover different utterances"
    n = len(a)
    text_same = sum(x["normalized_hypothesis"] == y["normalized_hypothesis"] for x, y in zip(a, b))
    raw_same = sum(x["hypothesis"] == y["hypothesis"] for x, y in zip(a, b))
    tok_pairs = [(x.get("tokens"), y.get("tokens")) for x, y in zip(a, b)]
    tok_same = sum(p == q for p, q in tok_pairs) if all(p is not None and q is not None for p, q in tok_pairs) else None
    diffs = [
        {"audio": x["audio"], a_stem: x["hypothesis"], b_stem: y["hypothesis"]}
        for x, y in zip(a, b)
        if x["normalized_hypothesis"] != y["normalized_hypothesis"]
    ]
    delta = b_sum["wer_percent"] - a_sum["wer_percent"]
    passed = text_same / n >= MIN_MATCH and abs(delta) <= MAX_WER_DELTA
    return {
        "a": a_stem, "b": b_stem, "samples": n,
        "wer_a": a_sum["wer_percent"], "wer_b": b_sum["wer_percent"], "wer_delta_points": delta,
        "normalized_text_match": text_same / n, "raw_text_match": raw_same / n,
        "token_match": None if tok_same is None else tok_same / n,
        "rtfx_a": a_sum.get("rtfx"), "rtfx_b": b_sum.get("rtfx"),
        "vram_gb_a": a_sum.get("peak_vram_gb"), "vram_gb_b": b_sum.get("peak_vram_gb"),
        "lite_forbidden_modules": b_sum.get("forbidden_modules_loaded"),
        "pass": passed, "mismatches": diffs[:20],
    }


def main():
    stems = {p.name[: -len(".summary.json")] for p in RESULTS_DIR.glob("*.summary.json")}
    pairs = []
    for s in sorted(stems):
        m = re.fullmatch(r"ref_(offline|stream)_r(\d+)_(\d+)", s)
        if m and f"lite_{m.group(1)}_r{m.group(2)}_{m.group(3)}" in stems:
            pairs.append((s, f"lite_{m.group(1)}_r{m.group(2)}_{m.group(3)}"))
        m = re.fullmatch(r"lite_offline_r(\d+)_(\d+)", s)
        if m and f"lite_stream_r{m.group(1)}_{m.group(2)}" in stems:
            pairs.append((s, f"lite_stream_r{m.group(1)}_{m.group(2)}"))
    if not pairs:
        print("no comparable result pairs found in", RESULTS_DIR)
        sys.exit(1)
    report = [compare(a, b) for a, b in pairs]
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ["| A | B | N | WER A | WER B | Δ (pts) | text match | token match | RTFx A | RTFx B | pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report:
        tm = "-" if r["token_match"] is None else f"{r['token_match']:.2%}"
        fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
        lines.append(
            f"| {r['a']} | {r['b']} | {r['samples']} | {r['wer_a']:.2f} | {r['wer_b']:.2f} | {r['wer_delta_points']:+.3f} | "
            f"{r['normalized_text_match']:.2%} | {tm} | {fmt(r['rtfx_a'])} | {fmt(r['rtfx_b'])} | {'✅' if r['pass'] else '❌'} |"
        )
    table = "\n".join(lines)
    (RESULTS_DIR / "comparison.md").write_text(table + "\n")
    print(table)
    for r in report:
        for d in r["mismatches"][:5]:
            print(json.dumps(d, ensure_ascii=False))


if __name__ == "__main__":
    main()
