#!/usr/bin/env python3
"""Tables for the streaming load test (loadtest.py): open-workload sweep (capacity at the SLO) and burst recovery."""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
lines = []


def f(x, d=0):
    return "-" if x is None or x != x else f"{x:.{d}f}"


sweeps = {}
for p in (out / "sweep").glob("*_open_c*.json"):  # per-level files, so resumed sweeps merge
    r = json.loads(p.read_text())
    sweeps.setdefault(r["backend"], []).append(r)
for rows in sweeps.values():
    rows.sort(key=lambda r: r["target_concurrency"])
if sweeps:
    any_row = next(iter(sweeps.values()))[0]
    lines += [f"## Open workload sweep (chunk {any_row['chunk_ms']:.0f} ms, {any_row['dtype']}, max batch {any_row['max_batch']})", "",
              "| backend | target streams | mean live | chunks/s | mean batch | lag p50/p95/p99 ms | finalize p50/p95/p99 ms | e2e finalize p95 ms | GPU % | WER % | SLO |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    caps = {}
    for b, rows in sweeps.items():
        for r in rows:
            l, fz, e = r["lag_ms"], r["finalize_ms"], r["finalize_e2e_ms"]
            lines.append(
                f"| {b} | {r['target_concurrency']} | {f(r['mean_concurrency'])} | {f(r['chunks_per_s'])} | {f(r['mean_batch'], 1)} | "
                f"{f(l['50'] if '50' in l else l[50])}/{f(l['95'] if '95' in l else l[95])}/{f(l['99'] if '99' in l else l[99])} | "
                f"{f(fz['50'] if '50' in fz else fz[50])}/{f(fz['95'] if '95' in fz else fz[95])}/{f(fz['99'] if '99' in fz else fz[99])} | "
                f"{f(e['95'] if '95' in e else e[95])} | {f(r['gpu_util_mean'])} | {f(r['wer'], 1)} | {'✅' if r['slo_ok'] else '❌'} |")
        ok = [r["target_concurrency"] for r in rows if r["slo_ok"]]
        caps[b] = max(ok) if ok else 0
    lines += ["", "Capacity (largest tested level meeting p95 finalize ≤ SLO, p95 lag ≤ chunk, no rejects): "
              + ", ".join(f"**{b}: {c} streams**" for b, c in caps.items()), ""]

for p in sorted((out / "burst").glob("*_burst_c*.json")):
    r = json.loads(p.read_text())
    tl, b = r["timeline"], r["burst"]
    chunk = r["chunk_ms"]
    after = [x for x in tl if x["t"] >= b["at"]]
    peak = max((x["lag_p95_ms"] for x in after if x["lag_p95_ms"] == x["lag_p95_ms"]), default=None)
    rec = None
    for i, x in enumerate(after):
        win = after[i : i + 5]
        if x["t"] >= b["at"] + b["spread"] and len(win) == 5 and all(w["lag_p95_ms"] <= chunk for w in win if w["lag_p95_ms"] == w["lag_p95_ms"]):
            rec = x["t"] - b["at"]
            break
    fz = r["finalize_ms"]
    lines += [f"## Burst: {r['backend']} — {r['target_concurrency']} streams steady, +{b['sessions']} sessions within {b['spread']:.0f} s at t={b['at']:.0f} s", "",
              f"- peak live streams: {max(x['streams'] for x in tl)}",
              f"- peak per-second p95 chunk lag: {f(peak)} ms (chunk = {chunk:.0f} ms)",
              f"- recovered (p95 lag ≤ chunk for 5 s straight) after: {f(rec, 0)} s" if rec is not None else "- did not recover within the run",
              f"- whole window: finalize p95 {f(fz.get('95', fz.get(95)))} ms, p99 {f(fz.get('99', fz.get(99)))} ms, rejected sessions {r['rejected']}", ""]

text = "\n".join(lines)
(out / "report.md").write_text(text + "\n")
print(text)
