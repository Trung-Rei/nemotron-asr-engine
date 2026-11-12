"""Where does the engine worker's time go under the streaming load test, and how much of it is waiting for the GIL?

Same workload as engine_loadtest.py. The engine is not modified: its worker methods are wrapped from outside.
Inside the measurement window [warmup, duration] of each run:

  worker phases  wall and thread-CPU time of drain / prepare / finish (= decode + settle) / flush / submit / idle.
                 CUDA syncs spin (CPU time), so wall - CPU of a phase is time the thread was blocked: the GIL.
  GPU timeline   CUDA events per batch: "encoder" = submit start -> submit end (the whole step's graph since greedy
                 moved into it), "greedy span" = -> collect end, idle = collect end -> next submit start.
  threads        CPU % of the worker thread and of the event-loop thread (/proc/self/task).

Blocked time that grows with the event loop's work (20 ms vs 100 ms packets) at the same GPU work is GIL waiting.

    python bench/profiling/profile_engine.py --runs 448:20 512:20 512:100 --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_loadtest as lt  # noqa: E402  (same workload)
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

PHASES = ["drain", "prepare", "finish", "decode", "flush", "submit", "idle"]


class Prof:
    def __init__(self):
        self.on = False
        self.reset()

    def reset(self):
        self.wall = collections.defaultdict(float)
        self.cpu = collections.defaultdict(float)
        self.n = collections.defaultdict(int)
        self.batches = []  # (e0, e1, e2, b, host time at e0)
        self.trace = []  # (phase, t0, t1) worker spans inside [trace_t0, trace_t0 + trace_s]
        self.trace_t0 = float("inf")
        self.trace_s = 0.0

    def wrap(self, name, fn):
        def w(*a, **k):
            if not self.on:
                return fn(*a, **k)
            t, c = time.perf_counter(), time.thread_time()
            try:
                return fn(*a, **k)
            finally:
                t1 = time.perf_counter()
                self.wall[name] += t1 - t
                self.cpu[name] += time.thread_time() - c
                self.n[name] += 1
                if self.trace_t0 <= t < self.trace_t0 + self.trace_s:
                    self.trace.append((name, t, t1))
        return w


def instrument(engine, prof: Prof) -> None:
    T = prof.wrap
    engine._drain = T("drain", engine._drain)
    engine._prepare = T("prepare", engine._prepare)
    engine._flush_out = T("flush", engine._flush_out)
    engine._wake.wait = T("idle", engine._wake.wait)
    engine.stepper.collect = T("decode", engine.stepper.collect)  # wait for the step + parse (+ eager redo)
    submit, finish = engine._submit, engine._finish

    def _submit(bt):
        e0 = torch.cuda.Event(enable_timing=True)
        bt.t_host = time.perf_counter()
        e0.record()
        submit(bt)
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        bt.ev = (e0, e1, prof.on)

    def _finish(bt):
        finish(bt)
        if bt.ev[2] and prof.on:
            e2 = torch.cuda.Event(enable_timing=True)
            e2.record()
            prof.batches.append((bt.ev[0], bt.ev[1], e2, len(bt.states), bt.t_host))

    engine._submit = T("submit", _submit)
    engine._finish = T("finish", _finish)


def thread_cpu_s(native_id: int) -> float:
    f = open(f"/proc/self/task/{native_id}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


async def profile_run(engine, prof, args, level, packet_ms, out: Path):
    a = argparse.Namespace(**vars(args))
    a.level, a.packet_ms = level, packet_ms
    rng = np.random.default_rng(args.seed + level)
    arrivals = lt.open_arrivals(rng, level, args.session_s, args.duration)
    timing = lt.TimingModel()
    sessions = [lt.make_session(i, x, rng, timing, args.pool, args.session_s) for i, x in enumerate(arrivals)]
    label = f"c{level}_p{packet_ms}"
    print(f"=== {label}: {len(sessions)} sessions", flush=True)
    prof.reset()
    tids = {"worker": engine._thread.native_id, "loop": threading.main_thread().native_id}
    win = {}

    async def window():
        await asyncio.sleep(1.0 + args.warmup)
        win["cpu0"] = {k: thread_cpu_s(v) for k, v in tids.items()}
        win["p0"], win["t0"] = time.process_time(), time.perf_counter()
        prof.trace_t0, prof.trace_s = win["t0"] + args.trace_at, args.trace_s
        prof.on = True
        await asyncio.sleep(args.duration - args.warmup)
        prof.on = False
        win["t1"], win["p1"] = time.perf_counter(), time.process_time()
        win["cpu1"] = {k: thread_cpu_s(v) for k, v in tids.items()}

    s0 = engine.stats()
    r, _ = await asyncio.gather(lt.run_level(engine, sessions, a), window())
    s1 = engine.stats()
    res = lt.summarize(label, a, sessions, *r)
    T = win["t1"] - win["t0"]
    res["window_s"] = T
    res["greedy_continuations_per_step"] = (s1["greedy_continuations"] - s0["greedy_continuations"]) / max(1, s1["steps"] - s0["steps"])
    res["thread_cpu_pct"] = {k: 100 * (win["cpu1"][k] - win["cpu0"][k]) / T for k in tids}
    res["process_cpu_pct_window"] = 100 * (win["p1"] - win["p0"]) / T
    rounds = prof.n["finish"] or 1  # one round per batch
    res["rounds_per_s"] = prof.n["finish"] / T
    res["phases"] = {p: {"pct_wall": 100 * prof.wall[p] / T, "calls_per_round": prof.n[p] / rounds, "ms_per_round": 1000 * prof.wall[p] / rounds,
                         "blocked_ms_per_round": 1000 * (prof.wall[p] - prof.cpu[p]) / rounds} for p in PHASES}
    top = sum(prof.wall[p] for p in ["drain", "prepare", "finish", "flush", "submit", "idle"])
    res["phases"]["other"] = {"pct_wall": 100 * (T - top) / T}
    torch.cuda.synchronize()
    b = prof.batches
    if len(b) > 1:
        enc = sum(x[0].elapsed_time(x[1]) for x in b)
        greedy = sum(x[1].elapsed_time(x[2]) for x in b)
        gap = sum(b[i][2].elapsed_time(b[i + 1][0]) for i in range(len(b) - 1))
        span = b[0][0].elapsed_time(b[-1][2])
        res["gpu"] = {"batches": len(b), "mean_batch": float(np.mean([x[3] for x in b])),
                      "encoder_pct": 100 * enc / span, "greedy_span_pct": 100 * greedy / span, "idle_pct": 100 * gap / span,
                      "encoder_ms": enc / len(b), "greedy_span_ms": greedy / len(b), "idle_ms_per_batch": gap / (len(b) - 1)}
    # trace: worker spans, GPU spans (host clock of the first traced submit + GPU event offsets), chunk lifelines
    t0 = prof.trace_t0
    tb = [x for x in b if t0 <= x[4] < t0 + prof.trace_s]
    if tb:
        ref_e, ref_t = tb[0][0], tb[0][4]
        g = lambda e: (ref_t + ref_e.elapsed_time(e) / 1000 - t0) * 1000  # noqa: E731
        res["trace"] = {
            "span_ms": prof.trace_s * 1000,
            "worker": [(n, (a - t0) * 1000, (z - t0) * 1000) for n, a, z in prof.trace],
            "gpu": [{"b": x[3], "submit": (x[4] - t0) * 1000, "start": g(x[0]), "graph_end": g(x[1]), "end": g(x[2])} for x in tb],
            "chunks": [((r_ - t0) * 1000, (s_ - t0) * 1000, (e_ - t0) * 1000, n_) for r_, s_, e_, n_ in r[1] if t0 <= r_ < t0 + prof.trace_s],
        }
    (out / f"{label}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return res


async def main_async(args):
    from common import load_rows_with_audio

    from nemotron_engine import AsyncNemotronEngine, EngineConfig

    rows, _ = load_rows_with_audio(args.pool_size, 42)
    args.pool = lt.AudioPool(rows)
    engine = await AsyncNemotronEngine.create(EngineConfig(
        language="vi-VN", lookahead=args.lookahead, max_streams=args.max_streams, max_batch=args.max_batch,
        matmul_precision=args.matmul_precision, record_chunks=True, max_lag_ms=None))
    prof = Prof()
    if not args.no_instrument:
        instrument(engine, prof)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.runs:
        level, packet = spec.split(":")
        results.append(await profile_run(engine, prof, args, int(level), int(packet), out))
    await engine.aclose()

    f = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
    lines = ["| run | batch | chunks/s | finalize p95 | rounds/s | worker CPU % | loop CPU % | GPU enc / greedy / idle % | "
             + " | ".join(f"{p} ms/round (blocked)" for p in ["drain", "prepare", "decode", "finish", "flush", "submit", "idle"]) + " |",
             "|" + "---|" * 15]
    for r in results:
        ph, g = r["phases"], r.get("gpu", {})
        lines.append(f"| {r['label']} | {r['mean_batch'] or 0:.1f} | {r['chunks_per_s']:.0f} | {f(r['finalize_ms'][95])} | "
                     f"{r['rounds_per_s']:.1f} | {r['thread_cpu_pct']['worker']:.0f} | {r['thread_cpu_pct']['loop']:.0f} | "
                     f"{g.get('encoder_pct', 0):.0f} / {g.get('greedy_span_pct', 0):.0f} / {g.get('idle_pct', 0):.0f} | "
                     + " | ".join(f"{ph[p]['ms_per_round']:.2f} ({ph[p]['blocked_ms_per_round']:.2f})"
                                  for p in ["drain", "prepare", "decode", "finish", "flush", "submit", "idle"]) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["448:20", "512:20", "512:100"], help="level:packet_ms")
    p.add_argument("--lookahead", type=int, default=3)
    p.add_argument("--session-s", type=float, default=40.0)
    p.add_argument("--warmup", type=float, default=40.0)
    p.add_argument("--duration", type=float, default=70.0)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-streams", type=int, default=768)
    p.add_argument("--slo-ms", type=float, default=300.0)
    p.add_argument("--matmul-precision", default="high")
    p.add_argument("--pool-size", type=int, default=1500)
    p.add_argument("--no-instrument", action="store_true", help="same run without the wrappers / CUDA events (A/B check)")
    p.add_argument("--trace-at", type=float, default=10.0, help="seconds into the window where the trace starts")
    p.add_argument("--trace-s", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
