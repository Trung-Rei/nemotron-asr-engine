"""Real-time streaming load test through AsyncNemotronEngine (the asyncio API an application embeds).

Same workload as loadtest.py (Poisson session arrivals, synthetic utterance timing, VietSuperSpeech audio), but
every session is an asyncio task that pushes `--packet-ms` packets on the wall clock and calls end_turn() the
moment an utterance ends (an oracle endpointer), and transcripts come back as events.

Metrics (measurement window [warmup, duration]):
  chunk lag   t_emit - t_ready per decoded chunk (worker side, same as loadtest.py's lag)
  finalize    final event received on the loop - end_turn() call   (what a client waits for after its VAD fires)
  push late   actual push time - scheduled packet time   (event loop health: GIL shared with the worker)
  loop lag    oversleep of a 10 ms ticker on the event loop
SLO: p95 finalize <= slo, p95 chunk lag <= chunk, no rejected sessions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

# isort: split
from workload import SR, AudioPool, TimingModel, make_session, open_arrivals, strip_tags  # noqa: E402


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


async def run_level(engine, sessions, args):
    from nemotron_engine import EngineFull

    t0 = time.perf_counter() + 1.0
    pkt = int(SR * args.packet_ms / 1000)
    ends, late, rejected = [], [], [0]
    loop_lag, stop = [], asyncio.Event()

    async def ticker():
        while not stop.is_set():
            a = time.perf_counter()
            await asyncio.sleep(0.01)
            loop_lag.append((time.perf_counter() - a - 0.01) * 1000)

    async def run_session(s):
        await asyncio.sleep(max(0.0, t0 + s.arrival - time.perf_counter()))
        try:
            st = engine.open_stream(encoding="f32")
        except EngineFull:
            rejected[0] += 1
            return
        got = []

        async def reader():
            async for ev in st:
                if ev.kind == "final":
                    got.append((ev, time.perf_counter()))

        rd = asyncio.create_task(reader())
        start = t0 + s.arrival
        turn_ends = [int(t.end * SR) for t in s.turns]
        ti, my_ends = 0, []
        end_at = min(len(s.audio), int((args.duration + 5.0 - s.arrival) * SR))
        for i in range(0, max(0, end_at), pkt):
            due = start + (i + pkt) / SR  # packet i..i+pkt is complete at this time
            d = due - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            now = time.perf_counter()
            late.append((now - due) * 1000)
            st.push_nowait(s.audio[i : i + pkt])
            while ti < len(turn_ends) and turn_ends[ti] <= i + pkt:
                st.end_turn()
                my_ends.append((time.perf_counter(), s.turns[ti]))
                ti += 1
        await st.close()
        await rd
        for k, (t_end, turn) in enumerate(my_ends):
            if k < len(got):
                ev, t_recv = got[k]
                ends.append((t_end - t0, (t_recv - t_end) * 1000, (ev.t_emit - t_end) * 1000, turn.reference, ev.text))

    engine.chunk_log.clear()
    tk = asyncio.create_task(ticker())
    cpu0, w0 = time.process_time(), time.perf_counter()
    await asyncio.gather(*(run_session(s) for s in sessions))
    cpu = (time.process_time() - cpu0) / (time.perf_counter() - w0) * 100
    stop.set()
    await tk
    return ends, list(engine.chunk_log), late, loop_lag, rejected[0], cpu, t0


def summarize(label, args, sessions, ends, chunks, late, loop_lag, rejected, cpu, t0):
    import jiwer
    from common import normalize_for_wer

    w0, w1 = args.warmup, args.duration
    ch = np.array([(r - t0, (e - r) * 1000, b) for r, s, e, b in chunks]) if chunks else np.zeros((0, 3))
    m = (ch[:, 0] >= w0) & (ch[:, 0] < w1)
    lag, batch = ch[m, 1], ch[m, 2]
    fe = [e for e in ends if w0 <= e[0] < w1]
    fin = [e[1] for e in fe]
    fin_worker = [e[2] for e in fe]
    pairs = [(normalize_for_wer(strip_tags(r)), normalize_for_wer(h)) for _, _, _, r, h in fe if r]
    pairs = [(r, h) for r, h in pairs if r]
    chunk_ms = (args.lookahead + 1) * 80
    res = {
        "label": label, "target_concurrency": args.level, "sessions": len(sessions), "rejected": rejected,
        "packet_ms": args.packet_ms, "chunks_per_s": float(m.sum() / (w1 - w0)), "mean_batch": float(batch.mean()) if len(batch) else None,
        "lag_ms": {q: pct(lag, q) for q in (50, 95, 99)},
        "finalize_ms": {q: pct(fin, q) for q in (50, 95, 99)},
        "finalize_worker_ms": {q: pct(fin_worker, q) for q in (50, 95, 99)},
        "turns": len(fe), "push_late_ms": {q: pct(late, q) for q in (50, 95, 99)},
        "loop_lag_ms": {q: pct(loop_lag, q) for q in (50, 95, 99)}, "process_cpu_pct": cpu,
        "wer": 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]) if pairs else None,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }
    res["slo_ok"] = bool(fin and res["finalize_ms"][95] <= args.slo_ms and len(lag) and res["lag_ms"][95] <= chunk_ms and rejected == 0)
    return res


async def main_async(args):
    from common import load_rows_with_audio

    from nemotron_engine import AsyncNemotronEngine, EngineConfig

    rows, _ = load_rows_with_audio(args.pool_size, 42)
    pool = AudioPool(rows, telephone=args.telephone)
    timing = TimingModel()
    engine = await AsyncNemotronEngine.create(EngineConfig(
        language="vi-VN", lookahead=args.lookahead, max_streams=args.max_streams, max_batch=args.max_batch,
        matmul_precision=args.matmul_precision, record_chunks=True, max_lag_ms=args.max_lag_ms or None,
        cuda_graphs=not args.no_cuda_graphs))
    if args.switch_interval_ms:
        sys.setswitchinterval(args.switch_interval_ms / 1000)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, fails = [], 0
    for level in args.levels:
        args.level = level
        rng = np.random.default_rng(args.seed + level)
        arrivals = open_arrivals(rng, level, args.session_s, args.duration)
        sessions = [make_session(i, a, rng, timing, pool, args.session_s) for i, a in enumerate(arrivals)]
        label = f"engine_open_c{level}_p{args.packet_ms}{args.tag}"
        torch.cuda.reset_peak_memory_stats()
        print(f"=== {label}: {len(sessions)} sessions", flush=True)
        r = await run_level(engine, sessions, args)
        res = summarize(label, args, sessions, *r)
        res["engine_stats"] = engine.stats()
        results.append(res)
        (out / f"{label}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)
        fails = 0 if res["slo_ok"] else fails + 1
        if fails >= args.stop_after_fail:
            print("stopping sweep: SLO missed", flush=True)
            break
    await engine.aclose()
    lines = ["| level | packet | sessions | rej | chunks/s | batch | lag p50/p95 | finalize p50/p95/p99 | push late p95 | loop lag p95 | CPU % | WER | SLO |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
    for r in results:
        lines.append(f"| {r['target_concurrency']} | {r['packet_ms']} ms | {r['sessions']} | {r['rejected']} | {r['chunks_per_s']:.0f} | "
                     f"{r['mean_batch'] or 0:.1f} | {f(r['lag_ms'][50])}/{f(r['lag_ms'][95])} | "
                     f"{f(r['finalize_ms'][50])}/{f(r['finalize_ms'][95])}/{f(r['finalize_ms'][99])} | {f(r['push_late_ms'][95])} | "
                     f"{f(r['loop_lag_ms'][95])} | {r['process_cpu_pct']:.0f} | {r['wer']:.1f} | {'✅' if r['slo_ok'] else '❌'} |")
    (out / f"report_p{args.packet_ms}{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=int, nargs="+", default=[256, 448, 512])
    p.add_argument("--packet-ms", type=int, default=20)
    p.add_argument("--lookahead", type=int, default=3)
    p.add_argument("--session-s", type=float, default=40.0)
    p.add_argument("--warmup", type=float, default=40.0)
    p.add_argument("--duration", type=float, default=90.0)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-streams", type=int, default=768)
    p.add_argument("--slo-ms", type=float, default=300.0)
    p.add_argument("--matmul-precision", default="high")
    p.add_argument("--pool-size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stop-after-fail", type=int, default=1)
    p.add_argument("--max-lag-ms", type=float, default=0.0, help="engine admission threshold; 0 disables it")
    p.add_argument("--switch-interval-ms", type=float, default=0.0, help="sys.setswitchinterval; 0 keeps 5 ms")
    p.add_argument("--telephone", action="store_true", help="pass the audio through an 8 kHz band")
    p.add_argument("--tag", default="")
    p.add_argument("--no-cuda-graphs", action="store_true")
    p.add_argument("--out", required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
