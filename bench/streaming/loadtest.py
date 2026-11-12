#!/usr/bin/env python3
"""Open-loop real-time load test for streaming ASR (lite or NeMo backend, same scheduler).

Sessions arrive on a wall clock (Poisson, optionally with a burst) and stream audio in real time whether or not
the server keeps up. A chunk becomes available when its last mel frame's audio (20 ms packets) has arrived.
The scheduler repeatedly takes the oldest available chunks of one phase (first / later chunk), up to
--max-batch streams, and runs one batched step.

Per chunk:   lag = done - available                          (system latency, includes queueing)
Per turn:    finalize = done(chunk holding the turn's last sample) - available(that chunk)
             finalize_e2e = done(that chunk) - time the user stopped speaking  (adds the chunk-alignment wait)

  python bench/streaming/loadtest.py --backend lite --mode sweep --levels 16 32 64 --out results/loadtest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

# isort: split
from workload import (  # noqa: E402
    SR,
    AudioPool,
    TimingModel,
    burst_arrivals,
    make_session,
    open_arrivals,
    strip_tags,
)

HOP, HALF_FFT, PACKET = 160, 256, 0.02


class GpuSampler(threading.Thread):
    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_flag = period, [], False

    def run(self):
        while not self.stop_flag:
            try:
                self.samples.append((time.perf_counter(), torch.cuda.utilization()))
            except Exception:
                return
            time.sleep(self.period)


def compute_mel(sessions, frontend):
    """Session audio -> (T, n_mels) CPU tensors, same frames the incremental front-end would produce."""
    for s in sessions:
        a = torch.from_numpy(s.audio)[None].cuda()
        feats, _ = frontend(a, torch.tensor([a.shape[1]], device="cuda"))
        s.mel = feats[0].cpu().pin_memory()


def chunk_plan(s, first_n, later_n, chunk_enc_frames):
    """Start frame, size and availability time (s after arrival) of every chunk of session `s`."""
    total = len(s.audio) // HOP
    starts, sizes = [0], [first_n]
    while starts[-1] + sizes[-1] + later_n <= total:
        starts.append(starts[-1] + sizes[-1])
        sizes.append(later_n)
    if starts[-1] + sizes[-1] > total:
        starts, sizes = [], []
    avail = [math.ceil(((st + n - 1) * HOP + HALF_FFT) / SR / PACKET) * PACKET for st, n in zip(starts, sizes)]
    return starts, sizes, avail


def chunk_of_time(t: float, lookahead: int) -> int:
    """Chunk whose encoder frames cover audio time t (encoder frame e uses mel frames up to 8e)."""
    e = math.ceil(math.floor(t * 100) / 8)
    return e // (lookahead + 1)


def run(backend, sessions, args, label):
    first_n, later_n = backend.first_chunk_frames(), backend.chunk_frames()
    plans = {s.sid: chunk_plan(s, first_n, later_n, args.lookahead + 1) for s in sessions}
    pending = sorted(sessions, key=lambda s: s.arrival)
    active = {}  # sid -> dict(slot, next, tokens_by_chunk)
    chunk_log, rejected = [], 0
    gpu = GpuSampler()
    gpu.start()
    t0 = time.perf_counter() + 1.0
    pi = 0
    end_time = args.duration + 5.0  # measurement window ends at args.duration; stragglers after it are dropped
    while True:
        now = time.perf_counter() - t0
        if now > end_time:
            break
        while pi < len(pending) and pending[pi].arrival <= now:
            s = pending[pi]
            pi += 1
            if not backend_has_slot(backend):
                rejected += 1
                continue
            active[s.sid] = {"s": s, "slot": backend.open(), "next": 0, "emitted": {}}
        ready = []
        next_due = math.inf
        for sid, st in active.items():
            starts, sizes, avail = plans[sid]
            if st["next"] >= len(starts):
                continue
            due = st["s"].arrival + avail[st["next"]]
            if due <= now:
                ready.append((due, st["next"] == 0, sid))
            else:
                next_due = min(next_due, due)
        if not ready:
            if pi >= len(pending) and not any(st["next"] < len(plans[sid][0]) for sid, st in active.items()):
                break
            nxt = min(next_due, pending[pi].arrival if pi < len(pending) else math.inf)
            wait = nxt - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(min(wait, 0.002))
            continue
        ready.sort()
        phase = ready[0][1]
        batch = [r for r in ready if r[1] == phase][: args.max_batch]
        sids = [r[2] for r in batch]
        slots = [active[sid]["slot"] for sid in sids]
        chunks = []
        for sid in sids:
            st = active[sid]
            starts, sizes, _ = plans[sid]
            chunks.append((st["s"].mel, starts[st["next"]], sizes[st["next"]]))
        t_start = time.perf_counter() - t0
        new = backend.step(slots, chunks)
        t_done = time.perf_counter() - t0
        for (due, _, sid), toks in zip(batch, new):
            st = active[sid]
            c = st["next"]
            st["emitted"][c] = toks
            chunk_log.append((sid, c, due, t_start, t_done, len(batch)))
            st["next"] += 1
            if st["next"] >= len(plans[sid][0]):
                backend.close(st["slot"])
                st["done"] = True
        for sid in [sid for sid, st in active.items() if st.get("done")]:
            st = active.pop(sid)
            finished[sid] = st["emitted"]
    gpu.stop_flag = True
    for sid, st in active.items():  # still running at the end of the window
        finished[sid] = st["emitted"]
        backend.close(st["slot"])
    return chunk_log, rejected, gpu.samples, t0


def backend_has_slot(backend):
    free = getattr(backend, "free", None) or getattr(getattr(backend, "engine", None), "free", None)
    return bool(free)


finished: dict = {}


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarize(backend, sessions, chunk_log, rejected, gpu_samples, t0, args, label):
    import jiwer

    by_sid = {s.sid: s for s in sessions}
    log = np.array([(c[2], c[3], c[4], c[5]) for c in chunk_log]) if chunk_log else np.zeros((0, 4))
    w0, w1 = args.warmup, args.duration
    sel = (log[:, 0] >= w0) & (log[:, 0] < w1) if len(log) else np.zeros(0, bool)
    lag = (log[sel, 2] - log[sel, 0]) * 1000
    done_at = {(c[0], c[1]): (c[2], c[4]) for c in chunk_log}  # (sid, chunk) -> (avail_abs, done)
    fin, fin_e2e, refs, hyps, missing = [], [], [], [], 0
    for sid, emitted in finished.items():
        s = by_sid[sid]
        last = max(emitted, default=-1) + 1
        cut = [chunk_of_time(t.start, args.lookahead) for t in s.turns] + [last]
        for i, t in enumerate(s.turns):
            end_abs = s.arrival + t.end
            if not (w0 <= end_abs < w1):
                continue
            c_end = chunk_of_time(t.end, args.lookahead)
            if (sid, c_end) not in done_at:
                missing += 1
                continue
            avail_abs, done = done_at[(sid, c_end)]
            fin.append((done - avail_abs) * 1000)
            fin_e2e.append((done - end_abs) * 1000)
            if t.reference is not None:
                toks = [tok for c in range(cut[i], cut[i + 1]) for tok in emitted.get(c, [])]
                refs.append(t.reference)
                hyps.append(strip_tags(backend.decode(toks)))
    from common import normalize_for_wer

    nrefs = [normalize_for_wer(r) for r in refs]
    nhyps = [normalize_for_wer(h) for h in hyps]
    keep = [i for i, r in enumerate(nrefs) if r]
    wer = jiwer.wer([nrefs[i] for i in keep], [nhyps[i] for i in keep]) * 100 if keep else float("nan")
    util = [u for t, u in gpu_samples if w0 <= t - t0 < w1]
    conc = active_streams(sessions, w0, w1)
    chunk_s = (args.lookahead + 1) * 0.08
    res = {
        "label": label, "backend": backend.name, "mode": args.mode, "lookahead": args.lookahead,
        "chunk_ms": chunk_s * 1000, "dtype": args.dtype, "max_batch": args.max_batch,
        "target_concurrency": getattr(args, "level", None), "mean_concurrency": conc,
        "sessions": len(sessions), "rejected": rejected,
        "chunks": int(sel.sum()), "chunks_per_s": float(sel.sum() / max(w1 - w0, 1e-9)),
        "mean_batch": float(log[sel, 3].mean()) if sel.any() else None,
        "lag_ms": {q: pct(lag, q) for q in (50, 95, 99)}, "lag_max_ms": float(lag.max()) if len(lag) else None,
        "late_chunk_frac": float((lag > chunk_s * 1000).mean()) if len(lag) else None,
        "finalize_ms": {q: pct(fin, q) for q in (50, 95, 99)},
        "finalize_e2e_ms": {q: pct(fin_e2e, q) for q in (50, 95, 99)},
        "turns": len(fin), "turns_missing": missing, "wer": wer, "wer_turns": len(refs),
        "gpu_util_mean": float(np.mean(util)) if util else None,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    res["slo_ok"] = bool(res["finalize_ms"][95] <= args.slo_ms and (res["lag_ms"][95] <= chunk_s * 1000) and rejected == 0)
    return res, log


def active_streams(sessions, w0, w1):
    grid = np.arange(w0, w1, 0.5)
    return float(np.mean([sum(s.arrival <= t < s.arrival + s.duration for s in sessions) for t in grid]))


def timeline(log, sessions, bin_s=1.0):
    """Per-second p95 lag, chunk count and live streams, for burst plots."""
    if not len(log):
        return []
    t_end = log[:, 0].max()
    rows = []
    for b in np.arange(0, t_end, bin_s):
        m = (log[:, 0] >= b) & (log[:, 0] < b + bin_s)
        lag = (log[m, 2] - log[m, 0]) * 1000
        rows.append({"t": float(b), "chunks": int(m.sum()), "lag_p95_ms": pct(lag, 95),
                     "lag_max_ms": float(lag.max()) if len(lag) else None,
                     "streams": int(sum(s.arrival <= b < s.arrival + s.duration for s in sessions))})
    return rows


def build_sessions(arrivals, rng, timing, pool, args, frontend):
    sessions = [make_session(i, a, rng, timing, pool, args.session_s) for i, a in enumerate(arrivals)]
    compute_mel(sessions, frontend)
    return sessions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["lite", "nemo"], required=True)
    p.add_argument("--mode", choices=["sweep", "burst"], default="sweep")
    p.add_argument("--levels", type=int, nargs="+", default=[16, 32, 64, 96, 128, 192, 256])
    p.add_argument("--lookahead", type=int, default=3)
    p.add_argument("--session-s", type=float, default=60.0, help="length of each session")
    p.add_argument("--warmup", type=float, default=60.0)
    p.add_argument("--duration", type=float, default=150.0, help="arrivals stop and measurement ends here")
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-streams", type=int, default=512)
    p.add_argument("--slo-ms", type=float, default=300.0)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--matmul-precision", default="high")
    p.add_argument("--pool-size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--burst-base", type=int, default=32)
    p.add_argument("--burst-sessions", type=int, default=64)
    p.add_argument("--burst-at", type=float, default=90.0)
    p.add_argument("--burst-spread", type=float, default=5.0)
    p.add_argument("--stop-after-fail", type=int, default=1, help="sweep: stop after this many levels miss the SLO")
    p.add_argument("--telephone", action="store_true", help="pass the audio through an 8 kHz band")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    from common import load_rows_with_audio
    from huggingface_hub import hf_hub_download
    from nemotron_lite.config import NemotronLiteConfig
    from nemotron_lite.features import LogMelFeatures

    tp = time.perf_counter()
    rows, _ = load_rows_with_audio(args.pool_size, 42)
    pool = AudioPool(rows, telephone=args.telephone)
    print(f"=== audio pool: {len(rows)} utterances in {time.perf_counter() - tp:.0f}s", flush=True)
    timing = TimingModel()
    cfg_dir = Path(hf_hub_download("nvidia/nemotron-3.5-asr-streaming-0.6b", "config.json")).parent
    hf_hub_download("nvidia/nemotron-3.5-asr-streaming-0.6b", "processor_config.json")
    frontend = LogMelFeatures(NemotronLiteConfig.from_dir(cfg_dir)).cuda()

    from backends import make_backend

    backend = make_backend(args.backend, lookahead=args.lookahead, max_streams=args.max_streams,
                           language="vi-VN", dtype=dtype, max_batch=args.max_batch)
    print(f"backend={backend.name} chunk mel frames first={backend.first_chunk_frames()} later={backend.chunk_frames()}", flush=True)

    # short warm-up so kernels/allocator are hot before the clock starts
    warm = build_sessions([0.0] * 4, np.random.default_rng(0), timing, pool, args, frontend)
    sl = [backend.open() for _ in warm]
    for k in range(5):
        n = backend.first_chunk_frames() if k == 0 else backend.chunk_frames()
        st = 0 if k == 0 else backend.first_chunk_frames() + (k - 1) * backend.chunk_frames()
        backend.step(sl, [(s.mel, st, n) for s in warm])
    for x in sl:
        backend.close(x)
    torch.cuda.synchronize()

    levels = args.levels if args.mode == "sweep" else [args.burst_base]
    results, fails = [], 0
    for level in levels:
        args.level = level
        rng = np.random.default_rng(args.seed + level)
        if args.mode == "sweep":
            arrivals = open_arrivals(rng, level, args.session_s, args.duration)
            label = f"{backend.name}_open_c{level}"
        else:
            arrivals = burst_arrivals(rng, level, args.session_s, args.duration, args.burst_at,
                                      args.burst_sessions, args.burst_spread)
            label = f"{backend.name}_burst_c{level}_b{args.burst_sessions}"
        tb = time.perf_counter()
        sessions = build_sessions(arrivals, rng, timing, pool, args, frontend)
        t_build = time.perf_counter() - tb
        finished.clear()
        torch.cuda.reset_peak_memory_stats()
        print(f"=== {label}: {len(sessions)} sessions", flush=True)
        tr = time.perf_counter()
        log_rows, rejected, gpu_samples, t0 = run(backend, sessions, args, label)
        t_run = time.perf_counter() - tr
        ts = time.perf_counter()
        res, log = summarize(backend, sessions, log_rows, rejected, gpu_samples, t0, args, label)
        res["wall_s"] = {"build": t_build, "run": t_run, "summarize": time.perf_counter() - ts}
        print(f"=== {label}: build {t_build:.0f}s run {t_run:.0f}s summarize {res['wall_s']['summarize']:.0f}s", flush=True)
        if args.mode == "burst":
            res["timeline"] = timeline(log, sessions)
            res["burst"] = {"at": args.burst_at, "sessions": args.burst_sessions, "spread": args.burst_spread}
        print(json.dumps({k: v for k, v in res.items() if k != "timeline"}), flush=True)
        (out / f"{label}.json").write_text(json.dumps(res, indent=1))
        np.save(out / f"{label}_chunks.npy", log)
        results.append(res)
        if args.mode == "sweep":
            fails = fails + 1 if not res["slo_ok"] else 0
            if fails >= args.stop_after_fail:
                print("stopping sweep: SLO missed at consecutive levels", flush=True)
                break
    (out / f"{backend.name}_{args.mode}_summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
