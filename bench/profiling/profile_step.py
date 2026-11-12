#!/usr/bin/env python3
"""Where does one batched streaming step of the lite engine spend its time?

Builds streaming sessions (synthetic utterance timing + VietSuperSpeech audio, as in loadtest.py), then for each batch
size runs consecutive 320 ms chunks through BatchedStreamEngine.step and reports:

  1. wall time per step, split into host input prep (stack + H2D copy) and the engine step itself
  2. per-section GPU time (CUDA events) and CPU time (perf_counter) inside the step
  3. greedy RNN-T counters: frames, symbol loops, tokens, host syncs
  4. torch.profiler: sum of GPU kernel time vs wall (GPU busy %), top ops, a chrome trace for the largest batch

  python bench/profiling/profile_step.py --out results/profile
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from workload import AudioPool, TimingModel, make_session  # noqa: E402


def chunk_start(k: int, first_n: int, later_n: int) -> int:
    return 0 if k == 0 else first_n + (k - 1) * later_n


def kernel_time_us(prof) -> float:
    total = 0.0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            total += e.device_time if hasattr(e, "device_time") else e.cuda_time
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32, 64, 96, 128])
    p.add_argument("--steps", type=int, default=30, help="timed steps per batch size")
    p.add_argument("--warm", type=int, default=4, help="untimed chunks before timing (incl. the first chunk)")
    p.add_argument("--lookahead", type=int, default=3)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--matmul-precision", default="high")
    p.add_argument("--pool-size", type=int, default=300)
    p.add_argument("--trace-batch", type=int, default=64)
    p.add_argument("--parity-streams", type=int, default=64)
    p.add_argument("--configs", nargs="+", default=None, help="subset of config names (first = parity reference)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    from common import load_rows_with_audio
    from loadtest import compute_mel
    from nemotron_lite import NemotronLite
    from nemotron_lite.batch_stream import BatchedStreamEngine

    rows, _ = load_rows_with_audio(args.pool_size, 42)
    pool = AudioPool(rows)
    timing = TimingModel()
    model = NemotronLite.from_pretrained(device="cuda", dtype=dtype)
    max_b = max(max(args.batches), args.parity_streams)
    rng = np.random.default_rng(3)
    sessions = [make_session(i, 0.0, rng, timing, pool, 40.0) for i in range(max_b)]
    compute_mel(sessions, model.preprocessor)
    first_n = 1 + model.cfg.subsampling_factor * args.lookahead
    later_n = model.cfg.subsampling_factor * (args.lookahead + 1)
    n_chunks = args.warm + args.steps
    print(f"gpu={torch.cuda.get_device_name(0)} dtype={args.dtype} matmul={args.matmul_precision} "
          f"chunk mel frames first={first_n} later={later_n} sessions={len(sessions)}", flush=True)

    from nemotron_lite.encoder import ConvModule

    def make_engine(cfg):
        ConvModule.use_linear = cfg["linear"]
        torch.cuda.empty_cache()
        e = BatchedStreamEngine(model, args.lookahead, max_b, "vi-VN", cuda_graphs=cfg["graphs"],
                                fast_greedy=cfg["greedy"], max_batch=max_b,
                                fused_attn=cfg.get("fused", False))
        if cfg["graphs"]:
            e.capture_graphs()
        return e

    def prep(sess, k):
        n = first_n if k == 0 else later_n
        st = chunk_start(k, first_n, later_n)
        t0 = time.perf_counter()
        cpu = torch.stack([s.mel[st : st + n] for s in sess])
        t1 = time.perf_counter()
        gpu = cpu.to("cuda", non_blocking=True)
        torch.cuda.current_stream().synchronize()  # the copy is part of what the scheduler waits for
        t2 = time.perf_counter()
        return gpu, (t1 - t0) * 1e3, (t2 - t1) * 1e3

    def run_stream(engine, k_range, slots, sess, sink):
        for k in k_range:
            mel, t_stack, t_h2d = prep(sess, k)
            t0 = time.perf_counter()
            engine.step(slots, mel)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if sink is not None:
                sink.append({"stack_ms": t_stack, "h2d_ms": t_h2d, "step_ms": (t1 - t0) * 1e3})

    def transcripts(engine, n_streams, n_chunk):
        """Full streams in one batch; tokens per stream."""
        sess = sessions[:n_streams]
        slots = [engine.open() for _ in sess]
        run_stream(engine, range(n_chunk), slots, sess, None)
        out = [list(engine.tokens[s]) for s in slots]
        for x in slots:
            engine.close(x)
        return out

    def agreement(ref, hyp):
        import difflib
        same = sum(a == b for a, b in zip(ref, hyp))
        sm = [difflib.SequenceMatcher(a=a, b=b, autojunk=False) for a, b in zip(ref, hyp)]
        n_ref = sum(len(a) for a in ref)
        diff = sum(max(len(a), len(b)) - sum(m.size for m in s.get_matching_blocks()) for a, b, s in zip(ref, hyp, sm))
        return {"streams_identical": same, "streams": len(ref), "tokens_ref": n_ref, "tokens_differing": diff}

    configs = [
        {"name": "baseline", "linear": False, "graphs": False, "greedy": False},
        {"name": "+pointwise_linear", "linear": True, "graphs": False, "greedy": False},
        {"name": "+cuda_graph", "linear": True, "graphs": True, "greedy": False},
        {"name": "+fast_greedy", "linear": True, "graphs": True, "greedy": True},
        {"name": "+fused_attn", "linear": True, "graphs": True, "greedy": True, "fused": True},
    ]
    if args.configs:
        configs = [c for n in args.configs for c in configs if c["name"] == n]

    # ---------------------------------------------------------------- parity (strict fp32, then TF32)
    from common import set_strict_fp32

    parity = {}
    n_par_chunks = int((40.0 - 1.0) / ((args.lookahead + 1) * 0.08))
    for mode in ("strict_fp32", args.matmul_precision):
        if mode == "strict_fp32":
            set_strict_fp32()
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision(args.matmul_precision)
        ref = None
        for cfg in configs:
            engine = make_engine(cfg)
            toks = transcripts(engine, args.parity_streams, n_par_chunks)
            del engine
            if ref is None:
                ref = toks
            a = agreement(ref, toks)
            parity.setdefault(mode, {})[cfg["name"]] = a
            print(f"parity[{mode}] {cfg['name']:18s} identical streams {a['streams_identical']}/{a['streams']}  "
                  f"tokens differing {a['tokens_differing']}/{a['tokens_ref']}", flush=True)
    (out / "parity.json").write_text(json.dumps(parity, indent=1))

    # ---------------------------------------------------------------- timing (matmul precision as in the load test)
    results = []
    for cfg in configs:
        engine = make_engine(cfg)
        greedy = model.greedy
        sl = [engine.open() for _ in range(max_b)]
        run_stream(engine, range(3), sl, sessions[:max_b], None)  # warm-up
        for x in sl:
            engine.close(x)
        for b in args.batches:
            sess = sessions[:b]
            slots = [engine.open() for _ in range(b)]
            run_stream(engine, range(args.warm), slots, sess, None)
            plain = []
            run_stream(engine, range(args.warm, n_chunks), slots, sess, plain)
            for x in slots:
                engine.close(x)

            slots = [engine.open() for _ in range(b)]
            run_stream(engine, range(args.warm), slots, sess, None)
            sec_gpu, sec_cpu = defaultdict(float), defaultdict(float)
            greedy.counters = {}
            for k in range(args.warm, n_chunks):
                mel, _, _ = prep(sess, k)
                engine.sections = []
                engine.step(slots, mel)
                torch.cuda.synchronize()
                for name, s_, e_, c0, c1 in engine.sections:
                    sec_gpu[name] += s_.elapsed_time(e_)
                    sec_cpu[name] += (c1 - c0) * 1e3
                engine.sections = None
            counters = greedy.counters
            greedy.counters = None
            for x in slots:
                engine.close(x)

            slots = [engine.open() for _ in range(b)]
            run_stream(engine, range(args.warm), slots, sess, None)
            acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=acts) as prof:
                run_stream(engine, range(args.warm, n_chunks), slots, sess, None)
            for x in slots:
                engine.close(x)
            kern_ms = kernel_time_us(prof) / 1e3
            ka = prof.key_averages()
            n_launch = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch"))
            tag = cfg["name"].lstrip("+")
            if b == args.trace_batch:
                prof.export_chrome_trace(str(out / f"trace_{tag}_b{b}.json"))
                sort_dev = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
                (out / f"ops_{tag}_b{b}_gpu.txt").write_text(ka.table(sort_by=sort_dev, row_limit=40))
                (out / f"ops_{tag}_b{b}_cpu.txt").write_text(ka.table(sort_by="self_cpu_time_total", row_limit=40))

            n = args.steps
            med = lambda key: statistics.median(r[key] for r in plain)  # noqa: E731
            step_ms = med("step_ms")
            r = {
                "config": cfg["name"], "batch": b, "steps": n,
                "stack_ms": med("stack_ms"), "h2d_ms": med("h2d_ms"),
                "step_ms_p50": step_ms,
                "step_ms_p90": float(np.percentile([x["step_ms"] for x in plain], 90)),
                "chunks_per_s": b / ((step_ms + med("stack_ms") + med("h2d_ms")) / 1e3),
                "sections_gpu_ms": {k: v / n for k, v in sec_gpu.items()},
                "sections_cpu_ms": {k: v / n for k, v in sec_cpu.items()},
                "greedy_per_step": {k: v / n for k, v in counters.items()},
                "kernel_ms_per_step": kern_ms / n,
                "gpu_busy_pct": 100 * (kern_ms / n) / step_ms,  # vs the un-profiled step
                "launches_per_step": n_launch / n,
            }
            results.append(r)
            g = r["greedy_per_step"]
            print(f"{cfg['name']:18s} B={b:4d} step p50 {step_ms:7.2f} ms  {r['chunks_per_s']:7.0f} chunks/s  "
                  f"kernels {r['kernel_ms_per_step']:6.2f} ms  GPU busy {r['gpu_busy_pct']:5.1f}%  "
                  f"launches {r['launches_per_step']:6.0f}  syncs {g.get('syncs', 0):5.1f}", flush=True)
        del engine
    (out / "profile_summary.json").write_text(json.dumps({"parity": parity, "timing": results}, indent=1))

    # markdown report
    L = [f"# Lite step profile ({torch.cuda.get_device_name(0)}, {args.dtype}, matmul {args.matmul_precision}, chunk {(args.lookahead + 1) * 80} ms)", "",
         f"## Token parity vs baseline ({args.parity_streams} streams x {n_par_chunks} chunks in one batch)", "",
         "| mode | config | identical streams | differing tokens |", "|---|---|---|---|"]
    for mode, d in parity.items():
        for name, a in d.items():
            L.append(f"| {mode} | {name} | {a['streams_identical']}/{a['streams']} | {a['tokens_differing']}/{a['tokens_ref']} |")
    L += ["", "## Step time", "",
          "| config | batch | step p50 ms | step p90 ms | chunks/s | GPU kernel ms/step | GPU busy % | launches/step | greedy syncs/step |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['config']} | {r['batch']} | {r['step_ms_p50']:.2f} | {r['step_ms_p90']:.2f} | {r['chunks_per_s']:.0f} | "
                 f"{r['kernel_ms_per_step']:.2f} | {r['gpu_busy_pct']:.0f} | {r['launches_per_step']:.0f} | "
                 f"{r['greedy_per_step'].get('syncs', 0):.1f} |")
    L += ["", "## Sections (ms per step, GPU span / CPU)", ""]
    for r in results:
        if r["batch"] in (8, 64, 128):
            parts = ", ".join(f"{k} {r['sections_gpu_ms'][k]:.2f}/{r['sections_cpu_ms'][k]:.2f}" for k in r["sections_gpu_ms"])
            L.append(f"- {r['config']} B={r['batch']}: {parts}")
    (out / "profile_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
