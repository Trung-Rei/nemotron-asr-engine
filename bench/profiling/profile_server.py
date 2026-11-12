"""Run the nemotron-engine WebSocket server with the profile_engine.py instrumentation, to see what the server's
event loop costs the GPU worker. Adds two endpoints that ws_loadtest.py --prof-url calls around its window:

    POST /prof/start   reset and start measuring
    POST /prof/stop    stop; returns worker phases per batch (wall, blocked = wall - thread CPU), GPU busy / idle
                       from CUDA events, CPU % of the worker and of the event-loop thread, event-loop lag

    python bench/profiling/profile_server.py --port 8091
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_engine as pe  # noqa: E402
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

from nemotron_engine import server  # noqa: E402

app = server.app
prof = pe.Prof()
win: dict = {}
loop_lag: list[float] = []


async def ticker() -> None:
    while True:
        a = time.perf_counter()
        await asyncio.sleep(0.01)
        if prof.on:
            loop_lag.append((time.perf_counter() - a - 0.01) * 1000)


_orig = app.router.lifespan_context


@asynccontextmanager
async def lifespan(a):
    async with _orig(a):
        pe.instrument(a.state.engine, prof)
        task = asyncio.create_task(ticker())
        yield
        task.cancel()


app.router.lifespan_context = lifespan


@app.post("/prof/start")
async def prof_start() -> dict:
    eng = app.state.engine
    prof.reset()
    loop_lag.clear()
    win.update(tids={"worker": eng._thread.native_id, "loop": threading.get_native_id()},
               t0=time.perf_counter(), p0=time.process_time(), s0=eng.stats())
    win["cpu0"] = {k: pe.thread_cpu_s(v) for k, v in win["tids"].items()}
    prof.on = True
    return {"ok": True}


@app.post("/prof/stop")
async def prof_stop() -> dict:
    prof.on = False
    eng = app.state.engine
    T = time.perf_counter() - win["t0"]
    s1 = eng.stats()
    n = prof.n["finish"] or 1
    res = {
        "window_s": T, "batches_per_s": prof.n["finish"] / T,
        "chunks_per_batch": (s1["chunks"] - win["s0"]["chunks"]) / n,
        "thread_cpu_pct": {k: 100 * (pe.thread_cpu_s(v) - win["cpu0"][k]) / T for k, v in win["tids"].items()},
        "process_cpu_pct": 100 * (time.process_time() - win["p0"]) / T,
        "loop_lag_ms": {q: float(np.percentile(loop_lag, q)) if loop_lag else None for q in (50, 95, 99)},
        "phases_per_batch": {p: {"calls": prof.n[p] / n, "ms": 1000 * prof.wall[p] / n, "blocked_ms": 1000 * (prof.wall[p] - prof.cpu[p]) / n}
                             for p in pe.PHASES},
    }
    torch.cuda.synchronize()
    b = prof.batches
    if len(b) > 1:
        busy = sum(x[0].elapsed_time(x[2]) for x in b)
        span = b[0][0].elapsed_time(b[-1][2])
        gap = sum(b[i][2].elapsed_time(b[i + 1][0]) for i in range(len(b) - 1))
        res["gpu"] = {"busy_pct": 100 * busy / span, "idle_pct": 100 * gap / span, "idle_ms_per_batch": gap / (len(b) - 1),
                      "step_ms": busy / len(b)}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    args = p.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
