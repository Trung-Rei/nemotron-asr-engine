"""Real-time streaming load test through the nemotron-engine WebSocket server, with a memory monitor.

Same workload as engine_loadtest.py (Poisson session arrivals, synthetic utterance timing, VietSuperSpeech audio,
16 kHz pcm16 packets on the wall clock, end_turn when an utterance ends), but every session is a WebSocket to the
server, and the sessions are spread over --procs client processes so the client is not the bottleneck.

Metrics per level (measurement window [warmup, duration]):
  finalize     final received by the client - end_turn sent by the client (includes the socket hop)
  chunk lag    server-side t_emit - t_ready of every event (chunks that produced tokens)
  push late    actual send time - scheduled packet time (client health)
Memory: a monitor polls GET /health (PyTorch allocated / reserved, process RSS) and nvidia-smi every second during
the whole run, including idle gaps between levels, so growth shows as a rising baseline.

    python bench/streaming/ws_loadtest.py --url ws://127.0.0.1:8091/v1/stream --levels 256 448 512 576 --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from workload import SR, AudioPool, TimingModel, make_session, open_arrivals, strip_tags  # noqa: E402

SESSIONS: list = []  # the current level's sessions, shared with the forked client processes


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


# ----------------------------------------------------------------------------- client process
async def run_sessions(sessions, t0: float, args) -> dict:
    import websockets

    pkt = SR * args.packet_ms // 1000
    url = f"{args.url}?sample_rate={SR}&encoding=pcm16&language=vi-VN"
    ends, lat, late, rejected = [], [], [], [0]

    async def run_session(s):
        await asyncio.sleep(max(0.0, t0 + s.arrival - time.monotonic()))
        try:
            ws = await websockets.connect(url, max_size=None, ping_interval=None)
        except Exception:  # noqa: BLE001
            rejected[0] += 1
            return
        finals, closed = [], {}

        async def reader():
            try:
                async for raw in ws:
                    ev = json.loads(raw)
                    if ev["type"] in ("partial", "final"):
                        lat.append((time.monotonic() - t0, ev["latency_ms"]))
                    if ev["type"] == "final":
                        finals.append((time.monotonic(), ev["text"]))
                    elif ev["type"] == "closed":
                        return
            except websockets.ConnectionClosed as e:
                closed["code"] = e.rcvd.code if e.rcvd else None

        rd = asyncio.create_task(reader())
        start = t0 + s.arrival
        turn_ends = [int(t.end * SR) for t in s.turns]
        ti, my_ends = 0, []
        end_at = min(len(s.audio), int((args.duration + 5.0 - s.arrival) * SR))
        try:
            for i in range(0, max(0, end_at), pkt):
                due = start + (i + pkt) / SR
                d = due - time.monotonic()
                if d > 0:
                    await asyncio.sleep(d)
                late.append((time.monotonic() - due) * 1000)
                await ws.send((np.clip(s.audio[i : i + pkt], -1, 1) * 32767).astype("<i2").tobytes())
                while ti < len(turn_ends) and turn_ends[ti] <= i + pkt:
                    await ws.send('{"type": "end_turn"}')
                    my_ends.append((time.monotonic(), ti))
                    ti += 1
            await ws.send('{"type": "close"}')
        except websockets.ConnectionClosed:
            pass
        await rd
        await ws.close()
        if closed.get("code") == 1013:
            rejected[0] += 1
            return
        for k, (t_end, turn) in enumerate(my_ends):
            if k < len(finals):
                t_recv, text = finals[k]
                ends.append((t_end - t0, (t_recv - t_end) * 1000, s.turns[turn].reference, text))

    await asyncio.gather(*(run_session(s) for s in sessions))
    return {"ends": ends, "lat": lat, "late": late, "rejected": rejected[0]}


def client_proc(k: int, t0: float, args, path: str) -> None:
    res = asyncio.run(run_sessions(SESSIONS[k :: args.procs], t0, args))
    Path(path).write_text(json.dumps(res))


# ----------------------------------------------------------------------------- monitor
class Monitor(threading.Thread):
    def __init__(self, health_url: str, period: float = 1.0):
        super().__init__(daemon=True)
        self.url, self.period, self.t0 = health_url, period, time.monotonic()
        self.samples, self.label, self.stop = [], "idle", threading.Event()

    def run(self):
        while not self.stop.is_set():
            row = {"t": time.monotonic() - self.t0, "label": self.label}
            try:
                h = json.loads(urllib.request.urlopen(self.url, timeout=2).read())
                row.update({k: h.get(k) for k in ("gpu_allocated_mb", "gpu_reserved_mb", "rss_mb", "active", "lag_ms", "free_slots")})
            except Exception as e:  # noqa: BLE001
                row["health_error"] = repr(e)
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.split()
                row["nvsmi_used_mb"] = float(out[0]) if out else None
            except Exception:  # noqa: BLE001
                row["nvsmi_used_mb"] = None
            self.samples.append(row)
            self.stop.wait(self.period)


def health(url: str) -> dict:
    return json.loads(urllib.request.urlopen(url, timeout=5).read())


# ----------------------------------------------------------------------------- main
def run_level(level: int, args, pool, timing, out: Path) -> dict:
    global SESSIONS
    rng = np.random.default_rng(args.seed + level)
    arrivals = open_arrivals(rng, level, args.session_s, args.duration)
    SESSIONS = [make_session(i, a, rng, timing, pool, args.session_s) for i, a in enumerate(arrivals)]
    t0 = time.monotonic() + 2.0
    prof: dict = {}
    timers = []
    if args.prof_url:  # server-side profile over the same window (bench/profiling/profile_server.py)
        post = lambda path: json.loads(urllib.request.urlopen(urllib.request.Request(args.prof_url + path, method="POST"), timeout=60).read())  # noqa: E731
        timers = [threading.Timer(t0 + args.warmup - time.monotonic(), lambda: post("/prof/start")),
                  threading.Timer(t0 + args.duration - time.monotonic(), lambda: prof.update(post("/prof/stop")))]
        for t in timers:
            t.start()
    ctx = mp.get_context("fork")
    paths = [str(out / f"tmp_{level}_{k}.json") for k in range(args.procs)]
    procs = [ctx.Process(target=client_proc, args=(k, t0, args, paths[k])) for k in range(args.procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    parts = [json.loads(Path(p).read_text()) for p in paths]
    for p in paths:
        Path(p).unlink()
    ends = [e for r in parts for e in r["ends"]]
    lat = [e for r in parts for e in r["lat"]]
    late = [x for r in parts for x in r["late"]]
    rejected = sum(r["rejected"] for r in parts)
    for t in timers:
        t.join()
    res = summarize(level, args, len(SESSIONS), ends, lat, late, rejected)
    if prof:
        res["server_profile"] = prof
    return res


def summarize(level, args, calls, ends, lat, late, rejected) -> dict:
    import jiwer
    from common import normalize_for_wer

    w0, w1 = args.warmup, args.duration
    fe = [e for e in ends if w0 <= e[0] < w1]
    fin = [e[1] for e in fe]
    lg = [x for t, x in lat if w0 <= t < w1]
    pairs = [(normalize_for_wer(strip_tags(r)), normalize_for_wer(h)) for _, _, r, h in fe if r]
    pairs = [(r, h) for r, h in pairs if r]
    chunk_ms = (args.lookahead + 1) * 80
    res = {"level": level, "sessions": calls, "rejected": rejected, "turns": len(fe),
           "finalize_ms": {q: pct(fin, q) for q in (50, 95, 99)}, "chunk_lag_ms": {q: pct(lg, q) for q in (50, 95, 99)},
           "push_late_ms": {q: pct(late, q) for q in (50, 95, 99)},
           "wer": 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]) if pairs else None}
    res["slo_ok"] = bool(fin and res["finalize_ms"][95] <= args.slo_ms and lg and res["chunk_lag_ms"][95] <= chunk_ms and rejected == 0)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8091/v1/stream")
    p.add_argument("--levels", type=int, nargs="+", default=[256, 448, 512, 576])
    p.add_argument("--packet-ms", type=int, default=20)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--lookahead", type=int, default=3)
    p.add_argument("--session-s", type=float, default=40.0)
    p.add_argument("--warmup", type=float, default=40.0)
    p.add_argument("--duration", type=float, default=90.0)
    p.add_argument("--gap-s", type=float, default=20.0, help="idle time between levels (memory baseline)")
    p.add_argument("--slo-ms", type=float, default=300.0)
    p.add_argument("--pool-size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prof-url", default=None, help="http://host:port of profile_server.py to profile the server's window")
    p.add_argument("--telephone", action="store_true", help="pass the audio through an 8 kHz band")
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from common import load_rows_with_audio

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    health_url = args.url.replace("ws://", "http://").split("/v1/")[0] + "/health"
    rows, _ = load_rows_with_audio(args.pool_size, 42)
    pool, timing = AudioPool(rows, telephone=args.telephone), TimingModel()
    mon = Monitor(health_url)
    mon.start()
    time.sleep(args.gap_s)
    results = []
    for level in args.levels:
        mon.label = f"c{level}"
        print(f"=== level {level}", flush=True)
        r = run_level(level, args, pool, timing, out)
        mon.label = "idle"
        time.sleep(args.gap_s)
        r["health_after"] = health(health_url)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    mon.stop.set()
    mon.join()
    out = out / args.tag if args.tag else out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    (out / "memory_timeline.json").write_text(json.dumps(mon.samples))

    f = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
    lines = ["| level | sessions | rej | finalize p50/p95/p99 ms | chunk lag p50/p95 | push late p95 | WER | SLO |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['level']} | {r['sessions']} | {r['rejected']} | {f(r['finalize_ms'][50])}/{f(r['finalize_ms'][95])}/{f(r['finalize_ms'][99])} | "
                     f"{f(r['chunk_lag_ms'][50])}/{f(r['chunk_lag_ms'][95])} | {f(r['push_late_ms'][95])} | {'-' if r['wer'] is None else format(r['wer'], '.1f')} | {'✅' if r['slo_ok'] else '❌'} |")
    lines += ["", "| phase | samples | GPU allocated MB min/max | GPU reserved MB min/max | nvidia-smi used MB min/max | RSS MB min/max |", "|---|---|---|---|---|---|"]
    phases: dict[str, list] = {}
    for i, s in enumerate(mon.samples):  # consecutive runs of the same label
        key = f"{len(phases):02d} {s['label']}" if not phases or list(phases)[-1].split(" ", 1)[1] != s["label"] else list(phases)[-1]
        phases.setdefault(key, []).append(s)
    mm = lambda xs, k: "-" if not [x[k] for x in xs if x.get(k) is not None] else f"{min(x[k] for x in xs if x.get(k) is not None):.0f}/{max(x[k] for x in xs if x.get(k) is not None):.0f}"  # noqa: E731
    for k, xs in phases.items():
        lines.append(f"| {k} | {len(xs)} | {mm(xs, 'gpu_allocated_mb')} | {mm(xs, 'gpu_reserved_mb')} | {mm(xs, 'nvsmi_used_mb')} | {mm(xs, 'rss_mb')} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
