"""Smoke test of the nemotron-engine WebSocket server (start it first, e.g. with NEMOTRON_MAX_STREAMS=128).

1. GET /health
2. --n VietSuperSpeech utterances as concurrent 8 kHz mu-law streams in real time: WER and client-side finalize
3. fill every free slot with idle sockets; one more must be closed with code 1013

    python bench/streaming/ws_smoke.py --url ws://127.0.0.1:8091/v1/stream --n 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from ws_client import stream_one  # noqa: E402


def health(url: str) -> dict:
    base = url.replace("ws://", "http://").split("/v1/")[0]
    return json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())


async def overflow(url: str, free: int) -> str:
    import websockets

    held = [await websockets.connect(url + "?sample_rate=8000&encoding=mulaw") for _ in range(free)]
    await asyncio.sleep(0.5)
    extra = await websockets.connect(url + "?sample_rate=8000&encoding=mulaw")
    try:
        await asyncio.wait_for(extra.recv(), 10)
        result = "no close"
    except websockets.ConnectionClosed as e:
        result = f"closed {e.rcvd.code if e.rcvd else None}: {e.rcvd.reason if e.rcvd else ''}"
    for ws in held:
        await ws.send(json.dumps({"type": "close"}))
        await ws.close()
    return result


async def main_async(args):
    import jiwer
    import soundfile as sf
    from common import load_rows_with_audio, normalize_for_wer
    from scipy.signal import resample_poly

    print("health:", health(args.url), flush=True)
    rows, _ = load_rows_with_audio(args.n, 42)
    audio = []
    for r in rows:
        a, sr = sf.read(r["local_audio"], dtype="float32", always_2d=True)
        audio.append(resample_poly(a.mean(axis=1), 1, 2).astype(np.float32))  # 16 kHz -> 8 kHz
    res = await asyncio.gather(*(stream_one(args.url, a, 8000, "mulaw", 20, language="vi-VN") for a in audio))
    errs = [r["error"] for r in res if r["error"]]
    pairs = [(normalize_for_wer(str(r0["text"])), normalize_for_wer(r["text"] or "")) for r0, r in zip(rows, res)]
    pairs = [(a, b) for a, b in pairs if a]
    lat = [r["finalize_ms"] for r in res if r["finalize_ms"] is not None]
    out = {"streams": len(res), "errors": errs[:3], "n_errors": len(errs),
           "wer": 100 * jiwer.wer([a for a, _ in pairs], [b for _, b in pairs]),
           "finalize_ms_p50_p95_max": [float(np.percentile(lat, 50)), float(np.percentile(lat, 95)), max(lat)] if lat else None,
           "partials_per_stream": float(np.mean([r["partials"] for r in res]))}
    print("stream:", json.dumps(out, ensure_ascii=False), flush=True)
    print("example:", res[0]["text"], flush=True)
    await asyncio.sleep(1.0)
    h = health(args.url)
    print("overflow:", await overflow(args.url, h["free_slots"]), flush=True)
    await asyncio.sleep(1.0)
    h = health(args.url)
    print("health after:", h, flush=True)
    assert not errs and h["active"] == 0, (errs, h)
    print("SMOKE OK", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8091/v1/stream")
    p.add_argument("--n", type=int, default=64)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
