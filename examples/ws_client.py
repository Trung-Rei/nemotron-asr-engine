"""Stream WAV files to a nemotron-asr-engine server in real time, one WebSocket per file, all files concurrently.

    pip install websockets soundfile numpy scipy
    python examples/ws_client.py a.wav b.wav --url ws://HOST:8090/v1/stream [--sample-rate 8000 --encoding mulaw]

Each file is one utterance: packets go out on the wall clock, then {"type": "end_turn"} and {"type": "close"}.
Prints the final text per file and the client-side finalize latency (end_turn sent -> final received).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np


def mulaw_encode(x: np.ndarray) -> bytes:
    """float in [-1, 1] -> G.711 mu-law bytes."""
    s = np.clip(np.round(x * 32768), -32635, 32635).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    s = np.abs(s) + 0x84
    exp = np.clip(np.floor(np.log2(s)).astype(np.int32) - 7, 0, 7)
    mant = (s >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


def encode(x: np.ndarray, encoding: str) -> bytes:
    if encoding == "mulaw":
        return mulaw_encode(x)
    if encoding == "pcm16":
        return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    return x.astype("<f4").tobytes()


def load(path: str, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != sample_rate:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, sample_rate)
        x = resample_poly(x, sample_rate // g, sr // g).astype(np.float32)
    return x


async def stream_one(url: str, audio: np.ndarray, sample_rate: int, encoding: str, packet_ms: int,
                     realtime: bool = True, language: str | None = None) -> dict:
    """Send one turn; returns {"text", "finalize_ms", "partials", "error"}."""
    import websockets

    q = f"?sample_rate={sample_rate}&encoding={encoding}" + (f"&language={language}" if language else "")
    out = {"text": None, "finalize_ms": None, "partials": 0, "error": None}
    n = sample_rate * packet_ms // 1000
    try:
        async with websockets.connect(url + q, max_size=None) as ws:
            t_end = None

            async def reader():
                async for raw in ws:
                    ev = json.loads(raw)
                    if ev["type"] == "partial":
                        out["partials"] += 1
                    elif ev["type"] == "final" and out["text"] is None:
                        out["text"], out["finalize_ms"] = ev["text"], (time.perf_counter() - t_end) * 1000
                    elif ev["type"] == "closed":
                        return

            rd = asyncio.create_task(reader())
            t0 = time.perf_counter()
            for i in range(0, len(audio), n):
                if realtime:
                    d = t0 + (i + n) / sample_rate - time.perf_counter()
                    if d > 0:
                        await asyncio.sleep(d)
                await ws.send(encode(audio[i : i + n], encoding))
            t_end = time.perf_counter()
            await ws.send(json.dumps({"type": "end_turn"}))
            await ws.send(json.dumps({"type": "close"}))
            await rd
    except Exception as e:  # noqa: BLE001 - report per stream (e.g. 1013 engine full)
        out["error"] = repr(e)
    return out


async def main_async(args) -> None:
    audios = [load(p, args.sample_rate) for p in args.files]
    res = await asyncio.gather(*(stream_one(args.url, a, args.sample_rate, args.encoding, args.packet_ms,
                                            not args.fast, args.language) for a in audios))
    for p, r in zip(args.files, res):
        lat = "-" if r["finalize_ms"] is None else f"{r['finalize_ms']:.0f} ms"
        print(f"{p}\t{lat}\t{r['error'] or r['text']}")
    lats = [r["finalize_ms"] for r in res if r["finalize_ms"] is not None]
    if lats:
        print(f"finalize p50 {np.percentile(lats, 50):.0f} ms, p95 {np.percentile(lats, 95):.0f} ms over {len(lats)} streams")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--url", default="ws://localhost:8090/v1/stream")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--encoding", choices=["mulaw", "pcm16", "f32"], default="pcm16")
    p.add_argument("--packet-ms", type=int, default=20)
    p.add_argument("--language", default=None)
    p.add_argument("--fast", action="store_true", help="send as fast as possible instead of in real time")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
