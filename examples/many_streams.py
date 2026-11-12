"""Many concurrent streams on one GPU: every WAV file becomes its own live stream, all running at once.

    python examples/many_streams.py a.wav b.wav c.wav ... [--copies 100] [--language en-US]

Each stream is an ordinary asyncio task. The engine batches the chunks of all streams into one GPU step (up to
max_batch streams per step), so 100 streams cost far less than 100 x one stream. --copies replays the files to
reach a larger number of streams.
"""

import argparse
import asyncio
import time

import numpy as np

from nemotron_engine import AsyncNemotronEngine, EngineConfig
from nemotron_engine.cli import read_audio


async def run_stream(engine, audio: np.ndarray, sr: int, offset_s: float) -> tuple[str, float]:
    await asyncio.sleep(offset_s)  # stagger the starts like real clients
    stream = engine.open_stream(sample_rate=sr, encoding="f32")
    finals = []

    async def collect():
        async for ev in stream:
            if ev.kind == "final":
                finals.append(ev)

    reader = asyncio.create_task(collect())
    packet, start = sr // 50, time.perf_counter()
    for i in range(0, len(audio), packet):
        stream.push_nowait(audio[i : i + packet])
        await asyncio.sleep(max(0.0, start + (i + packet) / sr - time.perf_counter()))
    t_end = time.perf_counter()
    stream.end_turn()
    await stream.close()
    await reader
    return finals[0].text, (finals[0].t_emit - t_end) * 1000


async def main(args) -> None:
    audios = [read_audio(p) for p in args.files] * args.copies
    engine = await AsyncNemotronEngine.create(EngineConfig(max_streams=len(audios), language=args.language))
    rng = np.random.default_rng(0)
    results = await asyncio.gather(*(run_stream(engine, a, sr, float(rng.uniform(0, 2))) for a, sr in audios))
    for path, (text, _) in zip(args.files, results):
        print(f"{path}: {text}")
    lat = np.array([ms for _, ms in results])
    print(f"{len(results)} streams; end-of-utterance -> final: p50 {np.percentile(lat, 50):.0f} ms, "
          f"p95 {np.percentile(lat, 95):.0f} ms; engine: {engine.stats()}")
    await engine.aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--copies", type=int, default=1)
    p.add_argument("--language", default="auto")
    asyncio.run(main(p.parse_args()))
