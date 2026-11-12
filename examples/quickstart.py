"""Transcribe one WAV file through the streaming engine, printing partial results as they arrive.

    python examples/quickstart.py speech.wav [--language en-US]

The file is pushed in 20 ms packets at real-time speed, like a live microphone. A partial event comes with every
320 ms chunk that adds text; end_turn() (here: at the end of the file) produces the final.
"""

import argparse
import asyncio
import time

from nemotron_engine import AsyncNemotronEngine, EngineConfig
from nemotron_engine.cli import read_audio


async def main(path: str, language: str) -> None:
    audio, sr = read_audio(path)  # mono float32 at 8 or 16 kHz
    engine = await AsyncNemotronEngine.create(EngineConfig(max_streams=4, language=language))
    stream = engine.open_stream(sample_rate=sr, encoding="f32")

    async def show_events():
        async for ev in stream:
            if ev.kind == "partial":
                print(f"\r… {ev.text}", end="", flush=True)
            else:
                print(f"\r✔ {ev.text}  ({(ev.t_emit - ev.t_ready) * 1000:.0f} ms after end of utterance)")

    reader = asyncio.create_task(show_events())
    packet = sr // 50  # 20 ms
    start = time.perf_counter()
    for i in range(0, len(audio), packet):
        await stream.push(audio[i : i + packet])
        await asyncio.sleep(max(0.0, start + (i + packet) / sr - time.perf_counter()))  # real time
    stream.end_turn()  # in a live app, your VAD / endpointer calls this when the speaker stops
    await stream.close()  # waits for the final, then frees the slot
    await reader
    await engine.aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--language", default="auto")
    args = p.parse_args()
    asyncio.run(main(args.wav, args.language))
