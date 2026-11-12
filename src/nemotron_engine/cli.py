"""Command line: transcribe files, serve the WebSocket API, or measure how many streams a GPU sustains.

    nemotron-engine transcribe a.wav b.wav [--language en-US] [--realtime] [--partials]
    nemotron-engine serve [--port 8090] [--max-streams 256]
    nemotron-engine bench [--streams 64 128 256] [--seconds 60] [--wav speech.wav]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import wave

import numpy as np

SUPPORTED_SR = (8000, 16000)


# --------------------------------------------------------------------------- audio files
def read_audio(path: str) -> tuple[np.ndarray, int]:
    """Mono float32 audio at 8 or 16 kHz. WAV PCM16 needs only the standard library; other formats use soundfile."""
    try:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                raise wave.Error("not 16-bit PCM")
            sr, ch = w.getframerate(), w.getnchannels()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            x = x.reshape(-1, ch).mean(axis=1)
    except (wave.Error, EOFError):
        try:
            import soundfile as sf
        except ImportError:
            sys.exit(f"{path}: only 16-bit PCM WAV is readable without soundfile (pip install soundfile)")
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    if sr not in SUPPORTED_SR:
        x, sr = _resample(x, sr, 16000), 16000
    return np.ascontiguousarray(x, dtype=np.float32), sr


def _resample(x: np.ndarray, sr: int, to: int) -> np.ndarray:
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, to)
        return resample_poly(x, to // g, sr // g).astype(np.float32)
    except ImportError:  # linear interpolation: fine for a quick try, scipy is better
        n = int(round(len(x) * to / sr))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _engine_config(args, **over):
    from .engine import EngineConfig

    kw = dict(language=args.language, matmul_precision=args.matmul_precision, cuda_graphs=not args.no_cuda_graphs)
    if args.model_dir:
        kw["model_dir"] = args.model_dir
    kw.update(over)
    return EngineConfig(**kw)


# --------------------------------------------------------------------------- transcribe
async def _transcribe(args) -> None:
    from .engine import AsyncNemotronEngine

    audios = [read_audio(p) for p in args.files]
    t0 = time.perf_counter()
    engine = await AsyncNemotronEngine.create(_engine_config(args, max_streams=max(1, len(audios)), max_lag_ms=None))
    print(f"[engine ready in {time.perf_counter() - t0:.1f} s]", file=sys.stderr)
    multi = len(audios) > 1

    async def one(path: str, x: np.ndarray, sr: int) -> str:
        st = engine.open_stream(sample_rate=sr, encoding="f32")
        finals: list[str] = []

        async def reader():
            async for ev in st:
                if ev.kind == "final":
                    if ev.text:
                        finals.append(ev.text)
                elif args.partials:
                    print(f"{path + ': ' if multi else ''}… {ev.text}", file=sys.stderr, flush=True)

        task = asyncio.create_task(reader())
        n = sr * args.packet_ms // 1000
        start = time.perf_counter()
        for i in range(0, len(x), n):
            st.push_nowait(x[i : i + n])
            if args.realtime:
                await asyncio.sleep(max(0.0, start + (i + n) / sr - time.perf_counter()))
            elif i // n % 50 == 0:
                await asyncio.sleep(0)
        await st.close()  # finishes the open turn
        await task
        return " ".join(finals)

    t1 = time.perf_counter()
    texts = await asyncio.gather(*(one(p, x, sr) for p, (x, sr) in zip(args.files, audios)))
    dt = time.perf_counter() - t1
    for p, t in zip(args.files, texts):
        print(f"{p}\t{t}" if multi else t)
    total = sum(len(x) / sr for x, sr in audios)
    print(f"[{total:.1f} s of audio in {dt:.2f} s]", file=sys.stderr)
    await engine.aclose()


# --------------------------------------------------------------------------- serve
def _serve(args) -> None:
    try:
        import uvicorn
    except ImportError:
        sys.exit('serve needs the server extra: pip install "nemotron-engine[server]"')
    env = {"NEMOTRON_LANGUAGE": args.language, "NEMOTRON_MAX_STREAMS": args.max_streams, "NEMOTRON_MAX_BATCH": args.max_batch,
           "NEMOTRON_MAX_LAG_MS": args.max_lag_ms, "NEMOTRON_MATMUL_PRECISION": args.matmul_precision,
           "NEMOTRON_MODEL_DIR": args.model_dir, "NEMOTRON_CUDA_GRAPHS": "0" if args.no_cuda_graphs else None}
    for k, v in env.items():
        if v is not None:
            os.environ[k] = str(v)
    # one worker only: the engine owns the GPU, and a second process would load a second engine
    uvicorn.run("nemotron_engine.server:app", host=args.host, port=args.port, workers=1, log_level=args.log_level)


# --------------------------------------------------------------------------- bench
async def _bench(args) -> None:
    """N concurrent real-time streams; each speaks for --utt-s, pauses for --pause-s, and ends every utterance."""
    from .engine import AsyncNemotronEngine, EngineFull

    if args.wav:
        x, sr = read_audio(args.wav)
        if sr != 16000:
            x = _resample(x, sr, 16000)
    else:  # no speech at hand: noise still exercises the encoder; token counts (greedy) will be lower than speech
        x = (np.random.default_rng(0).standard_normal(16000 * 30) * 0.05).astype(np.float32)
    sr = 16000
    utt, pause = int(args.utt_s * sr), int(args.pause_s * sr)
    engine = await AsyncNemotronEngine.create(_engine_config(args, max_streams=max(args.streams), max_batch=args.max_batch,
                                                             max_lag_ms=None))
    print("| streams | chunks/s | mean batch | finalize p50 / p95 / p99 ms | SLO p95 ≤ %d ms |" % args.slo_ms)
    print("|---|---|---|---|---|")
    for n_streams in args.streams:
        fin_ms: list[float] = []
        rng = np.random.default_rng(n_streams)
        pkt = sr * args.packet_ms // 1000

        async def one(k: int) -> None:
            try:
                st = engine.open_stream(sample_rate=sr, encoding="f32")
            except EngineFull:
                return
            ends: dict[int, float] = {}

            async def reader():
                async for ev in st:
                    if ev.kind == "final" and ev.turn in ends and t_measure <= ends[ev.turn]:
                        fin_ms.append((time.perf_counter() - ends[ev.turn]) * 1000)

            task = asyncio.create_task(reader())
            pos, sent, turn = int(rng.integers(len(x))), 0, 0
            phase = int(rng.integers(utt + pause))  # streams start at random points of the speak/pause cycle
            start = time.perf_counter() + k * 0.001
            while sent < total:
                p = (phase + sent) % (utt + pause)
                if p < utt:
                    chunk = np.take(x, range(pos, pos + pkt), mode="wrap")
                    pos += pkt
                else:
                    chunk = np.zeros(pkt, np.float32)
                st.push_nowait(chunk)
                sent += pkt
                if p < utt <= p + pkt:  # utterance ends inside this packet
                    ends[turn] = time.perf_counter()
                    st.end_turn()
                    turn += 1
                await asyncio.sleep(max(0.0, start + sent / sr - time.perf_counter()))
            await st.close()
            await task

        total = int(args.seconds * sr)
        t_measure = time.perf_counter() + args.warmup_s
        s0 = dict(engine.stats())
        t0 = time.perf_counter()
        await asyncio.gather(*(one(k) for k in range(n_streams)))
        s1, dt = engine.stats(), time.perf_counter() - t0
        chunks, steps = s1["chunks"] - s0["chunks"], max(1, s1["steps"] - s0["steps"])
        q = np.percentile(fin_ms, [50, 95, 99]) if fin_ms else [float("nan")] * 3
        ok = "✅" if fin_ms and q[1] <= args.slo_ms else "❌"
        print(f"| {n_streams} | {chunks / dt:.0f} | {chunks / steps:.1f} | {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} | {ok} |", flush=True)
    await engine.aclose()


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="nemotron-engine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--language", default="auto", help="language prompt: auto, en-US, vi-VN, de-DE, ... (default auto)")
        sp.add_argument("--model-dir", default=None, help="local checkpoint dir (default: download from the HF hub)")
        sp.add_argument("--matmul-precision", default="high", choices=["high", "highest"], help="high = TF32 matmuls")
        sp.add_argument("--no-cuda-graphs", action="store_true", help="run steps eagerly (slower)")

    t = sub.add_parser("transcribe", help="stream audio files through the engine and print the transcripts")
    t.add_argument("files", nargs="+", help="WAV files (16-bit PCM; other formats need soundfile)")
    t.add_argument("--realtime", action="store_true", help="push audio at real-time speed instead of as fast as possible")
    t.add_argument("--partials", action="store_true", help="print partial results to stderr")
    t.add_argument("--packet-ms", type=int, default=20)
    common(t)

    s = sub.add_parser("serve", help="run the WebSocket server (needs the [server] extra)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--max-streams", type=int, default=None, help="concurrent stream slots (default 512)")
    s.add_argument("--max-batch", type=int, default=None, help="streams per GPU step (default 64)")
    s.add_argument("--max-lag-ms", default=None, help="refuse new streams above this chunk lag (default 200; 'none' disables)")
    s.add_argument("--log-level", default="warning")
    common(s)

    b = sub.add_parser("bench", help="concurrent real-time streams: throughput and end-of-utterance latency")
    b.add_argument("--streams", type=int, nargs="+", default=[64, 128, 256])
    b.add_argument("--seconds", type=float, default=60.0, help="length of each level")
    b.add_argument("--warmup-s", type=float, default=15.0, help="latencies before this are not counted")
    b.add_argument("--utt-s", type=float, default=3.0, help="speech per utterance")
    b.add_argument("--pause-s", type=float, default=5.0, help="pause after each utterance")
    b.add_argument("--packet-ms", type=int, default=20)
    b.add_argument("--max-batch", type=int, default=64)
    b.add_argument("--slo-ms", type=float, default=300.0)
    b.add_argument("--wav", default=None, help="speech to loop over (default: noise)")
    common(b)

    args = p.parse_args(argv)
    if args.cmd == "serve":
        _serve(args)
    else:
        try:
            asyncio.run(_transcribe(args) if args.cmd == "transcribe" else _bench(args))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
