"""GPU checks for nemotron_engine (from the repo root): python tests/test_engine.py [--strict]

1. audio: the CPU checks of tests/test_audio_cpu.py
2. parity: AsyncNemotronEngine (concurrent asyncio streams, 20 ms packets) vs the reference port
   (research/nemotron_lite BatchedStreamEngine + StreamingFeatures, separate model load) on the same zero-padded audio,
   all streams in one batch per chunk; then 100 ms packets must give the same tokens; run again with
   --greedy-steps 1 --parity-only, where every chunk needs continuation graphs, to check those are exact too
3. 8 kHz mu-law input vs 16 kHz telephone-band input: per-utterance WER through the engine
4. stress: EngineFull, close mid-turn, end_turn with no audio, aclose with open streams, no leaked slots

The engine runs push packets as fast as possible, so admission control (max_lag_ms) is off.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from test_audio_cpu import mulaw_encode  # noqa: E402
from test_audio_cpu import run_all as check_audio  # noqa: E402

from nemotron_engine import AsyncNemotronEngine, EngineConfig, EngineFull  # noqa: E402

SR = 16000
RESULTS: dict = {}


async def run_engine(engine, audios, packet_ms, sample_rate=SR, encoding="f32", payload=None):
    """All streams concurrently, packets as fast as possible, one end_turn then close. Returns finals per stream."""
    async def one(i):
        st = engine.open_stream(sample_rate=sample_rate, encoding=encoding)
        events = []

        async def reader():
            async for ev in st:
                events.append(ev)

        rt = asyncio.create_task(reader())
        data = payload[i] if payload is not None else audios[i]
        step = int(sample_rate * packet_ms / 1000)  # samples (f32 arrays) or bytes (mu-law)
        for j in range(0, len(data), step):
            await st.push(data[j : j + step])
            if j // step % 8 == 0:
                await asyncio.sleep(0)  # let other streams interleave
        st.end_turn()
        await st.close()
        await rt
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, [e.kind for e in events]
        parts = [e for e in events if e.kind == "partial"]
        assert all(p.turn == 0 for p in parts)
        return finals[0]

    return await asyncio.gather(*(one(i) for i in range(len(audios))))


def reference_tokens(engine, audios):
    """Same streams through the lite code: zero padding as the engine's flush, lockstep batches."""
    from nemotron_lite import NemotronLite
    from nemotron_lite.batch_stream import BatchedStreamEngine
    from nemotron_lite.features import StreamingFeatures

    m = NemotronLite.from_pretrained(device="cuda", dtype=engine.cfg.dtype)
    bse = BatchedStreamEngine(m, engine.cfg.lookahead, len(audios), engine.cfg.language, cuda_graphs=False,
                              fast_greedy=True, fused_attn=True)
    ch = engine.chunking
    feats, n_chunks = [], []
    for a in audios:
        f = StreamingFeatures(m.preprocessor)
        n = ch.chunks_covering(len(a))
        need = ch.samples_for_chunks(n)
        f.accept(torch.from_numpy(np.concatenate([a, np.zeros(max(0, need - len(a)), np.float32)])))
        feats.append(f)
        n_chunks.append(n)
    slots = [bse.open() for _ in audios]
    for c in range(max(n_chunks)):
        live = [i for i in range(len(audios)) if c < n_chunks[i]]
        nf = bse.chunk_mel_frames(c == 0)
        mel = torch.stack([feats[i].take(nf, device="cuda") for i in live]).to(m.dtype)
        bse.step([slots[i] for i in live], mel)
    return [list(bse.tokens[s]) for s in slots]


def token_diff(a, b):
    import difflib

    return sum(max(len(x), len(y)) - sum(blk.size for blk in difflib.SequenceMatcher(a=x, b=y, autojunk=False).get_matching_blocks())
               for x, y in zip(a, b))


def wer(refs, hyps):
    import jiwer
    from common import normalize_for_wer

    r = [normalize_for_wer(x) for x in refs]
    h = [normalize_for_wer(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x]
    return 100 * jiwer.wer([r[i] for i in keep], [h[i] for i in keep])


async def main_async(args):
    import soundfile as sf
    from common import load_rows_with_audio, set_strict_fp32
    from scipy.signal import resample_poly

    check_audio()
    if args.strict:
        set_strict_fp32()
    rows, _ = load_rows_with_audio(max(args.n, args.n_8k), 42)
    audio16 = []
    for r in rows:
        a, sr = sf.read(r["local_audio"], dtype="float32", always_2d=True)
        assert sr == SR
        audio16.append(a.mean(axis=1).astype(np.float32))
    refs = [str(r["text"]) for r in rows]

    cfg = EngineConfig(language="vi-VN", max_streams=max(args.n, args.n_8k, 64), max_batch=64, max_lag_ms=None, greedy_steps=args.greedy_steps,
                       matmul_precision="highest" if args.strict else "high")
    t0 = time.perf_counter()
    engine = await AsyncNemotronEngine.create(cfg)
    print(f"engine up in {time.perf_counter() - t0:.1f}s, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved", flush=True)

    # ---- parity
    aud = audio16[: args.n]
    t0 = time.perf_counter()
    fin20 = await run_engine(engine, aud, 20)
    dt = time.perf_counter() - t0
    st20 = engine.stats()
    fin100 = await run_engine(engine, aud, 100)
    ref = reference_tokens(engine, aud)
    eng20 = [f.tokens for f in fin20]
    eng100 = [f.tokens for f in fin100]
    par = {
        "streams": len(aud), "audio_s": sum(len(a) for a in aud) / SR, "wall_s_20ms": dt,
        "vs_direct_identical": sum(a == b for a, b in zip(eng20, ref)), "vs_direct_tokens_diff": token_diff(ref, eng20),
        "pkt100_identical": sum(a == b for a, b in zip(eng20, eng100)), "pkt100_tokens_diff": token_diff(eng20, eng100),
        "tokens": sum(len(x) for x in ref), "wer_engine": wer(refs[: args.n], [f.text for f in fin20]),
        "steps": st20["steps"], "greedy_continuations": st20["greedy_continuations"],
    }
    RESULTS["parity"] = par
    print("parity:", json.dumps(par), flush=True)
    assert engine.stats()["free_slots"] == cfg.max_streams and engine.stats()["active"] == 0, engine.stats()
    assert par["vs_direct_identical"] == par["pkt100_identical"] == len(aud), par
    if args.parity_only:
        await engine.aclose()
        return

    # ---- 8 kHz mu-law vs 16 kHz telephone band
    aud8 = audio16[: args.n_8k]
    tel16 = [resample_poly(resample_poly(a, 1, 2), 2, 1).astype(np.float32) for a in aud8]
    mulaw = [mulaw_encode(resample_poly(a, 1, 2).astype(np.float32)) for a in aud8]
    fin_tel = await run_engine(engine, tel16, 20)
    fin_mu = await run_engine(engine, [None] * len(aud8), 20, sample_rate=8000, encoding="mulaw", payload=mulaw)
    r8 = {"utts": len(aud8), "wer_16k_telephone": wer(refs[: args.n_8k], [f.text for f in fin_tel]),
          "wer_8k_mulaw": wer(refs[: args.n_8k], [f.text for f in fin_mu]),
          "wer_16k_clean": wer(refs[: args.n_8k], [f.text for f in (await run_engine(engine, aud8, 20))])}
    RESULTS["audio_8k"] = r8
    print("8k:", json.dumps(r8), flush=True)

    # ---- multi-turn on one stream: finals per turn, tokens split at the boundaries
    st = engine.open_stream(encoding="f32")
    evs = []

    async def rd():
        async for e in st:
            evs.append(e)

    rt = asyncio.create_task(rd())
    for k in range(3):
        a = audio16[k]
        for j in range(0, len(a), 320):
            await st.push(a[j : j + 320])
        st.end_turn()
    st.end_turn()  # no audio since the last end_turn -> empty final
    await st.close()
    await rt
    finals = [e for e in evs if e.kind == "final"]
    assert [f.turn for f in finals] == [0, 1, 2, 3], [(e.kind, e.turn) for e in evs]
    assert finals[3].text == "" and all(f.text for f in finals[:3])
    RESULTS["multi_turn"] = {"turns": [f.text[:60] for f in finals],
                             "wer": wer(refs[:3], [f.text for f in finals[:3]])}
    print("multi-turn:", RESULTS["multi_turn"], flush=True)
    await engine.aclose()

    # ---- stress with a small engine
    small = await AsyncNemotronEngine.create(EngineConfig(language="vi-VN", max_streams=4, max_batch=4, cuda_graphs=False, max_lag_ms=None))
    ss = [small.open_stream(encoding="f32") for _ in range(4)]
    try:
        small.open_stream()
        raise AssertionError("expected EngineFull")
    except EngineFull:
        pass
    a = audio16[0]
    await ss[0].push(a[: len(a) // 2])
    await ss[0].close()  # mid-turn: final then release
    await ss[1].close()  # never pushed: no final
    got0 = [e async for e in ss[0]]
    got1 = [e async for e in ss[1]]
    assert [e.kind for e in got0][-1:] == ["final"] and got1 == [], (got0, got1)
    s5 = small.open_stream(encoding="f32")  # slot came back
    await s5.push(a[:SR])
    await small.aclose()  # with open streams: iterators must end
    rest = await asyncio.wait_for(asyncio.gather(*([_drain(x) for x in ss[2:]] + [_drain(s5)])), 10)
    RESULTS["stress"] = {"ok": True, "events_after_aclose": [len(r) for r in rest]}
    print("stress: OK", flush=True)


async def _drain(st):
    return [e async for e in st]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--n-8k", type=int, default=200)
    p.add_argument("--strict", action="store_true", help="strict fp32 (no TF32) for the parity check")
    p.add_argument("--greedy-steps", type=int, default=EngineConfig.greedy_steps)
    p.add_argument("--parity-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    asyncio.run(main_async(args))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "engine_test.json").write_text(json.dumps(RESULTS, indent=1, ensure_ascii=False))
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
