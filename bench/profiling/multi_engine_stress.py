"""Several engines on one GPU at the same time (CUDA graphs on): do they fault (CUDA error 700) or corrupt tokens?

Each round, every engine in this process decodes the same --n VietSuperSpeech utterances concurrently (20 ms
packets as fast as possible, so the GPU stays busy). Tokens of every round and engine are compared with round 0 of
engine 0, and WER is tracked per round. A CUDA fault raises in the engine worker and ends the run with an error.
Run two copies of this script at once (--engines 1 each) to test separate processes on one GPU.

    python bench/profiling/multi_engine_stress.py --engines 2 --rounds 10 --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from test_engine import run_engine, token_diff, wer  # noqa: E402

from nemotron_engine import AsyncNemotronEngine, EngineConfig  # noqa: E402


async def main_async(args):
    import soundfile as sf
    from common import load_rows_with_audio

    rows, _ = load_rows_with_audio(args.n, 42)
    audio = [sf.read(r["local_audio"], dtype="float32", always_2d=True)[0].mean(axis=1).astype(np.float32) for r in rows]
    refs = [str(r["text"]) for r in rows]
    cfg = EngineConfig(language="vi-VN", max_streams=max(64, args.n), max_batch=64, max_lag_ms=None, matmul_precision=args.matmul_precision)
    engines = [await AsyncNemotronEngine.create(cfg) for _ in range(args.engines)]
    print(f"pid {os.getpid()}: {args.engines} engine(s) up, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved", flush=True)
    base, log = None, []
    for rnd in range(args.rounds):
        t = time.perf_counter()
        outs = await asyncio.gather(*(run_engine(e, audio, 20) for e in engines))
        dt = time.perf_counter() - t
        toks = [[f.tokens for f in o] for o in outs]
        if base is None:
            base = toks[0]
        row = {"round": rnd, "wall_s": dt,
               "token_diff_vs_base": [token_diff(base, tk) for tk in toks],
               "streams_identical": [sum(a == b for a, b in zip(base, tk)) for tk in toks],
               "wer": [wer(refs, [f.text for f in o]) for o in outs]}
        log.append(row)
        print(json.dumps(row), flush=True)
    for e in engines:
        await e.aclose()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"stress_{os.getpid()}.json").write_text(json.dumps(log, indent=1))
    print("STRESS DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines", type=int, default=2)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--matmul-precision", default="high")
    p.add_argument("--out", required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
