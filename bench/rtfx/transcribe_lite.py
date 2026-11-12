#!/usr/bin/env python3
"""nemotron_lite counterpart of NeMo's examples/asr/transcribe_speech.py (RTFx methodology).

Takes the same key=value arguments and measures the same thing: one timed call that transcribes the whole
manifest (audio loading included), presorted by duration (longest first), after `warmup_steps` untimed
passes; RTFx = total manifest duration / model time, averaged over `run_steps`.

  python bench/transcribe_lite.py dataset_manifest=m.json output_filename=out.json batch_size=32 \
      att_context_size=[56,13] target_lang=vi-VN calculate_rtfx=true warmup_steps=1 run_steps=3
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

# isort: split
from nemotron_lite import NemotronLite, load_audio  # noqa: E402

DEFAULTS = {
    "dataset_manifest": None,
    "output_filename": None,
    "model_dir": None,
    "batch_size": 32,
    "presort_manifest": True,
    "att_context_size": None,
    "target_lang": None,
    "matmul_precision": "high",
    "compute_dtype": "float32",
    "calculate_rtfx": False,
    "warmup_steps": 0,
    "run_steps": 1,
}


def parse_args(argv: list[str]) -> dict:
    cfg = dict(DEFAULTS)
    for arg in argv:
        key, _, value = arg.partition("=")
        key = key.lstrip("+")
        if key not in cfg:
            raise SystemExit(f"unknown option {key!r}; known: {sorted(cfg)}")
        default = DEFAULTS[key]
        if isinstance(default, bool):
            cfg[key] = value.lower() in {"1", "true", "yes"}
        elif isinstance(default, int):
            cfg[key] = int(value)
        elif key == "att_context_size":
            cfg[key] = [int(v) for v in value.strip("[] ").split(",")]
        else:
            cfg[key] = None if value in {"", "None", "null"} else value
    return cfg


def transcribe(model: NemotronLite, paths: list[str], cfg: dict) -> list[str]:
    """Counterpart of `asr_model.transcribe(audio=filepaths)`: load, batch, encode, decode, detokenize."""
    lookahead = cfg["att_context_size"][1] if cfg["att_context_size"] else None
    language = cfg["target_lang"] or "auto"
    texts: list[str] = []
    for i in range(0, len(paths), cfg["batch_size"]):
        audios = [load_audio(p, model.cfg.sample_rate) for p in paths[i : i + cfg["batch_size"]]]
        texts.extend(model.transcribe_batch(audios, language=language, lookahead=lookahead, strip_lang_tag=False))
    return texts


def main():
    cfg = parse_args(sys.argv[1:])
    print("config:", json.dumps(cfg), flush=True)
    torch.set_float32_matmul_precision(cfg["matmul_precision"])
    torch.set_grad_enabled(False)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg["compute_dtype"]]
    model = NemotronLite.from_pretrained(cfg["model_dir"], device="cuda", dtype=dtype)

    items = [json.loads(l) for l in Path(cfg["dataset_manifest"]).read_text().splitlines() if l.strip()]
    order = list(range(len(items)))
    if cfg["presort_manifest"] and all(it.get("duration") is not None for it in items):
        order = sorted(order, reverse=True, key=lambda i: items[i]["duration"])  # same as NeMo
    paths = [items[i]["audio_filepath"] for i in order]
    total_duration = sum(float(it["duration"]) for it in items)

    measurements = []
    for step in range(cfg["warmup_steps"] + cfg["run_steps"]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        texts = transcribe(model, paths, cfg)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tag = "warmup" if step < cfg["warmup_steps"] else "run"
        print(f"Model time for iteration {step} ({tag}): {elapsed:.3f}", flush=True)
        if step >= cfg["warmup_steps"]:
            measurements.append(elapsed)

    m = np.asarray(measurements)
    print(f"Model time avg: {m.mean():.3f} (std: {m.std():.3f})")
    if cfg["calculate_rtfx"]:
        r = total_duration / m
        print(f"Model RTFx on the dataset: {r.mean():.3f} (std: {r.std():.3f})")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    pred = [None] * len(items)
    for i, t in zip(order, texts):
        pred[i] = t
    if cfg["output_filename"]:
        with open(cfg["output_filename"], "w", encoding="utf-8") as f:
            for it, t in zip(items, pred):
                f.write(json.dumps({**it, "pred_text": t}, ensure_ascii=False) + "\n")
        print("Wrote", cfg["output_filename"])


if __name__ == "__main__":
    main()
