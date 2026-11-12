#!/usr/bin/env python3
"""Write the 500-utterance VietSuperSpeech benchmark subset as a NeMo manifest (audio_filepath, duration, text, lang)."""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

# isort: split
from common import load_rows_with_audio

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--max-samples", type=int, default=500)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

rows, _ = load_rows_with_audio(args.max_samples, args.seed)
with open(args.out, "w", encoding="utf-8") as f:
    for r in rows:
        info = sf.info(r["local_audio"])
        f.write(json.dumps({
            "audio_filepath": r["local_audio"], "duration": info.frames / info.samplerate,
            "text": r["text"], "lang": "vi-VN", "audio": r["audio"],
        }, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} items, {sum(1 for _ in open(args.out))} lines -> {args.out}")
