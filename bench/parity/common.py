"""Dataset, text normalization and result I/O shared by the reference (NeMo) and lite benchmarks.

The benchmark subset is a stable, seeded sample of the public VietSuperSpeech validation split (500 utterances,
seed 42 by default), so every run and every backend sees the same audio.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

DATASET_NAME = "thanhnew2001/VietSuperSpeech"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
LANG_TAG_AT_END = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Z]{2})?>\s*$")


def normalize_for_wer(text: str) -> str:
    """Normalize case/punctuation while preserving Vietnamese word characters."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_language_tag(text: str) -> str:
    return LANG_TAG_AT_END.sub("", text).strip()


def load_validation_rows(split: str, max_samples: int | None, sample_seed: int) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset

    if split not in {"validation", "dev"}:
        raise ValueError("Only the VietSuperSpeech validation/dev split is supported.")
    candidates = [split, "dev" if split == "validation" else "validation"]
    last_error: Exception | None = None
    for split_name in candidates:
        try:
            ds = load_dataset(DATASET_NAME, split=split_name)
            rows = [dict(row) for row in ds]
            if max_samples is not None:
                rows = select_stable_subset(rows, max_samples, sample_seed)
            return rows, split_name
        except Exception as exc:  # Dataset split names changed on HF before.
            last_error = exc
    raise RuntimeError(f"Could not load split {split!r} from {DATASET_NAME}") from last_error


def select_stable_subset(rows: list[dict[str, Any]], max_samples: int, sample_seed: int) -> list[dict[str, Any]]:
    def stable_key(row: dict[str, Any]) -> tuple[str, str]:
        audio = str(row["audio"])
        digest = hashlib.sha256(f"{sample_seed}:{audio}".encode("utf-8")).hexdigest()
        return digest, audio

    return sorted(rows, key=stable_key)[: min(max_samples, len(rows))]


def download_audio_file(relative_path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=DATASET_NAME, repo_type="dataset", filename=relative_path)


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def parse_att_context(value: str) -> list[int]:
    parts = [p.strip() for p in value.strip("[] ").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("att_context must look like '56,13'")
    return [int(parts[0]), int(parts[1])]


def load_rows_with_audio(max_samples: int, seed: int, split: str = "validation"):
    from huggingface_hub import snapshot_download

    rows, loaded_split = load_validation_rows(split, max_samples if max_samples > 0 else None, seed)
    # one parallel snapshot instead of one request per file (per-file downloads stalled for 30+ min)
    paths = [str(row["audio"]) for row in rows]
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    root = snapshot_download(repo_id=DATASET_NAME, repo_type="dataset", allow_patterns=paths, max_workers=16)
    for row, rel in zip(rows, paths):
        row["local_audio"] = os.path.join(root, rel)
    print(f"audio ready: {len(paths)} files under {root}", flush=True)
    return rows, loaded_split


def write_results(stem: str, records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    import jiwer

    refs = [r["normalized_reference"] for r in records]
    hyps = [r["normalized_hypothesis"] for r in records]
    m = jiwer.process_words(refs, hyps)
    summary.update(
        samples=len(records),
        wer=m.wer,
        wer_percent=m.wer * 100.0,
        substitutions=m.substitutions,
        deletions=m.deletions,
        insertions=m.insertions,
        hits=m.hits,
        normalization="NFC, lowercase, punctuation stripped, whitespace collapsed",
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RESULTS_DIR / f"{stem}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def make_record(row: dict[str, Any], raw_text: str, tokens: list[int] | None, strip: bool = True) -> dict[str, Any]:
    reference = str(row["text"])
    hypothesis = strip_language_tag(raw_text) if strip else raw_text
    return {
        "audio": row["audio"],
        "duration": row.get("duration"),
        "source": row.get("source"),
        "reference": reference,
        "raw_hypothesis": raw_text,
        "hypothesis": hypothesis,
        "tokens": tokens,
        "normalized_reference": normalize_for_wer(reference),
        "normalized_hypothesis": normalize_for_wer(hypothesis),
    }


def set_strict_fp32() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
