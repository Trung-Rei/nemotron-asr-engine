#!/usr/bin/env python3
"""Summarize transcribe_speech.py (NeMo) vs transcribe_lite.py runs: RTFx from the logs, WER and agreement from the outputs."""

import json
import re
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from common import normalize_for_wer, strip_language_tag

out = Path(sys.argv[1])


def rtfx(log):
    text = log.read_text(errors="replace") if log.exists() else ""
    m = re.findall(r"Model RTFx on the dataset: ([\d.]+)(?: \(std: ([\d.]+)\))?", text)
    t = re.findall(r"Model time avg: ([\d.]+)", text)
    v = re.findall(r"Peak VRAM: ([\d.]+)", text)
    return (float(m[-1][0]) if m else None, float(m[-1][1]) if m and m[-1][1] else None,
            float(t[-1]) if t else None, float(v[-1]) if v else None)


def preds(path):
    items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [normalize_for_wer(i["text"]) for i in items], [normalize_for_wer(strip_language_tag(i["pred_text"])) for i in items]


rows = ["| att | impl | model time (s) | RTFx (std) | peak VRAM | WER % | same hyp as NeMo |", "|---|---|---|---|---|---|---|"]
summary = []
for r in sorted({int(p.stem.split("_r")[1]) for p in out.glob("*_r*.json") if p.stem.split("_r")[1].isdigit()}, reverse=True):
    ref_hyp = None
    for impl in ("nemo", "lite"):
        js, log = out / f"{impl}_r{r}.json", out / f"{impl}_r{r}.log"
        if not js.exists():
            continue
        refs, hyps = preds(js)
        wer = jiwer.wer(refs, hyps) * 100
        if impl == "nemo":
            ref_hyp = hyps
        same = None if ref_hyp is None else sum(a == b for a, b in zip(ref_hyp, hyps)) / len(hyps)
        x, sd, t, vram = rtfx(log)
        summary.append({"att": r, "impl": impl, "rtfx": x, "rtfx_std": sd, "model_time_s": t, "peak_vram_gb": vram, "wer": wer, "same_as_nemo": same})
        rows.append(f"| [56,{r}] | {impl} | {t} | {x} ({sd}) | {vram if vram else '-'} | {wer:.2f} | {'-' if same is None else f'{same:.1%}'} |")
(out / "rtfx_summary.json").write_text(json.dumps(summary, indent=2))
(out / "rtfx_summary.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
