#!/usr/bin/env python3
"""Benchmark runs and tensor parity for the pure-PyTorch port (nemotron_lite).

  offline  batched transcription, same batches as the NeMo reference
  stream   cache-aware streaming, audio fed in 20 ms pieces like a live source
  parity   compare per-layer tensors / tokens with a NeMo dump (bench/run_reference.py dump)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

# isort: split
from common import (  # noqa: E402
    RESULTS_DIR,
    batched,
    load_rows_with_audio,
    make_record,
    parse_att_context,
    set_strict_fp32,
    write_results,
)
from nemotron_lite import NemotronLite, load_audio  # noqa: E402


def forbidden_modules() -> list[str]:
    return sorted({m.split(".")[0] for m in sys.modules if m.split(".")[0] in {"nemo", "transformers", "lightning", "pytorch_lightning"}})


def load(args) -> NemotronLite:
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    return NemotronLite.from_pretrained(args.model_dir, device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)


def base_summary(args, att, split, model) -> dict:
    return {
        "impl": "nemotron_lite", "split": split, "sample_seed": args.seed, "target_lang": args.target_lang,
        "att_context_size": att, "dtype": args.dtype, "device": str(model.device),
        "torch_version": torch.__version__,
    }


def cmd_offline(args):
    att = parse_att_context(args.att_context)
    rows, split = load_rows_with_audio(args.max_samples, args.seed)
    model = load(args)
    records, audio_sec = [], 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for bi, batch in enumerate(batched(rows, args.batch_size), start=1):
        audios = [load_audio(r["local_audio"], model.cfg.sample_rate) for r in batch]
        tensors = [torch.from_numpy(a) for a in audios]
        lengths = torch.tensor([t.numel() for t in tensors])
        pad = torch.zeros(len(tensors), int(lengths.max()))
        for i, t in enumerate(tensors):
            pad[i, : t.numel()] = t
        enc, enc_lens = model.encode(pad, lengths, args.target_lang, att[1])
        state = model.greedy.decode(enc, enc_lens)
        for row, audio, toks in zip(batch, audios, state.tokens):
            records.append(make_record(row, model.tokenizer.decode(toks), toks))
            audio_sec += len(audio) / model.cfg.sample_rate
        if bi == 1 or bi % 10 == 0:
            print(f"[lite offline] {len(records)}/{len(rows)} in {time.time() - started:.1f}s", flush=True)
    elapsed = time.time() - started
    write_results(f"lite_offline_r{att[1]}_{len(rows)}", records, {
        **base_summary(args, att, split, model), "mode": "offline", "batch_size": args.batch_size,
        "elapsed_seconds": elapsed, "audio_seconds": audio_sec, "rtfx": audio_sec / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "forbidden_modules_loaded": forbidden_modules(),
    })


def cmd_stream(args):
    att = parse_att_context(args.att_context)
    rows, split = load_rows_with_audio(args.max_samples, args.seed)
    model = load(args)
    piece = int(model.cfg.sample_rate * args.piece_ms / 1000)
    records, audio_sec = [], 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for i, row in enumerate(rows, start=1):
        audio = load_audio(row["local_audio"], model.cfg.sample_rate)
        sess = model.stream(language=args.target_lang, lookahead=att[1], strip_lang_tag=False)
        for s in range(0, len(audio), piece):
            sess.accept_audio(audio[s : s + piece])
        sess.finish()
        records.append(make_record(row, sess.text, list(sess.tokens)))
        audio_sec += len(audio) / model.cfg.sample_rate
        if i == 1 or i % 50 == 0:
            print(f"[lite stream] {i}/{len(rows)} in {time.time() - started:.1f}s", flush=True)
    elapsed = time.time() - started
    write_results(f"lite_stream_r{att[1]}_{len(rows)}", records, {
        **base_summary(args, att, split, model), "mode": "stream", "batch_size": 1, "piece_ms": args.piece_ms,
        "latency_ms": (att[1] + 1) * 80, "elapsed_seconds": elapsed, "audio_seconds": audio_sec,
        "rtfx": audio_sec / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "forbidden_modules_loaded": forbidden_modules(),
    })


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (b.norm() + 1e-12))


def cmd_parity(args):
    """Feed the exact NeMo input audio and compare every captured tensor on the valid frames."""
    att = parse_att_context(args.att_context)
    dump = torch.load(RESULTS_DIR / f"ref_dump_r{att[1]}{args.dump_suffix}.pt", weights_only=False)[0]
    print(f"parity vs ref_dump_r{att[1]}{args.dump_suffix}.pt  rel_tol={args.rel_tol}")
    model = load(args)
    audio, lens = dump["audio"], dump["audio_lens"]
    ref = dump["tensors"]
    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = t.detach().float().cpu()
        return fn

    hs = [model.encoder.subsampling.register_forward_hook(hook("pre_encode"))]
    hs += [l.register_forward_hook(hook(f"layer{i}")) for i, l in enumerate(model.encoder.layers)]
    hs.append(model.prompt_projector.register_forward_hook(hook("prompt")))
    hs.append(model.encoder_projector.register_forward_hook(hook("enc_proj")))
    feats, feat_lens = model.preprocessor(audio.to(model.device), lens.to(model.device))
    captured["mel"] = feats.float().cpu()
    enc, enc_lens = model.encode(audio, lens, dump["prompt_id"], att[1])
    state = model.greedy.decode(enc, enc_lens)
    for h in hs:
        h.remove()

    enc_len = dump["encoded_len"]
    report, ok = [], True
    for name, r in ref.items():
        l = captured.get(name)
        if l is None:
            report.append((name, "missing in lite"))
            continue
        if name == "mel":
            r = r.transpose(1, 2)  # NeMo (B, F, T) -> (B, T, F)
            valid_t = feat_lens.cpu()
        else:
            valid_t = enc_len
        t = min(r.shape[1], l.shape[1])
        errs, maxabs = [], 0.0
        for b in range(r.shape[0]):
            n = int(valid_t[b])
            rb, lb = r[b, : min(n, t)], l[b, : min(n, t)]
            errs.append(rel_err(lb, rb))
            maxabs = max(maxabs, float((lb - rb).abs().max()))
        tol = args.mel_tol if name == "mel" else args.rel_tol
        metric = max(errs)
        passed = metric < tol
        ok &= passed
        report.append((name, f"shape ref {tuple(r.shape)} lite {tuple(l.shape)}  max_rel {max(errs):.2e}  max_abs {maxabs:.2e}  {'OK' if passed else 'FAIL'}"))
    same_tokens = [a == b for a, b in zip(state.tokens, dump["tokens"])]
    for name, line in report:
        print(f"{name:12s} {line}")
    print("enc_len ref", enc_len.tolist(), "lite", enc_lens.tolist())
    print("tokens identical per sample:", same_tokens)
    for a, b in zip(dump["texts"], state.tokens):
        print("  ref :", a, "\n  lite:", model.tokenizer.decode(b))
    ok &= all(same_tokens) and enc_len.tolist() == enc_lens.cpu().tolist()
    print("PARITY", "PASS" if ok else "FAIL")
    if not ok and args.strict:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["offline", "stream", "parity"])
    p.add_argument("--att-context", default="56,13")
    p.add_argument("--target-lang", default="vi-VN")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--piece-ms", type=int, default=20)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--rel-tol", type=float, default=1e-4)
    p.add_argument("--mel-tol", type=float, default=1e-5)
    p.add_argument("--dump-suffix", default="")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    set_strict_fp32()
    torch.set_grad_enabled(False)
    {"offline": cmd_offline, "stream": cmd_stream, "parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    main()
