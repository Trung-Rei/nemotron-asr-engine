#!/usr/bin/env python3
"""Reference runs with the original NeMo model (nvidia/nemotron-3.5-asr-streaming-0.6b).

  inspect  dump NeMo config, preprocessor buffers and checkpoint facts the lite port relies on
  offline  model.transcribe() on the benchmark subset
  stream   cache-aware streaming (same loop as NeMo's speech_to_text_cache_aware_streaming_infer.py)
  dump     per-layer tensors on a few samples for the lite parity test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from common import (
    RESULTS_DIR,
    batched,
    load_rows_with_audio,
    make_record,
    parse_att_context,
    set_strict_fp32,
    write_results,
)

MODEL_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_FILENAME = "nemotron-3.5-asr-streaming-0.6b.nemo"


def nemo_version() -> dict:
    import nemo

    info = {"nemo_version": getattr(nemo, "__version__", "?")}
    try:
        out = subprocess.run(["pip", "show", "-f", "nemo_toolkit"], capture_output=True, text=True).stdout
        info["nemo_pip"] = [l for l in out.splitlines() if l.startswith(("Version", "Location"))]
        direct = subprocess.run(
            ["python", "-c", "import importlib.metadata as m;print(m.distribution('nemo_toolkit').read_text('direct_url.json'))"],
            capture_output=True, text=True,
        ).stdout.strip()
        info["nemo_direct_url"] = direct
    except Exception as exc:  # informational only
        info["nemo_pip_error"] = str(exc)
    return info


def load_model(att_context: list[int], target_lang: str):
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_REPO, MODEL_FILENAME)
    model = nemo_asr.models.ASRModel.restore_from(restore_path=path, map_location="cuda")
    model = model.to("cuda").eval()
    model.encoder.set_default_att_context_size(att_context)
    if hasattr(model, "set_inference_prompt"):
        model.set_inference_prompt(target_lang)
    if hasattr(model.decoding, "set_strip_lang_tags"):
        model.decoding.set_strip_lang_tags(False)  # raw text; the lang tag is stripped the same way for both sides
    return model


def hyp_tokens(h) -> list[int]:
    seq = getattr(h, "y_sequence", None)
    if seq is None:
        return []
    return [int(t) for t in (seq.tolist() if hasattr(seq, "tolist") else seq)]


def hyp_text(h) -> str:
    return h if isinstance(h, str) else str(getattr(h, "text", h))


# ---------------------------------------------------------------------------- inspect
def cmd_inspect(args):
    from omegaconf import OmegaConf

    att = parse_att_context(args.att_context)
    model = load_model(att, args.target_lang)
    out = RESULTS_DIR / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_config.yaml").write_text(OmegaConf.to_yaml(model.cfg))
    sd = model.state_dict()
    (out / "state_dict_keys.txt").write_text("\n".join(f"{k} {tuple(v.shape)} {v.dtype}" for k, v in sd.items()))

    facts = {"class": type(model).__name__, **nemo_version()}
    pre = model.preprocessor.featurizer
    facts["preprocessor"] = {
        k: str(getattr(pre, k, None))
        for k in ["n_fft", "hop_length", "win_length", "preemph", "log_zero_guard_value", "mag_power",
                  "pad_to", "pad_value", "dither", "normalize", "exact_pad", "stft_pad_amount", "log"]
    }
    facts["fb_dtype"] = str(pre.fb.dtype)
    facts["window_dtype"] = str(pre.window.dtype) if pre.window is not None else None
    torch.save({"fb": pre.fb.detach().cpu(), "window": pre.window.detach().cpu()}, out / "preprocessor_buffers.pt")
    enc = model.encoder
    facts["encoder"] = {
        "att_context_style": getattr(enc, "att_context_style", None),
        "att_context_size": getattr(enc, "att_context_size", None),
        "att_context_size_all": str(getattr(enc, "att_context_size_all", None)),
        "self_attention_model": getattr(enc, "self_attention_model", None),
        "xscale": str(getattr(enc, "xscale", None)),
        "streaming_cfg": str(getattr(enc, "streaming_cfg", None)),
        "use_pytorch_sdpa": str(getattr(enc.layers[0].self_attn, "use_pytorch_sdpa", None)),
        "conv_norm": type(enc.layers[0].conv.batch_norm).__name__,
        "pre_encode_fuse_triton": str(getattr(getattr(enc.pre_encode, "conv", None), "fuse_triton", None)),
    }
    emb = model.decoder.prediction["embed"].weight
    facts["decoder"] = {
        "blank_idx": int(model.decoder.blank_idx),
        "blank_as_pad": bool(getattr(model.decoder, "blank_as_pad", False)),
        "embed_shape": list(emb.shape),
        "blank_row_abs_sum": float(emb[model.decoder.blank_idx].abs().sum()),
    }
    facts["decoding_cfg"] = OmegaConf.to_container(model.cfg.decoding, resolve=True)
    facts["prompt_index_vi"] = model.resolve_prompt_language(args.target_lang) if hasattr(model, "resolve_prompt_language") else None
    tok = model.tokenizer
    facts["tokenizer"] = {"type": type(tok).__name__, "vocab_size": tok.vocab_size}
    try:
        facts["tokenizer"]["ids_to_text_sample"] = tok.ids_to_text([12944])
    except Exception as exc:
        facts["tokenizer"]["ids_to_text_error"] = str(exc)
    (out / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(facts, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------- offline
def cmd_offline(args):
    att = parse_att_context(args.att_context)
    rows, split = load_rows_with_audio(args.max_samples, args.seed)
    model = load_model(att, args.target_lang)
    trcfg = model.get_transcribe_config()
    trcfg.use_lhotse = False
    trcfg.batch_size = args.batch_size
    trcfg.num_workers = 0
    trcfg.target_lang = args.target_lang
    trcfg.return_hypotheses = True

    records, started, audio_sec = [], time.time(), 0.0
    torch.cuda.reset_peak_memory_stats()
    for bi, batch in enumerate(batched(rows, args.batch_size), start=1):
        paths = [r["local_audio"] for r in batch]
        outs = model.transcribe(audio=paths, batch_size=args.batch_size, return_hypotheses=True, override_config=trcfg)
        if isinstance(outs, tuple):
            outs = outs[0]
        for row, h in zip(batch, outs):
            records.append(make_record(row, hyp_text(h), hyp_tokens(h)))
            audio_sec += float(row.get("duration") or 0.0)
        if bi == 1 or bi % 10 == 0:
            print(f"[ref offline] {len(records)}/{len(rows)} in {time.time() - started:.1f}s", flush=True)
    elapsed = time.time() - started
    stem = f"ref_offline_r{att[1]}_{len(rows)}"
    write_results(stem, records, {
        "impl": "nemo", "mode": "offline", "model_repo": MODEL_REPO, "split": split, "sample_seed": args.seed,
        "batch_size": args.batch_size, "target_lang": args.target_lang, "att_context_size": att,
        "elapsed_seconds": elapsed, "audio_seconds": audio_sec, "rtfx": audio_sec / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, **nemo_version(),
    })


# ---------------------------------------------------------------------------- streaming
def cmd_stream(args):
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    att = parse_att_context(args.att_context)
    rows, split = load_rows_with_audio(args.max_samples, args.seed)
    model = load_model(att, args.target_lang)
    enc = model.encoder
    print("streaming_cfg:", enc.streaming_cfg, flush=True)

    records, started, audio_sec = [], time.time(), 0.0
    torch.cuda.reset_peak_memory_stats()
    for i, row in enumerate(rows, start=1):
        buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
        buf.append_audio_file(row["local_audio"], stream_id=-1)
        cache_ch, cache_t, cache_len = enc.get_initial_cache_state(batch_size=1)
        prev_hyps, pred_out = None, None
        texts = None
        with torch.inference_mode():
            for step, (chunk, chunk_len) in enumerate(iter(buf)):
                drop = 0 if step == 0 else enc.streaming_cfg.drop_extra_pre_encoded
                pred_out, texts, cache_ch, cache_t, cache_len, prev_hyps = model.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=chunk_len,
                    cache_last_channel=cache_ch,
                    cache_last_time=cache_t,
                    cache_last_channel_len=cache_len,
                    keep_all_outputs=buf.is_buffer_empty(),
                    previous_hypotheses=prev_hyps,
                    previous_pred_out=pred_out,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
        h = prev_hyps[0] if prev_hyps else (texts[0] if texts else "")
        records.append(make_record(row, hyp_text(h), hyp_tokens(h)))
        audio_sec += float(row.get("duration") or 0.0)
        if i == 1 or i % 50 == 0:
            print(f"[ref stream] {i}/{len(rows)} in {time.time() - started:.1f}s", flush=True)
    elapsed = time.time() - started
    stem = f"ref_stream_r{att[1]}_{len(rows)}"
    write_results(stem, records, {
        "impl": "nemo", "mode": "stream", "model_repo": MODEL_REPO, "split": split, "sample_seed": args.seed,
        "batch_size": 1, "target_lang": args.target_lang, "att_context_size": att,
        "streaming_cfg": str(enc.streaming_cfg), "elapsed_seconds": elapsed, "audio_seconds": audio_sec,
        "rtfx": audio_sec / elapsed, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, **nemo_version(),
    })


# ---------------------------------------------------------------------------- tensor dump
def cmd_dump(args):
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    att = parse_att_context(args.att_context)
    rows, _ = load_rows_with_audio(args.max_samples, args.seed)
    rows = rows[: args.dump_samples]
    model = load_model(att, args.target_lang)
    prompt_id = model.resolve_prompt_language(args.target_lang)
    if args.no_triton:  # plain PyTorch subsampling instead of NeMo's fused Triton kernels
        model.encoder.pre_encode.conv.fuse_triton = False

    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = t.detach().float().cpu()
        return fn

    enc = model.encoder
    handles = [model.preprocessor.register_forward_hook(hook("mel")), enc.pre_encode.register_forward_hook(hook("pre_encode"))]
    handles += [layer.register_forward_hook(hook(f"layer{i}")) for i, layer in enumerate(enc.layers)]
    handles.append(model.prompt_kernel.register_forward_hook(hook("prompt")))
    handles.append(model.joint.enc.register_forward_hook(hook("enc_proj")))

    audios = [torch.tensor(AudioSegment.from_file(r["local_audio"], target_sr=16000).samples, dtype=torch.float32) for r in rows]
    dumps = []
    # batch of all samples (exercises padding) - same as transcribe with batch_size=len(rows)
    lengths = torch.tensor([a.numel() for a in audios])
    batch = torch.zeros(len(audios), int(lengths.max()))
    for i, a in enumerate(audios):
        batch[i, : a.numel()] = a
    with torch.inference_mode():
        encoded, encoded_len = model.forward(
            input_signal=batch.cuda(), input_signal_length=lengths.cuda(),
            language_prompt_indices=torch.full((len(audios),), prompt_id, dtype=torch.long, device="cuda"),
        )
        hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
    dumps.append({
        "audio": batch, "audio_lens": lengths, "prompt_id": prompt_id, "att_context": att,
        "encoded_len": encoded_len.cpu(), "tensors": dict(captured),
        "tokens": [hyp_tokens(h) for h in hyps], "texts": [hyp_text(h) for h in hyps],
        "paths": [r["audio"] for r in rows],
    })
    for h in handles:
        h.remove()
    out = RESULTS_DIR / f"ref_dump_r{att[1]}{'_notriton' if args.no_triton else ''}.pt"
    torch.save(dumps, out)
    print(f"saved {out}: {[(k, tuple(v.shape)) for k, v in captured.items()][:6]} ...", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["inspect", "offline", "stream", "dump"])
    p.add_argument("--att-context", default="56,13")
    p.add_argument("--target-lang", default="vi-VN")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dump-samples", type=int, default=4)
    p.add_argument("--no-triton", action="store_true")
    args = p.parse_args()
    set_strict_fp32()
    torch.set_grad_enabled(False)
    {"inspect": cmd_inspect, "offline": cmd_offline, "stream": cmd_stream, "dump": cmd_dump}[args.cmd](args)


if __name__ == "__main__":
    main()
