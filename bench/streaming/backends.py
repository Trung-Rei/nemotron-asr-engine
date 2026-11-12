"""Streaming ASR backends behind one interface, so the scheduler and load are identical for both.

  open() -> slot        close(slot)
  step(slots, chunks) -> new tokens per slot, where chunks[i] = (session_mel_cpu (T, n_mels), start_frame, n)

Both use the same log-mel front-end (computed once per session, frame-exact with incremental streaming).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

MODEL_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"


class LiteBackend:
    name = "lite"

    def __init__(self, lookahead: int, max_streams: int, language: str, dtype: torch.dtype, max_batch: int = 64):
        import os

        from nemotron_lite import NemotronLite
        from nemotron_lite.batch_stream import BatchedStreamEngine

        self.model = NemotronLite.from_pretrained(device="cuda", dtype=dtype)
        graphs = os.environ.get("LITE_CUDA_GRAPHS", "1") == "1"
        self.engine = BatchedStreamEngine(self.model, lookahead, max_streams, language, cuda_graphs=graphs,
                                          fast_greedy=os.environ.get("LITE_FAST_GREEDY", "1") == "1", max_batch=max_batch,
                                          fused_attn=os.environ.get("LITE_FUSED_ATTN", "1") == "1")
        if graphs:
            self.engine.capture_graphs()
        self.tokenizer = self.model.tokenizer

    def first_chunk_frames(self) -> int:
        return self.engine.chunk_mel_frames(True)

    def chunk_frames(self) -> int:
        return self.engine.chunk_mel_frames(False)

    def open(self) -> int:
        return self.engine.open()

    def close(self, slot: int) -> None:
        self.engine.close(slot)

    def step(self, slots, chunks):
        mel = torch.stack([m[s : s + n] for m, s, n in chunks]).to("cuda", non_blocking=True)
        return self.engine.step(slots, mel)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class NemoBackend:
    """NeMo `conformer_stream_step` with pooled per-stream caches and hypotheses, as in
    examples/asr/asr_cache_aware_streaming (pre-encode cache of earlier mel frames, drop_extra_pre_encoded)."""

    name = "nemo"

    def __init__(self, lookahead: int, max_streams: int, language: str, dtype: torch.dtype, max_batch: int = 64):
        import nemo.collections.asr as nemo_asr
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(MODEL_REPO, "nemotron-3.5-asr-streaming-0.6b.nemo")
        m = nemo_asr.models.ASRModel.restore_from(restore_path=path, map_location="cuda").eval()
        m.encoder.set_default_att_context_size([m.encoder.att_context_size[0], lookahead])
        m.set_inference_prompt(language)
        if hasattr(m.decoding, "set_strip_lang_tags"):
            m.decoding.set_strip_lang_tags(False)
        self.model = m.to(device="cuda", dtype=dtype)
        cfg = m.encoder.streaming_cfg
        self.first_n, self.later_n = int(cfg.chunk_size[0]), int(cfg.chunk_size[1])
        self.pre_cache = int(cfg.pre_encode_cache_size[1])
        self.drop = int(cfg.drop_extra_pre_encoded)
        ch, t, ln = m.encoder.get_initial_cache_state(batch_size=max_streams)
        self.cache_ch, self.cache_t, self.cache_len = ch.to(dtype), t.to(dtype), ln
        self.hyps: list = [None] * max_streams
        self.ntok = [0] * max_streams
        self.free = list(range(max_streams - 1, -1, -1))
        self.tokenizer = m.tokenizer

    def first_chunk_frames(self) -> int:
        return self.first_n

    def chunk_frames(self) -> int:
        return self.later_n

    def open(self) -> int:
        slot = self.free.pop()
        self.cache_ch[:, slot] = 0
        self.cache_t[:, slot] = 0
        self.cache_len[slot] = 0
        self.hyps[slot], self.ntok[slot] = None, 0
        return slot

    def close(self, slot: int) -> None:
        self.hyps[slot] = None
        self.free.append(slot)

    @torch.no_grad()
    def step(self, slots, chunks):
        first = self.hyps[slots[0]] is None
        assert all((self.hyps[s] is None) == first for s in slots), "mixed first/later chunks in one batch"
        if first:
            sig = torch.stack([m[s : s + n] for m, s, n in chunks])
        else:
            sig = torch.stack([m[s - self.pre_cache : s + n] for m, s, n in chunks])
        sig = sig.transpose(1, 2).to("cuda", dtype=self.cache_ch.dtype, non_blocking=True)  # (B, F, T)
        idx = torch.tensor(slots, device="cuda")
        lens = torch.full((len(slots),), sig.shape[-1], device="cuda", dtype=torch.long)
        out = self.model.conformer_stream_step(
            processed_signal=sig,
            processed_signal_length=lens,
            cache_last_channel=self.cache_ch[:, idx],
            cache_last_time=self.cache_t[:, idx],
            cache_last_channel_len=self.cache_len[idx],
            keep_all_outputs=False,
            previous_hypotheses=None if first else [self.hyps[s] for s in slots],
            drop_extra_pre_encoded=0 if first else self.drop,
            return_transcription=True,
        )
        _, hyps, ch, t, ln = out[:5]
        self.cache_ch[:, idx] = ch
        self.cache_t[:, idx] = t
        self.cache_len[idx] = ln
        new = []
        for s, h in zip(slots, hyps):
            seq = h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)
            new.append([int(x) for x in seq[self.ntok[s] :]])
            self.ntok[s] = len(seq)
            self.hyps[s] = h
        return new

    def decode(self, tokens):
        return self.tokenizer.ids_to_text(tokens)


def make_backend(name: str, **kw):
    return {"lite": LiteBackend, "nemo": NemoBackend}[name](**kw)
