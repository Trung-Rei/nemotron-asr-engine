"""Multi-stream cache-aware streaming: many live streams, one batched encoder/decoder step.

Each stream owns a slot in pooled cache tensors. `step(slots, mel)` gathers the slots, runs one chunk for all
of them, scatters the caches back and returns the new tokens per stream. Streams in one call must be in the
same phase (all first chunks or all later chunks), since the two carry a different number of mel frames.

The attention cache always has `left_chunks * chunk` positions; a stream that has seen less audio keeps zeros
in the oldest positions and masks them, so relative positions stay exact (as NeMo's cache_last_channel_len).

Speed options (all give the same tokens up to float rounding):
  cuda_graphs   the 24 Conformer layers + prompt projection run as one CUDA graph per batch bucket. The batch is
                padded up to the bucket; padded rows compute on stale data and are discarded (rows never mix).
  fast_greedy   greedy RNN-T keeps labels on the GPU and syncs once per symbol loop instead of once per token.
  fused_attn    attention core runs as one SDPA kernel with the rel-pos term as an additive bias (the shifted
                rel-pos keys are precomputed, since they do not depend on the input), Q/K/V are one GEMM, and the
                chunk's K/V are written straight into a (cache + chunk) buffer: no concat and no write-back
                inside the encoder; the cache roll happens in the scatter (buffer[T:] -> pool).
Pools are slot-major ([S, L, ...]) so a batch gathers / scatters each cache with one indexed copy.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

from .model import NemotronLite
from .rnnt import DecoderState


def _buckets(max_batch: int) -> list[int]:
    b = {1, 2, 4, 8, max_batch} | set(range(16, max_batch + 1, 8))
    return sorted(x for x in b if x <= max_batch)


class BatchedStreamEngine:
    def __init__(self, model: NemotronLite, lookahead: int, max_streams: int, language="auto",
                 cuda_graphs: bool = False, fast_greedy: bool = False, max_batch: int | None = None,
                 fused_attn: bool = False):
        self.model = model
        self.cfg = model.cfg
        self.lookahead = lookahead
        self.chunk = lookahead + 1
        enc = model.encoder
        self.cache_len = (self.cfg.left_context // self.chunk) * self.chunk
        self.device, self.dtype = model.device, model.dtype
        S, L = max_streams, len(enc.layers)
        self.n_layers = L
        H, dk, D = self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads, self.cfg.d_model
        z = dict(device=self.device, dtype=self.dtype)
        self.k = torch.zeros(S, L, H, self.cache_len, dk, **z)
        self.v = torch.zeros(S, L, H, self.cache_len, dk, **z)
        self.kv_len = torch.zeros(S, dtype=torch.long, device=self.device)
        self.conv = torch.zeros(S, L, D, self.cfg.conv_kernel - 1, **z)
        self.sub: list[torch.Tensor | None] = [None] * len(enc.subsampling.strided_convs())
        self.first = torch.ones(S, dtype=torch.bool)
        H2 = self.cfg.pred_hidden
        self.pred_out = torch.zeros(S, H2, **z)
        self.h = torch.zeros(self.cfg.pred_layers, S, H2, **z)
        self.c = torch.zeros(self.cfg.pred_layers, S, H2, **z)
        self.tokens: list[list[int]] = [[] for _ in range(S)]
        self.free = list(range(S - 1, -1, -1))
        self.language = language
        self._lang_rows: dict[str, torch.Tensor] = {}
        self.one_hot_row = self._lang_row(language)
        self.lang = self.one_hot_row[None].repeat(S, 1)  # per-slot language prompt (one-hot)
        self.pos = enc.pos_emb(self.cache_len + self.chunk, self.device, self.dtype)
        self.init_state = model.greedy.init_state(1, self.device, self.dtype)
        self.sections: list | None = None  # set to [] to record (name, cuda start, cuda end, cpu t0, cpu t1)

        self.fast_greedy = fast_greedy
        self.fused_attn = fused_attn
        self.n_keys = self.cache_len + self.chunk
        kv_seq = self.n_keys if fused_attn else self.cache_len
        if fused_attn:
            self.keys_pad = -(-self.n_keys // 16) * 16  # SDPA bias rows aligned to 16 elements
            self.pos_keys = [l.self_attn.stream_pos_keys(self.pos, self.chunk, self.n_keys, self.keys_pad)
                             for l in enc.layers]
        self.cuda_graphs = cuda_graphs
        self.max_batch = min(max_batch or max_streams, max_streams)
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        if cuda_graphs:
            Bm = self.max_batch
            self.buckets = _buckets(Bm)
            self._x = torch.zeros(Bm, self.chunk, D, **z)
            self._kb = torch.zeros(Bm, L, H, kv_seq, dk, **z)
            self._vb = torch.zeros(Bm, L, H, kv_seq, dk, **z)
            self._cb = torch.zeros(Bm, L, D, self.cfg.conv_kernel - 1, **z)
            self._kl = torch.zeros(Bm, dtype=torch.long, device=self.device)
            self._oh = self.one_hot_row[None].repeat(Bm, 1)

    # ------------------------------------------------------------------ profiling
    def _sec(self, name: str):
        return nullcontext() if self.sections is None else self._record(name)

    @contextmanager
    def _record(self, name: str):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.profiler.record_function(name):
            t0 = time.perf_counter()
            s.record()
            try:
                yield
            finally:
                e.record()
                self.sections.append((name, s, e, t0, time.perf_counter()))

    # ------------------------------------------------------------------ slots
    def _lang_row(self, language) -> torch.Tensor:
        key = str(language)
        if key not in self._lang_rows:
            self._lang_rows[key] = self.model._one_hot(language, 1)[0]
        return self._lang_rows[key]

    def open(self, language=None) -> int:
        """Take a free slot and reset it. `language` (None = the engine default) sets the stream's prompt."""
        slot = self.free.pop()
        self.lang[slot] = self._lang_row(self.language if language is None else language)
        self.kv_len[slot] = 0
        self.conv[slot] = 0
        self.first[slot] = True
        self.tokens[slot] = []
        self.pred_out[slot] = self.init_state.pred_out[0]
        self.h[:, slot] = self.init_state.hc[0][:, 0]
        self.c[:, slot] = self.init_state.hc[1][:, 0]
        return slot

    def close(self, slot: int) -> None:
        self.free.append(slot)

    def chunk_mel_frames(self, first: bool) -> int:
        sf = self.cfg.subsampling_factor
        return 1 + sf * self.lookahead if first else sf * self.chunk

    # ------------------------------------------------------------------ encoder core
    def _encode(self, x, kb, vb, cb, kv_len, one_hot):
        """x (B, t, D) subsampled frames; kb/vb (B, L, H, C, dk) and cb (B, L, D, K-1) are read and updated in
        place; kv_len (B,) cached frames per stream. Returns the projected encoder output (B, t, pred_hidden)."""
        b, t = x.shape[0], x.shape[1]
        C = self.cache_len
        pos_ids = torch.arange(C, device=x.device)
        cache_valid = pos_ids[None, :] >= (C - kv_len)[:, None]
        key_valid = torch.cat([cache_valid, cache_valid.new_ones(b, t)], dim=1)
        att_mask = ~key_valid[:, None, :].expand(b, t, -1)
        pad_mask = torch.zeros(b, t, dtype=torch.bool, device=x.device)
        for li, layer in enumerate(self.model.encoder.layers):
            x, (kk, vv), conv_cache = layer(x, self.pos, att_mask, pad_mask, (kb[:, li], vb[:, li]), cb[:, li])
            kb[:, li] = kk[:, :, -C:]
            vb[:, li] = vv[:, :, -C:]
            cb[:, li] = conv_cache
        return self.model.encoder_projector(self.model.prompt_projector(x, one_hot))

    def _encode_fused(self, x, kb, vb, cb, kv_len, one_hot):
        """Like `_encode`, with kb/vb (B, L, H, C+T, dk): [:C] is the cache on entry, [C:] gets the chunk."""
        b, t = x.shape[0], x.shape[1]
        C, Kp = self.cache_len, self.keys_pad
        cols = torch.arange(Kp, device=x.device)
        valid = (cols[None, :] >= (C - kv_len)[:, None]) & (cols[None, :] < self.n_keys)
        key_masked = ~valid[:, None, None, :]
        pad_mask = torch.zeros(b, t, dtype=torch.bool, device=x.device)
        for li, layer in enumerate(self.model.encoder.layers):
            x, conv_cache = layer.forward_stream(x, kb[:, li], vb[:, li], self.pos_keys[li], key_masked,
                                                 pad_mask, cb[:, li])
            cb[:, li] = conv_cache
        return self.model.encoder_projector(self.model.prompt_projector(x, one_hot))

    def _enc_fn(self):
        return self._encode_fused if self.fused_attn else self._encode

    @torch.no_grad()
    def _graph(self, bp: int):
        if bp in self._graphs:
            return self._graphs[bp]
        args = (self._x[:bp], self._kb[:bp], self._vb[:bp], self._cb[:bp], self._kl[:bp], self._oh[:bp])
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._enc_fn()(*args)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = self._enc_fn()(*args)
        self._graphs[bp] = (g, y)
        return g, y

    @torch.no_grad()
    def capture_graphs(self) -> None:
        """Capture every bucket up front so no capture lands inside a timed run."""
        for bp in self.buckets:
            self._graph(bp)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, slots: list[int], mel: torch.Tensor) -> list[list[int]]:
        """mel (B, n, n_mels) for streams `slots` (same phase). Returns the tokens emitted per stream."""
        first = bool(self.first[slots[0]])
        assert all(bool(self.first[s]) == first for s in slots), "mixed first/later chunks in one batch"
        idx = torch.tensor(slots, device=self.device)
        enc = self.model.encoder
        sub = enc.subsampling
        k, s = sub.k, sub.s

        # causal subsampling with per-stream frame caches
        with self._sec("subsampling"):
            x = mel.to(self.dtype).unsqueeze(1)
            for i, conv in enumerate(sub.strided_convs()):
                x = conv.pad_freq(x)
                if first:
                    cache = x.new_zeros(x.shape[0], x.shape[1], k - 1, x.shape[3])
                else:
                    cache = self.sub[i][idx]
                x = torch.cat([cache, x], dim=2)
                if self.sub[i] is None:
                    self.sub[i] = x.new_zeros(self.k.shape[0], x.shape[1], k - s, x.shape[3])
                self.sub[i][idx] = x[:, :, -(k - s) :]
                x = F.conv2d(x, conv.weight, conv.bias, stride=conv.stride, groups=conv.groups)
                if i > 0:
                    x = sub.layers[i - 1].pointwise_conv(x)
                x = torch.relu(x)
            b, ch, t, f = x.shape
            x = sub.linear(x.transpose(1, 2).reshape(b, t, ch * f))

        kv_len = self.kv_len[idx]
        if self.cuda_graphs and b <= self.max_batch:
            assert t == self.chunk, (t, self.chunk)
            bp = next(n for n in self.buckets if n >= b)
            g, y_static = self._graph(bp)
            C, T = self.cache_len, self.chunk
            with self._sec("cache_gather"):
                self._x[:b].copy_(x)
                sl = slice(None, C) if self.fused_attn else slice(None)  # fused: [C:] is written by the graph
                torch.index_select(self.k, 0, idx, out=self._kb[:b, :, :, sl])
                torch.index_select(self.v, 0, idx, out=self._vb[:b, :, :, sl])
                torch.index_select(self.conv, 0, idx, out=self._cb[:b])
                self._kl[:b].copy_(kv_len)
                torch.index_select(self.lang, 0, idx, out=self._oh[:b])
            with self._sec("encoder_graph"):
                g.replay()
            with self._sec("cache_scatter"):
                sl = slice(T, None) if self.fused_attn else slice(None)  # fused: roll the cache by T here
                self.k.index_copy_(0, idx, self._kb[:b, :, :, sl])
                self.v.index_copy_(0, idx, self._vb[:b, :, :, sl])
                self.conv.index_copy_(0, idx, self._cb[:b])
            y = y_static[:b]
        else:
            with self._sec("cache_gather"):
                if self.fused_attn:
                    assert t == self.chunk, (t, self.chunk)
                    C, T = self.cache_len, t
                    kb = self.k.new_empty(b, *self.k.shape[1:3], C + T, self.k.shape[-1])
                    vb = torch.empty_like(kb)
                    kb[:, :, :, :C] = self.k[idx]
                    vb[:, :, :, :C] = self.v[idx]
                else:
                    kb, vb = self.k[idx], self.v[idx]
                cb = self.conv[idx]
            with self._sec("encoder_eager"):
                y = self._enc_fn()(x, kb, vb, cb, kv_len, self.lang[idx])
            with self._sec("cache_scatter"):
                if self.fused_attn:
                    kb, vb = kb[:, :, :, T:], vb[:, :, :, T:]
                self.k.index_copy_(0, idx, kb)
                self.v.index_copy_(0, idx, vb)
                self.conv.index_copy_(0, idx, cb)
        self.kv_len[idx] = torch.clamp(kv_len + t, max=self.cache_len)
        self.first[slots] = False

        with self._sec("dec_gather"):
            state = DecoderState(pred_out=self.pred_out[idx], hc=(self.h[:, idx], self.c[:, idx]), tokens=[[] for _ in slots])
        with self._sec("greedy_rnnt"):
            if self.fast_greedy:
                state = self.model.greedy.decode_full(y, state)
            else:
                state = self.model.greedy.decode(y, torch.full((b,), t, device=self.device), state)
        with self._sec("dec_scatter"):
            self.pred_out[idx] = state.pred_out
            self.h[:, idx] = state.hc[0]
            self.c[:, idx] = state.hc[1]
        for slot, new in zip(slots, state.tokens):
            self.tokens[slot].extend(new)
        return state.tokens
