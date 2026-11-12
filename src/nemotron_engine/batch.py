"""One batched step for many streams: every stream owns a slot in pooled caches on the GPU.

`submit(windows, idx, first)` copies the batch's raw sample windows and slot numbers into static device buffers
and replays one CUDA graph for the whole step: preemphasis + log-mel, subsampling, gather of the slots' caches,
the encoder (24 layers + prompt + projection), scatter of the caches, and `greedy_steps` steps of greedy RNN-T.
It then queues the labels' copy to pinned host memory. Nothing waits for the GPU, and the host does one replay
instead of hundreds of kernel launches (each of which releases and re-acquires the GIL). `collect()` waits for the
labels; while some stream has not finished its chunk, it replays a continuation graph of `chunk` more steps.

Greedy RNN-T (NeMo greedy_batch semantics, at most max_symbols_per_step symbols per frame) runs branch-free with a
frame pointer per stream: each step, a stream that is not done either emits a token, or (blank, or the symbol
limit reached) moves to its next frame. A stream needs T + (its tokens) steps; per stream this is the same
sequence of joint / prediction-network evaluations as the frame-synchronous loop, so the tokens are identical.

There is one graph per (phase, batch bucket), first chunks (fewer mel frames, no subsampling cache; usually small
batches) with power-of-two buckets only, and one continuation graph over all max_batch rows. Rows past the batch
point at a scratch slot, so their state updates land there. Graphs share one memory pool and are captured from the
largest bucket down, so the smaller ones reuse its blocks.

The attention cache always has C = left_chunks * chunk positions; a stream that has seen less audio keeps
stale values in the oldest positions and masks them, so relative positions stay exact. K/V buffers hold
C + T positions: the encoder writes the chunk's keys to [C:] and the scatter keeps [T:], which rolls the cache.
"""

from __future__ import annotations

import torch

from .model import NemotronModel


def _buckets(max_batch: int) -> dict[str, list[int]]:
    later = {1, 2, 4, 8, max_batch} | set(range(16, max_batch, 8))
    first = {1 << i for i in range(max_batch.bit_length())} | {max_batch}
    return {"later": sorted(b for b in later if b <= max_batch), "first": sorted(b for b in first if b <= max_batch),
            "continue": [max_batch]}


class BatchStepper:
    def __init__(self, model: NemotronModel, lookahead: int, max_streams: int, max_batch: int,
                 cuda_graphs: bool = True, greedy_steps: int = 12):
        cfg = self.cfg = model.cfg
        self.model = model
        T = self.chunk = lookahead + 1
        self.first_mel = 1 + cfg.subsampling_factor * lookahead
        self.later_mel = cfg.subsampling_factor * T
        self.greedy_steps = greedy_steps
        C = self.cache_len = (cfg.left_context // T) * T
        K = self.n_keys = C + T
        self.keys_pad = -(-K // 16) * 16  # SDPA bias rows aligned to 16 elements
        S, L, H, D = max_streams + 1, cfg.n_layers, cfg.n_heads, cfg.d_model
        self.scratch = max_streams  # slot of the padded rows
        dk, B, P = D // H, max_batch, cfg.pred_hidden
        p = model.encoder_projector.weight
        z = dict(device=p.device, dtype=p.dtype)
        i64 = dict(device=p.device, dtype=torch.long)
        self.device, self.dtype = p.device, p.dtype

        # per-slot state
        sub = model.encoder.subsampling
        self.sub = [torch.zeros(S, *shape, **z) for shape in sub.cache_shapes]
        self.k = torch.zeros(S, L, H, C, dk, **z)
        self.v = torch.zeros(S, L, H, C, dk, **z)
        self.conv = torch.zeros(S, L, D, cfg.conv_kernel - 1, **z)
        self.kv_len = torch.zeros(S, **i64)
        self.lang = torch.zeros(S, cfg.num_prompts, **z)
        self.pred_out = torch.zeros(S, P, **z)
        self.h = torch.zeros(cfg.pred_layers, S, P, **z)
        self.c = torch.zeros_like(self.h)
        blank = torch.full((1,), cfg.blank_id, **i64)
        h0 = torch.zeros(cfg.pred_layers, 1, P, **z)
        self.init_pred = model.decoder(blank, (h0, h0.clone()))  # state after <blank>

        # static graph buffers. Inputs: windows (raw samples plus the one before, see stream_audio), slot numbers.
        self.width = {first: (n - 1) * cfg.hop_length + cfg.n_fft + 1 for first, n in ((True, self.first_mel), (False, self.later_mel))}
        self._win = {first: torch.zeros(B, w, device=self.device) for first, w in self.width.items()}
        self._idx = torch.full((B,), self.scratch, **i64)
        self._kb = torch.zeros(B, L, H, K, dk, **z)
        self._vb = torch.zeros(B, L, H, K, dk, **z)
        pos = model.encoder.pos_emb(K).to(self.dtype)
        self.pos_keys = [l.self_attn.pos_keys(pos, T, K, self.keys_pad) for l in model.encoder.layers]
        # greedy state of the batch, carried across the step graph and continuation graphs
        self.max_labels = T * cfg.max_symbols_per_step
        self._y = torch.zeros(B, T, P, **z)
        self._p, self._h, self._c = torch.zeros(B, P, **z), torch.zeros(cfg.pred_layers, B, P, **z), torch.zeros(cfg.pred_layers, B, P, **z)
        self._t, self._sym, self._n = torch.zeros(B, **i64), torch.zeros(B, **i64), torch.zeros(B, **i64)
        self._lab = torch.full((B, self.max_labels + 1), -1, **i64)  # labels; last column: 1 while not done
        self._host = torch.empty(B, self.max_labels + 1, dtype=torch.long, pin_memory=True)
        self._done = torch.cuda.Event()
        self._pending: int | None = None  # rows of the submitted batch
        self.continuations = 0

        self.max_batch, self.buckets = B, _buckets(B)
        self.cuda_graphs = cuda_graphs
        self._graphs: dict[tuple[str, int], torch.cuda.CUDAGraph] = {}
        if cuda_graphs:
            pool = torch.cuda.graph_pool_handle()  # graphs replay one at a time, so they share memory
            for bp, kind in sorted(((bp, kind) for kind, bps in self.buckets.items() for bp in bps), reverse=True):
                self._capture(kind, bp, pool)
            torch.cuda.synchronize()

    # ---------------------------------------------------------------- slots
    def reset(self, slot: int, language) -> None:
        """Start a new stream in `slot`."""
        self.kv_len[slot] = 0
        self.conv[slot] = 0
        self.lang[slot] = self.model.one_hot(language)
        self.pred_out[slot] = self.init_pred[0][0]
        self.h[:, slot] = self.init_pred[1][0][:, 0]
        self.c[:, slot] = self.init_pred[1][1][:, 0]

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def submit(self, windows: torch.Tensor, idx: torch.Tensor, first: bool) -> None:
        """windows (B, width[first]) and idx (B,) slot numbers, pinned CPU tensors that stay untouched until
        `collect`. All first chunks or all later chunks. Queues the step and returns at once."""
        b = idx.shape[0]
        assert self._pending is None and b <= self.max_batch and windows.shape == (b, self.width[first])
        kind = "first" if first else "later"
        bp = next(n for n in self.buckets[kind] if n >= b)
        self._win[first][:b].copy_(windows, non_blocking=True)
        self._idx[:b].copy_(idx, non_blocking=True)
        self._idx[b:].fill_(self.scratch)  # also the rows a continuation runs past bp
        self._launch(kind, bp, b)
        self._pending = b

    def ready(self) -> bool:
        """Whether the submitted step's first labels are on the host (collect would not wait)."""
        return self._done.query()

    @torch.no_grad()
    def collect(self) -> list[list[int]]:
        """Wait for the submitted step; returns the tokens emitted per stream."""
        b, self._pending = self._pending, None
        L = self.max_labels
        while True:
            self._done.synchronize()
            rows = self._host[:b].tolist()
            if not any(r[L] for r in rows):
                return [[x for x in r[:L] if x >= 0] for r in rows]
            self.continuations += 1
            self._launch("continue", self.max_batch, b)

    def _launch(self, kind: str, bp: int, b: int) -> None:
        if self.cuda_graphs:
            self._graphs[kind, bp].replay()
        else:
            self._run(kind, bp)
        self._host[:b].copy_(self._lab[:b], non_blocking=True)
        self._done.record()

    def _run(self, kind: str, bp: int) -> None:
        """On the first bp rows of the static buffers: "first" / "later" = the encoder step, then greedy_steps
        greedy steps; "continue" = `chunk` more greedy steps. Slot state is updated in place."""
        if kind == "continue":
            return self._greedy(bp, self.chunk)
        first = kind == "first"
        m, C, T = self.model, self.cache_len, self.chunk
        idx = self._idx[:bp]
        mel = m.frontend(self._win[first][:bp]).to(self.dtype)
        x, sub = m.encoder.subsampling(mel, None if first else [s.index_select(0, idx) for s in self.sub])
        for pool, new in zip(self.sub, sub):
            pool.index_copy_(0, idx, new)

        kb, vb = self._kb[:bp], self._vb[:bp]
        torch.index_select(self.k, 0, idx, out=kb[:, :, :, :C])
        torch.index_select(self.v, 0, idx, out=vb[:, :, :, :C])
        conv = self.conv.index_select(0, idx)
        kv_len = self.kv_len.index_select(0, idx)
        cols = torch.arange(self.keys_pad, device=x.device)
        valid = (cols[None, :] >= (C - kv_len)[:, None]) & (cols[None, :] < self.n_keys)
        key_masked = ~valid[:, None, None, :]
        for li, layer in enumerate(m.encoder.layers):
            x, conv[:, li] = layer(x, kb[:, li], vb[:, li], self.pos_keys[li], key_masked, conv[:, li])
        self._y[:bp] = m.encoder_projector(m.prompt_projector(x, self.lang.index_select(0, idx)))
        self.k.index_copy_(0, idx, kb[:, :, :, T:])
        self.v.index_copy_(0, idx, vb[:, :, :, T:])
        self.conv.index_copy_(0, idx, conv)
        self.kv_len.index_copy_(0, idx, torch.clamp(kv_len + T, max=C))

        torch.index_select(self.pred_out, 0, idx, out=self._p[:bp])
        self._h[:, :bp] = self.h.index_select(1, idx)
        self._c[:, :bp] = self.c.index_select(1, idx)
        for buf in (self._t, self._sym, self._n):
            buf[:bp].zero_()
        self._lab[:bp].fill_(-1)
        self._greedy(bp, self.greedy_steps)

    def _greedy(self, bp: int, steps: int) -> None:
        """`steps` branch-free greedy steps on the batch's greedy state; then the decoder state goes back to the
        slots and the last label column flags the streams that have frames left."""
        cfg, m, T, L = self.cfg, self.model, self.chunk, self.max_labels
        y, lab, idx = self._y[:bp], self._lab[:bp], self._idx[:bp]
        p, h, c = self._p[:bp], self._h[:, :bp].contiguous(), self._c[:, :bp].contiguous()
        t, sym, n = self._t[:bp], self._sym[:bp], self._n[:bp]
        for _ in range(steps):
            done = t >= T
            f = y.gather(1, t.clamp(max=T - 1)[:, None, None].expand(-1, 1, y.shape[2]))[:, 0]
            k = m.joint(f, p).argmax(-1)
            emit = ~done & (k != cfg.blank_id)
            lab.scatter_(1, torch.where(emit, n, L)[:, None], k[:, None])
            n = n + emit
            new_out, (nh, nc) = m.decoder(k.masked_fill(~emit, cfg.blank_id), (h, c))
            e = emit[:, None]
            p = torch.where(e, new_out, p)
            h = torch.where(e[None], nh, h)
            c = torch.where(e[None], nc, c)
            sym = sym + emit
            nxt = ~done & (~emit | (sym >= cfg.max_symbols_per_step))  # blank or symbol limit: next frame
            sym = torch.where(nxt, 0, sym)
            t = t + nxt
        lab[:, L] = t < T
        self._p[:bp], self._h[:, :bp], self._c[:, :bp] = p, h, c
        self._t[:bp], self._sym[:bp], self._n[:bp] = t, sym, n
        self.pred_out.index_copy_(0, idx, p)
        self.h.index_copy_(1, idx, h)
        self.c.index_copy_(1, idx, c)

    @torch.no_grad()
    def _capture(self, kind: str, bp: int, pool) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm-up (allocator, cuBLAS / cuFFT / SDPA kernel selection); writes the scratch slot
                self._run(kind, bp)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.empty_cache()  # give the warm-up's cached blocks back, so the graph pool can use that memory
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            self._run(kind, bp)
        self._graphs[kind, bp] = g
