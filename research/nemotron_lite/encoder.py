"""Cache-aware FastConformer encoder (causal dw-striding subsampling, chunked_limited rel-pos attention).

Module names mirror the keys of the HF `model.safetensors`, so weights load with a strict state_dict.
The offline path follows NeMo `ConformerEncoder.forward_internal`; `stream_step` is the cache-aware
equivalent that gives the same frames when fed fixed-size chunks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .config import NemotronLiteConfig

NEG_INF = -10000.0  # NeMo INF_VAL; masked rows are zeroed after softmax


def _time_mask(x: torch.Tensor, lengths: torch.Tensor, start: int | torch.Tensor = 0) -> torch.Tensor:
    """Zero frames whose global index (start + local index) >= lengths. x: (B, C, T, F)."""
    t = torch.arange(x.shape[2], device=x.device)[None, :] + start
    return x * (t < lengths[:, None]).to(x.dtype)[:, None, :, None]


def _conv_out_len(lengths: torch.Tensor, kernel: int, stride: int) -> torch.Tensor:
    # causal padding: left = kernel - 1, right = stride - 1
    return torch.div(lengths + (kernel - 1) + (stride - 1) - kernel, stride, rounding_mode="floor") + 1


class SubsamplingConv2d(nn.Conv2d):
    def pad_freq(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.kernel_size[1] - 1, self.stride[1] - 1))


class SubsamplingLayer(nn.Module):
    def __init__(self, ch: int, k: int, s: int):
        super().__init__()
        self.depthwise_conv = SubsamplingConv2d(ch, ch, k, stride=s, groups=ch)
        self.pointwise_conv = nn.Conv2d(ch, ch, kernel_size=1)


class Subsampling(nn.Module):
    def __init__(self, cfg: NemotronLiteConfig):
        super().__init__()
        ch, k, s = cfg.subsampling_channels, cfg.subsampling_kernel, cfg.subsampling_stride
        self.k, self.s = k, s
        n = int(math.log2(cfg.subsampling_factor))
        self.conv_in = SubsamplingConv2d(1, ch, k, stride=s)
        self.layers = nn.ModuleList(SubsamplingLayer(ch, k, s) for _ in range(n - 1))
        freq = cfg.n_mels
        for _ in range(n):
            freq = (freq + (k - 1) + (s - 1) - k) // s + 1
        self.linear = nn.Linear(ch * freq, cfg.d_model)

    def strided_convs(self) -> list[SubsamplingConv2d]:
        return [self.conv_in] + [layer.depthwise_conv for layer in self.layers]

    def _run(self, x, lengths, time_pad, starts=None):
        """Shared body. `time_pad(i, x)` adds the time padding (or cache) in front of strided conv i.
        `starts[i]` is the global index of the first output frame of conv i (streaming), else 0."""
        convs = self.strided_convs()
        for i, conv in enumerate(convs):
            x = time_pad(i, conv.pad_freq(x))
            x = F.conv2d(x, conv.weight, conv.bias, stride=conv.stride, groups=conv.groups)
            lengths = _conv_out_len(lengths, self.k, self.s)
            start = 0 if starts is None else starts[i]
            if i > 0:
                x = self.layers[i - 1].pointwise_conv(x)
            x = torch.relu(_time_mask(x, lengths, start))
        b, c, t, f = x.shape
        return self.linear(x.transpose(1, 2).reshape(b, t, c * f)), lengths

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        """feats (B, T, n_mels) -> (B, T', d_model), lengths'."""
        x = _time_mask(feats.unsqueeze(1), lengths)
        pad = (self.k - 1, self.s - 1)
        return self._run(x, lengths, lambda i, h: F.pad(h, (0, 0, *pad)))


class FeedForward(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.linear1 = nn.Linear(d, hidden, bias=False)
        self.linear2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class RelPosAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h, self.dk = n_heads, d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.relative_k_proj = nn.Linear(d, d, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(n_heads, self.dk))
        self.bias_v = nn.Parameter(torch.zeros(n_heads, self.dk))

    def _heads(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dk).transpose(1, 2)  # (B, H, T, dk)

    @staticmethod
    def _rel_shift(x):
        b, h, q, p = x.shape
        x = F.pad(x, (1, 0)).view(b, h, -1, q)
        return x[:, :, 1:].reshape(b, h, q, p)

    def forward(self, x, pos_emb, mask, kv_cache=None):
        """x (B, T, d) normalized input; pos_emb (1, 2L-1, d) for L = cache + T keys;
        mask (B, T, L) bool, True = masked; kv_cache (k, v) each (B, H, Tc, dk) or None.
        Returns output and the full (k, v) for this call (cache + current)."""
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        p = self._heads(self.relative_k_proj(pos_emb))  # (1, H, 2L-1, dk)
        q_u = q + self.bias_u[None, :, None, :]
        q_v = q + self.bias_v[None, :, None, :]
        scale = 1.0 / math.sqrt(self.dk)
        bd = self._rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))[..., : k.shape[2]] * scale
        m = mask[:, None]
        bd = bd.masked_fill(m, NEG_INF)
        scores = torch.matmul(q_u, k.transpose(-2, -1)) * scale + bd
        attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.o_proj(out), (k, v)

    # ------------------------------------------------------------ fused streaming path
    @torch.no_grad()
    def stream_pos_keys(self, pos_emb: torch.Tensor, t: int, n_keys: int, pad_to: int) -> torch.Tensor:
        """Relative-position keys for a streaming chunk, already shifted: query i (the i-th of the last t of
        n_keys positions) and key j use pos_emb row t-1-i+j (what `_rel_shift` selects). pos_emb is
        input-independent, so this is computed once. Returns (H, t, dk, pad_to); columns >= n_keys are 0."""
        p = self._heads(self.relative_k_proj(pos_emb))[0]  # (H, 2L-1, dk)
        i = torch.arange(t, device=p.device)[:, None]
        j = torch.arange(n_keys, device=p.device)[None, :]
        sel = p[:, t - 1 - i + j]  # (H, t, n_keys, dk)
        out = sel.new_zeros(self.h, t, self.dk, pad_to)
        out[..., :n_keys] = sel.transpose(-2, -1)
        return out.contiguous()

    def forward_stream(self, x, kbuf, vbuf, pos_keys, key_masked):
        """One streaming chunk with the attention core in a single fused SDPA kernel.

        x (B, T, d) normalized input. kbuf / vbuf (B, H, C+T, dk): positions [:C] hold the cache on entry;
        the chunk's keys / values are written to [C:] (the caller keeps [T:] as the next cache, so no
        concat and no write-back here). pos_keys from `stream_pos_keys`; key_masked (B, 1, 1, Kp) bool,
        True = invalid key (Kp >= C+T, padded columns True). Same result as `forward` up to rounding:
        the bias is the same rel-pos term with masked keys set to NEG_INF, and no query row is fully
        masked (the chunk's own keys are always valid), so the post-softmax zeroing is a no-op."""
        b, t, _ = x.shape
        n_keys = kbuf.shape[2]
        if not hasattr(self, "_w_qkv"):
            self._w_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], 0)
        q, k, v = F.linear(x, self._w_qkv).view(b, t, 3, self.h, self.dk).unbind(2)
        kbuf[:, :, n_keys - t :] = k.transpose(1, 2)
        vbuf[:, :, n_keys - t :] = v.transpose(1, 2)
        q_u = (q + self.bias_u).transpose(1, 2)  # (B, H, T, dk)
        q_v = (q + self.bias_v).permute(2, 1, 0, 3)  # (H, T, B, dk)
        scale = 1.0 / math.sqrt(self.dk)
        bd = torch.matmul(q_v, pos_keys).permute(2, 0, 1, 3)  # (B, H, T, Kp)
        bias = torch.where(key_masked, NEG_INF, bd * scale)
        out = F.scaled_dot_product_attention(q_u, kbuf, vbuf, attn_mask=bias[..., :n_keys], scale=scale)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.pointwise_conv2 = nn.Conv1d(d, d, 1, bias=False)

    # The 1x1 convs are GEMMs over channels. cuDNN runs them ~10x slower than F.linear for short chunks
    # (T = 4 frames when streaming), so they go through F.linear on (B, T, C) with the same weights.
    use_linear = True

    def forward(self, x, pad_mask, cache=None):
        """x (B, T, d); pad_mask (B, T) True = padding; cache (B, d, kernel-1) or None (zeros).
        Returns output and the new cache (last kernel-1 GLU frames)."""
        if self.use_linear:
            h = F.glu(F.linear(x, self.pointwise_conv1.weight[:, :, 0]), dim=-1)
            h = h.masked_fill(pad_mask[:, :, None], 0.0).transpose(1, 2)
        else:
            h = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
            h = h.masked_fill(pad_mask[:, None, :], 0.0)
        if cache is None:
            h = F.pad(h, (self.kernel - 1, 0))
        else:
            h = torch.cat([cache, h], dim=2)
        new_cache = h[:, :, -(self.kernel - 1) :]
        h = self.depthwise_conv(h)
        h = self.norm(h.transpose(1, 2))
        if self.use_linear:
            return F.linear(F.silu(h), self.pointwise_conv2.weight[:, :, 0]), new_cache
        h = self.pointwise_conv2(F.silu(h).transpose(1, 2))
        return h.transpose(1, 2), new_cache


class ConformerBlock(nn.Module):
    def __init__(self, cfg: NemotronLiteConfig):
        super().__init__()
        d = cfg.d_model
        self.feed_forward1 = FeedForward(d, cfg.ff_dim)
        self.self_attn = RelPosAttention(d, cfg.n_heads)
        self.conv = ConvModule(d, cfg.conv_kernel)
        self.feed_forward2 = FeedForward(d, cfg.ff_dim)
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.norm_self_att = nn.LayerNorm(d)
        self.norm_conv = nn.LayerNorm(d)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x, pos_emb, att_mask, pad_mask, kv_cache=None, conv_cache=None):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        a, kv = self.self_attn(self.norm_self_att(x), pos_emb, att_mask, kv_cache)
        x = x + a
        c, conv_cache = self.conv(self.norm_conv(x), pad_mask, conv_cache)
        x = x + c
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x), kv, conv_cache

    def forward_stream(self, x, kbuf, vbuf, pos_keys, key_masked, pad_mask, conv_cache):
        """Streaming chunk with fused attention; see `RelPosAttention.forward_stream`."""
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn.forward_stream(self.norm_self_att(x), kbuf, vbuf, pos_keys, key_masked)
        c, conv_cache = self.conv(self.norm_conv(x), pad_mask, conv_cache)
        x = x + c
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x), conv_cache


@dataclass
class EncoderStreamState:
    """Caches for one batch of streams. Encoder frames are produced `lookahead + 1` at a time."""

    lookahead: int
    sub_cache: list[torch.Tensor | None] = field(default_factory=list)  # per strided conv
    sub_frames_out: list[int] = field(default_factory=list)  # global frames emitted per strided conv
    kv: list[tuple[torch.Tensor, torch.Tensor] | None] = field(default_factory=list)
    conv: list[torch.Tensor | None] = field(default_factory=list)
    mel_frames_in: int = 0
    frames_out: int = 0


class Encoder(nn.Module):
    def __init__(self, cfg: NemotronLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.subsampling = Subsampling(cfg)
        self.layers = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, cfg.d_model, 2, dtype=torch.float32) / cfg.d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def pos_emb(self, length: int, device, dtype) -> torch.Tensor:
        """Relative positions length-1 .. -(length-1), interleaved sin/cos -> (1, 2*length-1, d)."""
        pos = torch.arange(length - 1, -length, -1, device=device, dtype=torch.float32)
        ang = pos[:, None] * self.inv_freq.to(device)[None, :]
        pe = torch.stack([ang.sin(), ang.cos()], dim=-1).reshape(1, 2 * length - 1, -1)
        return pe.to(dtype)

    def _chunk_params(self, lookahead: int) -> tuple[int, int]:
        chunk = lookahead + 1
        left_chunks = self.cfg.left_context // chunk
        return chunk, left_chunks

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor, lookahead: int):
        """Offline: feats (B, T, n_mels) -> (B, T', d_model), lengths (B,)."""
        x, lengths = self.subsampling(feats, feat_lens)
        b, t, _ = x.shape
        chunk, left_chunks = self._chunk_params(lookahead)
        idx = torch.arange(t, device=x.device)
        cidx = torch.div(idx, chunk, rounding_mode="floor")
        diff = cidx[:, None] - cidx[None, :]
        allowed = (diff >= 0) & (diff <= left_chunks)
        valid = idx[None, :] < lengths[:, None]  # (B, T)
        att_mask = ~(allowed[None] & valid[:, None, :] & valid[:, :, None])
        pad_mask = ~valid
        pos = self.pos_emb(t, x.device, x.dtype)
        for layer in self.layers:
            x, _, _ = layer(x, pos, att_mask, pad_mask)
        return x, lengths

    # ---------------------------------------------------------------- streaming
    def init_stream(self, lookahead: int) -> EncoderStreamState:
        n_conv = len(self.subsampling.strided_convs())
        return EncoderStreamState(
            lookahead=lookahead,
            sub_cache=[None] * n_conv,
            sub_frames_out=[0] * n_conv,
            kv=[None] * len(self.layers),
            conv=[None] * len(self.layers),
        )

    def chunk_mel_frames(self, state: EncoderStreamState) -> int:
        """Mel frames the next chunk must carry: 1 + sf*r for the first, sf*(r+1) afterwards."""
        sf, r = self.cfg.subsampling_factor, state.lookahead
        return 1 + sf * r if state.mel_frames_in == 0 else sf * (r + 1)

    def stream_step(self, state: EncoderStreamState, feats: torch.Tensor, total_mel_frames: int | None = None):
        """Encode one chunk of mel frames (B, n, n_mels), n == chunk_mel_frames(state).

        `total_mel_frames`, known once the stream is finished, masks frames past the end exactly like the
        offline length masks. Returns (B, T_new, d_model) and the number of valid frames in it.
        """
        sub = self.subsampling
        k, s = sub.k, sub.s
        b = feats.shape[0]
        mel_start = state.mel_frames_in
        state.mel_frames_in += feats.shape[1]

        # global valid length after each strided conv (huge when the stream is still open)
        big = 1 << 40
        n_conv = len(sub.strided_convs())
        lens = []
        cur = torch.full((b,), total_mel_frames if total_mel_frames is not None else big, device=feats.device)
        for _ in range(n_conv):
            cur = _conv_out_len(cur, k, s) if total_mel_frames is not None else cur
            lens.append(cur)
        mel_len = torch.full((b,), total_mel_frames if total_mel_frames is not None else big, device=feats.device)

        def time_pad(i, h):
            cache = state.sub_cache[i]
            if cache is None:  # first chunk: the offline causal left padding
                cache = h.new_zeros(h.shape[0], h.shape[1], k - 1, h.shape[3])
            out = torch.cat([cache, h], dim=2)
            state.sub_cache[i] = out[:, :, -(k - s) :]
            return out

        x = _time_mask(feats.unsqueeze(1), mel_len, mel_start)
        starts = list(state.sub_frames_out)
        convs = sub.strided_convs()
        for i, conv in enumerate(convs):
            x = time_pad(i, conv.pad_freq(x))
            x = F.conv2d(x, conv.weight, conv.bias, stride=conv.stride, groups=conv.groups)
            if i > 0:
                x = sub.layers[i - 1].pointwise_conv(x)
            x = torch.relu(_time_mask(x, lens[i], starts[i]))
            state.sub_frames_out[i] += x.shape[2]
        bb, c, t, f = x.shape
        x = sub.linear(x.transpose(1, 2).reshape(bb, t, c * f))

        start = state.frames_out
        state.frames_out += t
        enc_len = lens[-1] if total_mel_frames is not None else torch.full((b,), big, device=x.device)
        valid = (torch.arange(t, device=x.device)[None, :] + start) < enc_len[:, None]  # (B, T)
        n_valid = int(valid.sum(1).max().item()) if total_mel_frames is not None else t

        chunk, left_chunks = self._chunk_params(state.lookahead)
        max_cache = left_chunks * chunk
        cache_len = 0 if state.kv[0] is None else state.kv[0][0].shape[2]
        # every query of this chunk sees the whole cache and the valid frames of the chunk
        key_valid = torch.cat([valid.new_ones(b, cache_len), valid], dim=1)  # (B, Tc+T)
        att_mask = ~(key_valid[:, None, :] & valid[:, :, None])
        pad_mask = ~valid
        pos = self.pos_emb(cache_len + t, x.device, x.dtype)
        for li, layer in enumerate(self.layers):
            x, (kk, vv), conv_cache = layer(x, pos, att_mask, pad_mask, state.kv[li], state.conv[li])
            state.kv[li] = (kk[:, :, -max_cache:], vv[:, :, -max_cache:])
            state.conv[li] = conv_cache
        return x, n_valid
