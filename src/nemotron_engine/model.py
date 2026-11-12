"""nvidia/nemotron-3.5-asr-streaming-0.6b, streaming inference only.

Module names mirror the keys of the HF `model.safetensors`, so the checkpoint loads with a strict state_dict.
Only the cache-aware chunk path is implemented: causal dw-striding subsampling with per-stream frame caches,
24 FastConformer layers with rel-pos attention as one SDPA call over (cache + chunk) keys, language prompt
fusion, and frame-synchronous greedy RNN-T (NeMo greedy_batch semantics).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig

MODEL_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_FILES = ("config.json", "processor_config.json", "tokenizer.json", "model.safetensors")
NEG_INF = -10000.0  # NeMo INF_VAL for masked attention scores
LOG_ZERO_GUARD = 2.0**-24


# ------------------------------------------------------------------------------------------- front-end
def _slaney_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney")."""
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, math.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp

    def hz_to_mel(f):
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)

    def mel_to_hz(m):
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(np.array(0.0)), hz_to_mel(np.array(sample_rate / 2.0)), n_mels + 2))
    fdiff, ramps = np.diff(mel_f), np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights


class LogMel(nn.Module):
    """Raw samples -> preemphasis -> log-mel frames, STFT without centering (the caller supplies the context)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("fb", torch.from_numpy(_slaney_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels)), persistent=False)
        self.register_buffer("window", torch.hann_window(cfg.win_length, periodic=False), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, (n-1)*hop + n_fft + 1) float32, the first column the sample before the window -> (B, n, n_mels)."""
        c = self.cfg
        x = x[:, 1:] - x[:, :-1] * c.preemphasis  # two kernels, so the rounding matches x - k*prev in numpy
        spec = torch.stft(x, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length, window=self.window,
                          center=False, return_complex=True)
        power = torch.view_as_real(spec).pow(2).sum(-1).sqrt().pow(2)
        return torch.log(torch.matmul(self.fb, power) + LOG_ZERO_GUARD).transpose(1, 2)


# --------------------------------------------------------------------------------------------- encoder
class SubsamplingConv2d(nn.Conv2d):
    def pad_freq(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.kernel_size[1] - 1, self.stride[1] - 1))


class SubsamplingLayer(nn.Module):
    def __init__(self, ch: int, k: int, s: int):
        super().__init__()
        self.depthwise_conv = SubsamplingConv2d(ch, ch, k, stride=s, groups=ch)
        self.pointwise_conv = nn.Conv2d(ch, ch, kernel_size=1)


class Subsampling(nn.Module):
    """Causal dw-striding subsampling x8. In time, each strided conv sees the last k-s input frames of the
    previous chunk (k-1 zeros before the first chunk)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ch, k, s = cfg.subsampling_channels, cfg.subsampling_kernel, cfg.subsampling_stride
        self.k, self.s = k, s
        n = int(math.log2(cfg.subsampling_factor))
        self.conv_in = SubsamplingConv2d(1, ch, k, stride=s)
        self.layers = nn.ModuleList(SubsamplingLayer(ch, k, s) for _ in range(n - 1))
        self.cache_shapes = []  # (channels, k - s, padded freq) per strided conv
        freq, cin = cfg.n_mels, 1
        for _ in range(n):
            self.cache_shapes.append((cin, k - s, freq + k + s - 2))
            freq, cin = (freq + s - 2) // s + 1, ch
        self.linear = nn.Linear(ch * freq, cfg.d_model)

    def strided_convs(self) -> list[SubsamplingConv2d]:
        return [self.conv_in] + [layer.depthwise_conv for layer in self.layers]

    def forward(self, mel: torch.Tensor, caches: list[torch.Tensor] | None):
        """mel (B, n, n_mels); caches per conv (B, C, k-s, F), None for first chunks.
        Returns (B, t, d_model) and the new caches."""
        x, new = mel.unsqueeze(1), []
        for i, conv in enumerate(self.strided_convs()):
            x = conv.pad_freq(x)
            prev = x.new_zeros(x.shape[0], x.shape[1], self.k - 1, x.shape[3]) if caches is None else caches[i]
            x = torch.cat([prev, x], dim=2)
            new.append(x[:, :, -(self.k - self.s) :])
            x = F.conv2d(x, conv.weight, conv.bias, stride=conv.stride, groups=conv.groups)
            if i > 0:
                x = self.layers[i - 1].pointwise_conv(x)
            x = torch.relu(x)
        b, c, t, f = x.shape
        return self.linear(x.transpose(1, 2).reshape(b, t, c * f)), new


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
        self.w_qkv: torch.Tensor | None = None  # set by fuse_qkv() after loading

    def fuse_qkv(self) -> None:
        self.w_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], 0).detach()

    @torch.no_grad()
    def pos_keys(self, pos_emb: torch.Tensor, t: int, n_keys: int, pad_to: int) -> torch.Tensor:
        """Relative-position keys, already shifted: query i (the i-th of the last t of n_keys positions) and key j
        use pos_emb row t-1-i+j (Transformer-XL rel_shift). Input-independent, so computed once.
        Returns (H, t, dk, pad_to); columns >= n_keys are 0."""
        p = self.relative_k_proj(pos_emb).view(-1, self.h, self.dk).transpose(0, 1)  # (H, 2L-1, dk)
        i = torch.arange(t, device=p.device)[:, None]
        j = torch.arange(n_keys, device=p.device)[None, :]
        out = p.new_zeros(self.h, t, self.dk, pad_to)
        out[..., :n_keys] = p[:, t - 1 - i + j].transpose(-2, -1)
        return out

    def forward(self, x, kbuf, vbuf, pos_keys, key_masked):
        """x (B, T, d). kbuf / vbuf (B, H, C+T, dk) hold the cache in [:C]; the chunk's keys / values are written
        to [C:] (the caller keeps [T:] as the next cache). key_masked (B, 1, 1, Kp) bool, True = invalid key.
        No query row is fully masked (the chunk's own keys are valid), so softmax needs no post-masking."""
        b, t, _ = x.shape
        n_keys = kbuf.shape[2]
        q, k, v = F.linear(x, self.w_qkv).view(b, t, 3, self.h, self.dk).unbind(2)
        kbuf[:, :, n_keys - t :] = k.transpose(1, 2)
        vbuf[:, :, n_keys - t :] = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.dk)
        bd = torch.matmul((q + self.bias_v).permute(2, 1, 0, 3), pos_keys).permute(2, 0, 1, 3)  # (B, H, T, Kp)
        bias = torch.where(key_masked, NEG_INF, bd * scale)
        out = F.scaled_dot_product_attention((q + self.bias_u).transpose(1, 2), kbuf, vbuf,
                                             attn_mask=bias[..., :n_keys], scale=scale)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1, bias=False)  # run as F.linear (cuDNN is slow for T=4)
        self.depthwise_conv = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.pointwise_conv2 = nn.Conv1d(d, d, 1, bias=False)

    def forward(self, x, cache):
        """x (B, T, d); cache (B, d, kernel-1) of the previous GLU outputs. Returns output and the new cache."""
        h = F.glu(F.linear(x, self.pointwise_conv1.weight[:, :, 0]), dim=-1).transpose(1, 2)
        h = torch.cat([cache, h], dim=2)
        new_cache = h[:, :, -cache.shape[2] :]
        h = self.norm(self.depthwise_conv(h).transpose(1, 2))
        return F.linear(F.silu(h), self.pointwise_conv2.weight[:, :, 0]), new_cache


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
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

    def forward(self, x, kbuf, vbuf, pos_keys, key_masked, conv_cache):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), kbuf, vbuf, pos_keys, key_masked)
        c, conv_cache = self.conv(self.norm_conv(x), conv_cache)
        x = x + c
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x), conv_cache


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.subsampling = Subsampling(cfg)
        self.layers = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, cfg.d_model, 2, dtype=torch.float32) / cfg.d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def pos_emb(self, length: int) -> torch.Tensor:
        """Relative positions length-1 .. -(length-1), interleaved sin/cos -> (2*length-1, d), float32."""
        pos = torch.arange(length - 1, -length, -1, device=self.inv_freq.device, dtype=torch.float32)
        ang = pos[:, None] * self.inv_freq.float()[None, :]
        return torch.stack([ang.sin(), ang.cos()], dim=-1).reshape(2 * length - 1, -1)


class PromptProjector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(cfg.d_model + cfg.num_prompts, cfg.prompt_hidden)
        self.linear_2 = nn.Linear(cfg.prompt_hidden, cfg.d_model)

    def forward(self, x: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, one_hot[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        return self.linear_2(torch.relu(self.linear_1(h)))


# --------------------------------------------------------------------------------------------- decoder
class PredictionNetwork(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.pred_hidden)
        self.lstm = nn.LSTM(cfg.pred_hidden, cfg.pred_hidden, num_layers=cfg.pred_layers, batch_first=True)
        self.decoder_projector = nn.Linear(cfg.pred_hidden, cfg.pred_hidden)

    def forward(self, tokens: torch.Tensor, state):
        """tokens (B,) -> projected output (B, H), new (h, c)."""
        out, state = self.lstm(self.embedding(tokens)[:, None, :], state)
        return self.decoder_projector(out[:, 0]), state


class Joint(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.head = nn.Linear(cfg.pred_hidden, cfg.vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(enc + pred))


class Tokenizer:
    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as HFTokenizer

        self.tok = HFTokenizer.from_file(str(path))

    def decode(self, ids: list[int]) -> str:
        # SentencePiece (NeMo) renders the unknown piece as "⁇", HF tokenizers as "<unk>"
        return self.tok.decode(ids, skip_special_tokens=False).replace("<unk>", "⁇ ")


class NemotronModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = LogMel(cfg)
        self.encoder = Encoder(cfg)
        self.prompt_projector = PromptProjector(cfg)
        self.encoder_projector = nn.Linear(cfg.d_model, cfg.pred_hidden)
        self.decoder = PredictionNetwork(cfg)
        self.joint = Joint(cfg)
        self.tokenizer: Tokenizer | None = None

    @classmethod
    def load(cls, model_dir: str | Path | None = None, device: str = "cuda", dtype=torch.float32) -> "NemotronModel":
        """From a directory with MODEL_FILES, or downloaded from the HF hub when model_dir is None."""
        if model_dir is None:
            from huggingface_hub import hf_hub_download

            model_dir = Path(hf_hub_download(MODEL_REPO, "model.safetensors")).parent
            for name in MODEL_FILES:
                hf_hub_download(MODEL_REPO, name)
        model_dir = Path(model_dir)
        from safetensors.torch import load_file

        model = cls(ModelConfig.from_dir(model_dir))
        model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
        model.tokenizer = Tokenizer(model_dir / "tokenizer.json")
        model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        model.frontend.float()  # front-end and positions always run in float32
        model.encoder.inv_freq = model.encoder.inv_freq.float()
        for layer in model.encoder.layers:
            layer.self_attn.fuse_qkv()
        return model

    def one_hot(self, language: str | int | None) -> torch.Tensor:
        p = self.encoder_projector.weight
        return F.one_hot(torch.tensor(self.cfg.prompt_id(language), device=p.device), self.cfg.num_prompts).to(p.dtype)
