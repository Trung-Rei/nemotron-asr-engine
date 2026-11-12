"""Log-mel front-end matching NeMo's AudioToMelSpectrogramPreprocessor for this checkpoint.

preemphasis -> STFT (hann, periodic=False, center=True, zero padded) -> power -> slaney mel -> log(x + 2**-24).
No per-feature normalization, no dither at inference.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .config import NemotronLiteConfig

LOG_ZERO_GUARD = 2.0**-24


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    # librosa slaney mel scale (htk=False)
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = freq >= min_log_hz
    mels = np.where(log_t, min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    return np.where(log_t, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


def slaney_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Same numbers as `librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney")`."""
    fmin, fmax = 0.0, sample_rate / 2.0
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


class LogMelFeatures(nn.Module):
    def __init__(self, cfg: NemotronLiteConfig):
        super().__init__()
        self.cfg = cfg
        fb = torch.from_numpy(slaney_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels))
        self.register_buffer("fb", fb, persistent=False)  # (n_mels, n_fft//2+1)
        self.register_buffer("window", torch.hann_window(cfg.win_length, periodic=False), persistent=False)

    def preemphasize(self, audio: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        k = self.cfg.preemphasis
        out = torch.cat([audio[:, :1], audio[:, 1:] - k * audio[:, :-1]], dim=1)
        mask = torch.arange(audio.shape[1], device=audio.device)[None, :] < lengths[:, None]
        return out.masked_fill(~mask, 0.0)

    def _logmel(self, x: torch.Tensor, center: bool) -> torch.Tensor:
        spec = torch.stft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window.to(x.dtype),
            center=center,
            pad_mode="constant",
            return_complex=True,
        )
        power = torch.view_as_real(spec).pow(2).sum(-1).sqrt().pow(2)
        mel = torch.matmul(self.fb.to(power.dtype), power)
        return torch.log(mel + LOG_ZERO_GUARD)  # (B, n_mels, frames)

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """audio (B, S) float32 zero padded -> features (B, T, n_mels), feature lengths (B,)."""
        x = self.preemphasize(audio.float(), lengths)
        feats = self._logmel(x, center=True)
        feat_lens = torch.div(lengths, self.cfg.hop_length, rounding_mode="floor")
        mask = torch.arange(feats.shape[-1], device=feats.device)[None, :] < feat_lens[:, None]
        feats = feats.masked_fill(~mask[:, None, :], 0.0)
        return feats.transpose(1, 2), feat_lens


class StreamingFeatures:
    """Incremental log-mel for one stream, frame-for-frame equal to `LogMelFeatures` on the whole audio.

    Frame j of the centered STFT covers preemphasized samples [j*hop - n_fft/2, j*hop + n_fft/2).
    """

    def __init__(self, frontend: LogMelFeatures):
        self.frontend = frontend
        self.cfg = frontend.cfg
        self.buffer = torch.zeros(0)  # preemphasized samples from global index `self.offset`
        self.offset = 0
        self.total_samples = 0
        self.last_sample: float | None = None
        self.frames_done = 0
        self.finished = False

    def accept(self, samples: torch.Tensor) -> None:
        samples = samples.float().flatten().cpu()
        if samples.numel() == 0:
            return
        prev = samples.new_tensor([self.last_sample]) if self.last_sample is not None else samples[:1]
        shifted = torch.cat([prev, samples[:-1]])
        pre = samples - self.cfg.preemphasis * shifted
        if self.last_sample is None:
            pre[0] = samples[0]
        self.last_sample = float(samples[-1])
        self.buffer = torch.cat([self.buffer, pre])
        self.total_samples += samples.numel()

    def finish(self) -> None:
        self.finished = True

    def num_valid_frames(self) -> int | None:
        """Total number of feature frames once the stream is finished (None while open)."""
        return self.total_samples // self.cfg.hop_length if self.finished else None

    def available_frames(self) -> int:
        """How many frames (from frame 0) can be computed with the samples seen so far."""
        hop, half = self.cfg.hop_length, self.cfg.n_fft // 2
        if self.finished:
            return self.total_samples // hop
        # frame j needs samples up to j*hop + half (exclusive)
        return max(0, (self.total_samples - half) // hop + 1)

    def take(self, n_frames: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Compute the next `n_frames` frames -> (n_frames, n_mels). Frames past the end are zeros."""
        seg, n_real = self.take_segment(n_frames)
        out = torch.zeros(n_frames, self.cfg.n_mels, device=device)
        if n_real > 0:
            feats = self.frontend._logmel(seg[None].to(device), center=False)[0]  # (n_mels, n_real)
            out[:n_real] = feats.transpose(0, 1)
        return out

    def segment_len(self, n_frames: int) -> int:
        """Samples `take_segment` returns for n_frames complete frames (same for every stream)."""
        return (n_frames - 1) * self.cfg.hop_length + self.cfg.n_fft

    def take_segment(self, n_frames: int) -> tuple[torch.Tensor, int]:
        """Advance by `n_frames` frames and return the preemphasized samples they cover plus how many of them
        are real (the rest are past the end of a finished stream). With all frames real, the segment has
        `segment_len(n_frames)` samples, so segments of several streams stack into one batched `_logmel`."""
        hop, half = self.cfg.hop_length, self.cfg.n_fft // 2
        j0 = self.frames_done
        n_real = min(n_frames, max(0, self.available_frames() - j0))
        seg = self._slice(j0 * hop - half, (j0 + n_real - 1) * hop + half) if n_real > 0 else torch.zeros(0)
        self.frames_done += n_frames
        self._trim()
        return seg, n_real

    def _slice(self, start: int, end: int) -> torch.Tensor:
        """Preemphasized samples [start, end) in global indices, zero outside [0, total_samples)."""
        lo, hi = start - self.offset, end - self.offset
        left = max(0, -start)
        right = max(0, end - self.total_samples)
        seg = self.buffer[max(lo, 0) : max(min(hi, self.buffer.numel()), 0)]
        return torch.nn.functional.pad(seg, (left, right))

    def _trim(self) -> None:
        keep_from = self.frames_done * self.cfg.hop_length - self.cfg.n_fft // 2
        drop = keep_from - self.offset
        if drop > 0:
            self.buffer = self.buffer[drop:]
            self.offset += drop
