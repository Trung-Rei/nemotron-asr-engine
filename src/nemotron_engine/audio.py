"""Incoming audio for the streaming engine: decode PCM16 / G.711 mu-law / float32 and bring it to 16 kHz.

Everything keeps per-stream state, so the output does not depend on how the caller cuts the packets.
"""

from __future__ import annotations

import numpy as np

MODEL_SR = 16000


def _mulaw_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign, exponent, mantissa = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    mag = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return (np.where(sign != 0, -mag, mag) / 32768.0).astype(np.float32)


MULAW_TABLE = _mulaw_table()  # G.711 mu-law byte -> float in [-1, 1)


def mulaw_decode(data: bytes) -> np.ndarray:
    return MULAW_TABLE[np.frombuffer(data, dtype=np.uint8)]


def pcm16_decode(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def lowpass_2x(taps: int = 64, cutoff_hz: float = 3800.0, out_sr: int = MODEL_SR, beta: float = 8.0) -> np.ndarray:
    """Windowed-sinc interpolation filter for 2x upsampling (gain 2 to make up for the inserted zeros)."""
    n = np.arange(taps) - (taps - 1) / 2
    fc = cutoff_hz / out_sr
    h = 2 * fc * np.sinc(2 * fc * n) * np.kaiser(taps, beta)
    return (2 * h / h.sum()).astype(np.float64)


class Upsampler2x:
    """Streaming polyphase 2x upsampler. Output y[2m+p] = sum_k h[2k+p] x[m-k]; delay (taps-1)/2 output samples."""

    def __init__(self, h: np.ndarray | None = None):
        h = lowpass_2x() if h is None else h
        self.phases = [h[0::2], h[1::2]]
        self.hist = np.zeros(len(self.phases[0]) - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, np.float32)
        xx = np.concatenate([self.hist, x.astype(np.float64)])
        n_hist = len(self.hist)
        y = np.empty(2 * x.size)
        for p, hp in enumerate(self.phases):
            y[p::2] = np.convolve(xx, hp)[n_hist : n_hist + x.size]
        self.hist = xx[-n_hist:]
        return y.astype(np.float32)


class AudioDecoder:
    """bytes / ndarray packets of one stream -> float32 samples at 16 kHz."""

    ENCODINGS = ("pcm16", "mulaw", "f32")

    def __init__(self, sample_rate: int = MODEL_SR, encoding: str = "pcm16"):
        if encoding not in self.ENCODINGS:
            raise ValueError(f"encoding must be one of {self.ENCODINGS}, got {encoding!r}")
        if sample_rate not in (MODEL_SR, MODEL_SR // 2):
            raise ValueError(f"sample_rate must be 16000 or 8000, got {sample_rate}")
        self.encoding = encoding
        self.width = {"pcm16": 2, "mulaw": 1, "f32": 4}[encoding]
        self.rest = b""  # partial sample left over from the previous packet
        self.up = Upsampler2x() if sample_rate != MODEL_SR else None

    def __call__(self, data) -> np.ndarray:
        if isinstance(data, np.ndarray):
            x = data.astype(np.float32, copy=False).ravel()
            if data.dtype == np.int16:
                x = x / 32768.0
        else:
            data = self.rest + bytes(data)
            cut = len(data) - len(data) % self.width
            data, self.rest = data[:cut], data[cut:]
            if self.encoding == "pcm16":
                x = pcm16_decode(data)
            elif self.encoding == "mulaw":
                x = mulaw_decode(data)
            else:
                x = np.frombuffer(data, dtype="<f4").astype(np.float32)
        return self.up(x) if self.up is not None else x
