"""Per-stream audio bookkeeping on the CPU: chunk geometry and the raw-sample window of each chunk.

Frame j of the model's centered STFT covers preemphasized samples [j*hop - n_fft/2, j*hop + n_fft/2) (zeros before
the start), so the samples of frames [j0, j0+n) are one contiguous window of (n-1)*hop + n_fft samples. A window is
handed out raw with one extra leading sample, so preemphasis (y[i] = x[i] - k*x[i-1], x = 0 before the start) runs
batched on the GPU inside `LogMel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chunking:
    first_mel: int  # mel frames in the first chunk
    later_mel: int  # mel frames in every later chunk
    hop: int
    n_fft: int

    def end(self, c: int) -> int:
        """Mel frames needed to complete chunks 0..c."""
        return self.first_mel + c * self.later_mel

    def chunks_covering(self, samples: int) -> int:
        """Chunks needed to decode every frame whose center is one of the first `samples` samples."""
        frames = -(-samples // self.hop)
        if frames <= 0:
            return 0
        return 1 if frames <= self.first_mel else 1 + -(-(frames - self.first_mel) // self.later_mel)

    def frames_available(self, samples: int) -> int:
        """Frames computable from the first `samples` samples (frame j needs samples < j*hop + n_fft/2)."""
        return max(0, (samples - self.n_fft // 2) // self.hop + 1)

    def samples_for_chunks(self, n: int) -> int:
        """Samples needed so that chunks 0..n-1 are complete."""
        return (self.end(n - 1) - 1) * self.hop + self.n_fft // 2 + 1


class StreamAudio:
    """Raw samples of one stream in a growable buffer, and the frame cursor."""

    def __init__(self, chunking: Chunking):
        self.ch = chunking
        self.buf = np.zeros(16000, np.float32)  # samples from global index `offset`
        self.n = 0  # valid samples in buf
        self.offset = 0
        self.total = 0  # samples received
        self.frames_done = 0

    def append(self, x: np.ndarray) -> None:
        end = self.n + x.size
        if end > self.buf.size:
            buf = np.empty(max(end, 2 * self.buf.size), np.float32)
            buf[: self.n] = self.buf[: self.n]
            self.buf = buf
        self.buf[self.n : end] = x
        self.n, self.total = end, self.total + x.size

    def window(self, n_frames: int, out: np.ndarray) -> None:
        """Write the next n_frames frames' raw samples, plus the one before, into `out`
        ((n-1)*hop + n_fft + 1 samples) and advance the cursor. All frames must be available."""
        assert self.ch.frames_available(self.total) - self.frames_done >= n_frames
        start = self.frames_done * self.ch.hop - self.ch.n_fft // 2 - 1 - self.offset
        end = start + out.size
        if start < 0:  # before the first sample (only while offset == 0)
            out[:-start] = 0
            out[-start:] = self.buf[:end]
        else:
            out[:] = self.buf[start:end]
        self.frames_done += n_frames
        drop = self.frames_done * self.ch.hop - self.ch.n_fft // 2 - 1 - self.offset  # before the next window
        if drop > self.buf.size // 2:
            self.buf[: self.n - drop] = self.buf[drop : self.n]
            self.n, self.offset = self.n - drop, self.offset + drop
