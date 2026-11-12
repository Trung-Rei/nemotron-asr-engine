"""Real-time streaming workload: long sessions with utterances separated by pauses, filled with public audio.

A session is one continuous 16 kHz stream that stays open for its whole length, the way a live client (a
dictation app, a conversational agent, a meeting transcriber) streams to ASR. Utterances and pauses are drawn
from log-normal distributions (`TimingModel`), and each utterance is filled with VietSuperSpeech audio of about
the same length. Pauses carry low-level noise, so the stream never goes silent. `AudioPool(telephone=True)`
passes the audio through an 8 kHz band first, to mimic phone audio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

SR = 16000
LANG_TAG = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Z]{2})?>")


@dataclass
class Turn:
    start: float  # seconds from session start
    end: float
    reference: str | None  # None when the turn is a cut-down utterance


@dataclass
class Session:
    sid: int
    arrival: float  # seconds from run start
    audio: np.ndarray  # float32, 16 kHz
    turns: list[Turn] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.audio) / SR


class AudioPool:
    """Utterances (audio + reference) to fill the utterance slots of a session, indexed by duration."""

    def __init__(self, rows: list[dict], telephone: bool = False):
        import soundfile as sf
        from scipy.signal import resample_poly

        self.audio, self.text, durs = [], [], []
        for r in rows:
            a, sr = sf.read(r["local_audio"], dtype="float32", always_2d=True)
            a = a.mean(axis=1)
            assert sr == SR, f"{r['local_audio']}: {sr} Hz"
            if telephone:  # 8 kHz band
                a = resample_poly(resample_poly(a, 1, 2), 2, 1).astype(np.float32)
            self.audio.append(a)
            self.text.append(str(r["text"]))
            durs.append(len(a) / SR)
        self.dur = np.asarray(durs)

    def pick(self, rng: np.random.Generator, target: float) -> tuple[np.ndarray, str | None]:
        """Audio of about `target` seconds and its reference text.

        VietSuperSpeech segments are 10-15 s while most utterance slots are 1-5 s, so a shorter slot is the
        head of a longer utterance (faded out, reference None: no word timings to cut the text). Longer slots
        join whole utterances and keep their references.
        """
        if target < self.dur.min():
            i = int(rng.integers(len(self.audio)))
            n = max(int(target * SR), int(0.3 * SR))
            a = self.audio[i][:n].copy()
            fade = min(int(0.02 * SR), n // 2)
            a[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            return a, None
        parts, texts, total = [], [], 0.0
        while True:
            want = target - total
            near = np.argsort(np.abs(self.dur - want))[:8]
            i = int(rng.choice(near))
            parts.append(self.audio[i])
            texts.append(self.text[i])
            total += self.dur[i]
            if total >= 0.85 * target or len(parts) >= 6:
                break
            parts.append(np.zeros(int(0.3 * SR), np.float32))
            total += 0.3
        return np.concatenate(parts), " ".join(texts)


class TimingModel:
    """Synthetic utterance timing: log-normal utterance lengths and pauses.

    Defaults: utterance p50 2.4 s / p90 12.3 s, pause p50 6.5 s / p90 25 s, so speech fills about 30% of a
    session. Short utterances with long pauses are the hard case for a streaming engine: most streams are idle
    at any moment, yet every one of them still needs a chunk decoded every 320 ms and a fast final at each
    utterance end.
    """

    Z90 = 1.2816  # standard normal 90th percentile

    def __init__(self, utt_p50: float = 2.4, utt_p90: float = 12.3, pause_p50: float = 6.5, pause_p90: float = 25.0,
                 utt_range: tuple[float, float] = (0.3, 30.0), pause_min: float = 0.3):
        self.utt = (np.log(utt_p50), np.log(utt_p90 / utt_p50) / self.Z90)
        self.pause = (np.log(pause_p50), np.log(pause_p90 / pause_p50) / self.Z90)
        self.utt_range, self.pause_min = utt_range, pause_min

    def window(self, rng: np.random.Generator, length: float) -> list[tuple[float, float]]:
        """Utterance (start, end) times that start within a `length`-second session.

        The session starts at a random point of the speech/pause cycle, as if it had been running already.
        """
        t = -float(rng.uniform(0, 60.0))
        out = []
        while True:
            dur = float(np.clip(rng.lognormal(*self.utt), *self.utt_range))
            if 0.0 <= t < length:
                out.append((t, t + dur))
            t += dur + max(self.pause_min, float(rng.lognormal(*self.pause)))
            if t >= length:
                return out


def make_session(sid: int, arrival: float, rng: np.random.Generator, timing: TimingModel, pool: AudioPool,
                 window: float, noise_dbfs: float = -60.0) -> Session:
    turns_t = timing.window(rng, window)
    placed, shift = [], 0.0
    for s, e in turns_t:
        audio, ref = pool.pick(rng, e - s)
        start = max(s + shift, placed[-1][1] + 0.3 if placed else 0.2)
        shift = start - s
        placed.append((start, start + len(audio) / SR, audio, ref))
    length = max(window, (placed[-1][1] + 1.0) if placed else 0.0)
    n = int(length * SR)
    out = (rng.standard_normal(n) * 10 ** (noise_dbfs / 20)).astype(np.float32)
    turns = []
    for start, end, audio, ref in placed:
        i = int(start * SR)
        out[i : i + len(audio)] += audio[: n - i]
        turns.append(Turn(start, end, ref))
    return Session(sid, arrival, out, turns)


# --------------------------------------------------------------------------- arrival processes
def open_arrivals(rng, concurrency: float, session_s: float, duration: float) -> list[float]:
    """Poisson arrivals at rate concurrency / session_s (Little's law), for `duration` seconds."""
    rate = concurrency / session_s
    t, out = 0.0, []
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= duration:
            return out
        out.append(t)


def burst_arrivals(rng, base_concurrency: float, session_s: float, duration: float,
                   burst_at: float, burst_sessions: int, burst_spread: float) -> list[float]:
    """Steady Poisson background plus `burst_sessions` extra sessions arriving uniformly within `burst_spread` s."""
    base = open_arrivals(rng, base_concurrency, session_s, duration)
    burst = list(burst_at + rng.uniform(0, burst_spread, burst_sessions))
    return sorted(base + burst)


def strip_tags(text: str) -> str:
    return LANG_TAG.sub("", text).strip()
