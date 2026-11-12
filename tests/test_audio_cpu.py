"""CPU-only checks of the audio front of the engine (no torch, no GPU): pytest tests/test_audio_cpu.py

- G.711 mu-law table and round trip
- the streaming 2x upsampler gives the same output as a one-shot call, however the input is cut
- PCM16 decoding does not depend on packet boundaries (even ones that split a sample)
- the upsampler keeps passband gain
- StreamAudio chunk windows, across buffer growth and compaction, equal slices of the whole signal
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemotron_engine.audio import MULAW_TABLE, AudioDecoder, Upsampler2x  # noqa: E402
from nemotron_engine.stream_audio import Chunking, StreamAudio  # noqa: E402

SR = 16000


def mulaw_encode(x: np.ndarray) -> bytes:
    """Linear float -> G.711 mu-law bytes (reference encoder for the tests)."""
    s = np.clip(np.round(x * 32768), -32635, 32635).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    s = np.abs(s) + 0x84
    exp = np.clip(np.floor(np.log2(s)).astype(np.int32) - 7, 0, 7)
    mant = (s >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


def _noise(n: int, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


def test_mulaw():
    assert MULAW_TABLE[0xFF] == 0.0 and MULAW_TABLE[0x7F] == 0.0
    assert abs(MULAW_TABLE[0x00] * 32768 + 32124) < 1e-3 and abs(MULAW_TABLE[0x80] * 32768 - 32124) < 1e-3
    x = _noise(8000 * 3)
    rt = MULAW_TABLE[np.frombuffer(mulaw_encode(x), np.uint8)]
    assert np.max(np.abs(rt - x)) < 0.01, "mu-law round trip"


def test_upsampler_streaming_equals_one_shot():
    rng = np.random.default_rng(1)
    x = _noise(8000 * 3)
    one = Upsampler2x()(x)
    up, pieces, i = Upsampler2x(), [], 0
    for n in rng.integers(1, 400, 1000):
        pieces.append(up(x[i : i + n]))
        i += n
        if i >= x.size:
            break
    pieces.append(up(x[i:]))
    assert np.allclose(np.concatenate(pieces), one, atol=1e-6)


def test_pcm16_packetization():
    pcm = (_noise(8000 * 3) * 32767 / 0.5).clip(-32768, 32767).astype("<i2").tobytes()
    a = AudioDecoder(8000, "pcm16")(pcm)
    d = AudioDecoder(8000, "pcm16")
    b = np.concatenate([d(pcm[j : j + 333]) for j in range(0, len(pcm), 333)])  # odd cuts split samples
    assert np.allclose(a, b, atol=1e-6)


def test_upsampler_passband_gain():
    tone = np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000).astype(np.float32)
    y = Upsampler2x()(tone)[200:-200]
    assert abs(np.sqrt(np.mean(y**2)) - np.sqrt(0.5)) < 0.02


def test_stream_audio_windows() -> int:
    rng = np.random.default_rng(2)
    ch = Chunking(25, 32, 160, 512)
    sig = rng.standard_normal(SR * 20).astype(np.float32)
    padded = np.concatenate([np.zeros(ch.n_fft // 2 + 1, np.float32), sig])
    sa, i, c = StreamAudio(ch), 0, 0
    for n in rng.integers(1, 3000, 10000):
        sa.append(sig[i : i + n])
        i += n
        while ch.frames_available(sa.total) >= ch.end(c):
            nf = ch.first_mel if c == 0 else ch.later_mel
            out = np.empty((nf - 1) * ch.hop + ch.n_fft + 1, np.float32)
            j0 = sa.frames_done * ch.hop
            sa.window(nf, out)
            assert np.array_equal(out, padded[j0 : j0 + out.size]), f"window of chunk {c}"
            c += 1
        if i >= sig.size:
            break
    assert c > 50


def run_all() -> None:
    """For tests/test_engine.py, which runs outside pytest."""
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("audio: OK", flush=True)


if __name__ == "__main__":
    run_all()
