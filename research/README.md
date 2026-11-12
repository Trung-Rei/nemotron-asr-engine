# nemotron_lite: the reference port

*[Tiếng Việt](README.vi.md)*

A pure-PyTorch, inference-only port of `nvidia/nemotron-3.5-asr-streaming-0.6b`:

- cache-aware FastConformer + RNN-T, language-prompted;
- offline and streaming;
- loads the checkpoint's `model.safetensors` directly;
- needs `torch`, `safetensors`, `tokenizers`, `numpy` and `soundfile`, and no NeMo.

It is the readable version of the model, and the yardstick the engine is tested against:

| Check | Result |
|---|---|
| Layer by layer vs NeMo | match within 1e-6 (NeMo's fused Triton subsampling off) |
| Offline, 500 utterances | WER equal to NeMo, ~99% identical hypotheses |
| Streaming vs offline | identical |

`nemotron_engine` (in `src/`) is the production path and does not import this package.

```python
import sys; sys.path.insert(0, "research")
from nemotron_lite import NemotronLite

model = NemotronLite.from_pretrained(device="cuda")
print(model.transcribe(["a.wav"], language="en-US", lookahead=13))

sess = model.stream(language="en-US", lookahead=3)   # lookahead 0/3/6/13 -> 80/320/560/1120 ms chunks
for piece in mic_chunks():                            # 16 kHz float32 pieces of any size
    partial = sess.accept_audio(piece)
final = sess.finish()
```

| File | Contents |
|---|---|
| `features.py` | log-mel front-end, and an incremental version for streaming that gives the same frames |
| `encoder.py` | causal subsampling, chunked-limited rel-pos attention, `stream_step` caches |
| `rnnt.py` | prediction network, joint, batched greedy decoding |
| `batch_stream.py` | `BatchedStreamEngine`: synchronous multi-stream batching with CUDA graphs; the engine's ancestor |

Parity runs against NeMo: `bash bench/parity/run_parity.sh` (see `bench/README.md`). `requirements.txt` here installs everything the benchmarks need, NeMo included.
