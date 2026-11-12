# Architecture

*[Tiếng Việt](architecture.vi.md)*

nemotron-engine runs [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for many live audio streams at once on one GPU. This page explains how one process turns hundreds of independent streams into a few large GPU steps, and why each design choice is there.

## The model in one paragraph

The model is a cache-aware FastConformer encoder with an RNN-T decoder (0.6B parameters):

- Audio becomes a 128-bin log-mel spectrogram (10 ms hop).
- A ×8 convolutional subsampler turns 8 mel frames into one encoder frame (80 ms).
- 24 Conformer layers process the frames. Each layer is FFN, relative-position self-attention, convolution, FFN.
- Attention is chunked and limited: a frame sees 56 frames of left context and the frames of its own chunk.
- A language prompt (a one-hot over ~100 locales, or `auto`) is mixed into the encoder output.
- An RNN-T prediction network and joint emit word pieces.

Because the attention and the convolutions only look left (plus inside the chunk), the encoder can run chunk by chunk with a cache:

- the last 56 frames of K/V per layer;
- the last 8 frames of conv state per layer;
- the subsampler's tail.

A chunk of 320 ms (4 encoder frames, `lookahead=3`) is the default. The model also supports 80, 560 and 1120 ms.

## Where the time goes in a naive streaming loop

Run the model one chunk at a time, or even batched with eager PyTorch, and the GPU is mostly idle:

| Problem | Why it hurts |
|---|---|
| ~1,500 kernel launches per step | 24 layers × (LayerNorms, GEMMs, attention pieces, conv, GLU, ...) issued from Python. At small chunk sizes the GPU finishes each kernel before Python issues the next: the step is bound by the host (CPU), not the GPU. |
| 1×1 convolutions through cuDNN | The pointwise convs of the conv module run ~10× slower than the same math as a GEMM. |
| Greedy RNN-T decoding on the host | One `.item()` / sync per emitted token: dozens of GPU↔CPU round trips per step. |
| One process, many streams, one GIL | Every PyTorch op releases and reacquires the GIL. If the thread that drives the GPU must win the GIL back from an event loop serving hundreds of sockets for every op, the GPU waits for Python. |

The engine removes each of these.

## Design

```
 asyncio event loop (your app / the server)            worker thread "nemotron-gpu" (owns the GPU)
 ─────────────────────────────────────────             ───────────────────────────────────────────────
 open_stream / push / end_turn / close                  1. while the GPU runs step N: poll its CUDA event,
   decode PCM16 / mu-law / f32, 8→16 kHz  ──inbox──▶       drain the inbox in between (append samples,
                                           (one deque)     mark complete chunks, utterance boundaries)
 detokenize, compute delta   ◀── call_soon_threadsafe ── 2. GPU done: collect step N, post partials / finals
 asyncio.Queue → `async for ev in stream`                3. drain the rest, pick batch N+1 (oldest chunk first,
                                                            same phase, ≤ max_batch), copy sample windows
                                                            into one pinned staging buffer
                                                         4. submit: H2D copies + ONE CUDA graph replay; no wait
```

### 1. One CUDA graph per step

Each batch-size bucket has a CUDA graph that covers the whole step:

1. preemphasis and log-mel (`torch.stft` without centering: the stream buffer supplies the context);
2. the ×8 subsampler, with its cache gathered from and scattered back to the slot pool;
3. gathering K/V, conv state, cache lengths and the language one-hot of the batch's slots;
4. the 24 Conformer layers;
5. the prompt projector;
6. scattering the caches back to the pool;
7. `greedy_steps` (12) steps of greedy RNN-T decoding;
8. copying the labels to pinned host memory and recording a CUDA event.

The host only copies the staging buffer to the GPU, replays one graph and later reads the labels: about half a dozen PyTorch calls per step instead of ~1,500. That removes the launch bound, and with it most GIL handoffs, which is what made the step's cost depend on how busy the event loop was.

There are separate graphs for the first chunk of a stream (25 mel frames, no cache yet; buckets 1, 2, 4, …, 64) and for later chunks (32 mel frames; buckets 1, 2, 4, 8, 16, 24, …, 64), plus one continuation graph (see below): 19 graphs at `max_batch=64`. They share one memory pool and are captured largest first, so smaller graphs reuse the larger graphs' blocks. A batch is padded up to its bucket. Padding rows point to a scratch slot, so their cache writes never touch a real stream.

### 2. Slot-major cache pool

Every stream owns a slot for its whole life. The pool holds, per slot:

- K/V for 24 layers × 56 frames;
- conv state;
- the subsampler cache;
- the prediction network's LSTM state and output;
- the language prompt one-hot.

The pool is allocated once at start-up (`max_streams` slots). Memory stays flat no matter how streams come and go: there is no allocation on the hot path. Gathering a batch's slots and scattering them back is an indexed copy inside the graph.

### 3. Attention as one SDPA call

Relative-position attention needs `q·k` plus `q·p` with a shifted position term. The engine:

- computes Q, K, V with one GEMM;
- precomputes the shifted position keys once per chunk shape;
- adds them as an attention bias to a single `scaled_dot_product_attention` call (memory-efficient kernel) over cache + chunk keys;
- rolls the K/V cache inside the scatter, with no extra copy.

The pointwise convolutions run as `F.linear`, not cuDNN.

### 4. Greedy RNN-T inside the graph, with a frame pointer per stream

Greedy RNN-T emits zero or more tokens per encoder frame. The number varies per stream, and that variation is what usually forces a host-side loop. The engine instead gives each stream its own frame pointer. At every decoding step, each stream either:

- emits one token and stays on its frame, or
- moves to its next frame (on blank, or after 10 symbols on one frame).

A stream needs T + (its token count) steps. For each stream, the sequence of joint / prediction-network evaluations is exactly that of the frame-by-frame loop, so **tokens are bit-identical** to the reference implementation.

Twelve such steps run inside the step graph. When a stream still has frames left, `collect` replays a small "continuation" graph of 4 more steps. This is rare at a 320 ms chunk.

An alternative was rejected: a fixed number of symbols per frame inside the graph is simpler, but real speech regularly emits ≥3 tokens on one frame, so nearly every batch overflowed.

### 5. A worker that picks the batch when the GPU frees up

The worker thread loop is:

1. **While the GPU runs a step,** poll its CUDA event and drain the inbox in between. Draining means appending samples to per-stream buffers, marking chunks complete, and recording utterance boundaries.
2. **When the GPU finishes,** collect the labels and post all results of the round to the event loop in one `call_soon_threadsafe`.
3. **Pick the next batch** among streams whose next chunk is complete. The oldest-waiting chunk goes first, and a batch holds only first chunks or only later chunks, up to `max_batch`.
4. **Copy each stream's sample window** straight into the pinned staging buffer, and submit.

Picking at the moment the GPU frees up, instead of preparing the next batch earlier, means a chunk waits for at most the one step in flight. Everything runs on one CUDA stream, so a stream that appears in two consecutive batches still sees its cache updates in order.

### 6. The event loop does as little as possible

- `push` decodes the packet (mu-law / PCM16 / float32, streaming 8→16 kHz polyphase upsampler) and appends it to the shared inbox. No per-stream lock and no scan over all streams.
- `wake.set()` is called only when the worker is asleep.
- Detokenization runs on the event loop, off the GPU thread.
- `open_stream` takes a slot from a free list under a lock. It raises `EngineFull` at once when none is left, and `EngineOverloaded` when the oldest chunk waiting for the GPU is older than `max_lag_ms` (default 200 ms). A saturated engine refuses new streams instead of letting every stream's latency grow.

### 7. Utterance ends without waiting for audio

`end_turn(pad=True)` pads the stream with zeros to the end of the current chunk, so the final comes right after the next GPU step instead of waiting for up to 320 ms of real audio. The stream keeps its caches; the next utterance continues after the padding.

The known trade-off: a token the model emits late, after the boundary (most often punctuation), lands in the next utterance.

## Files

| File | What it does |
|---|---|
| [`engine.py`](../src/nemotron_engine/engine.py) | asyncio API (`AsyncNemotronEngine`, `AsrStream`, `TranscriptEvent`), the worker loop, slots, utterance boundaries |
| [`batch.py`](../src/nemotron_engine/batch.py) | `BatchStepper`: slot pool, graph capture, frame-pointer greedy, `submit` / `ready` / `collect` |
| [`model.py`](../src/nemotron_engine/model.py) | the model's streaming modules, log-mel with preemphasis, weight and tokenizer loading |
| [`stream_audio.py`](../src/nemotron_engine/stream_audio.py) | per-stream raw sample buffer, chunk geometry, sample windows |
| [`audio.py`](../src/nemotron_engine/audio.py) | PCM16 / mu-law / f32 decoding, streaming 8→16 kHz upsampler |
| [`server.py`](../src/nemotron_engine/server.py) | FastAPI WebSocket server and demo page |
| [`cli.py`](../src/nemotron_engine/cli.py) | `nemotron-engine transcribe / serve / bench` |

## Where one step spends its time

Measured on the A100 with [`bench/profiling/graph_breakdown.py`](../bench/profiling/graph_breakdown.py). Each stage is captured as its own CUDA graph on static inputs for a batch of later chunks, then replayed and timed with CUDA events, next to a replay of the real whole-step graph.

| Stage, batch 64 | ms | Share of the stage sum |
|---|---|---|
| FFN 1 + FFN 2 (24 layers) | 5.55 | 27% |
| Gather + scatter of the slot caches | 4.49 | 22% |
| Conv module (24 layers) | 2.72 | 13% |
| Attention: SDPA (24 layers) | 2.37 | 11% |
| Greedy RNN-T, 12 steps | 2.36 | 11% |
| Attention: QKV + cache write, rel-pos bias, output projection | 1.99 | 10% |
| Subsampling + its cache | 1.09 | 5% |
| Log-mel, final LayerNorms, prompt projector | 0.30 | 1% |
| **Sum of stages** | **20.88** | |
| **Whole-step graph** | **27.67** | |
| Same step, eager PyTorch | 41.24 | |

| Batch | Whole-step graph | Eager | One continuation graph (4 greedy steps) |
|---|---|---|---|
| 8 | 9.35 ms | 41.46 ms | 0.77 ms |
| 32 | 22.92 ms | 42.21 ms | 0.68 ms |
| 64 | 27.67 ms | 41.24 ms | 0.77 ms |

What this shows:

- **Eager PyTorch costs ~41 ms at any batch size.** That is the launch bound: the GPU waits for Python. The graph makes small batches cheap (9 ms at batch 8) and large batches 1.5× faster.
- **The whole-step graph takes ~7 ms more than the sum of its stages at batch 64.** The stages share memory traffic, and timed alone they run with warm caches. The sum is a guide to where time goes, not an exact split.
- **Gathering and scattering the caches is about a fifth of the step.** Attention that reads K/V in place from the slot pool would remove most of it. It is the clearest next optimization.
- **Greedy decoding costs about the same at batch 8 and 64 (~2.3 ms).** Its 12 sequential steps are latency-bound, not compute-bound.

Without CUDA graphs (`EngineConfig(cuda_graphs=False)`), the same load test fails already at 256 streams on this A100: finalize p95 973 ms, batch pinned at 63.6, one CPU core saturated. With graphs, 256 streams run at p95 30 ms.

## Correctness

The engine is checked against [`research/nemotron_lite`](../research/nemotron_lite), a straightforward PyTorch port that is itself checked layer by layer against NeMo:

- every layer matches NeMo within 1e-6 relative error with NeMo's fused Triton subsampling turned off;
- WER on 500 utterances matches NeMo.

[`tests/test_engine.py`](../tests/test_engine.py) runs 64 concurrent streams through the engine in strict fp32 and requires **identical tokens** to the reference. The same holds:

- with 100 ms packets instead of 20 ms;
- with `greedy_steps=1`, where every chunk goes through continuation graphs.

See [benchmarks.md](benchmarks.md) for the numbers.
