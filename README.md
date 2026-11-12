# nemotron-engine

*[Tiếng Việt](README.vi.md)*

**Serve hundreds of live speech-to-text streams from one GPU.**

nemotron-engine is a real-time, multi-stream streaming ASR engine for NVIDIA's [Nemotron ASR streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) model (FastConformer + RNN-T, ~100 languages). It is written in plain PyTorch, with no NeMo at run time. You can:

- embed it in your asyncio app;
- run it as a WebSocket server;
- try it from the command line or in the browser.

On one A100 40 GB, the engine serves **384 concurrent real-time streams** with the final transcript **96 ms** (p95) after each utterance ends. NeMo's own cache-aware streaming, batched by the same scheduler on the same GPU, holds 192.

- **Same transcripts.** Tokens are bit-identical to a reference port that matches NeMo layer by layer (strict fp32). WER on 500 utterances is equal to NeMo's.
- **Many streams per GPU.** Chunks from all live streams are batched into one GPU step. The whole step (log-mel → encoder → greedy RNN-T) is a single CUDA graph replay.
- **Fast finals.** When your VAD says an utterance ended, the final transcript arrives after one GPU step instead of up to 320 ms of extra audio.
- **Easy to use.** An `async for event in stream` API, a WebSocket server with a demo page, PCM16 / μ-law / float32 input at 8 or 16 kHz, and admission control when the GPU is saturated.

> Unofficial project. Not affiliated with or endorsed by NVIDIA. The model weights are downloaded from Hugging Face and governed by NVIDIA's [OpenMDW-1.1](https://openmdw.ai/license/1-1/) license.

## Quickstart

You need an NVIDIA GPU with CUDA, Python ≥ 3.10 and PyTorch ≥ 2.4.

```bash
pip install "nemotron-engine[server] @ git+https://github.com/Trung-Rei/nemotron-engine"

# transcribe files (each file is its own concurrent stream); the model downloads on first use (~2.5 GB)
nemotron-engine transcribe speech.wav --language en-US

# WebSocket server + a microphone demo page at http://localhost:8090
nemotron-engine serve --port 8090 --max-streams 256
```

The demo page needs microphone access, which browsers allow only on `localhost` or HTTPS. For a remote GPU box, open a tunnel first: `ssh -L 8090:localhost:8090 gpu-box`.

### In Python

```python
import asyncio
from nemotron_engine import AsyncNemotronEngine, EngineConfig

async def main():
    engine = await AsyncNemotronEngine.create(EngineConfig(max_streams=256, language="en-US"))
    stream = engine.open_stream(sample_rate=16000, encoding="pcm16")   # one per live audio source

    async def read():
        async for ev in stream:                 # TranscriptEvent
            if ev.kind == "partial":
                print("…", ev.text)             # full text of the current utterance so far
            else:
                print("✔", ev.text)             # "final": the utterance is done

    reader = asyncio.create_task(read())
    for packet in microphone_packets():         # bytes of any size, as they arrive
        await stream.push(packet)
        if vad_says_utterance_ended():
            stream.end_turn()                   # -> one "final" event for this utterance
    await stream.close()                        # finishes the open utterance, frees the slot
    await reader
    await engine.aclose()

asyncio.run(main())
```

Runnable versions are in [`examples/`](examples):

| Example | What it shows |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | one file, real-time packets, live partials |
| [`many_streams.py`](examples/many_streams.py) | hundreds of concurrent streams in one process |
| [`ws_client.py`](examples/ws_client.py) | WebSocket client with client-side latency |

## Benchmarks

All rows: one NVIDIA A100 40 GB, 320 ms chunks, 20 ms packets in real time, VietSuperSpeech audio. SLO: finalize p95 ≤ 300 ms, chunk lag p95 ≤ 320 ms, no refused stream.

| | Concurrent streams within SLO | Finalize p50 / p95 at that load | WER |
|---|---|---|---|
| NeMo cache-aware streaming (batched, synchronous scheduler) | 192 | 95 / 154 ms | equal |
| **nemotron-engine, asyncio API** | **384** | **51 / 96 ms** | equal |
| nemotron-engine, WebSocket server | 256 | 26 / 65 ms | equal |

- **Accuracy.** The reference port matches NeMo: WER 23.91% vs 23.91% at 1120 ms chunks and 25.19% vs 25.18% at 320 ms, on 500 utterances. The engine's tokens are identical to the port's (64/64 streams, 0 of 9,272 tokens differ).
- **Scope.** The engine is built for live streams. For offline batch transcription of files, it is not faster than NeMo's offline path: the reference port measured within a few percent of NeMo's RTFx on an A100.

Methodology, all numbers and how to reproduce them: [docs/benchmarks.md](docs/benchmarks.md).

## API

### Input

`engine.open_stream(language=None, sample_rate=16000, encoding="pcm16")`

| `encoding` | `push()` payload | `sample_rate` |
|---|---|---|
| `pcm16` | bytes, 16-bit little-endian | 16000 or 8000 |
| `mulaw` | bytes, G.711 μ-law | 8000 (or 16000) |
| `f32` | bytes, float32 | 16000 or 8000 |

- `push()` also takes a numpy array (float32 or int16).
- Packets may have any size. The output does not depend on how the audio is cut.
- 8 kHz audio is upsampled to 16 kHz by a streaming 64-tap polyphase filter (≈2 ms delay).
- `language` takes a locale such as `en-US`, `vi-VN`, `de-DE` or `ja-JP`, or `auto`. Each stream can use a different language.

### Events

`TranscriptEvent` fields: `kind` (`"partial"` / `"final"`), `turn` (utterance index), `text` (the utterance so far), `delta` (new text), `tokens`, `audio_ms`, and the timestamps `t_ready` / `t_emit` (`time.perf_counter`).

- A **partial** comes with every 320 ms chunk that adds text. RNN-T only appends, so a partial is never revised.
- A **final** comes exactly once per `end_turn()`.

### Ending an utterance

| Call | Behavior | Trade-off |
|---|---|---|
| `end_turn(pad=True)` (default) | Pads the current chunk with silence and decodes it at once. The final follows one GPU step. | Up to 320 ms of silence is inserted into the stream's timeline. A token the model emits late, usually punctuation, may start the next utterance. |
| `end_turn(pad=False)` | Waits until real audio completes the chunk. | Adds 0–320 ms of latency. |

The engine does not detect utterance ends itself: use your own VAD or endpointer. The server's demo page has a simple energy-based one.

### Errors and admission control

| Exception | When |
|---|---|
| `EngineFull` | `open_stream` with all `max_streams` slots in use. |
| `EngineOverloaded` (subclass of `EngineFull`) | The oldest chunk waiting for the GPU is older than `max_lag_ms` (default 200 ms). A saturated engine refuses new streams instead of slowing every stream down. |
| `EngineClosed` | Using a closed stream or engine. |

### Configuration (`EngineConfig`)

| Field | Default | Meaning |
|---|---|---|
| `max_streams` | 512 | Concurrent stream slots. Their caches are allocated up front (see sizing below). |
| `max_batch` | 64 | Streams per GPU step. |
| `language` | `"auto"` | Default language prompt. |
| `lookahead` | 3 | Chunk = (lookahead + 1) × 80 ms. The model supports 0, 3, 6, 13. |
| `max_lag_ms` | 200 | Admission threshold; `None` disables it. |
| `matmul_precision` | `"high"` | TF32 matmuls; `"highest"` for strict fp32. |
| `cuda_graphs` | `True` | `False` runs steps eagerly: slower, for debugging. |
| `model_dir` | `None` | A local folder with `config.json`, `processor_config.json`, `tokenizer.json` and `model.safetensors`, instead of downloading. |

### Sizing

GPU memory is reserved at start-up and stays flat. Most of it is the per-slot cache (24 layers × 56 frames of K/V). The 704 row was measured on the A100; the other rows are estimated from it.

| `max_streams` | PyTorch allocated | Total GPU memory (≈ +1.3 GB CUDA context and allocator) |
|---|---|---|
| 128 | 4.9 GB | 6.2 GB |
| 256 | 6.4 GB | 7.7 GB |
| 512 | 9.4 GB | 10.7 GB |
| 704 | 11.66 GB (measured) | 12.99 GB (measured) |
| 1024 | 15.5 GB | 16.8 GB |

Rule of thumb from the A100 run: **about 3.4 GB + 11.8 MB per slot**. Set `max_streams` from your GPU memory. Set it above the load the GPU can serve within latency targets, and let `max_lag_ms` refuse streams when the GPU is saturated.

Throughput is bounded by GPU compute, not memory: see the benchmark table for how many active streams one GPU sustains.

## WebSocket server

```bash
nemotron-engine serve --host 0.0.0.0 --port 8090 --max-streams 256 [--language en-US]
# equivalent: NEMOTRON_MAX_STREAMS=256 uvicorn nemotron_engine.server:app --port 8090   (one worker per GPU)
```

**`WS /v1/stream?language=en-US&sample_rate=16000&encoding=pcm16`**

| Direction | Message |
|---|---|
| client → server | Binary frames: audio in the stream's encoding, any size. |
| client → server | Text `{"type": "end_turn", "pad": true}`: end the utterance. |
| client → server | Text `{"type": "close"}`: finish the open utterance and close. |
| server → client | `{"type": "partial" \| "final", "turn", "text", "delta", "audio_ms", "latency_ms"}` |
| server → client | `{"type": "closed"}` after the last event. |

Close codes: `1013` engine full or overloaded (retry later), `1003` bad parameters or command, `1011` engine error.

- **`GET /`** serves the microphone demo page.
- **`GET /health`** returns engine stats (active streams, free slots, chunk lag, step time) and memory.

Run one server process per GPU, for example with `CUDA_VISIBLE_DEVICES`. Put a load balancer in front if you have several.

## How it works

Read [docs/architecture.md](docs/architecture.md) for the design, and [docs/optimization-journey.md](docs/optimization-journey.md) for how it got there: what was profiled, what changed, and what did not work. In short:

1. One worker thread owns the GPU. Your event loop only decodes packets and appends them to an inbox.
2. Every stream owns a slot in pre-allocated cache tensors. A batch is gather → 24 layers → scatter over slot indices.
3. The whole step, from raw samples to RNN-T labels, is one CUDA graph per batch bucket. The host does about 6 PyTorch calls per step, which also keeps the Python GIL out of the GPU's way.
4. Greedy RNN-T runs inside the graph with a frame pointer per stream: branch-free and bit-exact.
5. The next batch is picked the moment the GPU finishes. A chunk waits for at most the step in flight.

## Repository layout

| Path | Contents |
|---|---|
| [`src/nemotron_engine/`](src/nemotron_engine) | The engine package: what `pip install` gives you. |
| [`examples/`](examples) | Small runnable programs. |
| [`research/nemotron_lite/`](research/nemotron_lite) | Reference port: offline + streaming, simple code, checked against NeMo. Used for parity tests and experiments. |
| [`bench/`](bench) | Accuracy parity vs NeMo, offline RTFx, streaming load tests (in-process and WebSocket), profilers. |
| [`tests/`](tests) | CPU unit tests (`pytest`) and the GPU parity/stress test (`python tests/test_engine.py`). |
| [`docs/`](docs) | Architecture, benchmarks, optimization journey (English and Vietnamese). |

## Reproduce the benchmarks

```bash
pip install -r research/requirements.txt          # engine + benchmark deps + NeMo for the reference runs
STAGES="test parity capacity ws cli" bash bench/run_all.sh
```

To run on a rented GPU, [`vkong.yaml`](vkong.yaml) is a ready recipe: `vkong run --detach`.

## Limitations

- One process uses one GPU. For several GPUs, run several processes.
- Greedy decoding only; no beam search, no word timestamps yet.
- The chunk size is fixed per engine (`lookahead`). 320 ms is the default and the tested configuration.
- Through the WebSocket server, a single Python event loop handles every frame. At very high stream counts with 20 ms packets it becomes the bottleneck before the GPU does. Larger packets (60–100 ms) or several front-end processes help.
- No authentication on the server. Put it behind your own gateway.

## Acknowledgements and license

- The model is by NVIDIA, released under [OpenMDW-1.1](https://openmdw.ai/license/1-1/). The implementation follows [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/Speech) (Apache-2.0).
- The benchmarks use the public [VietSuperSpeech](https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech) dataset.
- The code in this repository is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE).

If you use this in your work, see [CITATION.cff](CITATION.cff).
