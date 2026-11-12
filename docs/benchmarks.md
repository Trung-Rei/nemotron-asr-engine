# Benchmarks

*[Tiếng Việt](benchmarks.vi.md)*

All numbers on this page come from one run of [`bench/run_all.sh`](../bench/run_all.sh) on a single **NVIDIA A100 40 GB** (rented, not shared with other jobs): A100-SXM4-40GB, driver 610.57, PyTorch 2.7.1 + CUDA 12.8, fp32 weights with TF32 matmuls unless noted, 12 host CPU cores. Run date: 2026-11-12. Raw summaries: [`docs/results/a100-2026-11-12/`](results/a100-2026-11-12).

## What is measured, and why

Offline throughput (RTFx: seconds of audio per second of compute) is the usual ASR benchmark. It does not answer the question a live service has:

> How many concurrent real-time streams can one GPU serve while every utterance's final transcript still arrives quickly after the speaker stops?

In a live service:

- audio arrives in small packets on the wall clock, whether or not the server keeps up;
- every stream stays open for minutes, including pauses;
- what users feel is the delay between the end of an utterance and its final text.

### Workload

[`bench/streaming/workload.py`](../bench/streaming/workload.py):

- **Sessions** arrive as a Poisson process. Each one is a 40 s continuous 16 kHz stream.
- **Utterances and pauses** are drawn from log-normal distributions: utterance p50 2.4 s / p90 12.3 s, pause p50 6.5 s / p90 25 s. About 30% of the time is speech.
- **Utterance audio** is from VietSuperSpeech (public, with references, so WER is computed too). Pauses carry low-level noise.
- **Packets** of 20 ms are pushed on the wall clock. `end_turn()` is called the moment an utterance ends (an ideal VAD).
- **Chunk size** is 320 ms (`lookahead=3`).

### Metrics

| Metric | Definition |
|---|---|
| **Finalize** | From `end_turn()` to the final event at the client. |
| **Chunk lag** | From the moment a chunk's audio is complete to the moment its tokens come out. |
| **SLO** | Finalize p95 ≤ 300 ms, chunk lag p95 ≤ 320 ms, no refused stream. Measured from second 40 to second 90 of each load level. |

### NeMo baseline

NeMo cannot take an asyncio stream API as is. It is driven by [`bench/streaming/loadtest.py`](../bench/streaming/loadtest.py), a synchronous scheduler that:

- precomputes features;
- batches the oldest ready chunks (up to 64);
- calls NeMo's cache-aware streaming step (`conformer_stream_step`).

The scheduler costs nothing on the event loop, so this is a best case for NeMo.

## Results

### Capacity: concurrent real-time streams on one A100

| | Streams within SLO | Finalize p50 / p95 / p99 at that level | Chunk lag p50 / p95 | Throughput at that level |
|---|---|---|---|---|
| NeMo cache-aware streaming, synchronous scheduler | **192** | 95 / 154 / 171 ms ¹ | 97 / 156 ms | 607 chunks/s, batch 45.6 |
| **nemotron-engine, asyncio API** | **384** | **51 / 96 / 120 ms** | 48 / 86 ms | 1,278 chunks/s, batch 45.0 |
| nemotron-engine, WebSocket server (4 client processes) | **256** | 26 / 65 / 206 ms | 19 / 30 ms | |

¹ NeMo's "finalize" is measured server-side: from the moment the chunk holding the utterance's last sample is complete. It does not include the wait for that chunk to fill. Measured end to end, from the end of speech, NeMo's p95 at 192 streams is **421 ms**. The engine's numbers are always end to end: from `end_turn()` to the final event at the client, including the padding of the open chunk. The capacity comparison is therefore conservative toward NeMo.

All levels:

**NeMo** (`bench/streaming/nemo_vs_lite.sh`, backend `nemo`)

| Target streams | Chunks/s | Mean batch | Chunk lag p50/p95 ms | Finalize p50/p95/p99 ms | End-to-end finalize p95 | WER % | SLO |
|---|---|---|---|---|---|---|---|
| 64 | 246 | 16.9 | 77 / 137 | 77 / 207 / 1340 | 491 | 19.3 | ✅ |
| 128 | 418 | 30.0 | 90 / 148 | 87 / 147 / 163 | 421 | 20.7 | ✅ |
| 192 | 607 | 45.6 | 97 / 156 | 95 / 154 / 171 | 421 | 24.6 | ✅ |
| 256 | 853 | 62.5 | 173 / 778 | 176 / 763 / 887 | 970 | 23.9 | ❌ |

**nemotron-engine, asyncio API** (`bench/streaming/engine_loadtest.py`)

| Target streams | Chunks/s | Mean batch | Chunk lag p50/p95 ms | Finalize p50/p95/p99 ms | Process CPU | WER % | SLO |
|---|---|---|---|---|---|---|---|
| 256 | 868 | 11.8 | 18 / 26 | 19 / 30 / 37 | 38% | 25.0 | ✅ |
| 384 | 1,278 | 45.0 | 48 / 86 | 51 / 96 / 120 | 48% | 25.1 | ✅ |
| 448 | 1,501 | 63.7 | 1,457 / 2,723 | 1,426 / 2,732 / 2,772 | 57% | 23.2 | ❌ |
| 512 | 1,781 | 63.9 | 8,106 / 14,165 | 8,083 / 14,126 / 14,693 | 60% | 23.6 | ❌ (13 refused: slots full) |

At 448 streams, the mean batch reaches the 64-stream cap and chunks queue up. The GPU is saturated near 1,500–1,800 chunks/s. One whole-step graph of 64 streams takes 27.7 ms on this A100 in isolation. Under load, small first-chunk batches, continuation graphs and the gaps between steps add to it (see [architecture.md](architecture.md#where-one-step-spends-its-time)).

The edge between 384 and 448 is not sharp. In a second, shorter run (`bench/profiling/profile_engine.py`, 70 s per level), 448 streams passed with finalize p95 41 ms, and 512 streams failed at p95 516 ms. The table above keeps the stricter result.

**Without CUDA graphs** (`--no-cuda-graphs`), 256 streams already fail: finalize p95 973 ms, batch pinned at 63.6, process CPU 102%. Every step is launch-bound at ~41 ms.

**`nemotron-engine bench`**, the CLI's quick check, uses a simpler workload: every stream speaks 3 s and pauses 5 s, starting at a random point of that cycle, and all streams stay open for the whole level (no arrivals, no first-chunk bursts). On this A100 it gave:

| Streams | Chunks/s | Mean batch | Finalize p50 / p95 / p99 |
|---|---|---|---|
| 128 | 416 | 3.7 | 22 / 27 / 28 ms |
| 256 | 829 | 10.0 | 31 / 39 / 41 ms |
| 448 | 1,445 | 35.5 | 73 / 138 / 204 ms |

Use it to size your own GPU quickly. Use the full load test for numbers you want to compare.

**nemotron-engine, WebSocket server** (`bench/streaming/ws_bench.sh`, 16 kHz PCM16, 20 ms frames, clients in 4 processes on the same machine)

| Target streams | Refused | Finalize p50/p95/p99 ms | Chunk lag p50/p95 ms | WER % | SLO |
|---|---|---|---|---|---|
| 256 | 0 | 26 / 65 / 206 | 19 / 30 | 25.0 | ✅ |
| 384 | 0 | 337 / 849 / 938 | 116 / 588 | 25.2 | ❌ |
| 448 | 58 | 1,862 / 2,703 / 3,473 | 1,613 / 2,255 | 23.2 | ❌ |
| 512 | 161 | 1,526 / 3,614 / 3,775 | 1,144 / 3,183 | 23.7 | ❌ |

The server loses capacity relative to the in-process API. One Python event loop receives 50 WebSocket frames per second per stream, and it shares the GIL with the GPU worker. Larger frames (60–100 ms) or separate front-end processes recover most of the difference; see [optimization-journey.md](optimization-journey.md#where-it-stands-and-what-is-next).

### Memory

The WebSocket run used `max_streams=704`. GPU memory was allocated at start-up and stayed flat through every load level and idle gap:

| | Value |
|---|---|
| PyTorch allocated | 11,657 MB |
| PyTorch reserved | 12,206 MB |
| nvidia-smi | 12,989 MB |
| Server process RSS | 1,585 MB at start → 1,972 MB after the overload levels; it grew mostly while the server was overloaded and stayed there while idle (a soak test at one level is still to do) |

Each slot costs about 11.8 MB, mostly K/V for 24 layers × 56 frames. The fixed part (weights, graphs, buffers) is about 3.4 GB. A rough guide: **VRAM ≈ 3.4 GB + 11.8 MB × max_streams**, plus ~1.3 GB of CUDA context and allocator slack.

### Correctness under load and on 8 kHz audio

From `tests/test_engine.py --strict`: 64 concurrent streams, strict fp32.

| Check | Result |
|---|---|
| Engine vs reference port, 20 ms packets | 64/64 streams identical, 0 of 9,272 tokens differ |
| 100 ms packets vs 20 ms | identical |
| `greedy_steps=1` (169 continuation-graph replays) | identical |
| 50 utterances: 16 kHz original / 16 kHz telephone band / 8 kHz μ-law | WER 26.4% / 29.5% / 28.9% |
| Stress: full engine, close mid-utterance, `end_turn` with no audio, `aclose` with open streams | pass, no leaked slots |

## Accuracy

The reference port (`research/nemotron_lite`) against NeMo, 500 VietSuperSpeech validation utterances (seed 42), strict fp32 (`bench/parity/run_parity.sh`):

| Chunk | NeMo offline | Port offline | NeMo streaming | Port streaming |
|---|---|---|---|---|
| 1120 ms (`56,13`) | 23.91% | 23.91% | 23.94% | 23.91% |
| 320 ms (`56,3`) | 25.18% | 25.19% | 25.27% | 25.19% |

Per-utterance agreement of the normalized text (`$RESULTS_DIR/comparison.md`):

| Pair | 1120 ms | 320 ms |
|---|---|---|
| Port offline vs NeMo offline | 99.2% identical | 99.0% identical |
| Port streaming vs port offline | 100% identical (tokens too) | 100% identical (tokens too) |
| NeMo streaming vs NeMo offline | 98.8% identical | 83.0% identical |

NeMo's own streaming loop (its `conformer_stream_step` example) returns a strict prefix of its offline transcript on 6 of 500 utterances at 1120 ms, and on 85 of 500 at 320 ms: it drops tail tokens at the end of the audio. The port's streaming path flushes the last chunk and matches offline exactly.

The engine's tokens are identical to the port's (see above), so the engine inherits the port's accuracy.

## Reproduce

```bash
pip install -r research/requirements.txt
STAGES="test parity capacity ws cli profile" RESULTS_DIR=results bash bench/run_all.sh
```

Or on a rented GPU: `vkong run --detach` with the repo's [`vkong.yaml`](../vkong.yaml).

Single stages:

```bash
python tests/test_engine.py --strict --n 64                                   # token parity, stress
bash bench/parity/run_parity.sh ref bench stream-ref stream-bench compare      # accuracy vs NeMo
VA_BACKENDS=nemo VA_PHASES=sweep bash bench/streaming/nemo_vs_lite.sh          # NeMo capacity
python bench/streaming/engine_loadtest.py --levels 256 448 512 --out results/engine   # engine capacity
bash bench/streaming/ws_bench.sh 256 384 448                                   # through the WebSocket server
nemotron-engine bench --streams 64 128 256 --wav speech.wav                    # quick check on your GPU
```

Measurements near capacity vary from run to run. At the highest passing level, finalize p95 of the same configuration can differ by tens of milliseconds between runs. Compare numbers only within the same GPU and run.
