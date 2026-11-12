# Benchmarks

*[Tiếng Việt](README.vi.md)*

Install everything first, NeMo included, on a machine with an NVIDIA GPU:

```bash
pip install -r research/requirements.txt
```

`bash bench/run_all.sh` runs everything. Pick stages with `STAGES="test parity rtfx capacity ws cli profile"`. Results go to `$RESULTS_DIR` (default `results/`). Results and methodology are in [docs/benchmarks.md](../docs/benchmarks.md).

| Folder | Script | What it measures |
|---|---|---|
| `parity/` | `run_parity.sh` | Accuracy of the reference port vs NeMo: offline and streaming, per-utterance agreement, WER. Also layer-by-layer tensor parity (`dump` / `parity` stages). |
| `rtfx/` | `bench_rtfx.sh` | Offline throughput (RTFx) with NeMo's own `transcribe_speech.py` vs the port, same manifest and settings. |
| `streaming/` | `engine_loadtest.py` | Real-time load test through the engine's asyncio API: Poisson stream arrivals, 20 ms packets on the wall clock, `end_turn` at each utterance end. Reports finalize latency, chunk lag, throughput and WER. |
| `streaming/` | `loadtest.py`, `nemo_vs_lite.sh` | The same workload driven by one synchronous scheduler for NeMo's cache-aware streaming model and the reference port. This is how NeMo's capacity is measured. |
| `streaming/` | `ws_loadtest.py`, `ws_bench.sh` | The same workload through the WebSocket server, with clients in separate processes and a GPU/RSS memory monitor. |
| `streaming/` | `ws_smoke.sh` | Server smoke test: WER over concurrent 8 kHz μ-law streams, and close code 1013 when full. |
| `profiling/` | `graph_breakdown.py`, `profile_engine.py`, `profile_server.py`, `profile_step.py`, `multi_engine_stress.sh` | Time per part of one step, worker phases and GIL wait, server event-loop cost, two engines on one GPU. |

**Workload** (`streaming/workload.py`):
- Each session is one continuous 16 kHz stream.
- Utterances and pauses are drawn from log-normal distributions: utterance p50 2.4 s / p90 12.3 s, pause p50 6.5 s / p90 25 s, ~30% speech.
- Utterances are filled with VietSuperSpeech audio, and pauses carry low-level noise.
- `--telephone` band-limits the audio to 8 kHz.
