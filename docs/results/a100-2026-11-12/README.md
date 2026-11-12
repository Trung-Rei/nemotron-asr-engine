# Raw results: A100-SXM4-40GB, 2026-11-12

The summaries behind [docs/benchmarks.md](../../benchmarks.md), as written by `bench/run_all.sh` (`STAGES="test parity capacity ws cli profile"`). Per-utterance transcripts and per-chunk logs are left out to keep the repo small; rerun the benchmarks to regenerate them.

| Path | From |
|---|---|
| `*.summary.json`, `comparison.*` | `bench/parity/run_parity.sh` (NeMo = `ref_*`, reference port = `lite_*`; `r13` = 1120 ms, `r3` = 320 ms chunks) |
| `nemo_vs_lite/` | `bench/streaming/nemo_vs_lite.sh`, NeMo backend |
| `capacity_engine/`, `capacity_engine_nograph/` | `bench/streaming/engine_loadtest.py` with and without CUDA graphs |
| `ws_bench/` | `bench/streaming/ws_bench.sh` (WebSocket server, memory timeline) |
| `test_engine/` | `tests/test_engine.py --strict` |
| `profile/` | `bench/profiling/graph_breakdown.py`, `bench/profiling/profile_engine.py` |
