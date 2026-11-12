# Contributing

Issues and pull requests are welcome, in English or Vietnamese.

## Setup

```bash
pip install -e ".[server,bench,dev]"   # plus torch for your CUDA version
ruff check .
pytest -q                              # CPU tests
python tests/test_engine.py --strict   # GPU: token parity with the reference port, stress tests
```

## Rules

1. **Exactness is the contract.** A change to the model path must keep `tests/test_engine.py --strict` at 64/64 identical streams. If a change is meant to trade exactness for speed (bf16, approximate attention, ...), make it opt-in and report the WER change on the benchmark set.
2. **Show the numbers.** For a performance change, include `nemotron-engine bench` (or `bench/streaming/engine_loadtest.py`) before and after, on the same GPU.
3. **Keep the engine lean.** Experiments belong in `research/` and `bench/`. `src/nemotron_engine` holds only the fast path that ships.
