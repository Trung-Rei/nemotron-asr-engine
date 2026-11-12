"""Put the repo's code on sys.path for the benchmark and test scripts, so they run from a plain checkout.

  src/                 nemotron_engine (also fine when it is pip-installed)
  research/            nemotron_lite, the reference port
  bench/parity         common.py: dataset, text normalization, result I/O
  bench/streaming      workload.py and the load tests
  bench/profiling      profilers
  tests/               test_engine.py (helpers reused by the stress test)
  examples/            ws_client.py (reused by the WebSocket smoke test)
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("examples", "tests", "bench/profiling", "bench/streaming", "bench/parity", "research", "src"):
    p = str(REPO / p)
    if p not in sys.path:
        sys.path.insert(0, p)
