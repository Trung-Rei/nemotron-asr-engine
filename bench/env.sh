# Shared settings for the benchmark scripts (sourced, not run).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export RESULTS_DIR="${RESULTS_DIR:-results}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
mkdir -p "$RESULTS_DIR"

wait_health() {  # wait_health PORT: block until the server answers /health (up to 5 min for the first model download)
  for _ in $(seq 300); do
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$1/health', timeout=2)" 2>/dev/null && return 0
    sleep 1
  done
  echo "server on port $1 did not come up"; return 1
}
