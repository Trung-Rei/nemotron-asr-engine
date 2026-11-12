#!/usr/bin/env bash
# Start the WebSocket server in the background, wait for /health, run ws_smoke.py, stop the server.
#   bash bench/streaming/ws_smoke.sh [n_streams]
set -x
source "$(dirname "$0")/../env.sh"
export NEMOTRON_MAX_STREAMS=${NEMOTRON_MAX_STREAMS:-128}
PORT=${PORT:-8091}
python -m nemotron_engine.cli serve --host 127.0.0.1 --port $PORT &
server=$!
wait_health $PORT || { kill $server; exit 1; }
python bench/streaming/ws_smoke.py --url ws://127.0.0.1:$PORT/v1/stream --n "${1:-64}"
rc=$?
kill $server
wait $server
exit $rc
