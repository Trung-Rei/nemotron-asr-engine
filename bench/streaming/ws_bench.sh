#!/usr/bin/env bash
# Load test through the WebSocket server, with the memory monitor (see ws_loadtest.py).
# The server and the client processes run on one machine and talk over localhost.
#   [PROFILE=1] [PACKETS="20 100"] bash bench/streaming/ws_bench.sh [levels...]
# PROFILE=1 runs the server under bench/profiling/profile_server.py and records a server-side profile per level.
set -x
source "$(dirname "$0")/../env.sh"
export NEMOTRON_MAX_STREAMS=${NEMOTRON_MAX_STREAMS:-704} NEMOTRON_MAX_LAG_MS=${NEMOTRON_MAX_LAG_MS:-none}
PORT=${PORT:-8091}
if [[ -n "${PROFILE:-}" ]]; then
  python bench/profiling/profile_server.py --port $PORT &
  PROF="--prof-url http://127.0.0.1:$PORT"
else
  python -m nemotron_engine.cli serve --host 127.0.0.1 --port $PORT &
fi
server=$!
wait_health $PORT || { kill $server; exit 1; }
levels=${*:-256 448 512 576}
rc=0
for pk in ${PACKETS:-20}; do
  python bench/streaming/ws_loadtest.py --url ws://127.0.0.1:$PORT/v1/stream --levels $levels --packet-ms $pk \
    ${PROF:-} ${WS_ARGS:-} --tag p$pk --out "$RESULTS_DIR/ws_bench" || rc=$?
done
kill $server
wait $server
exit $rc
