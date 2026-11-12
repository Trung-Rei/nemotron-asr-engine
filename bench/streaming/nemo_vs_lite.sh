#!/usr/bin/env bash
# Streaming load test with one shared scheduler (loadtest.py): smoke -> open-workload sweep -> burst, for the original
# NeMo model and the reference port. Needs NeMo: pip install -r research/requirements.txt
#   [VA_BACKENDS="nemo lite"] [VA_LEVELS="64 128 192 256 384"] [VA_PHASES="sweep burst"] bash bench/streaming/nemo_vs_lite.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
OUT="${VA_OUT:-$RESULTS_DIR/nemo_vs_lite}"
BACKENDS="${VA_BACKENDS:-nemo lite}"
SWEEP_BACKENDS="${VA_SWEEP_BACKENDS:-$BACKENDS}"
BURST_BACKENDS="${VA_BURST_BACKENDS:-$BACKENDS}"
LEVELS="${VA_LEVELS:-64 128 192 256 384}"
TIMING=(--session-s "${VA_SESSION_S:-40}" --warmup "${VA_WARMUP:-40}" --duration "${VA_DURATION:-90}")
LOOKAHEAD="${VA_LOOKAHEAD:-3}"
DTYPE="${VA_DTYPE:-float32}"
PHASES="${VA_PHASES:-sweep burst}"
mkdir -p "$OUT"
common=(--lookahead "$LOOKAHEAD" --dtype "$DTYPE")

lt() {  # run one loadtest, keep NeMo's verbose output in a file, echo only progress and results
  local log="$1"; shift
  set +o pipefail
  python bench/streaming/loadtest.py "$@" 2>&1 | tee "$log" | grep --line-buffered -E '^(===|\{|!!|stopping|backend=|Traceback)' | awk '{print substr($0,1,900); fflush()}'
  local rc=${PIPESTATUS[0]}
  set -o pipefail
  [ "$rc" -eq 0 ] || { tail -60 "$log"; return 1; }
}

for phase in $PHASES; do
  echo "================ phase: $phase ================"
  case "$phase" in sweep) list=$SWEEP_BACKENDS ;; burst) list=$BURST_BACKENDS ;; *) list=$BACKENDS ;; esac
  for b in $list; do
    case "$phase" in
      smoke) lt "$OUT/smoke_$b.log" --backend "$b" --mode sweep --levels 8 --session-s 20 --warmup 10 --duration 40 \
               --pool-size 200 "${common[@]}" --out "$OUT/smoke" ;;
      sweep) lt "$OUT/sweep_$b.log" --backend "$b" --mode sweep --levels $LEVELS "${TIMING[@]}" "${common[@]}" --out "$OUT/sweep" ;;
      burst)
        # steady load at half of the weaker backend's capacity, then as many extra sessions as the stronger one sustains
        read -r BASE EXTRA < <(python - "$OUT/sweep" <<'PY'
import json, sys, pathlib
caps = []
by_backend = {}
for p in pathlib.Path(sys.argv[1]).glob("*_open_c*.json"):
    r = json.loads(p.read_text())
    by_backend.setdefault(r["backend"], []).append(r)
for rows in by_backend.values():
    ok = [r["target_concurrency"] for r in rows if r["slo_ok"]]
    caps.append(max(ok) if ok else 16)
print(max(8, min(caps) // 2), max(16, max(caps)))
PY
)
        echo "burst: base=$BASE extra=$EXTRA"
        lt "$OUT/burst_$b.log" --backend "$b" --mode burst --burst-base "$BASE" --burst-sessions "$EXTRA" \
           --session-s 40 --warmup 40 --burst-at 60 --burst-spread 5 --duration 120 "${common[@]}" --out "$OUT/burst" ;;
    esac
  done
done
python bench/streaming/report.py "$OUT"
