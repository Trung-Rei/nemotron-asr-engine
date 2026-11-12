#!/usr/bin/env bash
# Two engines on one GPU at once: A) in one process, B) in two processes. Checks for CUDA faults and token drift.
#   bash bench/profiling/multi_engine_stress.sh [rounds]
source "$(dirname "$0")/../env.sh"
R=${1:-10}
S=bench/profiling/multi_engine_stress.py
O="$RESULTS_DIR/multi_engine"
mkdir -p "$O"
echo "=== A: 2 engines, 1 process"
python $S --engines 2 --rounds $R --out "$O/A"; echo "A rc=$?"
echo "=== B: 2 processes, 1 engine each"
python $S --engines 1 --rounds $R --out "$O/B" > "$O/B1.log" 2>&1 & p1=$!
python $S --engines 1 --rounds $R --out "$O/B" > "$O/B2.log" 2>&1 & p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
grep -hE "round|engine|Error|error|DONE" "$O/B1.log" "$O/B2.log"
echo "B rc=$r1,$r2"
