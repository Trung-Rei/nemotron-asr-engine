#!/usr/bin/env bash
# Every benchmark in this repo, in one script. Run it on a GPU machine after
#   pip install -r research/requirements.txt        (the engine + benchmark deps + NeMo for the reference runs)
#
#   STAGES="test parity capacity ws" bash bench/run_all.sh
#
# Stages:
#   test       CPU unit tests; GPU tests: token parity engine vs reference port (strict fp32), 8 kHz mu-law, stress
#   parity     accuracy: reference port vs NeMo, offline and streaming, 500 VietSuperSpeech utterances, 320 ms + 1120 ms
#   rtfx       offline throughput (RTFx): NeMo's transcribe_speech.py vs the reference port
#   capacity   concurrent real-time streams at p95 finalize <= 300 ms on this GPU: NeMo (cache-aware streaming,
#              shared scheduler) and nemotron-engine (asyncio API), same workload
#   ws         the WebSocket server: smoke test, then the load test through the server
#   cli        nemotron-engine transcribe and nemotron-engine bench
#   profile    time per part of one step (graph breakdown), CUDA graphs on vs off, worker phases and GIL wait
# Results go to $RESULTS_DIR (default results/).
set -uo pipefail
source "$(dirname "$0")/env.sh"
STAGES="${STAGES:-test parity capacity ws cli}"
NEMO_LEVELS="${NEMO_LEVELS:-64 128 192 256 320}"
ENGINE_LEVELS="${ENGINE_LEVELS:-256 384 448 512 576 640}"
status=()
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.get_device_name(0))"

run() {  # run NAME CMD...: keep going after a failure, report at the end
  local name=$1; shift
  echo "================ $name ================"
  local t0=$SECONDS
  "$@"; local rc=$?
  status+=("$name rc=$rc $((SECONDS - t0))s")
  echo "== $name done rc=$rc in $((SECONDS - t0)) s"
}

for stage in $STAGES; do
  case "$stage" in
    test)
      run pytest-cpu python -m pytest -q tests/test_audio_cpu.py
      run test-engine python tests/test_engine.py --strict --n 64 --n-8k 50 --out "$RESULTS_DIR/test_engine"
      run test-engine-greedy1 python tests/test_engine.py --strict --n 64 --greedy-steps 1 --parity-only ;;
    parity)
      run parity bash bench/parity/run_parity.sh ref bench stream-ref stream-bench compare ;;
    rtfx)
      run rtfx bash bench/rtfx/bench_rtfx.sh ;;
    capacity)
      run capacity-nemo env VA_BACKENDS=nemo VA_PHASES=sweep VA_LEVELS="$NEMO_LEVELS" bash bench/streaming/nemo_vs_lite.sh
      run capacity-engine python bench/streaming/engine_loadtest.py --levels $ENGINE_LEVELS --max-streams 704 \
        --stop-after-fail 2 --out "$RESULTS_DIR/capacity_engine" ;;
    ws)
      run ws-smoke bash bench/streaming/ws_smoke.sh 64
      run ws-bench bash bench/streaming/ws_bench.sh 256 384 448 512 ;;
    cli)
      wav=$(python - <<'PY'
import sys
sys.path.insert(0, "bench/parity")
import soundfile as sf
from common import load_rows_with_audio
rows, _ = load_rows_with_audio(4, 42)
for i, r in enumerate(rows):
    a, sr = sf.read(r["local_audio"], dtype="int16")
    sf.write(f"/tmp/sample{i}.wav", a, sr, subtype="PCM_16")
print(" ".join(f"/tmp/sample{i}.wav" for i in range(len(rows))))
PY
)
      wav=$(echo "$wav" | tail -1)  # load_rows_with_audio prints progress on stdout too
      run cli-transcribe nemotron-engine transcribe $wav --language vi-VN
      run cli-bench nemotron-engine bench --streams 128 256 448 --seconds 60 --language vi-VN --wav ${wav%% *} ;;
    profile)
      run profile-graph python bench/profiling/graph_breakdown.py --batch 64 32 8 --stages --out "$RESULTS_DIR/profile/graph"
      run profile-engine python bench/profiling/profile_engine.py --max-streams 704 --runs 448:20 512:20 512:100 \
        --out "$RESULTS_DIR/profile/engine"
      run capacity-engine-nograph python bench/streaming/engine_loadtest.py --levels 256 384 448 --max-streams 704 \
        --no-cuda-graphs --stop-after-fail 1 --tag _nograph --out "$RESULTS_DIR/capacity_engine_nograph" ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
done
echo "================ summary ================"
printf '%s\n' "${status[@]}"
