#!/usr/bin/env bash
# Offline RTFx: NeMo's own examples/asr/transcribe_speech.py vs bench/rtfx/transcribe_lite.py, same manifest and settings.
# NeMo prints a very large config; its output goes to a log file and only the timing lines are echoed.
#   [DTYPES="float32 bfloat16"] [LOOKAHEADS="13 3"] bash bench/rtfx/bench_rtfx.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
OUT="$RESULTS_DIR/rtfx"
mkdir -p "$OUT"
LOOKAHEADS="${LOOKAHEADS:-13 3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-3}"
IMPLS="${IMPLS:-nemo lite}"
DTYPES="${DTYPES:-float32 bfloat16}"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0))"
python bench/parity/make_manifest.py --out "$OUT/manifest.json"
NEMO_PATH=$(python -c "from huggingface_hub import hf_hub_download as d;print(d('nvidia/nemotron-3.5-asr-streaming-0.6b','nemotron-3.5-asr-streaming-0.6b.nemo'))")

for dt in $DTYPES; do
D="$OUT/$dt"; mkdir -p "$D"
common=(dataset_manifest="$OUT/manifest.json" batch_size="$BATCH_SIZE" target_lang=vi-VN compute_dtype="$dt"
        calculate_rtfx=true warmup_steps="$WARMUP" run_steps="$RUNS")
for r in $LOOKAHEADS; do
  for impl in $IMPLS; do
    echo "================ $impl r=$r $dt ================"
    if [ "$impl" = nemo ]; then
      python bench/rtfx/nemo_examples/transcribe_speech.py model_path="$NEMO_PATH" "${common[@]}" \
        att_context_size="[56,$r]" output_filename="$D/nemo_r$r.json" > "$D/nemo_r$r.log" 2>&1 \
        || { tail -40 "$D/nemo_r$r.log"; exit 1; }
      grep -E "Model time|Model RTFx" "$D/nemo_r$r.log"
    else
      python bench/rtfx/transcribe_lite.py "${common[@]}" att_context_size="[56,$r]" \
        output_filename="$D/lite_r$r.json" 2>&1 | tee "$D/lite_r$r.log"
    fi
  done
done
python bench/rtfx/summarize_rtfx.py "$D"
done
