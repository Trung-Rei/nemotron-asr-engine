#!/usr/bin/env bash
# Accuracy parity: the reference port (research/nemotron_lite) against the original NeMo model, on a seeded subset of
# VietSuperSpeech validation. Needs NeMo for the ref/dump/inspect stages: pip install -r research/requirements.txt
#
#   bash bench/parity/run_parity.sh ref bench stream-ref stream-bench compare
#
# Stages (run in the order given):
#   inspect        NeMo config, preprocessor buffers, decoder facts -> $RESULTS_DIR/inspect/
#   dump           NeMo per-layer tensors on 4 samples (batched, so padding is covered)
#   parity         lite on the same audio, compared layer by layer with the dump
#   ref / bench    NeMo transcribe() / lite offline on the subset
#   stream-ref     NeMo cache-aware streaming (NeMo's own streaming loop)
#   stream-bench   lite streaming, audio fed in 20 ms pieces
#   smoke          lite offline on 16 samples
#   compare        per-utterance agreement and WER delta -> $RESULTS_DIR/comparison.md
# Env: ATT_CONTEXTS ("56,13 56,3"), MAX_SAMPLES (500), SEED (42), BATCH_SIZE (8), TARGET_LANG (vi-VN)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
ATT_CONTEXTS="${ATT_CONTEXTS:-56,13 56,3}"
common=(--max-samples "${MAX_SAMPLES:-500}" --seed "${SEED:-42}" --target-lang "${TARGET_LANG:-vi-VN}")
bs=(--batch-size "${BATCH_SIZE:-8}")
P=bench/parity
for stage in "$@"; do
  echo "================ stage: $stage ================"
  for att in $ATT_CONTEXTS; do
    case "$stage" in
      inspect)         python $P/run_reference.py inspect --att-context "$att"; break ;;
      ref)             python $P/run_reference.py offline --att-context "$att" "${bs[@]}" "${common[@]}" ;;
      dump)            python $P/run_reference.py dump --att-context "$att" "${common[@]}" ;;
      dump-notriton)   python $P/run_reference.py dump --no-triton --att-context "$att" "${common[@]}" ;;
      parity)          python $P/run_lite.py parity --rel-tol 5e-3 --att-context "$att" "${common[@]}" ;;
      parity-notriton) python $P/run_lite.py parity --dump-suffix _notriton --att-context "$att" "${common[@]}" ;;
      bench)           python $P/run_lite.py offline --att-context "$att" "${bs[@]}" "${common[@]}" ;;
      stream-ref)      python $P/run_reference.py stream --att-context "$att" "${common[@]}" ;;
      stream-bench)    python $P/run_lite.py stream --att-context "$att" "${common[@]}" ;;
      smoke)           python $P/run_lite.py offline --att-context "$att" "${bs[@]}" --max-samples 16 --seed "${SEED:-42}" --target-lang "${TARGET_LANG:-vi-VN}" ;;
      compare)         python $P/compare.py; break ;;
      *) echo "unknown stage $stage"; exit 2 ;;
    esac
  done
done
echo "done; results in $RESULTS_DIR"
