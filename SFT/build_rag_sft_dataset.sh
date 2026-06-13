#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_CAR_CONTROL_DIR="${EVAL_CAR_CONTROL_DIR:-${SCRIPT_DIR}/../evaluation/car_control}"

if [[ -f "${EVAL_CAR_CONTROL_DIR}/env_qwen3_4b.sh" ]]; then
  source "${EVAL_CAR_CONTROL_DIR}/env_qwen3_4b.sh"
fi

cd "${SCRIPT_DIR}"

build_args=(
  --use-eval-rag
  --rag-topk "${RAG_TOPK:-10}"
  --single single_intent_5k.json
  --multi multi_intent_5k.json
  --negative car_control_negative_10000_by_sheet.xlsx
  --negative-sheet all_10000
  --output "${OUTPUT:-generated/car_control_sft_mix.jsonl}"
  --stats-output "${STATS_OUTPUT:-generated/car_control_sft_mix.stats.json}"
  --single-ratio "${SINGLE_RATIO:-1}"
  --multi-ratio "${MULTI_RATIO:-1}"
  --negative-ratio "${NEGATIVE_RATIO:-1}"
)

if [[ -n "${TARGET_TOTAL:-}" ]]; then
  build_args+=(--target-total "${TARGET_TOTAL}")
fi
if [[ -n "${MAX_SINGLE_CASES:-}" ]]; then
  build_args+=(--max-single-cases "${MAX_SINGLE_CASES}")
fi
if [[ -n "${MAX_MULTI_CASES:-}" ]]; then
  build_args+=(--max-multi-cases "${MAX_MULTI_CASES}")
fi
if [[ -n "${MAX_NEGATIVE_ROWS:-}" ]]; then
  build_args+=(--max-negative-rows "${MAX_NEGATIVE_ROWS}")
fi

python build_sft_dataset.py "${build_args[@]}"
