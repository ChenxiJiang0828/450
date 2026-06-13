#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export ROOT="${ROOT:-${DEFAULT_ROOT}}"
EVAL_CAR_CONTROL_DIR="${EVAL_CAR_CONTROL_DIR:-${SCRIPT_DIR}/../evaluation/car_control}"

if [[ -f "${EVAL_CAR_CONTROL_DIR}/env_qwen3_4b.sh" ]]; then
  # Reuse evaluation embedding/model environment so SFT RAG matches inference.
  source "${EVAL_CAR_CONTROL_DIR}/env_qwen3_4b.sh"
fi

MODEL_PATH="${MODEL_PATH:-${ROOT}/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/models/qwen4b-car-control-sft-lora}"
TRAIN_FILE="${TRAIN_FILE:-${SCRIPT_DIR}/generated/car_control_sft_mix.jsonl}"

SINGLE_RATIO="${SINGLE_RATIO:-1}"
MULTI_RATIO="${MULTI_RATIO:-1}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1}"
TARGET_TOTAL="${TARGET_TOTAL:-}"
USE_EVAL_RAG="${USE_EVAL_RAG:-1}"
RAG_TOPK="${RAG_TOPK:-10}"

cd "${SCRIPT_DIR}"

build_args=(
  --single single_intent_5k.json
  --multi multi_intent_5k.json
  --negative car_control_negative_10000_by_sheet.xlsx
  --negative-sheet all_10000
  --output "${TRAIN_FILE}"
  --single-ratio "${SINGLE_RATIO}"
  --multi-ratio "${MULTI_RATIO}"
  --negative-ratio "${NEGATIVE_RATIO}"
  --rag-topk "${RAG_TOPK}"
)

if [[ "${USE_EVAL_RAG}" == "1" ]]; then
  build_args+=(--use-eval-rag)
fi

if [[ -n "${TARGET_TOTAL}" ]]; then
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

train_args=(
  --model "${MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --max-seq-length "${MAX_SEQ_LENGTH:-4096}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-2}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --save-steps "${SAVE_STEPS:-500}"
  --logging-steps "${LOGGING_STEPS:-10}"
)

if [[ "${MERGE_AND_SAVE:-0}" == "1" ]]; then
  train_args+=(--merge-and-save)
fi

python train_qwen4b_lora.py \
  "${train_args[@]}" \
  ${EXTRA_TRAIN_ARGS:-}

echo "SFT finished: ${OUTPUT_DIR}"
