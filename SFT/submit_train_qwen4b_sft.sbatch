#!/usr/bin/env bash
#SBATCH --job-name=car-qwen4b-sft
#SBATCH --partition=gpu_llm_small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs generated
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

source /public/home/sjtu_jiangnan/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-450}"

python -m pip install -r requirements_sft.txt

export ROOT="${ROOT:-${DEFAULT_ROOT}}"
export MODEL_PATH="${MODEL_PATH:-${ROOT}/Qwen3-4B-Instruct-2507}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/models/qwen4b-car-control-sft-lora}"
export TRAIN_FILE="${TRAIN_FILE:-${SCRIPT_DIR}/generated/car_control_sft_mix.jsonl}"

export TARGET_TOTAL="${TARGET_TOTAL:-12000}"
export SINGLE_RATIO="${SINGLE_RATIO:-2}"
export MULTI_RATIO="${MULTI_RATIO:-2}"
export NEGATIVE_RATIO="${NEGATIVE_RATIO:-1}"
export RAG_TOPK="${RAG_TOPK:-20}"
export MERGE_AND_SAVE="${MERGE_AND_SAVE:-1}"

export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"

bash run_qwen4b_sft.sh
