#!/usr/bin/env bash
#SBATCH --job-name=car-qwen4b-eval
#SBATCH --partition=gpu_llm_small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs eval_results
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

source /public/home/sjtu_jiangnan/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-450}"

export ROOT="${ROOT:-${DEFAULT_ROOT}}"
export MODEL_PATH="${MODEL_PATH:-${ROOT}/Qwen3-4B-Instruct-2507}"
export INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/generated/car_control_sft_mix.jsonl}"
export RUN_NAME="${RUN_NAME:-qwen4b_instruct_baseline}"
export OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/eval_results/${RUN_NAME}.jsonl}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${SCRIPT_DIR}/eval_results/${RUN_NAME}.summary.json}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

cmd=(
  python evaluate_qwen4b_toolcall.py
  --model "${MODEL_PATH}"
  --input "${INPUT_FILE}"
  --output "${OUTPUT_FILE}"
  --summary-output "${SUMMARY_OUTPUT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${LIMIT:-}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"
