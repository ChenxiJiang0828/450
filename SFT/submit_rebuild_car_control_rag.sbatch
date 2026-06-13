#!/usr/bin/env bash
#SBATCH --job-name=car-rag-index
#SBATCH --partition=gpu_llm_small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${REPO_ROOT}/evaluation/llm_function_call_tsl_deploy-main"
DEFAULT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

cd "${SCRIPT_DIR}"
mkdir -p logs

source /public/home/sjtu_jiangnan/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-450}"

python -m pip install -r requirements_sft.txt

export ROOT="${ROOT:-${DEFAULT_ROOT}}"
export TSMRT_EMBEDDING_MODE="${TSMRT_EMBEDDING_MODE:-local}"
export TSMRT_EMBEDDING_MODEL="${TSMRT_EMBEDDING_MODEL:-${ROOT}/models/bge-m3}"
export TSMRT_EMBEDDING_DEVICE="${TSMRT_EMBEDDING_DEVICE:-cuda}"
export TSMRT_EMBEDDING_WORKERS="${TSMRT_EMBEDDING_WORKERS:-1}"
export TSMRT_UPDATE_EMBEDDING_DOMAINS="${TSMRT_UPDATE_EMBEDDING_DOMAINS:-car_control}"

cd "${DEPLOY}"
python update_embedding.py
