#!/usr/bin/env bash
#SBATCH --job-name=car-sft-rag-build
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
mkdir -p logs generated

source /public/home/sjtu_jiangnan/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-450}"

python -m pip install -r requirements_sft.txt

export TARGET_TOTAL="${TARGET_TOTAL:-12000}"
export SINGLE_RATIO="${SINGLE_RATIO:-2}"
export MULTI_RATIO="${MULTI_RATIO:-2}"
export NEGATIVE_RATIO="${NEGATIVE_RATIO:-1}"
export RAG_TOPK="${RAG_TOPK:-20}"
export OUTPUT="${OUTPUT:-generated/car_control_sft_mix.jsonl}"
export STATS_OUTPUT="${STATS_OUTPUT:-generated/car_control_sft_mix.stats.json}"

bash build_rag_sft_dataset.sh
