#!/usr/bin/env bash
# Qwen3-4B 本地测评环境变量（source 后生效）
# 注意：不在此设置 http_proxy，避免 SLURM 计算节点误用登录节点代理

ROOT="${ROOT:-/public/home/sjtu_jiangnan/jiangchenxi}"
DEPLOY="${ROOT}/llm_function_call_tsl_deploy-main"
MODELS_DIR="${ROOT}/models"
BGE_MODEL_DIR="${MODELS_DIR}/bge-large-zh-v1.5"

# HuggingFace 缓存也放在当前 ROOT 目录下
export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

# 本地 vLLM OpenAI 兼容接口
export TSMRT_QWEN_URL="${TSMRT_QWEN_URL:-http://127.0.0.1:8000/v1/chat/completions}"
export TSMRT_QWEN_API_KEY="${TSMRT_QWEN_API_KEY:-EMPTY}"

# RAG embedding：默认走本地模型（1024 维 bge-large-zh-v1.5，与向量库一致）
export TSMRT_EMBEDDING_MODE="${TSMRT_EMBEDDING_MODE:-local}"
export TSMRT_EMBEDDING_MODEL="${TSMRT_EMBEDDING_MODEL:-${BGE_MODEL_DIR}}"
export TSMRT_EMBEDDING_DEVICE="${TSMRT_EMBEDDING_DEVICE:-cuda}"
# 若仍用内网 API，设置 TSMRT_EMBEDDING_MODE=remote 并指定 TSMRT_EMBEDDING_URL
export TSMRT_EMBEDDING_URL="${TSMRT_EMBEDDING_URL:-http://10.12.7.83:50030/embed}"

# 模型名需与 vLLM --served-model-name 一致
export TSM_AGENT_MODEL="${TSM_AGENT_MODEL:-Qwen3-4B}"
export TSM_CLASSIFIER_MODEL="${TSM_CLASSIFIER_MODEL:-Qwen3-4B}"

# 本地模型路径
export QWEN3_4B_MODEL_PATH="${QWEN3_4B_MODEL_PATH:-${ROOT}/Qwen3-4B-Instruct-2507}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

export PYTHONPATH="${DEPLOY}:${PYTHONPATH:-}"

# 防止后续 pip 操作把 numpy 升到 2.x（faiss 1.8 不兼容）
export PIP_CONSTRAINT="${ROOT}/car_control/constraints_numpy126.txt"
