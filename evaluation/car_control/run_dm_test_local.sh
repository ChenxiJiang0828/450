#!/usr/bin/env bash
# 在 GPU 节点上：启动 vLLM(Qwen3-4B) + 跑 car_control 批量测评
set -euo pipefail

ROOT=/public/home/sjtu_jiangnan/anruitong
DEPLOY="${ROOT}/llm_function_call_tsl_deploy-main"
SCRIPT_DIR="${ROOT}/car_control"

source /public/home/sjtu_jiangnan/anaconda3/etc/profile.d/conda.sh
conda activate 450
source "${SCRIPT_DIR}/env_qwen3_4b.sh"
source "${SCRIPT_DIR}/network_env.sh"
configure_network_for_current_node

mkdir -p "${ROOT}/logs" "${ROOT}/eval_results"

# vLLM 安装可能升级 numpy，测评依赖 faiss 1.8 需要 numpy<2
if ! python -c "import faiss" 2>/dev/null; then
  echo "检测到 faiss/numpy 不兼容，正在修复..."
  source "${SCRIPT_DIR}/pip_mirrors.sh"
  fix_tsm_numpy_faiss "$(which pip)"
fi

if ! python -c "import vllm" 2>/dev/null; then
  echo "错误: 环境 450 中未安装 vLLM/torch。"
  echo "请先一次性安装（清华镜像，约 10-20 分钟）:"
  echo "  bash ${ROOT}/submit_setup_vllm_gpu.sh"
  echo "完成后再提交测评任务。"
  exit 1
fi

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "停止 vLLM (pid=${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

free_vllm_port() {
  if command -v fuser >/dev/null 2>&1; then
    echo "释放端口 ${VLLM_PORT}（如有残留 vLLM 进程）..."
    fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
    sleep 2
  fi
}

wait_for_vllm() {
  local url="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
  local i
  echo "等待 vLLM 就绪（首次加载+编译约 2-5 分钟，日志: logs/vllm_qwen3_4b.log）..."
  for i in $(seq 1 120); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      echo "vLLM 就绪: ${url}"
      return 0
    fi
    if (( i % 6 == 0 )); then
      echo "仍在等待 vLLM... 已 ${i}0s"
      tail -1 "${ROOT}/logs/vllm_qwen3_4b.log" 2>/dev/null || true
    fi
    sleep 5
  done
  echo "vLLM 启动超时，请检查 ${ROOT}/logs/vllm_qwen3_4b.log"
  return 1
}

free_vllm_port
echo "启动 vLLM: ${QWEN3_4B_MODEL_PATH} (port ${VLLM_PORT})"
python -m vllm.entrypoints.openai.api_server \
  --model "${QWEN3_4B_MODEL_PATH}" \
  --served-model-name "${TSM_AGENT_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "${ROOT}/logs/vllm_qwen3_4b.log" 2>&1 &
VLLM_PID=$!

wait_for_vllm

cd "${SCRIPT_DIR}"
python dm_test_by_case_local.py "$@"
