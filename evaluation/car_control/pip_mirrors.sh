#!/usr/bin/env bash
# vLLM + PyTorch 安装（cu121 成套，适配 A800 + CUDA 12.2 驱动）

TSINGHUA_PYPI=https://pypi.tuna.tsinghua.edu.cn/simple
TSINGHUA_PYTORCH_CU121=https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121
VLLM_VERSION=0.8.5.post1

pip_mirror_args() {
  echo -i "${TSINGHUA_PYPI}"
}

pip_trusted_hosts() {
  echo --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org --trusted-host mirrors.tuna.tsinghua.edu.cn
}

# 不要单独装 torch：vLLM 0.8.5.post1 属于 cu121 时代，会拉匹配的 torch/cuda 版本
# transformers>=5 需要更高版本 torch，必须锁在 4.x
install_vllm_stack() {
  local pip_bin=$1
  local pip_flags
  pip_flags=($(pip_trusted_hosts))
  local pkgs=(
    "vllm==${VLLM_VERSION}"
    "transformers>=4.51.1,<5.0.0"
  )

  echo "[vllm] 安装 vLLM ${VLLM_VERSION} + 匹配 torch (cu121)..."
  echo "[vllm] 尝试清华 PyPI + 清华 PyTorch cu121 镜像..."
  if "${pip_bin}" install "${pip_flags[@]}" \
    -i "${TSINGHUA_PYPI}" \
    --extra-index-url "${TSINGHUA_PYTORCH_CU121}" \
    "${pkgs[@]}"; then
    return 0
  fi

  echo "[vllm] 清华镜像失败，尝试 PyPI + 官方 cu121..."
  if "${pip_bin}" install "${pip_flags[@]}" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "${pkgs[@]}"; then
    return 0
  fi

  echo "[vllm] 最后尝试 PyPI 默认源..."
  "${pip_bin}" install "${pip_flags[@]}" "${pkgs[@]}"
}

verify_gpu_stack() {
  python - <<'PY'
import torch, vllm, faiss, numpy
assert torch.cuda.is_available(), "CUDA 不可用，请在 GPU 节点执行"
cuda = torch.version.cuda or ""
print("numpy:", numpy.__version__)
print("faiss:", faiss.__version__)
print("torch:", torch.__version__, "built_cuda:", cuda, "device:", torch.cuda.get_device_name(0))
print("vllm:", vllm.__version__)
if not cuda.startswith("12."):
    print("警告: 期望 cu12x 构建，当前为", cuda)
PY
}

# vLLM 会把 numpy 升到 2.x，导致 faiss-cpu 1.8 无法 import；锁回 numpy 1.26
fix_tsm_numpy_faiss() {
  local pip_bin=$1
  local pip_flags
  pip_flags=($(pip_trusted_hosts))
  local constraints="${ROOT:-/public/home/sjtu_jiangnan/anruitong}/car_control/constraints_numpy126.txt"

  echo "[tsm] 锁定 numpy==1.26.4 + faiss-cpu==1.8.0.post1 ..."
  echo "[tsm] 说明: cupy/opencv 可能提示需要 numpy>=2，对 TSM 测评可忽略"
  "${pip_bin}" install "${pip_flags[@]}" -i "${TSINGHUA_PYPI}" \
    -c "${constraints}" \
    --force-reinstall \
    "numpy==1.26.4" "faiss-cpu==1.8.0.post1" || true

  # 确保 numpy 未被其他包再次拉高
  "${pip_bin}" install "${pip_flags[@]}" -i "${TSINGHUA_PYPI}" \
    "numpy==1.26.4" --force-reinstall --no-deps

  python - <<'PY'
import faiss, numpy
assert numpy.__version__.startswith("1.26"), numpy.__version__
print("faiss ok:", faiss.__version__, "numpy:", numpy.__version__)
PY
}
