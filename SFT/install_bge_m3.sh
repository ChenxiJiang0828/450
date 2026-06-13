#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

ROOT="${ROOT:-${DEFAULT_ROOT}}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/models/bge-m3}"
HF_MODEL_ID="${HF_MODEL_ID:-BAAI/bge-m3}"
MS_MODEL_ID="${MS_MODEL_ID:-BAAI/bge-m3}"
CONDA_ENV="${CONDA_ENV:-450}"

echo "[bge-m3] ROOT=${ROOT}"
echo "[bge-m3] MODEL_DIR=${MODEL_DIR}"
echo "[bge-m3] HF_MODEL_ID=${HF_MODEL_ID}"
echo "[bge-m3] MS_MODEL_ID=${MS_MODEL_ID}"

source /public/home/sjtu_jiangnan/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

python -m pip install -U \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host mirrors.tuna.tsinghua.edu.cn \
  "huggingface_hub>=0.34.0,<1.0"

mkdir -p "$(dirname "${MODEL_DIR}")"

download_with_hf() {
  local endpoint="${1:-}"
  if [[ -n "${endpoint}" ]]; then
    export HF_ENDPOINT="${endpoint}"
    echo "[bge-m3] trying HuggingFace endpoint: ${HF_ENDPOINT}"
  else
    unset HF_ENDPOINT || true
    echo "[bge-m3] trying default HuggingFace endpoint"
  fi

  huggingface-cli download "${HF_MODEL_ID}" \
    --local-dir "${MODEL_DIR}" \
    --local-dir-use-symlinks False \
    --exclude "imgs/*" "onnx/*" "*/.DS_Store" ".DS_Store"
}

download_with_modelscope() {
  echo "[bge-m3] trying ModelScope: ${MS_MODEL_ID}"
  MODEL_DIR="${MODEL_DIR}" MS_MODEL_ID="${MS_MODEL_ID}" python - <<'PY'
import os
import sys

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print("[bge-m3] modelscope is not installed; skip ModelScope fallback", file=sys.stderr)
    raise

snapshot_download(
    os.environ["MS_MODEL_ID"],
    local_dir=os.environ["MODEL_DIR"],
    ignore_file_pattern=[r"imgs/.*", r"onnx/.*", r".*\.DS_Store"],
)
PY
}

if [[ -f "${MODEL_DIR}/config.json" && -f "${MODEL_DIR}/pytorch_model.bin" ]]; then
  echo "[bge-m3] model already exists at ${MODEL_DIR}"
else
  download_with_hf "https://hf-mirror.com" \
    || download_with_hf "" \
    || download_with_modelscope
fi

MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import os
import torch
from tokenizers import Tokenizer
from transformers import AutoModel

model_dir = os.environ["MODEL_DIR"]
tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
tokenizer.enable_truncation(max_length=32)
model = AutoModel.from_pretrained(model_dir)
model.eval()
encoded = tokenizer.encode("hello")
input_ids = torch.tensor([encoded.ids], dtype=torch.long)
attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long)
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    vec = outputs.last_hidden_state[:, 0]
    vec = torch.nn.functional.normalize(vec, p=2, dim=1)[0]
print("[bge-m3] installed:", model_dir)
print("[bge-m3] embedding_dim:", len(vec))
PY

echo "[bge-m3] done"
echo "export TSMRT_EMBEDDING_MODEL=${MODEL_DIR}"
