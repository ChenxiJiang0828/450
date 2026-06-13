# Car-Control Qwen4B SFT

This folder builds a mixed SFT set from:

- `single_intent_5k.json`
- `multi_intent_5k.json`
- `car_control_negative_10000_by_sheet.xlsx`, sheet `all_10000`

The output follows the current evaluation flow:

- each train turn first runs the same evaluation RAG path, `Level_2_Classifier.rag_search_function(..., top_k=RAG_TOPK)`, to get candidate functions
- candidate functions are converted to OpenAI-compatible `tools` from `evaluation/.../function_call_data/car_control_func.jsonl`
- input is formatted like `function_call.py`: recent history becomes `历史输入/历史输出`, and the active utterance becomes `当前输入`
- positive turns supervise one assistant message containing OpenAI-compatible `tool_calls` plus one short NLG sentence in `content`
- negative rows still receive RAG candidate tools, but supervise no tool call and only a short refusal/clarification NLG
- functions absent from the current evaluation tool definition file are skipped and counted in the stats JSON

## Install

```bash
pip install -r User-Simulator/SFT/requirements_sft.txt
```

Use the same GPU environment as evaluation when possible.

## Build mixed data

Default ratio is `single:multi:negative = 1:1:1`.

```bash
cd User-Simulator/SFT
python build_sft_dataset.py \
  --use-eval-rag \
  --rag-topk 10 \
  --single-ratio 1 \
  --multi-ratio 1 \
  --negative-ratio 1 \
  --output generated/car_control_sft_mix.jsonl
```

For a fixed-size experiment:

```bash
python build_sft_dataset.py --use-eval-rag --target-total 12000 --single-ratio 2 --multi-ratio 2 --negative-ratio 1
```

On the cluster, prefer the wrapper because it sources `evaluation/car_control/env_qwen3_4b.sh` first, matching evaluation RAG and local embedding settings:

```bash
cd User-Simulator/SFT
TARGET_TOTAL=12000 SINGLE_RATIO=2 MULTI_RATIO=2 NEGATIVE_RATIO=1 RAG_TOPK=20 bash build_rag_sft_dataset.sh
```

## Train LoRA

```bash
cd User-Simulator/SFT
MODEL_PATH=/public/home/sjtu_jiangnan/jiangchenxi/Qwen3-4B-Instruct-2507 \
OUTPUT_DIR=/public/home/sjtu_jiangnan/jiangchenxi/models/qwen4b-car-control-sft-lora \
MERGE_AND_SAVE=1 \
bash run_qwen4b_sft.sh
```

## Server / SLURM

The login node should only be used for upload, environment checks, and `sbatch`.
Do not run `bash run_qwen4b_sft.sh` directly on the login node.

First run a tiny smoke job and inspect logs:

```bash
cd User-Simulator/SFT
TARGET_TOTAL=100 MAX_SINGLE_CASES=50 MAX_MULTI_CASES=50 MAX_NEGATIVE_ROWS=50 sbatch submit_build_rag_sft.sbatch
tail -f logs/car-sft-rag-build-<jobid>.out
```

Build the full RAG dataset:

```bash
cd User-Simulator/SFT
TARGET_TOTAL=12000 SINGLE_RATIO=2 MULTI_RATIO=2 NEGATIVE_RATIO=1 RAG_TOPK=20 sbatch submit_build_rag_sft.sbatch
```

Train LoRA:

```bash
cd User-Simulator/SFT
MODEL_PATH=/public/home/sjtu_jiangnan/jiangchenxi/Qwen3-4B-Instruct-2507 \
OUTPUT_DIR=/public/home/sjtu_jiangnan/jiangchenxi/models/qwen4b-car-control-sft-lora \
TRAIN_FILE=$PWD/generated/car_control_sft_mix.jsonl \
sbatch submit_train_qwen4b_sft.sbatch
```

With your current server layout:

```text
/public/home/sjtu_jiangnan/jiangchenxi/
  Qwen3-4B-Instruct-2507/
  car-control-qwen4b-sft/
```

the default `ROOT` and `MODEL_PATH` are already correct. You can submit directly:

```bash
cd /public/home/sjtu_jiangnan/jiangchenxi/car-control-qwen4b-sft/SFT
TARGET_TOTAL=100 MAX_SINGLE_CASES=50 MAX_MULTI_CASES=50 MAX_NEGATIVE_ROWS=50 sbatch submit_build_rag_sft.sbatch
sbatch submit_train_qwen4b_sft.sbatch
```

If the local BGE embedding model is not at `$ROOT/models/bge-large-zh-v1.5`, override it:

```bash
TSMRT_EMBEDDING_MODEL=/public/share/model/bge-large-zh-v1.5 sbatch submit_build_rag_sft.sbatch
```

If your server copy uses a different home root, override `ROOT`, `MODEL_PATH`, and `OUTPUT_DIR` in the `sbatch` command.

Useful knobs:

```bash
SINGLE_RATIO=2 MULTI_RATIO=2 NEGATIVE_RATIO=1 TARGET_TOTAL=12000 RAG_TOPK=20 bash run_qwen4b_sft.sh
USE_EVAL_RAG=0 bash run_qwen4b_sft.sh  # debug only; uses gold tools for positive samples
```

`MERGE_AND_SAVE=1` writes a merged model under:

```text
$OUTPUT_DIR/merged
```

Point evaluation vLLM at that merged path:

```bash
export QWEN3_4B_MODEL_PATH=/public/home/sjtu_jiangnan/jiangchenxi/models/qwen4b-car-control-sft-lora/merged
export TSM_AGENT_MODEL=Qwen3-4B
bash User-Simulator/evaluation/car_control/run_dm_test_local.sh --input your_eval.jsonl
```

If you want to serve the LoRA adapter directly instead of a merged model, start vLLM with LoRA support and keep `TSM_AGENT_MODEL` aligned with the served model name.
