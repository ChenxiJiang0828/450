# Car-Control Qwen4B SFT

This repository is a standalone package for building RAG-conditioned SFT data and fine-tuning the final Qwen4B car-control function-call model.

Main entry points:

- `SFT/build_rag_sft_dataset.sh`: build mixed SFT JSONL with the same RAG candidate-tool retrieval path used by evaluation.
- `SFT/run_qwen4b_sft.sh`: build data and run LoRA SFT.
- `SFT/submit_build_rag_sft.sbatch`: submit RAG data build on the cluster.
- `SFT/submit_train_qwen4b_sft.sbatch`: submit LoRA SFT on the cluster.
- `SFT/README_SFT.md`: detailed usage.

The copied evaluation runtime is sanitized: API keys must be supplied through environment variables if a remote provider is used. The intended cluster path uses local vLLM and local embedding settings from `evaluation/car_control/env_qwen3_4b.sh`.
