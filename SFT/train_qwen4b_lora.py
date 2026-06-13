#!/usr/bin/env python3
"""LoRA SFT entrypoint for Qwen car-control tool-call data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl_messages(path: str) -> Any:
    return load_dataset("json", data_files=path, split="train")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        msg = dict(message)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls = []
            for index, call in enumerate(msg["tool_calls"]):
                if "function" in call and "type" in call:
                    calls.append(call)
                    continue
                args = call.get("arguments", {})
                calls.append(
                    {
                        "id": call.get("id") or f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                        },
                    }
                )
            msg["tool_calls"] = calls
            msg.setdefault("content", "")
        normalized.append(msg)
    return normalized


def build_text_formatter(tokenizer: AutoTokenizer):
    def formatter(example: dict[str, Any]) -> str:
        messages = normalize_messages(example["messages"])
        tools = example.get("tools") or None
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )

    return formatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA on car-control SFT data.")
    parser.add_argument("--model", required=True, help="Base Qwen Instruct model path, e.g. /public/.../Qwen3-4B-Instruct-2507")
    parser.add_argument("--train-file", required=True, help="JSONL produced by build_sft_dataset.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names.",
    )
    parser.add_argument("--merge-and-save", action="store_true", help="Also save a merged full model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 and not args.fp16 else torch.float16 if args.fp16 else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataset = load_jsonl_messages(args.train_file)
    formatter = build_text_formatter(tokenizer)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        report_to="none",
        packing=False,
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatter,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.merge_and_save:
        merged_dir = output_dir / "merged"
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))


if __name__ == "__main__":
    main()
