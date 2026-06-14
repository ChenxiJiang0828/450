#!/usr/bin/env python3
"""Evaluate a Qwen chat/tool-call model on generated car-control SFT JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def normalize_args(args: Any) -> dict[str, Any]:
    if args is None or args == "":
        return {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {"__raw__": args}
        return parsed if isinstance(parsed, dict) else {"__raw__": parsed}
    return args if isinstance(args, dict) else {"__raw__": args}


def normalize_tool_calls(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized = []
    for call in calls or []:
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name") or ""
        args = fn.get("arguments") if "arguments" in fn else call.get("arguments")
        if name:
            normalized.append({"name": name, "arguments": normalize_args(args)})
    return normalized


def canonical_calls(calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    result = []
    for call in calls:
        args_json = json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result.append((call.get("name", ""), args_json))
    return result


def parse_json_tool_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        parsed = []
        for item in payload:
            parsed.extend(parse_json_tool_payload(item))
        return parsed
    if not isinstance(payload, dict):
        return []

    if "tool_calls" in payload:
        return normalize_tool_calls(payload.get("tool_calls") or [])
    if "function" in payload or "name" in payload:
        return normalize_tool_calls([payload])
    return []


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    candidates = []
    candidates.extend(match.group(1) for match in TOOL_BLOCK_RE.finditer(text))
    candidates.extend(match.group(1) for match in JSON_BLOCK_RE.finditer(text))
    candidates.append(text.strip())

    parsed_calls = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        parsed_calls.extend(parse_json_tool_payload(payload))
    return parsed_calls


def make_prompt(tokenizer: AutoTokenizer, example: dict[str, Any]) -> str:
    messages = []
    for message in example["messages"]:
        if message.get("role") == "assistant":
            break
        messages.append(message)
    tools = example.get("tools") or None
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def eval_one(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    example: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = make_prompt(tokenizer, example)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    assistant = example["messages"][-1]
    gold_calls = normalize_tool_calls(assistant.get("tool_calls"))
    pred_calls = parse_tool_calls(generated_text)
    return {
        "id": example.get("id"),
        "source": example.get("source"),
        "task_type": example.get("task_type"),
        "gold_calls": gold_calls,
        "pred_calls": pred_calls,
        "gold_names": [call["name"] for call in gold_calls],
        "pred_names": [call["name"] for call in pred_calls],
        "generated_text": generated_text,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    counters = Counter()
    by_task: dict[str, Counter] = {}
    by_source: dict[str, Counter] = {}

    for item in results:
        gold = item["gold_calls"]
        pred = item["pred_calls"]
        gold_names = item["gold_names"]
        pred_names = item["pred_names"]
        task = item.get("task_type") or ""
        source = item.get("source") or ""
        by_task.setdefault(task, Counter())
        by_source.setdefault(source, Counter())

        has_gold = bool(gold)
        has_pred = bool(pred)
        name_exact = gold_names == pred_names
        name_set_exact = set(gold_names) == set(pred_names)
        call_exact = canonical_calls(gold) == canonical_calls(pred)

        counters["total"] += 1
        counters["gold_tool"] += int(has_gold)
        counters["pred_tool"] += int(has_pred)
        counters["tool_presence_correct"] += int(has_gold == has_pred)
        counters["name_exact"] += int(name_exact)
        counters["name_set_exact"] += int(name_set_exact)
        counters["call_exact"] += int(call_exact)
        counters["no_tool_correct"] += int(not has_gold and not has_pred)
        counters["tool_name_exact_on_gold"] += int(has_gold and name_exact)
        counters["tool_call_exact_on_gold"] += int(has_gold and call_exact)

        for group in (by_task[task], by_source[source]):
            group["total"] += 1
            group["gold_tool"] += int(has_gold)
            group["pred_tool"] += int(has_pred)
            group["tool_presence_correct"] += int(has_gold == has_pred)
            group["name_exact"] += int(name_exact)
            group["name_set_exact"] += int(name_set_exact)
            group["call_exact"] += int(call_exact)
            group["no_tool_correct"] += int(not has_gold and not has_pred)
            group["tool_name_exact_on_gold"] += int(has_gold and name_exact)
            group["tool_call_exact_on_gold"] += int(has_gold and call_exact)

    def rates(counter: Counter) -> dict[str, Any]:
        total = counter["total"] or 1
        gold_total = counter["gold_tool"] or 1
        no_tool_total = counter["total"] - counter["gold_tool"] or 1
        return {
            **dict(counter),
            "tool_presence_acc": counter["tool_presence_correct"] / total,
            "name_exact_acc": counter["name_exact"] / total,
            "name_set_exact_acc": counter["name_set_exact"] / total,
            "call_exact_acc": counter["call_exact"] / total,
            "tool_name_exact_on_gold_acc": counter["tool_name_exact_on_gold"] / gold_total,
            "tool_call_exact_on_gold_acc": counter["tool_call_exact_on_gold"] / gold_total,
            "no_tool_acc": counter["no_tool_correct"] / no_tool_total,
        }

    return {
        "overall": rates(counters),
        "by_task": {key: rates(value) for key, value in by_task.items()},
        "by_source": {key: rates(value) for key, value in by_source.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen tool-call outputs on SFT JSONL.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 and not args.fp16 else torch.float16 if args.fp16 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows = read_jsonl(args.input, args.limit)
    results = []
    with args.output.open("w", encoding="utf-8") as out:
        for example in tqdm(rows, desc="evaluating"):
            item = eval_one(tokenizer, model, example, args.max_new_tokens)
            results.append(item)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()

    summary = summarize(results)
    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
