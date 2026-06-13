#!/usr/bin/env python3
"""Build mixed car-control SFT data for Qwen tool-call fine-tuning."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SYSTEM_PROMPT = (
    "你是车载工具调用助手。根据用户输入和对话历史，输出一个或多个车控工具调用；"
    "如果不需要或不允许调用车控工具，不要输出工具调用，只给一句简短中文回复。"
    "工具调用必须严格使用提供的函数名和参数。完成工具调用后，根据工具执行结果给一句简短自然语言回复。"
)

NEGATIVE_NLG = {
    "chat": "这个不需要执行车控操作，我可以继续和你聊。",
    "asr_reject": "我没听清明确的车控需求，请再说一遍要控制的功能。",
    "out_of_domain": "这个不属于车控执行范围，我不能调用车控工具。",
}


def load_function_defs(path: Path) -> dict[str, dict[str, Any]]:
    funcs = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            funcs[item["name"]] = item
    return funcs


def function_to_tool(func: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    required = []
    for name, param in (func.get("input_param") or {}).items():
        define = param.get("define") or {}
        schema_type = define.get("type") or "string"
        if schema_type == "float":
            schema_type = "number"
        prop: dict[str, Any] = {
            "type": schema_type,
            "description": param.get("description", ""),
        }
        choices = define.get("choice") or []
        if choices:
            prop["enum"] = choices
        properties[name] = prop
        if define.get("required"):
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def collect_gold_function_names(trace: list[dict[str, Any]], start: int, end: int) -> list[str]:
    names = []
    for msg in trace[start:end]:
        for call in msg.get("tool_calls") or []:
            name = call.get("name") or call.get("function", {}).get("name")
            if name:
                names.append(name)
    return list(dict.fromkeys(names))


class CandidateToolProvider:
    def __init__(
        self,
        deploy_root: Path,
        function_defs: dict[str, dict[str, Any]],
        topk: int,
        use_eval_rag: bool,
        ensure_gold_tools: bool,
    ) -> None:
        self.deploy_root = deploy_root
        self.function_defs = function_defs
        self.topk = topk
        self.use_eval_rag = use_eval_rag
        self.ensure_gold_tools = ensure_gold_tools
        self.classifier = None
        self.instances = None
        self.cache: dict[str, list[str]] = {}
        self.missing_gold_counts: dict[str, int] = {}
        self.rag_disabled_reason = ""

    def _init_eval_rag(self) -> None:
        if self.classifier is not None:
            return
        sys.path.insert(0, str(self.deploy_root))
        from tsmrt import classify  # type: ignore
        import tsmrt.tsm_rag as tsm_rag  # type: ignore

        if not hasattr(tsm_rag, "traceback"):
            tsm_rag.traceback = traceback

        self.instances = [
            {
                "_description": "智能座舱",
                "all_functions": list(self.function_defs.values()),
                "all_functions_define": {"CarControl": list(self.function_defs)},
            }
        ]
        self.classifier = classify.Level_2_Classifier("")
        self.classifier.update_instances(self.instances)

    async def retrieve_names(
        self,
        eval_text_list: list[list[str]],
        gold_names: list[str],
    ) -> list[str]:
        names: list[str] = []
        if self.use_eval_rag and not self.rag_disabled_reason:
            try:
                self._init_eval_rag()
                cache_key = json.dumps(eval_text_list[-3:], ensure_ascii=False)
                if cache_key in self.cache:
                    names = list(self.cache[cache_key])
                else:
                    rag_result, _vocab_hit_result, _modified_input = await self.classifier.rag_search_function(
                        ["CarControl"],
                        [list(item) for item in eval_text_list],
                        top_k=self.topk,
                    )
                    for funcs in rag_result.values():
                        names.extend(funcs)
                    names = [name for name in dict.fromkeys(names) if name in self.function_defs]
                    self.cache[cache_key] = names
            except Exception as exc:
                self.rag_disabled_reason = f"{type(exc).__name__}: {exc}"
                print(f"[WARN] eval RAG failed, falling back to gold-only tools: {self.rag_disabled_reason}")
                names = []

        if self.ensure_gold_tools:
            names.extend(gold_names)

        names = [name for name in dict.fromkeys(names) if name in self.function_defs]
        return names

    async def tools_for(
        self,
        eval_text_list: list[list[str]],
        gold_names: list[str],
    ) -> list[dict[str, Any]]:
        names = await self.retrieve_names(eval_text_list, gold_names)
        return [function_to_tool(self.function_defs[name]) for name in names]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def normalize_tool_call(call: dict[str, Any], index: int) -> dict[str, Any]:
    name = call.get("name") or call.get("function", {}).get("name")
    args = call.get("arguments")
    if args is None and isinstance(call.get("function"), dict):
        raw_args = call["function"].get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = raw_args
        else:
            args = raw_args
    if args is None:
        args = {}
    return {
        "id": call.get("id") or f"call_{index}",
        "type": "function",
        "function": {
            "name": name or "",
            "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
        },
    }


def user_prompt(history: list[tuple[str, str, str]], current: str) -> str:
    if not history:
        return current
    parts = []
    for user_text, assistant_text, _func_name in history[-5:]:
        parts.append(f"历史输入:{user_text}\n历史输出:{assistant_text}")
    parts.append(f"当前输入:{current}")
    return "\n".join(parts)


def eval_text_list(history: list[tuple[str, str, str]], current: str) -> list[list[str]]:
    items = [[user_text, nlg, func_name] for user_text, nlg, func_name in history[-5:]]
    items.append([current, "", ""])
    return items


async def extract_positive_examples(
    item: dict[str, Any],
    source: str,
    system_prompt: str,
    candidate_provider: CandidateToolProvider,
) -> list[dict[str, Any]]:
    trace = item.get("trace", [])
    examples: list[dict[str, Any]] = []
    history: list[tuple[str, str, str]] = []
    idx = 0

    while idx < len(trace):
        turn = trace[idx]
        if turn.get("role") != "user":
            idx += 1
            continue

        current_user = (turn.get("content") or "").strip()
        prompt = user_prompt(history, current_user)
        assistant_tool_calls: list[dict[str, Any]] = []
        assistant_content = ""
        j = idx + 1

        while j < len(trace) and trace[j].get("role") != "user":
            msg = trace[j]
            if msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    assistant_tool_calls.extend(msg["tool_calls"])
                if msg.get("content"):
                    assistant_content = str(msg["content"]).strip()
            j += 1

        gold_names = collect_gold_function_names(trace, idx + 1, j)
        tools = await candidate_provider.tools_for(
            eval_text_list(history, current_user),
            gold_names,
        )

        if assistant_tool_calls:
            missing_gold = [name for name in gold_names if name not in candidate_provider.function_defs]
            if missing_gold:
                for name in missing_gold:
                    candidate_provider.missing_gold_counts[name] = (
                        candidate_provider.missing_gold_counts.get(name, 0) + 1
                    )
                idx = j
                continue
            normalized_calls = [
                normalize_tool_call(call, call_idx)
                for call_idx, call in enumerate(assistant_tool_calls)
            ]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": normalized_calls,
                },
            ]
            examples.append(
                {
                    "id": f"{item.get('case_id', source)}__turn_{len(examples)}__tool",
                    "source": source,
                    "task_type": "tool_call_with_nlg",
                    "tools": tools,
                    "candidate_function_names": [
                        tool["function"]["name"] for tool in tools
                    ],
                    "gold_function_names": gold_names,
                    "messages": messages,
                }
            )
        elif assistant_content:
            examples.append(
                {
                    "id": f"{item.get('case_id', source)}__turn_{len(examples)}__nlg",
                    "source": source,
                    "task_type": "no_tool_nlg",
                    "tools": tools,
                    "candidate_function_names": [
                        tool["function"]["name"] for tool in tools
                    ],
                    "gold_function_names": [],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_content},
                    ],
                }
            )

        if current_user:
            history.append((current_user, assistant_content or "好的", gold_names[0] if gold_names else ""))
        idx = j

    return examples


def read_negative_rows(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    required = {"id", "query", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}:{sheet_name} missing columns: {sorted(missing)}")
    rows = []
    for row in df.to_dict(orient="records"):
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        rows.append(row)
    return rows


async def make_negative_examples(
    rows: list[dict[str, Any]],
    system_prompt: str,
    candidate_provider: CandidateToolProvider,
) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        category = str(row.get("category") or "").strip()
        response = NEGATIVE_NLG.get(category, "这个不属于车控执行范围，我不能调用车控工具。")
        query = str(row["query"]).strip()
        tools = await candidate_provider.tools_for([[query, "", ""]], [])
        examples.append(
            {
                "id": str(row.get("id") or f"negative_{len(examples)}"),
                "source": f"negative:{category}",
                "task_type": "negative_no_tool",
                "tools": tools,
                "candidate_function_names": [
                    tool["function"]["name"] for tool in tools
                ],
                "gold_function_names": [],
                "negative_meta": {
                    "category": category,
                    "subcategory": row.get("subcategory"),
                    "domain": row.get("domain"),
                    "expected_action": row.get("expected_action"),
                    "response_policy": row.get("response_policy"),
                },
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ],
            }
        )
    return examples


def sample_by_ratio(
    pools: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    target_total: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    active = {k: v for k, v in pools.items() if ratios.get(k, 0) > 0 and v}
    if not active:
        return []

    if target_total is None:
        selected = []
        for key, pool in active.items():
            selected.extend(pool)
        rng.shuffle(selected)
        return selected

    ratio_sum = sum(ratios[k] for k in active)
    selected = []
    remaining = target_total
    keys = list(active)
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            count = remaining
        else:
            count = int(round(target_total * ratios[key] / ratio_sum))
            count = min(count, remaining)
        pool = active[key]
        if count <= len(pool):
            selected.extend(rng.sample(pool, count))
        else:
            selected.extend(pool)
            selected.extend(rng.choices(pool, k=count - len(pool)))
        remaining -= count
    rng.shuffle(selected)
    return selected


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed SFT JSONL data.")
    parser.add_argument("--single", type=Path, default=Path("single_intent_5k.json"))
    parser.add_argument("--multi", type=Path, default=Path("multi_intent_5k.json"))
    parser.add_argument("--negative", type=Path, default=Path("car_control_negative_10000_by_sheet.xlsx"))
    parser.add_argument("--negative-sheet", default="all_10000")
    parser.add_argument("--output", type=Path, default=Path("generated/car_control_sft_mix.jsonl"))
    parser.add_argument("--stats-output", type=Path, default=Path("generated/car_control_sft_mix.stats.json"))
    parser.add_argument("--single-ratio", type=float, default=1.0)
    parser.add_argument("--multi-ratio", type=float, default=1.0)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--rag-topk", type=int, default=10)
    parser.add_argument(
        "--use-eval-rag",
        action="store_true",
        help="Use evaluation tsmrt RAG to populate candidate tools for every sample.",
    )
    parser.add_argument(
        "--deploy-root",
        type=Path,
        default=Path("../evaluation/llm_function_call_tsl_deploy-main"),
    )
    parser.add_argument(
        "--function-defs",
        type=Path,
        default=Path("../evaluation/llm_function_call_tsl_deploy-main/function_call_data/car_control_func.jsonl"),
    )
    parser.add_argument(
        "--no-ensure-gold-tools",
        action="store_false",
        dest="ensure_gold_tools",
        help="Do not force gold functions into candidate tools when RAG misses them.",
    )
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--max-single-cases", type=int, default=None)
    parser.add_argument("--max-multi-cases", type=int, default=None)
    parser.add_argument("--max-negative-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    single_path = args.single if args.single.is_absolute() else base_dir / args.single
    multi_path = args.multi if args.multi.is_absolute() else base_dir / args.multi
    negative_path = args.negative if args.negative.is_absolute() else base_dir / args.negative
    output_path = args.output if args.output.is_absolute() else base_dir / args.output
    stats_path = args.stats_output if args.stats_output.is_absolute() else base_dir / args.stats_output
    deploy_root = args.deploy_root if args.deploy_root.is_absolute() else base_dir / args.deploy_root
    function_defs_path = args.function_defs if args.function_defs.is_absolute() else base_dir / args.function_defs
    function_defs = load_function_defs(function_defs_path)
    candidate_provider = CandidateToolProvider(
        deploy_root=deploy_root.resolve(),
        function_defs=function_defs,
        topk=args.rag_topk,
        use_eval_rag=args.use_eval_rag,
        ensure_gold_tools=args.ensure_gold_tools,
    )

    single_examples = []
    single_items = load_json(single_path)
    if args.max_single_cases is not None:
        single_items = single_items[: args.max_single_cases]
    for item in single_items:
        single_examples.extend(
            await extract_positive_examples(item, "single_intent", args.system_prompt, candidate_provider)
        )

    multi_examples = []
    multi_items = load_json(multi_path)
    if args.max_multi_cases is not None:
        multi_items = multi_items[: args.max_multi_cases]
    for item in multi_items:
        multi_examples.extend(
            await extract_positive_examples(item, "multi_intent", args.system_prompt, candidate_provider)
        )

    negative_rows = read_negative_rows(negative_path, args.negative_sheet)
    if args.max_negative_rows is not None:
        negative_rows = negative_rows[: args.max_negative_rows]
    negative_examples = await make_negative_examples(negative_rows, args.system_prompt, candidate_provider)

    pools = {
        "single": single_examples,
        "multi": multi_examples,
        "negative": negative_examples,
    }
    ratios = {
        "single": args.single_ratio,
        "multi": args.multi_ratio,
        "negative": args.negative_ratio,
    }
    rng = random.Random(args.seed)
    mixed = sample_by_ratio(pools, ratios, args.target_total, rng)
    write_jsonl(output_path, mixed)

    stats = {
        "output": str(output_path),
        "total": len(mixed),
        "pools": {key: len(value) for key, value in pools.items()},
        "ratios": ratios,
        "target_total": args.target_total,
        "use_eval_rag": args.use_eval_rag,
        "rag_topk": args.rag_topk,
        "ensure_gold_tools": args.ensure_gold_tools,
        "rag_disabled_reason": candidate_provider.rag_disabled_reason,
        "max_single_cases": args.max_single_cases,
        "max_multi_cases": args.max_multi_cases,
        "max_negative_rows": args.max_negative_rows,
        "task_type_counts": {},
        "source_counts": {},
        "avg_candidate_tools": 0.0,
        "skipped_missing_gold_function_counts": candidate_provider.missing_gold_counts,
    }
    total_candidate_tools = 0
    for item in mixed:
        stats["task_type_counts"][item["task_type"]] = stats["task_type_counts"].get(item["task_type"], 0) + 1
        stats["source_counts"][item["source"]] = stats["source_counts"].get(item["source"], 0) + 1
        total_candidate_tools += len(item.get("tools") or [])
    if mixed:
        stats["avg_candidate_tools"] = total_candidate_tools / len(mixed)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(async_main())
