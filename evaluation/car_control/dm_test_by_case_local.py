import sys
import os
import time
import logging
import uuid
import asyncio
import json
import contextlib
import io
import argparse
import copy
from typing import Any, List

DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../llm_function_call_tsl_deploy-main"))
sys.path.insert(0, DEPLOY_ROOT)

import tsmrt.manager as dialog_manager
from tsmrt.tsm_input import TsmInput, Session, DialogHistory, DialogOutput, SpeakOutput, Functions, FunctionRequest, RequestBody
from tsmrt.tsm_output import TsmOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_ORIG_FROM_CURRENT_MESSAGE = TsmOutput.from_current_message.__func__


def params_str_to_dict(params: str) -> dict:
    if not params:
        return {}
    args = {}
    for part in params.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        args[key] = value
    return args


def preds_to_tool_calls(preds: list[dict]) -> list[dict]:
    tool_calls = []
    for pred in preds:
        tool_calls.append(
            {
                "id": "",
                "name": pred.get("function_name", ""),
                "arguments": params_str_to_dict(pred.get("params", "")),
                "requestor": "assistant",
            }
        )
    return tool_calls


def serialize_llm_responses(llm_responses: list[Any]) -> list[Any]:
    serialized = []
    for item in llm_responses:
        if item == "null":
            serialized.append("null")
            continue
        if isinstance(item, dict):
            serialized.append(copy.deepcopy(item))
        else:
            serialized.append(str(item))
    return serialized


def get_assistant_content(llm_responses: list[Any]) -> str:
    if not llm_responses:
        return ""
    last = llm_responses[-1]
    if last == "null" or not isinstance(last, dict):
        return ""
    return (last.get("content") or "").strip()


def build_trace(turn_records: list[dict]) -> list[dict]:
    trace = []
    for turn in turn_records:
        trace.append({"role": "user", "content": turn["user"]})
        assistant: dict[str, Any] = {"role": "assistant"}
        content = turn.get("assistant_content", "")
        tool_calls = turn.get("tool_calls", [])
        if content:
            assistant["content"] = content
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        if content or tool_calls:
            trace.append(assistant)
        elif turn.get("error"):
            assistant["content"] = ""
            assistant["error"] = turn["error"]
            trace.append(assistant)
    return trace


def extract_turn_record(
    case: str,
    tsm_output: TsmOutput | None,
    current_message: dict | None,
    llm_responses: list[Any],
    preds: list[dict],
    error: str | None = None,
) -> dict:
    rag_candidates = {}
    metric = {}
    hotfix_used = False

    if current_message:
        l2_result = current_message.get("l2_result")
        if l2_result:
            try:
                rag_candidates = json.loads(l2_result)
            except json.JSONDecodeError:
                rag_candidates = {"raw": l2_result}
        metric = {
            key: current_message.get(key)
            for key in ("systime", "hottime", "clstime", "fuctime")
            if current_message.get(key) is not None
        }
        hotfix_used = bool(current_message.get("hotfix_used"))

    if tsm_output and tsm_output.metric:
        metric.update(
            {
                key: value
                for key, value in tsm_output.metric.model_dump().items()
                if value is not None
            }
        )

    return {
        "user": case,
        "assistant_content": get_assistant_content(llm_responses),
        "tool_calls": preds_to_tool_calls(preds),
        "preds": preds,
        "rag_candidates": rag_candidates,
        "llm_responses": serialize_llm_responses(llm_responses),
        "metric": metric,
        "hotfix_used": hotfix_used,
        "error": error,
    }


class TestDialogManager:
    def __init__(self, instances, c_info=None):
        self.dialog_manager = dialog_manager.Dialog_Manager(instances, c_info)
        self.session = Session(dialogHistory=[])

    def refresh_session(self):
        self.session = Session(dialogHistory=[])
        self.dialog_manager.refresh_message()

    def add_dialog_history(self, user_input: str, nlg_output: str = "", function_name: str = "", skill: str = "", source: str = "", data_source: str = ""):
        dialog_output = None
        if nlg_output:
            dialog_output = DialogOutput(speak=SpeakOutput(text=nlg_output))

        functions = None
        if function_name:
            functions = Functions(request=FunctionRequest(name=function_name))

        dialog_history = DialogHistory(
            timestamp=int(time.time() * 1000),
            input=user_input,
            skill=skill,
            source=source,
            output=dialog_output,
            functions=functions,
            dataSource=data_source,
        )

        if self.session.dialogHistory is None:
            self.session.dialogHistory = []
        self.session.dialogHistory.append(dialog_history)

    async def chat_process(self, user_input: str) -> tuple[TsmOutput, dict | None]:
        request_id = str(uuid.uuid4())
        request_body = RequestBody(input=user_input, requestId=request_id, recordId=request_id)
        tsm_input = TsmInput(request=request_body, session=self.session)

        captured: dict[str, dict | None] = {"current_message": None}

        @classmethod
        def capture_from_current_message(cls, current_message, *args, **kwargs):
            captured["current_message"] = copy.deepcopy(current_message)
            return _ORIG_FROM_CURRENT_MESSAGE(cls, current_message, *args, **kwargs)

        TsmOutput.from_current_message = capture_from_current_message
        try:
            tsm_output = await self.dialog_manager.chat_process(tsm_input)
        finally:
            TsmOutput.from_current_message = classmethod(_ORIG_FROM_CURRENT_MESSAGE)

        dlg_input = tsm_output.request.text
        llm_responses = getattr(self.dialog_manager.function_call_agent, "llm_responses", [])
        assistant_content = get_assistant_content(llm_responses)
        dlg_func = ""
        if tsm_output.result and tsm_output.result.tsmResult and len(tsm_output.result.tsmResult) > 0:
            if tsm_output.result.tsmResult[0].functions and tsm_output.result.tsmResult[0].functions.request:
                dlg_func = tsm_output.result.tsmResult[0].functions.request.name or ""

        if assistant_content:
            dlg_nlg = assistant_content
        elif dlg_func:
            dlg_nlg = "好的"
        else:
            dlg_nlg = ""

        self.add_dialog_history(dlg_input, dlg_nlg, dlg_func)
        return tsm_output, captured["current_message"]


def get_function_from_output(tsm_output: TsmOutput):
    if not tsm_output.result or not tsm_output.result.tsmResult or len(tsm_output.result.tsmResult) == 0:
        return []

    out_data = []
    for tsm_result_item in tsm_output.result.tsmResult:
        if not tsm_result_item.functions or not tsm_result_item.functions.request:
            continue

        func_name = tsm_result_item.functions.request.name
        input_params = {}
        if tsm_result_item.functions.request.params:
            for param in tsm_result_item.functions.request.params:
                param_name = param.name
                param_value = param.value if hasattr(param, "value") else ""
                if param_value:
                    input_params[param_name] = param_value

        input_params_new = []
        for param_name in input_params.keys():
            param_value = input_params[param_name]
            if param_value == "":
                continue
            input_params_new.append(f"{param_name}={param_value}")
        input_params_new = ",".join(input_params_new)

        out_data.append({"function_name": func_name, "params": input_params_new})

    return out_data


def refresh_dm(classifier_model, agent_model):
    test_dm = TestDialogManager(None)
    test_dm.dialog_manager.l2_classifier.model = classifier_model
    test_dm.dialog_manager.function_call_agent.model = agent_model
    test_dm.dialog_manager.used_rag_search = True
    test_dm.dialog_manager.rag_topk = 20
    return test_dm


async def main(fn_in, fn_out, classifier_model, agent_model, max_samples=None):
    out_lines = []
    sample_count = 0

    with open(fn_in, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if fn_in.endswith(".jsonl"):
                data = json.loads(line)
            elif "=>" in line:
                case, label = [part.strip() for part in line.split("=>", 1)]
                data = {"cases": [case], "category": "单轮单意图", "labs": [[{"function_name": label, "params": ""}]]}
            else:
                data = {"cases": [line], "category": "单轮单意图", "labs": []}

            cases = data["cases"]
            print("-" * 80)
            print("开始本次对话 session")

            func_outs = []
            turn_records = []
            test_dm = refresh_dm(classifier_model, agent_model)

            for case in cases:
                tsm_output = None
                current_message = None
                llm_responses = []
                error = None
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        tsm_output, current_message = await asyncio.wait_for(
                            test_dm.chat_process(case),
                            timeout=60.0,
                        )
                    c_func_out = get_function_from_output(tsm_output)
                    llm_responses = getattr(test_dm.dialog_manager.function_call_agent, "llm_responses", [])
                except Exception as e:
                    logger.error(f"对话失败: {type(e).__name__}: {e!r}")
                    c_func_out = []
                    error = f"{type(e).__name__}: {e!r}"

                func_outs.append(c_func_out)
                turn_records.append(
                    extract_turn_record(
                        case=case,
                        tsm_output=tsm_output,
                        current_message=current_message,
                        llm_responses=llm_responses,
                        preds=c_func_out,
                        error=error,
                    )
                )
                logger.info(
                    f"classify: {classifier_model} agent: {agent_model} case: {case} => function call: {c_func_out}"
                )

            print("-" * 80)

            c_out_line = {"category": data["category"], "cases": cases}
            if "case_id" in data:
                c_out_line["case_id"] = data["case_id"]
            if "labs" in data:
                c_out_line["labs"] = data["labs"]
            c_out_line["preds"] = func_outs
            c_out_line["turns"] = turn_records
            c_out_line["trace"] = build_trace(turn_records)
            out_lines.append(json.dumps(c_out_line, ensure_ascii=False))

            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break

    os.makedirs(os.path.dirname(fn_out) or ".", exist_ok=True)
    with open(fn_out, "w", encoding="utf-8") as f:
        for out_line in out_lines:
            f.write(out_line + "\n")

    logger.info(f"写入 {len(out_lines)} 条结果 -> {fn_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="car_control 本地 Qwen3-4B 批量测评")
    parser.add_argument(
        "--input",
        default=os.path.join(DEPLOY_ROOT, "function_call_data/car_control_examples.txt"),
        help="测试集路径 (.txt / .jsonl)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出 jsonl 路径（默认写到 eval_results/）",
    )
    parser.add_argument(
        "--classifier-model",
        default=os.getenv("TSM_CLASSIFIER_MODEL", "Qwen3-4B"),
    )
    parser.add_argument(
        "--agent-model",
        default=os.getenv("TSM_AGENT_MODEL", "Qwen3-4B"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多跑多少条（debug 用）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classifier_model = args.classifier_model
    agent_model = args.agent_model
    fn_in = args.input

    if args.output:
        fn_out = args.output
    else:
        base = os.path.basename(fn_in)
        stem = base.rsplit(".", 1)[0]
        fn_out = os.path.join(
            os.path.dirname(__file__),
            "../eval_results",
            f"{stem}_pred_classify_{classifier_model}_agent_{agent_model}_local.jsonl",
        )
        fn_out = os.path.abspath(fn_out)

    asyncio.run(main(fn_in, fn_out, classifier_model, agent_model, args.max_samples))
