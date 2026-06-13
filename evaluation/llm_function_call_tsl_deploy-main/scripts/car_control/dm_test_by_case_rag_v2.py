import sys
import os
import time
import logging
import uuid
import asyncio
import json
import contextlib
import io
import signal
from typing import List
import pdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tsmrt.manager as dialog_manager
from tsmrt.tsm_input import TsmInput, Session, DialogHistory, DialogOutput, SpeakOutput, Functions, FunctionRequest, RequestBody
from tsmrt.tsm_output import TsmOutput

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDialogManager:
    def __init__(self, instances, c_info=None):
        self.dialog_manager = dialog_manager.Dialog_Manager(instances, c_info)
        self.session = Session(dialogHistory=[])

    def refresh_session(self):
        self.session = Session(dialogHistory=[])
        self.dialog_manager.refresh_message()

    def update_dialog_state(self, tsm_input: TsmInput):
        self.session = Session(dialogHistory=[])
        if tsm_input.session and tsm_input.session.dialogHistory:
            self.session.dialogHistory = tsm_input.session.dialogHistory.copy()

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
            dataSource=data_source
        )
        
        if self.session.dialogHistory is None:
            self.session.dialogHistory = []
        self.session.dialogHistory.append(dialog_history)

    def get_dialog_history(self) -> List[DialogHistory]:
        return self.session.dialogHistory if self.session.dialogHistory else []

    async def chat_process(self, user_input: str) -> TsmOutput:
        request_id = str(uuid.uuid4())
        request_body = RequestBody(input=user_input, requestId=request_id, recordId=request_id)
        tsm_input = TsmInput(request=request_body, session=self.session)
        
        tsm_output = await self.dialog_manager.chat_process(tsm_input)
        
        dlg_input = tsm_output.request.text
        dlg_nlg = "好的"
        dlg_func = ""
        if tsm_output.result and tsm_output.result.tsmResult and len(tsm_output.result.tsmResult) > 0:
            if tsm_output.result.tsmResult[0].functions and tsm_output.result.tsmResult[0].functions.request:
                dlg_func = tsm_output.result.tsmResult[0].functions.request.name or ""
        
        self.add_dialog_history(dlg_input, dlg_nlg, dlg_func)

        return tsm_output

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
                param_value = param.value if hasattr(param, 'value') else ""
                if param_value:
                    input_params[param_name] = param_value
        
        input_params_new = []
        for param_name in input_params.keys():
            param_value = input_params[param_name]
            if param_value == "":
                continue
            input_params_new.append(f"{param_name}={param_value}")
        input_params_new = ",".join(input_params_new)
        
        c_dict = {"function_name": func_name, "params": input_params_new}
        out_data.append(c_dict)
    
    return out_data

def refresh_dm(classifier_model, agent_model):

    # c_instance = {
    #     "_description": "智能座舱",
    #     "all_functions": [json.loads(it.strip()) for it in open("../../function_call_data/car_control_func.jsonl", "r").readlines()] + 
    #                     [json.loads(it.strip()) for it in open("../../function_call_data/map_func.jsonl", "r").readlines()] +
    #                     [json.loads(it.strip()) for it in open("../../function_call_data/multimedia_func.jsonl", "r").readlines()]
    # }
    # c_instances = [c_instance]
    c_instances = None

    test_dm = TestDialogManager(c_instances)
    test_dm.dialog_manager.l2_classifier.model = classifier_model
    test_dm.dialog_manager.function_call_agent.model = agent_model
    test_dm.dialog_manager.used_rag_search = True
    test_dm.dialog_manager.rag_topk = 20
    
    return test_dm

async def main(fn_in, fn_out, classifier_model, agent_model):
    out_lines = []
    for line in open(fn_in):
        line = line.strip()
        if fn_in.endswith(".jsonl"):
            data = json.loads(line)
        else:
            data = {}
            data["cases"] = [line]
            data["category"] = "单轮单意图"
            data["labs"] = []
        cases = data["cases"]
        
        print("-"*80)
        print("开始本次对话session")
        
        func_outs = []
        test_dm = refresh_dm(classifier_model, agent_model)
        
        for case in cases:
            for attempt in range(1):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        try:
                            tsm_output = await asyncio.wait_for(
                                test_dm.chat_process(case),
                                timeout=30.0
                            )
                        except asyncio.TimeoutError:
                            raise TimeoutError("对话超时")
                    
                    c_func_out = get_function_from_output(tsm_output)
                    break
                except Exception as e:
                    logger.error(f"对话失败 (attempt {attempt+1}): {e}")
                    c_func_out = []
            
            func_outs.append(c_func_out)
            logger.info(f"classify: {classifier_model} agent: {agent_model} case: {case} => function call: {c_func_out}")
        
        print("-"*80)
        
        c_out_line = {"category": data["category"], "cases": cases}
        if "labs" in data.keys():
            c_out_line["labs"] = data["labs"]
        c_out_line["preds"] = func_outs
        out_lines.append(json.dumps(c_out_line, ensure_ascii=False))

    with open(fn_out, "w") as f:
        for line in out_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    classifier_model = "dfm3_tsl_built_in_1.0"
    agent_model = "dfm3_tsl_built_in_1.0"

    dir_in = "../../data/car_control/"
    fn_in = "test_data_merged_v1.2503_v1_8037.convert.jsonl"
    # dir_in = "../../data/map/"
    fn_in = "changcheng_examples.txt"
    fn_in = "toy_example.txt"
    # fn_in = "scene_gen.txt"
    # fn_in = "人设切换.txt"
    # fn_in = "车控function例句.txt"
    # fn_in = "sort_session_all_2025.10000_dedup.txt"
    fn_in = "multi_toy.jsonl"
    fn_in = "leapmotor_tsm_failed_actual_carcontrol_pure.txt"
    fn_in = "lingpao_car.txt"
    fn_in = "leapmortor_0519.txt"
    fn_in = "multimedia_case_20260519.txt"
    fn_in = "wuling_regression_0526.txt"

    if fn_in.endswith(".jsonl"):
        fn_out = fn_in.replace(".jsonl", f"_pred_classify_{classifier_model}_agent_{agent_model}_v2.jsonl")
    else:
        fn_out = fn_in.replace(".txt", f"_pred_classify_{classifier_model}_agent_{agent_model}_v2.jsonl")
    fn_in = os.path.join(dir_in, fn_in)
    fn_out = os.path.join(dir_in, fn_out)
    
    asyncio.run(main(fn_in, fn_out, classifier_model, agent_model))
