import sys
import os
import asyncio

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tsmrt.manager as manager
import json
import contextlib, io
import pdb
import signal

# "DFM3-Turbo" "qwen3-max" "qwen3-235b-a22b-instruct-2507" "deepseek-v3-250324" "doubao-seed-1-6-251015"



def refresh_dm(classifier_model, agent_model):
    # 测试dialog manager
    ## 构建实例
    c_instance = {} 
    c_instance["_description"] = "空调"
    c_instance["all_functions"] = [json.loads(it.strip()) for it in open("../../function_call_data/air_conditioner_func.jsonl", "r").readlines()]
    c_instances = [c_instance]

    dm = manager.Dialog_Manager(c_instances)
    # 配置dm的classify model
    dm.l2_classifier.model=classifier_model
    # 配置dm的function call模型
    dm.function_call_agent.model=agent_model
    dm.used_rag_search = True
    return dm

def get_function(message):
    if "call_functions" not in message:
        return {}
    all_func = message["call_functions"]
    out_data = []
    for it in all_func["modify_functions"]:
        func_name = it["function_name"]
        input_params = it["input_params"]
        input_params_new = []
        for param_name in input_params.keys():
            param_value = input_params[param_name]
            if param_value == "":
                continue
            input_params_new.append(f"{param_name}={param_value}")
        input_params_new = ",".join(input_params_new)
        c_dict = {"function_name": func_name, "params": input_params_new}
        out_data.append(c_dict)
    
    if "query_funcs" not in all_func.keys():
        return out_data
    
    for it in all_func["query_funcs"]:
        func_name = it["function_name"]
        input_params = it["input_params"]
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

async def main(fn_in, fn_out, classifier_model, agent_model):
    out_lines = []
    for line in open(fn_in):
        line = line.strip()
        data = json.loads(line)
        cases = data["cases"]
        print("-"*80)
        print("开始本次对话session")
        func_outs = []
        dm = refresh_dm(classifier_model, agent_model)
        for case in cases:
            for i in range(2):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        def timeout_handler(signum, frame):
                            raise TimeoutError("对话超时")
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)
                        try:
                            c_messages = await dm.in_session_chat(case)
                        finally:
                            signal.alarm(0)
                    c_func_out = get_function(c_messages)
                    break
                except:
                    c_func_out = "对话失败"
            print(c_messages)      
            func_outs.append(c_func_out)
            print(f"classify: {classifier_model} agent: {agent_model} case: {case}" + " => " + f"function call: {c_func_out}")
        print("-"*80)
        c_out_line = {"category":data["category"], "cases":cases}
        if "labs" in data.keys():
            c_out_line["labs"] = data["labs"]
        c_out_line["preds"] = func_outs
        out_lines.append(json.dumps(c_out_line, ensure_ascii=False))

    with open(fn_out, "w") as f:
        for line in out_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    classifier_model = "qwen3-235b-a22b-instruct-2507"
    agent_model = "qwen3-235b-a22b-instruct-2507"
    #classifier_model = "dfm3_tsl_built_in_1.0"
    #agent_model = "dfm3_tsl_built_in_1.0"


    dir_in = "../../data/home_control/ac"
    fn_in = "hc_aircontioner_uniq.jsonl"
    fn_out = fn_in.replace(".jsonl", "_pred_classify_"+classifier_model+"_agent_"+agent_model+".jsonl")
    fn_in = os.path.join(dir_in, fn_in)
    fn_out = os.path.join(dir_in, fn_out)
    
    asyncio.run(main(fn_in, fn_out, classifier_model, agent_model))


