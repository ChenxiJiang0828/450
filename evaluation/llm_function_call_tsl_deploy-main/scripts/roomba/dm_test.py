import sys
import os
import asyncio
import pdb
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tsmrt.manager as dialog_manager
import json

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
    return out_data


# 测试dialog manager
## 构建实例
c_instance = {} 
c_instance["_description"] = "扫地机器人"
c_instance["all_functions"] = [json.loads(it.strip()) for it in open("../../function_call_data/roomba_func.jsonl", "r").readlines()]
c_instances = [c_instance]

dm = dialog_manager.Dialog_Manager(c_instances)
# 配置dm的classify model(Qwen3-235B-A22B, DFM3-Turbo, qwen3-max, qwen3-235b-a22b-instruct-2507, deepseek-v3-250324, doubao-seed-1-6-251015)
dm.l2_classifier.model="qwen3-235b-a22b-instruct-2507"

# 配置dm的function call model(Qwen3-235B-A22B, DFM3-Turbo, qwen3-max, qwen3-235b-a22b-instruct-2507, deepseek-v3-250324, doubao-seed-1-6-251015)
dm.function_call_agent.model="qwen3-235b-a22b-instruct-2507"

'''
dm.in_session_chat("水量调到中")
dm.in_session_chat("算了最大吧")
'''

# dm.in_session_chat("打开拖地模式")


async def main():
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        current_message = await dm.chat_entry(user_input)
        print(current_message)
        func_out = get_function(current_message)
        print(func_out)

asyncio.run(main())
