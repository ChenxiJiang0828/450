from calendar import c
from typing import Any
import logging
import json
import time
import pdb
import uuid

# 导入 TsmConfig 全局配置实例
from .tsm_config import tsm_config
from .tsm_input import TsmInput
from .utils import requests
from tsmrt import tsm_input

logger = logging.getLogger(__name__)

class Agent():
    def __init__(self, model="DFM3-Pro", sys_prompt= "", url= ""):
        self.model = model
        self.sys_prompt = sys_prompt
        self.url = url
        self.instances = []
        # 模型列表配置
        self.doubao_models = ["doubao-seed-1-6-251015", "deepseek-v3-250324"]
        self.glm_models = ["glm-4.7"]
        self.qwen_models = ["qwen3-max", "qwen3-235b-a22b-instruct-2507", "qwen3-30b-a3b-instruct-2507", "qwen3.5-35b-a3b", "qwen3.5-27b", "Qwen3-4B"]
        self.deepseek_v4_models = ["deepseek-v4-flash", "deepseek-v4-pro"]
        self.tsm_input = None

    async def request_model(self, tools=[], messages=[]):
        MODEL = self.model
        # 豆包模型
        if MODEL in self.doubao_models:
            BASE_URL = tsm_config.doubao.url
            COMPLETIONS_PATH = ""
            URL = BASE_URL + COMPLETIONS_PATH
            API_KEY = tsm_config.doubao.api_key
        # glm模型
        elif MODEL == "glm-4.7":
            BASE_URL = tsm_config.glm.url
            COMPLETIONS_PATH = ""
            URL = BASE_URL + COMPLETIONS_PATH
            API_KEY = tsm_config.glm.api_key
        # qwen模型
        elif MODEL in self.qwen_models:
            BASE_URL = tsm_config.qwen.url
            COMPLETIONS_PATH = ""
            URL = BASE_URL + COMPLETIONS_PATH
            API_KEY = tsm_config.qwen.api_key
        # deepseek-v4模型
        elif MODEL in self.deepseek_v4_models:
            BASE_URL = tsm_config.deepseek_v4.url
            COMPLETIONS_PATH = ""
            URL = BASE_URL + COMPLETIONS_PATH
            API_KEY = tsm_config.deepseek_v4.api_key
        else :
            BASE_URL = tsm_config.dfm.url
            API_KEY = tsm_config.dfm.api_key
            COMPLETIONS_PATH = "/chat/completions"
            URL = BASE_URL + COMPLETIONS_PATH

        # ========== 构造请求体（与 Bash 的 heredoc 完全一致） ==========
        if MODEL in self.doubao_models or MODEL in self.deepseek_v4_models:
            PAYLOAD = {
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "max_tokens": 512,
                "temperature": 0.2,
                "repetitionPenalty": 1.2,
                "top_k": 1,
                "top_p": 0.95,
                "thinking": {"type":"disabled"}
                }
        else:
            PAYLOAD = {
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "max_tokens": 512,
                "temperature": 0,        # 1. 温度=0：完全关闭随机性
                "top_k": 1,              # 2. 只选概率最高的1个词（贪心采样）
                # "top_p": 1,              # 3. 关闭核采样，必须设为1
                # "repetitionPenalty": 1.0, # 4. 关闭重复惩罚（避免影响结果）
                "enable_thinking": False,
                "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": False},
                }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        if "dfm3" in MODEL or "DFM" in MODEL:
            URL += "?modelName=" + MODEL
            if self.tsm_input is None:
                URL += "&productId=" + tsm_config.dfm.product_id
                URL += "&apikey=" + tsm_config.dfm.api_key
                URL += "&requestId=" + str(uuid.uuid4())
            else :
                if "https://dfm.duiopen.com" in URL :
                    URL += "&productId=" + tsm_config.dfm.product_id
                    URL += "&apikey=" + tsm_config.dfm.api_key
                    URL += "&requestId=" + self.tsm_input.request.requestId
                    URL += "&recordId=" + self.tsm_input.request.recordId
                else :
                    URL += "&productId=" + self.tsm_input.context.product.productId
                    URL += "&requestId=" + self.tsm_input.request.requestId
                    URL += "&recordId=" + self.tsm_input.request.recordId
        try:
            logger.info(f"function call request url : {URL}")
            #logger.info(f"function call request body : {json.dumps(PAYLOAD, ensure_ascii=False)}")
            # print(messages)
            resp = await requests.post(URL, headers=headers, json=PAYLOAD, timeout=30)
            resp.raise_for_status()
            logger.info(f"function call resp : {resp.text}")
            c_res = resp.json()["choices"][0]["message"]
        except Exception as e:
            logger.error(f"function call error:{e}")
            c_res = "null"
        return c_res

    def update_prompt(self, prompt):
        self.sys_prompt = prompt

class Llm_Function_Call_Agent(Agent):
    # 物模型 function call Agent
    def __init__(self, model="DFM3-Pro", sys_prompt= "", url= ""):
        super().__init__(model, sys_prompt, url)
        c_prompt = """
    "你是车载工具调用助手。根据输入内容输出一个或多个工具调用。"
    "若无需调用工具，固定输出：<no_tool>。禁止闲聊。"
    "当输入为视频/图像生成、天气查询、车辆说明书及功能查询、POI识别或涉及色情暴力等敏感内容时，固定输出：<no_tool>。禁止闲聊。"
"""

        self.sys_prompt = c_prompt
        self.tools = []
        self.messages = []
        self.llm_responses = []

    def update_input(self, tsm_input:TsmInput):
        self.tsm_input = tsm_input

    def update_functions(self, instances, l2_result):
        self.instances = instances
        self.tools = []
        l2_instance_names = list(l2_result.keys())

        for instance in self.instances:
            all_functions = instance["all_functions"]
            all_functions_dict = {it["name"]:it for it in all_functions}
            # 如果实例不在l2结果中，直接跳过
            if instance["_description"] not in l2_instance_names:
                continue

            for c_function_name in l2_result[instance["_description"]]:
                if c_function_name not in all_functions_dict.keys():
                    continue
                
                c_function = all_functions_dict[c_function_name]
                c_tool = {}
                c_tool["type"] = "function"
                c_tool["function"] = {}
                # c_tool["function"]["name"] = instance["_description"] + "__func__" + c_function["name"]
                c_tool["function"]["name"] = c_function["name"]
                c_tool["function"]["description"] = c_function["description"]
                c_tool["function"]["parameters"] = {}
                c_tool["function"]["parameters"]["type"] = "object"
                c_tool["function"]["parameters"]["properties"] = {}
                c_tool["function"]["parameters"]["required"] = []
                if c_function["input_param"] != "None":
                    for input_param_name in c_function["input_param"].keys():
                        c_tool["function"]["parameters"]["properties"][input_param_name] = {}
                        if c_function["input_param"][input_param_name]["define"]["type"] == "float":
                            # 适配sglang/deepseek_v4 不支持float类型，统一转换为number
                            if self.model in self.deepseek_v4_models:
                                c_tool["function"]["parameters"]["properties"][input_param_name]["type"] = "number"
                        # c_tool["function"]["parameters"]["properties"][input_param_name]["type"] = c_function["input_param"][input_param_name]["define"]["type"]
                        else:
                            c_tool["function"]["parameters"]["properties"][input_param_name]["type"] = c_function["input_param"][input_param_name]["define"]["type"]
                        c_tool["function"]["parameters"]["properties"][input_param_name]["description"] = c_function["input_param"][input_param_name]["description"]
                        if "choice" in c_function["input_param"][input_param_name]["define"].keys():
                            if c_function["input_param"][input_param_name]["define"]["choice"] != []:
                                c_tool["function"]["parameters"]["properties"][input_param_name]["enum"] = c_function["input_param"][input_param_name]["define"]["choice"]
                        # c_tool["function"]["parameters"]["required"].append(input_param_name)
                self.tools.append(c_tool)
        # print(json.dumps(self.tools, ensure_ascii=False, indent=4))
        return self.tools

    async def request_session(self, text_list):
        self.messages = []
        text_str = []
        # 限制在最近5轮对话
        if len(text_list) > 6:
            text_list = text_list[-6:]
        if len(text_list) > 1:
            for i in range(len(text_list)):
                if i == (len(text_list) -1 ):
                    text_str.append(f"当前输入:{text_list[i][0]}")
                else:
                    text_str.append(f"历史输入:{text_list[i][0]}\n历史输出:{text_list[i][1]}")
        else:
            text_str.append(f"{text_list[0][0]}")
        self.messages.append({"role": "system", "content": self.sys_prompt})
        self.messages.append({"role": "user", "content": "\n".join(text_str)})
        modify_funcs = []
        query_funcs = []
        dm_input_messages = []
        self.llm_responses = []

        while True:
            c_res = await self.request_model(self.tools, self.messages)
            if c_res == "null":
                c_llm_nlg = c_res
            else:
                content = c_res.get("content") or ""
                c_llm_nlg = content.strip() if content.strip() else "null"
            self.llm_responses.append(c_res)
            self.messages.append(c_res)
            # pdb.set_trace()
            if c_res == "null" or "tool_calls" not in c_res.keys() or len(c_res["tool_calls"]) == 0:
                # 不存在工具调用，直接结束
                dm_input_messages.append(c_llm_nlg)
                break
            dm_input_messages.append(c_llm_nlg)
            # print(c_llm_nlg)
            input_params_list = []
            for i in range(len(c_res["tool_calls"])):
                cur_func = c_res["tool_calls"][i]
                input_params = json.loads(cur_func["function"]["arguments"])
                func_name = cur_func["function"]["name"]
                instance_name = func_name.split("__func__")[0]
                func_name = func_name.split("__func__")[-1]

                # 保存调用的function
                obs = '{"status": "success"}'
                # modify_funcs.append({"instance_name":instance_name, "function_name": func_name, "input_params": input_params})
                ### JS 临时修改 "queryAliasAddress" 转为deep
                if func_name == "queryAliasAddress":
                    func_name = "mapDeep"
                    input_params = {}
                ### JS 临时修改 "queryAliasAddress" 转为deep


                modify_funcs.append({"function_name": func_name, "input_params": input_params})
                input_params_list.append(str(input_params))
            # '''
                # print("Function Name: " + func_name + "\nInput Param:" + str(input_params))
                # print("Observation:", obs)
            # '''
                dm_input_messages.append("Action: " + func_name + "\nInput:" + input_params_list[i])
                dm_input_messages.append("Observation: " + obs)
                self.messages.append({"role": "tool", "tool_call_id": c_res["tool_calls"][0]["id"], "content": obs})
            # 不采用react模式，直接结束
            break
        # 执行修改函数
        return modify_funcs,query_funcs, self.instances, dm_input_messages




