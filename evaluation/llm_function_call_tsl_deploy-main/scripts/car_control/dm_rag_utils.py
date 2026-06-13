# 单独调用rag模块获取funtion结果

import sys
import os
import asyncio
import time
import logging
import uuid
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tsmrt.manager as dialog_manager
from tsmrt.tsm_input import TsmInput, Session, DialogHistory, DialogOutput, SpeakOutput, Functions, FunctionRequest, RequestBody
from tsmrt.tsm_output import TsmOutput
import json
import pdb

c_instance = {
        "_description": "智能座舱",
        "all_functions": [json.loads(it.strip()) for it in open("../../function_call_data/car_control_func.jsonl", "r").readlines()] + 
                        [json.loads(it.strip()) for it in open("../../function_call_data/map_func.jsonl", "r").readlines()] +
                        [json.loads(it.strip()) for it in open("../../function_call_data/multimedia_func.jsonl", "r").readlines()]
    }
c_instances = [c_instance]
test_dm = dialog_manager.Dialog_Manager(c_instances)
test_dm.used_rag_search = True
rag_topk = 10

# l2_classify_input 格式：[["query1", "nlg1"], ["query2", "nlg2"], ..., ["queryN", ""]]
# 获取 queryN需要的functions



## 单轮示例
request_domains = []
l2_classify_input = [["退出地图", ""]]
rag_result = asyncio.run(test_dm.l2_classifier.rag_search_function(request_domains, l2_classify_input, top_k=rag_topk))
print(rag_result)



'''
## 多轮示例，需要注意，历史轮次的function需要手动加入
request_domains = []
l2_classify_input = [["打开车窗", "好的,已为您打开"], ["查询当前位置", ""]]
rag_result = asyncio.run(test_dm.l2_classifier.rag_search_function(request_domains, l2_classify_input, top_k=rag_topk))
print(rag_result)
'''
