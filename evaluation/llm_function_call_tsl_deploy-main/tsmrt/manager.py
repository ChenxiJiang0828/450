import logging
import os
from . import classify
from . import function_call
from .utils import utils
import json
import copy
import time
from typing import List, Dict, Any

from .tsm_input import TsmInput
from .tsm_output import TsmOutput
from .tsm_config import tsm_config
from .tsm_post import TsmPost

logger = logging.getLogger(__name__)

class Dialog_Manager:
    def __init__(self, instances=None, c_info={}, description="鏅鸿兘搴ц埍"):
        self.l2_classifier = None
        self.function_call_agent = None
        self.tsm_post = None
        self.session_messages = []
        self.used_rag_search = tsm_config.res.func_rag_config["use_rag"]
        self.rag_topk = tsm_config.res.func_rag_config["topk"]
        self.c_info = c_info

        self.initialize_instances(instances, description)
        self.initialize_classifier()
        self.initialize_agent()
        self.initialize_post()

        logger.info("Dialog_Manager init Done")

    def initialize_instances(self, instances, description="鏅鸿兘搴ц埍"):
        """鍒濆鍖栧疄渚嬪垪琛?

        濡傛灉澶栭儴浼犲叆 instances锛屽垯浣跨敤澶栭儴鐨勫垵濮嬪寲 self.instances
        鍚﹀垯浠?tsm_config 鏋勫缓瀹炰緥鍒楄〃

        Args:
            instances: 澶栭儴浼犲叆鐨勫疄渚嬪垪琛紝濡傛灉涓?None 鍒欎粠 tsm_config 鏋勫缓
            description: 瀹炰緥鎻忚堪锛岄粯璁や负"鏅鸿兘搴ц埍"

        Structure:
        c_instance 鏂板缁撴瀯锛屼緵tsm_output杈撳嚭domain淇℃伅锛岀ず渚嬪涓?
            "all_functions_define": {
                    "CarControl": [
                        "setTemperatureDirect"
                    ],
                    "Map": [
                        "controlHighwayPilotAssistance"
                    ]
            }
        """
        if instances is not None:
            self.instances = instances
        else:
            c_instance = {}
            c_instance["_description"] = description
            all_functions = []
            all_functions_define = {}
            for domain, config in tsm_config.res.func_def_domains.items():
                file_path = f"{tsm_config.res.func_def_res}/{config['fn']}"
                with open(file_path, "r") as f:
                    domain_functions = [json.loads(line.strip()) for line in f.readlines()]
                    all_functions.extend(domain_functions)
                    all_functions_define[domain] = [func["name"] for func in domain_functions]
            c_instance["all_functions"] = all_functions
            c_instance["all_functions_define"] = all_functions_define
            self.instances = [c_instance]

    def initialize_classifier(self):
        l2_url = os.getenv("TSMRT_L2_CLASSIFIER_PATH", "/chat/completions")
        self.l2_classifier = classify.Level_2_Classifier(l2_url)
        self.l2_classifier.update_instances(self.instances)
        self.l2_classifier.model="DFM3-Pro"

    def initialize_agent(self):
        self.function_call_agent = function_call.Llm_Function_Call_Agent()
        self.function_call_agent.model="dfm3_tsl_built_in_1.0"

    def initialize_post(self):
        self.tsm_post = TsmPost()

    def refresh_message(self):
        self.session_messages = []

    async def in_session_chat(self, current_user_input):
        current_message = {"query": current_user_input}
        # c_info淇℃伅鍔犲叆
        if self.c_info is not None:
            hit_c_info = {}
            for key in self.c_info.keys():
                if key in current_user_input:
                    hit_c_info[key] = self.c_info[key]
            for key in hit_c_info.keys():
                current_user_input = current_user_input + " (棰濆淇℃伅:\'" + key + "\'鍙兘涓? + hit_c_info[key] + ")"


        start_time = time.time()
        l2_classify_input = []
        # 榛樿閰嶇疆鍘嗗彶杞鐨刵lg涓?濂界殑"
        defult_nlg = "濂界殑"
        for item_message in self.session_messages:
            l2_classify_input.append([item_message["query"], defult_nlg])
        l2_classify_input.append([current_user_input, ""])
        print("="*80 + "\nLevel 2 Classifier Input:\n" + json.dumps(l2_classify_input, ensure_ascii=False))
        start_time = time.time()
        # 濡傛灉rag鎼滅储寮€鍚紝浣跨敤rag鎼滅储
        if self.used_rag_search:
            l2_result, vocab_hit_result, modify_l2_classify_input = await self.l2_classifier.rag_search_function(None, l2_classify_input, top_k=20)
            if vocab_hit_result["hitFlag"]:
                l2_classify_input = modify_l2_classify_input
            l2_result = json.dumps(l2_result, ensure_ascii=False)
        else:
            l2_result_dict = await self.l2_classifier.classify(l2_classify_input)
            ## 鎵撳嵃浜岀骇鍒嗙被缁撴灉

            l2_result = json.dumps(l2_result_dict, ensure_ascii=False)
        ##his_func
        print("="*80 + "\nLevel 2 Classifier Result:\n" + l2_result)
        current_message["l2_result"] = l2_result
        end_time = time.time()
        print(f"Level 2 Classifier Time Cost: {end_time - start_time}")

        start_time = time.time()
        if l2_result == '{}':
            # 濡傛灉浜岀骇鍒嗙被娌℃湁缁撴灉锛屾嫆璇嗗鐞嗭紝缁撴潫
            current_message["nlg"] = ""
            self.session_messages.append(copy.deepcopy(current_message))

            return current_message
        else:
            if utils.is_valid_json(l2_result):
                # 浜岀骇鍒嗙被鏈夌粨鏋滐紝杩涜agent璋冪敤
                l2_result = json.loads(l2_result)
                self.function_call_agent.update_functions(self.instances, l2_result)
                print("="*80 + "\nLLM Function Call:")
                modify_functions, query_funcs, self.instances, c_messages = await self.function_call_agent.request_session(l2_classify_input)
                current_message["call_functions"] = {"modify_functions": modify_functions, "query_funcs": query_funcs}


        self.session_messages.append(copy.deepcopy(current_message))
        end_time = time.time()
        print(f"LLM Function Call Time Cost: {end_time - start_time}")

        return current_message

    async def chat_entry(self, current_user_input):
        start_time = time.time()
        current_message = await self.in_session_chat(current_user_input)
        end_time = time.time()
        current_message["systime"] = end_time - start_time
        print("chat_entry current_message : ", json.dumps(current_message, ensure_ascii=False))
        return current_message

    def get_request_domains(self, tsm_input: TsmInput) -> List[str]:
        domains = []

        if tsm_config.app.uri == "ctsm":
            if tsm_input.context and tsm_input.context.skills:
                for skill in tsm_input.context.skills:
                    if skill.useTsm and not skill.tsm.domain and skill.task in tsm_config.res.func_def_domains:
                        domains.append(skill.task)
        else :
            domains = list(tsm_config.res.func_def_domains.keys())
        return domains

    async def chat_process(self, tsm_input: TsmInput) -> TsmOutput:
        """澶勭悊 TsmInput 骞惰繑鍥?TsmOutput

        Args:
            tsm_input: TsmInput 瀵硅薄锛屽寘鍚?request銆乻ession 鍜?context 淇℃伅

        Returns:
            TsmOutput: 鏍囧噯鍖栫殑杈撳嚭瀵硅薄
        """
        start_time = time.time()
        l2_classify_input = []
        dialog_history_list = []
        dlg_function = ""
        request_domains = self.get_request_domains(tsm_input)
        logger.info(f"function classify request domains: {request_domains}")

        if tsm_input.session and tsm_input.session.dialogHistory:
            dialog_history_list = tsm_input.session.dialogHistory

        for dialog_history in dialog_history_list:
            dlg_input = dialog_history.input
            dlg_nlg = ""
            dlg_function = ""
            if dialog_history.output:
                if dialog_history.output.widget and dialog_history.output.widget.displayText:
                    dlg_nlg = dialog_history.output.widget.displayText
                elif dialog_history.output.speak and dialog_history.output.speak.text:
                    dlg_nlg = dialog_history.output.speak.text

            if dialog_history.functions and dialog_history.dataSource == tsm_config.app.uri:
                dlg_function = dialog_history.functions.request.name
            l2_classify_input.append([dlg_input, dlg_nlg, dlg_function])

        # 浠?TsmInput 涓彁鍙栧綋鍓嶇敤鎴疯緭鍏?
        current_user_input = tsm_input.request.input
        l2_classify_input.append([current_user_input, "", ""])
        logger.info(f"function classify input : {json.dumps(l2_classify_input, ensure_ascii=False)}")

        current_message = {"query": current_user_input, "model": self.function_call_agent.model}

        hot_start_time = time.time()
        hotfix_result = await self.l2_classifier.rag_search_hotfix(request_domains, current_user_input, dlg_function)
        hot_end_time = time.time()
        current_message["hottime"] = hot_end_time - hot_start_time
        if hotfix_result:
            # 鑻ュ懡涓嫆璇嗙殑hotfix锛岀洿鎺ヨ繑鍥炴嫆璇嗙粨鏋?
            if hotfix_result["modify_functions"][0]["function_name"] == "":
                current_message["call_functions"] = {
                    "modify_functions": [],
                    "query_funcs": []
                }
            # 鑻ュ懡涓叾浠杊otfix锛屾甯稿鐞?
            else:
                current_message["call_functions"] = {
                    "modify_functions": hotfix_result["modify_functions"],
                    "query_funcs": hotfix_result["query_funcs"]
                }
            current_message["nlg"] = "宸叉墽琛屽懡浠?
            current_message["hotfix_used"] = True

            self.refresh_message()

            end_time = time.time()
            current_message["systime"] = end_time - start_time

            tsm_output = TsmOutput.from_current_message(
                current_message=current_message,
                tsm_input=tsm_input,
                tsm_post=self.tsm_post,
                instances=self.instances,
                request_domains=request_domains
            )

            return tsm_output

        logger.info("No hotfix match found, using normal processing flow")

        # 璋冪敤浜岀骇鍒嗙被鍣?
        classify_start_time = time.time()
        if self.used_rag_search:
            l2_result, vocab_hit_result, modify_l2_classify_input = await self.l2_classifier.rag_search_function(request_domains, l2_classify_input, top_k=self.rag_topk)
            if vocab_hit_result["hitFlag"]:
                l2_classify_input = modify_l2_classify_input
            l2_result = json.dumps(l2_result, ensure_ascii=False)
        else:
            l2_result = await self.l2_classifier.classify(l2_classify_input)
        classify_end_time = time.time()
        logger.info(f"function classify resp : {l2_result}")
        logger.info(f"num of classify functions : {len(list(json.loads(l2_result).values())[0])}")

        current_message["l2_result"] = l2_result
        current_message["clstime"] = classify_end_time - classify_start_time

        # 澶勭悊鍒嗙被缁撴灉
        if l2_result == '{}':
            # 濡傛灉浜岀骇鍒嗙被娌℃湁缁撴灉锛屾嫆璇嗗鐞嗭紝缁撴潫
            current_message["nlg"] = ""
        else:
            if utils.is_valid_json(l2_result):
                # 浜岀骇鍒嗙被鏈夌粨鏋滐紝杩涜agent璋冪敤
                l2_result = json.loads(l2_result)
                self.function_call_agent.update_functions(self.instances, l2_result)
                self.function_call_agent.update_input(tsm_input)
                #print("="*80 + "\nLLM Function Call:")
                agent_start_time = time.time()
                modify_functions, query_funcs, self.instances, c_messages = await self.function_call_agent.request_session(l2_classify_input)
                current_message["call_functions"] = {"modify_functions": modify_functions, "query_funcs": query_funcs}
                agent_end_time = time.time()
                current_message["fuctime"] = agent_end_time - agent_start_time

        self.refresh_message()

        end_time = time.time()
        current_message["systime"] = end_time - start_time
        # 浣跨敤 TsmOutput.from_current_message 鏋勫缓 TsmOutput 瀵硅薄
        tsm_output = TsmOutput.from_current_message(
            current_message=current_message,
            tsm_input=tsm_input,
            tsm_post=self.tsm_post,
            instances=self.instances,
            request_domains=request_domains
        )

        return tsm_output



