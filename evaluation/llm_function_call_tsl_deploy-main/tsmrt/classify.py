import re
from typing import Any, List, Dict, Optional
from pathlib import Path
import asyncio
import json
import logging

from .tsm_config import tsm_config
from .tsm_rag import TsmRag, TsmRags
from .utils import requests


logger = logging.getLogger(__name__)

class Classifier:
    # 分类器 抽象类
    def __init__(self, model_url):
        self.model_url = model_url
        self.prompt = ""
        

    async def request_model(self, text):
        """请求模型"""
        response = await requests.post(self.model_url, json={"text": text})
        return response.json()

    async def classify(self, text):
        """分类文本"""
        domain_result = await self.request_model(text)
        # 分类文本
        return domain_result

class Level_2_Classifier(Classifier):
    def __init__(self, model_url):
        super().__init__(model_url)
        self.instances = []
        self.model = "DFM3-Pro"
        self.ins = ""
        # 模型列表配置
        self.doubao_models = ["doubao-seed-1-6-251015", "deepseek-v3-250324"]
        self.glm_models = ["glm-4.7"]
        self.qwen_models = ["qwen3-max", "qwen3-235b-a22b-instruct-2507", "qwen3-30b-a3b-instruct-2507", "qwen3.5-35b-a3b", "qwen3.5-27b", "Qwen3-4B"]

        # 使用 TsmRag 处理向量相关功能
        self.tsm_rags = TsmRags()
        # 词库相关
        self.tsm_res = tsm_config.res
        self.vocabs = {}
        for k, v in self.tsm_res.vocab_res_path.items():
            with open(v, "r", encoding="utf-8") as f:
                self.vocabs[k] = set(f.read().splitlines())

    async def request_model(self, text):
        MODEL = self.model

        if MODEL in self.doubao_models:
            BASE_URL = tsm_config.doubao.url
            API_KEY = tsm_config.doubao.api_key
            URL = BASE_URL
        elif MODEL in self.glm_models:
            BASE_URL = tsm_config.glm.url
            API_KEY = tsm_config.glm.api_key
            URL = BASE_URL
        elif MODEL in self.qwen_models:
            BASE_URL = tsm_config.qwen.url
            API_KEY = tsm_config.qwen.api_key
            URL = BASE_URL
        else:
            # DFM 模型（默认）
            BASE_URL = tsm_config.dfm.url
            API_KEY = tsm_config.dfm.api_key
            # DFM 模型需要拼接 modelName 参数
            COMPLETIONS_PATH = self.model_url
            URL = BASE_URL + COMPLETIONS_PATH + f"&modelName={MODEL}"

        # ========== 构造请求体（与 Bash 的 heredoc 完全一致） ==========
        c_ins = text
        if MODEL in self.doubao_models:
            PAYLOAD = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": c_ins
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.2,
                "repetitionPenalty": 1.2,
                "topK": 40,
                "topP": 0.95,
                "thinking": {"type":"disabled"}
            }
        else:
            PAYLOAD = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": c_ins
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.2,
                "repetitionPenalty": 1.2,
                "topK": 40,
                "topP": 0.95,
                "enable_thinking": False,
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        try:
            resp = await requests.post(URL, headers=headers, json=PAYLOAD, timeout=20)
            #print(resp.text)
            resp.raise_for_status()          # 非 2xx 会抛异常
            # print("\n===== 原始响应 =====")
            # print(c_ins)

            c_classify = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print("请求失败:", e)
            c_classify = "{}"
        return c_classify

    async def classify(self, text_list):
        text_str = []
        if len(text_list) > 1:
            for i in range(len(text_list)):
                if i == (len(text_list) -1 ):
                    text_str.append(f"当前输入:{text_list[i][0]}\n当前输出:{text_list[i][1]}")
                else:
                    text_str.append(f"历史输入:{text_list[i][0]}\n历史输出:{text_list[i][1]}")
        else:
            text_str.append(f"当前输入:{text_list[0][0]}\n当前输出:{text_list[0][1]}")


        self.ins = self.prompt + "\n" + "\n".join(text_str)
        c_classify = await self.request_model(self.ins)
        c_classify = c_classify.strip("```json").strip("```").strip()

        if len(text_list) > 1 and c_classify != '{}':
            c_classify_dict = json.loads(c_classify)
            c_classify_dict = self.append_history_func(text_list, c_classify_dict)
            c_classify = json.dumps(c_classify_dict, ensure_ascii=False)
        return c_classify

    def update_prompt(self, prompt):
        """更新指令"""
        self.prompt = prompt
    def update_instances(self, instances):
        """更新实例"""
        self.instances = instances
        self.tsm_rags.update_instances(instances)
        l1_prompt = []
        l1_prompt.append("你是一个智能助手，你的任务是根据用户的指令和上下文以及提供的实例的功能列表，选取涉及到的实例的功能。")
        l1_prompt.append("你只能根据实例的功能列表中的功能进行选择，不能编造新的功能。")
        l1_prompt.append("每个实例都具有状态的修改和查询功能，你可以同时选择状态查询和修改，后续模块会通过状态查询获取结果后进行分析再进行状态修改。")
        l1_prompt.append("通常情况下，如果用户的指令清晰且明确，直接选取相关的若干实例的状态修改功能即可，否则你需要同时选取相关的若干实例的状态查询和状态修改功能。")
        l1_prompt.append("对于需要隐式推理的情况，你可以尽可能多的返回实例的多种状态的查询和修改功能.")
        l1_prompt.append("如果在提供的实例中无法根据用户的输入确定相关的实例，你需要返回空对象{}。")
        l1_prompt.append("示例1:")
        l1_prompt.append("实例功能列表：")
        l1_prompt.append('{"instance_name":"卧室空调", "functions":[{"function_name":"query_temperature", "function_description":"查询温度"}, {"function_name":"modify_temperature", "function_description":"修改温度"}]},')
        l1_prompt.append('{"instance_name":"客厅空调", "functions":[{"function_name":"query_temperature", "function_description":"查询温度"}, {"function_name":"modify_temperature", "function_description":"修改温度"}]},')
        l1_prompt.append("当前输入:空调太热了")
        l1_prompt.append('你输出:{"卧室空调":["query_temperature", "modify_temperature"], "客厅空调":["query_temperature", "modify_temperature"]}')
        l1_prompt.append("\n")
        l1_prompt.append("示例2:")
        l1_prompt.append("实例功能列表：")
        l1_prompt.append('{"instance_name":"卧室空调", "functions":[{"function_name":"query_temperature", "function_description":"查询温度"}, {"function_name":"modify_temperature", "function_description":"修改温度"}]},')
        l1_prompt.append('{"instance_name":"客厅空调", "functions":[{"function_name":"query_temperature", "function_description":"查询温度"}, {"function_name":"modify_temperature", "function_description":"修改温度"}]},')
        l1_prompt.append("当前输入:打开窗户")
        l1_prompt.append('你输出:{}')
        l1_prompt.append("\n")
        # l1_prompt.append("请根据用户指令和上下文，判断用户的意图。")
        l1_prompt.append("以下为本次请求提供的实例")
        for it in self.instances:
            instance_description = it["_description"]
            c_instance_json = {"instance_name": instance_description, "functions": []}
            for f in it["all_functions"]:
                c_instance_json["functions"].append({"function_name": f["name"], "function_description": f["description"]})
            l1_prompt.append(json.dumps(c_instance_json, ensure_ascii=False))
        l1_prompt.append("\n")
        l1_prompt.append("请根据用户指令和上下文，判断用户的意图。")
        l1_prompt.append("请注意，你只需要输出json即可，不要有任何其他的解释。")
        l1_prompt.append("以下为本次请求的上下文和用户指令：")
        self.update_prompt("\n".join(l1_prompt))

    def _format_historical_context(self, text_list: List[List[str]]) -> str:
        """
        格式化历史上下文

        Args:
            text_list: 用户输入列表，格式为[[输入文本, 输出文本, dlg_function]]

        Returns:
            格式化后的历史上下文字符串
        """
        if len(text_list) <= 1:
            return ""

        text_str = []
        # 只取除了最后一个之外的所有条目作为历史上下文
        for i in range(len(text_list) - 1):
            text_str.append(f"历史输入:{text_list[i][0]}\n历史输出:{text_list[i][1]}")

        return "\n".join(text_str)

    def _format_historical_and_current_input(self, text_list: List[List[str]]) -> str:
        """
        格式化历史上下文与当前输入合并

        Args:
            text_list: 用户输入列表，格式为[[输入文本, 输出文本, dlg_function]]

        Returns:
            格式化后的历史上下文与当前输入合并后的字符串
        """
        if len(text_list) <= 1:
            return ""
        text_str = []
        for i in range(len(text_list) - 1):
            text_str.append(f"历史输入:{text_list[i][0]}\n历史输出:{text_list[i][1]}")
        # 合并当前输入
        text_str.append(f"当前输入:{text_list[-1][0]}")

        return "\n".join(text_str)

    def _format_current_input(self, text_list: List[List[str]]) -> str:
        """
        格式化当前输入（不带"当前输入"字符串）

        Args:
            text_list: 用户输入列表，格式为[[输入文本, 输出文本, dlg_function]]

        Returns:
            当前输入文本
        """
        if not text_list:
            return ""

        # 只取最后一个条目的输入部分
        return text_list[-1][0]

    def append_history_func(self, text_list: List[List[str]], result: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        将历史轮次的function加入结果中，避免重复

        Args:
            text_list: 用户输入列表，格式为[[输入文本, 输出文本, dlg_function]]
            result: RAG搜索结果字典，键为实例名，值为功能名列表

        Returns:
            合并了历史function的结果字典
        """
        history_functions = []
        if not text_list or len(text_list) <= 1:
            return result
        else:
            for i in range(len(text_list) - 1):
                if len(text_list[i]) > 2 and text_list[i][2]:
                    history_functions.append(text_list[i][2])
        if not history_functions:
            return result

        subclassify_functions = self._get_subclassify_functions(history_functions)
        history_functions = list(dict.fromkeys(history_functions + subclassify_functions))
        for instance, funcs in result.items():
            merged_funcs = funcs + history_functions
            result[instance] = list(dict.fromkeys(merged_funcs))
        return result

    def _get_subclassify_functions(self, history_functions: List[str]) -> List[str]:
        subclassify_data = tsm_config.res.func_cluster_subclassify
        if not subclassify_data:
            return []

        result = []
        #只寻找上一轮的function的subclassify
        for func_name in history_functions[-1:]:
            if len(result) > 0:
                break
            for domain, subclasses in subclassify_data.items():
                if len(result) > 0:
                    break
                for subclass_name, functions in subclasses.items():
                    if func_name in functions:
                        result.extend(functions)
                        break
        return list(set(result) - set(history_functions))

    def _merge_rag_results(self, current_result: Dict[str, List[str]],
                           historical_result: Dict[str, List[str]],
                           historical_and_current_input_result: Dict[str, List[str]],
                           text_list: List[List[str]]) -> Dict[str, List[str]]:
        """
        合并三路RAG搜索结果：当前输入、历史上下文、历史上下文与当前输入合并

        Args:
            current_result: 当前输入的RAG搜索结果
            historical_result: 历史上下文的RAG搜索结果
            historical_and_current_input_result: 历史上下文与当前输入合并的RAG搜索结果
            text_list: 用户输入列表，格式为[[输入文本, 输出文本, dlg_function]]

        Returns:
            合并后的最终结果字典，键为实例名，值为功能名列表
        """
        final_result = {}

        # 先合并当前输入的结果
        for instance, funcs in current_result.items():
            final_result[instance] = funcs

        # 再合并历史上下文的结果
        for instance, funcs in historical_result.items():
            if instance not in final_result:
                final_result[instance] = funcs
            else:
                # 合并不同的功能名
                final_result[instance].extend(funcs)
                # 去重, 保留原始顺序
                final_result[instance] = list(dict.fromkeys(final_result[instance]))

        # 再合并历史上下文与当前输入的结果
        for instance, funcs in historical_and_current_input_result.items():
            if instance not in final_result:
                final_result[instance] = funcs
            else:
                # 合并不同的功能名
                final_result[instance].extend(funcs)
                # 去重, 保留原始顺序
                final_result[instance] = list(dict.fromkeys(final_result[instance]))

        # 将历史轮次的function加入final_result
        final_result = self.append_history_func(text_list, final_result)

        return final_result
    
    def check_vocab_hit(self, text_list: List[List[str]]) -> bool:
        """
        检查是否有vocab_data中的vocab_xxx.txt文件中的词汇命中

        Args:
            text_list: 用户输入列表，格式为[[输入文本, 输出文本, dlg_function]]

        Returns:
            {"hitFlag": True/False, "hitvocab":"/xxx", "hittype":"video/song"}
        """
        # 先检查歌曲词库
        for vocab in self.vocabs["song"]:
            if vocab in text_list[-1][0]:
                return {"hitFlag": True, "hitvocab": vocab, "hittype": "song"}
        # 再检查视频词库
        for vocab in self.vocabs["video"]:
            if vocab in text_list[-1][0]:
                return {"hitFlag": True, "hitvocab": vocab, "hittype": "video"}
        
        
        return {"hitFlag": False, "hitvocab":"", "hittype":""}

    async def rag_search_function(self, domain_list: List[str], text_list: List[List[str]], top_k: int = 5) -> Dict[str, List[str]]:
        rag_search_result = await self.rag_search_function_module(domain_list, text_list, top_k)
        # 若rag_search_result召回的前3个function包含multimedia领域下的fucntion，进行vocab检查
        vocab_hit_result = {"hitFlag": False, "hitvocab":"", "hittype":""}
        for instance in rag_search_result.items():
            for func in instance[1][0:3]:
                # 检查功能是否属于multimedia领域
                if self.tsm_rags.search_domain_by_function(func) == "Multimedia":
                    # 执行词汇表命中检查
                    vocab_hit_result = self.check_vocab_hit(text_list)
                    if vocab_hit_result["hitFlag"]:
                        # 若有命中，修改输入并重新rag一次
                        hit_vocab = vocab_hit_result["hitvocab"]
                        if vocab_hit_result["hittype"] == "song":
                            text_list[-1][0] = text_list[-1][0].replace(hit_vocab, hit_vocab+"(歌曲名)")
                        elif vocab_hit_result["hittype"] == "video":
                            text_list[-1][0] = text_list[-1][0].replace(hit_vocab, hit_vocab+"(影视名)")
                        # 重新rag一次
                        rag_search_result = await self.rag_search_function_module(domain_list, text_list, top_k)
                        break
                        
        return rag_search_result, vocab_hit_result, text_list


    async def rag_search_function_module(self, domain_list: List[str], text_list: List[List[str]], top_k: int = 5) -> Dict[str, List[str]]:
        """
        单一模块
        使用RAG搜索功能替代传统的classify方法
        通过余弦相似度匹配用户输入与向量库中的功能查询

        Args:
            domain_list: 领域列表，格式为[领域1, 领域2, ...]
            text_list: 用户输入列表，格式为[[输入文本, 输出文本, dlg_function]]
            top_k: 返回前k个最相似的结果

        Returns:
            符合原有classify方法输出格式的字典，键为实例名，值为功能名列表
        """
        domain_list = self.tsm_rags.get_available_domains(domain_list)
        if not domain_list:
            logger.error("function classify rag_search_function input domains is empty")
            return {}
        else:
            logger.info(f"function classify rag_search_function input domains: {domain_list}")

        # 检查是否为多轮输入
        if len(text_list) > 1:
            # 限制在最近三轮对话
            if len(text_list) > 3:
                text_list = text_list[-3:]
            # 多轮输入情况：两路RAG召回

            # 1. 历史上下文单独RAG一次
            historical_context = self._format_historical_context(text_list)
            historical_result = {}
            # 2. 历史上下文与当前输入合并RAG一次
            historical_and_current_input = self._format_historical_and_current_input(text_list)
            historical_and_current_input_result = {}

            # 只有当历史上下文不为空时才进行RAG搜索
            if historical_context.strip():
                historical_text_list = [[historical_context, ""]]
                historical_and_current_input_text_list = [[historical_and_current_input, ""]]

                historical_result_des, historical_result_query, historical_and_current_result_des, historical_and_current_result_query = await asyncio.gather(
                    self.tsm_rags.rag_search_function_des(domain_list, historical_text_list, round(top_k/2)),
                    self.tsm_rags.rag_search_function_query(domain_list, historical_text_list, top_k=2),
                    self.tsm_rags.rag_search_function_des(domain_list, historical_and_current_input_text_list, round(top_k/2)),
                    self.tsm_rags.rag_search_function_query(domain_list, historical_and_current_input_text_list, top_k=2)
                )

                # 合并历史上下文的结果
                for instance, funcs in historical_result_query.items():
                    if instance not in historical_result_des:
                        historical_result[instance] = funcs
                    else:
                        historical_result[instance] = historical_result_des[instance]
                        # 合并不同的功能名
                        historical_result[instance].extend(funcs)
                        # 去重, 保留原始顺序
                        historical_result[instance] = list(dict.fromkeys(historical_result[instance]))
                # 合并历史上下文与当前输入的结果
                for instance, funcs in historical_and_current_result_query.items():
                    if instance not in historical_and_current_result_des:
                        historical_and_current_input_result[instance] = funcs
                    else:
                        historical_and_current_input_result[instance] = historical_and_current_result_des[instance]
                        # 合并不同的功能名
                        historical_and_current_input_result[instance].extend(funcs)
                        # 去重, 保留原始顺序
                        historical_and_current_input_result[instance] = list(dict.fromkeys(historical_and_current_input_result[instance]))

            # 3. 当前输入单独RAG一次（不带"当前输入"字符串）
            current_input = self._format_current_input(text_list)
            # 构建当前输入的text_list格式（用于传递给现有RAG方法）
            current_text_list = [[current_input, ""]]

            current_result_des, current_result_query = await asyncio.gather(
                self.tsm_rags.rag_search_function_des(domain_list, current_text_list, top_k),
                self.tsm_rags.rag_search_function_query(domain_list, current_text_list, top_k=2)
            )

            # 合并当前输入的结果
            current_result = {}
            for instance, funcs in current_result_query.items():
                if instance not in current_result_des:
                    current_result[instance] = funcs
                else:
                    current_result[instance] = current_result_des[instance]
                    # 合并不同的功能名
                    current_result[instance].extend(funcs)
                    # 去重, 保留原始顺序
                    current_result[instance] = list(dict.fromkeys(current_result[instance]))

            # 3. 合并三路RAG的结果
            final_result = self._merge_rag_results(current_result, historical_result,
                                                   historical_and_current_input_result, text_list)


            # 加入deep函数
            deep_functions = self.tsm_rags.get_domains_deep_functions(domain_list)
            for instance in final_result.keys():
                final_result[instance].extend(deep_functions)


            return final_result
        else:
            result_des, result_query = await asyncio.gather(
                self.tsm_rags.rag_search_function_des(domain_list, text_list, top_k),
                self.tsm_rags.rag_search_function_query(domain_list, text_list, top_k=2)
            )

            # 合并结果，优先保留功能描述中的结果
            for instance, funcs in result_query.items():
                if instance not in result_des:
                    result_des[instance] = funcs
                else:
                    # 合并不同的功能名
                    result_des[instance].extend(funcs)
                    # 去重, 保留原始顺序
                    result_des[instance] = list(dict.fromkeys(result_des[instance]))

            # 加入deep函数
            deep_functions = self.tsm_rags.get_domains_deep_functions(domain_list)
            for instance in result_des.keys():
                result_des[instance].extend(deep_functions)

            return result_des

    async def rag_search_hotfix(self, domain_list: List[str], query: str, dlg_function: str, similarity_threshold: float = 0.98) -> Optional[Dict[str, Any]]:
        """
        使用RAG搜索热修复功能
        通过余弦相似度匹配用户输入与向量库中的功能查询

        Args:
            domain_list: 领域列表，格式为[领域1, 领域2, ...]
            query: 用户输入文本
            dlg_function: 当前对话函数
            similarity_threshold: 相似度阈值，默认0.95

        Returns:
            符合原有classify方法输出格式的字典，键为实例名，值为功能名列表
        """
        domain_list = self.tsm_rags.get_available_domains(domain_list)
        if not domain_list:
            logger.error("function classify rag_search_hotfix input domains is empty")
            return None
        else:
            logger.info(f"function classify rag_search_hotfix input domains: {domain_list}")
            return await self.tsm_rags.rag_search_hotfix(domain_list, query, dlg_function, similarity_threshold)
