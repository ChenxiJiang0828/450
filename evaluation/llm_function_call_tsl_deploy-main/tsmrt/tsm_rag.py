import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from .tsm_config import tsm_config
from .utils import requests
import faiss
import time

logger = logging.getLogger(__name__)

API_URL = tsm_config.embedding.url

async def get_embedding(text: str) -> List[float]:
    if tsm_config.embedding.mode == "local":
        from .local_embedding import get_local_embedding
        return await get_local_embedding(text)

    payload = {"inputs": text}
    resp = await requests.post(API_URL, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[0]

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

class TsmRag:
    def __init__(self, domain:str = "CarControl", shared_embedding_cache: Optional[Dict[str, List[float]]] = None):
        self.instances = []

        self.domain = domain
        self.embedding_cache = shared_embedding_cache if shared_embedding_cache is not None else {}
        self.deep_function_name = self.get_deep_function_name()
        self.func_to_skill_map = self.get_func_to_skill_map()

        self.vector_db_des = self.load_vector_db_description()
        self.vector_db_query = self.load_vector_db_query()
        self.vector_db_hotfix = self.load_vector_db_hotfix()
        self.functions = self._build_function_set()
        self.faiss_index_des = None
        self.faiss_index_query = None
        self.faiss_index_hotfix = None
        self.vector_metadata = []
        self.vector_metadata_query = []
        self.vector_metadata_hotfix = []
        self.build_faiss_index_description()
        self.build_faiss_index_query()
        self.build_faiss_index_hotfix()

    def update_instances(self, instances):
        self.instances = instances

    def _build_function_set(self) -> set:
        functions = set()
        for item in self.vector_db_des:
            func_name = item.get("function_name", "")
            if func_name:
                functions.add(func_name)
        for item in self.vector_db_query:
            func_name = item.get("function_name", "")
            if func_name:
                functions.add(func_name)
        for item in self.vector_db_hotfix:
            func_result = item.get("function_result", {})
            if isinstance(func_result, dict):
                func_name = func_result.get("name", "")
                if func_name:
                    functions.add(func_name)
        return functions

    def get_deep_function_name(self) -> str:
        if self.domain not in tsm_config.res.func_def_domains:
            return ""

        domain_config = tsm_config.res.func_def_domains[self.domain]
        return domain_config.get("deep", "")

    def get_func_to_skill_map(self) -> Dict[str, str]:
        func_to_skill = {}
        for instance in self.instances:
            instance_name = instance.get("_description", "")
            if not instance_name:
                continue

            for func in (instance.get("all_functions", []) or []):
                func_name = func.get("name", "")
                skill = func.get("skill", "")
                if func_name:
                    func_to_skill[func_name] = skill
        return func_to_skill

    def load_vector_db_description(self) -> List[Dict]:
        if self.domain not in tsm_config.res.func_def_domains:
            raise ValueError(f"不支持的领域: {self.domain}")

        rag_res = Path(tsm_config.res.func_rag_res) / tsm_config.res.func_def_domains[self.domain]["rag"]
        vector_db_path = rag_res / "vector_db_description.json"

        if not vector_db_path.exists():
            raise FileNotFoundError(f"向量库文件不存在，请先运行embedding_res/embedding_res_iov/update_embedding_index.py生成: {vector_db_path}")

        with open(vector_db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_vector_db_query(self) -> List[Dict]:
        if self.domain not in tsm_config.res.func_def_domains:
            raise ValueError(f"不支持的领域: {self.domain}")

        rag_res = Path(tsm_config.res.func_rag_res) / tsm_config.res.func_def_domains[self.domain]["rag"]
        vector_db_path = rag_res / "vector_db_query.json"

        if not vector_db_path.exists():
            raise FileNotFoundError(f"向量库文件不存在，请先运行embedding_res/embedding_res_iov/update_embedding_index.py生成: {vector_db_path}")

        with open(vector_db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_vector_db_hotfix(self) -> List[Dict]:
        if self.domain not in tsm_config.res.func_def_domains:
            return []

        rag_res = Path(tsm_config.res.func_rag_res) / tsm_config.res.func_def_domains[self.domain]["rag"]
        vector_db_path = rag_res / "vector_db_hotfix.json"

        if not vector_db_path.exists():
            return []

        with open(vector_db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_faiss_index_description(self):
        try:
            vectors = []
            self.vector_metadata = []

            for item in self.vector_db_des:
                vectors.append(item["embedding"])
                self.vector_metadata.append({
                    "function_name": item.get("function_name", ""),
                    "description": item.get("description", "")
                })

            vectors = np.array(vectors, dtype="float32")
            dimension = vectors.shape[1]
            self.faiss_index_des = faiss.IndexFlatIP(dimension)
            self.faiss_index_des.add(vectors)
        except ImportError:
            logger.warning("FAISS library not installed, using brute-force similarity search")
            self.faiss_index_des = None

    def build_faiss_index_query(self):
        try:
            vectors_query = []
            self.vector_metadata_query = []

            for item in self.vector_db_query:
                vectors_query.append(item["embedding"])
                self.vector_metadata_query.append({
                    "function_name": item.get("function_name", ""),
                    "description": item.get("description", "")
                })

            vectors_query = np.array(vectors_query, dtype="float32")
            dimension_query = vectors_query.shape[1]
            self.faiss_index_query = faiss.IndexFlatIP(dimension_query)
            self.faiss_index_query.add(vectors_query)
        except ImportError:
            logger.warning("FAISS library not installed, using brute-force similarity search")
            self.faiss_index_query = None

    def build_faiss_index_hotfix(self):
        try:
            vectors = []
            self.vector_metadata_hotfix = []
            if not self.vector_db_hotfix:
                self.faiss_index_hotfix = None
                return

            for item in self.vector_db_hotfix:
                vectors.append(item["embedding"])
                self.vector_metadata_hotfix.append({
                    "function_result": item.get("function_result", ""),
                    "dlg_function": item.get("dlg_function", ""),
                    "dlg_domain": item.get("dlg_domain", "")
                })

            vectors = np.array(vectors, dtype="float32")
            self.dimension_hotfix = vectors.shape[1]
            self.faiss_index_hotfix = faiss.IndexFlatIP(self.dimension_hotfix)
            self.faiss_index_hotfix.add(vectors)
        except ImportError:
            logger.warning("FAISS library not installed, using brute-force similarity search")
            self.faiss_index_hotfix = None

    async def get_embedding_with_cache(self, text: str) -> List[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        embedding = await get_embedding(text)
        self.embedding_cache[text] = embedding

        return embedding

    def build_format_input(self, text_list: List[List[str]]):
        if not text_list:
            return ""
        text_str = []
        if len(text_list) > 1:
            for i in range(len(text_list)):
                if i == (len(text_list) -1 ):
                    text_str.append(f"当前输入:{text_list[i][0]}")
                else:
                    text_str.append(f"历史输入:{text_list[i][0]}\n历史输出:{text_list[i][1]}")
        else:
            text_str.append(f"{text_list[0][0]}")

        format_input = "\n".join(text_str)
        return format_input

    async def search(self, rag_type: str, text_list: List[List[str]], top_k: int = 5) -> Dict[str, List[str]]:
        try:
            similarity_results = []
            if rag_type == "description":
                faiss_index = self.faiss_index_des
                meta_data = self.vector_metadata
                search_count = max(top_k, 20)
            elif rag_type == "query":
                faiss_index = self.faiss_index_query
                meta_data = self.vector_metadata_query
                search_count = max(top_k * 3, 20)
            else :
                logger.warning(f"rag search not support type: {rag_type}")
                return similarity_results

            input_format = self.build_format_input(text_list)
            if not input_format.strip():
                return similarity_results
            input_embedding = await self.get_embedding_with_cache(input_format)
            input_vector = np.array([input_embedding], dtype="float32")

            D, I = faiss_index.search(input_vector, search_count)

            for i in range(len(I[0])):
                idx = I[0][i]
                score = D[0][i]
                metadata = meta_data[idx]
                similarity_results.append({
                    "function_name": metadata["function_name"],
                    "description": metadata["description"],
                    "similarity": score
                })
            return similarity_results
        except Exception as e:
            logger.error(f"rag search error : {traceback.format_exc()} : {e}")
            return []

    async def retrieve_similar_queries(self, query: str, similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        query_embedding = await self.get_embedding_with_cache(query)
        query_vector = np.array([query_embedding], dtype="float32")

        all_results = []

        if self.faiss_index_hotfix:
            D, I = self.faiss_index_hotfix.search(query_vector, len(self.vector_metadata_hotfix))
            for i in range(len(I[0])):
                idx = I[0][i]
                score = D[0][i]
                if score < similarity_threshold:
                    continue

                metadata = self.vector_metadata_hotfix[idx]
                all_results.append({
                    "function_result": metadata["function_result"],
                    "dlg_function": metadata["dlg_function"],
                    "dlg_domain": metadata["dlg_domain"],
                    "similarity": score,
                    "source": "description"
                })
        else:
            for item in self.vector_db_hotfix:
                score = cosine_similarity(query_embedding, item["embedding"])
                if score < similarity_threshold:
                    continue

                all_results.append({
                    "function_result": item.get("function_result", ""),
                    "dlg_function": item.get("dlg_function", ""),
                    "dlg_domain": item.get("dlg_domain", ""),
                    "similarity": score,
                    "source": "description"
                })

        if self.faiss_index_hotfix:
            D, I = self.faiss_index_hotfix.search(query_vector, len(self.vector_metadata_hotfix))
            for i in range(len(I[0])):
                idx = I[0][i]
                score = D[0][i]
                if score < similarity_threshold:
                    continue

                metadata = self.vector_metadata_hotfix[idx]
                all_results.append({
                    "function_result": metadata["function_result"],
                    "dlg_function": metadata["dlg_function"],
                    "dlg_domain": metadata["dlg_domain"],
                    "similarity": score,
                    "source": "query"
                })
        else:
            for item in self.vector_db_hotfix:
                score = cosine_similarity(query_embedding, item["embedding"])
                if score < similarity_threshold:
                    continue

                all_results.append({
                    "function_result": item.get("function_result", ""),
                    "dlg_function": item.get("dlg_function", ""),
                    "dlg_domain": item.get("dlg_domain", ""),
                    "similarity": score,
                    "source": "query",
                })

        unique_results = {}
        for result in all_results:
            key = f"{result['function_result']}"
            if key not in unique_results or result['similarity'] > unique_results[key]['similarity']:
                unique_results[key] = result

        sorted_results = sorted(unique_results.values(), key=lambda x: x['similarity'], reverse=True)

        return sorted_results

    def filter_and_rank_results(self, results: List[Dict[str, Any]], historical_functions: List[str]) -> List[Dict[str, Any]]:
        historical_skills = []
        for func in historical_functions:
            if func in self.func_to_skill_map:
                historical_skills.append(self.func_to_skill_map[func])
        historical_skills = list(set(historical_skills))

        class1_results = []
        class2_results = []
        class3_results = []

        for result in results:
            func_name = result["function_result"]["name"]
            dlg_function = result["dlg_function"]
            dlg_domain = result["dlg_domain"]
            if dlg_function in historical_functions:
                class1_results.append(result)
            elif dlg_function == "" and dlg_domain in historical_skills:
                class2_results.append(result)
            elif dlg_function == "" and dlg_domain == "":
                class3_results.append(result)

        class1_results.sort(key=lambda x: x['similarity'], reverse=True)
        class2_results.sort(key=lambda x: x['similarity'], reverse=True)
        class3_results.sort(key=lambda x: x['similarity'], reverse=True)

        final_results = class1_results + class2_results + class3_results

        return final_results

    async def process_hotfix_match(self, query: str, dlg_function: str, similarity_threshold: float = 0.96) -> Optional[Dict[str, Any]]:
        if not self.faiss_index_hotfix:
            return None

        similar_results = await self.retrieve_similar_queries(query, similarity_threshold)
        if not similar_results:
            return None

        historical_functions = []
        if dlg_function and dlg_function != "":
            historical_functions.append(dlg_function)

        ranked_results = self.filter_and_rank_results(similar_results, historical_functions)

        best_function_name = None
        best_result = None
        for result in ranked_results:
            if result["function_result"]:
                best_function_name = result["function_result"]["name"]
                best_result = result
                break

        if best_function_name is None:
            return None

        logger.info(f"Using embedding hotfix to call function directly: {best_function_name}")

        input_params = {}
        for it in best_result["function_result"]["param"].keys():
            input_params[it] = best_result["function_result"]["param"][it]

        modify_functions = [{"function_name": best_function_name, "input_params": input_params}]
        query_funcs = []
        c_messages = [
            f"Action: {best_function_name}\nInput:{json.dumps(input_params)}",
            "Observation: {\"status\": \"success\"}"
        ]

        return {
            "modify_functions": modify_functions,
            "query_funcs": query_funcs,
            "c_messages": c_messages,
            "function_name": best_function_name,
            "input_params": input_params
        }

class TsmRags:
    def __init__(self):
        self.rags = {}
        self.domains = []
        self.shared_embedding_cache = {}

        self.rags = self.load_domains_rags()
        self.domains = self.get_builtin_domains()

    def update_instances(self, instances):
        for rag in self.rags.values():
            if rag is not None:
                rag.update_instances(instances)

    def load_domains_rags(self):
        domain_rags = {}
        for domain in tsm_config.res.func_def_domains.keys():
            try:
                domain_rags[domain] = TsmRag(domain, self.shared_embedding_cache)
                logger.info(f"succ to load rag for domain : {domain}")
            except Exception as e:
                logger.warning(f"fail to load rag for domain : {domain} with error : {e}")
                domain_rags[domain] = None
        return domain_rags

    def get_domain_rag(self, domain: str) -> Optional[TsmRag]:
        return self.rags.get(domain)

    def get_available_domains(self, domain_list: List[str]) -> List[str]:
        if not domain_list:
            logger.warning(f"function classify request domain list is empty, return default all domains : {self.domains}")
            return self.domains
        else :
            if tsm_config.app.uri == "ctsm":
                domains = [domain for domain in domain_list if domain in self.domains]
            else:
                domains = self.get_builtin_domains()
        return domains

    def get_available_domains_v2(self, domain_list: List[str]) -> List[str]:
        if not domain_list:
            return self.domains
        else :
            for domain in domain_list:
                if domain in tsm_config.res.func_builtin_domains:
                    return tsm_config.res.func_builtin_domains
                elif domain in tsm_config.res.func_other_domains:
                    return tsm_config.res.func_other_domains
                else:
                    return self.domains

    def get_builtin_domains(self) -> List[str]:
        return [domain for domain, rag in self.rags.items() if rag is not None]

    def get_format_functions(self, functions: List[Dict]) -> Dict[str, List[str]]:
        result = {}
        if functions:
            for func_info in functions:
                function_name = func_info["function_name"]
                function_similarity = func_info["similarity"]
                if "智能座舱" not in result:
                    result["智能座舱"] = []
                if function_name not in result["智能座舱"]:
                    # result["智能座舱"].append({"function_name": function_name, "similarity": function_similarity})
                    result["智能座舱"].append(function_name)
        return result

    def get_domains_deep_functions(self, domain_list: List[str]) -> List[str] :
        deep_functions = []
        for domain_name in domain_list:
            domain_rag = self.get_domain_rag(domain_name)
            if domain_rag is not None:
                deep_functions.append(domain_rag.deep_function_name)
        return deep_functions
    
    def search_domain_by_function(self, function_name: str) -> Optional[str]:
        for domain_name, domain_rag in self.rags.items():
            if domain_rag is not None and function_name in domain_rag.functions:
                return domain_name
        return None

    async def get_domains_functions(self, rag_type:str, domain_list: List[str], text_list: List[List[str]], top_k: int) -> Dict[str, List[str]]:
        total_functions = []

        for domain_name in domain_list:
            domain_rag = self.get_domain_rag(domain_name)
            if domain_rag is not None:
                similarity_results = await domain_rag.search(rag_type, text_list, top_k)
                for item in similarity_results:
                    total_functions.append(item)
            else:
                logger.warning(f"fail to get domains functions with {domain_name}")

        total_functions.sort(key=lambda x: x["similarity"], reverse=True)

        return self.get_format_functions(total_functions[:top_k])

    async def rag_search_function_des(self, domain_list: List[str], text_list: List[List[str]], top_k: int = 5) -> Dict[str, List[str]]:
        domains_functions = await self.get_domains_functions("description", domain_list, text_list, top_k)
        deep_functions = self.get_domains_deep_functions(domain_list)
        # domains_functions["智能座舱"].extend(deep_functions)
        # 过滤所有Deep函数
        domains_functions.setdefault("智能座舱", [])
        domains_functions["智能座舱"] = [func for func in domains_functions["智能座舱"] if func not in deep_functions]
        return domains_functions

    async def rag_search_function_query(self, domain_list: List[str], text_list: List[List[str]], top_k: int = 2) -> Dict[str, List[str]]:
        return await self.get_domains_functions("query", domain_list, text_list, top_k)

    async def rag_search_hotfix(self, domain_list: List[str], query: str, dlg_function: str, similarity_threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        for domain_name in domain_list:
            domain_rag = self.get_domain_rag(domain_name)
            if domain_rag is not None:
                hotfix_result = await domain_rag.process_hotfix_match(query, dlg_function, similarity_threshold)
                if hotfix_result is not None:
                    return hotfix_result
            else:
                logger.warning(f"fail to get domains hotfix with {domain_name}")
        return None
