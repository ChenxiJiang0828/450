#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: sj123(sheng.jiang@aispeech.com)

import sys
import json
import math
import requests
from pathlib import Path
from typing import Union, List, Dict, Any
import faiss
import numpy as np
import pdb
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# API_URL = "http://llm-text2embedding.gk-internal.prod.duiopen.com/embed"
EMBEDDING_MODE = os.getenv("TSMRT_EMBEDDING_MODE", "remote")
EMBEDDING_URL = os.getenv("TSMRT_EMBEDDING_URL", "http://10.12.7.83:50030").rstrip("/") + "/embed"
EMBEDDING_MODEL = os.getenv("TSMRT_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("TSMRT_EMBEDDING_DEVICE", "cuda")
EMBEDDING_WORKERS = int(os.getenv("TSMRT_EMBEDDING_WORKERS", "20"))
_LOCAL_MODEL = None


def _load_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer

        print(f"[embedding] loading local model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
        _LOCAL_MODEL = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        print(f"[embedding] dim={_LOCAL_MODEL.get_sentence_embedding_dimension()}")
    return _LOCAL_MODEL


def get_embedding(text: str):
    """Return one normalized embedding from local model or remote API."""
    if EMBEDDING_MODE == "local":
        model = _load_local_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    payload = {"inputs": text}
    resp = requests.post(EMBEDDING_URL, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[0]



def update_embedding(VECTOR_DB_PATH_DES, VECTOR_DB_PATH_QUERY, VECTOR_DB_PATH_HOTFIX, support_domain):
    support_domains = [support_domain]
    func_jsonl_paths = ["function_call_data/"+it+"_func.jsonl" for it in support_domains]
    example_txt_paths = ["function_call_data/"+it+"_examples.txt" for it in support_domains]
    hotfix_txt_paths = ["function_call_data/"+it+"_bugfix.txt" for it in support_domains]
    save_descriptions_to_vector_db(VECTOR_DB_PATH_DES, VECTOR_DB_PATH_QUERY, VECTOR_DB_PATH_HOTFIX,func_jsonl_paths, example_txt_paths, hotfix_txt_paths)


def save_descriptions_to_vector_db(VECTOR_DB_PATH_DES, VECTOR_DB_PATH_QUERY, VECTOR_DB_PATH_HOTFIX,func_jsonl_paths, example_txt_paths, hotfix_txt_paths):
    """
    从JSONL文件中提取每个tool的description，生成嵌入向量并保存到向量库
    """
    jsonl_paths = [Path(it) for it in func_jsonl_paths]
    for jsonl_path in jsonl_paths:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL文件不存在: {jsonl_path}")
    

    vector_db_hotfix = []
    for hotfix_txt_path in hotfix_txt_paths:
        if not os.path.exists(hotfix_txt_path):
            print(f"查询热修文件不存在: {hotfix_txt_path}")
            continue
        
        lines_data = []
        for line in open(hotfix_txt_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            print(line)
            parsed = json.loads(line)
            lines_data.append({
                "example": parsed["case"],
                "function_result": parsed["function_result"],
                "dlg_function": parsed["dlg_function"],
                "dlg_domain": parsed["dlg_domain"]
            })
        
        def process_hotfix_item(item: Dict[str, Any]) -> Dict[str, Any]:
            embedding = get_embedding(item["example"])
            return {
                "description": item["example"],
                "function_result": item["function_result"],
                "dlg_function": item["dlg_function"],
                "dlg_domain": item["dlg_domain"],
                "embedding": embedding
            }
        
        pbar = tqdm(total=len(lines_data), desc=hotfix_txt_path+"生成向量库进度（hotfix级别）", unit="条")
        
        with ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS) as executor:
            futures = {executor.submit(process_hotfix_item, item): item for item in lines_data}
            for future in as_completed(futures):
                result = future.result()
                vector_db_hotfix.append(result)
                pbar.update(1)
        
        pbar.close()
    
    if not os.path.exists(os.path.dirname(VECTOR_DB_PATH_HOTFIX)):
        os.makedirs(os.path.dirname(VECTOR_DB_PATH_HOTFIX))
    with open(VECTOR_DB_PATH_HOTFIX, "w", encoding="utf-8") as f:
        json.dump(vector_db_hotfix, f, ensure_ascii=False, indent=2)
    print(f"向量库已保存至: {VECTOR_DB_PATH_HOTFIX}")

    vector_db = []
    for jsonl_path in jsonl_paths:
        lines_data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tool = json.loads(line)
                description = tool.get("description")
                function_name = tool.get("name")
                if description:
                    lines_data.append({
                        "description": description,
                        "function_name": function_name
                    })
        
        def process_description_item(item: Dict[str, Any]) -> Dict[str, Any]:
            embedding = get_embedding(item["description"])
            return {
                "description": item["description"],
                "function_name": item["function_name"],
                "embedding": embedding
            }
        
        pbar = tqdm(total=len(lines_data), desc=str(jsonl_path) + "生成向量库进度（description级别）", unit="条")
        
        with ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS) as executor:
            futures = {executor.submit(process_description_item, item): item for item in lines_data}
            for future in as_completed(futures):
                result = future.result()
                vector_db.append(result)
                pbar.update(1)
        
        pbar.close()
    
    with open(VECTOR_DB_PATH_DES, "w", encoding="utf-8") as f:
        json.dump(vector_db, f, ensure_ascii=False, indent=2)
    print(f"向量库已保存至: {VECTOR_DB_PATH_DES}")

    vector_db_query = []
    for exmaple_txt_path in example_txt_paths:
        lines_data = []
        for line in open(exmaple_txt_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(" => ")
            if len(parts) >= 2:
                lines_data.append({
                    "example": parts[0],
                    "function_name": parts[1]
                })
        
        def process_query_item(item: Dict[str, Any]) -> Dict[str, Any]:
            embedding = get_embedding(item["example"])
            return {
                "description": item["example"],
                "function_name": item["function_name"],
                "embedding": embedding
            }
        
        pbar = tqdm(total=len(lines_data), desc=exmaple_txt_path+"生成向量库进度（query级别）", unit="条")
        
        with ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS) as executor:
            futures = {executor.submit(process_query_item, item): item for item in lines_data}
            for future in as_completed(futures):
                result = future.result()
                vector_db_query.append(result)
                pbar.update(1)
        
        pbar.close()
    
    # 保存向量库到JSON文件
    with open(VECTOR_DB_PATH_QUERY, "w", encoding="utf-8") as f:
        json.dump(vector_db_query, f, ensure_ascii=False, indent=2)
    print(f"向量库已保存至: {VECTOR_DB_PATH_QUERY}")


support_domains = os.getenv("TSMRT_UPDATE_EMBEDDING_DOMAINS", "multimedia,map,car_control,air_conditioner,roomba,home_control,calendar,weather").split(",")
support_domains = [domain.strip() for domain in support_domains if domain.strip()]
for support_domain in support_domains:
    VECTOR_DB_PATH_DES = f"./embedding_res/embedding_res_{support_domain}/vector_db_description.json"
    VECTOR_DB_PATH_QUERY = f"./embedding_res/embedding_res_{support_domain}/vector_db_query.json"
    VECTOR_DB_PATH_HOTFIX = f"./embedding_res/embedding_res_{support_domain}/vector_db_hotfix.json"
    update_embedding(VECTOR_DB_PATH_DES, VECTOR_DB_PATH_QUERY, VECTOR_DB_PATH_HOTFIX, support_domain)
