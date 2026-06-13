#!/usr/bin/env python3
"""
从car_control_func.jsonl中提取例句，保存为"例句 => function_name"格式的txt文件
"""

import json
from pathlib import Path
import re
import os

def extract_examples(input_file, output_file):
    """提取例句并保存到txt文件"""
    # 输入输出文件路径
    # input_file = Path("function_call_data/car_control_func.jsonl")
    # output_file = Path("function_call_data/car_control_examples.txt")
    # input_file = Path("function_call_data/air_conditioner_func.jsonl")
    # output_file = Path("function_call_data/air_conditioner_examples.txt")
    # input_file = Path("function_call_data/map_func.jsonl")
    # output_file = Path("function_call_data/map_examples.txt")
    # input_file = Path("function_call_data/roomba_func.jsonl")
    # output_file = Path("function_call_data/roomba_examples.txt")
    # input_file = Path("function_call_data/multimedia_func.jsonl")
    # output_file = Path("function_call_data/multimedia_examples.txt")
    input_file = Path(input_file)
    output_file = Path(output_file)
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"输入文件不存在: {input_file}")
        return
    
    examples = []
    
    # 读取并处理JSONL文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析JSON
                data = json.loads(line)
                function_name = data.get("name", "")
                description = data.get("description", "")
                
                if not function_name or not description:
                    continue
                
                # 提取例句部分
                example_match = re.search(r"例句：(.*)", description)
                
                if not example_match:
                    continue
                
                examples_text = example_match.group(1)
                
                # 分割例句
                # 注意：例句之间可能用分号分隔，但需要处理可能的中英文分号
                split_examples = re.split(r"[；;]", examples_text)
                
                # 清理并添加到结果列表
                for example in split_examples:
                    example = example.strip()
                    if example:
                        examples.append(f"{example} => {function_name}")
                
            except json.JSONDecodeError:
                print(f"解析JSON失败: {line}")
                continue
    
    # 保存到输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(f"{example}\n")
    
    print(f"成功提取 {len(examples)} 个例句，保存至: {output_file}")

if __name__ == "__main__":
    for domain in ["car_control", "map", "multimedia","home_control", "roomba", "calendar", "weather"]:
        extract_examples(f"function_call_data/{domain}_func.jsonl", f"function_call_data/{domain}_examples.txt")
        if os.path.exists(f"function_call_data/{domain}_add_examples.txt"):
            os.system(f"cat function_call_data/{domain}_add_examples.txt >> function_call_data/{domain}_examples.txt")
