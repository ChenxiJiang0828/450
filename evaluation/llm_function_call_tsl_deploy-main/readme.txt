# LLM Function Call TSL Deploy Project

## 项目概述

本项目是一个基于大语言模型(LLM)的函数调用系统，支持对话管理、意图分类和函数调用能力。主要用于智能设备控制场景，例如扫地机器人的控制。

## 项目结构

```
llm_function_call_tsl_deploy/
├── src/                    # 核心源代码
│   ├── manager.py          # 对话管理器
│   ├── classify.py         # 意图分类器
│   ├── function_call.py    # 函数调用模块
│   └── utils/              # 工具函数
├── scripts/                # 测试脚本
│   └── roomba/             # 扫地机器人相关测试
│       └── dm_test.py      # 对话管理器测试脚本
├── function_call_data/     # 函数调用数据
│   ├── roomba_func.jsonl   # 扫地机器人函数定义
│   └── tmp/                # 临时文件
└── readme.txt              # 项目说明文档
```

## 核心功能

1. **对话管理**：通过 `Dialog_Manager` 类处理用户输入，维护对话状态
2. **意图分类**：识别用户输入的意图，确定需要调用的功能
3. **函数调用**：根据意图生成相应的函数调用，控制智能设备
4. **多模型支持**：可配置不同的模型用于分类、函数调用和对话管理

## 支持的模型

- Qwen3-235B-A22B
- DFM3-Turbo
- qwen3-max
- qwen3-235b-a22b-instruct-2507
- deepseek-v3-250324
- doubao-seed-1-6-251015

## 使用方法

### 测试对话管理器

1. 进入测试脚本目录：
   ```bash
   cd scripts/roomba
   ```

2. 运行测试脚本：
   ```bash
   python dm_test.py
   ```

3. 输入命令控制扫地机器人，例如：
   ```
   User: 打开拖地模式
   ```

4. 输入 `exit` 退出测试

### 配置模型

在 `dm_test.py` 文件中，可以修改以下配置：

```python
# 配置分类模型
dm.l2_classifier.model="qwen3-235b-a22b-instruct-2507"

# 配置函数调用模型
dm.function_call_agent.model="qwen3-235b-a22b-instruct-2507"

# 配置对话模型
dm.dm_agent.model="DFM3-Turbo"
```

## 函数调用格式

函数调用结果以 JSON 格式返回，包含以下信息：

```json
{
  "function_name": "函数名称",
  "params": "参数1=值1,参数2=值2"
}
```

## 示例命令

- 打开拖地模式
- 调整温度
- 播放音乐

## 注意事项

1. 确保已安装所有必要的依赖
2. 确保模型配置正确且可用
3. 函数调用数据文件 `roomba_func.jsonl` 必须存在

## 联系方式

如有问题，请联系项目维护人员。