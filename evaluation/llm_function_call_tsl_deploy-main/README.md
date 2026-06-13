# TSM - Things Specification Model Runtime

基于大模型计算的物模型运行时项目，负责解析和执行物模型协议，计算请求对应的物模型 function call。

## 项目概述

TSM（Things Specification Model）是一个基于大语言模型的物模型计算引擎，通过预先定义的物模型协议内容，将请求转换为对应的物模型函数调用，实现智能化的设备控制和管理。

### 核心功能

- **物模型解析**：解析和验证 TSM 协议定义
- **Function Call 生成**：基于 LLM 计算生成物模型函数调用
- **设备状态管理**：维护设备状态和数据模型
- **API 服务**：提供 RESTful API 接口
- **监控与健康检查**：实时监控服务状态

## 目录结构

```
tsmserver/
├── serve/                    # 对外服务功能代码
│   ├── __init__.py
│   ├── healthz/             # 健康检查模块
│   │   ├── __init__.py
│   │   ├── routes.py        # 健康检查路由
│   │   └── models.py        # 健康状态数据模型
│   ├── monitor/             # 监控模块
│   │   ├── __init__.py
│   │   ├── routes.py        # 监控指标路由
│   │   ├── collector.py     # 指标采集
│   │   └── prometheus.py    # Prometheus 集成
│ └── main.py               # FastAPI 应用入口
├── service/                # 业务服务层
│   ├── __init__.py
│   └── metric/             # 指标服务
│       ├── __init__.py
│       ├── handler.py      # 指标处理逻辑
│       └── models.py       # 指标数据模型
├── tsmrt/                  # 核心物模型运行代码
│   ├── __init__.py
│   ├── parser.py           # 物模型协议解析
│   ├── executor.py         # Function call 执行器
│   ├── model.py            # 物模型数据结构
│   ├── llm_client.py       # LLM 客户端接口
│   └── validator.py        # 请求验证
├── tests/                  # 测试代码
│   ├── __init__.py
│   ├── unit/               # 单元测试
│   │   ├── __init__.py
│   │   ├── test_parser.py  # 解析器测试
│   │   └── test_executor.py # 执行器测试
│   ├── functional/         # 功能测试
│   │   ├── __init__.py
│   │   └── test_tsm_flow.py # 物模型流程测试
│   └── api/                # 接口测试
│       ├── __init__.py
│       └── test_endpoints.py # API 端点测试
├── tools/                  # 工具集
│   ├── __init__.py
│   ├── gen_schema.py       # 物模型 Schema 生成工具
│   └── convert.py          # 数据格式转换工具
├── 3rdparty/               # 第三方库源码
│   └── README.md           # 第三方库说明文档
├── scripts/                # 辅助脚本
│   ├── start.sh            # 服务启动脚本
│   ├── stop.sh             # 服务停止脚本
│   ├── setup.sh            # 环境安装脚本
│   └── test.sh             # 测试运行脚本
├── misc/                   # 杂项预留
│   └── README.md           # 说明文档
├── docs/                   # 对外文档
│   ├── architecture.md     # 架构设计文档
│   ├── workflow.md         # 工作流程说明
│   ├── api.md              # API 接口文档
│   └── tsm_spec.md         # 物模型协议规范
├── requirements.txt        # Python 依赖包
├── README.md               # 项目说明文档（本文件）
├── .gitignore             # Git 忽略文件配置
└── pyproject.toml         # 项目配置文件
```

## 模块说明

### serve/

负责对外提供 HTTP 服务，包括健康检查、监控指标、API 路由等。

| 模块 | 功能 |
|------|------|
| `server_entry.py` | FastAPI应用入口，定义API路由和端点 |
| `launch_server.py` | 服务启动脚本，支持命令行参数和配置文件 |
| `server_args.py` | 服务启动参数定义和解析 |
| `healthz/` | 提供服务健康检查接口，返回服务状态和版本信息 |
| `monitor/` | 服务监控模块，采集性能指标、资源使用情况等 |
| `metric/` | 指标服务，实现自定义指标计算和上报 |

### tsmrt/

核心物模型运行时代码，包含协议解析、Function Call 生成和执行等核心逻辑。

| 模块 | 功能 |
|------|------|
| `parser.py` | 解析 TSM 协议文档，构建物模型数据结构 |
| `executor.py` | 执行物模型 Function Call，处理返回结果 |
| `model.py` | 定义物模型数据模型和结构 |
| `llm_client.py` | 对接 LLM 服务，生成 Function Call |
| `validator.py` | 验证请求参数、物模型定义的有效性 |

### tests/

包含单元测试、功能测试和接口测试，确保代码质量和功能正确性。

| 目录 | 功能 |
|------|------|
| `unit/` | 单元测试，测试各个模块的独立功能 |
| `functional/` | 功能测试，测试完整业务流程 |
| `api/` | 接口测试，测试 HTTP API 端点 |

### tools/

提供开发和运维所需的辅助工具。

| 工具 | 功能 |
|------|------|
| `gen_schema.py` | 根据物模型定义生成 JSON Schema |
| `convert.py` | 数据格式转换工具 |

### scripts/

部署和运维脚本。

| 脚本 | 功能 |
|------|------|
| `start.sh` | 启动 TSM 服务 |
| `stop.sh` | 停止 TSM 服务 |
| `setup.sh` | 初始化安装环境和依赖 |
| `test.sh` | 运行测试套件 |

### docs/

项目文档，包含架构设计、工作流程、API 文档等。

| 文档 | 内容 |
|------|------|
| `architecture.md` | 系统架构设计，模块间关系说明 |
| `workflow.md` | 物模型计算流程说明 |
| `api.md` | RESTful API 接口详细文档 |
| `tsm_spec.md` | TSM 协议规范文档 |

## 快速开始

### 环境要求

- Python 3.9+
- Conda 环境（推荐）
- LLM 服务接入（支持 OpenAI、Azure OpenAI 等）

### 安装依赖

```bash
# 创建 conda 环境
conda create -n tsm python=3.9
conda activate tsm

# 安装依赖
pip install -r requirements.txt
```

### 启动服务

```bash
# 使用脚本启动
./scripts/start.sh

# 或使用启动脚本
python -m serve.launch_server --port 8000
```

服务启动后访问：
- API 地址：`http://localhost:25988`
- API 文档：`http://localhost:25988/docs`
- 健康检查：`http://localhost:25988/healthz`
- 监控指标：`http://localhost:25988/monitor/metrics`

## 核心流程

### 物模型计算流程

```
1. 接收请求
   ↓
2. 解析物模型规则 (tsmrt/parser.py)
   ↓
3. 调用 LLM 生成 Function Call (tsmrt/llm_client.py)
   ↓
4. 验证 Function Call (tsmrt/validator.py)
   ↓
5. 执行 Function Call (tsmrt/executor.py)
   ↓
6. 返回结果
```

### API 请求示例

```python
import requests

# 物模型计算请求
response = requests.post("http://localhost:25988/parse", json={
    "device_id": "sensor_001",
    "specification": "...",
    "user_query": "打开空调，温度设置为26度"
})

print(response.json())
```

## 开发指南

### 代码规范

- 遵循 PEP 8 编码规范
- 使用 Type Hints 进行类型注解
- 编写单元测试，覆盖率要求 80%+
- 使用异步编程（async/await）

### 添加新功能

1. 在 `tsmrt/` 中实现核心逻辑
2. 在 `service/` 中添加业务服务层
3. 在 `serve/` 中添加 API 路由
4. 在 `tests/` 中添加测试用例
5. 在 `docs/` 中更新文档

### 运行测试

```bash
# 运行所有测试
./scripts/test.sh

# 运行单元测试
pytest tests/unit/

# 运行功能测试
pytest tests/functional/

# 生成覆盖率报告
pytest --cov=tsmrt tests/
```

## 部署

### Docker 部署

```bash
# 构建镜像
docker build -t tsm-server:latest .

# 运行容器
docker run -d -p 25988:25988 --name tsm-server tsm-server:latest
```

### 生产环境配置

使用 Gunicorn + Uvicorn Worker 部署：

```bash
gunicorn serve.server_entry:app -c scripts/gunicorn_config.py
```

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请联系项目维护团队。

