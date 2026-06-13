"""
TSM Runtime 閰嶇疆绠＄悊妯″潡
鏀寔浠庣幆澧冨彉閲忚鍙栭厤缃紝浼樺厛绾э細鐜鍙橀噺 > 榛樿鍊?

娉ㄦ剰锛歅ython 妯″潡瀵煎叆鏈哄埗澶╃劧鏀寔鍗曚緥妯″紡
- 妯″潡鍙湪绗竴娆″鍏ユ椂鎵ц涓€娆?
- 鍚庣画瀵煎叆浼氫娇鐢ㄧ紦瀛樼殑妯″潡瀵硅薄
- 鍥犳 tsm_config 瀹炰緥鍦ㄦ暣涓簲鐢ㄤ腑鏄敮涓€鐨?
"""

import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

class AppConfig(BaseModel):
    app_name: str = Field(
        default_factory=lambda: os.getenv("PROJECT_APP_NAME", "tsmserver"),
        description="搴旂敤鍚嶇О锛宼smserver 鎴?ctsmserver, 鑻ヤ娇鐢ㄥ畾鍒剁墿妯″瀷锛岀幆澧冨彉閲忚缃负tsm-custom-service"
        # export PROJECT_APP_NAME=tsm-custom-service
    )
    host_name: str = Field(
        default_factory=lambda: os.getenv("HOSTNAME", "local"),
        description="涓绘満鍚嶇О"
    )

    @property
    def uri(self) -> str:
        return "tsm" if self.app_name == "tsmserver" else "ctsm"

class EmbedConfig(BaseModel):
    """Embedding 閰嶇疆妯″潡"""
    mode: str = Field(
        default_factory=lambda: os.getenv("TSMRT_EMBEDDING_MODE", "remote"),
        description="remote=HTTP API, local=鏈湴 sentence-transformers 妯″瀷",
    )
    url: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_EMBEDDING_URL",
            "http://10.12.7.83:50030"
        ).rstrip('/') + '/embed',
        description="Embedding API 鍦板潃锛坢ode=remote 鏃朵娇鐢級",
    )
    model: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_EMBEDDING_MODEL",
            "BAAI/bge-large-zh-v1.5",
        ),
        description="鏈湴 embedding 妯″瀷锛?024 缁达紝闇€涓庡悜閲忓簱涓€鑷达級",
    )
    device: str = Field(
        default_factory=lambda: os.getenv("TSMRT_EMBEDDING_DEVICE", "cuda"),
        description="鏈湴 embedding 璁惧锛歝uda / cpu",
    )
    backend: str = Field(
        default_factory=lambda: os.getenv("TSMRT_EMBEDDING_BACKEND", "transformers"),
        description="local embedding backend: transformers, flagembedding, or sentence-transformers",
    )

class DfmConfig(BaseModel):
    """DFM 妯″瀷閰嶇疆"""
    url: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_DFM_URL",
            "https://dfm.duiopen.com"
        ).rstrip('/') + "/dfm/v1/compatible-mode",
        description="DFM API 鍦板潃"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_DFM_API_KEY",
            ""
        ),
        description="DFM API 瀵嗛挜"
    )
    product_id: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_DFM_PRODUCT_ID",
            "279631892"
        ),
        description="DFM 浜у搧 ID"
    )


class DoubaoConfig(BaseModel):
    """璞嗗寘妯″瀷閰嶇疆"""
    url: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_DOUBAO_URL",
            "https://ark.cn-beijing.volces.com"
        ).rstrip('/') + "/api/v3/chat/completions",
        description="璞嗗寘 API 鍦板潃"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_DOUBAO_API_KEY",
            ""
        ),
        description="璞嗗寘 API 瀵嗛挜"
    )


class GlmConfig(BaseModel):
    """GLM 妯″瀷閰嶇疆"""
    url: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_GLM_URL",
            "https://open.bigmodel.cn"
        ).rstrip('/') + "/api/paas/v4/chat/completions",
        description="GLM API 鍦板潃"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_GLM_API_KEY",
            ""
        ),
        description="GLM API 瀵嗛挜"
    )

    
class DeepseekV4Config(BaseModel):
    """Deepseek V4 妯″瀷閰嶇疆"""
    url: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_DEEPSEEKV4_URL",
            "https://api.deepseek.com"
        ).rstrip('/') + "/chat/completions",
        description="Deepseek V4 API 鍦板潃"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_DEEPSEEKV4_API_KEY",
            ""
        ),
        description="Deepseek V4 API 瀵嗛挜"
    )

def _qwen_chat_completions_url() -> str:
    raw = os.getenv("TSMRT_QWEN_URL", "https://dashscope.aliyuncs.com").rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    return raw + "/compatible-mode/v1/chat/completions"


class QwenConfig(BaseModel):
    """Qwen 妯″瀷閰嶇疆"""
    url: str = Field(
        default_factory=_qwen_chat_completions_url,
        description="Qwen API 鍦板潃"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_QWEN_API_KEY",
            ""
        ),
        description="Qwen API 瀵嗛挜"
    )

_TSM_FUNC_DEF_DOMAINS = {
    "CarControl": {"fn":"car_control_func.jsonl", "rag":"embedding_res_car_control", "deep":"carControlDeep"},
    "Map":{"fn": "map_func.jsonl", "rag":"embedding_res_map", "deep":"mapDeep"},
    "Multimedia":{"fn": "multimedia_func.jsonl", "rag":"embedding_res_multimedia", "deep":"multimediaDeep"},
}

_CTSM_FUNC_DEF_DOMAINS = {
    "Roomba":{"fn": "roomba_func.jsonl", "rag":"embedding_res_roomba", "deep":"roombaDeep", "norm":"roomba_special_normalization.json"},
    "HomeControl":{"fn": "home_control_func.jsonl", "rag":"embedding_res_home_control", "deep":"homeControlDeep"},
    "Calendar":{"fn": "calendar_func.jsonl", "rag":"embedding_res_calendar", "deep":"calendarDeep"},
    "Weather":{"fn": "weather_func.jsonl", "rag":"embedding_res_weather", "deep":"weatherDeep"}
}

class TsmRes(BaseModel):
    func_def_res: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_FUNC_DEF_RES",
            str(Path(__file__).parent.parent/"function_call_data")
        ),
        description="鐗╂ā鍨婩unction瀹氫箟璧勬簮璺緞"
    )

    func_rag_res: str = Field(
        default_factory=lambda: os.getenv(
            "TSMRT_FUNC_RAG_RES",
            str(Path(__file__).parent.parent/"embedding_res")
        ),
        description="Function embedding resource path"
    )

    func_rag_config: dict = {
        "topk": 10,
        "use_rag": True,
    }
    
    vocab_res_path: dict = {
        "video": str(Path(__file__).parent.parent/"vocab_data/vocab_video.txt"),
        "song": str(Path(__file__).parent.parent/"vocab_data/vocab_song.txt")
    }

    func_cluster_subclassify_path: str = Field(
        default_factory=lambda: str(Path(__file__).parent.parent/"function_call_data/function_cluster_subclassify.json"),
        description="Function subclass cluster file path"
    )

    @property
    def func_cluster_subclassify(self) -> dict:
        path = Path(self.func_cluster_subclassify_path)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)



    @property
    def func_def_domains(self) -> dict:
        return _TSM_FUNC_DEF_DOMAINS if tsm_config.app.uri == "tsm" else _CTSM_FUNC_DEF_DOMAINS

    @property
    def func_builtin_domains(self) -> list:
        return list(_TSM_FUNC_DEF_DOMAINS.keys()) if tsm_config.app.uri == "tsm" else []

    @property
    def func_other_domains(self) -> list:
        return [] if tsm_config.app.uri == "tsm" else list(_CTSM_FUNC_DEF_DOMAINS.keys())

class TsmConfig(BaseModel):
    """TSM runtime config."""
    app: AppConfig = Field(default_factory=AppConfig, description="搴旂敤閰嶇疆")
    embedding: EmbedConfig = Field(default_factory=EmbedConfig, description="Embedding 閰嶇疆")
    dfm: DfmConfig = Field(default_factory=DfmConfig, description="DFM 妯″瀷閰嶇疆")
    doubao: DoubaoConfig = Field(default_factory=DoubaoConfig, description="璞嗗寘妯″瀷閰嶇疆")
    glm: GlmConfig = Field(default_factory=GlmConfig, description="GLM 妯″瀷閰嶇疆")
    qwen: QwenConfig = Field(default_factory=QwenConfig, description="Qwen 妯″瀷閰嶇疆")
    deepseek_v4: DeepseekV4Config = Field(default_factory=DeepseekV4Config, description="Deepseek V4 妯″瀷閰嶇疆")

    res: TsmRes = Field(default_factory=TsmRes, description="TSM 璧勬簮閰嶇疆")



# 鍏ㄥ眬閰嶇疆瀹炰緥
# Python 妯″潡瀵煎叆鏈哄埗淇濊瘉姝ゅ疄渚嬪湪鏁翠釜搴旂敤涓敮涓€锛堝ぉ鐒跺崟渚嬶級
# 鍏朵粬妯″潡鍙洿鎺?import 浣跨敤锛歠rom tsmrt.tsm_config import tsm_config
tsm_config = TsmConfig()

