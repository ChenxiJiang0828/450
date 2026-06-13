import ast
import json
import os
import logging
from typing import Optional, Any
from .tsm_config import tsm_config

logger = logging.getLogger(__name__)

NORM_SEPARATOR = "[sep]"


class TsmPost:
    def __init__(self):
        self._norm_cache = {}
        self._load_normalization()

    def _load_normalization(self):
        for domain, config in tsm_config.res.func_def_domains.items():
            if "norm" not in config:
                continue
            norm_path = os.path.join(tsm_config.res.func_rag_res, config["rag"], config["norm"])
            if os.path.exists(norm_path):
                with open(norm_path, "r", encoding="utf-8") as f:
                    self._norm_cache[domain] = json.load(f)
                logger.info(f"Loaded normalization for domain: {domain}")

    @staticmethod
    def convert_param_value(value: Any, param_type: str) -> Any:
        try:
            if param_type == "float":
                return float(value)
            elif param_type == "int":
                return int(value)
            elif param_type == "integer":
                return int(value)
            elif param_type == "bool":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            elif param_type == "array":
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, list):
                            return parsed
                    except (ValueError, SyntaxError):
                        pass
                return [value] if value else []
            else:
                return str(value)
        except:
            return str(value)

    def normalize_param(self, domain: str, function_name: str, param_name: str, param_value: Any) -> Any:
        norm_data = self._norm_cache.get(domain)
        if not norm_data:
            return param_value
        if function_name not in norm_data:
            return param_value
        if param_name not in norm_data[function_name]:
            return param_value
        str_value = str(param_value)
        if str_value not in norm_data[function_name][param_name]:
            return param_value
        return f"{param_value}{NORM_SEPARATOR}{norm_data[function_name][param_name][str_value]}"

    def process_param(self, domain: str, function_name: str, param_name: str, param_value: Any, param_type: str = "string") -> Any:
        converted_value = self.convert_param_value(param_value, param_type)
        converted_value = self.normalize_param(domain, function_name, param_name, converted_value)
        return converted_value
