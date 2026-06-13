#!/usr/bin/env python3
"""TSM Response 输出模型定义

定义 TSM 返回的 JSON 格式的数据模型。
"""
import logging
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from .tsm_input import TsmInput
from .tsm_input import Functions, FunctionRequest, FunctionParam
from .tsm_post import TsmPost
from .tsm_config import tsm_config
import json
import pdb

logger = logging.getLogger(__name__)

class TsmResultItem(BaseModel):
    """TSM结果项模型"""
    res: Optional[str] = Field(default="", description="响应结果")
    input: Optional[str] = Field(default="", description="文本输入")
    functions: Optional[Functions] = Field(default=None, description="函数信息")
    skill: Optional[str] = Field(default=None, description="技能名称")
    skillId: Optional[str] = Field(default=None, description="技能ID")
    


class Metric(BaseModel):
    """指标模型"""
    systime: Optional[float] = Field(default=0.0, description="系统总耗时(毫秒)")
    hottime: Optional[float] = Field(default=0.0, description="高频检索耗时(毫秒)")
    clstime: Optional[float] = Field(default=0.0, description="二级分类耗时(毫秒)")
    fuctime: Optional[float] = Field(default=0.0, description="函数解析耗时(毫秒)")


class RequestInfo(BaseModel):
    """请求信息模型"""
    pinyin: Optional[str] = Field(default="", description="拼音")
    text: Optional[str] = Field(default="", description="文本输入")
    requestId: Optional[str] = Field(default="", description="请求ID")
    recordId: Optional[str] = Field(default="", description="录音ID")


class Result(BaseModel):
    """结果模型"""
    tsmResult: List[TsmResultItem] = Field(default_factory=list, description="TSM结果列表")


def update_ctsm_skill_info(tsm_result_item: TsmResultItem, tsm_input: TsmInput, function_domain: str):
    tsm_result_item.functions.request.domain = ""
    if tsm_input.context and tsm_input.context.skills:
        for skill in tsm_input.context.skills:
            if skill.useTsm and skill.task == function_domain:
                tsm_result_item.skill = skill.name
                tsm_result_item.skillId = skill.id
                tsm_result_item.functions.request.task = skill.task
                break


class TsmOutput(BaseModel):
    """TSM 服务输出模型

    基于协议定义的完整输出结构，包含：
    - errno: 错误码
    - error: 错误信息
    - metric: 性能指标
    - request: 请求信息
    - result: 结果信息（错误场景下为空）
    """
    errno: int = Field(default=0, description="错误码，0表示成功")
    error: str = Field(default="", description="错误信息")
    metric: Metric = Field(default_factory=Metric, description="性能指标")
    request: RequestInfo = Field(default_factory=RequestInfo, description="请求信息")
    result: Optional[Result] = Field(default_factory=Result, description="结果信息，错误场景下为空")

    @classmethod
    def from_current_message(
        cls,
        current_message: dict,
        tsm_input: TsmInput,
        tsm_post: TsmPost,
        instances: list = None,
        request_domains : List[str] = None
    ) -> "TsmOutput":
        """基于 current_message 和 tsm_input 构建 TsmOutput 对象

        Args:
            current_message: 处理后的消息字典，包含 call_functions、clstime、fuctime、systime 等信息
            tsm_input: TsmInput 对象，包含 request、session 和 context
            instances: 函数实例列表，用于获取函数定义信息

        Returns:
            TsmOutput: 构建好的输出对象
        """

        def get_function_definition(function_name: str) -> dict:
            """根据函数名查找函数定义"""
            if not instances:
                return None
            for instance in instances:
                all_functions = instance.get("all_functions", [])
                for func in all_functions:
                    if (isinstance(func, dict) and func.get("name") == function_name):
                        return func
            return None

        def get_function_domain(function_name: str) -> str:
            """根据函数名查找函数所属的 domain"""
            if not instances:
                return None
            for instance in instances:
                all_functions_define = instance.get("all_functions_define", {})
                for domain, function_names in all_functions_define.items():
                    if function_name in function_names:
                        return domain
            return None

        def get_function_classify(function_name: str) -> str:
            """根据函数名查找函数所属的分类"""
            if not instances:
                return None
            for instance in instances:
                all_functions = instance.get("all_functions", [])
                for func in all_functions:
                    if (isinstance(func, dict) and func.get("name") == function_name):
                        return func.get("classify", "")
            return None

        metric = Metric(
            systime=round(current_message.get("systime", 0) * 1000, 2) if current_message.get("systime") is not None else None,
            hottime=round(current_message.get("hottime", 0) * 1000, 2) if current_message.get("hottime") is not None else None,
            clstime=round(current_message.get("clstime", 0) * 1000, 2) if current_message.get("clstime") is not None else None,
            fuctime=round(current_message.get("fuctime", 0) * 1000, 2) if current_message.get("fuctime") is not None else None
        )

        tsm_result_items: List[TsmResultItem] = []

        if "call_functions" in current_message:
            call_functions = current_message["call_functions"]

            if "modify_functions" in call_functions:
                for func in call_functions["modify_functions"]:
                    if isinstance(func, dict):
                        function_params = []
                        function_name = func.get("function_name", "")
                        function_def = get_function_definition(function_name)
                        function_domain = get_function_domain(function_name)
                        function_classify = get_function_classify(function_name)

                        if "input_params" in func:
                            for param_name, param_value in func["input_params"].items():
                                param_type = "string"
                                if function_def and "input_param" in function_def:
                                    if param_name in function_def["input_param"]:
                                        param_type = function_def["input_param"][param_name].get("define", {}).get("type", "string")

                                converted_value = tsm_post.process_param(function_domain, function_name, param_name, param_value, param_type)
                                function_params.append(
                                    FunctionParam(name=param_name, value=converted_value)
                                )

                        function_request = FunctionRequest(
                            name=function_name,
                            params=function_params if function_params else None,
                            confidence=1.0,
                            domain=function_domain or "Unsupport",
                            task=function_domain or "Unsupport",
                            classify=function_classify or "Unsupport"
                        )

                        tsm_result_item = TsmResultItem(
                            res=current_message.get("model", ""),
                            input=tsm_input.request.input,
                            functions=Functions(request=function_request)
                        )

                        if tsm_config.app.uri == "ctsm":
                            update_ctsm_skill_info(tsm_result_item, tsm_input, function_domain)

                        tsm_result_items.append(tsm_result_item)

        tsm_output = cls(
            errno=0,
            error="",
            metric=metric,
            request=RequestInfo(
                pinyin=tsm_input.request.pinyin or None,
                text=tsm_input.request.input,
                requestId=tsm_input.request.requestId or "",
                recordId=tsm_input.request.recordId or ""
            ),
            result=Result(tsmResult=tsm_result_items)
        )

        return tsm_output


if __name__ == "__main__":
    print("=== TSM Output 测试 ===")

    output = TsmOutput(
        errno=0,
        error="",
        metric=Metric(
            systime=5,
            clstime=3,
            fuctime=2
        ),
        request=RequestInfo(
            pinyin="da kai hou pai yue du deng",
            text="打开后排阅读灯",
            requestId="uuid",
            recordId="uuid"
        ),
        result=Result(
            tsmResult=[
                TsmResultItem(
                    res="built-in-xx",
                    functions=Functions(
                        request=FunctionRequest(
                            name="controlReadingLight",
                            confidence=1,
                            domain="CarControl",
                            params=[
                                FunctionParam(name="position", value="REAR_SEAT"),
                                FunctionParam(name="action", value="OPEN")
                            ]
                        )
                    ),
                    classify="airConditioning"
                )
            ]
        )
    )

    print("\n完整输出（JSON 格式）：")
    print(output.model_dump_json(indent=2, ensure_ascii=False))

    print("\n字典格式：")
    print(output.model_dump())

    print("\n=== 错误情况测试 ===")
    error_output = TsmOutput(
        errno=1001,
        error="无效的请求参数",
        metric=Metric(systime=100.0),
        request=RequestInfo(text="测试输入"),
        result=None
    )
    print("\n错误输出（result 为空）：")
    print(error_output.model_dump_json(indent=2, ensure_ascii=False))

    print("\n测试完成！")
