#!/usr/bin/env python3
"""TSM Request 输入模型定义

定义 TSM 接收的 JSON body 格式的数据模型。
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any

class FunctionParam(BaseModel):
    """函数参数模型"""
    name: str = Field(..., description="参数名称")
    value: Any = Field(..., description="参数值")

class FunctionRequest(BaseModel):
    """函数请求模型"""
    name: str = Field(..., description="函数名称")
    confidence: float = Field(default=1.0, description="置信度")
    domain: Optional[str] = Field(default="", description="功能域")
    classify: Optional[str] = Field(default=None, description="分类")
    task: Optional[str] = Field(default=None, description="任务名称")
    params: Optional[List[FunctionParam]] = Field(default=None, description="参数列表")

class Functions(BaseModel):
    """函数包装模型"""
    request: FunctionRequest = Field(..., description="函数请求")

class SpeakOutput(BaseModel):
    """输出中的 speak 部分"""
    type: str = Field(default="text", description="输出类型")
    text: Optional[str] = Field(default="", description="输出的文本内容")


class WidgetOutput(BaseModel):
    """输出中的 widget 部分"""
    streamType: str = Field(default="intermediate", description="流类型")
    type: str = Field(default="displayText", description="widget 类型")
    displayText: Optional[str] = Field(default="", description="显示的文本")


class DialogOutput(BaseModel):
    """对话历史中的输出"""
    speak: Optional[SpeakOutput] = None
    widget: Optional[WidgetOutput] = None

class DialogHistory(BaseModel):
    """对话历史记录"""
    timestamp: int = Field(default=0, description="时间戳")
    input: str = Field(default="", description="用户输入")
    skill: str = Field(default="", description="技能名称")
    source: str = Field(default="", description="来源")
    output: Optional[DialogOutput] = Field(default=None, description="输出信息")
    functions: Optional["Functions"] = Field(default=None, description="函数调用信息")
    dataSource: Optional[str] = Field(default="", description="数据源")


class Session(BaseModel):
    """会话信息"""
    dialogHistory: Optional[List[DialogHistory]] = Field(default_factory=list, description="对话历史")


class Product(BaseModel):
    """产品信息"""
    productId: str = Field(..., description="产品 ID")
    productVersion: str = Field(..., description="产品版本")


class TsmInfo(BaseModel):
    """TSM 信息"""
    ready: Optional[bool] = Field(default=False, description="是否存在定制Tsm资源")
    domain: Optional[str] = Field(default=None, description="领域")


class Skill(BaseModel):
    """技能信息"""
    id: str = Field(..., description="技能 ID")
    name: str = Field(..., description="技能名称")
    task: Optional[str] = Field(default=None, description="任务名称")
    version: Optional[str] = Field(default=None, description="版本")
    domain: Optional[str] = Field(default=None, description="领域")
    tsm: Optional[TsmInfo] = Field(default=None, description="物模型信息")
    useTsm: Optional[bool] = Field(default=False, description="是否使用物模型")

class Context(BaseModel):
    """上下文信息"""
    product: Optional[Product] = Field(default=None, description="产品信息")
    skills: Optional[List[Skill]] = Field(default_factory=list, description="技能列表")


class RequestBody(BaseModel):
    """请求信息"""
    input: str = Field(..., description="用户输入")
    pinyin: Optional[str] = Field(default=None, description="用户输入的拼音")
    requestId: Optional[str] = Field(default=None, description="请求ID")
    recordId: Optional[str] = Field(default=None, description="记录ID， 一般情况下与requestId相同")


class TsmInput(BaseModel):
    """服务器输入的完整数据模型"""
    request: RequestBody = Field(..., description="请求信息")
    session: Optional[Session] = Field(default=None, description="会话信息")
    context: Optional[Context] = Field(default=None, description="上下文信息")
