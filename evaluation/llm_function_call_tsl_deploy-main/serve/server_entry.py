#!/usr/bin/env python3
"""TSM Server 服务入口点

定义FastAPI应用实例，供launch_server.py和Gunicorn使用。
"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from contextvars import ContextVar
import logging
import tsmrt.manager as dialog_manager
from tsmrt.tsm_config import tsm_config
from serve.server_process import process_tsm_request
import json

dm = dialog_manager.Dialog_Manager(description="智能座舱")

request_info_var: ContextVar[str] = ContextVar("request_info", default="")


class RequestInfoFilter(logging.Filter):
    def filter(self, record):
        info = request_info_var.get("")
        record.request_info = f', request: "{info}"' if info else ""
        return True

app = FastAPI(
    title="TSM Server",
    description="Things Specification Model Runtime Server",
    version="1.0.0"
)

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path
    raw_query = request.scope.get("query_string", b"").decode("latin-1")
    full_path = f"{path}?{raw_query}" if raw_query else path
    http_version = request.scope.get("http_version", "1.1")
    token = request_info_var.set(f"{method} {full_path} HTTP/{http_version}")
    try:
        return await call_next(request)
    finally:
        request_info_var.reset(token)

class EchoRequest(BaseModel):
    """Echo请求模型"""
    message: Optional[str] = ""

@app.get("/")
async def root():
    """根路径"""
    return {"message": "TSM Server is running", "status": "ok"}

@app.get("/healthz")
async def healthz():
    """Healthz接口"""
    return {"message": "ok"}

@app.post(f"/{tsm_config.app.uri}/v1/functions")
async def tsm_functions(request: Request):
    """
    TSM 功能处理接口

    Args:
        request: FastAPI Request 对象，用于获取原始 POST body 和 URI 参数

    Returns:
        处理结果
    """
    try:
        request_body = await request.json()
    except Exception:
        return {"errno": -1, "error": "invalid json body"}
    # 获取 URI 查询参数
    request_params = dict(request.query_params)

    # 调用业务处理函数
    return await process_tsm_request(request_body, request_params, dm)
