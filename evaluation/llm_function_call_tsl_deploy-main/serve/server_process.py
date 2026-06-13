#!/usr/bin/env python3
"""TSM Server 业务处理模块

封装 TSM 功能处理的核心逻辑，将 HTTP 请求处理与业务逻辑分离。
"""

import logging
from tsmrt.tsm_input import TsmInput
from tsmrt.tsm_output import TsmOutput
import json
import uuid
import traceback

logger = logging.getLogger(__name__)


async def process_tsm_request(request_body: dict, request_params: dict, dm) -> TsmOutput:
    """
    处理 TSM 功能请求的核心逻辑

    Args:
        request_body: POST 请求的 JSON body 数据
        request_params: URI 查询参数字典
        dm: Dialog_Manager 实例，用于处理对话

    Returns:
        TsmOutput: 标准化的输出对象
    """
    tsm_input = None
    try:
        logger.info(f"process-input: {json.dumps(request_body, ensure_ascii=False)}")
        # 序列化为 TsmInput
        tsm_input = TsmInput(**request_body)
        tsm_input.request.requestId = request_params.get("requestId", str(uuid.uuid4()))
        tsm_input.request.recordId = request_params.get("recordId", str(uuid.uuid4()))

        # 调用 dialog_manager 处理，返回 TsmOutput
        tsm_output = await dm.chat_process(tsm_input)
        tsm_output = tsm_output.model_dump(exclude_none=True)

        logger.info(f"process-output: {json.dumps(tsm_output, ensure_ascii=False)}")

        return tsm_output
    except Exception as e:
        logger.error(f"process-error: {traceback.format_exc()} : {e}")
        # 返回错误结果
        tsm_output = TsmOutput(
            errno=-1,
            error=str(e)
        )
        tsm_output = tsm_output.model_dump(exclude_none=True)
        logger.info(f"process-error: {json.dumps(tsm_output, ensure_ascii=False)}")
        return tsm_output
