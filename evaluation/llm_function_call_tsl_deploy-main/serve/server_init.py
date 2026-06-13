import os
import sys
import logging

def on_starting(server):
    """在 master 进程启动时调用"""
    pass

def post_worker_init(worker):
    from serve.server_entry import RequestInfoFilter
    from tsmrt.tsm_config import tsm_config

    log_level = os.getenv("TSMRT_LOG_LEVEL", "info")
    log_dir = os.getenv("TSMRT_LOG_DIR", "./logs")

    host_name = tsm_config.app.host_name
    app_name = tsm_config.app.app_name

    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    log_format = f'{host_name} {app_name} %(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s%(request_info)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    os.makedirs(log_dir, exist_ok=True)

    request_info_filter = RequestInfoFilter()
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, 'server.log'), encoding='utf-8')
    ]
    for handler in handlers:
        handler.addFilter(request_info_filter)

    logging.basicConfig(
        level=log_level_value,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
