#!/usr/bin/env python3
"""TSM Server 启动脚本

使用方式:
    python -m serve.launch_server --help           # 查看帮助
    python -m serve.launch_server --port 8080     # 指定端口
    python -m serve.launch_server --config config.yaml  # 使用配置文件
    python -m serve.launch_server --reload        # 开发模式（热重载）
    python -m serve.launch_server --workers 4     # 生产模式（多 worker，使用 Gunicorn）
"""

import sys
import os
import logging
import subprocess
import uvicorn

# 添加项目根目录到 Python 路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from serve.server_args import prepare_server_args, ServerArgs
from serve.server_entry import RequestInfoFilter
from tsmrt.tsm_config import tsm_config


def setup_logging(args: ServerArgs) -> None:
    """配置日志系统。

    Args:
        args: 服务启动参数
    """
    # 创建日志目录
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 获取 host name 和 app name
    host_name = tsm_config.app.host_name
    app_name = tsm_config.app.app_name

    # 配置日志格式
    log_format = f'{host_name} {app_name} %(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s%(request_info)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    # 配置日志处理器
    request_info_filter = RequestInfoFilter()
    handlers = [
        logging.StreamHandler(sys.stdout),
        #logging.StreamHandler(sys.stderr),
        logging.FileHandler(os.path.join(log_dir, 'server.log'), encoding='utf-8')
    ]
    for handler in handlers:
        handler.addFilter(request_info_filter)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    # 降低 Uvicorn 日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def launch_server(args: ServerArgs) -> None:
    """启动 TSM Server。

    Args:
        args: 服务启动参数
    """
    # 配置日志
    setup_logging(args)

    logger = logging.getLogger(__name__)

    # 打印启动配置
    logger.info("=" * 50)
    logger.info("TSM Server startup config:")
    logger.info(f"  Listen: {args.host}:{args.port}")
    logger.info(f"  Log level: {args.log_level}")
    logger.info(f"  Log dir: {args.log_dir}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Max concurrency: {args.limit_concurrency}")
    logger.info(f"  Keep-Alive timeout: {args.timeout_keep_alive}s")
    logger.info(f"  Worker timeout: {args.worker_timeout}s")
    logger.info(f"  Backlog: {args.backlog}")
    if args.config_path:
        logger.info(f"  Config file: {args.config_path}")
    logger.info("=" * 50)

    # 根据是否启用热重载选择启动方式
    if args.reload:
        # 开发模式：热重载
        logger.info("Starting in dev mode (hot reload enabled)...")
        uvicorn.run(
            "serve.server_entry:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            reload_dirs=args.reload_dirs,
            log_level=args.log_level,
            access_log=True,
            limit_concurrency=args.limit_concurrency,
            timeout_keep_alive=args.timeout_keep_alive,
            backlog=args.backlog
        )
    else:
        # 生产模式
        if args.workers > 1:
            # 多进程模式（使用 Gunicorn）
            logger.info(f"Starting in production mode ({args.workers} workers, Gunicorn)...")
            config_file = os.path.join(os.path.dirname(__file__), 'server_init.py')
            cmd = (
                f"gunicorn serve.server_entry:app "
                f"-c {config_file} "
                f"-k uvicorn.workers.UvicornWorker "
                f"-b {args.host}:{args.port} "
                f"-w {args.workers} "
                f"--timeout {args.worker_timeout} "
                f"--keep-alive {args.timeout_keep_alive} "
                f"--worker-connections {args.limit_concurrency} "
                f"--backlog {args.backlog} "
                f"--env TSMRT_LOG_LEVEL={args.log_level} "
                f"--env TSMRT_LOG_DIR={args.log_dir} "
                f"--env TSMRT_LOG_OUTPUT={args.log_output}"
            )
            logger.info(f"Command: {cmd}")
            subprocess.run(cmd, shell=True, cwd=_project_root)
        else:
            # 单进程模式
            logger.info("Starting in production mode (single worker, Uvicorn)...")
            uvicorn.run(
                "serve.server_entry:app",
                host=args.host,
                port=args.port,
                workers=1,
                log_level=args.log_level,
                access_log=True,
                limit_concurrency=args.limit_concurrency,
                timeout_keep_alive=args.timeout_keep_alive,
                backlog=args.backlog
            )


def main() -> int:
    """主函数。

    Returns:
        退出代码，0 表示成功，非 0 表示失败
    """
    try:
        # 解析命令行参数
        args = prepare_server_args(sys.argv[1:])

        # 启动服务器
        launch_server(args)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...")
        return 0

    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        logging.error("Startup failed", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
