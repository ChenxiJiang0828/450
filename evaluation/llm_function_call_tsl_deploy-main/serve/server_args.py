from typing import List, Optional
import argparse
import logging

logger = logging.getLogger(__name__)


class ServerArgs:
    """服务初始化参数类"""
    # 服务配置
    config_path: Optional[str] = None
    config_format: str = "yaml"  # yaml, json, toml

    # 网络配置
    host: str = "0.0.0.0"
    port: int = 25988
    workers: int = 1

    # 日志配置
    log_level: str = "info"  # debug, info, warning, error, critical
    log_dir: str = "./logs"
    log_output: str = "console"  # console, file, both

    # 性能配置
    limit_concurrency: int = 200
    timeout_keep_alive: int = 3660
    worker_timeout: int = 300
    backlog: int = 2048

    # 其他配置
    reload: bool = False
    reload_dirs: Optional[List[str]] = None

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_format: str = "yaml",
        host: str = "0.0.0.0",
        port: int = 25988,
        workers: int = 1,
        log_level: str = "info",
        log_dir: str = "./logs",
        log_output: str = "console",
        limit_concurrency: int = 200,
        timeout_keep_alive: int = 3660,
        worker_timeout: int = 300,
        backlog: int = 2048,
        reload: bool = False,
        reload_dirs: Optional[List[str]] = None,
    ):
        self.config_path = config_path
        self.config_format = config_format
        self.host = host
        self.port = port
        self.workers = workers
        self.log_level = log_level
        self.log_dir = log_dir
        self.log_output = log_output
        self.limit_concurrency = limit_concurrency
        self.timeout_keep_alive = timeout_keep_alive
        self.worker_timeout = worker_timeout
        self.backlog = backlog
        self.reload = reload
        self.reload_dirs = reload_dirs if reload_dirs is not None else ["."]


def prepare_server_args(argv: List[str]) -> ServerArgs:
    """解析命令行参数，返回 ServerArgs 实例。

    Args:
        argv: 命令行参数列表，通常使用 sys.argv[1:]

    Returns:
        ServerArgs: 解析后的服务配置参数
    """
    parser = argparse.ArgumentParser(
        description="TSM Server - Things Specification Model Runtime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 服务配置
    config_group = parser.add_argument_group("服务配置")
    config_group.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径"  
    )
    config_group.add_argument(
        "--config-format",
        type=str,
        default="yaml",
        choices=["yaml", "json", "toml"],
        help="配置文件格式"
    )

    # 网络配置
    network_group = parser.add_argument_group("网络配置")
    network_group.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="服务监听地址"
    )
    network_group.add_argument(
        "--port", "-p",
        type=int,
        default=25988,
        help="服务监听端口"
    )
    network_group.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="工作进程数"
    )

    # 日志配置
    log_group = parser.add_argument_group("日志配置")
    log_group.add_argument(
        "--log-level", "-l",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="日志级别"
    )
    log_group.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="日志文件目录"
    )
    log_group.add_argument(
        "--log-output",
        type=str,
        default="both",
        choices=["console", "file", "both"],
        help="日志输出方式：console（仅控制台）、file（仅文件）、both（控制台和文件）"
    )

    # 性能配置
    performance_group = parser.add_argument_group("性能配置")
    performance_group.add_argument(
        "--limit-concurrency",
        type=int,
        default=200,
        help="最大并发连接数"
    )
    performance_group.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=3660,
        help="Keep-Alive 超时时间（秒）"
    )
    performance_group.add_argument(
        "--worker-timeout",
        type=int,
        default=300,
        help="Gunicorn worker 处理请求超时时间（秒，仅多 worker 生产模式生效）"
    )
    performance_group.add_argument(
        "--backlog",
        type=int,
        default=2048,
        help="连接队列长度"
    )

    # 其他配置
    other_group = parser.add_argument_group("其他配置")
    other_group.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载（开发环境）"
    )
    other_group.add_argument(
        "--reload-dirs",
        type=str,
        nargs="*",
        default=None,
        help="热重载监听目录"
    )

    # 解析命令行参数
    args = parser.parse_args(argv)

    # 创建基础 ServerArgs 实例（使用命令行参数）
    server_args = ServerArgs(
        config_path=args.config,
        config_format=args.config_format,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_output=args.log_output,
        limit_concurrency=args.limit_concurrency,
        timeout_keep_alive=args.timeout_keep_alive,
        worker_timeout=args.worker_timeout,
        backlog=args.backlog,
        reload=args.reload,
        reload_dirs=args.reload_dirs
    )

    return server_args


if __name__ == "__main__":
    # 测试示例
    import sys
    args_instance = prepare_server_args(sys.argv[1:])
    print("ServerArgs:")
    for field_name, field_value in args_instance.__dict__.items():
        print(f"  {field_name}: {field_value}")
