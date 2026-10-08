"""
结构化日志配置。

使用标准库 logging，
输出带时间、级别、模块、消息的结构化日志。
"""

import logging
import sys
from datetime import datetime


# 日志格式
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s:%(lineno)d | %(message)s"
)

# 日期格式
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
) -> None:
    """
    初始化全局日志配置。

    参数：
        level: 日志级别，默认 INFO
    """

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有 handler，避免重复输出
    root_logger.handlers.clear()

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )
    )

    root_logger.addHandler(console_handler)

    # 降低第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志器。

    参数：
        name: 日志器名称，通常用 __name__

    返回：
        Logger 实例
    """
    return logging.getLogger(name)
