# -*- coding: utf-8 -*-
"""
日志工具 — 提供统一的日志配置
"""

import logging
import sys


def setup_logger(name: str = "dunhuang_kb", level: int = logging.INFO) -> logging.Logger:
    """创建并返回格式化的 Logger 实例。

    Args:
        name: Logger 名称。
        level: 日志级别，默认 INFO。

    Returns:
        logging.Logger: 配置好的 Logger。
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # 避免重复添加 handler

    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger
