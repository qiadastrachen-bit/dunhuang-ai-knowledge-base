# -*- coding: utf-8 -*-
"""
配置模块 — 集中管理所有可配置项
"""

import os
import yaml
from pathlib import Path

# 项目根目录（向上两级定位）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认配置文件路径
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(config_path: str = None) -> dict:
    """加载 YAML 配置文件，若未指定则使用默认路径。

    Args:
        config_path: 配置文件路径，默认为 config/settings.yaml。

    Returns:
        dict: 配置字典。
    """
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 将路径字段转为绝对路径（基于项目根目录）
    path_keys = ("data_dir", "pdf_dir", "cache_dir", "index_file", "docs_file", "metadata_file")
    for key in path_keys:
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = str(PROJECT_ROOT / cfg["paths"][key])

    return cfg


# 便捷访问：模块级默认配置
_config = None


def get_config() -> dict:
    """获取（惰性初始化的）全局配置单例。"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
