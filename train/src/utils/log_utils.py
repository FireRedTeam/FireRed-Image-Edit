# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
# Copyright (c) REDEdit. Logging utilities for multi-process / multi-node training.
"""
分布式训练友好的彩色日志工具。

- 日志行首带 [R<rank>/<world_size> L<local_rank>/<local_size>] 前缀，便于多进程/多机下区分来源。
- 按级别着色：INFO=绿，WARNING=黄，ERROR=红，DEBUG=灰。
- 通过环境变量 REDEDIT_NO_COLOR=1 可关闭颜色；非 TTY 时自动关闭颜色。
- 通过环境变量 REDEDIT_LOG_LEVEL=DEBUG 可显示 DEBUG 日志（默认 INFO）。
"""

import logging
import os
import sys
from typing import Optional

# ANSI 颜色码（终端可读）
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"

# 级别 -> 颜色
_LEVEL_COLORS = {
    logging.DEBUG: _DIM,
    logging.INFO: _GREEN,
    logging.WARNING: _YELLOW,
    logging.ERROR: _RED,
    logging.CRITICAL: _BOLD + _RED,
}


def _env_int(key: str, default: int = -1) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def get_default_log_level() -> int:
    """
    从环境变量 REDEDIT_LOG_LEVEL 读取日志级别，便于开启 DEBUG。
    可选: DEBUG, INFO, WARNING, ERROR。默认 INFO。
    """
    name = (os.environ.get("REDEDIT_LOG_LEVEL") or "").strip().upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }.get(name, logging.INFO)


def get_dist_prefix() -> str:
    """
    从环境变量获取分布式训练前缀，便于多进程/多机下区分日志来源。
    格式: [R<rank>/<world_size> L<local_rank>/<local_size>]
    torchrun / accelerate launch 会设置 RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE。
    """
    rank = _env_int("RANK", -1)
    local_rank = _env_int("LOCAL_RANK", -1)
    world_size = _env_int("WORLD_SIZE", 1)
    local_size = _env_int("LOCAL_WORLD_SIZE", 1)
    if rank < 0 and local_rank < 0:
        return ""
    if rank < 0:
        rank = local_rank
    if world_size <= 0:
        world_size = 1
    if local_size <= 0:
        local_size = 1
    return f"[R{rank}/{world_size} L{local_rank}/{local_size}] "


def _use_color() -> bool:
    """是否使用彩色输出：非 TTY 或 REDEDIT_NO_COLOR=1 时不用颜色。"""
    if os.environ.get("REDEDIT_NO_COLOR", "").strip() in ("1", "true", "yes"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class DistributedColoredFormatter(logging.Formatter):
    """
    带分布式前缀和级别颜色的 Formatter。
    多进程/多机时每行带 [R?/N L?/M] 前缀，便于 grep 和阅读。
    """

    def __init__(self, use_color: Optional[bool] = None, dist_prefix: Optional[str] = None):
        super().__init__(
            datefmt="%m/%d %H:%M:%S",
        )
        self._use_color = use_color if use_color is not None else _use_color()
        self._dist_prefix = dist_prefix if dist_prefix is not None else get_dist_prefix()

    def format(self, record: logging.LogRecord) -> str:
        # 级别颜色（先填充再着色，保证多进程对齐）
        levelname_raw = (record.levelname + " " * 8)[:8]
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, _RESET)
            reset = _RESET
            levelname = f"{color}{levelname_raw}{reset}"
        else:
            levelname = levelname_raw
        # 统一前缀: 时间 | 级别 | [R?/N L?/M] name | 消息
        msg = record.getMessage()
        prefix = f"{self._dist_prefix}" if self._dist_prefix else ""
        return (
            f"{self.formatTime(record, datefmt='%m/%d %H:%M:%S')} | "
            f"{levelname} | {prefix}{record.name} | {msg}"
        )


def get_logger(
    name: str,
    level: Optional[int] = None,
    use_color: Optional[bool] = None,
    dist_prefix: Optional[str] = None,
) -> logging.Logger:
    """
    获取带分布式前缀和颜色的 logger。
    建议各模块使用同一 name 风格（如 __name__），便于过滤。

    Args:
        name: logger 名称，如 __name__。
        level: 日志级别，None 时使用环境变量 REDEDIT_LOG_LEVEL（默认 INFO）。
        use_color: 是否着色，None 表示自动判断。
        dist_prefix: 分布式前缀，None 表示从环境变量自动生成。
    """
    if level is None:
        level = get_default_log_level()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            DistributedColoredFormatter(use_color=use_color, dist_prefix=dist_prefix)
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def log_once(logger: logging.Logger, level: int, msg: str, *args, **kwargs):
    """仅在主进程（LOCAL_RANK 0）记录一次。RANK 未设置时始终记录。"""
    local_rank = _env_int("LOCAL_RANK", 0)
    if local_rank <= 0:
        logger.log(level, msg, *args, **kwargs)
