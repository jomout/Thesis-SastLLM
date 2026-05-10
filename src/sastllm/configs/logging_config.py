from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
import yaml
from structlog.typing import EventDict, Processor, WrappedLogger

SUPPORTED_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _validate_level(level: str, *, field_name: str = "level") -> str:
    level = str(level).upper()

    if level not in SUPPORTED_LEVELS:
        raise ValueError(f"Logging {field_name} not supported: {level}. Supported levels: {', '.join(sorted(SUPPORTED_LEVELS))}")

    return level


def _uppercase_log_level(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    level = event_dict.get("level")

    if isinstance(level, str):
        event_dict["level"] = level.upper()

    return event_dict


def _load_yaml_logging(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise RuntimeError(f"Path doesn't exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return _convert_simple_log_block(data.get("log"))


def _make_console_renderer() -> Processor:
    styles = structlog.dev.ConsoleRenderer.get_default_level_styles(colors=True)

    # structlog's default level styles are keyed by lowercase names.
    # Since we uppercase the level field, duplicate styles under uppercase keys.
    styles.update(
        {
            "DEBUG": styles.get("debug", ""),
            "INFO": styles.get("info", ""),
            "WARNING": styles.get("warning", ""),
            "ERROR": styles.get("error", ""),
            "CRITICAL": styles.get("critical", ""),
        }
    )

    return structlog.dev.ConsoleRenderer(
        colors=True,
        force_colors=True,
        pad_event=0,
        pad_level=False,
        level_styles=styles,
    )


def _make_json_renderer() -> Processor:
    return structlog.processors.JSONRenderer()


def _convert_simple_log_block(block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Expected YAML:

    log:
      level: INFO
      file: logs/sastllm.log
      file_level: INFO
      max_bytes: 10485760
      backup_count: 5

    Console output:
      - colored
      - uppercase levels
      - no level padding

    File output:
      - JSON Lines
      - rotating file handler
    """
    if not block:
        raise RuntimeError(
            """Block doesn't exist. Expected following format:
log:
  level: INFO
  file: logs/sastllm.log
  file_level: INFO
  max_bytes: 10485760
  backup_count: 5
"""
        )

    level = _validate_level(block.get("level", "INFO"))
    file_level = _validate_level(block.get("file_level", level), field_name="file_level")

    file_path = block.get("file")
    max_bytes = int(block.get("max_bytes", 10 * 1024 * 1024))
    backup_count = int(block.get("backup_count", 5))

    handlers = ["console"]

    cfg: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": _make_console_renderer(),
                "foreign_pre_chain": [
                    structlog.contextvars.merge_contextvars,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    _uppercase_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                ],
            },
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": _make_json_renderer(),
                "foreign_pre_chain": [
                    structlog.contextvars.merge_contextvars,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    _uppercase_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                ],
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": handlers,
            "level": level,
        },
        "loggers": {
            "urllib3": {
                "level": "WARNING",
                "propagate": True,
            },
            "botocore": {
                "level": "WARNING",
                "propagate": True,
            },
        },
    }

    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        handlers.append("file")

        cfg["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": file_level,
            "formatter": "json",
            "filename": str(file_path),
            "mode": "a",
            "encoding": "utf-8",
            "maxBytes": max_bytes,
            "backupCount": backup_count,
        }

    return cfg


def _configure_structlog() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            _uppercase_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def setup_logging(
    config_path: Union[str, Path] = "configs/base.yaml",
    default_level: str = "INFO",
) -> None:
    default_level = _validate_level(default_level)
    path = Path(config_path)

    if path.exists():
        cfg = _load_yaml_logging(path)
    else:
        cfg = _convert_simple_log_block(
            {
                "level": default_level,
            }
        )

    logging.config.dictConfig(cfg)
    _configure_structlog()


def get_logger(name: str = "sastllm") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
