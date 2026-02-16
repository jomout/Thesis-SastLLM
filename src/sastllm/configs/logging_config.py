from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

SUPPORTED_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _load_yaml_logging(config_path: Path) -> Dict:
    """Return a dictConfig 'logging' section from YAML if available."""
    if not config_path.exists():
        raise RuntimeError(f"Path doesn't exist: {config_path}")
    
    with config_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    
    # Expect a 'log' block
    return _convert_simple_log_block(data.get("log"))


def _convert_simple_log_block(block: Optional[Dict]) -> Dict:
    """
    Allow a simple block like:
      log:
        level: INFO
        file: logs/app.log
        format: "%(asctime)s ..."
    """
    if not block:
        raise RuntimeError(
            """Block doesn't exist. Expected following format:
            log:
                level: INFO
                file: logs/app.log
                format: '%(asctime)s ...'
            """
        )

    level = str(block.get("level", "INFO")).upper()
    
    if level not in SUPPORTED_LEVELS:
        raise ValueError(f"Logging level not supported: {level}. Supported levels: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")
    
    fmt = block.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_path = block.get("file")

    handlers = ["console"] + (["file"] if file_path else [])
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": fmt}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {"handlers": handlers, "level": level, "propagate": False},
            "urllib3": {"level": "WARNING"},
            "botocore": {"level": "WARNING"},
        },
    }

    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        cfg["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": block.get("file_level", "INFO"),
            "formatter": "standard",
            "filename": str(file_path),
            "mode": "a",
            "encoding": "utf-8",
        }

    return cfg


def setup_logging(
    config_path: Union[str, Path] = "configs/base.yaml",
    default_level: str = "INFO",
) -> None:
    """
    Configure logging for the project.
    Priority:
      1) YAML config (if provided)
      2) Built-in fallback defaults
    """
    
    yaml_dict = _load_yaml_logging(Path(config_path))

    if yaml_dict:
        logging.config.dictConfig(yaml_dict)
        return

    # fallback config (if no YAML found)
    logging.basicConfig(
        level=default_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_logger(name: str = "sastllm") -> logging.Logger:
    return logging.getLogger(name)
