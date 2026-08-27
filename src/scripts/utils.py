from os import getenv
from pathlib import Path
from typing import Literal

import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI

from argus.configs import get_logger

logger = get_logger(__name__)


def load_yaml(config_path: str | Path = "configs/base.yaml") -> dict:
    """
    Load a YAML config file and return its contents as a dict.

    Raises:
        FileNotFoundError if the file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        logger.exception("Configuration file was not found", path=str(path))
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        config = yaml.safe_load(f) or {}
    logger.debug("Loaded YAML configuration", path=str(path), top_level_keys=sorted(config))
    return config


def get_model(
    processor: Literal[
        "snippet_processor",
        "file_processor",
        "cluster_processor",
        "repository_processor",
    ],
    config_path: str | Path = "configs/llms.yaml",
):
    """
    Load YAML and return an instantiated LangChain Chat model.

    Supports:
      - host: "google" -> ChatGoogleGenerativeAI
      - host: "openai" -> ChatOpenAI

    YAML example:
      models:
        llm:
          host: "google"
          name: "gemini-2.5-flash"
          params:
            temperature: 0.2
            max_tokens: 8192
    """

    PROCESSORS = {
        "snippet_processor",
    }

    if processor not in PROCESSORS:
        raise ValueError(f"Invalid argument processor: {processor}. Should be: 'snippet_processor'.")

    config = load_yaml(config_path)
    try:
        llm_cfg = config["models"][processor]
    except KeyError as e:
        raise KeyError(f"Missing 'models.{processor}' section in {config_path}") from e

    host = str(llm_cfg.get("host", "")).lower()
    name = str(llm_cfg.get("name", ""))
    if not host or not name:
        raise ValueError(f"Both 'models.{processor}.host' and 'models.{processor}.name' must be set in YAML.")

    params = llm_cfg.get("params", {})
    logger.info("Building configured LLM", processor=processor, host=host, model=name)

    if host == "google":
        # expects GOOGLE_API_KEY to be available in the environment or configured globally
        return ChatGoogleGenerativeAI(model=name, **params)
    if host == "openai":
        # expects OPENAI_API_KEY to be available in the environment or configured globally
        return OpenAI(model=name)
    if host == "issel":
        from argus.utils import CustomLLM

        endpoint_url = getenv("ENDPOINT_URL", "")
        access_token = getenv("ACCESS_TOKEN", "")
        if not endpoint_url or not access_token:
            raise ValueError(f"'models.{processor}.params' must include 'endpoint_url' and 'access_token' for host 'issel'.")
        return CustomLLM(endpoint_url=endpoint_url, access_token=access_token)

    logger.exception("Unsupported configured LLM host", processor=processor, host=host)
    raise ValueError(f"Unsupported LLM host: {host!r}. Supported: 'google', 'openai', 'issel'.")


def get_classification_config(
    mode: Literal["train", "test"],
    config_path: str | Path = "configs/classification.yaml",
) -> tuple[str | Path, dict]:
    MODES = {"train", "test"}

    if mode not in MODES:
        raise ValueError(f"Invalid argument mode: {mode}. Should be one of: 'train', 'test'.")

    config = load_yaml(config_path)
    try:
        cls_cfg = config["classification"][mode]
    except KeyError as e:
        raise KeyError(f"Missing 'classification.{mode}' section in {config_path}") from e

    params = cls_cfg.get("params", {})
    if mode == "train":
        save_model_dir = cls_cfg.get("save_model_dir")
        if not save_model_dir:
            raise ValueError(f"'classification.{mode}.save_model_dir' must be set in YAML.")
        return save_model_dir, params
    else:
        load_model_dir = cls_cfg.get("load_model_dir")
        if not load_model_dir:
            raise ValueError(f"'classification.{mode}.load_model_dir' must be set in YAML.")
        return load_model_dir, params
