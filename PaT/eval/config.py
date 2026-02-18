import pathlib
from typing import Literal

import pydantic
import toml
from cachetools import cached

from ..llm import LLMConfig
from ..utils.logger import LoggerEventType


class LangRTPythonConfig(pydantic.BaseModel):
    lang: Literal["python"] = "python"
    sandbox_root: str  # relative to config file's locating directory
    parallelism: int  # max workers
    macos_sandbox_bin: str | None = None
    python_bin: str | None = None
    pass


LangRTConfig = LangRTPythonConfig | LangRTPythonConfig


class LoggerConfig(pydantic.BaseModel):
    hide_event_types: list[LoggerEventType]
    pass


class MiscConfig(pydantic.BaseModel):
    wandb_enabled: bool
    wandb_project: str | None
    silent: bool
    default_proxy: str | Literal[False] | None
    pass


class EvalConfig(pydantic.BaseModel):
    """DQLLM configurations. File should not be uploaded to Git."""

    langrt: dict[str, LangRTConfig]  # langrt_key -> any langrt config
    llm: dict[str, LLMConfig]  # llm_key -> any llm config
    logger: LoggerConfig
    misc: MiscConfig
    pass


@cached(cache={})
def get_eval_config() -> EvalConfig:
    config_path = pathlib.Path(__file__).parent / "config.toml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_raw = f.read()
    cfg_dict = toml.loads(cfg_raw)
    cfg = EvalConfig.model_validate(cfg_dict)

    # patches & resolutions
    for langrt_cfg in cfg.langrt.values():
        if langrt_cfg.lang == "python":
            langrt_cfg.sandbox_root = (config_path.parent / langrt_cfg.sandbox_root).resolve().as_posix()

    return cfg
