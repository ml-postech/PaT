from typing import Callable, Literal

import pydantic

from ..utils.types import guard_never
from .mixins import (
    CodeModelFormatLLMMixin,
    DefaultHttpClientLLMMixin,
    MergedLLMMixin,
    MockFewShotPromptLLMMixin,
    MockSystemRoleLLMMixin,
    TextCompletionLLMMixin,
)
from .openai_gpt_engine import OpenAIGptLLMEngine
from .types import LLMEngine, LLMEngineMixin


class LLMConfig(pydantic.BaseModel):
    """Local configuration for LLMs. Subject to change as new engines are being
    added gradually."""

    kind: Literal["gpt"]

    # [gpt llm engine config]
    endpoint: str
    key: pydantic.SecretStr
    api_type: Literal["open_ai", "azure"]
    api_version: str | None = None  # Azure OpenAI only
    api_dialect: Literal["completions", "chat_completions"]
    engine: str | None = None  # model switching for Azure OpenAI
    model: str | None = None  # model switch for OpenAI API

    # [client behavior]
    opt_max_output_tokens: int | None = None
    opt_min_output_tokens: int | None = None
    opt_stop_tokens: list[str] | None = None  # up to 4 sequences. hotfix for certain text models
    opt_max_sampling: int | None = None  # parallelism supported by model
    opt_retry_timeout: float | None = None  # 429 prevention, None to use default

    # [mixins]
    # proxy URL for bad connectivity issues
    mixin_proxy: str | None = None
    # True to fix model not supporting system roles (Special few-shot role,
    # uses {"role": "system", "name": "user|assistant"} for few-shot examples)
    mixin_mock_few_shot_prompt: bool = False
    # True to fix model not supporting "role"="system" (the system prompt will
    # be concatenated to the first message)
    mixin_mock_system_role: bool = True
    # True if model using other code block delimiters rather than markdown code
    # blocks (e.g. `[PYTHON]...[/PYTHON]`)
    mixin_code_model_format: bool = False

    pass


def create_llm_engine(
    config: LLMConfig,
    opt_on_error: Callable[[str], None] = lambda _: None,
    opt_silent: bool = False,
) -> LLMEngine:
    """Initialize an LLM engine based on the configuration."""

    mixins: list[LLMEngineMixin | None] = [
        TextCompletionLLMMixin(),
        DefaultHttpClientLLMMixin(config.mixin_proxy),
        MockFewShotPromptLLMMixin() if config.mixin_mock_few_shot_prompt else None,
        MockSystemRoleLLMMixin() if config.mixin_mock_system_role else None,
        CodeModelFormatLLMMixin() if config.mixin_code_model_format else None,
    ]
    mixin = MergedLLMMixin([m for m in mixins if m is not None])

    if config.kind == "gpt":
        return OpenAIGptLLMEngine(
            mixin=mixin,
            # endpoint settings
            endpoint=config.endpoint,
            key=config.key.get_secret_value(),
            api_type=config.api_type,
            api_version=config.api_version,
            api_dialect=config.api_dialect,
            engine=config.engine,
            model=config.model,
            # client data behavior
            opt_max_output_tokens=config.opt_max_output_tokens,
            opt_min_output_tokens=config.opt_min_output_tokens,
            opt_stop_tokens=config.opt_stop_tokens or [],
            opt_max_sampling=config.opt_max_sampling,
            opt_retry_timeout=config.opt_retry_timeout,
            # misc & exceptions
            opt_on_error=opt_on_error,
            opt_silent=opt_silent,
        )
    guard_never(config.kind)
