from typing import Literal, TypeAlias

import httpx
import pydantic
from typing_extensions import NotRequired, TypedDict

from ..utils.treestore import TreeStore


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    name: NotRequired[str]
    content: str
    pass


class ChatResponseDebugInfo(pydantic.BaseModel):
    prompt: list[ChatMessage]
    completions: list[ChatMessage]
    input_tokens: int
    output_tokens: int

    def __add__(self, other: "ChatResponseDebugInfo") -> "ChatResponseDebugInfo":
        return ChatResponseDebugInfo(
            prompt=self.prompt + other.prompt,
            completions=self.completions + other.completions,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @staticmethod
    def default() -> "ChatResponseDebugInfo":
        return ChatResponseDebugInfo(prompt=[], completions=[], input_tokens=0, output_tokens=0)

    pass


class ChatResponseOk(pydantic.BaseModel):
    status: Literal["ok"] = "ok"
    ok: list[str]
    err: None = None

    debug_info: ChatResponseDebugInfo
    backoff_tokens: None = None
    pass


class ChatResponseErr(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    status: Literal["err"] = "err"  # for recoverable errors only
    ok: None = None
    err: list[Exception]

    debug_info: ChatResponseDebugInfo | None
    backoff_tokens: int | None  # or negative. for retrying
    pass


ChatResponse: TypeAlias = ChatResponseOk | ChatResponseErr


inspect_llm = TreeStore[ChatResponseDebugInfo](key="inspect_llm")
"""Inspect inference engine calls using a tree store."""


class LLMEngine:
    """Abstract method for implementing LLM inference engines."""

    async def call(self, messages: list[ChatMessage], n: int = 1, temperature: float = 0.0) -> ChatResponse:
        ret = await self._call_impl(messages, n, temperature)
        if ret.debug_info:
            inspect_llm.put(ret.debug_info)
        return ret

    async def _call_impl(self, messages: list[ChatMessage], n: int, temperature: float) -> ChatResponse:
        # abstract method, override this
        raise NotImplementedError()

    pass


class LLMEngineMixin:
    """Abstract method for adding plugins to the LLM which either manipulate
    the input / output, or change how the engine works.

    How these mixins are used depend on the LLM engine's implementation."""

    def preprocess_prompt(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Pre-process the prompt before sending it to the LLM."""
        return messages

    def serialize_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert the prompt for use in text-only LLMs."""
        raise NotImplementedError()

    def create_http_client(self) -> httpx.AsyncClient:
        """Creates an HTTP client for sending network requests."""
        raise NotImplementedError()

    def postprocess_completion(self, response: str) -> str:
        """Post-process the completion after receiving it from the LLM."""
        return response

    pass
