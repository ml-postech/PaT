import httpx

from ..utils.types import guard_never
from .types import ChatMessage, LLMEngineMixin


class MergedLLMMixin(LLMEngineMixin):
    """Sequentially combine multiple LLM mixins."""

    def __init__(self, mixins: list[LLMEngineMixin]):
        self._mixins = mixins

    def preprocess_prompt(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        for mixin in self._mixins:
            messages = mixin.preprocess_prompt(messages)
        return messages

    def serialize_prompt(self, messages: list[ChatMessage]) -> str:
        for mixin in reversed(self._mixins):  # precedence for overrides
            try:
                return mixin.serialize_prompt(messages)
            except NotImplementedError:
                pass
        raise NotImplementedError()

    def create_http_client(self) -> httpx.AsyncClient:
        for mixin in reversed(self._mixins):
            try:
                return mixin.create_http_client()
            except NotImplementedError:
                pass
        raise NotImplementedError()

    def postprocess_completion(self, response: str) -> str:
        for mixin in self._mixins:
            response = mixin.postprocess_completion(response)
        return response


class TextCompletionLLMMixin(LLMEngineMixin):
    """Legacy LLMs only support text completion APIs and do not accept
    dictionary values as input."""

    def serialize_prompt(self, messages: list[ChatMessage]) -> str:
        ret: list[str] = []
        for message in messages:
            ret.append("<|im_start|>" + message["role"] + "\n")
            ret.append(message["content"] + "\n")
            ret.append("<|im_end|>\n")
        ret.append("<|im_start|>assistant\n")
        return "".join(ret)


class DefaultHttpClientLLMMixin(LLMEngineMixin):
    """Default HTTP client for LLM engines sending requests."""

    def __init__(self, proxy: str | None):
        self._proxy = proxy

    def create_http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(proxy=self._proxy)


class MockFewShotPromptLLMMixin(LLMEngineMixin):
    """In later GPT versions, few-shot prompts may be specified in the
    `role=system,name=assistant` form. Earlier versions and open-source models,
    however, may not support this. This mixin will ensure such protocols are
    emulated in a compatible way."""

    def preprocess_prompt(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        mocked: list[ChatMessage] = []
        for message in messages:
            if message["role"] == "system":
                name = message.get("name", None)
                if name is not None:
                    assert name == "assistant" or name == "user"
                    mocked.append({"role": name, "content": message["content"]})  # few-shot examples
                else:
                    mocked.append(message)  # system message
            else:
                mocked.append(message)  # typical user / assistant
        return mocked


class MockSystemRoleLLMMixin(LLMEngineMixin):
    """In certain open-source models, the `role=system` message may not be
    recognized. This mixin prepends system messages to all user prompts."""

    def preprocess_prompt(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        system_prompts = [msg["content"] for msg in messages if msg["role"] == "system"]
        mocked: list[ChatMessage] = []
        for message in messages:
            if message["role"] == "system":
                pass
            elif message["role"] == "user":
                message = message.copy()
                content = system_prompts + [message["content"]]
                content = [blk.strip("\n") for blk in content]
                message["content"] = "\n\n".join(content)
                mocked.append(message)
            elif message["role"] == "assistant":
                mocked.append(message)
            else:
                guard_never(message["role"])
        return mocked


class CodeModelFormatLLMMixin(LLMEngineMixin):
    """In certain code models code blocks are not enclosed in triple-tildes
    (```), but rather in named brackets([PYTHON]...[/PYTHON])."""

    def postprocess_completion(self, response: str) -> str:
        languages = ["C", "C++", "PYTHON", "JAVASCRIPT", "JS", "TYPESCRIPT", "TS"]
        for lang in languages:
            if f"[{lang}]" in response and f"[/{lang}]" in response:
                response = response.replace(f"[{lang}]", f"```{lang.lower()}").replace(f"[/{lang}]", "```")
        return response
