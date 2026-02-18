import asyncio
import re
from typing import Any, Callable, Coroutine, Literal, TypeVar, cast

import openai
import rich

from ..utils.logger import Console
from ..utils.types import coalesce, guard_never, not_null
from .types import (
    ChatMessage,
    ChatResponse,
    ChatResponseDebugInfo,
    ChatResponseErr,
    ChatResponseOk,
    LLMEngine,
    LLMEngineMixin,
)

T = TypeVar("T")


class OpenAIGptLLMEngine(LLMEngine):
    def __init__(
        self,
        mixin: LLMEngineMixin,
        # endpoint settings
        endpoint: str,
        key: str,
        api_type: Literal["open_ai", "azure"],
        api_version: str | None,  # Azure OpenAI only
        api_dialect: Literal["completions", "chat_completions"],
        engine: str | None,  # model switching for Azure OpenAI
        model: str | None,  # model switch for OpenAI API
        # client data behavior
        opt_max_output_tokens: int | None,
        opt_min_output_tokens: int | None,  # after backoff
        opt_stop_tokens: list[str],  # up to 4 sequences. hotfix for certain text models
        opt_max_sampling: int | None,  # parallelism
        opt_retry_timeout: float | None,
        # misc & exceptions
        opt_on_error: Callable[[str], None],
        opt_silent: bool,
    ):
        # endpoint settings
        self._endpoint = endpoint
        self._key = key
        self._api_type: Literal["open_ai", "azure"] = api_type
        self._api_version = api_version
        self._api_dialect: Literal["completions", "chat_completions"] = api_dialect
        self._engine = engine
        self._model = model
        # client data behavior
        self._opt_max_output_tokens = opt_max_output_tokens
        self._opt_min_output_tokens = opt_min_output_tokens
        self._opt_stop_tokens = opt_stop_tokens
        self._opt_max_sampling = opt_max_sampling
        self._opt_retry_timeout = opt_retry_timeout if opt_retry_timeout is not None else 5.0  # in seconds
        # misc & exceptions
        self._opt_on_error = opt_on_error
        self._opt_silent = opt_silent
        # setup
        self._mixin = mixin
        self._console = rich.get_console()
        self._client = self._get_client()

    async def _call_impl(self, messages: list[ChatMessage], n: int = 1, temperature: float = 0.0) -> ChatResponse:
        return await self._call_with_sampling(
            n=n,
            func=lambda n: self._call_with_retry(
                func=lambda max_tokens: self._call_raw_once(
                    messages=messages,
                    n=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            ),
        )

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._api_type == "open_ai":
            assert self._api_version is None
            assert self._engine is None
            assert self._model is not None
            client = self._mixin.create_http_client()
            return openai.AsyncOpenAI(
                api_key=self._key,
                base_url=self._endpoint,
                http_client=client,
                max_retries=1,
            )
        elif self._api_type == "azure":
            assert self._api_version is not None
            assert self._engine is not None
            assert self._model is None
            client = self._mixin.create_http_client()
            return openai.AsyncAzureOpenAI(
                api_key=self._key,
                base_url=f"{self._endpoint.strip('/')}/openai",  # lib v1 bug
                api_version=self._api_version,
                http_client=client,
                max_retries=1,
            )
        else:
            guard_never(self._api_type)

    async def _call_raw_once(
        self,
        messages: list[ChatMessage],
        n: int,
        max_tokens: int,
        temperature: float,
    ) -> ChatResponse:
        if self._api_dialect == "completions":
            fmt_messages = self._mixin.preprocess_prompt(messages)
            prompt = self._mixin.serialize_prompt(fmt_messages)
            res = await self._client.completions.create(
                model=not_null(coalesce(self._model, self._engine)),
                prompt=prompt,
                max_tokens=max_tokens,
                n=n,
                temperature=temperature,
                stop=self._opt_stop_tokens,
            )
            completions = [choice.text for choice in res.choices]
            return ChatResponseOk(
                ok=completions,
                debug_info=ChatResponseDebugInfo(
                    prompt=messages,
                    completions=[{"role": "assistant", "content": c} for c in completions],
                    input_tokens=res.usage.prompt_tokens if res.usage else 0,
                    output_tokens=res.usage.completion_tokens if res.usage else 0,
                ),
            )
        elif self._api_dialect == "chat_completions":
            fmt_messages = self._mixin.preprocess_prompt(messages)
            res = await self._client.chat.completions.create(
                model=not_null(coalesce(self._model, self._engine)),
                messages=cast(Any, messages),
                max_tokens=max_tokens,
                n=n,
                temperature=temperature,
                stop=self._opt_stop_tokens,
            )
            completions = [choice.message.content for choice in res.choices if choice.message.content]
            return ChatResponseOk(
                ok=completions,
                debug_info=ChatResponseDebugInfo(
                    prompt=messages,
                    completions=[{"role": "assistant", "content": c} for c in completions],
                    input_tokens=res.usage.prompt_tokens if res.usage else 0,
                    output_tokens=res.usage.completion_tokens if res.usage else 0,
                ),
            )
        guard_never(self._api_dialect)

    async def _call_with_sampling(
        self, n: int, func: Callable[[int], Coroutine[Any, Any, ChatResponse]]
    ) -> ChatResponse:
        """(N) -> (n_i) -> response"""

        tot_response = ChatResponseOk(ok=[], debug_info=ChatResponseDebugInfo.default())
        chunk = self._opt_max_sampling or n
        for lb in range(0, n, chunk):
            rb = min(lb + chunk, n)  # exclusive
            upd = await func(rb - lb)
            if isinstance(upd, ChatResponseOk):
                tot_response = ChatResponseOk(
                    ok=tot_response.ok + upd.ok,
                    debug_info=tot_response.debug_info + upd.debug_info,
                )
            elif isinstance(upd, ChatResponseErr):
                return upd
            else:
                guard_never(upd)
        return tot_response

    async def _call_with_retry(
        self, func: Callable[[int], Coroutine[Any, Any, T]]
    ) -> T | ChatResponseOk | ChatResponseErr:
        """() -> (max_tokens) -> response"""

        exceptions: list[Exception] = []
        _retry_cnt = 1
        _retry_cnt_max_tokens = 1
        max_tokens = self._opt_max_output_tokens or 4096
        min_output_tokens = self._opt_min_output_tokens or 512

        with Console.get_status("[bold green]LLM call[/bold green]", silent=self._opt_silent) as status:
            while True:
                _retry_cnt += 1
                try:
                    c_max_tokens = max(max_tokens, min_output_tokens)
                    res = await func(c_max_tokens)
                    return res

                # fatal errors that will render this engine completely unusable
                # under current config (known for sure)
                except openai.AuthenticationError as err:
                    self._opt_on_error(f"fatal error: cannot authenticate with inference endpoint: {err}")
                    raise err from err
                except openai.PermissionDeniedError as err:
                    self._opt_on_error(f"fatal error: permission denied for inference endpoint: {err}")
                    raise err from err
                except openai.BadRequestError as err:
                    if "content_filter" in str(err):
                        self._opt_on_error("fatal error: cannot call chat completions due to content filter trigger")
                        raise err from err  # not recoverable

                    new_max_tokens = self._backoff_tokens(err, max_tokens)
                    if new_max_tokens is None:
                        raise err from err  # not recoverable

                    exceptions.append(err)
                    if _retry_cnt_max_tokens > 4:  # normally should work on first try
                        backoff_tokens = max_tokens - min_output_tokens
                        backoff_tokens = backoff_tokens if backoff_tokens < 0 else None
                        return ChatResponseErr(
                            err=exceptions, debug_info=None, backoff_tokens=backoff_tokens
                        )  # n-shot back-off

                    status.update(
                        f"[bold green]LLM call, retry {_retry_cnt_max_tokens} ({max_tokens} -> {new_max_tokens} tokens)...[/bold green]"
                    )
                    max_tokens = new_max_tokens
                    _retry_cnt_max_tokens += 1
                    continue

                # soft errors that can be retried
                except openai.RateLimitError as _:
                    status.update(f"[bold green]LLM call, retry {_retry_cnt} (rate limit)...[/bold green]")
                    if (tle := self._opt_retry_timeout) > 0:
                        await asyncio.sleep(tle)
                    continue
                except openai.APITimeoutError as _:
                    status.update(f"[bold green]LLM call, retry {_retry_cnt} (timeout)...[/bold green]")
                    if (tle := self._opt_retry_timeout) > 0:
                        await asyncio.sleep(tle)
                    continue
                except openai.APIConnectionError as _:
                    status.update(f"[bold green]LLM call, retry {_retry_cnt} (connection error)...[/bold green]")
                    if (tle := self._opt_retry_timeout) > 0:
                        await asyncio.sleep(tle)
                    continue

                # we don't know what's going on.
                except Exception as err:
                    return ChatResponseErr(err=[err], debug_info=None, backoff_tokens=None)
        pass

    def _backoff_tokens(self, err: openai.BadRequestError, prev_output_tokens: int) -> int | None:
        # openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, you requested 32391 tokens (16007 in the messages, 16384 in the completion). Please reduce the length of the messages or completion.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
        # openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 46055 tokens. Please reduce the length of the messages.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
        # openai.BadRequestError: Error code: 400 - {'error': {'message': 'max_tokens is too large: 4738. This model supports at most 4096 completion tokens, whereas you provided 4738.', 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': None}}
        # vLLM would only display the latter as there's no input / output token difference
        msg = err.message
        tk_max_tokens = re.findall(r"maximum context length is (\d+) tokens", msg)
        tk_max_completions = re.findall(r"at most (\d+) completion tokens", msg)
        tk_requested = re.findall(r"requested (\d+) tokens", msg)

        if tk_max_tokens and tk_requested:
            backoff = int(tk_requested[0]) - int(tk_max_tokens[0])
            return prev_output_tokens - backoff - 128
        elif tk_max_tokens:
            return int(tk_max_tokens[0]) - 1
        elif tk_max_completions:
            return int(tk_max_completions[0])
        # no fallback policy
        return prev_output_tokens - 512

    pass
