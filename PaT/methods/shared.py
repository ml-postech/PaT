import dataclasses
from typing import Any, get_args

from typing_extensions import NotRequired, TypedDict

from ..langrt import CodeBlock, LangRT, LrtFunctionDef, LrtNode, LrtProgram, SymbolName
from ..llm import ChatMessage, LLMConfig, LLMEngine, create_llm_engine, inspect_llm
from ..utils.logger import Logger, LoggerEventType


@dataclasses.dataclass
class CodeGenContext:
    """Code generators are ran within this context."""

    log: Logger
    llm: LLMEngine
    llm_planner: LLMEngine
    lrt: LangRT

    cfg_silent: bool
    pass


def create_code_gen_context(
    llm_config: LLMConfig,
    lrt: LangRT,
    silent: bool = True,
) -> CodeGenContext:
    """Creates a code generation context in the easiest way possible."""

    return CodeGenContext(
        log=Logger(hide_event_types=list(get_args(LoggerEventType)) if silent else ["chat_history", "exec_result"]),
        llm=create_llm_engine(llm_config, opt_silent=silent),
        lrt=lrt,
        cfg_silent=silent,
    )


class _CodeGenJournal_LLMCall(TypedDict):
    prompt: list[ChatMessage]
    completions: list[ChatMessage]
    input_tokens: int  # 0 when error
    output_tokens: int  # 0 when error


class CodeGenJournal(TypedDict):
    kind: str
    error: str | None
    input: str
    _input_ancestors: str
    _input_func: str
    _input_descendants: str
    # https://github.com/pydantic/pydantic/issues/7953
    # metagpt specifically requires that pydantic = 2.5.3.
    # f__k metagpt.
    # this needs to be "CodeGenJournal".
    children: list[Any]
    result: str | list[str]
    _result_func: str | list[str]
    _result_descendants: str | list[str]
    _result_type: NotRequired[str | list[str]]
    llm_calls: list[_CodeGenJournal_LLMCall]
    pass


class CodeGenJournalist:
    """A (not) simple hack that makes saving logs easier."""

    def __init__(self, ctx: CodeGenContext, kind: str, input: tuple[list[LrtNode], LrtFunctionDef, list[LrtNode]]):
        self._ctx = ctx
        self._kind = kind
        self._input = input
        self._llm_trap = inspect_llm.trap(offset=1)
        self._children: list[CodeGenJournal] = []

    def append(self, child: CodeGenJournal | None) -> None:
        if child is not None:
            self._children.append(child)

    def collect_gen(self, result: tuple[LrtFunctionDef, list[LrtNode]], **kwargs: Any) -> CodeGenJournal:
        """Mark this as a stage on the generation tree and report 1 layer up."""
        journal = self.__generate()
        journal["result"] = self._ctx.lrt.fmt([result[0], *result[1]])
        journal["_result_func"] = self._ctx.lrt.fmt(result[0])
        journal["_result_descendants"] = self._ctx.lrt.fmt(result[1])
        journal.update(kwargs)  # type: ignore
        return journal

    def collect_gen_multi(self, results: list[tuple[LrtFunctionDef, list[LrtNode]]], **kwargs: Any) -> CodeGenJournal:
        journal = self.__generate()
        journal["result"] = [self._ctx.lrt.fmt([result[0], *result[1]]) for result in results]
        journal["_result_func"] = [self._ctx.lrt.fmt(result[0]) for result in results]
        journal["_result_descendants"] = [self._ctx.lrt.fmt(result[1]) for result in results]
        journal.update(kwargs)  # type: ignore
        return journal

    def collect_test(self, samples: list[tuple[Any, LrtProgram, LrtFunctionDef]], **kwargs: Any) -> CodeGenJournal:
        """Mark this as a stage on the generation tree and report 1 layer up."""
        journal = self.__generate()
        journal["result"] = [self._ctx.lrt.fmt(sample[1]) for sample in samples]
        journal["_result_type"] = [sample[0] for sample in samples]  # type: ignore
        journal["_result_func"] = [self._ctx.lrt.fmt(sample[2]) for sample in samples]
        journal["_result_descendants"] = ["" for _ in samples]
        journal.update(kwargs)  # type: ignore
        return journal

    def collect_err(self, error: str, **kwargs: Any) -> CodeGenJournal:
        """Mark this as a stage on the generation tree but report error up."""
        journal = self.__generate()
        journal["error"] = error
        journal.update(kwargs)  # type: ignore
        return journal

    @staticmethod
    def just_error(kind: str, error: str) -> CodeGenJournal:
        return {
            "kind": kind,
            "error": error,
            "input": "",
            "_input_ancestors": "",
            "_input_func": "",
            "_input_descendants": "",
            "children": [],
            "result": "",
            "_result_func": "",
            "_result_descendants": "",
            "llm_calls": [],
        }

    def __generate(self) -> CodeGenJournal:
        input = self._input
        llm_calls: list[_CodeGenJournal_LLMCall] = [
            {
                "prompt": src.prompt,
                "completions": src.completions,
                "input_tokens": src.input_tokens,
                "output_tokens": src.output_tokens,
            }
            for src in self._llm_trap.gather()
        ]
        return {
            "kind": self._kind,
            "error": None,
            "input": self._ctx.lrt.fmt([*input[0], input[1], *input[2]]),
            "_input_ancestors": self._ctx.lrt.fmt(input[0]),
            "_input_func": self._ctx.lrt.fmt(input[1]),
            "_input_descendants": self._ctx.lrt.fmt(input[2]),
            "children": self._children,
            "result": "",
            "_result_func": "",
            "_result_descendants": "",
            "llm_calls": llm_calls,
        }

    pass


class CodeGenMethod:
    """The abstract base class for how to generate code from a function
    definition. Here we define:

      - `ancestors`: nodes that call the implemented `func`
      - `func`: to implement this function in this process
      - `descendants`: while writing `func`, these nodes may be leveraged
      - `n`: the (max) number of samples to generate

    Returns a tuple of `program ~= (func_impl, ...rest), log`:

      - `program`: the collection of nodes that includes:
          - `func_impl`: the implementation of `func`
          - `...rest`: and any remaining children of `func_impl`
      - `log`: may store debug information for the generation process
    """

    async def gen(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ) -> tuple[LrtProgram | None, CodeGenJournal]:
        """Implement `func` in the context of `ancestors` and `descendants`."""
        raise NotImplementedError()

    async def gen_simple(
        self,
        ctx: CodeGenContext,
        code: CodeBlock,
        entry_func: SymbolName,
    ) -> tuple[LrtProgram | None, CodeGenJournal]:
        """Implement `entry_func(...)` from raw `code`."""

        program = ctx.lrt.parse(module=(), code=code)
        func = program.find(LrtFunctionDef, entry_func)
        if func is None:
            raise ValueError(f"function `{entry_func}(...)` not found in code")
        rest = program.excluding(func)
        return await self.gen(ctx, rest, func, [])

    pass
