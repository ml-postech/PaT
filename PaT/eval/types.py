from typing import Generator, Generic, TypeVar, cast

# https://stackoverflow.com/questions/71944154/variadic-generic-type-for-pythons-typeddict
from typing_extensions import TypedDict

from ..langrt import CodeBlock
from ..methods.shared import CodeGenContext, CodeGenJournal, CodeGenMethod

_Input = TypeVar("_Input")

_Verdict = TypeVar("_Verdict")


class EvalResult(TypedDict, Generic[_Input, _Verdict]):
    id: str
    task: _Input

    code: CodeBlock | None
    _code_error: str | None
    _code_tree: None

    verdict: float | None
    _verdict_info: _Verdict | None

    pass


class CodeGenEvalTasks(Generic[_Input, _Verdict]):
    """The base class for all (code gen) task loaders. Examples are HumanEval,
    MATHS, xCodeEval, etc. A scheduler will use concrete implementations to
    evaluate the performance of a code gen method."""

    name: str = "(?)"

    def iter(self) -> Generator[tuple[str, _Input], None, None]:
        """function* () -> ...(id, task)"""
        raise NotImplementedError()

    def debug_fmt(self, task: _Input) -> dict:
        """OVERRIDE?: Remove redundant or excessive information from the 'task'
        so that it would not look too verbose on the console / log."""
        return cast(dict, task)

    async def execute(
        self,
        ctx: CodeGenContext,
        method: CodeGenMethod,
        task_id: str,
        task: _Input,
    ) -> tuple[EvalResult[_Input, _Verdict], CodeGenJournal]:
        """Execute method on a certain task."""
        raise NotImplementedError()

    async def judge(
        self,
        ctx: CodeGenContext,
        result: EvalResult[_Input, _Verdict],
    ) -> EvalResult[_Input, _Verdict]:
        """Judge the result of the method. You can make changes to the result
        in-place since we are not referencing the original result."""
        raise NotImplementedError()

    pass
