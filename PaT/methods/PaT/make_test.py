import enum
import random
from typing import Any, Callable, Coroutine, Generator, Iterable

from ...langrt import CodeBlock, LrtFunctionDef, LrtNode, LrtProgram, SymbolName
from ...llm import ChatMessage
from ..shared import CodeGenContext, CodeGenJournal, CodeGenJournalist
from .gen_once import _gen_once_fast_fmt


class TestType(enum.Enum):
    unittest = "unittest"  # very firm (nearly ground truth!) system unit tests
    call = "call"  # llm generated 'call-only' test cases
    assertion = "assertion"  # llm generated 'assertion' test cases
    pass


class MakeTestPrompt:
    """A prompt generator for `make_test`. These can be used to either extract,
    derive or generate test cases for a given problem. They can be evaluated
    upon with the `runner`. There are multiple ways to configure the prompt
    behavior, depending on the rarity of requirements:

      - [Most cases] `get_few_shots?`, `get_few_shot_ids()?`,
        `get_keep_ancestors?`, `get_next_turn?`, `parse_tests`.
      - [Rare] `wrap_prompt_iter?`, `parse_tests`.
      - [SSR] `make_tests`.

    The last case covers all possible scenarios and overrides all modifications
    to the subclass."""

    get_skip_closures: bool = False
    """OVERRIDE: When the function to-be-tested is pure, we may skip generating
    tests for them since just calling them is enough."""

    get_few_shots: list[ChatMessage] = []
    """Required by `wrap_prompt_iter`. Few shot demonstrations, the first turn
    is always a "system" message, and the rest comes in assistant and user
    conversation pairs."""

    def get_few_shot_ids(self) -> Iterable[list[int]]:
        """Required by `wrap_prompt_iter`. See `GenOncePrompt.get_few_shot_ids`,
        specifies which # of few shots will be used (e.g. default [0,1,2,3],
        [0,1,2], [0,1], [0], [])."""

        ls = list(self.get_few_shots)
        for i in range((len(ls) - 1) // 2, -1, -1):
            yield list(range(i))
        return

    get_keep_ancestors: bool = False
    """Required by `wrap_prompt_iter`. Whether or not to include depending
    functions in the code."""

    def get_next_turn(self, ctx: CodeGenContext, func_name: SymbolName, lang: str, code: CodeBlock) -> ChatMessage:
        """Required by `wrap_prompt_iter`. Format the next-turn user prompt:
        - `func_name`: `foo`
        - `lang`: `python`
        - `code`: `def foo(a: int, b: int) -> int:\n    raise ...\n\n\n...`"""
        raise NotImplementedError()

    def wrap_prompt_iter(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
    ) -> Generator[list[ChatMessage], None, None]:
        """OVERRIDE?: The generic way to generate prompts. In each iteration,
        we apply token backoff as requested by `get_few_shot_ids`.

        Note: This function is shadowed by `OVERRIDE_make_tests`."""

        func_name, code = _gen_once_fast_fmt(ctx, ancestors, func, [], self.get_keep_ancestors or False)

        p_system, *p_few_shots = self.get_few_shots
        p_next_turn = self.get_next_turn(ctx, func_name, ctx.lrt.lang, code)
        for nshots in self.get_few_shot_ids():
            few_shots = sum(([p_few_shots[2 * i], p_few_shots[2 * i + 1]] for i in nshots), [])
            yield [p_system, *few_shots, p_next_turn]
        return

    def parse_tests(
        self, ctx: CodeGenContext, func: LrtFunctionDef, raw_message: str
    ) -> Iterable[tuple[TestType, LrtProgram, LrtFunctionDef]]:
        """Extract test cases from a given message (just plain text). We require
        that all tests take exactly 1 argument as the seed. For example:

        - to_be_tested / func: `def foo(a: int, b: str) -> float: ...`
        - llm_returns: `foo(1, "hello")  # case A
                        foo(0, "world")  # case B`
        - program [0]: `def test_case_A(_seed: int) -> float:
                            return foo(1, "hello")  # case A`
        - program [1]: `def test_case_B(_seed: int) -> float:
                            return foo(0, "world")  # case B`
        - program [*]: `def test_main(seed: int) -> float:
                            random.seed(seed)
                            a = random.randint(1, 100)
                            b = random.choice(["hello", "world"])
                            return foo(a, b)`

        You may use the wrapper to inject specific environment hooks, e.g. stdio
        wrappers, initialization functions, etc.

        Note: This function is shadowed by `OVERRIDE_make_tests`."""
        raise NotImplementedError()

    async def make_tests(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
    ) -> Iterable[tuple[TestType, LrtProgram, LrtFunctionDef]]:
        """OVERRIDE: Directly create tests for a given function. This short
        circuits `wrap_prompt_iter` and `parse_tests`."""
        raise NotImplementedError()

    pass


def _make_random_sig() -> str:
    return hex(random.randrange(0, 2**24))[2:].rjust(6, "0")


MakeTestSig = Callable[
    [CodeGenContext, list[LrtNode], list[LrtFunctionDef]],
    Coroutine[Any, Any, tuple[list[tuple[TestType, LrtProgram, LrtFunctionDef]], CodeGenJournal]],
]
"""(ctx, ancestors[], impls[]) -> tests[]"""


async def PaT_make_test(
    ctx: CodeGenContext,
    opt_prompt: MakeTestPrompt,
    opt_temperature: float,
    opt_retries: int,
    ancestors: list[LrtNode],
    func_samples: list[LrtFunctionDef],
) -> tuple[list[tuple[TestType, LrtProgram, LrtFunctionDef]], CodeGenJournal]:
    """Creates a test method that generates one (or more) functions that either:

     1. calls the target function with generated arguments, or
     2. test the target function with generated arguments and compare it with
        generated results.

    The prompt should return multiple such functions, each in a different
    Markdown code block. The prompt will further be responsible for extracting
    the generated functions and wrap them up accordingly."""

    assert len(func_samples) >= 1, "func_samples must contain at least one function"
    func = random.choice(func_samples)
    ctx.log.in_scope(f"test_calling[{func.name}(...)]")
    _sj = CodeGenJournalist(ctx, "PaT_make_test", (ancestors, func, []))

    # closures are self-consistent: they should produce identical results
    # whatsoever
    if opt_prompt.get_skip_closures and len(func.args) == 0:
        results = _test_fixed_point(ctx, ancestors, func)
        return results, _sj.collect_test(results)

    results: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
    if opt_prompt.__class__.make_tests is MakeTestPrompt.make_tests:
        # use default implementation
        for _retry in range(1, opt_retries + 1):
            # apply nshot backoff during llm request
            responses = []
            for history in opt_prompt.wrap_prompt_iter(ctx, ancestors, func):
                # we use n=1 because sampling doesn't make sense for testing. llm
                # generally returns what it believes to be the best test(s).
                next_result = await ctx.llm.call(history, n=1, temperature=opt_temperature)
                if next_result.ok:
                    responses = next_result.ok
                    break
                elif next_result.backoff_tokens:
                    continue
                else:
                    raise (err := (next_result.err or [])[-1]) from err
            if len(responses) < 1:
                ctx.log.string(f"attempt {_retry} failed: llm did not return result")
                continue
            response = responses[0]

            # respect the prompt's test extraction algorithm
            parsed = opt_prompt.parse_tests(ctx, func, response)
            if not parsed:
                ctx.log.string(f"attempt {_retry} failed: no tests generated")
                continue
            results.extend(parsed)
            break
    else:
        # use custom implementation
        results.extend(await opt_prompt.make_tests(ctx, ancestors, func))

    for i, (_type, program, _func) in enumerate(results):
        code = ctx.lrt.pretty_fmt(program)
        ctx.log.code("python", f"generated test {i} of {len(results)}", code)
    return results, _sj.collect_test(results)


def wrap_test_case_stdio(ctx: CodeGenContext, func_name: str, input: str, output: str | None) -> LrtFunctionDef:
    """Wraps a test case with standard I/O hooks that provides stdin and
    captures stdout (which can be compared against the expected output)."""

    the_input = repr(input)
    the_output = repr(output)  # None / '...'

    if ctx.lrt.lang == "python":
        code = f"""
def _test_{func_name}_{_make_random_sig()}(seed: int):
    # from the outside world
    _exp_input = {the_input}
    _exp_output = {the_output}
    # hook stdio
    import io
    import sys
    the_stdin = sys.stdin
    the_stdout = sys.stdout
    hook_in = io.StringIO(_exp_input)
    hook_out = io.StringIO()
    try:
        sys.stdin = hook_in
        sys.stdout = hook_out
        real_ret = {func_name}()
        real_stdout = hook_out.getvalue()
    finally:
        sys.stdin = the_stdin
        sys.stdout = the_stdout
    # do compare
    if _exp_output is not None:
        _cmp_output = ' '.join(_exp_output.split())
        _cmp_stdout = ' '.join(real_stdout.split())
        assert _cmp_stdout == _cmp_output, repr(_cmp_stdout) + " != " + repr(_cmp_output)
    else:
        return real_stdout
"""
    else:
        raise NotImplementedError(ctx.lrt.lang)

    try:
        program = ctx.lrt.parse(module=(), code=code)
    except Exception as e:
        ctx.log.error("failed to parse test case")
        ctx.log.code("python", "failed test case", code)
        raise e from e
    return program.cast_as(LrtFunctionDef)


def wrap_test_case_expr(ctx: CodeGenContext, func_name: str, expr: str) -> LrtFunctionDef:
    """Wraps a test case with a single (?) expression that ought to be able to
    be evaluated and compared in the current environment."""

    if ctx.lrt.lang == "python":
        code = f"def _test_{func_name}_{_make_random_sig()}(seed: int):\n"
        for line in expr.split("\n"):
            code += f"    {line}\n"
    else:
        raise NotImplementedError(ctx.lrt.lang)

    try:
        program = ctx.lrt.parse(module=(), code=code)
    except Exception as e:
        ctx.log.error("failed to parse test case")
        ctx.log.code("python", "failed test case", code)
        raise e from e
    return program.cast_as(LrtFunctionDef)


def _test_fixed_point(
    ctx: CodeGenContext, ancestors: list[LrtNode], func: LrtFunctionDef
) -> list[tuple[TestType, LrtProgram, LrtFunctionDef]]:
    """Tests fixed point functions, e.g. `f: () -> T`."""

    assert len(func.args) == 0, "fixed point tests can only be applied to closures"
    wrapped = f"def _test_fixed_point_{_make_random_sig()}(_seed: int):\n    return {func.name}()\n"
    program = ctx.lrt.parse(module=(), code=wrapped)
    fn = program.cast_as(LrtFunctionDef)
    return [(TestType.call, program, fn)]
