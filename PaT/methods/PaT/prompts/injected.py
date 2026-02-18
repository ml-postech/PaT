from typing import Iterable

from ....langrt import LrtFunctionDef, LrtProgram
from ....utils.pyctx import PyCtx
from ..make_test import MakeTestPrompt, TestType, wrap_test_case_expr, wrap_test_case_stdio

_InjectedTestsStore = tuple[str, TestType, LrtProgram, LrtFunctionDef]
"""(key, test_type, program, function)"""

_InjectedStdioStore = tuple[str, TestType, str, str]
"""(key, test_type, stdin, stdout)"""

_InjectedExprStore = tuple[str, TestType, str]
"""(key, test_type, expr_code)"""


class PaTInjectedTestsPrompt(MakeTestPrompt):
    """Provide system tests / sample tests via PyCtx injection. Only calls
    happened under the scope in which `.inject(...)` happens are capable of
    receiving injected test cases."""

    _test_store = PyCtx[list[_InjectedTestsStore]]("PaT_injected_tests_store")

    def __init__(self, key: str = "default"):
        self._key = key

    get_skip_closures = False

    @classmethod
    def inject(
        cls, tests: list[tuple[TestType, LrtProgram, LrtFunctionDef]], key: str = "default", offset: int = 0
    ) -> None:
        for test_type, program, fn in tests:
            cls._test_store.update(
                lambda pre: (pre[0] if pre else []) + [(key, test_type, program, fn)], offset=1 + offset
            )
        return

    async def make_tests(self, ctx, ancestors, func) -> Iterable[tuple[TestType, LrtProgram, LrtFunctionDef]]:
        raw_tests = sum(self._test_store.get(), [])
        tests = [(test_type, program, fn) for key, test_type, program, fn in raw_tests if key == self._key]
        return tests

    pass


class PaTInjectedStdioPrompt(MakeTestPrompt):
    """Provide stdin / stdout as tests via PyCtx injection. Must specify test
    cases via `.inject(...)` directly within a parent call."""

    _test_store = PyCtx[list[_InjectedStdioStore]]("PaT_injected_stdio_store")

    def __init__(self, key: str = "default"):
        self._key = key

    get_skip_closures = False

    @classmethod
    def inject(cls, tests: list[tuple[TestType, str, str]], key: str = "default", offset: int = 0) -> None:
        for test_type, stdin, stdout in tests:
            cls._test_store.update(
                lambda pre: (pre[0] if pre else []) + [(key, test_type, stdin, stdout)], offset=1 + offset
            )
        return

    async def make_tests(self, ctx, ancestors, func) -> Iterable[tuple[TestType, LrtProgram, LrtFunctionDef]]:
        raw_tests = sum(self._test_store.get(), [])
        tests: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
        for key, test_type, stdin, stdout in raw_tests:
            if key != self._key:
                continue
            test_fn = wrap_test_case_stdio(ctx, func.name, stdin, stdout)
            test_prog = LrtProgram(module=(), nodes=[test_fn])
            tests.append((test_type, test_prog, test_fn))
        return tests

    pass


class PaTInjectedExprPrompt(MakeTestPrompt):
    """Provide one-line expression tests via PyCtx injection. Specify test
    cases via `.inject(...)` directly within a parent call."""

    _test_store = PyCtx[list[_InjectedExprStore]]("PaT_injected_expr_store")

    def __init__(self, key: str = "default"):
        self._key = key

    get_skip_closures = False

    @classmethod
    def inject(cls, tests: list[tuple[TestType, str]], key: str = "default", offset: int = 0) -> None:
        for test_type, expr_code in tests:
            cls._test_store.update(
                lambda pre: (pre[0] if pre else []) + [(key, test_type, expr_code)], offset=1 + offset
            )
        return

    async def make_tests(self, ctx, ancestors, func) -> Iterable[tuple[TestType, LrtProgram, LrtFunctionDef]]:
        raw_tests = sum(self._test_store.get(), [])
        tests: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
        for key, test_type, expr_code in raw_tests:
            if key != self._key:
                continue
            test_fn = wrap_test_case_expr(ctx, f"INJECTED_{func.name}", expr_code)
            test_prog = LrtProgram(module=(), nodes=[*ancestors, func, test_fn])
            tests.append((test_type, test_prog, test_fn))
        return tests

    pass
