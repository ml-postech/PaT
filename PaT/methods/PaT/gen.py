import json

from ...langrt import LrtFunctionDef, LrtNode, LrtProgram
from ..shared import CodeGenContext, CodeGenJournal, CodeGenJournalist, CodeGenMethod
from .dfs_2pass import PaTDfs2Pass
from .gen_once import GenOncePrompt, PaT_gen_once, gen_collect_program
from .make_test import MakeTestPrompt, TestType, PaT_make_test
from .runner import RunnerCaseResult, PaT_runner


class PaTGen(CodeGenMethod):
    """PaT is an efficient method of generating code iteratively."""

    def __init__(
        self,
        # dfs mechanism
        dfs_max_depth: int,
        # divide
        divide_gen_prompt: GenOncePrompt,
        divide_temperature: float,
        divide_retries: int,
        # functional consistency
        fc_root_test_prompt: MakeTestPrompt | None,
        fc_root_sys_test_prompt: MakeTestPrompt | None,
        fc_branch_test_prompt: MakeTestPrompt | None,
        fc_branch_sys_test_prompt: MakeTestPrompt | None,
        fc_temperature: float,
        fc_retries: int,
        # conquer
        conquer_gen_prompt: GenOncePrompt,
        conquer_temperature: float,
        conquer_samples: int,
        conquer_min_samples: int,
        conquer_retries: int,
    ):
        self.dfs_max_depth = dfs_max_depth
        self.divide_gen_prompt = divide_gen_prompt
        self.divide_temperature = divide_temperature
        self.divide_retries = divide_retries
        self.fc_root_test_prompt = fc_root_test_prompt
        self.fc_root_sys_test_prompt = fc_root_sys_test_prompt
        self.fc_branch_test_prompt = fc_branch_test_prompt
        self.fc_branch_sys_test_prompt = fc_branch_sys_test_prompt
        self.fc_temperature = fc_temperature
        self.fc_retries = fc_retries
        self.conquer_gen_prompt = conquer_gen_prompt
        self.conquer_temperature = conquer_temperature
        self.conquer_samples = conquer_samples
        self.conquer_min_samples = conquer_min_samples
        self.conquer_retries = conquer_retries

    async def gen(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ) -> tuple[LrtProgram | None, CodeGenJournal]:
        ctx.log.in_scope(f"PaT[{func.name}]")
        init_ancestors = ancestors
        init_func = func

        dfs_2p = PaTDfs2Pass(
            ctx=ctx,
            opt_max_depth=self.dfs_max_depth,
            opt_refine_leaf=True,
            opt_patch_refine_root_docstring=True,
            gen_pass_1=lambda _ctx, _anc, _func, _desc: self._pass_1(
                ctx=_ctx, ancestors=_anc, func=_func, descendants=_desc
            ),
            gen_pass_2=lambda _ctx, _anc, _func, _desc: self._pass_2(
                ctx=_ctx,
                init_ancestors=init_ancestors,
                init_func=init_func,
                ancestors=_anc,
                func=_func,
                descendants=_desc,
            ),
            ancestors=ancestors,
            func=func,
            descendants=descendants,
        )

        _results, journal = await dfs_2p.run()
        if _results is None:
            return None, journal
        func_impl, rest_impl = _results

        program = gen_collect_program(ctx, ancestors, func_impl, rest_impl, descendants)
        ctx.log.code("python", "final result", ctx.lrt.fmt(program))
        return program, journal

    async def _pass_1(
        self, ctx: CodeGenContext, ancestors: list[LrtNode], func: LrtFunctionDef, descendants: list[LrtNode]
    ):
        samples, sj = await PaT_gen_once(
            ctx=ctx,
            opt_retries=self.divide_retries,
            opt_prompt=self.divide_gen_prompt,
            opt_temperature=self.divide_temperature,
            opt_samples=1,
            opt_min_samples=1,
            ancestors=ancestors,
            func=func,
            descendants=descendants,
            divide=True,
        )
        return samples[0] if samples else None, sj

    async def _pass_2(
        self,
        ctx: CodeGenContext,
        init_ancestors: list[LrtNode],
        init_func: LrtFunctionDef,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ):
        return await PaT_runner(
            ctx=ctx,
            opt_include_architect=False,
            opt_samples=self.conquer_samples,
            gen_pass=lambda _ctx, _anc, _func, _desc, _n: self._pass_2_gen(
                ctx=_ctx, ancestors=_anc, func=_func, descendants=_desc, n=_n
            ),
            test_pass=lambda _ctx, _anc, _func_samples: self._pass_2_test(
                ctx=_ctx,
                init_ancestors=init_ancestors,
                init_func=init_func,
                pass_func=func,
                pass_descendants=descendants,
                ancestors=_anc,
                func_samples=_func_samples,
            ),
            score_pass=runner_score_functional_similarity,
            ancestors=ancestors,
            func=func,
            descendants=descendants,
        )

    async def _pass_2_gen(
        self, ctx: CodeGenContext, ancestors: list[LrtNode], func: LrtFunctionDef, descendants: list[LrtNode], n: int
    ):
        return await PaT_gen_once(
            ctx=ctx,
            opt_prompt=self.conquer_gen_prompt,
            opt_temperature=self.conquer_temperature,
            opt_samples=self.conquer_samples,
            opt_min_samples=self.conquer_min_samples,
            opt_retries=self.conquer_retries,
            ancestors=ancestors,
            func=func,
            descendants=descendants,
            divide=False,
        )

    async def _pass_2_test(
        self,
        ctx: CodeGenContext,
        init_ancestors: list[LrtNode],
        init_func: LrtFunctionDef,
        pass_func: LrtFunctionDef,
        pass_descendants: list[LrtNode],
        ancestors: list[LrtNode],
        func_samples: list[LrtFunctionDef],
    ):
        ctx.log.in_scope(f"PaT_pass_2_test")
        _sj = CodeGenJournalist(ctx, "PaT_pass_2_test", (ancestors, pass_func, pass_descendants))
        is_root = is_root_func(init_ancestors, init_func, ancestors, pass_func)
        fc_test_prompt = self.fc_root_test_prompt if is_root else self.fc_branch_test_prompt
        fc_sys_test_prompt = self.fc_root_sys_test_prompt if is_root else self.fc_branch_sys_test_prompt
        tests: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []

        # extract unit tests from requirements
        if fc_sys_test_prompt is not None:
            sys_tests, _sj_ch = await PaT_make_test(
                ctx=ctx,
                opt_prompt=fc_sys_test_prompt,
                opt_temperature=self.fc_temperature,
                opt_retries=self.fc_retries,
                ancestors=ancestors,
                func_samples=amend_func_samples_for_sys_tests(ctx, pass_func, func_samples),
            )
            _sj.append(_sj_ch)
            tests.extend(sys_tests)

        # collect self-tests (no ground truth)
        if fc_test_prompt is not None:
            self_tests, _sj_ch = await PaT_make_test(
                ctx=ctx,
                opt_prompt=fc_test_prompt,
                opt_temperature=self.fc_temperature,
                opt_retries=self.fc_retries,
                ancestors=ancestors,
                func_samples=func_samples,
            )
            _sj.append(_sj_ch)
            tests.extend(self_tests)

        return tests, _sj.collect_test(tests)

    pass


def is_root_func(
    initial_ancestors: list[LrtNode],
    initial_func: LrtFunctionDef,
    ancestors: list[LrtNode],
    func: LrtFunctionDef,
) -> bool:
    """Detects whether we're at (in DFS) the root of the tree or not."""

    return len(ancestors) <= len(initial_ancestors) or func.name == initial_func.name


def runner_score_functional_similarity(results: list[list[RunnerCaseResult]]) -> list[float]:
    """The more how implementations agree on a consensus, the higher that test
    case is scored. A `power` can be applied to the score so that $n$ players
    who achieve the same results will each get a score of $n^{power}$."""

    programs = len(results)
    tests = max(len(ri) for ri in results)
    scores = [0.0 for _ri in results]

    def _verdict(rij: RunnerCaseResult | None) -> str | None:
        if rij is None:
            return None
        if not rij.ok:
            return None
        return json.dumps(rij.result, indent=None, ensure_ascii=True, sort_keys=False)

    for j in range(tests):
        rj = [(_verdict(results[i][j]) if j < len(results[i]) else None) for i in range(programs)]
        cnt_j: dict[str, int] = {}
        for rij in rj:
            if rij is not None:
                cnt_j[rij] = cnt_j.get(rij, 0) + 1
        # four players vote the same answer, then each gets 3 points
        # anyone crashing would get a penalty
        score_j = {rij: float(cnt - 1) for rij, cnt in cnt_j.items()}
        for i, rij in enumerate(rj):
            if rij is not None:
                scores[i] += score_j[rij]
            else:
                scores[i] -= 100.0
        pass

    # if any unit test failed, mark it as 'almost dead'
    for i in range(programs):
        for j in range(tests):
            case = results[i][j]
            if case.test_type == TestType.unittest and not case.ok:
                scores[i] -= 10000.0
                break

    return scores


def amend_func_samples_for_sys_tests(
    ctx: CodeGenContext, ref_func: LrtFunctionDef, func_samples: list[LrtFunctionDef]
) -> list[LrtFunctionDef]:
    """Keep original docstrings (esp. for root functions) so that we can
    extract unit tests from them."""
    results: list[LrtFunctionDef] = []
    for func in func_samples:
        func = func.model_copy()
        func.docstring = ref_func.docstring or func.docstring
        func = ctx.lrt._parse.fmt_function_def(func)
        results.append(func)
    return results
