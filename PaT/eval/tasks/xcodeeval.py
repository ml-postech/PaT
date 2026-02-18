import json
import pathlib
import random
import re
from typing import Any, Generator

import pydantic
from typing_extensions import TypedDict

from ...eval.types import CodeGenEvalTasks
from ...langrt import LrtFunctionDef, LrtSolution
from ...methods.PaT.defaults import DEFAULT_IMPORTS
from ...methods.PaT.make_test import TestType
from ...methods.PaT.prompts.injected import PaTInjectedStdioPrompt
from ...methods.shared import CodeGenContext, CodeGenJournal
from ...utils.strings import compare_strings_cf
from ..types import EvalResult


class InputXCE(TypedDict):
    task_id: str  # 32 character hex string
    problem: dict  # _CfProblemDescription
    test_cases: list[dict]  # CfUnitTest


class VerdictXCE(TypedDict):
    # verdict: ok -> 1.0, fail -> 0.0, no code -> None
    ok: bool
    reason: str
    stdout: str
    result: Any


class _CfProblemDescription(pydantic.BaseModel):
    description: str
    input_from: str | None  # always "standard input" or "стандартный ввод"
    output_to: str | None  # always "standard output" or "стандартный вывод"
    time_limit: str  # e.g. "3 seconds"
    memory_limit: str | None  # e.g. "256 megabytes"
    input_spec: str | None
    output_spec: str | None
    notes: str | None
    sample_inputs: list[str]
    sample_outputs: list[str]
    tags: list[str]  # e.g. ["number theory", "greedy"]
    src_uid: str  # 32 character hex string
    difficulty: int | None  # cf difficulty rating
    created_at: int  # epoch


class _CfUnitTest(pydantic.BaseModel):
    input: str  # the only input. may end with '...'
    output: list[str]  # some of the possible correct outputs


class xCodeEvalEvalTasks(CodeGenEvalTasks[InputXCE, VerdictXCE]):
    """xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval
    https://github.com/ntunlp/xCodeEval
    https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval"""

    name = "xCodeEval"

    def __init__(self, dir_path: pathlib.Path, samples: int | None):
        self._dir_path = dir_path
        self._take_samples = samples

    def iter(self) -> Generator[tuple[str, InputXCE], None, None]:
        # descriptions are stored on one file
        problems: dict[str, _CfProblemDescription] = {}
        with open(pathlib.Path(self._dir_path) / "problem_descriptions.jsonl", "r", encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
            for p in lines:
                vp = self._cf_load_problem_desc(p)
                problems[vp.src_uid] = vp

        # but test cases are on another file
        test_cases: dict[str, list[_CfUnitTest]] = {}
        with open(pathlib.Path(self._dir_path) / "unittest_db.json", "r", encoding="utf-8") as f:
            _test_cases: dict[str, list[dict]] = json.load(f)
            for task_id, cases in _test_cases.items():
                test_cases[task_id] = [_CfUnitTest.model_validate(c) for c in cases]

        # group them together
        items: dict[str, InputXCE] = {}
        for task_id in sorted(problems.keys()):
            problem = problems[task_id]
            cases = test_cases.get(task_id, [])
            task = self._assemble_task(task_id, problem, cases)
            if task is not None:
                items[task_id] = task
        # and sample (reproducible)
        if self._take_samples is not None:
            shuffle_keys = list(items.keys())
            rand = random.Random()
            rand.seed(42)
            rand.shuffle(shuffle_keys)
            shuffle_keys = shuffle_keys[: self._take_samples]
            items = {k: v for k, v in items.items() if k in shuffle_keys}

        for task_id, item in items.items():
            the_id = task_id.split("/")[-1]  # 'HumanEval/233' -> '233'
            yield the_id, item
        return

    def _cf_load_problem_desc(self, p: dict) -> _CfProblemDescription:
        vp = _CfProblemDescription.model_validate(p)
        vp.description = _sanitize_html(vp.description)
        vp.input_spec = _sanitize_html(vp.input_spec) if vp.input_spec else None
        vp.output_spec = _sanitize_html(vp.output_spec) if vp.output_spec else None
        vp.notes = _sanitize_html(vp.notes) if vp.notes else None
        vp.sample_inputs = [_sanitize_html(i) for i in vp.sample_inputs]
        vp.sample_outputs = [_sanitize_html(i) for i in vp.sample_outputs]
        return vp

    def _assemble_task(self, task_id: str, problem: _CfProblemDescription, cases: list[_CfUnitTest]) -> InputXCE | None:
        if not cases:
            return None
        # skip russian probs
        cyrillic_cnt = sum(1 for c in problem.description if "\u0400" <= c <= "\u04ff")
        cyrillic_rate = cyrillic_cnt / len(problem.description)
        if cyrillic_rate > 0.5:
            return None
        # any extra-long input will render in probably-not-correct answers
        cleaned_cases = [self._cf_sanitize_test_case(c) for c in cases]
        good_cases = [c for c in cleaned_cases if c is not None]
        if len(good_cases) < len(cleaned_cases) or not good_cases:
            return None
        # no good sample
        if len(problem.sample_inputs) != len(problem.sample_outputs):
            return None
        # fix note
        if problem.notes is not None and problem.notes.lower().startswith("note"):
            problem.notes = problem.notes[4:].strip()
        # assemble
        return {
            "task_id": task_id,
            "problem": problem.model_dump(),
            "test_cases": [i.model_dump() for i in good_cases],
        }

    def _cf_sanitize_test_case(self, case: _CfUnitTest) -> _CfUnitTest | None:
        # make sure cases are not truncated
        outputs = [i for i in case.output if not i.endswith("...")]
        if case.input.endswith("...") or not outputs:
            return None
        return _CfUnitTest(input=case.input, output=outputs)

    def debug_fmt(self, task: InputXCE) -> dict:
        j = dict(task)
        j["test_cases"] = f"<{len(task['test_cases'])} test cases>"
        return j

    async def execute(self, ctx, method, task_id, task) -> tuple[EvalResult[InputXCE, VerdictXCE], CodeGenJournal]:
        # gap: default imports, stdio docstring formatter
        assert ctx.lrt.lang == "python"  # TODO: support other languages

        problem = _CfProblemDescription.model_validate(task["problem"])
        sections: list[tuple[str, str | None]] = [
            ("Description", problem.description),
            ("Input", problem.input_spec),
            ("Output", problem.output_spec),
        ]
        for i, (inp, out) in enumerate(zip(problem.sample_inputs, problem.sample_outputs)):
            sections.append((f"Sample Input {i + 1}", "```\n" + inp.strip() + "\n```"))
            sections.append((f"Sample Output {i + 1}", "```\n" + out.strip() + "\n```"))
        sections.append(("Notes", problem.notes))
        prompt_func = self._cf_fmt_problem(
            ctx=ctx,
            function_name="main",
            time_limit=problem.time_limit,
            memory_limit=problem.memory_limit,
            input_from=problem.input_from,
            output_to=problem.output_to,
            sections=sections,
        )

        # inject sample inputs / outputs
        sample_tests: list[tuple[TestType, str, str]] = []
        for inp, out in zip(problem.sample_inputs, problem.sample_outputs):
            sample_tests.append((TestType.unittest, inp, out))
        PaTInjectedStdioPrompt.inject(sample_tests)

        program, _sj = await method.gen(ctx, [], prompt_func, [])
        if program is None:
            raise ValueError("failed to generate program")
        code = ctx.lrt.pretty_fmt(program)

        return {
            "id": task_id,
            "task": task,
            "code": code,
            "_code_error": None,
            "_code_tree": None,
            "verdict": None,
            "_verdict_info": None,
        }, _sj

    def _cf_fmt_problem(
        self,
        ctx: CodeGenContext,
        function_name: str,
        time_limit: str,
        memory_limit: str | None,
        input_from: str | None,
        output_to: str | None,
        sections: list[tuple[str, str | None]],
    ) -> LrtFunctionDef:
        def _indented(s: str) -> str:
            lines = s.strip().split("\n")
            indented = [f"    {line}\n" if line.strip() else "\n" for line in lines]
            block = "".join(indented)
            # no unnecessary escapes please
            block = block.replace("'''", '"""')
            block = block.replace("$$$", "$")
            return block

        blk = ""
        blk += f"def {function_name}() -> None:\n"
        blk += f"    r'''\n"
        blk += f"    time limit per test: {time_limit}\n"
        if memory_limit:
            blk += f"    memory limit per test: {memory_limit}\n"
        blk += f"    input: {input_from or 'standard input'}\n"
        blk += f"    output: {output_to or 'standard output'}\n"
        for title, content in sections:
            if not content:
                continue
            blk += f"\n"
            blk += f"    ## {title}\n"
            blk += f"\n"
            blk += _indented(content)
        blk += f"    '''"
        blk += f"\n"
        blk += f"    raise NotImplementedError()\n"

        return ctx.lrt.parse(module=(), code=blk).cast_as(LrtFunctionDef)

    async def judge(self, ctx, result) -> EvalResult[InputXCE, VerdictXCE]:
        problem = _CfProblemDescription.model_validate(result["task"]["problem"])
        req_tests = [_CfUnitTest.model_validate(i) for i in result["task"]["test_cases"]]
        time_limit = self._parse_cf_time_limit(problem.time_limit) or 2.0

        draft_code = result["code"]
        if draft_code is None:
            result["verdict"] = None
            result["_verdict_info"] = {
                "ok": False,
                "reason": "Compile error: no code to execute",
                "stdout": "",
                "result": None,
            }
            return result
        draft_code = DEFAULT_IMPORTS + "\n\n\n" + draft_code

        exec_program = ctx.lrt.parse(module=(), code=draft_code)
        exec_fn = exec_program.find(LrtFunctionDef, "main")
        if not exec_fn:
            result["verdict"] = None
            result["_verdict_info"] = {
                "ok": False,
                "reason": "Compile error: no main function",
                "stdout": "",
                "result": None,
            }
            return result

        exec_solution = LrtSolution(modules=[exec_program])
        print(f"================ {result['id']} ================")
        judge_result = await codeforces_judge(
            ctx=ctx,
            sample_tests=list(zip(problem.sample_inputs, problem.sample_outputs)),
            system_tests=[(i.input, i.output) for i in req_tests],
            exec_solution=exec_solution,
            exec_module_name=exec_program.module,
            exec_fn_name=exec_fn.name,
            timeout=time_limit,
        )
        print((judge_result[0] if judge_result else "Accepted").ljust(48, " "))

        if judge_result is not None:
            reason, stdout, ret = judge_result
            result["verdict"] = 0.0
            result["_verdict_info"] = {"ok": False, "reason": reason, "stdout": stdout, "result": ret}
            return result
        result["verdict"] = 1.0
        result["_verdict_info"] = {"ok": True, "reason": "Accepted", "stdout": "", "result": None}
        return result

    def _parse_cf_time_limit(self, time_limit: str) -> float | None:
        ss = re.findall(r"(\d+(\.\d+)?)", time_limit)
        if not ss:
            return None
        return float(ss[0][0])

    pass


async def codeforces_judge(
    ctx: CodeGenContext,
    sample_tests: list[tuple[str, str]],
    system_tests: list[tuple[str, list[str]]],
    exec_solution: LrtSolution,
    exec_module_name: tuple[str, ...],
    exec_fn_name: str,
    timeout: float,
) -> tuple[str, str, Any] | None:
    """Judge I/O problems in the Codeforces way.
    - sample_tests: (stdin, stdout)[], since only one pair is provided in the
                    problem description;
    - system_tests: (stdin, stdout[])[], where matching any stdout would
                    consist a correct answer.
    - exec_*: function to execute.
    - Returns: (err_msg, stdout, result) or None if ok."""

    cnt_tot_tests = len(sample_tests) + len(system_tests)
    for i, (inp, out) in enumerate(sample_tests):
        print(f"running sample test {i + 1} of {cnt_tot_tests}...      ", end="\r")
        res = await ctx.lrt.run_solution(
            exec_solution, exec_module_name, exec_fn_name, args=[], kwargs={}, stdin=inp, timeout=timeout
        )
        if res.duration >= timeout:
            return f"Time limit exceeded on sample test {i + 1}", "", res.result
        if not res.ok:
            return f"Runtime error on sample test {i + 1}: ({res.ret_code}) {res.error}", "", res.result
        judge = compare_strings_cf([out], res.stdout or "")
        if judge is not None:
            return f"Wrong answer on sample test {i + 1}: {judge}", res.stdout or "", res.result
        pass

    for i, (inp, outs) in enumerate(system_tests):
        print(f"running system test {len(sample_tests) + i + 1} of {cnt_tot_tests}...      ", end="\r")
        res = await ctx.lrt.run_solution(
            exec_solution, exec_module_name, exec_fn_name, args=[], kwargs={}, stdin=inp, timeout=timeout
        )
        if res.duration >= timeout:
            return f"Time limit exceeded on system test {i + 1}", "", res.result
        if not res.ok:
            return f"Runtime error on system test {i + 1}: ({res.ret_code}) {res.error}", "", res.result
        judge = compare_strings_cf(outs, res.stdout or "")
        if judge is not None:
            return f"Wrong answer on system test {i + 1}: {judge}", res.stdout or "", res.result
        pass

    return None


def _sanitize_html(text: str) -> str:
    patterns: list[tuple[str, str]] = [
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&amp;", "&"),
        # haven't seen them appeared but have them escaped anyway (preventive)
        ("&quot;", '"'),
        ("&apos;", "'"),
    ]
    for pattern, repl in patterns:
        text = text.replace(pattern, repl)
    return text
