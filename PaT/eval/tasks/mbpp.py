import json
import pathlib
import random
from typing import Generator

from typing_extensions import TypedDict

from ...langrt import LrtFunctionDef, LrtNode
from ...methods.PaT.defaults import DEFAULT_IMPORTS
from ...methods.PaT.make_test import _make_random_sig
from ...methods.shared import CodeGenJournal
from ..types import CodeGenEvalTasks, EvalResult


class InputMBPP(TypedDict):
    """Sanitized version (`MBPP_typed.json`) derived from the original."""

    source_file: str
    task_id: str
    prompt: str
    code: str
    entry_point: str
    test_imports: list[str]
    test_list: list[str]


class VerdictMBPP(TypedDict):
    # verdict: ok -> 1.0, fail -> 0.0, no code -> None
    ok: bool
    ret_code: int
    stderr: str  # or empty string


class MBPPEvalTasks(CodeGenEvalTasks[InputMBPP, VerdictMBPP]):
    """Mostly Basic Python Problems Dataset
    Released as part of Program Synthesis with Large Language Models, Austin et. al., 2021.
    https://github.com/google-research/google-research/tree/master/mbpp"""

    name = "MBPP"

    def __init__(self, json_path: pathlib.Path, samples: int | None):
        self._json_path = json_path
        self._take_samples = samples

    def iter(self) -> Generator[tuple[str, InputMBPP], None, None]:
        # load data
        with open(pathlib.Path(self._json_path), "r", encoding="utf-8") as f:
            data: list[InputMBPP] = json.load(f)
            items = {item["task_id"]: item for item in data}

        # reproducible randomization
        if self._take_samples is not None:
            shuffle_keys = list(items.keys())
            rand = random.Random()
            rand.seed(42)
            rand.shuffle(shuffle_keys)
            shuffle_keys = shuffle_keys[: self._take_samples]
            items = {k: v for k, v in items.items() if k in shuffle_keys}

        for task_id, item in items.items():
            yield task_id, item
        return

    async def execute(self, ctx, method, task_id, task) -> tuple[EvalResult[InputMBPP, VerdictMBPP], CodeGenJournal]:
        # gap: default imports, test cases
        assert ctx.lrt.lang == "python"  # TODO: support other languages

        # WARNING: if you use these test cases in the program, you're testing
        #          your program with ground truth visible. this is leaky.
        visible_tests = task["test_list"]
        test_cases: list[str] = []
        for _line in visible_tests:
            if line := _line.strip():
                test_cases.append(line)

        prompt = ctx.lrt.parse(module=(), code=task["prompt"])
        prompt_funcs = prompt.find_all(LrtFunctionDef, None)
        prompt_func = prompt.find(LrtFunctionDef, task["entry_point"])
        assert prompt_func is not None
        prompt_funcs_sup: list[LrtNode] = [f for f in prompt_funcs if f.name != prompt_func.name]
        prompt_misc = prompt.excluding(prompt_func, prompt_funcs_sup)

        program, _sj = await method.gen(
            ctx=ctx,
            ancestors=prompt_misc,
            func=prompt_func,
            descendants=prompt_funcs_sup,
        )
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

    async def judge(self, ctx, result) -> EvalResult[InputMBPP, VerdictMBPP]:
        req_test_list = result["task"]["test_list"]
        draft_code = result["code"]
        if draft_code is None:
            result["verdict"] = None
            result["_verdict_info"] = {"ok": False, "ret_code": -1, "stderr": "no code to execute"}
            return result

        # MBPP comes with script-y tests, meaning that they should've be
        # executed at top-level, but we choose to place them inside functions
        # to isolate contexts.
        test_fn_name = f"_test_main_{_make_random_sig()}"
        test_wrapper = ""
        test_wrapper += f"def {test_fn_name}() -> bool:\n"
        for line in req_test_list:
            test_wrapper += f"    {line}\n"
        test_wrapper += f"    \n"
        test_wrapper += f"    return True\n"
        test_wrapper += f"\n"

        exec_code = DEFAULT_IMPORTS + "\n\n\n" + draft_code + "\n\n\n" + test_wrapper
        exec_program = ctx.lrt.parse(module=(), code=exec_code)
        exec_result = await ctx.lrt.run_program(exec_program, test_fn_name, args=[], kwargs={}, timeout=2.5)
        print(f"================ {result['id']} ================")
        print(exec_result.error)

        success = exec_result.ok and exec_result.ret_code == 0 and exec_result.result is True
        result["verdict"] = 1.0 if success else 0.0
        result["_verdict_info"] = {
            "ok": success,
            "ret_code": exec_result.ret_code,
            "stderr": exec_result.error,
        }
        return result

    pass
