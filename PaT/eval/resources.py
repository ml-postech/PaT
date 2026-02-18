import pathlib
from typing import Any, overload

from ..langrt import LangRT
from ..llm import LLMEngine, create_llm_engine
from ..methods.PaT.gen import PaTGen
from ..methods.PaT.gen_once import GenOncePrompt
from ..methods.PaT.make_test import MakeTestPrompt
from ..methods.PaT.prompts.humaneval import (
    PaTHumanEvalArgsMakerPrompt,
    PaTHumanEvalConquerPrompt,
    PaTHumanEvalDividePrompt,
    PaTHumanEvalFuncCallPrompt,
    PaTHumanEvalUnitTestPrompt,
)
from ..methods.PaT.prompts.injected import (
    PaTInjectedExprPrompt,
    PaTInjectedStdioPrompt,
    PaTInjectedTestsPrompt,
)
from ..methods.PaT.prompts.sys_test import PaTSysTestArgsPrompt
from ..methods.PaT.prompts.xcodeeval import (
    PaTXCodeEvalConquerPrompt,
    PaTXCodeEvalDividePrompt,
    PaTXCodeEvalFuncCallPrompt,
    PaTXCodeEvalUnitTestPrompt,
)
from ..methods.shared import CodeGenContext, CodeGenMethod
from ..methods.vanilla.gen import VanillaGen
from ..utils.logger import Logger
from ..utils.types import guard_never
from .config import EvalConfig
from .download_tasks.humaneval import download_humaneval_dataset
from .download_tasks.mbpp import download_mbpp_dataset
from .download_tasks.xcodeeval import download_xcodeeval_dataset
from .hparams import HParams, _HP_Method, _HP_Prompt_GenOnce, _HP_Prompt_MakeTest, _HP_Task
from .tasks.humaneval import HumanEvalEvalTasks
from .tasks.mbpp import MBPPEvalTasks
from .tasks.xcodeeval import xCodeEvalEvalTasks
from .types import CodeGenEvalTasks


def pick_code_gen_ctx(cfg: EvalConfig, hparams: HParams) -> CodeGenContext:
    return CodeGenContext(
        log=Logger(hide_event_types=cfg.logger.hide_event_types),
        llm=pick_llm(cfg, hparams.llm_engine),
        llm_planner=pick_llm(cfg, hparams.llm_engine_planner),
        lrt=pick_langrt(cfg, hparams.langrt),
        cfg_silent=cfg.misc.silent,
    )


def pick_llm(cfg: EvalConfig, llm_key: str) -> LLMEngine:
    llm_cfg = cfg.llm[llm_key]
    return create_llm_engine(llm_cfg)


def pick_langrt(cfg: EvalConfig, langrt_key: str) -> LangRT:
    langrt_cfg = cfg.langrt[langrt_key]
    if langrt_cfg.lang == "python":
        return LangRT.python(
            sandbox_root=pathlib.Path(langrt_cfg.sandbox_root),
            parallelism=langrt_cfg.parallelism,
            macos_sandbox_bin=langrt_cfg.macos_sandbox_bin,
            python_bin=langrt_cfg.python_bin,
        )
    guard_never(langrt_cfg.lang)


def download_all_tasks(cfg_proxy: str | None = None) -> None:
    datasets = pathlib.Path(__file__).parent / "../../datasets"
    download_humaneval_dataset(datasets / "HumanEval/", cfg_proxy=cfg_proxy)
    download_mbpp_dataset(datasets / "MBPP/", cfg_proxy=cfg_proxy)
    download_xcodeeval_dataset(datasets / "xCodeEval/", cfg_proxy=cfg_proxy)
    return


def pick_tasks(cfg: EvalConfig, task: _HP_Task) -> CodeGenEvalTasks[Any, Any]:
    datasets = pathlib.Path(__file__).parent / "../../datasets"
    if task.task_name == "HumanEval":
        return HumanEvalEvalTasks(
            json_path=datasets / "HumanEval/HumanEval_processed.json",
            samples=task.task_samples,
        )
    if task.task_name == "MBPP":
        return MBPPEvalTasks(
            json_path=datasets / "MBPP/MBPP_typed.json",
            samples=task.task_samples,
        )
    if task.task_name == "xCodeEval":
        return xCodeEvalEvalTasks(
            dir_path=datasets / "xCodeEval/",
            samples=task.task_samples,
        )
    guard_never(task)


def pick_gen_once_prompt(key: _HP_Prompt_GenOnce) -> GenOncePrompt:
    E = _HP_Prompt_GenOnce
    if key == E.humaneval_divide:
        return PaTHumanEvalDividePrompt()
    if key == E.humaneval_conquer:
        return PaTHumanEvalConquerPrompt()
    if key == E.maths_divide:
        return PaTMathsDividePrompt()
    if key == E.maths_conquer:
        return PaTMathsConquerPrompt()
    if key == E.xcodeeval_divide:
        return PaTXCodeEvalDividePrompt()
    if key == E.xcodeeval_conquer:
        return PaTXCodeEvalConquerPrompt()
    guard_never(key)


@overload
def pick_make_test_prompt(key: _HP_Prompt_MakeTest) -> MakeTestPrompt: ...
@overload
def pick_make_test_prompt(key: _HP_Prompt_MakeTest | None) -> MakeTestPrompt | None: ...
def pick_make_test_prompt(key: _HP_Prompt_MakeTest | None) -> MakeTestPrompt | None:
    E = _HP_Prompt_MakeTest
    if key is None:
        return None
    if key == E.humaneval_funccall:
        return PaTHumanEvalFuncCallPrompt()
    if key == E.humaneval_argsmaker:
        return PaTHumanEvalArgsMakerPrompt()
    if key == E.humaneval_unittest:
        return PaTHumanEvalUnitTestPrompt()
    if key == E.injected_expr:
        return PaTInjectedExprPrompt()
    if key == E.injected_stdio:
        return PaTInjectedStdioPrompt()
    if key == E.injected_tests:
        return PaTInjectedTestsPrompt()
    if key == E.sys_test_args:
        return PaTSysTestArgsPrompt()
    if key == E.xcodeeval_funccall:
        return PaTXCodeEvalFuncCallPrompt()
    if key == E.xcodeeval_unittest:
        return PaTXCodeEvalUnitTestPrompt()
    guard_never(key)


def pick_method(method: _HP_Method) -> CodeGenMethod:
    if method.method_name == "PaT":
        return PaTGen(
            dfs_max_depth=method.dfs_max_depth,
            divide_gen_prompt=pick_gen_once_prompt(method.divide_gen_prompt),
            divide_temperature=method.divide_temperature,
            divide_retries=method.divide_retries,
            fc_root_test_prompt=pick_make_test_prompt(method.fc_root_test_prompt),
            fc_root_sys_test_prompt=pick_make_test_prompt(method.fc_root_sys_test_prompt),
            fc_branch_test_prompt=pick_make_test_prompt(method.fc_branch_test_prompt),
            fc_branch_sys_test_prompt=pick_make_test_prompt(method.fc_branch_sys_test_prompt),
            fc_temperature=method.fc_temperature,
            fc_retries=method.fc_retries,
            conquer_gen_prompt=pick_gen_once_prompt(method.conquer_gen_prompt),
            conquer_temperature=method.conquer_temperature,
            conquer_samples=method.conquer_samples,
            conquer_min_samples=method.conquer_min_samples,
            conquer_retries=method.conquer_retries,
        )
    if method.method_name == "vanilla":
        return VanillaGen(
            gen_prompt=pick_gen_once_prompt(method.gen_prompt),
            temperature=method.temperature,
            retries=method.retries,
        )
    guard_never(method)
