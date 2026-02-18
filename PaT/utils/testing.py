import asyncio
import pathlib
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from ..llm.config import LLMConfig
    from ..methods.shared import CodeGenContext


class UnitTestConfig:
    """Configuration for unit tests and unit tests only."""

    def __init__(self):
        return

    def test_llm(self) -> bool:
        # WARNING: YOU WILL PAY FOR THIS (LITERALLY) IF SET TO TRUE
        return False

    def mk_llm_config(self) -> "LLMConfig":
        from ..eval.config import get_eval_config

        cfg = get_eval_config()
        llm_cfg = cfg.llm["for_unittest"]
        return llm_cfg

    def mk_code_gen_ctx(self) -> "CodeGenContext":
        from ..eval.config import get_eval_config
        from ..langrt import LangRT
        from ..llm import create_llm_engine
        from ..methods.shared import CodeGenContext
        from ..utils.logger import Logger

        cfg = get_eval_config()
        lrt_cfg = cfg.langrt["py3"]
        return CodeGenContext(
            log=Logger(),
            llm=create_llm_engine(self.mk_llm_config()),
            lrt=LangRT.python(
                sandbox_root=pathlib.Path(lrt_cfg.sandbox_root),
                parallelism=lrt_cfg.parallelism,
            ),
            cfg_silent=False,
        )

    pass


TArgs = ParamSpec("TArgs")


def async_test_case(fn: Callable[TArgs, Coroutine[Any, Any, None]]) -> Callable[TArgs, None]:
    """Decorate an async test case function to run it synchronously."""

    def _wrapper(*args, **kwargs) -> None:
        return asyncio.run(fn(*args, **kwargs))

    _wrapper.__name__ = fn.__name__
    return _wrapper
