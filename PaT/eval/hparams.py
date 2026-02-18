import enum
import json
import pathlib
from typing import Any, Literal, TypeVar

import pydantic

T = TypeVar("T", bound=pydantic.BaseModel)


class _HP_Task_HumanEval(pydantic.BaseModel):
    task_name: Literal["HumanEval"] = "HumanEval"
    task_samples: int | None


class _HP_Task_MBPP(pydantic.BaseModel):
    task_name: Literal["MBPP"] = "MBPP"
    task_samples: int | None


class _HP_Task_xCodeEval(pydantic.BaseModel):
    task_name: Literal["xCodeEval"] = "xCodeEval"
    task_samples: int | None


class _HP_Task_MATH(pydantic.BaseModel):
    task_name: Literal["MATH"] = "MATH"
    task_samples: int | None
    task_llm_engine: str


_HP_Task = _HP_Task_HumanEval | _HP_Task_MBPP | _HP_Task_xCodeEval | _HP_Task_MATH


class _HP_Prompt_GenOnce(enum.Enum):
    humaneval_divide = "humaneval_divide"
    humaneval_conquer = "humaneval_conquer"
    maths_divide = "maths_divide"
    maths_conquer = "maths_conquer"
    xcodeeval_divide = "xcodeeval_divide"
    xcodeeval_conquer = "xcodeeval_conquer"
    pass


class _HP_Prompt_MakeTest(enum.Enum):
    humaneval_funccall = "humaneval_funccall"
    humaneval_argsmaker = "humaneval_argsmaker"
    humaneval_unittest = "humaneval_unittest"
    injected_expr = "injected_expr"
    injected_stdio = "injected_stdio"
    injected_tests = "injected_tests"
    sys_test_args = "sys_test_args"
    xcodeeval_funccall = "xcodeeval_funccall"
    xcodeeval_unittest = "xcodeeval_unittest"
    pass


class _HP_Method_PaT(pydantic.BaseModel):
    method_name: Literal["PaT"] = "PaT"
    # dfs mechanism
    dfs_max_depth: int
    # divide
    divide_gen_prompt: _HP_Prompt_GenOnce
    divide_temperature: float
    divide_retries: int
    # functional consistency
    fc_root_test_prompt: _HP_Prompt_MakeTest | None
    fc_root_sys_test_prompt: _HP_Prompt_MakeTest | None
    fc_branch_test_prompt: _HP_Prompt_MakeTest | None
    fc_branch_sys_test_prompt: _HP_Prompt_MakeTest | None
    fc_temperature: float
    fc_retries: int
    # conquer
    conquer_gen_prompt: _HP_Prompt_GenOnce
    conquer_temperature: float
    conquer_samples: int
    conquer_min_samples: int
    conquer_retries: int
    pass

class _HP_Method_Vanilla(pydantic.BaseModel):
    method_name: Literal["vanilla"] = "vanilla"
    gen_prompt: _HP_Prompt_GenOnce
    temperature: float
    retries: int
    pass


_HP_Method = (
    _HP_Method_PaT
    | _HP_Method_Vanilla
)


class HParams(pydantic.BaseModel):
    schema_version: str | None = pydantic.Field(alias="$schema", serialization_alias="$schema")

    task: _HP_Task
    langrt: str  # key
    llm_engine: str  # key
    llm_engine_planner: str # key
    method: _HP_Method
    results_dir: pathlib.Path | None = None
    wandb_run_id: str | None = None

    def dump(self) -> dict[str, Any]:
        dumped = self.model_dump_json(by_alias=True, exclude_unset=True)
        dumped = json.loads(dumped)
        return dumped

    def dump_flattened(self) -> dict[str, Any]:
        dumped = self.dump()
        get_field = lambda k: {k: dumped[k]} if k in dumped else {}
        return {
            **get_field("$schema"),
            **dumped["task"],
            "langrt": dumped["langrt"],
            "llm_engine": dumped["llm_engine"],
            "llm_engine_planner": dumped["llm_engine_planner"],
            **dumped["method"],
            **get_field("results_dir"),
            **get_field("wandb_run_id"),
        }

    @staticmethod
    def load(data: dict[str, Any]) -> "HParams":
        return HParams.model_validate(data)

    @staticmethod
    def load_flattened(data: dict[str, Any]) -> "HParams":
        get_field = lambda k: {k: data[k]} if k in data else {}
        unflattened = {
            **get_field("$schema"),
            "task": data,
            "langrt": data["langrt"],
            "llm_engine": data["llm_engine"],
            "llm_engine_planner": data["llm_engine_planner"],
            "method": data,
            **get_field("results_dir"),
            **get_field("wandb_run_id"),
        }
        return HParams.model_validate(unflattened)

    pass


if __name__ == "__main__":
    schema_path = (pathlib.Path(__file__).parent / "./hparams.schema.json").resolve()
    with open(schema_path, "w", encoding="utf-8") as f:
        schema = HParams.model_json_schema()
        raw = json.dumps(schema, indent=2, ensure_ascii=False)
        f.write(raw + "\n")
    print(f"> schema written to: {schema_path}")
