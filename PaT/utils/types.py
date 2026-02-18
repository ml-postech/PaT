import json
from typing import Any, Type, TypeVar, cast

import pydantic
import pydantic_core
from typing_extensions import Never

T = TypeVar("T")

TModel = TypeVar("TModel", bound=pydantic.BaseModel)


def coalesce(x: T | None, y: T) -> T:
    """TS / `x ?? y`"""

    return y if x is None else x


def not_null(x: T | None) -> T:
    """TS / `x!`"""

    if x is None:
        raise ValueError("required value")
    return cast(T, x)


def guard_never(x: Never) -> Never:
    raise ValueError(f"unexpected value: {x}")


def anything_into_dict(typ: tuple[str, Type], value: Any) -> Any:
    """load(T.name, ^T, 'value') -> value@T"""

    model = pydantic.create_model(f"TypeParser<{typ[0]}>", the_value=(typ[1], ...))
    serialized: Any = model(the_value=value)
    try:
        j_s = serialized.model_dump_json()
    except pydantic_core.PydanticSerializationError:
        j_s = pydantic_core.to_json(serialized, serialize_unknown=True)
    j = json.loads(j_s)
    return j["the_value"]


def anything_from_dict(typ: tuple[str, Type], serialized: Any) -> Any:
    """dump(T.name, ^T, value@T) -> 'value'"""

    model = pydantic.create_model(f"TypeParser<{typ[0]}>", the_value=(typ[1], ...))
    parsed: Any = model(the_value=serialized)
    return parsed.the_value


def reshape(x: list[T], shape: tuple[int, int]) -> list[list[T]]:
    """Reinterpret (row, col) as another 2D array."""

    dim0, dim1 = shape
    if len(x) != dim0 * dim1:
        raise ValueError(f"expected {dim0 * dim1} elements, got {len(x)}")
    return [x[(i * dim1) : (i + 1) * dim1] for i in range(dim0)]
