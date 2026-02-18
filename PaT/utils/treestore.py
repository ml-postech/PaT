import dataclasses
from typing import Generic, TypeVar

from .pyctx import PyCtx

T = TypeVar("T")


@dataclasses.dataclass
class TreeStoreTrap(Generic[T]):
    _valid: bool
    _values: list[T]

    def gather(self) -> list[T]:
        self._valid = False
        return self._values

    pass


class TreeStore(Generic[T]):
    """Parent nodes `t = trap()`, descendants `put()`, and then parent nodes
    finally `t.gather()` the descendants that have not been trapped by an
    intermediate descendant."""

    def __init__(self, key: str):
        self._ctx = PyCtx[TreeStoreTrap[T]](f"dqllm_tree_store_scope[{key}]")
        return

    def trap(self, offset: int = 0) -> TreeStoreTrap[T]:
        node = TreeStoreTrap[T](_valid=True, _values=[])
        self._ctx.append(node, offset=offset + 1)
        return node

    def put(self, item: T, offset: int = 0) -> None:
        nodes = self._ctx.get(offset=offset + 1)
        for i in range(len(nodes) - 1, -1, -1):
            p = nodes[i]
            if p._valid:
                p._values.append(item)
                break
        return

    pass
