import dataclasses
import inspect
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class PyCtxPointer(Generic[T]):
    ancestors: list[T]
    current: tuple[T] | None
    pass


class PyCtx(Generic[T]):
    """React-like context for Python. A consumer may access context values from
    all ancestors of the current frame, which are stored in a variable under
    the frame's local variables.

    The context is preserved and indexable across typical calls and async calls.
    It is not preserved across threads or processes.

    WARNING: At most one value is stored in each frame."""

    def __init__(self, key: str):
        self._root_key = "__$__GENERIC_STORE__$__"
        self._key = key

    def get(self, offset: int = 0) -> list[T]:
        """Fetches stored values from all ancestors since current frame."""

        ptr, _off = self.__load(2 + offset)
        ret = list(ptr.ancestors)
        if ptr.current is not None:
            ret.append(ptr.current[0])
        return ret

    def append(self, value: T, offset: int = 0) -> None:
        """Writes value to current frame. Overwrites any existing value at
        current frame."""

        ptr, off = self.__load(2 + offset)
        tmp = list(ptr.ancestors)
        if off > 0 and ptr.current is not None:
            tmp.append(ptr.current[0])
        new_ptr = PyCtxPointer(ancestors=tmp, current=(value,))
        self.__store(2 + offset, new_ptr)
        return

    def update(self, fn: Callable[[tuple[T] | None], T], offset: int = 0) -> None:
        """Updates value at current frame, or creates a new value if no such
        value exists at current frame."""

        ptr, off = self.__load(2 + offset)
        if off == 0:
            new_val = fn(ptr.current)
            new_ptr = PyCtxPointer(ancestors=ptr.ancestors, current=(new_val,))
            self.__store(2 + offset, new_ptr)
        else:
            # nothing at current frame to append to
            self.append(fn(None), offset=1 + offset)
        return

    def __load(self, offset: int) -> tuple[PyCtxPointer[T], int]:
        stk = inspect.stack()
        pointer = PyCtxPointer[T](ancestors=[], current=None)
        res_offset = 0
        for i, frame in enumerate(stk[offset:]):
            vars = frame.frame.f_locals
            if self._root_key not in vars:
                continue
            root: dict[str, PyCtxPointer[T]] = vars[self._root_key]
            if self._key not in root:
                continue
            pointer = root[self._key]
            res_offset = i
            break
        return pointer, res_offset

    def __store(self, offset: int, ptr: PyCtxPointer[T]) -> None:
        stk = inspect.stack()
        if len(stk) < offset:
            return
        frame = stk[offset].frame
        vars = frame.f_locals
        if self._root_key not in vars:
            vars[self._root_key] = {}
        root: dict[str, PyCtxPointer[T]] = vars[self._root_key]
        root[self._key] = ptr
        return

    pass
