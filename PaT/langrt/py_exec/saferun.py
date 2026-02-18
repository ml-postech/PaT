import io
import subprocess
from typing import Any, TypeVar, cast

import pydantic


class ProtocolInput(pydantic.BaseModel):
    session_id: int
    mod_daemon: str
    mod_run: str
    # internal=True/dep=False, mod.ule.name, module_alias?, [...(symbols, aliases)]?
    imports: list[tuple[bool, str, str | None, list[tuple[str, str]] | None]]
    func_name: str
    func_args: list[str]
    func_kwargs: dict[str, str]
    func_ret: str
    call_args: list[Any]
    call_kwargs: dict[str, Any]
    io_stdin: str


class ProtocolOutput(pydantic.BaseModel):
    session_id: int
    call_err: None | str
    call_ret: Any
    call_stdout: str
    call_dt: float


class ProcessWrapper:
    """Provides an RAII-wise safe handle to a subprocess, ensuring that child
    processes are always killed if a runner was destroyed."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc

    def __del__(self):
        self._proc.terminate()
        self._proc.wait()
        try:
            not_null(self._proc.stdin).close()
        except Exception:
            pass
        try:
            not_null(self._proc.stdout).close()
        except Exception:
            pass
        try:
            not_null(self._proc.stderr).close()
        except Exception:
            pass
        return

    @property
    def stdin(self) -> io.TextIOWrapper:
        return cast(io.TextIOWrapper, self._proc.stdin)

    @property
    def stdout(self) -> io.TextIOWrapper:
        return cast(io.TextIOWrapper, self._proc.stdout)

    @property
    def stderr(self) -> io.TextIOWrapper:
        return cast(io.TextIOWrapper, self._proc.stderr)

    def poll(self) -> int | None:
        return self._proc.poll()

    def terminate(self) -> None:
        self._proc.terminate()
        return

    def kill(self) -> None:
        self._proc.kill()
        return

    def wait(self) -> int:
        return self._proc.wait()

    pass


T = TypeVar("T")


def not_null(x: T | None) -> T:
    if x is None:
        raise ValueError("required value")
    return cast(T, x)
