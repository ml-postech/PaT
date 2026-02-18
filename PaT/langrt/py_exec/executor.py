import pathlib
import random
import sys
import threading

import rich

from ..executor import LrtExecutor
from ..types import JsonSerializable, LrtExecutionEnv, LrtExecutionResult
from .m_client import MultiThreadedClient
from .saferun import ProtocolInput


class PyExecutor(LrtExecutor):
    def __init__(
        self,
        sandbox_root: pathlib.Path,
        parallelism: int,
        console: rich.console.Console = rich.get_console(),
        macos_sandbox_bin: str = "sandbox-exec",
        python_bin: str = sys.executable,
    ):
        self._lock = threading.Lock()
        self._m_client_factory = lambda: MultiThreadedClient(
            sandbox_root=sandbox_root,
            parallelism=parallelism,
            macos_sandbox_bin=macos_sandbox_bin,
            python_bin=python_bin,
            console=console,
        )
        self._m_client: MultiThreadedClient | None = None

    async def run(
        self,
        env: LrtExecutionEnv,
        args: list[JsonSerializable],
        kwargs: dict[str, JsonSerializable],
        stdin: str = "",
        timeout: float = 1.0,
    ) -> LrtExecutionResult:
        with self._lock:
            if self._m_client is None:
                self._m_client = self._m_client_factory()
            m_client = self._m_client

        inp = ProtocolInput(
            session_id=random.randrange(0, 2**31),
            mod_daemon="",
            mod_run="",
            imports=env.imports,
            func_name=env.func_name,
            func_args=env.func_args,
            func_kwargs=env.func_kwargs,
            func_ret=env.func_ret,
            call_args=args,
            call_kwargs=kwargs,
            io_stdin=stdin,
        )
        out = await m_client.run(code=env.code, timeout=timeout, inp=inp)

        return LrtExecutionResult(
            ok=out.call_err is None,
            ret_code=0 if out.call_err is None else 1073741823,
            error=out.call_err or "",
            result=out.call_ret,
            stdout=out.call_stdout,
            duration=out.call_dt,
        )

    def close(self):
        with self._lock:
            if self._m_client is not None:
                self._m_client.close()
                self._m_client = None
        return

    pass
