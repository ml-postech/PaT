import asyncio
import pathlib
import random
import string

import rich.console

from ..types import FileContent, ModuleName
from .s_client import SingleThreadedClient
from .saferun import ProtocolInput, ProtocolOutput


class MultiThreadedClient:
    """Thanks to single-threaded clients allowing module names, multiple
    processes or threads can now run in mutual isolation. Note that this is
    still a lower-level implementation and should be wrapped up."""

    def __init__(
        self,
        sandbox_root: pathlib.Path,
        parallelism: int,
        macos_sandbox_bin: str,
        python_bin: str,
        console: rich.console.Console,
    ):
        self._sandbox_root = sandbox_root
        self._parallelism = parallelism
        self._worker_prefix = self._roll_thread_prefix()
        self._workers: list[SingleThreadedClient] = []
        self._tokens = asyncio.Queue()

        for i in range(self._parallelism):
            info = f"[sky_blue1]starting Python code executor ({i + 1} / {self._parallelism})...[/sky_blue1]    "
            console.print(info, end="\r")  # won't trigger 'one live display required'
            client = SingleThreadedClient(
                sandbox_root=self._sandbox_root,
                daemon_module_name=f"{self._worker_prefix}_{i}",
                console=console,
                macos_sandbox_bin=macos_sandbox_bin,
                python_bin=python_bin,
            )
            self._workers.append(client)
            self._tokens._put(i)
        console.print(f"[sky_blue1]started {self._parallelism} Python code executors;[/sky_blue1]" + " " * 16)

    async def run(self, code: dict[ModuleName, FileContent], timeout: float, inp: ProtocolInput) -> ProtocolOutput:
        """Run the code in any one of the threads."""

        token = await self._tokens.get()
        try:
            proc = lambda: self._workers[token].run(code, timeout, inp)
            out = await asyncio.get_event_loop().run_in_executor(None, proc)
        finally:
            await self._tokens.put(token)
        return out

    def close(self):
        for worker in self._workers:
            worker._stop_daemon(force=True)
        for worker in self._workers:
            worker.close()
        return

    def __del__(self) -> None:
        self.close()

    def _roll_thread_prefix(self) -> str:
        if not self._sandbox_root.exists():
            self._sandbox_root.mkdir(parents=True, exist_ok=True)
        while True:
            prefix = "th_" + "".join(random.choices(string.ascii_letters + string.digits, k=6))
            files = list(self._sandbox_root.iterdir())
            if not any(f.name.startswith(prefix) for f in files):
                break
        return prefix

    pass
