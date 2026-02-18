import json
import pathlib
import random
import re
import shutil
import string
import subprocess
import sys
import threading
import time
import traceback

import rich

from ..types import FileContent, ModuleName
from .saferun import ProcessWrapper, ProtocolInput, ProtocolOutput, not_null


class SingleThreadedClient:
    """The single-threaded client connects to exactly one daemon at a time, and
    have all code execution delegated to that daemon process sequentially.
    Crashes in daemons will trigger restarts."""

    def __init__(
        self,
        sandbox_root: pathlib.Path,
        daemon_module_name: str,
        console: rich.console.Console,
        # binary paths
        macos_sandbox_bin: str,  # default: "sandbox-exec"
        python_bin: str,  # default: "python"
    ):
        self._closed = False
        self._sandbox_root = sandbox_root
        self._daemon_module_name = daemon_module_name  # module name of current client, e.g. "worker_1"
        self._workspace = self._sandbox_root / self._daemon_module_name
        self._console = console
        self._macos_sandbox_bin = macos_sandbox_bin
        self._python_bin = python_bin
        self._proc: ProcessWrapper | None = self._start_daemon()

    def run(self, code: dict[ModuleName, FileContent], timeout: float, inp: ProtocolInput) -> ProtocolOutput:
        opt_max_retries = 15
        tm_begin = time.time()
        for retry in range(opt_max_retries):
            out = self._run_once(code, timeout, inp, retry)
            if out is not None:
                return out
        return ProtocolOutput(
            session_id=inp.session_id,
            call_err=f"runner crashed for {opt_max_retries} times",
            call_ret=None,
            call_stdout="",
            call_dt=time.time() - tm_begin,
        )

    def close(self):
        if self._closed:
            return
        self._stop_daemon(force=True)
        shutil.rmtree(self._workspace, ignore_errors=True)
        self._proc = None
        self._closed = True

    # daemon related

    def _use_proc(self) -> ProcessWrapper:
        """Try our best to create a new Daemon process."""

        for _ in range(10):
            if self._proc is not None and self._proc.poll() is None:
                return self._proc
            self._stop_daemon(force=True)
            self._proc = self._start_daemon()
        raise RuntimeError("cannot start python runtime")

    def _start_daemon(self) -> ProcessWrapper | None:
        self._workspace.mkdir(parents=True, exist_ok=True)
        for ch in self._workspace.iterdir():
            shutil.rmtree(ch, ignore_errors=True)
        myself = pathlib.Path(__file__)
        self._copy_graceful(myself.parent / "sandbox_profile", self._workspace / ".sandbox_profile")
        self._copy_graceful(myself.parent / "daemon.py", self._workspace / "__main__.py")

        sandbox_available = sys.platform == "darwin"
        if sandbox_available:
            sbp_path = str((self._workspace / ".sandbox_profile").resolve())
            cmd = [self._macos_sandbox_bin, "-f", sbp_path, self._python_bin, "-m", self._daemon_module_name]
        else:
            cmd = [self._python_bin, "-m", self._daemon_module_name]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self._sandbox_root.resolve()),
        )

        worker_id = random.randrange(0, 2**16)
        not_null(proc.stdin).write(f"PING {worker_id}\n")
        not_null(proc.stdin).flush()
        line = not_null(proc.stdout).readline().strip()
        if line != f"PONG {worker_id}" or proc.poll() is not None:
            err = not_null(proc.stderr).read()
            self._console.print(f"<py_rt:{self._daemon_module_name}>\ncannot start daemon, reason:\n\n{err}")
            return None
        return ProcessWrapper(proc)

    def _stop_daemon(self, force: bool) -> None:  # noexcept
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            if not force:
                not_null(self._proc.stdin).write("null\n")
                time.sleep(0.5)
            self._proc.terminate()
            self._proc.wait()
            not_null(self._proc.stdin).close()
            not_null(self._proc.stdout).close()
            not_null(self._proc.stderr).close()
        except Exception:
            pass

    # execution

    def _roll_mod_name(self) -> str:
        """The code of each run is stored separately in a different folder
        (module name) under the cache dir to provide a basic isolation. We
        guarantee that there will be no name collision as long as there are not
        too many concurrent runs (like a lot)."""

        while True:
            mod = "exec_" + "".join(random.choices(string.ascii_letters + string.digits, k=6))
            if not (self._workspace / mod).exists():
                break
        return mod

    def _load_code(self, module_name: str, code: dict[ModuleName, FileContent]) -> None:
        """This initializes the workspace of each run."""

        for rel_path, content in code.items():
            path = self._workspace / module_name / self._module_to_path(rel_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return

    def _unload_code(self, mod: str) -> None:  # noexcept
        """Clean up the workspace of each run. Since there is little chance
        that the daemon will die even if we forgot to clean up we won't be in
        real trouble."""

        try:
            shutil.rmtree(self._workspace / mod, ignore_errors=True)
        except Exception:
            pass

    def _run_once(
        self, code: dict[ModuleName, FileContent], timeout: float, inp: ProtocolInput, retry: int
    ) -> ProtocolOutput | None:
        proc = self._use_proc()
        # leave inp.{mod_run, mod_daemon} empty. we'll fill them in
        mod = self._roll_mod_name()
        inp.mod_daemon = self._daemon_module_name
        inp.mod_run = mod
        self._load_code(mod, code)
        # need to also monitor tle
        running = [True]
        tm_begin = time.time()
        killer = threading.Thread(target=self._kill_after_tle, args=(proc, running, timeout))
        killer.start()
        out = self._run_only(proc, running, inp, retry, tm_begin, timeout)
        if (duration := time.time() - tm_begin) >= timeout:
            out = ProtocolOutput(
                session_id=inp.session_id,
                call_err=f"subprocess.TimeoutExpired: Program timed out after {timeout} seconds",
                call_ret=None,
                call_stdout="",
                call_dt=duration,
            )
        killer.join()
        self._unload_code(mod)
        return out

    def _run_only(
        self, proc: ProcessWrapper, running: list[bool], inp: ProtocolInput, retry: int, tm_begin: float, timeout: float
    ) -> ProtocolOutput | None:  # noexcept, None if crashes
        out_raw = ""
        try:
            inp_j = inp.model_dump_json(indent=None)
            inp_raw = json.dumps(json.loads(inp_j), indent=None, ensure_ascii=False)
            not_null(proc.stdin).write(f"{inp_raw.strip()}\n")
            not_null(proc.stdin).flush()
            out_raw: str = not_null(proc.stdout).readline()
            out_j = json.loads(out_raw)
            out = ProtocolOutput.model_validate(out_j)
            if out.session_id != inp.session_id:
                l1 = f"<py_rt:{self._daemon_module_name}, retry#{retry}>"
                l2 = f"runner session id mismatch, expects {inp.session_id}, got {out.session_id}"
                self._console.print(f"{l1}\n{l2}")
                proc.kill()  # must not leave dirty states behind
                return None
            return out
        except Exception:
            # this usually does not happen unless the daemon crashes
            proc_err = not_null(proc.stderr).read()
            call_err = traceback.format_exc()
            if not (time.time() - tm_begin > timeout):
                all_err = f"<py_rt:{self._daemon_module_name}, retry#{retry}>\n\n{out_raw}\n\n{proc_err}\n\n{call_err}"
                all_err = re.sub(r"\n\n+", "\n\n", all_err).strip("\n")
                self._console.print(all_err)
            return None
        finally:
            running[0] = False

    def _kill_after_tle(self, proc: ProcessWrapper, running: list[bool], timeout: float) -> None:  # noexcept
        """Runs on a separate Python thread and kills the given process if it
        runs for too long. For short durations this may incur some delay."""

        begin = time.time()
        while time.time() - begin < timeout:
            time.sleep(0.03)
            if not running[0]:  # done, no need to kill
                return
        try:
            not_null(proc.stdin).write("\nnull\n")
            proc.kill()
        except Exception:
            pass

    # utility functions

    def _copy_graceful(self, src: pathlib.Path, dst: pathlib.Path):
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)
        return

    def _module_to_path(self, mn: ModuleName) -> str:
        ps = self._sanitize_module_name(mn).split(".")
        ps = [i for i in ps if i]
        ps.append("__init__.py")
        return "/".join(ps)

    def _sanitize_module_name(self, mn: ModuleName) -> ModuleName:
        ps = mn.split(".")
        ps = [i.strip() for i in ps]
        ps = [i for i in ps if i]
        while len(ps) and ps[-1] == "__init__":
            ps.pop()
        return ".".join(ps)

    pass
