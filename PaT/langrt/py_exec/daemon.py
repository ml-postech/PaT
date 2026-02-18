# cspell: disable

import gc
import importlib
import io
import json
import pathlib
import platform
import sys
import time
import traceback
from typing import Any, Callable, TypeVar

import pydantic
import pydantic_core

__all__ = []


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


T = TypeVar("T")

__PLATFORM_SYSTEM__ = platform.system()


def __guard__() -> None:
    """
    Copied from https://github.com/openai/human-eval/blob/master/human_eval/execution.py
    with minor modification.

    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    ###########################################################################
    #   adds syntactic sugar to make this wrapper more readable

    class UnauthorizedAction:
        pass

    permissive = lambda *args, **kwargs: None  # nobody calls them. but there need to be sth
    never = UnauthorizedAction()  # code should never call them

    ###########################################################################

    import builtins
    import faulthandler
    import os
    import platform
    import shutil
    import subprocess
    import sys

    builtins.exit = never
    builtins.quit = never

    faulthandler.disable()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = never
    os.system = never
    os.putenv = permissive  # for numpy
    os.register_at_fork = permissive  # type: ignore  # for pydantic
    os.remove = never
    os.removedirs = never
    os.rmdir = never
    os.fchdir = never  # type: ignore
    os.setuid = never  # type: ignore
    os.fork = never  # type: ignore
    os.forkpty = never  # type: ignore
    os.killpg = never  # type: ignore
    os.rename = never
    os.renames = never
    os.truncate = never
    os.replace = never
    os.unlink = never
    os.fchmod = never  # type: ignore
    os.fchown = never  # type: ignore
    os.chmod = never
    os.chown = never  # type: ignore
    os.chroot = never  # type: ignore
    os.fchdir = never  # type: ignore
    os.lchflags = never  # type: ignore
    os.lchmod = never  # type: ignore
    os.lchown = never  # type: ignore
    os.getcwd = lambda *args, **kwargs: "."
    os.chdir = never

    platform.system = lambda *args, **kwargs: __PLATFORM_SYSTEM__

    maximum_memory_bytes = 4 * 2**30  # 4 GiB
    if sys.platform != "win32" and maximum_memory_bytes is not None:
        import resource

        if not (platform.uname().system == "Darwin" and platform.processor() == ""):
            # memory limit is not available when using sandbox on MacOS
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    shutil.rmtree = never
    shutil.move = never
    shutil.chown = never

    subprocess.Popen = never

    sys.modules["ipdb"] = never  # type: ignore
    sys.modules["joblib"] = never  # type: ignore
    sys.modules["resource"] = never  # type: ignore
    sys.modules["psutil"] = never  # type: ignore
    sys.modules["tkinter"] = never  # type: ignore

    return


def __main__() -> None:
    """Python Executor server follows a protocol which is also implemented in
    the single-threaded client.

      - DAEMON: <starts up>
      - CLIENT: sends `PING $random_number`
      - DAEMON: responds `PONG $random_number` with the same number
    <for each request>
      - CLIENT: <writes code to new tmp dir>
      - CLIENT: sends a JSON object, with the following format:
                  * exactly 1 object per line, contains only ascii chars
                  * the object deserialize into a legal ProtocolInput
      - DAEMON: <import code, execute, capture, serialize, collect>
      - DAEMON: responds with a JSON object, with the following format:
                  * also exactly 1 object per line, contains only ascii chars
                  * the ProtocolOutput of which is the previous result
      - CLIENT: <receives ProtocolOutput, deserialize, pass to caller>
    <end of daemon>
      - CLIENT: send empty line / EOF
      - DAEMON: <no response, exits>
    """

    ping_flag = sys.stdin.readline().strip()
    pong_flag = "PONG " + ping_flag.split(" ")[-1]
    sys.stdout.write(pong_flag + "\n")
    sys.stdout.flush()

    while True:
        inp_raw = input().strip()
        inp_j = json.loads(inp_raw)
        if inp_j is None:
            break

        inp = ProtocolInput.model_validate(inp_j)
        # here we allow absolute imports, relative to the program root
        # e.g. `import my_module.pkg`, instead of `import .my_module.pkg`
        exec_root = (pathlib.Path(__file__).parent / inp.mod_run).resolve().as_posix()
        try:
            sys.path.append(exec_root)
            out = __execute_program(inp)
        except Exception:
            sys.path = [p for p in sys.path if p != exec_root]  # restore
            trace = traceback.format_exc()
            out = ProtocolOutput(
                session_id=inp.session_id,
                call_err=trace,
                call_ret=None,
                call_stdout="",
                call_dt=0.0,
            )
        out_j = out.model_dump_json()
        out_raw = json.dumps(json.loads(out_j), indent=None, ensure_ascii=True)

        sys.stdout.write(out_raw + "\n")
        sys.stdout.flush()

    return


def __execute_program(inp: ProtocolInput) -> ProtocolOutput:
    """The safe task of executing the program, guaranteeing type safety and
    capturing all results and stdio."""

    eval_scope: dict[str, Any] = {}
    for internal, mod_name, mod_alias, symbols in inp.imports:
        mod_path = [i for i in mod_name.split(".") if i]
        # resolve import
        try:
            if internal:
                mod_path = [inp.mod_run] + mod_path
                pkg = importlib.import_module("." + ".".join(mod_path), inp.mod_daemon)
                pkg_entry = None
            else:
                pkg = importlib.import_module(".".join(mod_path))
                pkg_entry = __import__(".".join(mod_path))  # exception: pack.age (as `pack`)
        except ModuleNotFoundError:
            continue
        # insert into scope
        if symbols is None:
            # differ between the following situations:
            #     `import numpy` -> `numpy`
            #     `import numpy as np` -> `np`
            #     `import sys.path` -> `sys.path`
            #     `import sys.path as sys_path` -> `sys_path`
            to_insert = pkg_entry if mod_alias is None and pkg_entry is not None else pkg
            eval_scope[mod_alias or mod_path[-1]] = to_insert
        else:
            for symbol, alias in symbols:
                if loaded := getattr(pkg, symbol, None):
                    eval_scope[alias] = loaded
        pass

    # parse, call, serialize
    call_arg_types = [(arg_type_s, eval(arg_type_s, eval_scope)) for arg_type_s in inp.func_args]
    call_kwarg_types = {k: (arg_type_s, eval(arg_type_s, eval_scope)) for k, arg_type_s in inp.func_kwargs.items()}
    call_ret_type = (inp.func_ret, eval(inp.func_ret, eval_scope))
    invoke_impl = eval_scope[inp.func_name]
    invoke_args = [__type_parse(typ, arg) for typ, arg in zip(call_arg_types, inp.call_args)]
    invoke_kwargs = {k: __type_parse(call_kwarg_types[k], arg) for k, arg in inp.call_kwargs.items()}
    invoke_ts_0 = time.time()
    invoke_err, invoke_ret, invoke_stdout = __with_stdio(
        lambda: invoke_impl(*invoke_args, **invoke_kwargs), inp.io_stdin
    )
    invoke_ts_1 = time.time()
    serialized_ret = __type_serialize(call_ret_type, invoke_ret) if invoke_err is None else None

    # cleanup
    pkg = None
    eval_scope = {}
    gc.collect()

    return ProtocolOutput(
        session_id=inp.session_id,
        call_err=invoke_err,
        call_ret=serialized_ret,
        call_stdout=invoke_stdout,
        call_dt=invoke_ts_1 - invoke_ts_0,
    )


def __type_serialize(typ: tuple[str, Any], value: Any) -> Any:
    model = pydantic.create_model(
        f"TypeParser<{typ[0]}>",
        __config__=pydantic.ConfigDict(
            arbitrary_types_allowed=True,
        ),
        the_value=(typ[1] if typ[1] is not None else type(None), ...),
    )
    serialized: Any = model(the_value=value)
    try:
        j_s = serialized.model_dump_json()
    except pydantic_core.PydanticSerializationError:
        j_s = pydantic_core.to_json(serialized, serialize_unknown=True)
    j = json.loads(j_s)
    return j["the_value"]


def __type_parse(typ: tuple[str, Any], serialized: Any) -> Any:
    model = pydantic.create_model(
        f"TypeParser<{typ[0]}>",
        __config__=pydantic.ConfigDict(
            arbitrary_types_allowed=True,
        ),
        the_value=(typ[1] if typ[1] is not None else type(None), ...),
    )
    parsed: Any = model(the_value=serialized)
    return parsed.the_value


def __with_stdio(func: Callable[[], T], stdin: str) -> tuple[str | None, T | None, str]:
    """Pipe stdin and capture stdout of given function. Standard error is not
    blocked and inherits from the caller."""

    the_stdin = sys.stdin
    the_stdout = sys.stdout
    hook_in = io.StringIO(stdin)
    hook_out = io.StringIO()
    try:
        sys.stdin = hook_in
        sys.stdout = hook_out
        ret = func()
        stdout = hook_out.getvalue()
        return None, ret, stdout
    except Exception as _:
        trace = traceback.format_exc()
        return trace, None, hook_out.getvalue()
    finally:
        sys.stdin = the_stdin
        sys.stdout = the_stdout


if __name__ == "__main__":
    __guard__()
    __main__()
