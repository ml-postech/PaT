import datetime
import inspect
import json
import sys
import threading
import traceback
from typing import Any, Literal, Type

import rich.console
import rich.layout
import rich.live
import rich.markdown
import rich.panel
import rich.status
import rich.syntax
import rich.table

from .pyctx import PyCtx
from .types import anything_into_dict


class ParaStatus:
    """The (unified) entrypoint to rich.Status, with singleton-managed parallel
    status bars."""

    _status: dict[rich.console.Console, tuple[rich.live.Live, list[rich.status.Status]]] = {}

    def __init__(self, console: rich.console.Console, title: str, silent: bool):
        self._console = console
        self._title = title
        self._silent = silent
        self._it = rich.status.Status(self._title)

    def __enter__(self) -> "ParaStatus":
        if self._silent:
            return self
        if self._console not in ParaStatus._status:
            live = rich.live.Live(rich.console.Group(self._it))
            live.start()
            ParaStatus._status[self._console] = (live, [self._it])
        else:
            live, its = ParaStatus._status[self._console]
            its.append(self._it)
            live.update(rich.console.Group(*its))
            ParaStatus._status[self._console] = (live, its)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._silent:
            return
        live, its = ParaStatus._status[self._console]
        its = [i for i in its if i is not self._it]
        if its:
            live.update(rich.console.Group(*its))
            live.start()
            ParaStatus._status[self._console] = (live, its)
        else:
            live.stop()
            ParaStatus._status.pop(self._console)
        return

    def update(self, text: str) -> None:
        if self._silent:
            return
        self._it.update(text)

    pass


class Console:
    """The (unified) entrypoint to rich.Console."""

    _console: rich.console.Console | None = None

    @classmethod
    def get_console(cls) -> rich.console.Console:
        if cls._console is None:
            cls._console = rich.console.Console()
        return cls._console

    @classmethod
    def get_status(cls, title: str, silent: bool = False) -> ParaStatus:
        console = cls.get_console()
        return ParaStatus(console, title, silent)

    pass


LoggerEventType = Literal["epic", "error", "warn", "trace", "string", "exec_result", "code", "chat_history"]


class Logger:
    def __init__(self, hide_event_types: list[LoggerEventType] = []):
        self._scope_ctx = PyCtx[str]("dqllm_logger_scope")
        self._writer = StdoutLogWriter(hide_event_types=hide_event_types)
        self._lock = threading.Lock()

    def in_scope(self, scope: str) -> None:
        """Declare the entering of a new (deeper) layer of logging context."""

        with self._lock:
            self._scope_ctx.append(scope, offset=1)
        return

    def epic(self, content: str) -> None:
        """Big title. Very significant. Occupies many lines."""

        with self._lock:
            ts, position, title = self.__get_env()
            try:
                self._writer.write_epic(ts, position, title, content)
            except Exception:
                traceback.print_exc()
        return

    def error(self, text: str) -> None:
        with self._lock:
            ts, position, title = self.__get_env()
            try:
                self._writer.write_error(ts, position, title, text)
            except Exception:
                traceback.print_exc()
        return

    def warn(self, text: str) -> None:
        with self._lock:
            ts, position, title = self.__get_env()
            try:
                self._writer.write_warn(ts, position, title, text)
            except Exception:
                traceback.print_exc()
        return

    def string(self, content: str) -> None:
        """Report minor, retry-able issues with this."""

        with self._lock:
            ts, position, title = self.__get_env()
            try:
                self._writer.write_string(ts, position, title, content)
            except Exception:
                traceback.print_exc()
        return

    def object(self, typ: Type, obj: Any) -> None:
        obj_j = anything_into_dict((str(typ), typ), obj)
        content = json.dumps(obj_j, indent=2, ensure_ascii=False)
        with self._lock:
            ts, position, title = self.__get_env()
            try:
                self._writer.write_string(ts, position, title, content)
            except Exception:
                traceback.print_exc()
        return

    def exec_result(self, important: bool, content: str) -> None:
        with self._lock:
            ts, position, title = self.__get_env()
            try:
                self._writer.write_exec_result(ts, position, title, important, content)
            except Exception:
                traceback.print_exc()
        return

    def code(self, lang: str, label: str, content: str) -> None:
        with self._lock:
            ts, position, title = self.__get_env()
            try:
                self._writer.write_code(ts, position, title, lang, label, content)
            except Exception:
                traceback.print_exc()
        return

    def chat_history(self, messages: list[dict]) -> None:
        with self._lock:
            ts, position, title = self.__get_env()
            try:
                self._writer.write_chat_history(ts, position, title, messages)
            except Exception:
                traceback.print_exc()
        return

    def trace(self) -> None:
        with self._lock:
            ts, position, title = self.__get_env()
            _type, _value, _tb = sys.exc_info()
            if _tb is not None:
                trace = "".join(traceback.format_exception(_type, _value, _tb))
            else:
                lines: list[str] = []
                lines.append("Traceback (most recent call last):")
                lines.append("")
                for frm in inspect.stack():
                    _path = frm.filename
                    _no = frm.lineno
                    _func = frm.function
                    _line = frm.code_context
                    _line = _line[0] if _line else "<hidden code>"
                    lines.append(f'  File "{_path}", line {_no}, in {_func}')
                    lines.append("    " + _line.strip())
                lines.append("")
                lines.append("(): trace only, no exceptions found")
                trace = "\n".join(lines)
            trace = trace.strip().split("\n")
            try:
                self._writer.write_trace(ts, position, title, trace)
            except Exception:
                traceback.print_exc()
        return

    def __get_env(self) -> tuple[datetime.datetime, str, list[str]]:
        ts = datetime.datetime.now()
        title = self._scope_ctx.get(offset=2) or ["#root"]
        frame = inspect.stack()[2]
        frame_fn = frame.filename.replace("\\", "/").split("/")[-1]
        frame_no = frame.lineno
        return ts, f"{frame_fn}:{frame_no}", title

    pass


class StdoutLogWriter:
    def __init__(self, hide_event_types: list[LoggerEventType]):
        self._hide_event_types = hide_event_types
        self._con = Console.get_console()

    def write_epic(self, ts: datetime.datetime, position: str, title: list[str], content: str) -> None:
        if "epic" in self._hide_event_types:
            return
        table = rich.table.Table()
        head_l = " / ".join(title)
        head_r = "\\[" + position + "] " + str(ts)
        head_pad = " " * max(8, 72 - len(head_l) - len(head_r))
        head = f"{head_l}{head_pad}{head_r}"
        table.add_column(head, justify="left", header_style="grey62", style="bold italic green_yellow on grey11")
        table.add_row(content)
        self._con.print("\n")
        self._con.print(table)
        self._con.print("\n")

    def write_error(self, ts: datetime.datetime, position: str, title: list[str], content: str) -> None:
        if "error" in self._hide_event_types:
            return
        self.__write_header(ts, position, title)
        r_layout = rich.table.Table(show_header=False, box=None)
        r_layout.add_column("label", justify="right", width=2)
        r_layout.add_column("message")
        r_layout.add_row("", rich.panel.Panel(content, style="red1", border_style="red1"))
        self._con.print(r_layout)
        self._con.print("")

    def write_warn(self, ts: datetime.datetime, position: str, title: list[str], content: str) -> None:
        if "warn" in self._hide_event_types:
            return
        self.__write_header(ts, position, title)
        r_layout = rich.table.Table(show_header=False, box=None)
        r_layout.add_column("label", justify="right", width=2)
        r_layout.add_column("message")
        r_layout.add_row("", rich.panel.Panel(content, style="gold1", border_style="orange1"))
        self._con.print(r_layout)
        self._con.print("")

    def write_trace(self, ts: datetime.datetime, position: str, title: list[str], trace: list[str]) -> None:
        if "trace" in self._hide_event_types:
            return
        self.__write_header(ts, position, title)
        r_layout = rich.table.Table(show_header=False, box=None)
        r_layout.add_column("label", justify="right", width=2)
        r_layout.add_column("message")
        r_layout.add_row("", rich.panel.Panel("\n".join(trace), style="grey42", border_style="grey30"))
        self._con.print(r_layout)
        self._con.print("")

    def write_string(self, ts: datetime.datetime, position: str, title: list[str], content: str) -> None:
        if "string" in self._hide_event_types:
            return
        self.__write_header(ts, position, title)
        r_layout = rich.table.Table(show_header=False, box=None)
        r_layout.add_column("label", justify="right", width=2)
        r_layout.add_column("message")
        r_layout.add_row("", rich.panel.Panel(content))
        self._con.print(r_layout)
        self._con.print("")

    def write_exec_result(
        self, ts: datetime.datetime, position: str, title: list[str], important: bool, content: str
    ) -> None:
        if "exec_result" in self._hide_event_types and not important:
            return
        self.__write_header(ts, position, title)
        r_layout = rich.table.Table(show_header=False, box=None)
        r_layout.add_column("label", justify="right", width=2)
        r_layout.add_column("message")
        r_layout.add_row("", rich.panel.Panel(content))
        self._con.print(r_layout)
        self._con.print("")

    def write_code(
        self,
        ts: datetime.datetime,
        position: str,
        title: list[str],
        lang: str,
        label: str,
        content: str,
    ) -> None:
        if "code" in self._hide_event_types:
            return
        self.__write_header(ts, position, title)
        r_layout = rich.table.Table(show_header=False, box=None)
        r_layout.add_column("label", justify="right", width=2)
        r_layout.add_column("message")
        r_code = rich.syntax.Syntax(content, lang, word_wrap=True)
        r_code = rich.panel.Panel(r_code, title=f"{lang} :: [italic]{label}[/italic]", title_align="left")
        r_layout.add_row("", r_code)
        self._con.print(r_layout)
        self._con.print("")

    def write_chat_history(self, ts: datetime.datetime, position: str, title: list[str], messages: list[dict]) -> None:
        if "chat_history" in self._hide_event_types:
            return
        self.__write_header(ts, position, title)
        r_layout = rich.table.Table(show_header=False, box=None)
        r_layout.add_column("num", justify="right", width=2 + 10)
        r_layout.add_column("message")
        for i, message in enumerate(messages):
            role: str = message.get("role", "n/a")
            content: str | None = message.get("content", None)
            if not content:
                continue
            r_label = f"{'Out' if role == 'user' else 'In'} [{i}]:"
            r_msg = rich.panel.Panel(
                rich.markdown.Markdown(content, justify="left"),
                title=f"role: [underline]{role}[/underline]",
                title_align="left",
            )
            r_layout.add_row(r_label, r_msg)
        r_layout.add_row("End [ ]:", f"Total {len(messages)} messages.")
        self._con.print(r_layout)
        self._con.print("")

    def __write_header(self, ts: datetime.datetime, position: str, title: list[str]) -> None:
        self._con.rule(
            f"[bold green]──[/bold green] {' / '.join(title)} :: \\[{position}] {str(ts)}",
            align="left",
        )

    pass


class SilentStatus:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def stop(self):
        pass

    def __enter__(self) -> "SilentStatus":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    pass
