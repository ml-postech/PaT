import pathlib
import sys
from typing import Any, Iterable, Literal, cast, overload

import rich

from .executor import LrtExecutor
from .parser import LrtParser
from .py_exec.executor import PyExecutor
from .py_parse.parser import PyParser
from .types import (
    CodeBlock,
    JsonSerializable,
    LrtExecutionEnv,
    LrtExecutionResult,
    LrtImport,
    LrtNode,
    LrtProgram,
    LrtSolution,
    ModuleName,
    SymbolName,
    TLrtNode,
)


class LangRT:
    def __init__(self, lang: str, executor: LrtExecutor, parser: LrtParser):
        self.lang: Literal["python"] = cast(Any, lang)
        """Markdown language identifier."""
        self._exec = executor
        self._parse = parser

    @staticmethod
    def python(
        sandbox_root: pathlib.Path,
        parallelism: int,
        console: rich.console.Console = rich.get_console(),
        macos_sandbox_bin: str | None = None,
        python_bin: str | None = None,
    ) -> "LangRT":
        return LangRT(
            lang="python",
            executor=PyExecutor(
                sandbox_root=sandbox_root,
                parallelism=parallelism,
                console=console,
                macos_sandbox_bin=macos_sandbox_bin or "sandbox-exec",
                python_bin=python_bin or sys.executable,
            ),
            parser=PyParser(),
        )

    ###########################################################################

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def parse(self, module: tuple[SymbolName, ...], code: CodeBlock) -> LrtProgram:
        """Convert raw code to manipulable program structure."""

        code = self._parse.fmt_code(code)
        nodes = self._parse.parse_code(code)
        return LrtProgram(module=module, nodes=nodes)

    def fmt(self, *args: LrtProgram | LrtNode | list[LrtNode]) -> CodeBlock:
        """Just format the program(s) / node(s) into raw code. AST is not
        modified."""

        nodes = _as_nodes(*args)
        return self._parse.fmt_nodes(nodes, organize_imports=False)

    @overload
    def prettify(self, it: LrtSolution, organize_imports: bool = False) -> LrtSolution: ...
    @overload
    def prettify(self, it: LrtProgram, organize_imports: bool = False) -> LrtProgram: ...
    @overload
    def prettify(self, it: TLrtNode, organize_imports: bool = False) -> TLrtNode: ...
    def prettify(
        self, it: LrtSolution | LrtProgram | LrtNode, organize_imports: bool = False
    ) -> LrtSolution | LrtProgram | LrtNode:
        """Re-arrange the program structure to deduplicate nodes, organize
        imports and format raw code. Structure may change after prettify."""

        if isinstance(it, LrtSolution):
            return LrtSolution(modules=[self.prettify(prog) for prog in it.modules])
        elif isinstance(it, LrtProgram):
            nodes = self._parse.deduplicate_nodes(it.nodes)
            code = self._parse.fmt_nodes(nodes, organize_imports=organize_imports)
            return LrtProgram(module=it.module, nodes=self._parse.parse_code(code))
        else:
            code = self._parse.fmt_nodes([it], organize_imports=organize_imports)
            nodes = self._parse.parse_code(code)
            assert len(nodes) == 1
            node = nodes[0]
            assert isinstance(node, type(it))
            return node

    def pretty_fmt(self, *args: LrtProgram | LrtNode | list[LrtNode], organize_imports: bool = False) -> CodeBlock:
        """Prettify and format."""

        nodes = _as_nodes(*args)
        program = LrtProgram(module=(), nodes=nodes)
        program = self.prettify(program)
        return self._parse.fmt_nodes(program.nodes, organize_imports=organize_imports)

    async def run_program(
        self,
        program: LrtProgram,
        func_name: SymbolName,
        args: list[JsonSerializable],
        kwargs: dict[str, JsonSerializable],
        stdin: str = "",
        timeout: float = 1.0,
    ) -> LrtExecutionResult:
        solution = LrtSolution(modules=[program])
        return await self.run_solution(
            solution=solution,
            from_module=program.module,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            stdin=stdin,
            timeout=timeout,
        )

    async def run_solution(
        self,
        solution: LrtSolution,
        from_module: tuple[SymbolName, ...],
        func_name: SymbolName,
        args: list[JsonSerializable],
        kwargs: dict[str, JsonSerializable],
        stdin: str = "",
        timeout: float = 1.0,
    ) -> LrtExecutionResult:
        # get entrypoint module
        _entry_progs = [prog for prog in solution.modules if prog.module == from_module]
        if not _entry_progs:
            raise ValueError(f"entrypoint module {from_module} not found in solution")
        entry_prog = _entry_progs[0]
        # identify function
        _entry_funcs = [f for f in entry_prog.nodes if f.kind == "function" and f.name == func_name]
        if not _entry_funcs:
            raise ValueError(f"entrypoint function '{func_name}' not found in module")
        entry_func = _entry_funcs[0]

        # extract all code
        code: dict[ModuleName, CodeBlock] = {}
        for mod in solution.modules:
            mod_name = ".".join(mod.module)
            raw_code = self.fmt(mod)
            code[mod_name] = raw_code

        # pull all imports from entrypoint (best-effort resolution)
        imports: list[tuple[bool, ModuleName, SymbolName | None, list[tuple[SymbolName, SymbolName]] | None]] = []
        for node in entry_prog.nodes:
            if node.kind == "import":
                imports.extend(list(_cast_exec_env_imports(node)))
        # add additional imports that are always applicable
        imports.append((True, ".".join(from_module), None, [(func_name, func_name)]))  # entrypoint func
        imports.append((False, "typing", None, [("Any", "Any")]))  # add Any import

        # fetch function signature according to invocation input
        func_args: list[str] = []
        func_kwargs: dict[str, str] = {}
        for i, (arg_name, arg_type, arg_default) in enumerate(entry_func.args):
            if i < len(args):
                func_args.append(arg_type or "Any")
            else:
                func_kwargs[arg_name] = arg_type or "Any"
        func_ret = entry_func.ret or "Any"

        env = LrtExecutionEnv(
            code=code,
            imports=imports,
            func_name=func_name,
            func_args=func_args,
            func_kwargs=func_kwargs,
            func_ret=func_ret,
        )
        return await self._exec.run(env=env, args=args, kwargs=kwargs, stdin=stdin, timeout=timeout)

    def close(self):
        self._exec.close()

    pass


def _cast_exec_env_imports(
    imp: LrtImport,
) -> Iterable[tuple[bool, ModuleName, SymbolName | None, list[tuple[SymbolName, SymbolName]] | None]]:
    if imp.level == 0:
        if not imp.module:
            # import numpy / import numpy as np
            mod, alias = imp.symbols[0]
            yield (False, mod, alias if alias != mod else None, None)
            yield (True, mod, alias if alias != mod else None, None)
        else:
            # from numpy.types import NdArray (symbol)
            mod = ".".join(imp.module)
            yield (False, mod, None, imp.symbols)
            yield (True, mod, None, imp.symbols)
            # from numpy import types as np_types (module)
            if len(imp.symbols) == 1:
                sym_2, alias_2 = imp.symbols[0]
                mod_2 = ".".join(list(imp.module) + [sym_2])
                yield (False, mod_2, alias_2 if alias_2 != sym_2 else None, None)
    else:
        # from .types import MyType
        mod = ".".join(imp.module)
        yield (True, mod, None, imp.symbols)
    return


def _as_nodes(*its: LrtProgram | LrtNode | list[LrtNode]) -> list[LrtNode]:
    nodes: list[LrtNode] = []
    for it in its:
        if isinstance(it, LrtProgram):
            nodes.extend(it.nodes)
        elif isinstance(it, list):
            nodes.extend(it)
        else:
            nodes.append(it)
    return nodes
