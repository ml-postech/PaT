import ast
from typing import Callable

import black

from ...utils.strings import wrap_string_as_triple_quotes
from ...utils.types import guard_never
from ..parser import LrtParser
from ..types import (
    CodeBlock,
    LrtConstantDef,
    LrtEnumDef,
    LrtFunctionDef,
    LrtImport,
    LrtNode,
    LrtStructDef,
    SymbolName,
    TypeName,
    Value,
)
from .utils import ppy_get_python_packages, ppy_organize_code_imports


class PyParser(LrtParser):
    ###########################################################################
    #   parsing

    def parse_code(self, code: CodeBlock) -> list[LrtNode]:
        code = self.fmt_code(code)
        result: list[LrtNode] = []
        for chunk in self._parse_code_splits(code):
            root = ast.parse(chunk)
            if len(root.body) != 1:
                continue
            parsed = self.parse_node(chunk, root.body[0])
            if parsed is not None:
                result.append(parsed)
        return result

    def _parse_code_splits(self, code: CodeBlock) -> list[CodeBlock]:
        lines: list[tuple[int | None, str]] = []
        for line in code.split("\n"):
            line = line.rstrip()
            if not line:
                lines.append((None, line))
                continue
            indent = len(line) - len(line.lstrip())
            lines.append((indent, line))

        # special case 1:
        #     def foo(arg1: int, arg2: str, ...,
        #             more_args: list[str]
        #     ) -> None:
        # special case 2:
        #     the_str = """
        #     SOME_TEXT
        #     """
        # special case 3:
        #     import foo
        #     import bar
        # special case 4:
        #     from module import (
        #         foo,
        #         bar,
        #     )
        # blocking strategy: indent = [0 -> ... -> None] -> 0 | EOF
        buffer: list[str] = []
        chunks: list[list[str]] = []
        stage = 0  # 0: [], 1: [0], 2: [0, None]
        for indent, line in lines:
            if (line.startswith("from ") or line.startswith("import ")) and line[-1] != "(":
                chunks.append(buffer)
                chunks.append([line])
                buffer = []
                stage = 0
                continue
            # standard automata
            if stage == 0:
                if indent == 0:
                    stage = 1
                buffer.append(line)
            elif stage == 1:
                if indent is None:
                    stage = 2
                buffer.append(line)
            elif stage == 2:
                if indent == 0:
                    chunks.append(buffer)
                    buffer = []
                    stage = 1
                buffer.append(line)
        if buffer:
            chunks.append(buffer)

        ret: list[str] = []
        for chunk in chunks:
            block = "\n".join(chunk)
            if not block.strip():
                continue
            ret.append(self.fmt_code(block))
        return ret

    def parse_node(self, code: CodeBlock, node: ast.stmt) -> LrtNode | None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return self.parse_import(code, node)
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(i) for i in node.bases]
            if any("Enum" in _ for _ in bases):
                return self.parse_enum_def(code, node)
            elif any("BaseModel" in _ for _ in bases):
                return self.parse_struct_def(code, node)
        elif isinstance(node, ast.FunctionDef):
            return self.parse_function_def(code, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            return self.parse_constant_def(code, node)
        return None

    def parse_import(self, code: CodeBlock, node: ast.Import | ast.ImportFrom) -> LrtImport:
        if isinstance(node, ast.Import):
            if len(node.names) == 0:
                raise SyntaxError("import statement has no targets")
            if len(node.names) != 1:
                self._warn(f"import statement has multiple targets: {ast.unparse(node)}")
            name = node.names[0].name
            module = name.split(".")
            symbols = []
            if node.names[0].asname is not None:
                symbols.append((module.pop(), node.names[0].asname))
            return LrtImport(kind="import", code=self.fmt_code(code), module=tuple(module), symbols=symbols, level=0)
        elif isinstance(node, ast.ImportFrom):
            return LrtImport(
                kind="import",
                code=self.fmt_code(code),
                module=tuple(i for i in (node.module or "").split(".") if i),
                symbols=[(i.name, i.asname or i.name) for i in node.names],
                level=node.level or 0,
            )
        raise SyntaxError(f"unexpected import node: {ast.dump(node)}")

    def parse_enum_def(self, code: CodeBlock, node: ast.ClassDef) -> LrtEnumDef:
        return LrtEnumDef(
            kind="enum",
            code=self.fmt_code(code),
            name=node.name,
            docstring=self._parse_docstring(node.body),
            options=self._parse_enum_def_options(node.body),
        )

    def _parse_enum_def_options(self, nodes: list[ast.stmt]) -> dict[SymbolName, int | str]:
        ret = {}
        for node in nodes:
            opt = self._parse_enum_def_option(node)
            if opt is None:
                continue
            name, _typ, val = opt
            try:
                # real_val = json.loads(val or "null")
                real_val = ast.literal_eval(val or "None")
            except Exception:
                self._warn(f"enum option '{name}' has invalid value: {val}")
                continue
            if not isinstance(real_val, (int, str)):
                self._warn(f"enum option '{name}' has invalid value: {val}")
                continue
            ret[name] = real_val
        return ret

    def _parse_enum_def_option(self, node: ast.stmt) -> tuple[SymbolName, TypeName | None, Value | None] | None:
        return self._parse_struct_def_field(node)

    def parse_struct_def(self, code: CodeBlock, node: ast.ClassDef) -> LrtStructDef:
        return LrtStructDef(
            kind="struct",
            code=self.fmt_code(code),
            name=node.name,
            docstring=self._parse_docstring(node.body),
            fields=self._parse_struct_def_fields(node.body),
        )

    def _parse_struct_def_fields(self, nodes: list[ast.stmt]) -> dict[SymbolName, tuple[TypeName, Value | None]]:
        ret = {}
        for node in nodes:
            field = self._parse_struct_def_field(node)
            if field is None:
                continue
            name, typ, val = field
            if typ is None:
                self._warn(f"struct field '{name}' is missing type annotation")
            ret[name] = (typ, val)
        return ret

    def _parse_struct_def_field(self, node: ast.stmt) -> tuple[SymbolName, TypeName | None, Value | None] | None:
        if isinstance(node, ast.Assign):
            names = [i.id for i in node.targets if isinstance(i, ast.Name)]
            if len(names) != 1:
                self._warn(f"enum option has multiple targets: {ast.unparse(node)}")
                return None
            name = names[0]
            typ = None
            val = ast.unparse(node.value)
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                return None
            name = node.target.id
            typ = ast.unparse(node.annotation)
            val = ast.unparse(node.value) if node.value is not None else None
        else:
            return None
        return name, typ, val

    def parse_function_def(self, code: CodeBlock, node: ast.FunctionDef) -> LrtFunctionDef:
        return LrtFunctionDef(
            kind="function",
            code=self.fmt_code(code),
            name=node.name,
            docstring=self._parse_docstring(node.body),
            args=self._parse_function_def_args(node.args),
            ret=self._parse_function_def_ret(node),
            implemented=self._parse_function_def_if_implemented(node.body),
            body=self._parse_function_def_body(code),
        )

    def _parse_function_def_args(self, node: ast.arguments) -> list[tuple[SymbolName, TypeName | None, Value | None]]:
        ret: list[tuple[SymbolName, TypeName | None, Value | None]] = []
        for i, arg in enumerate(node.args):
            name, typ = self._parse_function_def_arg(arg)
            default = None
            if i + len(node.defaults) >= len(node.args):
                _d = node.defaults[i + len(node.defaults) - len(node.args)]
                default = ast.unparse(_d)
            ret.append((name, typ, default))
        return ret

    def _parse_function_def_ret(self, node: ast.FunctionDef) -> TypeName | None:
        if node.returns is None:
            return None
        return ast.unparse(node.returns)

    def _parse_function_def_if_implemented(self, nodes: list[ast.stmt]) -> bool:
        docstring = self._parse_docstring(nodes)
        if docstring is not None:
            nodes = nodes[1:]

        if len(nodes) == 0:
            return False
        last_node = nodes[-1]
        last_expr = ast.unparse(last_node).strip()
        if len(nodes) == 1 and last_expr == "pass":
            return False
        patterns = [
            "...",
            "raise NotImplementedError",
            "raise NotImplementedError()",
        ]
        return last_expr not in patterns

    def _parse_function_def_body(self, code: CodeBlock) -> CodeBlock:
        # shrink header to exactly only 1 line
        code = black.format_str(code, mode=black.Mode(line_length=100000000))
        lines = code.rstrip().split("\n")[1:]
        # skip docstring
        ptr = 0
        begin_flag, end_flag = "", ""
        while ptr < len(lines) and not lines[ptr].strip():
            ptr += 1
        if ptr < len(lines):
            first_line = lines[ptr].strip()
            if first_line.startswith("'''"):
                begin_flag, end_flag = "'''", "'''"
            elif first_line.startswith("r'''"):
                begin_flag, end_flag = "r'''", "'''"
            elif first_line.startswith('"""'):
                begin_flag, end_flag = '"""', '"""'
            elif first_line.startswith('r"""'):
                begin_flag, end_flag = 'r"""', '"""'
            # skip single line docstring header
            if first_line == begin_flag:
                ptr += 1
        if end_flag:
            while ptr < len(lines):
                if lines[ptr].strip().endswith(end_flag):
                    ptr += 1
                    break
                ptr += 1
        buffer = lines[ptr:]
        return "\n".join(buffer).strip("\n")

    def parse_constant_def(self, code: CodeBlock, node: ast.Assign | ast.AnnAssign) -> LrtConstantDef:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                self._warn(f"constant assignment has multiple targets: {ast.unparse(node)}")
            return LrtConstantDef(
                kind="constant",
                code=self.fmt_code(code),
                name=ast.unparse(node.targets[0]),
                type=ast.unparse(node.annotation) if isinstance(node, ast.AnnAssign) else None,
                value=ast.unparse(node.value),
            )
        elif isinstance(node, ast.AnnAssign):
            return LrtConstantDef(
                kind="constant",
                code=self.fmt_code(code),
                name=ast.unparse(node.target),
                type=ast.unparse(node.annotation),
                value=ast.unparse(node.value) if node.value is not None else "None",
            )
        else:
            guard_never(node)

    ###########################################################################
    #   custom parser tools

    def _parse_docstring(self, nodes: list[ast.stmt]) -> str | None:
        if len(nodes) == 0:
            return None
        expr = nodes[0]
        if not isinstance(expr, ast.Expr):
            return None
        const = expr.value
        if not isinstance(const, ast.Constant):
            return None
        val = const.value
        if not isinstance(val, str):
            return None
        return val

    def _parse_function_def_arg(self, node: ast.arg) -> tuple[SymbolName, TypeName | None]:
        name = node.arg
        if node.annotation is None:
            return name, None
        typ = ast.unparse(node.annotation)
        return name, typ

    ###########################################################################
    #   tools

    def make_import_from(
        self, module: tuple[SymbolName, ...], symbols: list[tuple[SymbolName, SymbolName]], level: int
    ) -> LrtImport:
        fmt_symbols = [src if alias == src else f"{src} as {alias}" for src, alias in symbols]

        if level == 0 and not module:
            code = f"import {symbols[0][0]} as {symbols[0][1]}"
        elif level == 0 and len(symbols) == 0:
            code = f"import {'.'.join(module)}"
        elif level == 0 and len(symbols) > 0:
            code = f"from {'.'.join(module)} import {', '.join(fmt_symbols)}"
        elif level > 0 and len(symbols) == 0:
            x_mod = list(module)
            x_symbols = [x_mod.pop(-1)]
            code = f"from {'.' * level}{'.'.join(x_mod)} import {', '.join(x_symbols)}"
        elif level > 0 and len(symbols) > 0:
            code = f"from {'.' * level}{'.'.join(module)} import {', '.join(fmt_symbols)}"
        else:
            raise NotImplementedError()

        return LrtImport(kind="import", code=self.fmt_code(code), module=module, symbols=symbols, level=level)

    def make_stub_function_def_from_params(
        self,
        name: SymbolName,
        docstring: str | None,
        args: list[tuple[SymbolName, TypeName | None, Value | None]],
        ret: TypeName | None,
    ) -> LrtFunctionDef:
        new_draft = LrtFunctionDef(
            kind="function",
            code="",
            name=name,
            docstring=docstring,
            args=args,
            ret=ret,
            implemented=False,
            body="    raise NotImplementedError()",
        )
        return self.fmt_function_def(new_draft)

    def make_stub_function_def_from_func(self, func: LrtFunctionDef) -> LrtFunctionDef:
        new_draft = LrtFunctionDef(
            kind="function",
            code=func.code,
            name=func.name,
            docstring=func.docstring,
            args=func.args,
            ret=func.ret,
            implemented=False,
            body="    raise NotImplementedError()",
        )
        return self.fmt_function_def(new_draft)

    def fmt_nodes(self, nodes: list[LrtNode], organize_imports: bool) -> CodeBlock:
        n_imports: list[LrtImport] = [n for n in nodes if n.kind == "import"]
        n_rest: list[LrtNode] = [n for n in nodes if n.kind != "import"]
        result = [self.fmt_imports(n_imports) + "\n\n"]
        for n in n_rest:
            result.append("\n\n" + n.code + "\n\n")
        code = "".join(result)
        code = self.fmt_code(code)
        if organize_imports:
            code = ppy_organize_code_imports(code)
        return code

    def fmt_code(self, code: CodeBlock) -> CodeBlock:
        # i know ast eats up comments and other styles that matters a lot to code
        # comprehension. so we'll update them back later as another pass.
        code = black.format_str(code, mode=black.Mode())
        return code

    def fmt_imports(self, imports: list[LrtImport]) -> CodeBlock:
        l_builtins: list[LrtImport] = []
        l_packages: list[LrtImport] = []
        l_local: list[LrtImport] = []
        for imp in imports:
            if len(imp.module) > 0 and imp.module[0] in ppy_get_python_packages(incl_std=True):
                l_builtins.append(imp)
            elif imp.level > 0:
                l_local.append(imp)
            else:
                l_packages.append(imp)

        def _fmt_imp(xs: list[LrtImport]) -> str:
            # merge modules
            sxs: dict[str, list[LrtImport]] = {}
            ret: list[LrtImport] = []
            for imp in xs:
                if imp.level == 0 and len(imp.module) + len(imp.symbols) == 1:
                    ret.append(imp)
                    continue
                key = "." * imp.level + ".".join(imp.module)
                sxs.setdefault(key, []).append(imp)
            l_sxs = sorted(sxs.items())
            for key, imps in l_sxs:
                symbols = sum([imp.symbols for imp in imps], [])
                symbols.sort()
                n_imp = self.make_import_from(imps[0].module, symbols, imps[0].level)
                ret.append(n_imp)
            ret.sort(key=lambda x: (x.level, x.module))
            # then format
            return "\n".join([imp.code.strip("\n") for imp in ret])

        s_imps = [_fmt_imp(l_builtins), _fmt_imp(l_packages), _fmt_imp(l_local)]
        s_imps = [s for s in s_imps if s]
        return "\n\n".join(s_imps)

    def fmt_function_sig(self, func: LrtFunctionDef) -> str:
        """... -> 'def name(arg: type, ...) -> type:'"""

        code = ""
        code += f"def {func.name}("
        f_args: list[str] = []
        for arg, typ, val in func.args:
            if typ is not None and val is not None:
                f_args.append(f"{arg}: {typ} = {val}")
            elif typ is not None and val is None:
                f_args.append(f"{arg}: {typ}")
            elif typ is None and val is not None:
                f_args.append(f"{arg} = {val}")
            else:
                f_args.append(f"{arg}")
        code += ", ".join(f_args)
        if func.ret is not None:
            code += f") -> {func.ret}:\n"
        else:
            code += "):\n"
        return code

    def fmt_function_def(self, func: LrtFunctionDef) -> LrtFunctionDef:
        code = self.fmt_function_sig(func)
        if func.docstring is not None:
            docstring = wrap_string_as_triple_quotes(func.docstring)
            code += f"    {docstring}\n\n"
        code += func.body
        code = code.strip("\n")
        if "\n" not in code:
            code += "\n    pass\n"
        code = self.fmt_code(code)

        return LrtFunctionDef(
            kind="function",
            code=code,
            name=func.name,
            docstring=func.docstring,
            args=func.args,
            ret=func.ret,
            implemented=func.implemented,
            body=func.body,
        )

    def is_function_code_compliant(self, impl: LrtFunctionDef, sig: LrtFunctionDef, strict_mode: bool) -> bool:
        """Check if the implementation is compliant with the signature. We allow
        the implementation to further restrict the signature, but not widen it."""

        if strict_mode:
            # TODO: doesn't support generic in unions yet ('list[str] | None')
            #       we honestly hope things don't run into this

            def _strip_generic(t: TypeName) -> tuple[TypeName, list[TypeName] | None]:
                if "[" not in t:
                    return t, None
                base, generics = t.split("[", 1)
                generics = generics[:-1].split(",")
                return base.strip(), [g.strip() for g in generics if g.strip()]

            def _type_compliant(cur: TypeName | None, sig: TypeName | None) -> bool:
                cur_t = cur or "Any"
                sig_t = sig or "Any"
                if sig_t == "Any":
                    return True
                cur_b, cur_g = _strip_generic(cur_t)
                sig_b, sig_g = _strip_generic(sig_t)
                if cur_b.lower() != sig_b.lower():
                    return False
                if sig_g is not None:
                    if cur_g is None:
                        return False
                    if len(cur_g) != len(sig_g):
                        return False
                    for c, s in zip(cur_g, sig_g):
                        if not _type_compliant(c, s):
                            return False
                return True

            if len(impl.args) != len(sig.args):
                return False
            for (cur_name, cur_type, cur_val), (sig_name, sig_type, sig_val) in zip(impl.args, sig.args):
                # TODO: do we care about typos in the sig?
                if cur_name != sig_name:  # this needs to be byte-exact however
                    return False
                if not _type_compliant(cur_type, sig_type):
                    return False
                if cur_val != sig_val:
                    return False
            # special case: if impl left ret None, ignore the result
            if impl.ret is not None:
                if not _type_compliant(impl.ret, sig.ret):
                    return False
            return True
        else:
            # non-strict mode
            if len(impl.args) != len(sig.args):
                return False
            return True
        pass

    def iter_repl_statements(self, flush_when: Callable[[str], bool], inp: str) -> list[str]:
        try:
            fmt_inp = black.format_str(inp, mode=black.Mode(line_length=1000000))
            inp = fmt_inp
        except Exception as _:
            pass

        buffer: list[str] = []
        result: list[str] = []
        for line in inp.split("\n"):
            buffer.append(line)
            if flush_when(line):
                result.append("\n".join(buffer))
                buffer.clear()
        if buffer:
            result.append("\n".join(buffer))

        cleaned: list[str] = []
        for case in result:
            case = case.strip()
            if not case:
                continue
            try:
                fmt_case = black.format_str(case, mode=black.Mode(line_length=1000000))
                case = fmt_case
            except Exception:
                pass
            cleaned.append(case.strip())
        return cleaned

    pass
