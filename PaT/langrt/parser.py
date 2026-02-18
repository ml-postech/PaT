from typing import Any, Callable

from ..utils.types import guard_never
from .types import (
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


class LrtParser:
    def __init__(self):
        pass

    ###########################################################################
    #   parsing

    def parse_code(self, code: CodeBlock) -> list[LrtNode]:
        """Convert a code block into a list of parsed nodes."""
        raise NotImplementedError()

    def _parse_code_splits(self, code: CodeBlock) -> list[CodeBlock]:
        """Split a code block into individually parsable sections."""
        raise NotImplementedError()

    def parse_node(self, code: CodeBlock, node: Any) -> LrtNode | None:
        """Parse a single node from a code block. This handles the loop logic."""
        raise NotImplementedError()

    def parse_import(self, code: CodeBlock, node: Any) -> LrtImport:
        """Parse `import` statement."""
        raise NotImplementedError()

    def parse_enum_def(self, code: CodeBlock, node: Any) -> LrtEnumDef:
        """Parse `enum` definition."""
        raise NotImplementedError()

    def _parse_enum_def_options(self, nodes: Any) -> dict[SymbolName, int | str]:
        raise NotImplementedError()

    def _parse_enum_def_option(self, node: Any) -> tuple[SymbolName, TypeName | None, Value | None] | None:
        raise NotImplementedError()

    def parse_struct_def(self, code: CodeBlock, node: Any) -> LrtStructDef:
        """Parse `struct` definition."""
        raise NotImplementedError()

    def _parse_struct_def_fields(self, nodes: Any) -> dict[SymbolName, tuple[TypeName, Value | None]]:
        raise NotImplementedError()

    def _parse_struct_def_field(self, node: Any) -> tuple[SymbolName, TypeName | None, Value | None] | None:
        raise NotImplementedError()

    def parse_function_def(self, code: CodeBlock, node: Any) -> LrtFunctionDef:
        """Parse `function` definition."""
        raise NotImplementedError()

    def _parse_function_def_args(self, node: Any) -> list[tuple[SymbolName, TypeName | None, Value | None]]:
        raise NotImplementedError()

    def _parse_function_def_ret(self, node: Any) -> TypeName | None:
        raise NotImplementedError()

    def _parse_function_def_if_implemented(self, nodes: Any) -> bool:
        raise NotImplementedError()

    def _parse_function_def_body(self, code: CodeBlock) -> CodeBlock:
        raise NotImplementedError()

    def parse_constant_def(self, code: CodeBlock, node: Any) -> LrtConstantDef:
        """Parse `constant` value definition."""
        raise NotImplementedError()

    ###########################################################################
    #   tools

    def make_import_from(
        self, module: tuple[SymbolName, ...], symbols: list[tuple[SymbolName, SymbolName]], level: int
    ) -> LrtImport:
        """Construct an import statement."""
        raise NotImplementedError()

    def make_stub_function_def_from_params(
        self,
        name: SymbolName,
        docstring: str | None,
        args: list[tuple[SymbolName, TypeName | None, Value | None]],
        ret: TypeName | None,
    ) -> LrtFunctionDef:
        """Construct a function definition from raw parameters."""
        raise NotImplementedError()

    def make_stub_function_def_from_func(self, func: LrtFunctionDef) -> LrtFunctionDef:
        """Copy function definition, but mark as unimplemented."""
        raise NotImplementedError()

    def fmt_nodes(self, nodes: list[LrtNode], organize_imports: bool) -> str:
        """Dump all nodes into an equivalent raw code block. No code should be
        assumed to be have been existent prior to this function call."""
        raise NotImplementedError()

    def fmt_code(self, code: CodeBlock) -> CodeBlock:
        """Prettify raw code block."""
        raise NotImplementedError()

    def fmt_imports(self, imports: list[LrtImport]) -> str:
        """Sort imports and export to code."""
        raise NotImplementedError()

    def fmt_function_sig(self, func: LrtFunctionDef) -> str:
        """Format function signature: -> `def name(arg: type, ...) -> type:`"""
        raise NotImplementedError()

    def fmt_function_def(self, func: LrtFunctionDef) -> LrtFunctionDef:
        """Prettify function."""
        raise NotImplementedError()

    def is_function_code_compliant(self, impl: LrtFunctionDef, sig: LrtFunctionDef, strict_mode: bool) -> bool:
        """Check if the implementation is compliant with the signature. We allow
        the implementation to further restrict the signature, but not widen it.
        This feature involves language specific checks."""
        raise NotImplementedError()

    def iter_repl_statements(self, flush_when: Callable[[str], bool], inp: str) -> list[str]:
        """Splits a 'multi-line' REPL statements into separable chunks. Ever
        when 'flush_when' returns 'True', all previously un-flushed lines are
        combined as a chunk, merged into one string."""
        raise NotImplementedError()

    ###########################################################################
    #   unified functions

    def deduplicate_nodes(self, nodes: list[LrtNode]) -> list[LrtNode]:
        """De-duplicate nodes. Implemented nodes always precedes unimplemented,
        imports are sorted by module and symbol name, and that latter values
        are always respected (like Python does) in case of conflicts."""

        # (level, (...module)) -> [...(symbols, alias)]
        m_imports: dict[tuple[int, tuple], list[tuple[SymbolName, SymbolName] | None]] = {}
        # (name) -> (implemented, func | enum | struct)
        m_impl: dict[SymbolName, tuple[bool, LrtNode]] = {}

        for node in nodes:
            if node.kind == "import":
                key = (node.level, node.module)
                if key not in m_imports:
                    m_imports[key] = []
                if not node.symbols:
                    m_imports[key].append(None)
                else:
                    m_imports[key].extend(node.symbols)
            elif node.kind == "enum":
                m_impl[node.name] = (True, node)
            elif node.kind == "struct":
                m_impl[node.name] = (True, node)
            elif node.kind == "function":
                key = node.name
                if not node.implemented:
                    if key not in m_impl or not m_impl[key][0]:
                        m_impl[key] = (False, node)
                else:
                    m_impl[key] = (True, node)
            elif node.kind == "constant":
                m_impl[node.name] = (True, node)
            else:
                guard_never(node)

        ret_imports: list[tuple[tuple[SymbolName, ...], LrtImport]] = []
        for (level, mod), symbols in m_imports.items():
            if level == 0 and not mod:
                has_none = any(sym is None for sym in symbols)
                has_others = [sym for sym in symbols if sym is not None]
                if has_none:
                    imp = self.make_import_from((), [], 0)
                    ret_imports.append(((), imp))
                if has_others:
                    imp = self.make_import_from((), sorted(set(has_others)), 0)
                    ret_imports.append(((), imp))
            else:
                has_none = any(sym is None for sym in symbols)
                has_others = [sym for sym in symbols if sym is not None]
                if has_none:
                    imp = self.make_import_from(mod, [], level)
                    ret_imports.append((mod, imp))
                if has_others:
                    imp = self.make_import_from(mod, sorted(set(has_others)), level)
                    ret_imports.append((mod, imp))
        ret_imports.sort(key=lambda x: x[0])

        ret: list[LrtNode] = []
        ret.extend([i for _, i in ret_imports])
        m_impl_picked = [n for _, n in m_impl.values()]
        for node in nodes:
            if any(node is i for i in m_impl_picked) and all(node is not i for i in ret):
                ret.append(node)
        return ret

    def _warn(self, msg: str) -> None:
        # TODO: use stack frame context
        print(msg)

    pass
