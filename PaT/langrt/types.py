from typing import Literal, Type, TypeAlias, TypeVar, overload

import pydantic

###############################################################################
#   parser


SymbolName: TypeAlias = str
"""'foo'"""

TypeName: TypeAlias = str  # SymbolName or else
"""'MyType', 'int', 'list[tuple[CompositeType, str]', ..."""

Value: TypeAlias = str
"""'1', '2.0', 'True', 'False', '[]', ..."""

CodeBlock: TypeAlias = str


class LrtImport(pydantic.BaseModel):
    """
    { module=['pydantic'], symbols=[], level=0 } ->
        [py]    import pydantic
        [ts]    TODO: import * as pydantic from "pydantic";
    { module=[], symbols=[('numpy', 'np')], level=0 } ->
        [py]    import numpy as np
        [ts]    TODO: import * as np from "numpy";
    { module=[], symbols=[('itertools', 'itertools')], level=1 } ->
        [py]    from . import itertools
        [ts]    TODO: import { itertools } from ".";
    { module=['pathlib'], symbols=[('Path', 'Path')], level=0 } ->
        [py]    from pathlib import Path
        [ts]    TODO: import { Path } from "pathlib";
    { module=['runner', 'types'], symbols=[('Enum', 'En'), ('Struct', 'Struct')], level=2 } ->
        [py]    from ..runner.types import Enum as En, Struct
        [ts]    TODO: import { Enum as En, Struct } from "../runner/types";
    """

    kind: Literal["import"]
    code: CodeBlock

    module: tuple[SymbolName, ...]
    symbols: list[tuple[SymbolName, SymbolName]]
    level: int
    pass


class LrtStructDef(pydantic.BaseModel):
    """
    Object interface definition. Possible formats are:

    [py]    class MyStruct(pydantic.BaseModel):
            TODO: dataclasses, Protocol, TypedDict are not supported yet
    [ts]    TODO: interface MyStruct { ... }
            TODO: type MyStruct = { ... };"""

    kind: Literal["struct"]
    code: CodeBlock

    name: SymbolName
    docstring: str | None
    fields: dict[SymbolName, tuple[TypeName, Value | None]]
    pass


class LrtEnumDef(pydantic.BaseModel):
    """Enumeration object. Possible formats are:

    [py]    class MyEnum(enum.Enum):
    [ts]    TODO: enum MyEnum { ... }"""

    kind: Literal["enum"]
    code: CodeBlock

    name: SymbolName
    docstring: str | None
    options: dict[SymbolName, int | str]
    pass


class LrtFunctionDef(pydantic.BaseModel):
    """Function definition. Possible formats are:

    [py]    def my_func(arg1: int, arg2: str = "default") -> float:
    [ts]    TODO: function my_func(arg1: number, arg2: string = "default"): number { ... }"""

    kind: Literal["function"]
    code: CodeBlock

    name: SymbolName
    docstring: str | None
    args: list[tuple[SymbolName, TypeName | None, Value | None]]  # (name, type?, default?)
    ret: TypeName | None
    implemented: bool
    body: CodeBlock
    pass


class LrtConstantDef(pydantic.BaseModel):
    """Global variable definition. Possible formats are:

    [py]    MY_CONSTANT: int = 42
    [ts]    TODO: const MY_CONSTANT: number = 42;"""

    kind: Literal["constant"]
    code: CodeBlock

    name: SymbolName
    type: TypeName | None
    value: Value
    pass


LrtNode = LrtImport | LrtStructDef | LrtEnumDef | LrtFunctionDef | LrtConstantDef
"""A top-level node in the program structure."""


TLrtNode = TypeVar("TLrtNode", bound=LrtNode)
"""Generic type argument for LrtNode."""


###############################################################################
#   program execution


ModuleName: TypeAlias = str
"""'mod.ule.implementation'"""

FileContent: TypeAlias = str
"""'def foo(...) -> ...'"""

ExceptionLog: TypeAlias = str
"""Traceback (most recent call last): ..."""

LineNum: TypeAlias = int

JsonSerializable: TypeAlias = str | int | float | bool | None | list | dict


class LrtExecutionEnv(pydantic.BaseModel):
    code: dict[ModuleName, FileContent]
    # internal=True/dep=False, mod.ule.name, module_alias?, [...(symbols, aliases)]?
    imports: list[tuple[bool, ModuleName, SymbolName | None, list[tuple[SymbolName, SymbolName]] | None]]
    func_name: SymbolName
    func_args: list[TypeName]
    func_kwargs: dict[SymbolName, TypeName]
    func_ret: TypeName
    pass


class LrtExecutionResult(pydantic.BaseModel):
    ok: bool
    ret_code: int
    error: ExceptionLog
    result: JsonSerializable | None
    stdout: str | None
    duration: float
    pass


###############################################################################
#   generic wrapper


class LrtProgram(pydantic.BaseModel):
    module: tuple[SymbolName, ...]

    nodes: list[LrtNode]

    def cast_as(self, typ: Type[TLrtNode]) -> TLrtNode:
        assert len(self.nodes) == 1
        head = self.nodes[0]
        assert isinstance(head, typ)
        return head

    @overload
    def find(self, typ: Type[TLrtNode], name: SymbolName, /) -> TLrtNode | None: ...
    @overload
    def find(self, node: TLrtNode, /) -> TLrtNode | None: ...
    def find(self, *args) -> LrtNode | None:
        if len(args) == 1:
            (node,) = args
            if name := getattr(args[0], "name", None):
                return self.find(type(args[0]), name)
            for node in self.nodes:
                if node == args[0]:
                    return node
            return None
        elif len(args) == 2:
            typ, name = args
            return ret[0] if (ret := self.find_all(typ, name)) else None

    def find_all(self, typ: Type[TLrtNode], name: SymbolName | None) -> list[TLrtNode]:
        ret: list[TLrtNode] = []
        for node in self.nodes:
            if isinstance(node, typ):
                if name is None or getattr(node, "name", None) == name:
                    ret.append(node)
        return ret

    def excluding(self, *args: LrtNode | list[LrtNode] | None) -> list[LrtNode]:
        excluding: list[LrtNode] = []
        for arg in args:
            if isinstance(arg, list):
                excluding.extend(arg)
            elif arg is None:
                continue
            else:
                excluding.append(arg)
        return [node for node in self.nodes if not any(node == excl for excl in excluding)]

    pass


class LrtSolution(pydantic.BaseModel):
    """A solution is a collection of source files."""

    modules: list[LrtProgram]

    def find(self, module: LrtProgram | tuple[SymbolName, ...]) -> LrtProgram | None:
        if isinstance(module, LrtProgram):
            module = module.module
        for prog in self.modules:
            if prog.module == module:
                return prog
        return None

    pass
