from typing import Any, Callable, Coroutine, Generator, Iterable, TypeAlias

from ...langrt import CodeBlock, LrtFunctionDef, LrtNode, LrtProgram, SymbolName
from ...llm import ChatMessage
from ...utils.strings import extract_md_code
from ..shared import CodeGenContext, CodeGenJournal, CodeGenJournalist
from .defaults import DEFAULT_IMPORTS


class GenOncePrompt:
    """A prompt generator for `gen_once`. For starters, you can override the
    `get_system`, `get_few_shots` and `get_next_turn` methods to customize, at
    a lower level, the prompt that will be generated.

    Although all our code follow the same paradigm, we know that inventing DSLs
    here are futile and redundant. This is why you can also override the
    `wrap_prompt` method to further customize the prompting process."""

    get_few_shots: list[ChatMessage] = []
    """OVERRIDE: Few shot demonstrations. The first turn is always a "system"
    message, and the rest comes in assistant and user conversation pairs."""

    def get_few_shot_ids(self) -> Iterable[list[int]]:
        """See `GenOncePrompt.get_few_shot_ids`. Specifies which # of few shots
        will be used (e.g. default [0,1,2,3], [0,1,2], [0,1], [0], [])."""
        ls = list(self.get_few_shots)
        for i in range((len(ls) - 1) // 2, -1, -1):
            yield list(range(i))
        return

    def get_next_turn(self, ctx: CodeGenContext, func_name: SymbolName, lang: str, code: CodeBlock) -> ChatMessage:
        """Format the next-turn user prompt:
        - `func_name`: `foo`
        - `lang`: `python`
        - `code`: `def foo(a: int, b: int) -> int:\n    raise ...\n\n\n...`"""
        raise NotImplementedError()

    def wrap_prompt_iter(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
        keep_ancestors: bool,
    ) -> Generator[list[ChatMessage], None, None]:
        """OVERRIDE?: The generic way to generate prompts. In each iteration,
        we apply token backoff as requested by `get_few_shot_ids`."""

        func_name, code = _gen_once_fast_fmt(ctx, ancestors, func, descendants, keep_ancestors)
        p_system, *p_few_shots = self.get_few_shots
        p_next_turn = self.get_next_turn(ctx, func_name, ctx.lrt.lang, code)
        for nshots in self.get_few_shot_ids():
            few_shots = sum(([p_few_shots[2 * i], p_few_shots[2 * i + 1]] for i in nshots), [])
            yield [p_system, *few_shots, p_next_turn]
        return

    pass


GenOnceSig: TypeAlias = Callable[
    [CodeGenContext, list[LrtNode], LrtFunctionDef, list[LrtNode]],
    Coroutine[Any, Any, tuple[tuple[LrtFunctionDef, list[LrtNode]] | None, CodeGenJournal]],
]
GenManySig: TypeAlias = Callable[
    [CodeGenContext, list[LrtNode], LrtFunctionDef, list[LrtNode], int],
    Coroutine[Any, Any, tuple[list[tuple[LrtFunctionDef, list[LrtNode]]], CodeGenJournal]],
]


async def PaT_gen_once(
    ctx: CodeGenContext,
    opt_prompt: GenOncePrompt,
    opt_temperature: float,
    opt_samples: int,
    opt_min_samples: int,
    opt_retries: int,
    ancestors: list[LrtNode],
    func: LrtFunctionDef,
    descendants: list[LrtNode],
    divide:bool,
) -> tuple[list[tuple[LrtFunctionDef, list[LrtNode]]], CodeGenJournal]:
    ctx.log.in_scope(f"PaT_gen_once[{func.name}(...)]")
    _sj = CodeGenJournalist(ctx, "PaT_gen_once", (ancestors, func, descendants))

    opt_validate_sig = True
    results: list[tuple[LrtFunctionDef, list[LrtNode], bool]] = []  # func, ...rest, has_err

    for _retry in range(1, opt_retries + 1):
        # apply nshot backoff during llm request
        responses = []
        for history in opt_prompt.wrap_prompt_iter(ctx, ancestors, func, descendants, keep_ancestors=True):
            if divide:
                next_result = await ctx.llm_planner.call(history, n=opt_samples, temperature=opt_temperature)
            else:
                next_result = await ctx.llm.call(history, n=opt_samples, temperature=opt_temperature)
            if next_result.ok:
                responses = next_result.ok
                break
            elif next_result.backoff_tokens:
                continue
            else:
                raise (err := (next_result.err or [])[-1]) from err
        if len(responses) < 1:
            ctx.log.string(f"attempt {_retry} failed: llm did not return result")
            continue

        for _idx, response in enumerate(responses):
            # parse code on a best-effort basis
            raw_py = [c for tag, c in extract_md_code(response) if tag in {"", "py", "python", "python3"}]
            if not raw_py:
                ctx.log.string(f"attempt {_retry} impl {_idx} failed: no code generated")
                continue
            parsing_nodes: list[LrtNode] = []
            for raw_chunk in raw_py:
                try:
                    parsing = ctx.lrt.parse(module=(), code=raw_chunk)
                    parsing_nodes.extend(parsing.nodes)
                except Exception:
                    ctx.log.trace()
                    continue
            if not parsing_nodes:
                ctx.log.string(f"attempt {_retry} impl {_idx} failed: cannot parse code")
                continue
            parsing = ctx.lrt.prettify(LrtProgram(module=(), nodes=parsing_nodes))

            # gpt has bad habits. sometimes it repeats the stub function before it
            # generates anything
            if not (root_impl := parsing.find(LrtFunctionDef, func.name)):
                ctx.log.string(f"attempt {_retry} impl {_idx} failed: missing {func.name}(...)")
                continue
            children = parsing.excluding(root_impl)

            # check implementation & type sig
            if not root_impl.implemented:
                ctx.log.string(f"attempt {_retry} impl {_idx} failed: expected implemented function")
                continue
            if opt_validate_sig and not ctx.lrt._parse.is_function_code_compliant(root_impl, func, strict_mode=True):
                err = f"attempt {_retry} impl {_idx} failed: signature mismatch\n"
                err += f"    got: {root_impl.args} -> {root_impl.ret}\n"
                err += f"    expected: {func.args} -> {func.ret}"
                ctx.log.string(err)
                ctx.log.warn("skipping result due to signature mismatch")
                results.append((root_impl, children, True))
                continue

            root_impl = _heal_generated_function(ctx, root_impl, func, heal_docstring=True)
            ctx.log.code(ctx.lrt.lang, f"generated code for impl {_idx}", ctx.lrt.fmt(root_impl, children))
            results.append((root_impl, children, False))

        if len(results) >= opt_min_samples:
            break
        pass

    if opt_samples == 1:
        # gen once mode
        if len(results) < 1:
            ctx.log.error(f"failed to generate code for:\n\n{func.code}")
            return [], _sj.collect_err("failed to generate code for function")
        _res = [(r, c) for r, c, _ in results]
        _res_func, _res_rest = _res[0]
        return _res, _sj.collect_gen((_res_func, _res_rest))
    else:
        # sampler mode
        if any(not err for _, _, err in results):
            results = [(r, c, err) for r, c, err in results if not err]
        if len(results) < opt_min_samples:
            ctx.log.warn(f"only {len(results)} samples generated, expected {opt_min_samples}")
        _res = [(r, c) for r, c, _ in results]
        return _res, _sj.collect_gen_multi(_res)
    pass


def _gen_once_fast_fmt(
    ctx: CodeGenContext,
    ancestors: list[LrtNode],
    func: LrtFunctionDef,
    descendants: list[LrtNode],
    keep_ancestors: bool,
) -> tuple[SymbolName, CodeBlock]:
    """(...) -> (func_name, code)"""

    nodes: list[LrtNode] = []
    if keep_ancestors:
        for node in ancestors:
            if node.kind == "function":
                nodes.append(node)
                # # causes performance issues since this loses context
                # nodes.append(ctx.lrt._parse.make_stub_function_def_from_func(node))
            else:
                nodes.append(node)
    nodes.append(func)
    nodes.extend(descendants)
    code_logic = ctx.lrt.prettify(LrtProgram(module=(), nodes=nodes))
    code = ctx.lrt.fmt(code_logic)
    return func.name, code


def _heal_generated_function(
    ctx: CodeGenContext,
    impl: LrtFunctionDef,
    sig: LrtFunctionDef,
    overwrite_name: bool = False,
    overwrite_arg_types: bool = False,
    overwrite_ret_type: bool = False,
    heal_docstring: bool = False,
    overwrite_docstring: bool = False,
) -> LrtFunctionDef:
    """Create a duplicate of the implementation with certain fields respecting
    the original signatures."""

    draft = impl.model_copy(deep=True)
    if overwrite_name:
        draft.name = sig.name
    if overwrite_arg_types:
        if len(draft.args) == len(sig.args):
            draft.args = [(i_sym, s_typ, s_val) for (_, s_typ, s_val), (i_sym, _, _) in zip(draft.args, sig.args)]
    if overwrite_ret_type:
        draft.ret = sig.ret
    if heal_docstring:
        if draft.docstring is None and sig.docstring is not None:
            draft.docstring = sig.docstring
    if overwrite_docstring:
        if sig.docstring is not None:
            draft.docstring = sig.docstring
    # prettify does not respect newly added fields
    return ctx.lrt._parse.fmt_function_def(draft)


def gen_collect_program(ctx: CodeGenContext, *args: LrtNode | list[LrtNode]) -> LrtProgram:
    nodes = ctx.lrt.parse(module=(), code=DEFAULT_IMPORTS).nodes
    for arg in args:
        if isinstance(arg, list):
            nodes.extend(arg)
        else:
            nodes.append(arg)
    return ctx.lrt.prettify(LrtProgram(module=(), nodes=nodes), organize_imports=True)
