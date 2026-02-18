from ...langrt import LrtFunctionDef, LrtNode, LrtProgram
from ..PaT.gen_once import GenOncePrompt, PaT_gen_once, gen_collect_program
from ..shared import CodeGenContext, CodeGenJournal, CodeGenMethod


class VanillaGen(CodeGenMethod):
    def __init__(
        self,
        gen_prompt: GenOncePrompt,
        temperature: float,
        retries: int,
    ):
        self.gen_prompt = gen_prompt
        self.temperature = temperature
        self.retries = retries

    async def gen(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ) -> tuple[LrtProgram | None, CodeGenJournal]:
        ctx.log.in_scope(f"vanilla_gen[{func.name}]")

        _results, journal = await PaT_gen_once(
            ctx=ctx,
            opt_prompt=self.gen_prompt,
            opt_temperature=self.temperature,
            opt_samples=1,
            opt_min_samples=1,
            opt_retries=self.retries,
            ancestors=ancestors,
            func=func,
            descendants=descendants,
        )
        # vanilla has no test. we just generate. if we test then that's CodeT.
        if not _results:
            return None, journal
        func_impl, rest_impl = _results[0]

        program = gen_collect_program(ctx, ancestors, func_impl, rest_impl, descendants)
        ctx.log.code("python", "final result", ctx.lrt.fmt(program))
        return program, journal

    pass
