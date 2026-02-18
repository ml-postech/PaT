from ....langrt import LrtFunctionDef, LrtProgram
from ....llm import ChatMessage
from ....utils.strings import extract_md_code
from ..make_test import MakeTestPrompt, TestType, _make_random_sig


class PaTSysTestArgsPrompt(MakeTestPrompt):
    """Extract argument calls or system unit tests from the input function's
    definition or requirements."""

    get_skip_closures = True

    @staticmethod
    def _format_user_request(lang: str, func_name: str, code: str) -> str:
        code = code.strip("\n")
        return f"""Extract tests from the following function `{func_name}(...)` definition.

```{lang}
{code}
```

Store extracted cases for `{func_name}(...)` as function calls or assertions, one per line."""

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a proficient software engineer and architect, specialized in testing. You will perform a series of very accurate tasks in extracting test cases from the user's requirements, and translate them into proper python expressions. Specifically:

  - You should invoke the function, or assert its behavior in a one-liner fashion.
  - Do not bring in imports other than what's already imported. Use the pre-declared imports in the original function only.
  - The callee may have multiple arguments, treat them with care and proper attribution.
  - You **must** respect the function signature and docstring, as well as what was written in the requirements.
  - Keep the invocations verbatim as-is. If there are nothing to extract, produce nothing.

Here follows a series of mutually uncorrelated cases to extract, one per conversation.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                func_name="check_valid_brackets",
                code='''
def check_valid_brackets(seq: str) -> bool:
    """Determine if a bracket sequence consisting of '(', ')', '{', '}', '['
    and ']' is valid."""

    raise NotImplementedError()
''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, here are some test cases for the `check_valid_brackets` function:

```python
```

It appears that there are no test cases to extract from the description.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                func_name="gcd",
                code='''
def lcm(a: int, b: int) -> int:
    """Find the least common multiple of `a` and `b`. Examples:
        - lcm(10**12, 12**10)
        - lcm(5, 15)
        - lcm(3, 5) == 15"""

    raise NotImplementedError()

def gcd(a: int, b: int) -> int:
    """Find the greatest common divisor of `a` and `b`. e.g. `gcd(12, 16)`.
    Other examples:

    >>> gcd(10, 15)
      # 5
    >>> gcd(10**12, 11**12)
    >>> gcd(5, 15)
    >>> assert gcd(3, 5) == 1

    Also, remain that the following cases also apply:
      - gcd(1, 1) -> 1
      - gcd(7, 3) -> 1
    """

    raise NotImplementedError()
''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, here are some test cases for the `gcd` function:

```python
gcd(12, 16)
assert gcd(10, 15) == 5
gcd(10**12, 11**12)
gcd(5, 15)
assert gcd(3, 5) == 1
assert gcd(1, 1) == 1
assert gcd(7, 3) == 1
```

These are the test cases extracted from the description, which are related to `gcd`.""",
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        return {
            "role": "user",
            "content": self._format_user_request(lang, func_name, code),
        }

    def parse_tests(self, ctx, func, raw_message):
        raw_md_tags = {"py", "python", "python3"}
        raw_md_blocks = [c for tag, c in extract_md_code(raw_message) if tag in raw_md_tags]

        raw_invokes: list[str] = []
        for md_block in raw_md_blocks:
            for invoke in ctx.lrt._parse.iter_repl_statements(
                lambda line: (
                    line.strip().startswith(f"{func.name}(") or line.strip().startswith(f"assert {func.name}(")
                ),
                md_block,
            ):
                raw_invokes.append(invoke)

        results: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
        for invoke in raw_invokes:
            lines = invoke.split("\n")
            # previous lines may represent intermediate variables
            if lines[-1].startswith("assert "):
                # test_type = TestType.assertion
                test_type = TestType.unittest
                lines.append("return True")
            else:
                # test_type = TestType.call
                test_type = TestType.unittest
                lines[-1] = "return " + lines[-1]
            lines = [f"    {line}\n" for line in lines]
            code = "".join(lines)
            code = f"def _test_{func.name}_{_make_random_sig()}(_seed: int):\n{code}"
            # parse & extract
            program = ctx.lrt.parse(module=(), code=code)
            if program is None:
                return None
            program = ctx.lrt.prettify(program)
            fn = program.cast_as(LrtFunctionDef)
            results.append((test_type, program, fn))

        return results

    pass


# TODO: missing stdio prompt
