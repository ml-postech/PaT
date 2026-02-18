import re
from typing import Callable

from ....langrt import LrtFunctionDef, LrtProgram
from ....llm import ChatMessage
from ....utils.strings import extract_md_code
from ..gen_once import GenOncePrompt
from ..make_test import MakeTestPrompt, TestType, wrap_test_case_stdio


class PaTXCodeEvalDividePrompt(GenOncePrompt):
    """Segregate requirement into multiple sub-tasks."""

    def get_few_shot_ids(self):
        # special process to 2 shot, use `solution()` and `is_common()`
        return [[0, 1, 2, 3], [0, 1, 2], [0, 3], [0]]

    @staticmethod
    def _format_user_request(lang: str, prev_code: str, cur_func_code: str) -> str:
        prev_code = prev_code.strip("\n")
        cur_func_code = cur_func_code.strip("\n")
        return f"""Current Code:
```{lang}
{prev_code}
```

Let's think step by step and complete the following Python function that solves:
```{lang}
{cur_func_code}
```"""

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a programming copilot, you can solve a CodeForces or ICPC problem by writing Python functions. Your task is to:
- The previous attempt to direct implement the target function is failed, indicating its overall logic might be too complex to implement directly.
- For every turn, you need to write a Python function that returns the answer based on Current Code (not code in chat history).
- Do not modify function name, arg names, docstring in given functions.
- You can import library to better solve the problem.
- The target function is too complex to implement directly. Therefore, you MUST decompose it into multiple smaller, manageable helper functions. This decomposition is a mandatory first step toward solving the problem.
- You can leave new function unimplemented for now, but write the function at the end of the code and comment what the function does.
- Do not write any code outside the function (import is accepted).
""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code="",
                cur_func_code='''
def main() -> None:
    """
    # Description:
    You are given two integers $$$a$$$ and $$$b$$$. Find the sum of all common prime factors of $$$a$$$ and $$$b$$$. A prime factor is a prime number that divides a number without leaving a remainder. And common prime factors are prime factors that are common in both numbers.

    # Input:
    The first line contains two integers $$$a$$$ and $$$b$$$.

    # Output:
    Print a single integer — the sum of all common prime factors of $$$a$$$ and $$$b$$$.

    # Sample Input 1:
    ```
    12 18
    ```

    # Sample Output 1:
    ```
    5
    ```

    # Sample Input 2:
    ```
    231 3135
    ```

    # Sample Output 2:
    ```
    14
    ```
    """
    raise NotImplementedError()''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''
First, I need to get the prime factors of $a$ and $b$.
Second, I can use `for` loop to find common element in two factors list.
Finally, sum the common factor list and return the answer.
Here is the `sum_common_factors` function:

```python
def main() -> None:
    a, b = list(map(int, input().split()))
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    print(sum(common_factors))

def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""
    raise NotImplementedError()

def get_common(a: list, b: list) -> list:
    """get common element in two list $a$ and $b$"""
    raise NotImplementedError()
```''',
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code='''
def main() -> None:
    """Output the sum of all common prime factors of $a$ and $b$"""
    a, b = list(map(int, input().split()))
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    print(sum(common_factors))
''',
                cur_func_code='''
def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""
    raise NotImplementedError()
''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''
First, factor number is a positive integer that divides the number exactly, leaving no remainder.
Second, I need to check whether a factor is prime number
Here is the `prime_factor` function:

```python
def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""
    ret = []
    i = 1
    while i * i <= x:
        i += 1
        if x % i == 0 and is_prime(i):
            ret.append(i)
    return ret

def is_prime(x: int) -> bool:
    """determine $x$ is a prime number or not"""
    raise NotImplementedError()
```''',
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code='''
def main() -> None:
    """Output the sum of all common prime factors of $a$ and $b$"""
    a, b = list(map(int, input().split()))
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    print(sum(common_factors))

def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""
    ret = []
    i = 1
    while i * i <= x:
        i += 1
        if x % i == 0 and is_prime(i):
            ret.append(i)
    return ret
''',
                cur_func_code='''
def is_prime(x: int) -> bool:
    """determine $x$ is a prime number or not"""
    raise NotImplementedError()
''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''
Here is the `is_prime` function:

```python
def is_prime(x: int) -> bool:
    """determine $x$ is a prime number or not"""
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True
```''',
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code='''
def main():
    """Output the sum of all common prime factors of $a$ and $b$"""
    a, b = list(map(int, input().split()))
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    print(sum(common_factors))
''',
                cur_func_code='''
def get_common(a: list, b: list) -> list:
    """get common element in two list $a$ and $b$"""
    raise NotImplementedError()
''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''
Here is the `get_common` function:

```python
def get_common(a: list, b: list) -> list:
    """get common element in two list $a$ and $b$"""
    ret = []
    for item in a:
        if item in b:
            ret.append(item)
    return ret
```''',
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        program = ctx.lrt.parse(module=(), code=code)
        func = program.find(LrtFunctionDef, func_name)
        if not func:
            raise ValueError(f"cannot find function: {func_name}")
        prev_code = program.excluding(func)
        prev_code_s = ctx.lrt.pretty_fmt(prev_code)
        cur_code_s = ctx.lrt.pretty_fmt(func)
        return {
            "role": "user",
            "content": self._format_user_request(lang, prev_code_s, cur_code_s),
        }

    pass


class PaTXCodeEvalConquerPrompt(GenOncePrompt):
    """Re-generate original problem based on already completed sub-tasks."""

    @staticmethod
    def _format_user_request(lang: str, prev_code: str, cur_func_name: str, cur_func_code: str) -> str:
        prev_code = prev_code.strip("\n")
        cur_func_code = cur_func_code.strip("\n")
        ret = ""
        if prev_code.strip():
            ret += f"""Current Code:
```python
{prev_code}
```
"""
        ret += f"""\nLet's think step by step and implemented the following method `{cur_func_name}` using existing functions:
```
{cur_func_code}
```"""
        return ret

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a programming copilot, you can solve a  CodeForces or ICPC problem by writing Python functions. Your task is to:

  - For every turn, you need to write a Python function that returns the answer, based on current code (not code in chat history) and problem description.
  - The `main()` function read input from STDIN and print the answer to STDOUT, please DONNOT print redundant characters.
  - Do not modify function name, arg names, docstring in given functions.
  - Consider reusing existing functions that are already implemented.
  - You can import libraries to better solve the problem.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code='''
def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""
    ret = []
    i = 1
    while i * i <= x:
        i += 1
        if x % i == 0 and is_prime(i):
            ret.append(i)
    return ret

def is_prime(x: int) -> bool:
    """determine $x$ is a prime number or not"""
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True

def get_common(a: list, b: list) -> list:
    """get common element in two list $a$ and $b$"""
    ret = []
    for item in a:
        if item in b:
            ret.append(item)
    return ret
    """
''',
                cur_func_name="main",
                cur_func_code='''
def main() -> None:
    """
    # Description:
    You are given two integers $$$a$$$ and $$$b$$$. Find the sum of all common prime factors of $$$a$$$ and $$$b$$$. A prime factor is a prime number that divides a number without leaving a remainder. And common prime factors are prime factors that are common in both numbers.

    # Input:
    The first line contains two integers $$$a$$$ and $$$b$$$.

    # Output:
    Print a single integer — the sum of all common prime factors of $$$a$$$ and $$$b$$$.

    # Sample Input 1:
    ```
    12 18
    ```

    # Sample Output 1:
    ```
    5
    ```

    # Sample Input 2:
    ```
    231 3135
    ```

    # Sample Output 2:
    ```
    14
    ```
    raise NotImplementedError()
''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """
First, I need to get the prime factors of $a$ and $b$.
Second, I can use `for` loop to find common element in two factors list.
Finally, sum the common factor list and return the answer.
Here is the `sum_common_factors` function:

```python
def main() -> None:
    a, b = list(map(int, input().split()))
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    print(sum(common_factors))
```
""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code="",
                cur_func_name="reverse",
                cur_func_code="""
def reverse(s: str) -> str:
    \"\"\"Return reversed input string\"\"\"
    raise NotImplementedError()
""",
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """
Here is the code for `reverse` function:
```python
def reverse(s: str) -> str:
    return s[::-1]
```
""",
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        func = ctx.lrt.parse(module=(), code=code).find(LrtFunctionDef, func_name)
        func_docstring = (func.docstring if func else None) or ""
        return {
            "role": "user",
            "content": f'''Below chats are examples, these code are not in your workspace, now you need to solve a new problem:
Current Code:
```python
{code}
```

Let's think step by step and complete the following Python function `{func_name}` that solves:
"{func_docstring}"''',
        }

    def wrap_prompt_iter(self, ctx, ancestors, func, descendants, keep_ancestors):
        # overriding impl here is necessary because:
        #   - function body must be hidden (becoming a stub)
        program_pre = LrtProgram(module=(), nodes=[*(ancestors if keep_ancestors else []), *descendants])
        code_pre = ctx.lrt.pretty_fmt(program_pre)
        func_stub = ctx.lrt._parse.make_stub_function_def_from_func(func)
        code_cur = ctx.lrt.pretty_fmt(func_stub)

        p_system, *p_few_shots = self.get_few_shots
        p_next_turn: ChatMessage = {
            "role": "user",
            "content": self._format_user_request(ctx.lrt.lang, code_pre, func.name, code_cur),
        }
        for nshots in self.get_few_shot_ids():
            few_shots = sum(([p_few_shots[2 * i], p_few_shots[2 * i + 1]] for i in nshots), [])
            yield [p_system, *few_shots, p_next_turn]
        return

    pass


class PaTXCodeEvalFuncCallPrompt(MakeTestPrompt):
    """Provide just stdin inputs for a given function. The sample input is to
    be provided verbatim by the LLM."""

    get_skip_closures = False

    @staticmethod
    def _format_user_request(func_name: str, lang: str, code: str) -> str:
        code = code.strip("\n")
        return f"""Let's think step by step and create some inputs for the following function `{func_name}(...)` in Python.

```{lang}
{code}
```

Store your test cases for `{func_name}(...)` as input blocks, separated by Markdown section titles '## Input i'. They will be called later."""

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a proficient software engineer and architect, specialized in testing, capable of observing and finding security holes and logical issues in code that spans across multiple computer science fields and mathematics. You are given a piece of Python function, and are expected to produce some test inputs and outputs for that function. Specifically:

  - You should provide exactly what the function would read from standard input, followed by a '## Input i' section header.
  - Additionally, an expected output should be provided, representing the correct standard output, followed by a '## Output i' section header.

Here follows a series of mutually uncorrelated functions to examine, one per conversation.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                func_name="main",
                lang="python",
                code='''
def main() -> None:
    """
    # Description:
    You are given two integers $$$a$$$ and $$$b$$$. Find the sum of all common prime factors of $$$a$$$ and $$$b$$$. A prime factor is a prime number that divides a number without leaving a remainder. And common prime factors are prime factors that are common in both numbers.

    # Input:
    The first line contains two integers $$$a$$$ and $$$b$$$.

    # Output:
    Print a single integer — the sum of all common prime factors of $$$a$$$ and $$$b$$$.

    # Sample Input 1:
    ```
    12 18
    ```

    # Sample Output 1:
    ```
    5
    ```

    # Sample Input 2:
    ```
    231 3135
    ```

    # Sample Output 2:
    ```
    14
    ```
    """
    raise NotImplementedError()''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, I can create some test inputs and outputs for the `main` function. The requirements stated that the function should read two integers from the standard input, and return the sum of all common prime factors of the two integers. Here are some cases of input / output pairs:

## Input 1

```
125 200
```

## Input 2

```
777 333
```

## Input 3

```
96 64
```

## Input 4

```
3 7
```

## Input 5

```
324 678
```
""",
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        return {
            "role": "user",
            "content": self._format_user_request(func_name, lang, code),
        }

    def parse_tests(self, ctx, func, raw_message):
        raw_md_sections = _split_md_by_header(raw_message, lambda line: bool(re.findall(r"^## Input \d+$", line)))

        raw_md_blocks: list[str] = []
        for _header, body in raw_md_sections:
            blocks = [c for tag, c in extract_md_code(body) if tag in {""}]
            raw_md_blocks.extend(blocks)

        results: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
        for raw_md_block in raw_md_blocks:
            test_func = wrap_test_case_stdio(ctx, func.name, input=raw_md_block, output=None)
            program = LrtProgram(module=(), nodes=[test_func])
            results.append((TestType.call, program, test_func))

        return results

    pass


class PaTXCodeEvalUnitTestPrompt(MakeTestPrompt):
    """Assert that the function has certain behavior. Both inputs and outputs
    are provided by the LLM, e.g. `assert gcd(10, 25) == 5`."""

    get_skip_closures = False

    @staticmethod
    def _format_user_request(func_name: str, lang: str, code: str) -> str:
        code = code.strip("\n")
        return f"""Let's think step by step and create some unit test (inputs) for the following function `{func_name}(...)` in Python.

```{lang}
{code}
```

Store your test cases for `{func_name}(...)` as input / output blocks, separated by Markdown section titles '## Input i' and '## Output i'. They will be called later."""

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a proficient software engineer and architect, specialized in testing, capable of observing and finding security holes and logical issues in code that spans across multiple computer science fields and mathematics. You are given a piece of Python function, and are expected to produce some test inputs and outputs for that function. Specifically:

  - You should provide exactly what the function would read from standard input, followed by a '## Input i' section header.
  - Additionally, an expected output should be provided, representing the correct standard output, followed by a '## Output i' section header.

Here follows a series of mutually uncorrelated functions to examine, one per conversation.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                func_name="main",
                lang="python",
                code='''
def main() -> None:
    """
    # Description:
    You are given two integers $$$a$$$ and $$$b$$$. Find the sum of all common prime factors of $$$a$$$ and $$$b$$$. A prime factor is a prime number that divides a number without leaving a remainder. And common prime factors are prime factors that are common in both numbers.

    # Input:
    The first line contains two integers $$$a$$$ and $$$b$$$.

    # Output:
    Print a single integer — the sum of all common prime factors of $$$a$$$ and $$$b$$$.

    # Sample Input 1:
    ```
    12 18
    ```

    # Sample Output 1:
    ```
    5
    ```

    # Sample Input 2:
    ```
    231 3135
    ```

    # Sample Output 2:
    ```
    14
    ```
    """
    raise NotImplementedError()''',
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, I can create some test inputs and outputs for the `main` function. The requirements stated that the function should read two integers from the standard input, and return the sum of all common prime factors of the two integers. Here are some cases of input / output pairs:

## Input 1

```
125 200
```

## Output 1

```
5
```

## Input 2

```
231 3135
```

## Output 2

```
14
```
""",
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        return {
            "role": "user",
            "content": f"""Extract tests for the following function `{func_name}(...)` in Python.

```{lang}
{code}
```

Store your test cases for `{func_name}(...)` as assertions, one per line. They will be called later.""",
        }

    def parse_tests(self, ctx, func, raw_message):
        raw_md_sections = _split_md_by_header(
            raw_message, lambda line: bool(re.findall(r"^#+ (Input|Output) \d+$", line))
        )

        raw_md_cases: list[tuple[str, str]] = []  # ("input" | "output", case)
        for header, body in raw_md_sections:
            case_type_raw = re.findall(r"^#+ (Input|Output) \d+$", header)[0]
            case_type = {"Input": "input", "Output": "output"}[case_type_raw]
            blocks = [c for tag, c in extract_md_code(body) if tag in {""}]
            if blocks:
                # there should be one and only one block
                raw_md_cases.append((case_type, blocks[0]))

        md_cases: list[tuple[str, str]] = []  # must be alternating (input_case, output_case)
        for i in range(0, len(raw_md_cases) - 1):
            if raw_md_cases[i][0] == "input" and raw_md_cases[i + 1][0] == "output":
                md_cases.append((raw_md_cases[i][1], raw_md_cases[i + 1][1]))

        results: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
        for case_inp, case_out in md_cases:
            test_func = wrap_test_case_stdio(ctx, func.name, input=case_inp, output=case_out)
            program = LrtProgram(module=(), nodes=[test_func])
            results.append((TestType.assertion, program, test_func))

        return results

    pass


def _split_md_by_header(text: str, is_header: Callable[[str], bool]) -> list[tuple[str, str]]:
    """Convert Markdown document to a list of sections, each identified by a
    header line and its body (excluding the header line)."""

    title: str | None = None
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []  # (header, body)
    for line in text.split("\n"):
        if is_header(line):
            if title:
                sections.append((title, "\n".join(buffer)))
            title = line
            buffer = []
        else:
            buffer.append(line)
    if title:
        sections.append((title, "\n".join(buffer)))
    return sections
