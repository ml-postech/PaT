from ....langrt import LrtFunctionDef, LrtProgram
from ....llm import ChatMessage
from ....utils.strings import extract_md_code
from ..gen_once import GenOncePrompt
from ..make_test import MakeTestPrompt, TestType


class PaTHumanEvalDividePrompt(GenOncePrompt):
    """Segregate requirement into multiple sub-tasks."""

    def get_few_shot_ids(self):
        # special process to 2 shot, use `solution()` and `is_common()`
        return [[0, 1, 2, 3], [0, 1, 2], [0, 3], [0]]

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a programming copilot, you can solve a LeetCode problem by writing Python functions. Your task is to:
  - The previous attempt to direct implement the target function is failed, indicating its overall logic might be too complex to implement directly.
  - For every turn, you need to write a Python function that returns the answer based on Current Code (not code in chat history).
  - Do not modify function name, arg names, docstring in given functions.
  - You can import libraries to better solve the problem.
  - The target function is too complex to implement directly. Therefore, you MUST decompose it into multiple smaller, manageable helper functions. This decomposition is a mandatory first step toward solving the problem.
  - You can leave new function unimplemented for now, but write the function at the end of the code and comment what the function does.""",
},
        {
            "role": "system",
            "name": "user",
            "content": f'''Current Code:
```python
def sum_common_factors(a: int, b: int) -> int:
    """Compute the sum of all common prime factors of $a$ and $b$"""
    raise NotImplementedError()
```

Let's think step by step and complete the following Python function `sum_common_factors` that solves:
"Compute the sum of all common prime factors of $a$ and $b$"''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''First, I need to get the prime factors of $a$ and $b$.
Second, I can use `for` loop to find common element in two factors list.
Finally, sum the common factor list and return the answer.
Here is the `sum_common_factors` function:

```python
def sum_common_factors(a: int, b: int) -> int:
    """Compute the sum of all common prime factors of $a$ and $b$"""
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)

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
            "content": f'''Current Code:
```python
def sum_common_factors(a: int, b: int) -> int:
    """Compute the sum of all common prime factors of $a$ and $b$"""
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)

def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""

    raise NotImplementedError()
```

Let's think step by step and complete the following Python function `prime_factor` that solves:
"get a list of prime factors of number $x$"''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''First, factor number is a positive integer that divides the number exactly, leaving no remainder.
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
            "content": f'''Current Code:
```python
def sum_common_factors(a: int, b: int) -> int:
    """Compute the sum of all common prime factors of $a$ and $b$"""
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)

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
```

Let's think step by step and complete the following Python function `is_prime` that solves:
"determine $x$ is a prime number or not"''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''Here is the `is_prime` function:

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
            "content": f'''Current Code:
```python
def sum_common_factors(a: int, b: int) -> int:
    """Compute the sum of all common prime factors of $a$ and $b$"""
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)

def get_common(a: list, b: list) -> list:
    """get common element in two list $a$ and $b$"""
    raise NotImplementedError()
```

Let's think step by step and complete the following Python function `get_common` that solves:
"get common element in two list $a$ and $b$"''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''Here is the `get_common` function:

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
        func = ctx.lrt.parse(module=(), code=code).find(LrtFunctionDef, func_name)
        func_docstring = (func.docstring if func else None) or ""
        return {
            "role": "user",
            "content": f'''Below chats are examples, these code are not in your workspace, now you need to solve a new problem:
Current Code:
```{lang}
{code}
```

Let's think step by step and complete the following Python function `{func_name}` that solves:
"{func_docstring}"''',
        }

    pass


class PaTHumanEvalConquerPrompt(GenOncePrompt):
    """Re-generate original problem based on already completed sub-tasks."""

    @staticmethod
    def _format_user_request(lang: str, prev_code: str, cur_func_name: str, cur_func_doc: str) -> str:
        prev_code = prev_code.strip("\n")
        return f'''Current Code:

```{lang}
{prev_code}
```

Let's think step by step and implemented the following method `{cur_func_name}` using existing functions that solves:
"{cur_func_doc}"'''

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a programming copilot, you can solve a LeetCode problem by writing Python functions. Your task is to:

  - For every turn, you need to write a Python function that returns the answer, based on current code (not code in chat history) and problem description.
  - Do not modify function name, arg names, docstring in given functions.
  - Consider reusing existing functions that are already implemented.
  - You can import libraries to better solve the problem.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code="""
def prime_factor(x: int) -> list:
    \"\"\"get a list of prime factors of number $x$\"\"\"
    ret = []
    i = 1
    while i * i <= x:
        i += 1
        if x % i == 0 and is_prime(i):
            ret.append(i)
    return ret

def is_prime(x: int) -> bool:
    \"\"\"determine $x$ is a prime number or not\"\"\"
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True

def get_common(a: list, b: list) -> list:
    \"\"\"get common element in two list $a$ and $b$\"\"\"
    ret = []
    for item in a:
        if item in b:
            ret.append(item)
    return ret

def sum_common_factors(a: int, b: int) -> int:
    \"\"\"Return the sum of all common prime factors of $a$ and $b$\"\"\"

    raise NotImplementedError()
""",
                cur_func_name="sum_common_factors",
                cur_func_doc="Return the sum of all common prime factors of $a$ and $b$",
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''First, I need to get the prime factors of $a$ and $b$.
Second, I can use `for` loop to find common element in two factors list.
Finally, sum the common factor list and return the answer.
Here is the `sum_common_factors` function:

```python
def sum_common_factors(a: int, b: int) -> int:
    """Compute the sum of all common prime factors of $a$ and $b$"""
    factors_a = prime_factor(a)
    factors_b = prime_factor(b)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)
```''',
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code="""
def reverse(s: str) -> str:
    \"\"\"Return reversed input string\"\"\"

    raise NotImplementedError()
""",
                cur_func_name="reverse",
                cur_func_doc="Return reversed input string",
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''Here is the code for `reverse` function:

```python
def reverse(s: str) -> str:
    """Return reversed input string"""
    return s[::-1]
```''',
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        func = ctx.lrt.parse(module=(), code=code).find(LrtFunctionDef, func_name)
        func_docstring = func.docstring if func else None
        return {
            "role": "user",
            "content": self._format_user_request(lang, code, func_name, func_docstring or ""),
        }

    def wrap_prompt_iter(self, ctx, ancestors, func, descendants, keep_ancestors):
        # overriding impl here is necessary because:
        #   - function body must be hidden (becoming a stub)
        #   - the to-be-conquered func must be the last one
        func_stub = ctx.lrt._parse.make_stub_function_def_from_func(func)
        program = LrtProgram(module=(), nodes=[*(ancestors if keep_ancestors else []), *descendants, func_stub])
        code = ctx.lrt.pretty_fmt(program)

        p_system, *p_few_shots = self.get_few_shots
        p_next_turn = self.get_next_turn(ctx, func.name, ctx.lrt.lang, code)
        for nshots in self.get_few_shot_ids():
            few_shots = sum(([p_few_shots[2 * i], p_few_shots[2 * i + 1]] for i in nshots), [])
            yield [p_system, *few_shots, p_next_turn]
        return

    pass


class PaTHumanEvalFuncCallPrompt(MakeTestPrompt):
    """Generates explicit function calls to a given function. Arguments are
    provided by the LLM, e.g. `gcd(10, 25)  # 5`."""

    get_skip_closures = True

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a proficient software engineer and architect, specialized in testing, capable of observing and finding security holes and logical issues in code that spans across multiple computer science fields and mathematics. You are given a piece of Python function, and are expected to produce some test cases for that function. Specifically:

  - You should invoke the function in a one-liner fashion.
  - Do not bring in imports other than what's already imported. Use the pre-declared imports in the original function only.
  - The callee may have multiple arguments, treat them with care.
  - You **must** respect the function signature and docstring, and be aware so you don't generate illegal inputs.
  - Keep the inputs simple but general, and that either edge cases or common cases are meaningful.

Here follows a series of mutually uncorrelated functions to test, one per conversation.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": '''Let's think step by step and create some tests for the following function `check_valid_brackets(...)` in Python.

```python
def check_valid_brackets(seq: str) -> bool:
    """Determine if a bracket sequence consisting of '(', ')', '{', '}', '['
    and ']' is valid."""

    mapping = {')': '(', '}': '{', ']': '['}
    stack = []
    for c in seq:
        if c in mapping:
            if not stack or stack[-1] != mapping[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return not stack
```

Store your test cases for `check_valid_brackets(...)` as function calls, one per line. They will be called later.''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, I can create some test cases for the `check_valid_brackets` function. We can either choose to test it with a valid bracket sequence or an invalid one. Empty strings are also considerable. Here are some test cases for the function:

```python
check_valid_brackets("()")  # True
check_valid_brackets("(([[]]))")  # True
check_valid_brackets("((())")  # False
check_valid_brackets("()[]{}")  # True
check_valid_brackets("([)]")  # False
check_valid_brackets("")  # True
check_valid_brackets(")(")  # False
```
""",
        },
        {
            "role": "system",
            "name": "user",
            "content": '''Let's think step by step and create some tests for the following function `gcd(...)` in Python.

```python
def lcm(a: int, b: int) -> int:
    """Find the least common multiple of `a` and `b`"""
    return round(a * b / gcd(a, b))

def gcd(a: int, b: int) -> int:
    """Find the greatest common divisor of `a` and `b`"""
    while b:
        a, b = b, a % b
    return a
```

Store your test cases for `gcd(...)` as function calls, one per line. They will be called later.''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, I can create some test cases for the `gcd` function. Here are some test cases for the function:

```python
gcd(1, 5)  # 1
gcd(15, 25)  # 5
gcd(5, 7)  # 1
gcd(32, 32)  # 32
gcd(64, 92)  # 4
```
""",
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        return {
            "role": "user",
            "content": f"""
Let's think step by step and create some tests for the following function `{func_name}(...)` in Python.

```{lang}
{code}
```

Store your test cases for `{func_name}(...)` as function calls, one per line. They will be called later.""",
        }

    def parse_tests(self, ctx, func, raw_message):
        raw_md_tags = {"py", "python", "python3"}
        raw_md_blocks = [c for tag, c in extract_md_code(raw_message) if tag in raw_md_tags]

        raw_invokes: list[str] = []
        for md_block in raw_md_blocks:
            for invoke in ctx.lrt._parse.iter_repl_statements(
                lambda line: line.strip().startswith(f"{func.name}("), md_block
            ):
                raw_invokes.append(invoke)

        results: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
        for invoke in raw_invokes:
            lines = invoke.split("\n")
            # previous lines may represent intermediate variables
            if not lines[-1].strip().startswith("return"):
                lines[-1] = "return " + lines[-1]
            lines = [f"    {line}\n" for line in lines]
            code = "".join(lines)
            code = f"def _test_main(_seed: int):\n{code}"
            # parse & extract
            program = ctx.lrt.parse(module=(), code=code)
            if program == None:
                continue
            program = ctx.lrt.prettify(program)
            fn = program.cast_as(LrtFunctionDef)
            results.append((TestType.call, program, fn))

        return results

    pass


class PaTHumanEvalArgsMakerPrompt(MakeTestPrompt):
    """Invokes function with arguments that are generated on-the-fly by a
    procedure, e.g. `gcd(i * 2, i * 3) for i in range(10)`."""

    get_skip_closures = True

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a proficient software engineer and architect, specialized in testing, capable of observing and finding security holes and logical issues in code that spans across multiple computer science fields and mathematics. You are given a piece of Python function, and are expected to produce some sample inputs for that function. Specifically:

  - You should create separate input generators functions for each kind of input, and call the function with the generated input.
  - Do not bring in imports other than what's already imported. Use the pre-declared imports in the original function only.
  - The callee may have multiple arguments, treat them with care.
  - You **must** respect the function signature and docstring, and be aware so you don't generate illegal inputs.
  - Keep the inputs simple but general, and that either edge cases or common cases are meaningful.

Here follows a series of mutually uncorrelated functions to inspect, one per conversation.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": '''Let's think step by step and create some inputs for the following function `check_valid_brackets(...)` in Python.

```python
def check_valid_brackets(seq: str) -> bool:
    """Determine if a bracket sequence consisting of '(', ')', '{', '}', '['
    and ']' is valid."""

    mapping = {')': '(', '}': '{', ']': '['}
    stack = []
    for c in seq:
        if c in mapping:
            if not stack or stack[-1] != mapping[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return not stack
```

Store your test cases for `check_valid_brackets(...)` as function calls per snippet. They will be called later.''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, I can create some inputs for the `check_valid_brackets` function. We can either choose to test it with a valid bracket sequence or an invalid one. Empty strings are also considerable. Here are some sample inputs for the function:

```python
def call_with_empty_string(seed: int) -> bool:
    random.seed(seed)
    return check_valid_brackets("")

def call_with_valid_brackets(seed: int) -> bool:
    random.seed(seed)
    length = random.randint(1, 20)
    brackets = ""
    for _ in range(length):
        l, r = random.choice(["()", "[]", "{}"])
        if random.random() < 0.5:
            brackets += l + r
        else:
            brackets = l + brackets + r
    return check_valid_brackets(brackets)

def call_with_any_brackets(seed: int) -> bool:
    random.seed(seed)
    length = random.randint(1, 8)
    brackets = ""
    for _ in range(length):
        brackets += random.choice("(){}[]")
    return check_valid_brackets(brackets)
```
""",
        },
        {
            "role": "system",
            "name": "user",
            "content": '''Let's think step by step and create some inputs for the following function `gcd(...)` in Python.

```python
def lcm(a: int, b: int) -> int:
    """Find the least common multiple of `a` and `b`"""
    return round(a * b / gcd(a, b))

def gcd(a: int, b: int) -> int:
    """Find the greatest common divisor of `a` and `b`"""
    while b:
        a, b = b, a % b
    return a
```

Store your test cases for `gcd(...)` as function calls per snippet. They will be called later.''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, I can create some test cases for the `gcd` function. Here are some test cases for the function:

```python
def call_with_one(seed: int) -> int:
    random.seed(seed)
    a = 1
    b = random.randint(1, 100)
    if random.random() < 0.5:
        a, b = b, a
    return gcd(a, b)

def call_with_known_divisor(seed: int) -> int:
    random.seed(seed)
    a = random.randint(1, 10**8)
    b = random.randint(1, 10**8)
    divisor = random.randint(1, 10**6)
    return gcd(a * divisor, b * divisor)

def call_with_multiples(seed: int) -> int:
    random.seed(seed)
    a = random.randint(1, 10**8)
    b = random.randint(1, 10**8)
    return gcd(a, a * b)
```
""",
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        return {
            "role": "user",
            "content": f"""
Let's think step by step and create some inputs for the following function `{func_name}(...)` in Python.

```{lang}
{code}
```

Store your test cases for `{func_name}(...)` as function calls per snippet. They will be called later.""",
        }

    def parse_tests(self, ctx, func, raw_message):
        raw_md_tags = {"py", "python", "python3"}
        raw_md_blocks = [c for tag, c in extract_md_code(raw_message) if tag in raw_md_tags]

        results: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
        for md_block in raw_md_blocks:
            program = ctx.lrt.parse(module=(), code=md_block)
            funcs = program.find_all(LrtFunctionDef, None)
            for func in funcs:
                func.name = f"_test_main_{func.name}"
                func = ctx.lrt._parse.fmt_function_def(func)
                program = LrtProgram(module=(), nodes=[func])
                results.append((TestType.call, program, func))

        return results

    pass


class PaTHumanEvalUnitTestPrompt(MakeTestPrompt):
    """Assert that the function has certain behavior. Both inputs and outputs
    are provided by the LLM, e.g. `assert gcd(10, 25) == 5`."""

    get_skip_closures = True

    get_few_shots = [
        {
            "role": "system",
            "content": """/no_think You are a proficient software engineer and architect, specialized in testing, capable of observing and finding security holes and logical issues in code that spans across multiple computer science fields and mathematics. You are given a piece of Python function, and are expected to produce some test cases for that function. Specifically:

  - You should invoke the function and assert its results in a one-liner fashion.
  - Do not bring in imports other than what's already imported. Use the pre-declared imports in the original function only.
  - The callee may have multiple arguments, treat them with care.
  - You **must** respect the function signature and docstring, and be aware so you don't generate illegal inputs.
  - Keep the inputs & outputs simple but general, and that either edge cases or common cases are meaningful.

Here follows a series of mutually uncorrelated functions to test, one per conversation.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": '''Let's think step by step and create some tests for the following function `lcm(...)` in Python.

```python
def lcm(a: int, b: int) -> int:
    """Find the least common multiple of `a` and `b`. Samples:

    >>> lcm(3, 5)
    15
    >>> lcm(4, 6)
    12
    """

    return round(a * b / gcd(a, b))
```

Store your test cases for `lcm(...)` as assertions, one per line. They will be called later.''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": """Sure, I can create some test cases for the `check_valid_brackets` function. We consider the following cases: 1. the two operands are not co-prime and has common factors; 2. the two operands are equal; 3. one of them is 1; 4. two of them is 1; 5. both operands are primes. Here is an example of these test cases in Python:

```python
assert lcm(15, 25) == 75
assert lcm(32, 32) == 32
assert lcm(1, 5) == 5
assert lcm(1, 1) == 1
assert lcm(17, 19) == 17 * 19
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
        raw_md_tags = {"py", "python", "python3"}
        raw_md_blocks = [c for tag, c in extract_md_code(raw_message) if tag in raw_md_tags]

        raw_invokes: list[str] = []
        for md_block in raw_md_blocks:
            for invoke in ctx.lrt._parse.iter_repl_statements(
                lambda line: line.strip().startswith(f"assert {func.name}("), md_block
            ):
                raw_invokes.append(invoke)

        results: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []
        for invoke in raw_invokes:
            lines = invoke.split("\n")
            lines = [f"    {line}\n" for line in lines]
            code = "".join(lines)
            code = f"def _test_main(_seed: int) -> bool:\n{code}    return True\n"
            # parse & extract
            program = ctx.lrt.parse(module=(), code=code)
            program = ctx.lrt.prettify(program)
            fn = program.cast_as(LrtFunctionDef)
            results.append((TestType.unittest, program, fn))

        return results

    pass
