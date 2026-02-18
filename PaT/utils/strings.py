from typing import overload


def wrap_code_inline(s: str) -> str:
    """.. so we can put code in Markdown."""

    lines = [i.rstrip() for i in s.split("\n")]
    s = "\n".join(lines).strip("\n")
    return s


@overload
def code_block(q: str, indent: int | None = None) -> str: ...
@overload
def code_block(q: list[str], indent: int | None = None) -> list[str]: ...
@overload
def code_block(q: dict[str, str], indent: int | None = None) -> dict[str, str]: ...
def code_block(q: str | list[str] | dict[str, str], indent: int | None = None) -> str | list[str] | dict[str, str]:
    """Un-indent text. Usage:

    ```py
    the_prompt = '''
        # use indent inside safely
    '''
    the_prompt = [code1, code2, ...]
    the_prompt = {'a': code1, 'b': code2, ...}
    the_prompt = code_block(the_prompt)
    ```"""

    if isinstance(q, list):
        return [code_block(x) for x in q]
    elif isinstance(q, dict):
        return {k: code_block(v) for k, v in q.items()}

    if indent is None:
        lines = [line.rstrip() for line in q.split("\n")]
        lines = [line for line in lines if line]
        indents = [len(line) - len(line.lstrip()) for line in lines]
        indent = -indents[0] if len(indents) else 0

    result: list[str] = []
    for line in q.split("\n"):
        line = line.rstrip()
        rem = min(len(line) - len(line.lstrip()), -indent)
        line = line[rem:] if rem > 0 else (" " * indent + line)
        result.append(line.rstrip())

    code = "\n".join(result)
    code = code.strip("\n")
    code += "\n"
    return code


def extract_md_code(markdown: str) -> list[tuple[str, str]]:
    """```py... -> ('py', '...'), ('ts', '...'), ..."""

    result: list[tuple[str, str]] = []
    buffer: list[str] = []
    current_tag: str | None = None

    for line in markdown.split("\n"):
        if current_tag is None:  # plain text mode
            if line.startswith("```"):
                current_tag = line[3:]
            elif line.startswith(" ```"):
                # some LLMs like Codestral has malformed output
                current_tag = line[4:]
            else:
                pass

        else:  # capturing mode
            if line.startswith("```") or line.startswith(" ```"):
                result.append((current_tag, "\n".join(buffer)))
                buffer = []
                current_tag = None
            else:
                buffer.append(line)
        pass

    result = [(t, c) for t, c in result]
    return result


def compare_strings_cf(gt: list[str], hyp: str) -> str | None:
    """Compare strings the OI/ACM way -- ignore whitespaces whatsoever. There
    provides a range of ground truths such that matching either would consist a
    correct. Returns the error reason if any, otherwise None."""

    _gt_keys = set[str]()
    gs_db: list[list[str]] = []
    for g in gt:
        gs = g.split()
        gs_key = " ".join(gs)
        if gs_key not in _gt_keys:
            _gt_keys.add(gs_key)
            gs_db.append(gs)

    def _fuzzy_eq(g: str, h: str) -> bool:
        try:
            ig, ih = int(g), int(h)
            return ig == ih
        except ValueError:
            pass
        try:
            fg, fh = float(g), float(h)
            return abs(fg - fh) < 1e-6
        except ValueError:
            pass
        if g.lower() in {"yes", "no"}:
            return g.lower() == h.lower()
        return g == h

    hs = hyp.split()
    err: str | None = None
    for gs in gs_db:
        ok = True
        for i in range(min(len(gs), len(hs))):
            if not _fuzzy_eq(gs[i], hs[i]):
                ok = False
                err = f"expected: '{gs[i]}', found: '{hs[i]}' [{i + 1}th token]"
                break
        if ok and len(hs) < len(gs):
            ok = False
            err = f"expected: '{gs[len(hs)]}', found: nothing [{len(hs) + 1}th token]"
        if ok and len(hs) > len(gs):
            ok = False
            err = f"expected: nothing, found: '{hs[len(gs)]}' [{len(gs) + 1}th token]"
        if ok:
            return None
    return err


def wrap_string_as_triple_quotes(s: str) -> str:
    """Put string into Pythonic triple quotes."""

    def _escape(li: str, single_quote: bool, double_quote: bool) -> tuple[bool, str]:
        escaped, ret = False, ""
        for ch in li:
            if ch == "'":
                ret += "\\'" if single_quote else "'"
            elif ch == '"':
                ret += '\\"' if double_quote else '"'
            elif ch == "\r":
                pass
            elif ch == "\n":
                ret += "\n"
            else:
                if ch == "\\":
                    escaped = True
                ret += ch
        return escaped, ret

    if "'" not in s and '"' not in s:
        escaped, s = _escape(s, single_quote=False, double_quote=False)
        ret = f'"""{s}"""'
    elif "'" not in s and '"' in s:
        escaped, s = _escape(s, single_quote=False, double_quote=False)
        ret = f"'''{s}'''"
    elif "'" in s and '"' not in s:
        escaped, s = _escape(s, single_quote=False, double_quote=False)
        ret = f'"""{s}"""'
    else:
        escaped, s = _escape(s, single_quote=False, double_quote=True)
        ret = f'"""{s}"""'
    prefix = "r" if escaped else ""
    return prefix + ret
