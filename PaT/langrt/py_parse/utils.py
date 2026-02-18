import pathlib
import site
import sys

import autoflake
import stdlib_list
from cachetools import cached

from ..types import CodeBlock


@cached(cache={})
def ppy_get_python_packages(incl_std: bool = False, incl_3p: bool = False) -> set[str]:
    pkgs: set[str] = set()
    pkgs_roots = site.getsitepackages() + [site.getusersitepackages()]
    roots: list[pathlib.Path] = []
    if incl_std:
        _known = stdlib_list.stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}")
        pkgs.update(_known)
        for pr in pkgs_roots:
            roots.append(pathlib.Path(pr).parent)
    if incl_3p:
        for pr in pkgs_roots:
            roots.append(pathlib.Path(pr))
    # hack through package dir
    for root in roots:
        if not root.exists():
            continue
        for p in root.iterdir():
            # not verified to work with symlinks or not
            if p.is_dir() and p.name.lower() != "site-packages":
                if p.name.isidentifier():
                    pkgs.add(p.name)
            elif p.is_file() and p.suffix.lower().startswith(".py"):
                if p.stem.isidentifier():
                    pkgs.add(p.stem)
    return pkgs


def ppy_organize_code_imports(code: CodeBlock) -> CodeBlock:
    code = autoflake.fix_code(code, remove_all_unused_imports=True)
    return code
