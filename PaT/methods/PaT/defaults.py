from ...utils.strings import code_block

DEFAULT_IMPORTS = """
    import cmath
    import collections
    import itertools
    import math
    import queue
    import random
    import re
    from typing import Any, AsyncGenerator, Callable, Dict, Generator, Generic, Iterable, List, Literal, Optional, Set, Tuple, TypeAlias, TypeVar, Union

    import numpy as np
    import pydantic
    import scipy
    import sympy as sp
"""
DEFAULT_IMPORTS = code_block(DEFAULT_IMPORTS).strip()
