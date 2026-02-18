import gzip
import json
import pathlib

from .github import GitHubFS


def download_humaneval_dataset(to_dir: pathlib.Path, cfg_proxy: str | None = None) -> None:
    print("[HumanEval dataset]")
    to_dir.mkdir(parents=True, exist_ok=True)

    # download
    the_repo = GitHubFS(
        repo="openai/human-eval",
        # ref="refs/heads/master",
        ref="6d43fb980f9fee3c892a914eda09951f772ad10d",
        cfg_proxy=cfg_proxy,
    )
    print(f"  - opening repo: {the_repo}")
    the_jsonl_gz_path = to_dir / "HumanEval.jsonl.gz"
    if not the_jsonl_gz_path.exists():
        print(f"  - downloading file: /data/HumanEval.jsonl.gz")
        the_jsonl_gz = the_repo.read("/data/HumanEval.jsonl.gz")
        with open(the_jsonl_gz_path, "wb") as f:
            f.write(the_jsonl_gz)
    else:
        print(f"  - file already exists: {to_dir.name}/HumanEval.jsonl.gz")

    # parse
    print(f"  - deflating HumanEval.jsonl.gz")
    with open(the_jsonl_gz_path, "rb") as f:
        the_jsonl_gz = f.read()
    the_jsonl = gzip.decompress(the_jsonl_gz)
    the_items: list[dict] = []
    for line in the_jsonl.decode("utf-8", "ignore").strip().split("\n"):
        if line.strip():
            the_items.append(json.loads(line))

    # patch bad data
    patches: list[dict] = [
        {
            # bad test case
            "task_id": "HumanEval/47",
            "[old]prompt": '\n\ndef median(l: list):\n    """Return median of elements in the list l.\n    >>> median([3, 1, 2, 4, 5])\n    3\n    >>> median([-10, 4, 6, 1000, 10, 20])\n    15.0\n    """\n',
            "[new]prompt": '\n\ndef median(l: list):\n    """Return median of elements in the list l.\n    >>> median([3, 1, 2, 4, 5])\n    3\n    >>> median([-10, 4, 6, 1000, 10, 20])\n    8.0\n    """\n',
        },
        {
            # no imports in functions
            "task_id": "HumanEval/115",
            "[old]prompt": '\ndef max_fill(grid, capacity):\n    import math\n    """\n    You are given a rectangular grid of wells. Each row represents a single well,\n    and each 1 in a row represents a single unit of water.\n    Each well has a corresponding bucket that can be used to extract water from it, \n    and all buckets have the same capacity.\n    Your task is to use the buckets to empty the wells.\n    Output the number of times you need to lower the buckets.\n\n    Example 1:\n        Input: \n            grid : [[0,0,1,0], [0,1,0,0], [1,1,1,1]]\n            bucket_capacity : 1\n        Output: 6\n\n    Example 2:\n        Input: \n            grid : [[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]\n            bucket_capacity : 2\n        Output: 5\n    \n    Example 3:\n        Input: \n            grid : [[0,0,0], [0,0,0]]\n            bucket_capacity : 5\n        Output: 0\n\n    Constraints:\n        * all wells have the same length\n        * 1 <= grid.length <= 10^2\n        * 1 <= grid[:,1].length <= 10^2\n        * grid[i][j] -> 0 | 1\n        * 1 <= capacity <= 10\n    """\n',
            "[new]prompt": '\ndef max_fill(grid, capacity):\n    """\n    You are given a rectangular grid of wells. Each row represents a single well,\n    and each 1 in a row represents a single unit of water.\n    Each well has a corresponding bucket that can be used to extract water from it, \n    and all buckets have the same capacity.\n    Your task is to use the buckets to empty the wells.\n    Output the number of times you need to lower the buckets.\n\n    Example 1:\n        Input: \n            grid : [[0,0,1,0], [0,1,0,0], [1,1,1,1]]\n            bucket_capacity : 1\n        Output: 6\n\n    Example 2:\n        Input: \n            grid : [[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]\n            bucket_capacity : 2\n        Output: 5\n    \n    Example 3:\n        Input: \n            grid : [[0,0,0], [0,0,0]]\n            bucket_capacity : 5\n        Output: 0\n\n    Constraints:\n        * all wells have the same length\n        * 1 <= grid.length <= 10^2\n        * 1 <= grid[:,1].length <= 10^2\n        * grid[i][j] -> 0 | 1\n        * 1 <= capacity <= 10\n    """\n',
        },
        {
            # bad test case
            "task_id": "HumanEval/116",
            "[old]prompt": '\ndef sort_array(arr):\n    """\n    In this Kata, you have to sort an array of non-negative integers according to\n    number of ones in their binary representation in ascending order.\n    For similar number of ones, sort based on decimal value.\n\n    It must be implemented like this:\n    >>> sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]\n    >>> sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]\n    >>> sort_array([1, 0, 2, 3, 4]) [0, 1, 2, 3, 4]\n    """\n',
            "[new]prompt": 'def sort_array(arr):\n    """\n    In this Kata, you have to sort an array of non-negative integers according to\n    number of ones in their binary representation in ascending order.\n    For similar number of ones, sort based on decimal value.\n\n    It must be implemented like this:\n    >>> sort_array([1, 5, 2, 3, 4]) == [1, 2, 4, 3, 5]\n    >>> sort_array([-2, -3, -4, -5, -6]) == [-4, -2, -6, -5, -3]\n    >>> sort_array([1, 0, 2, 3, 4]) == [0, 1, 2, 4, 3]\n    """',
        },
        {
            # malformed tuple syntax
            "task_id": "HumanEval/148",
            "[old]prompt": '\ndef bf(planet1, planet2):\n    \'\'\'\n    There are eight planets in our solar system: the closerst to the Sun \n    is Mercury, the next one is Venus, then Earth, Mars, Jupiter, Saturn, \n    Uranus, Neptune.\n    Write a function that takes two planet names as strings planet1 and planet2. \n    The function should return a tuple containing all planets whose orbits are \n    located between the orbit of planet1 and the orbit of planet2, sorted by \n    the proximity to the sun. \n    The function should return an empty tuple if planet1 or planet2\n    are not correct planet names. \n    Examples\n    bf("Jupiter", "Neptune") ==> ("Saturn", "Uranus")\n    bf("Earth", "Mercury") ==> ("Venus")\n    bf("Mercury", "Uranus") ==> ("Venus", "Earth", "Mars", "Jupiter", "Saturn")\n    \'\'\'\n',
            "[new]prompt": '\ndef bf(planet1, planet2):\n    \'\'\'\n    There are eight planets in our solar system: the closerst to the Sun \n    is Mercury, the next one is Venus, then Earth, Mars, Jupiter, Saturn, \n    Uranus, Neptune.\n    Write a function that takes two planet names as strings planet1 and planet2. \n    The function should return a tuple containing all planets whose orbits are \n    located between the orbit of planet1 and the orbit of planet2, sorted by \n    the proximity to the sun. \n    The function should return an empty tuple if planet1 or planet2\n    are not correct planet names. \n    Examples\n    bf("Jupiter", "Neptune") ==> ("Saturn", "Uranus")\n    bf("Earth", "Mercury") ==> ("Venus",)\n    bf("Mercury", "Uranus") ==> ("Venus", "Earth", "Mars", "Jupiter", "Saturn")\n    \'\'\'\n',
        },
    ]
    print(f"  - patching {len(patches)} tasks")
    patched_items: list[dict] = []
    for item in the_items:
        for patch in patches:
            if item["task_id"] == patch["task_id"]:
                for key, value in patch.items():
                    if key.startswith("[new]"):
                        item[key[5:]] = value
                break
        patched_items.append(item)

    # save
    print(f"  - saving to {to_dir.name}/HumanEval_processed.json")
    with open(to_dir / "HumanEval_processed.json", "w", encoding="utf-8") as f:
        patched_json = json.dumps(patched_items, ensure_ascii=True, indent=2)
        f.write(patched_json + "\n")

    print("  - done")
    return
