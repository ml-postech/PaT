import copy
import json
import pathlib
from typing import Any

from ...langrt.wrapper import LangRT
from .github import GitHubFS

DISABLED_FEATURE: Any = ...


def download_mbpp_dataset(to_dir: pathlib.Path, cfg_proxy: str | None = None) -> None:
    print("[MBPP dataset]")
    to_dir.mkdir(parents=True, exist_ok=True)

    # download
    the_repo = GitHubFS(
        repo="google-research/google-research",
        # ref="refs/heads/master",
        ref="619904fdca3f5a533383ae36dd4c564464ce91f3",
        cfg_proxy=cfg_proxy,
    )
    print(f"  - opening repo: {the_repo}")
    typed_repo = GitHubFS(
        repo="noahshinn/reflexion",
        # ref="refs/heads/main",
        ref="218cf0ef1df84b05ce379dd4a8e47f17766733a0",
        cfg_proxy=cfg_proxy,
    )
    print(f"  - opening repo: {typed_repo}")

    the_src_path = to_dir / "sanitized-mbpp.json"
    if not the_src_path.exists():
        print(f"  - downloading file: /mbpp/sanitized-mbpp.json")
        the_src_raw = the_repo.read("/mbpp/sanitized-mbpp.json")
        with open(the_src_path, "wb") as f:
            f.write(the_src_raw)
    else:
        print(f"  - file already exists: {to_dir.name}/sanitized-mbpp.json")

    the_types_path = to_dir / "mbpp-py.jsonl"
    if not the_types_path.exists():
        print(f"  - downloading file: /programming_runs/benchmarks/mbpp-py.jsonl")
        the_types_raw = typed_repo.read("/programming_runs/benchmarks/mbpp-py.jsonl")
        with open(the_types_path, "wb") as f:
            f.write(the_types_raw)
    else:
        print(f"  - file already exists: {to_dir.name}/mbpp-py.jsonl")

    print("  - loading sanitized-mbpp.json")
    with open(the_src_path, "r", encoding="utf-8") as f:
        the_src: dict = json.load(f)
    print("  - loading mbpp-py.jsonl")
    with open(the_types_path, "r", encoding="utf-8") as f:
        the_types: list[dict] = [json.loads(x) for x in f.read().split("\n") if x.strip()]
        the_types_dict = {int(line["name"].split("_")[1]): line for line in the_types}

    print("  - patching dataset irregularities")
    patches: list[dict] = [
        {
            # typo in entry function name
            "name": "mbpp_56_checks",
            "[old]entry_point": "checks",
            "[new]entry_point": "check",
        },
    ]
    for item in the_types:
        for patch in patches:
            if item["name"] == patch["name"]:
                for key, value in patch.items():
                    if key.startswith("[new]"):
                        item[key[5:]] = value
                break

    print("  - patching types")
    untyped_results: list[dict] = []
    typed_results: list[dict] = []
    lrt = LangRT.python(
        sandbox_root=DISABLED_FEATURE,
        parallelism=1,
        macos_sandbox_bin=DISABLED_FEATURE,
        python_bin=DISABLED_FEATURE,
    )
    for item in the_src:
        # extract entry
        task_id = item["task_id"]
        test_list = "\n".join(item["test_list"])
        i_program = lrt.parse(module=(), code=item["code"])
        i_entry_funcs = [f for f in i_program.nodes if f.kind == "function" and f.name in test_list]
        assert len(i_entry_funcs) == 1, i_program  # have one and only one entry function
        i_entry_func = i_entry_funcs[0]

        # combine entrypoint with problem description
        i_entry_func.docstring = item["prompt"]
        i_entry_func = lrt._parse.fmt_function_def(i_entry_func)  # commit changes
        item["prompt"] = lrt.pretty_fmt(i_entry_func)
        item["entry_point"] = i_entry_func.name
        untyped_results.append(copy.deepcopy(item))

        # add types
        if task_id not in the_types_dict:
            print(f"    ! task '{item['task_id']}' has to type annotations")
            continue
        item["prompt"] = the_types_dict[task_id]["prompt"]
        item["entry_point"] = the_types_dict[task_id]["entry_point"]
        assert item["entry_point"] in test_list, (item["entry_point"], test_list, item["code"])
        typed_results.append(copy.deepcopy(item))

    print(f"  - saving to {to_dir.name}/MBPP.json")
    with open(to_dir / "MBPP.json", "w", encoding="utf-8") as f:
        json.dump(untyped_results, f, indent=2, ensure_ascii=False)
    print(f"  - saving to {to_dir.name}/MBPP_typed.json")
    with open(to_dir / "MBPP_typed.json", "w", encoding="utf-8") as f:
        json.dump(typed_results, f, indent=2, ensure_ascii=False)

    print("  - done")
    return
