import pathlib

import requests


def download_xcodeeval_dataset(to_dir: pathlib.Path, cfg_proxy: str | None = None) -> None:
    print("[xCodeEval dataset]")
    to_dir.mkdir(parents=True, exist_ok=True)

    # just download
    proxies = {"http": cfg_proxy, "https": cfg_proxy} if cfg_proxy else None

    url_probs = "https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval/resolve/main/problem_descriptions.jsonl"
    path_probs = to_dir / "problem_descriptions.jsonl"
    if not path_probs.exists():
        print(f"  - downloading file: {to_dir.name}/problem_descriptions.jsonl")
        req = requests.get(url_probs, proxies=proxies)
        with open(path_probs, "wb") as f:
            f.write(req.content)

    url_tests = "https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval/resolve/main/unittest_db.json"
    path_tests = to_dir / "unittest_db.json"
    if not path_tests.exists():
        print(f"  - downloading file: {to_dir.name}/unittest_db.json")
        req = requests.get(url_tests, proxies=proxies)
        with open(path_tests, "wb") as f:
            f.write(req.content)

    print("  - done")
    return
