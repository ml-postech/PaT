from typing import Literal

import pydantic
import requests


class GitHubFile(pydantic.BaseModel):
    type: Literal["dir", "file"]
    name: str
    path: str
    sha: str
    size: int


class GitHubFS:
    def __init__(self, repo: str, ref: str, cfg_proxy: str | None = None) -> None:
        self.repo = repo
        self.ref = ref
        self.cfg_proxy = cfg_proxy

    def __repr__(self) -> str:
        return f"<GitHubFS repo='{self.repo}', ref='{self.ref}'>"

    def listdir(self, path: str) -> list[GitHubFile]:
        path = path.replace("\\", "/").strip("/")
        ref = self.ref[11:] if self.ref.startswith("refs/heads/") else self.ref
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}?ref={ref}"
        res = self.__get(url)
        results = []
        for _it in res.json():
            this_path = [i for i in path.split("/") + [_it["name"]] if i]
            item = GitHubFile(
                type=_it["type"],
                name=_it["name"],
                path="/" + "/".join(this_path),
                sha=_it["sha"],
                size=_it["size"],
            )
            results.append(item)
        return results

    def read(self, path: str) -> bytes:
        path = path.replace("\\", "/").strip("/")
        url = f"https://github.com/{self.repo}/raw/{self.ref}/{path}"
        res = self.__get(url)
        return res.content

    def __get(self, url: str) -> requests.Response:
        proxies = {}
        if self.cfg_proxy:
            proxies = {
                "http": self.cfg_proxy,
                "https": self.cfg_proxy,
            }
        res = requests.get(url, proxies=proxies)
        res.raise_for_status()
        return res

    pass
