from .types import JsonSerializable, LrtExecutionEnv, LrtExecutionResult


class LrtExecutor:
    def __init__(self):
        pass

    async def run(
        self,
        env: LrtExecutionEnv,
        args: list[JsonSerializable],
        kwargs: dict[str, JsonSerializable],
        stdin: str = "",
        timeout: float = 1.0,
    ) -> LrtExecutionResult:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    pass
