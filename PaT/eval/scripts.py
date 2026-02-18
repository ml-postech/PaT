import argparse
import asyncio
import json
import pathlib
import sys
import traceback
from typing import Any, Callable, Coroutine, Generator, cast

import wandb

from ..methods.shared import CodeGenContext, CodeGenJournal, CodeGenJournalist, CodeGenMethod
from ..utils.types import anything_into_dict
from .config import EvalConfig, get_eval_config
from .hparams import HParams
from .resources import download_all_tasks, pick_code_gen_ctx, pick_method, pick_tasks
from .types import CodeGenEvalTasks, EvalResult, _Input, _Verdict


def run_download_datasets(cfg: EvalConfig) -> None:
    """Preload all datasets to local disk. They were not checked in to the repo
    for size, clarity and licensing (?) concerns. This procedure will download
    and pre-process all datasets for the evaluation."""

    download_all_tasks(cfg_proxy=cfg.misc.default_proxy or None)
    return


async def run_draft(
    cfg: EvalConfig,
    hparams: HParams,
    results_dir: pathlib.Path,
    parallelism: int,
    skip_done: bool,
) -> None:
    """Generate programs for a test, but do not evaluate. This is useful for
    cases where GPU time is very costly, and we want to offload the judging to
    cheaper CPU workers. The stage:

      - Reads `$results_dir/.hparams.json` for experiment config,
      - Writes journal to `$results_dir/$task_id.dbg.json`,
      - Writes results to `$results_dir/$task_id.out.json`.

    Task is skipped iff `.out.json` contains generated code.
    TODO: Currently running all tasks sequentially. Consider parallel."""

    ctx = pick_code_gen_ctx(cfg, hparams)
    tasks = pick_tasks(cfg, hparams.task)
    method = pick_method(hparams.method)

    # resume wandb run
    if cfg.misc.wandb_enabled:
        if hparams.wandb_run_id:
            try:
                wandb.init(
                    project=cfg.misc.wandb_project or "PaT",
                    id=hparams.wandb_run_id,
                    resume="must",
                )
            except Exception:
                pass
        if wandb.run is None:
            code_dir = (pathlib.Path(__file__).parent / "../").resolve()
            wandb.init(
                project=cfg.misc.wandb_project or "PaT",
                config=hparams.dump_flattened(),
                settings=wandb.Settings(code_dir=code_dir.as_posix()),
                save_code=True,
            )
        else:
            hparams.wandb_run_id = wandb.run.id
        _save_hparams(results_dir, hparams)

    # main logic
    try:
        if parallelism <= 1:
            for task_counter, (task_id, task) in enumerate(tasks.iter()):
                await _run_draft_logic(ctx, method, tasks, task_counter, task_id, task, results_dir, skip_done)
        else:
            await _run_draft_parallel(
                ctx=ctx,
                tasks=tasks,
                call=lambda _ctx, task_counter, task_id, task: _run_draft_logic(
                    _ctx, method, tasks, task_counter, task_id, task, results_dir, skip_done
                ),
                parallelism=parallelism,
            )
    except KeyboardInterrupt:
        print("  ! user interrupted, stopping kernel")
        return

    # upload code
    if cfg.misc.wandb_enabled and wandb.run is not None:
        artifact = wandb.Artifact(name=f"{wandb.run.id}-draft", type="draft")
        artifact.add_dir(local_path=str(results_dir))
        wandb.run.log_artifact(artifact)
    return


async def _run_draft_parallel(
    ctx: CodeGenContext,
    tasks: CodeGenEvalTasks[_Input, _Verdict],
    call: Callable[[CodeGenContext, int, str, _Input], Coroutine[Any, Any, None]],
    parallelism: int,
) -> None:
    job_queue: asyncio.Queue[tuple[int, str, _Input] | None] = asyncio.Queue()
    for task_counter, (task_id, task) in enumerate(tasks.iter()):
        await job_queue.put((task_counter, task_id, task))
    for _ in range(parallelism * 16 + 17):
        await job_queue.put(None)

    async def _grace_kill() -> None:
        while not job_queue.empty():
            try:
                job_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        for _ in range(parallelism * 16 + 17):
            await job_queue.put(None)
        return

    async def _thread(thread_id: int) -> None:
        while True:
            job = await job_queue.get()
            if job is None:
                break
            task_counter, task_id, task = job
            try:
                await call(ctx, task_counter, task_id, task)
            except KeyboardInterrupt:
                ctx.log.warn(f"thread {thread_id} interrupted")
                await _grace_kill()
            except Exception as err:
                ctx.log.error(f"thread {thread_id} failed: {err}")
                ctx.log.trace()
                await _grace_kill()
        return

    ctx.log.string(f"running {parallelism} threads in parallel")
    threads: list[Coroutine[Any, Any, None]] = []
    for thread_id in range(parallelism):
        threads.append(_thread(thread_id))
    try:
        await asyncio.gather(*threads)
    except KeyboardInterrupt:
        ctx.log.warn("user interrupted process")
    return


async def _run_draft_logic(
    ctx: CodeGenContext,
    method: CodeGenMethod,
    tasks: CodeGenEvalTasks[_Input, _Verdict],
    task_counter: int,
    task_id: str,
    task: _Input,
    results_dir: pathlib.Path,
    skip_done: bool,
) -> None:
    ctx.log.in_scope(f"{tasks.name}#{task_counter}")

    out_json = results_dir / f"{task_id}.out.json"
    out_debug = results_dir / f"{task_id}.dbg.json"
    # is it completed? can we skip it just yet?
    out_ok = out_json.exists() and out_debug.exists()
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            out = json.load(f)
        out_ok = out_ok and (out["code"] is not None)
    except Exception:
        pass
    if skip_done and out_ok:
        print(f"  . skipping draft/{task_id} as is already completed")
        return

    # run problem & catch errors
    ctx.log.epic(f"PROBLEM '{task_id}' (#{task_counter})")
    ctx.log.string(json.dumps(tasks.debug_fmt(task), indent=2, ensure_ascii=False))
    try:
        result, _sj = await tasks.execute(ctx, method, task_id, task)
    except Exception:
        exc = "".join(traceback.format_exception(*sys.exc_info()))
        print(exc)
        result: EvalResult[_Input, _Verdict] = {
            "id": task_id,
            "task": task,
            "code": None,
            "_code_error": exc,
            "_code_tree": None,
            "verdict": None,
            "_verdict_info": None,
        }
        _sj = CodeGenJournalist.just_error("main", exc)

    # save results
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    journal_dict = anything_into_dict(("CodeGenJournal", CodeGenJournal), _sj)
    with open(out_debug, "w", encoding="utf-8") as f:
        json.dump(journal_dict, f, indent=2, ensure_ascii=False)
    return


def get_difficulty_category(difficulty_score: int) -> str:
    print(difficulty_score)
    if (difficulty_score == None) or (difficulty_score < 1200):
        return "Easy"
    elif 1200 <= difficulty_score <= 1599:
        return "Mid"
    elif 1600 <= difficulty_score <= 1999:
        return "Hard"
    else: # difficulty_score >= 2000
        return "Expert"


async def run_judge(
    cfg: EvalConfig,
    hparams: HParams,
    results_dir: pathlib.Path,
    skip_done: bool,
    upload_only: bool,
) -> None:
    """Judge the results for a `draft` session after completion. The stage:

      - Reads `$results_dir/.hparams.json` for experiment config,
      - Reads results from `$results_dir/$task_id.out.json`,
      - Appends verdicts to `$results_dir/.results.jsonl`,
      - Writes summary to `$results_dir/.results.txt`.

    Task is skipped iff `.results.jsonl` contains a verdict for that task ID."""
    ctx = pick_code_gen_ctx(cfg, hparams)
    tasks = pick_tasks(cfg, hparams.task)
    
    try:
        overall_stats, category_stats = await _run_judge_logic(ctx, tasks, hparams.task.task_name, results_dir, skip_done=skip_done, upload_only=upload_only)
    except KeyboardInterrupt:
        print("  ! user interrupted, stopping judge")
        return
    
    if hparams.task.task_name == 'xCodeEval':
        difficulty_report_filename = pathlib.Path(results_dir) / ".defficulty_accuracy_report.txt"
        total_problems = sum(s["total"] for s in category_stats.values())
        with open(difficulty_report_filename, 'w', encoding='utf-8') as f:
            if total_problems == 0:
                report_line = "no data.\n"
                f.write(report_line)
                print(report_line)
            else:
                category_order = ["Easy", "Mid", "Hard", "Expert"]
                for category in category_order:
                    stats = category_stats[category]
                    total = stats["total"]
                    passed = stats["passed"]
                    
                    accuracy = (passed / total * 100) if total > 0 else 0.0
                    
                    report_line = (
                        f"[{category} ({total})]:\n"
                        f"  - Total: {total}\n"
                        f"  - Generated: {passed}\n"
                        f"  - Accepted: {accuracy:.2f}%\n"
                        f"----------------------------------------\n"
                    )
                    f.write(report_line)
                    print(report_line, end='')

    # sync results to wandb
    if cfg.misc.wandb_enabled and hparams.wandb_run_id is not None:
        run = wandb.init(
            project=cfg.misc.wandb_project or "PaT",
            id=hparams.wandb_run_id,
            resume="must",
        )
        if run is not None:
            summary = hparams.dump_flattened()
            for k, v in summary.items():
                run.summary[k] = v
            artifact = wandb.Artifact(f"{hparams.wandb_run_id}-judge", "judge")
            artifact.add_file(local_path=(pathlib.Path(results_dir) / ".results.jsonl").as_posix())
            artifact.add_file(local_path=(pathlib.Path(results_dir) / ".results.txt").as_posix())
            if hparams.task.task_name == "xCodeEval":
                print('b')
                artifact.add_file(local_path=difficulty_report_filename.as_posix()) 
            run.log_artifact(artifact)

    return


async def _run_judge_logic(
    ctx: CodeGenContext,
    tasks: CodeGenEvalTasks[_Input, _Verdict],
    task_name: str,
    results_dir: pathlib.Path,
    skip_done: bool,
    upload_only: bool,
) -> dict[str, list[int | float]]:
    assert results_dir.exists() and results_dir.is_dir()
    results_jsonl = results_dir / ".results.jsonl"
    results_txt = results_dir / ".results.txt"

    # use cached results whenever possible
    all_generated: dict[str, bool] = {}
    all_verdicts: dict[str, float | None] = {}
    for _result in _load_jsonl(results_jsonl):
        result = cast(EvalResult[_Input, _Verdict], _result)
        all_generated[result["id"]] = result["code"] is not None
        all_verdicts[result["id"]] = result["verdict"]

    # main judge logic
    eval_results = _iter_eval_results(results_dir, ".out.json")
    f_jsonl = open(results_jsonl, "a", encoding="utf-8")
    for _i, result in enumerate(eval_results):
        if upload_only:
            continue
        if skip_done and result["id"] in all_verdicts:
            print(f"  . skipping judge/{result['id']} as is already completed")
            continue
        # we trust that the judge program does not fail
        verdict = await tasks.judge(ctx, result)
        all_generated[result["id"]] = result["code"] is not None
        all_verdicts[result["id"]] = verdict["verdict"]
        # print verdict to console
        print(f"  - ({_i + 1}/{len(eval_results)}) {result['id']}: {verdict['verdict']}")
        f_jsonl.write(json.dumps(verdict, indent=None, ensure_ascii=False) + "\n")
        f_jsonl.flush()
    f_jsonl.close()

    xcodeeval_category_stats = {
        "Easy": {"total": 0, "passed": 0},
        "Mid": {"total": 0, "passed": 0},
        "Hard": {"total": 0, "passed": 0},
        "Expert": {"total": 0, "passed": 0}
    }

    for result in eval_results:
        if result["id"] in all_verdicts:
            task_source = None
            if "task" in result and "task_id" in result["task"]:
                task_id_parts = result["task"]["task_id"].split('/', 1)
                task_source = task_id_parts[0] if task_id_parts else None
            
            if task_name == "xCodeEval":
                if "task" in result and "problem" in result["task"] and "difficulty" in result["task"]["problem"]:
                    difficulty_score = result["task"]["problem"]["difficulty"]
                    category = get_difficulty_category(difficulty_score)
                    
                    xcodeeval_category_stats[category]["total"] += 1
                    if (all_verdicts[result["id"]] is True) or ((all_verdicts[result["id"]] or 0) > 0.5):
                        xcodeeval_category_stats[category]["passed"] += 1

    # save results for txt
    with open(results_txt, "w", encoding="utf-8") as f:
        f_print = lambda x: f.write(x + "\n") and print(x)
        tot = len(all_generated)
        ok_gen = sum(int(i) for i in all_generated.values())
        ok_ac = sum(int((i or 0) > 0.5) for i in all_verdicts.values())
        f_print(f"Generated: {ok_gen} / {tot}, {ok_gen/tot:.2%}")
        f_print(f" Accepted: {ok_ac} / {tot}, {ok_ac/tot:.2%}")
        f_print(f"--------------------------------")
        for the_id, verdict in sorted(all_verdicts.items()):
            v_s = "-" if verdict is None else "Accepted" if verdict > 0.5 else "fail"
            f_print(f"{the_id}: {v_s}")

    return {
        "generated": [ok_gen, tot, ok_gen / tot],
        "accepted": [ok_ac, tot, ok_gen / tot],
    }, xcodeeval_category_stats


def _iter_eval_results(root: pathlib.Path, suffix: str) -> list[EvalResult[Any, Any]]:
    files: list[str] = []
    for item in root.iterdir():
        if item.is_file() and item.name.lower().endswith(suffix.lower()):
            files.append(item.name)
    max_file_len = max(len(item) for item in files)
    files.sort(key=lambda x: x.rjust(max_file_len, " "))
    results: list[EvalResult[Any, Any]] = []
    for item in files:
        with open(root / item, "r", encoding="utf-8") as f:
            try:
                results.append(json.load(f))
            except Exception as err:
                print(err)
                pass
    return results


def _load_jsonl(path: pathlib.Path) -> Generator[dict, None, None]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    pass
    return


def _save_hparams(results_dir: pathlib.Path, hparams: HParams) -> None:
    with open(results_dir / ".hparams.json", "w", encoding="utf-8") as f:
        raw = json.dumps(hparams.dump(), indent=2)
        f.write(raw + "\n")
    return


async def run_get_split(cfg: EvalConfig, hparams: HParams) -> None:
    """Discover all task IDs that will be used throughout the evaluation."""

    tasks = pick_tasks(cfg, hparams.task)

    ids: list[str] = []
    for task_id, _item in tasks.iter():
        ids.append(task_id)
    print(json.dumps(ids, indent=2, ensure_ascii=False))
    return


async def eval_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download-datasets", "draft", "judge", "get-split"])
    parser.add_argument("--results-dir", type=pathlib.Path)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    # for key, field_type in HParams.dump_all_flattened_keys().items():
    #     if field_type is bool:
    #         parser.add_argument(f"--{key}", type=eval, choices=[True, False], required=False)
    #     elif isinstance(field_type, type) and issubclass(field_type, enum.Enum):
    #         enum_values = [i.value for i in field_type]
    #         parser.add_argument(f"--{key}", type=field_type, choices=enum_values, required=False)
    #     else:
    #         parser_type = field_type if field_type in (int, float) else str
    #         parser.add_argument(f"--{key}", type=parser_type, required=False)

    args = parser.parse_args()
    cfg = get_eval_config()
    if args.results_dir is not None:
        results_dir = pathlib.Path(args.results_dir)
        with open(results_dir / ".hparams.json", "r", encoding="utf-8") as f:
            hparams = HParams.load(json.loads(f.read()))
    else:
        results_dir = None
        hparams = None

    if args.command == "download-datasets":
        run_download_datasets(cfg)
    elif args.command == "draft":
        assert results_dir is not None and hparams is not None
        await run_draft(cfg, hparams, results_dir, parallelism=args.parallelism, skip_done=bool(args.skip_done))
    elif args.command == "judge":
        assert results_dir is not None and hparams is not None
        await run_judge(cfg, hparams, results_dir, skip_done=bool(args.skip_done), upload_only=bool(args.upload_only))
    elif args.command == "get-split":
        assert results_dir is not None and hparams is not None
        await run_get_split(cfg, hparams)
    else:
        raise ValueError(f"unknown command: {args.command}")
    pass
