#!/usr/bin/env python3
"""WorldState-Bench harness 评测入口:render → judge → report 三阶段。

本模块是论文 Table 1 中 STATEFULHARNESS 行的产出链路:同一份 WorldState-Bench
数据集(worldstate_bench/data/cases,B01–B50 × A/B 共 100 clip)在
StatefulHarness(full harness)条件下渲染、判定、出指标。

数据:<数据集>/<CASE>/(input.png + case_{A,B}.json;instruction、gold、
judge 问题全部由数据预写;默认根由 evaluation/casebench/dataset.py 指向
../worldstate_bench/data/cases,可用 --cases 或环境变量
WORLDSTATE_BENCH_CASES 覆盖)。渲染 = instruction 原文包 H3 I2VA 信封
→ 常驻 H3 服务(i2v,5s);判定 = 常驻 Qwen VL 看采样帧逐字回答预写问题;
report 汇总指标(EndingFollow≈ENDFOLLOW / CounterAbsent≈CNTABSENT /
StrictFollow≈STRICT / PairFollow≈PAIRFOLLOW / PairStrict≈PAIRSTRICT,
对照表见 evaluation/README.md)。

用法:
    bash evaluation/ensure_services.sh            # H3 先装载、LLM 后启动
    python3 -m evaluation.run_cases --stage all

    # ref2va 首帧模式(不开白模):input.png 作纯图片 reference,
    # prompt 对齐行换成 ref2va 首帧锚定句;服务 = box2clay socket 上的
    # engine ref2va 常驻实例(复用 wbench runner 的 ensure_service):
    python3 -m evaluation.run_cases --stage all --ref2va

    # 只渲/只判/只出报告(幂等续跑;--case 可多选):
    python3 -m evaluation.run_cases --stage render
    python3 -m evaluation.run_cases --stage judge --case B01
    python3 -m evaluation.run_cases --stage report --run-id <id>

    # 无 GPU 干跑(只打通数据流,结果不可作正式结论):
    python3 -m evaluation.run_cases --stage all --renderer fake --judge heuristic

产物:evaluation/output/casebench/runs/<run_id>/(manifest.json / videos /
frames / verdicts / metrics.json / report.md),--force 重做。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

from evaluation.casebench import report as report_mod
from evaluation.casebench.dataset import Case, filter_cases, iter_twins, load_cases
from evaluation.casebench.frames import judge_images
from evaluation.casebench.headless import run_headless, write_headless
from evaluation.casebench.judge import build_judge
from evaluation.casebench.metrics import aggregate
from evaluation.casebench.render import (build_renderer, final_frame_for,
                                         render_twin, video_path_for)

logger = logging.getLogger("evaluation.run_cases")

OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "casebench"
RUNS_ROOT = OUTPUT_ROOT / "runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("all", "headless", "render", "judge",
                                            "report"),
                        default="all", help="要执行的阶段(all 先跑 headless 数据自洽检查)")
    parser.add_argument("--cases", type=Path, default=None,
                        help="案例根目录(默认 ../worldstate_bench/data/cases)")
    parser.add_argument("--case", action="append", default=[],
                        help="只评该案例(可重复;默认全部)")
    parser.add_argument("--judge", default="parallel",
                        choices=("heuristic", "resident", "parallel"),
                        help="判定后端:parallel=停 H3 后 N 路单卡 LLM 并行"
                             "(默认,最快);resident=与 H3 共存的单路常驻服务;"
                             "heuristic=干跑")
    parser.add_argument("--llm-workers", type=int, default=8,
                        help="parallel 后端的单卡 LLM worker 数(默认 8)")
    parser.add_argument("--keep-fleet", action="store_true",
                        help="判定完成后保留 LLM 机群(默认判完即释放)")
    parser.add_argument("--renderer", default="h3", choices=("fake", "h3"),
                        help="渲染后端(fake 供无 GPU 干跑)")
    parser.add_argument("--ref2va", action="store_true",
                        help="ref2va 首帧模式(不开白模):input.png 作纯图片"
                             " reference 渲染,prompt 对齐行换成 ref2va 首帧"
                             "锚定句;服务用 box2clay socket 的 engine ref2va"
                             " 实例(--socket 可覆盖 socket 路径,此时自管服务)")
    parser.add_argument("--prompt-mode", default="framework",
                        choices=("framework", "envelope", "remain", "raw",
                                 "raw_state"),
                        help="framework=完整游戏框架(GameLoop→规则结算→"
                             "VisualScript→PromptCompiler,默认);"
                             "envelope=instruction 原文包 I2VA 信封;"
                             "remain=摘除 must-NOT 句的直连消融;"
                             "raw=card txt 整段 Inputs 原文(standalone-H3 "
                             "baseline);raw_state=raw + 追加 gold 终态表现"
                             "(baseline 告知版)")
    parser.add_argument("--mode", default="remote",
                        choices=("remote", "session", "subprocess"),
                        help="h3 调用方式;批量渲染用 remote")
    parser.add_argument("--socket", type=Path, help="常驻服务 socket 路径")
    parser.add_argument("--seed", type=int, default=0, help="渲染种子(固定可复现)")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="判定帧采样帧率(默认 3fps ≈ 15 帧/5s 片段)")
    parser.add_argument("--max-frames", type=int, default=32,
                        help="送 VLM 的帧数上限(超出均匀降采样)")
    parser.add_argument("--max-tokens", type=int, default=768, help="判定输出 token 上限")
    parser.add_argument("--config", type=Path, help="config.toml 路径(默认 ./config.toml)")
    parser.add_argument("--run-id", help="运行 id(默认按时间生成;report 阶段可省略取最新)")
    parser.add_argument("--force", action="store_true", help="忽略已有产物,强制重做")
    return parser.parse_args(argv)


def _load_selected(args: argparse.Namespace) -> list[Case]:
    cases = load_cases(args.cases)
    if args.case:
        cases = filter_cases(cases, set(args.case))
    return cases


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    if args.stage == "report":
        runs = sorted(RUNS_ROOT.glob("run-*")) if RUNS_ROOT.is_dir() else []
        if not runs:
            raise SystemExit("没有已存在的 run;请先跑其他阶段或显式传 --run-id")
        return runs[-1].name
    return f"run-{datetime.now():%Y%m%d-%H%M%S}"


def _write_manifest(run_id: str, args: argparse.Namespace, cases: list[Case]) -> Path:
    path = RUNS_ROOT / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run_id": run_id,
        "run_dir": str(path.parent),
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "cases": [{"case_id": case.case_id, "genre": case.genre, "family": case.family,
                   "clips": [twin.clip_id for twin in case.twins]}
                  for case in cases],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# -- 阶段 ---------------------------------------------------------------------


def cmd_headless(args: argparse.Namespace, run_id: str, cases: list[Case]) -> bool:
    """数据自洽检查(零 GPU):五处结局声明交叉互指,不一致即中止后续阶段。"""
    passed, checked = run_headless(cases)
    path = write_headless(RUNS_ROOT / run_id, passed, checked)
    print(f"[headless] {passed}/{len(checked)} clips self-consistent -> {path}")
    for clip, problems in checked.items():
        for problem in problems:
            print(f"[headless]   {clip}: {problem}")
    return passed == len(checked)


def cmd_render(args: argparse.Namespace, run_id: str, cases: list[Case]) -> bool:
    run_dir = RUNS_ROOT / run_id
    socket_path = args.socket
    if args.ref2va and args.renderer == "h3" and socket_path is None:
        from evaluation.casebench.render import ref2va_socket_path

        socket_path = ref2va_socket_path()
        # 复用 wbench runner 的安全起步路径(只探活/锁内起步,绝不停任何
        # 在跑实例;ref2va 与 t2v 互斥由人工/编排层保证)
        from game.wbench.run_wbench import ensure_service

        if not ensure_service("first_frame"):
            raise SystemExit(
                f"ref2va 服务未就绪({socket_path});"
                "确认 8 卡空闲后重试,或检查 "
                f"{socket_path}.log")
    renderer = build_renderer(run_dir, backend=args.renderer, mode=args.mode,
                              socket_path=socket_path, config_path=args.config)
    twins = iter_twins(cases)
    failures: list[str] = []
    rendered = skipped = 0
    try:
        for twin in twins:
            try:
                if args.prompt_mode == "framework":
                    from evaluation.casebench.framework import run_framework

                    _, fresh = run_framework(renderer, twin, run_dir,
                                             seed=args.seed, force=args.force,
                                             ref2va=args.ref2va)
                else:
                    _, fresh = render_twin(renderer, twin, run_dir, seed=args.seed,
                                           force=args.force,
                                           prompt_mode=args.prompt_mode,
                                           ref2va=args.ref2va)
            except Exception as exc:  # noqa: BLE001 - 单 clip 失败不拖垮整批
                failures.append(twin.clip_id)
                logger.warning("[%s] render failed: %s", twin.clip_id, exc)
                continue
            rendered += int(fresh)
            skipped += int(not fresh)
            if fresh:
                print(f"[render] {twin.clip_id} done", flush=True)
    finally:
        renderer.close()
    ready = sum(1 for twin in twins
                if video_path_for(run_dir, twin).is_file()
                and video_path_for(run_dir, twin).stat().st_size > 0)
    print(f"[render] {ready}/{len(twins)} clips ready "
          f"({rendered} newly rendered, {skipped} reused)"
          + (f"; failures: {', '.join(failures)}" if failures else ""))
    return not failures


def cmd_judge(args: argparse.Namespace, run_id: str, cases: list[Case]) -> bool:
    from concurrent.futures import ThreadPoolExecutor

    from engine.renderer.base import FrameExtractionError

    run_dir = RUNS_ROOT / run_id
    fleet_size = args.llm_workers if args.judge == "parallel" else 0
    sockets = None
    if fleet_size:
        from evaluation.casebench.fleet import ensure_fleet, retire_fleet

        print(f"[fleet] retiring H3/shared LLM, starting {fleet_size} "
              "single-GPU workers ...", flush=True)
        sockets = ensure_fleet(fleet_size)
        print(f"[fleet] {fleet_size} workers ready", flush=True)
    judge = build_judge(args.judge, socket_path=args.socket,
                        max_tokens=args.max_tokens, sockets=sockets)
    counts = {"judged": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()

    def work(twin) -> None:
        verdict_file = report_mod.verdict_path(run_dir, twin.clip_id)
        if verdict_file.is_file() and not args.force:
            status, detail = "skipped", "already done"
        else:
            video_path = video_path_for(run_dir, twin)
            if not video_path.is_file() or video_path.stat().st_size == 0:
                status, detail = "failed", "no rendered clip"
            else:
                try:
                    images = judge_images(video_path, final_frame_for(run_dir, twin),
                                          run_dir / "frames" / twin.clip_id,
                                          fps=args.fps, max_frames=args.max_frames)
                    verdict = judge.judge(twin, images)
                except FrameExtractionError as exc:
                    status, detail = "failed", f"frame extraction: {exc}"
                except Exception as exc:  # noqa: BLE001 - 单 clip 失败不拖垮整批
                    status, detail = "failed", f"judge: {exc}"
                else:
                    report_mod.write_verdict(run_dir, twin, verdict,
                                             judge_name=args.judge,
                                             n_images=len(images), images=images,
                                             video_path=video_path)
                    status, detail = "judged", ""
                    print(f"[judge] {twin.clip_id}: outcome={verdict['outcome']!r} "
                          f"counter={verdict['counter']!r} "
                          f"(expected {twin.outcome_question['expected']!r}/no)",
                          flush=True)
        if status == "failed":
            logger.warning("[%s] %s: %s", twin.clip_id, status, detail)
        with lock:
            counts[status] += 1

    try:
        with ThreadPoolExecutor(max_workers=fleet_size or 1) as pool:
            list(pool.map(work, iter_twins(cases)))
    finally:
        judge.close()
        if fleet_size and not args.keep_fleet:
            from evaluation.casebench.fleet import retire_fleet

            retire_fleet(fleet_size)
            print("[fleet] retired (H3 stays down; restart for rendering)",
                  flush=True)
    print(f"[judge] {counts['judged']} newly judged, {counts['skipped']} already "
          f"done, {counts['failed']} failed (backend={args.judge}, fps={args.fps})")
    return counts["failed"] == 0


def cmd_report(args: argparse.Namespace, run_id: str, cases: list[Case]) -> bool:
    run_dir = RUNS_ROOT / run_id
    metrics_path, md_path = report_mod.write_report(run_dir, cases, meta={
        "run_id": run_id,
        "judge": args.judge,
        "renderer": args.renderer,
        "prompt_mode": args.prompt_mode,
        "ref2va": args.ref2va,
        "seed": args.seed,
        "fps": args.fps,
    })
    summary = aggregate(report_mod.load_rows(run_dir, cases), cases)
    print(f"[report] {metrics_path}")
    print(f"[report] {md_path}")
    print(f"[report] clips={summary['clips']} "
          f"EndingFollow={summary['EndingFollow']} CounterAbsent={summary['CounterAbsent']} "
          f"StrictFollow={summary['StrictFollow']} PairStrict={summary['PairStrict']}")
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    run_id = _resolve_run_id(args)
    wanted = ("headless", "render", "judge", "report") if args.stage == "all" \
        else (args.stage,)
    cases = _load_selected(args)
    if not any(stage == "report" for stage in wanted):
        _write_manifest(run_id, args, cases)

    for stage in wanted:
        if stage == "headless":
            if not cmd_headless(args, run_id, cases):
                print("[headless] 数据自洽检查未通过,中止后续阶段")
                return 1
        elif stage == "render":
            if not cmd_render(args, run_id, cases):
                return 1
        elif stage == "judge":
            if not cmd_judge(args, run_id, cases):
                return 1
        elif stage == "report":
            cmd_report(args, run_id, cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
