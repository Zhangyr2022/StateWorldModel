#!/usr/bin/env python3
"""评测编排:headless 自检 → 外部视频物化 → 按卡分片判定 → 指标报告。

评什么:baseline/ 产出的外部模型视频(runs/<model>/{raw,raw_state}/<CLIP>.mp4,
prompts/<CLIP>.<mode>.txt),每个 (model, mode) 物化成一个标准 run 目录:

    runs/<model>-raw/  或  runs/<model>-raw-state/
    ├─ videos/<CLIP>.mp4   (软链 → 外部视频)
    ├─ frames/<CLIP>.png   (ffmpeg 抽末帧)
    ├─ prompts/<CLIP>.txt  (推理时的实际 prompt,失败报告用)
    ├─ manifest.json
    └─ verdicts/ + metrics.json + report.md + FAILURES.md

判定按卡分片并行:--gpus 0,1,2,3 → 每卡 spawn 一个 worker 子进程
(kairos 环境解释器,加载 config.review.model),各裁 1/N 的 clip。
无 GPU 时 --judge heuristic 只打通数据流(结果不可作正式结论)。

用法(项目根目录):
    # 全链路:自检 → 物化 → 判定 → 报告(两个 prompt 模式都评)
    python3 -m evaluation.run_eval --models wan22 --gpus 0,1,2,3

    # 只物化 / 只判定 / 只出报告(断点续跑:verdicts 按存在性去重)
    python3 -m evaluation.run_eval --models wan22 --stage materialize
    python3 -m evaluation.run_eval --models wan22 --stage judge
    python3 -m evaluation.run_eval --models wan22 --stage report

    # 已物化的单个 run 直接判/报:
    python3 -m evaluation.run_eval --run-dir runs/wan22-raw --stage judge
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from worldstate_bench.config import load_config
from evaluation import report as report_mod
from evaluation.dataset import (Case, filter_cases,
                                                 iter_twins, load_cases)
from evaluation.frames import extract_final_frame
from evaluation.headless import run_headless, write_headless
from evaluation.metrics import aggregate
from evaluation.report import final_frame_for, video_path_for

logger = logging.getLogger("evaluation.run_eval")

ROOT = Path(__file__).resolve().parents[1]
MODE_RUN_SUFFIX = {"raw": "raw", "raw_state": "raw-state"}


def _runs_root(args: argparse.Namespace) -> Path:
    return args.runs_root if args.runs_root is not None else ROOT / "runs"


def _load_selected(args: argparse.Namespace) -> list[Case]:
    cases = load_cases(args.cases_root)
    if args.case:
        cases = filter_cases(cases, set(args.case))
    if args.limit:
        cases = cases[: args.limit]
    return cases


# -- 阶段 ---------------------------------------------------------------------


def cmd_headless(args: argparse.Namespace, run_dir: Path,
                 cases: list[Case]) -> bool:
    """数据自洽检查(零 GPU):结局声明交叉互指 + 规则重放,不过即中止。"""
    passed, checked = run_headless(cases)
    path = write_headless(run_dir, passed, checked)
    print(f"[headless] {passed}/{len(checked)} clips self-consistent -> {path}")
    for clip, problems in checked.items():
        for problem in problems:
            print(f"[headless]   {clip}: {problem}")
    return passed == len(checked)


def materialize(model: str, mode: str, cases: list[Case], *,
                runs_root: Path, force: bool = False) -> tuple[str, Path]:
    """外部视频 → 标准 run 布局(软链 + 末帧 + prompt 存档)。"""
    run_id = f"{model}-{MODE_RUN_SUFFIX[mode]}"
    run_dir = runs_root / run_id
    source_dir = runs_root / model / mode
    if not source_dir.is_dir():
        raise SystemExit(f"external source dir missing: {source_dir}")
    missing: list[str] = []
    for twin in iter_twins(cases):
        src = source_dir / f"{twin.clip_id}.mp4"
        if not src.is_file() or src.stat().st_size == 0:
            missing.append(twin.clip_id)
            continue
        dst = video_path_for(run_dir, twin)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if force and dst.is_symlink():
            dst.unlink()
        if not dst.exists() and not dst.is_symlink():
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copyfile(src, dst)
        frame = final_frame_for(run_dir, twin)
        if not frame.is_file():
            try:
                extract_final_frame(dst, frame)
            except Exception as exc:  # noqa: BLE001 - 末帧缺失只影响该 clip 判定输入
                logger.warning("[%s] final frame missing: %s", twin.clip_id, exc)
    if missing:
        logger.warning("[%s/%s] %d clips missing from submission: %s",
                       model, mode, len(missing), ", ".join(missing))
    prompt_src = runs_root / model / "prompts"
    if prompt_src.is_dir():
        for twin in iter_twins(cases):
            src = prompt_src / f"{twin.clip_id}.{mode}.txt"
            if src.is_file():
                dst = run_dir / "prompts" / f"{twin.clip_id}.txt"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.is_file():
                    shutil.copyfile(src, dst)
    manifest = run_dir / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "run_id": run_id, "model": model, "mode": mode,
        "source": str(source_dir), "external": True,
        "cases": [case.case_id for case in cases],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id, run_dir


def _selection_flags(args: argparse.Namespace) -> list[str]:
    """把 --case/--limit 选择透传给 worker 子进程(两边圈定范围必须一致)。"""
    flags: list[str] = []
    for case in args.case:
        flags += ["--case", case]
    if args.limit:
        flags += ["--limit", str(args.limit)]
    return flags


def cmd_judge(args: argparse.Namespace, run_dir: Path,
              cases: list[Case]) -> bool:
    """按卡分片:每张卡 spawn 一个 worker 子进程,等待全部完成。"""
    if args.judge == "heuristic":  # 无 GPU 干跑:进程内直接跑
        from evaluation.worker import main as worker_main

        return worker_main([
            "--run-dir", str(run_dir),
            *(["--cases-root", str(args.cases_root)] if args.cases_root else []),
            *_selection_flags(args),
            "--judge", "heuristic",
            "--fps", str(args.fps), "--max-frames", str(args.max_frames),
            *(["--force"] if args.force else []),
        ]) == 0

    cfg = load_config()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--judge local 需要 --gpus(如 --gpus 0,1,2,3)")
    python = args.worker_python or cfg.review_python
    if not Path(python).exists():
        raise SystemExit(f"config.yaml 的解释器不存在: {python}\n"
                         "(改 envs.review_python,或传 --worker-python)")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("HF_HUB_OFFLINE", "1")
    procs = []
    for shard, gpu in enumerate(gpus):
        cmd = [python, "-m", "evaluation.worker",
               "--run-dir", str(run_dir),
               *(["--cases-root", str(args.cases_root)] if args.cases_root else []),
               *_selection_flags(args),
               "--judge", "local",
               *(["--model", args.model] if args.model else []),
               "--gpus", gpu, "--shard", str(shard), "--nshards", str(len(gpus)),
               "--fps", str(args.fps), "--max-frames", str(args.max_frames),
               "--max-tokens", str(args.max_tokens),
               *(["--force"] if args.force else [])]
        log = run_dir / f"judge_shard{shard}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as lf:
            procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=lf,
                                          env=env))
    rc = 0
    for proc in procs:
        rc |= proc.wait()
    print(f"[judge] {len(procs)} shard worker(s) done rc={rc}")
    return rc == 0


def cmd_report(args: argparse.Namespace, run_dir: Path,
               cases: list[Case]) -> bool:
    metrics_path, md_path = report_mod.write_report(run_dir, cases, meta={
        "run_id": run_dir.name,
        "judge": args.judge,
        "prompt_mode": args.prompt_mode or "",
        "fps": args.fps,
        "max_frames": args.max_frames,
    })
    summary = aggregate(report_mod.load_rows(run_dir, cases), cases)
    print(f"[report] {metrics_path}")
    print(f"[report] {md_path}")
    print(f"[report] clips={summary['clips']} "
          f"ENDFOLLOW={summary['ENDFOLLOW']} "
          f"CNTABSENT={summary['CNTABSENT']} "
          f"STRICT={summary['STRICT']} "
          f"PAIRSTRICT={summary['PAIRSTRICT']}")
    return True


# -- 入口 ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--models", help="逗号分隔的外部模型名(runs/ 下目录名;"
                                        "对每个模型的 raw 与 raw_state 都评测)")
    target.add_argument("--run-dir", type=Path,
                        help="已物化的单个 run 目录(只做 judge/report)")
    parser.add_argument("--stage", choices=("all", "headless", "materialize",
                                            "judge", "report"), default="all")
    parser.add_argument("--cases-root", type=Path, default=None,
                        help="数据集根(默认 data/cases)")
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="run 产物根(默认 <项目>/runs)")
    parser.add_argument("--case", action="append", default=[],
                        help="只评该案例(可重复;默认全部)")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评前 N 个案例(冒烟用)")
    parser.add_argument("--judge", default="local", choices=("local", "heuristic"),
                        help="判定后端:local=transformers 直载判定 VLM;"
                             "heuristic=干跑(打通数据流,结果不可用)")
    parser.add_argument("--model", default=None,
                        help="判定 VLM 路径(默认 config.review.model)")
    parser.add_argument("--gpus", default="0",
                        help="local 后端的分片卡(逗号分隔,每卡一个 worker)")
    parser.add_argument("--worker-python", default=None,
                        help="worker 解释器(默认 config.envs.review_python)")
    parser.add_argument("--prompt-mode", choices=("raw", "raw_state"), default=None,
                        help="写入报告 meta 的 prompt 模式(--run-dir 时可用)")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="判定帧采样帧率(3fps ≈ 15 帧/5s 片段)")
    parser.add_argument("--max-frames", type=int, default=32,
                        help="送 VLM 的帧数上限(超出均匀降采样)")
    parser.add_argument("--max-tokens", type=int, default=768,
                        help="判定输出 token 上限")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有 verdicts/软链,强制重做")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    runs_root = _runs_root(args)
    cases = _load_selected(args)

    if args.run_dir is not None:  # 单 run 模式:judge / report
        if args.stage in ("all", "materialize", "headless"):
            raise SystemExit("--run-dir 模式只支持 --stage judge|report")
        if not args.case and not args.limit:  # 按 run 的 manifest 圈定范围
            manifest = args.run_dir / "manifest.json"
            if manifest.is_file():
                wanted = set(json.loads(
                    manifest.read_text(encoding="utf-8")).get("cases", []))
                if wanted:
                    cases = [c for c in cases if c.case_id in wanted]
        ok = True
        if args.stage in ("all", "judge"):
            ok = cmd_judge(args, args.run_dir, cases)
        if args.stage in ("all", "report"):
            cmd_report(args, args.run_dir, cases)
        return 0 if ok else 1

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    evaluated = 0
    for model in models:
        for mode in ("raw", "raw_state"):
            if not (runs_root / model / mode).is_dir():
                # 部分提交(如只生成了 raw_state)时跳过该模式,不硬退出
                logger.warning("[materialize] skip %s/%s: runs/%s/%s missing",
                               model, mode, model, mode)
                continue
            evaluated += 1
            run_id, run_dir = materialize(model, mode, cases,
                                          runs_root=runs_root, force=args.force)
            if args.stage == "materialize":
                print(f"[materialize] {run_id} ready", flush=True)
                continue
            if args.stage in ("all", "headless"):
                if not cmd_headless(args, run_dir, cases):
                    print("[headless] 数据自洽检查未通过,中止后续阶段")
                    return 1
            if args.stage == "headless":
                continue
            ns_args = argparse.Namespace(**{**vars(args), "prompt_mode": mode})
            ok = True
            if args.stage in ("all", "judge"):
                ok = cmd_judge(ns_args, run_dir, cases)
            if args.stage in ("all", "report"):
                cmd_report(ns_args, run_dir, cases)
            if not ok:
                return 1
    if not evaluated:
        if args.stage == "headless":
            # 没有提交的 clip 时,headless 退化为数据集自洽检查(与 USAGE.md §3 代码段等价)
            from evaluation.headless import run_headless

            passed, checked = run_headless(cases)
            print(f"[headless] no clips under {runs_root}; dataset self-check: "
                  f"{passed}/{len(checked)} clips self-consistent", flush=True)
            return 0 if passed == len(checked) else 1
        # 静默成功会掩盖"先放 clip"的前提:没有任何 (model, mode) 被评测就按失败退出
        print(f"[run_eval] nothing evaluated: no clips under "
              f"{runs_root}/<model>/{{raw,raw_state}}/ for --models {args.models}. "
              "Generate or copy the clips first (USAGE.md § 3).", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
