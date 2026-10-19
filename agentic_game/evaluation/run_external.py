#!/usr/bin/env python3
"""外部模型视频结果评测:runs/<model>/{raw,raw_state}/<CLIP>.mp4 → 标准产物。

外部提交目录约定:<RUNS_ROOT>/<model>/raw/<CLIP>.mp4 与
raw_state/<CLIP>.mp4(各 100 个,B01–B50 × A/B),prompts/<CLIP>.<mode>.txt。
评测把每个 (model, mode) 物化成标准 run 布局(不复制视频,软链):
    <RUNS_ROOT>/<model>-raw/        或  <model>-raw-state/
    ├─ videos/<CLIP>.mp4   (symlink → 外部视频)
    ├─ frames/<CLIP>.png   (ffmpeg 抽末帧)
    ├─ prompts/<CLIP>.txt  (外部提供的实际 prompt,失败报告用)
    ├─ manifest.json       (model/mode/来源)
    └─ verdicts/ + metrics.json + report.md + FAILURES.md (判定产物)
然后走与内部评测完全相同的 8 路并行判定与报告链路。

用法:
    python3 -m evaluation.run_external --models astra,echo            # 判定+报告
    python3 -m evaluation.run_external --models astra --stage materialize
服务:判定自动用 8 路单卡 LLM 机群(--keep-fleet 跨 run 复用,结束统一退役);
不需要 H3(视频已由外部模型产出)。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from evaluation import run_cases
from evaluation.casebench import report as report_mod
from evaluation.casebench.dataset import iter_twins, load_cases
from evaluation.casebench.frames import extract_final_frame
from evaluation.casebench.render import final_frame_for, video_path_for

logger = logging.getLogger("evaluation.run_external")

RUNS_ROOT = run_cases.RUNS_ROOT
MODE_RUN_SUFFIX = {"raw": "raw", "raw_state": "raw-state"}


def materialize(model: str, mode: str, cases, *, force: bool = False) -> tuple[str, Path]:
    """外部视频 → 标准 run 布局(软链 + 末帧)。返回 (run_id, run_dir)。"""
    run_id = f"{model}-{MODE_RUN_SUFFIX[mode]}"
    run_dir = RUNS_ROOT / run_id
    source_dir = RUNS_ROOT / model / mode
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
    prompt_src = RUNS_ROOT / model / "prompts"
    if prompt_src.is_dir():
        for twin in iter_twins(cases):
            src = prompt_src / f"{twin.clip_id}.{mode}.txt"
            if src.is_file():
                dst = run_dir / "prompts" / f"{twin.clip_id}.txt"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.is_file():
                    shutil.copyfile(src, dst)
    manifest = run_dir / "manifest.json"
    if not manifest.is_file() or force:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "run_id": run_id, "model": model, "mode": mode,
            "source": str(source_dir), "external": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id, run_dir


def judge_namespace(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    """构造与 run_cases.cmd_judge/cmd_report 兼容的参数。"""
    return argparse.Namespace(
        judge=args.judge, llm_workers=args.llm_workers, keep_fleet=True,
        force=args.force, socket=args.socket, max_tokens=args.max_tokens,
        fps=args.fps, max_frames=args.max_frames,
        renderer="external", prompt_mode=mode, seed=0,
        config=None,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True,
                        help="逗号分隔的外部模型名(runs/ 下的目录名)")
    parser.add_argument("--stage", choices=("all", "materialize"), default="all")
    parser.add_argument("--judge", default="parallel",
                        choices=("heuristic", "resident", "parallel"))
    parser.add_argument("--llm-workers", type=int, default=8)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--socket", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评前 N 个案例(冒烟用;默认全部)")
    args = parser.parse_args(argv)

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    fleet_started = False
    try:
        for model in models:
            for mode in ("raw", "raw_state"):
                run_id, _run_dir = materialize(model, mode, cases, force=args.force)
                if args.stage == "materialize":
                    print(f"[materialize] {run_id} ready", flush=True)
                    continue
                if args.judge == "parallel" and not fleet_started:
                    from evaluation.casebench.fleet import ensure_fleet

                    print(f"[fleet] starting {args.llm_workers} workers ...", flush=True)
                    ensure_fleet(args.llm_workers)
                    fleet_started = True
                ns = judge_namespace(args, mode)
                if not run_cases.cmd_judge(ns, run_id, cases):
                    logger.warning("[%s] judge reported failures", run_id)
                run_cases.cmd_report(ns, run_id, cases)
    finally:
        if fleet_started and args.judge == "parallel" and args.stage == "all":
            from evaluation.casebench.fleet import retire_fleet

            retire_fleet(args.llm_workers)
            print("[fleet] retired", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
