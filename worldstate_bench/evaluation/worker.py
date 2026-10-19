#!/usr/bin/env python3
"""判定 worker:单卡进程裁一个 shard,采样帧 → VLM 判定 → verdicts/。

由 run_eval.py 按 --gpus 每卡spawn 一个本模块进程(与数据生产侧
render/review 的分片习惯一致);也可手动单独运行断点续判(按
verdicts/<CLIP>.json 存在性去重)。

用法(一般经 run_eval.py 调,手动示例):
    python -m evaluation.worker \
        --run-dir runs/wan22-raw --gpus 0 --shard 0 --nshards 1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation import report as report_mod
from evaluation.dataset import (filter_cases, iter_twins,
                                                 load_cases)
from evaluation.frames import (FrameExtractionError,
                                                judge_images)
from evaluation.judge import build_judge
from evaluation.report import final_frame_for, video_path_for

logger = logging.getLogger("evaluation.worker")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="evaluation.worker", description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, default=None)
    parser.add_argument("--case", action="append", default=[],
                        help="只判该案例(可重复;与 run_eval 的选择一致)")
    parser.add_argument("--limit", type=int, default=None,
                        help="只判前 N 个案例(与 run_eval 的选择一致)")
    parser.add_argument("--judge", default="local", choices=("local", "heuristic"))
    parser.add_argument("--model", default=None, help="判定 VLM 路径(默认取 config)")
    parser.add_argument("--gpus", default="0", help="本 worker 用的卡(CUDA 设备串)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    cases = load_cases(args.cases_root)
    if args.case:
        cases = filter_cases(cases, set(args.case))
    if args.limit:
        cases = cases[: args.limit]
    twins = iter_twins(cases)[args.shard::args.nshards]

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus)
    device = "cuda:0" if args.judge == "local" else "cpu"
    judge = build_judge(args.judge, model=args.model, device=device,
                        max_tokens=args.max_tokens)

    n_judged = n_skipped = n_failed = 0
    try:
        for twin in twins:
            verdict_file = report_mod.verdict_path(run_dir, twin.clip_id)
            if verdict_file.is_file() and not args.force:
                n_skipped += 1
                continue
            video_path = video_path_for(run_dir, twin)
            if not video_path.is_file() or video_path.stat().st_size == 0:
                n_failed += 1
                logger.warning("[%s] no clip at %s", twin.clip_id, video_path)
                continue
            try:
                images = judge_images(video_path,
                                      final_frame_for(run_dir, twin),
                                      run_dir / "frames" / twin.clip_id,
                                      fps=args.fps, max_frames=args.max_frames)
                verdict = judge.judge(twin, images)
            except FrameExtractionError as exc:
                n_failed += 1
                logger.warning("[%s] frame extraction: %s", twin.clip_id, exc)
                continue
            except Exception as exc:  # noqa: BLE001 - 单 clip 失败不拖垮整批
                n_failed += 1
                logger.warning("[%s] judge: %s", twin.clip_id, exc)
                continue
            report_mod.write_verdict(run_dir, twin, verdict,
                                     judge_name=args.judge,
                                     n_images=len(images), images=images,
                                     video_path=video_path)
            n_judged += 1
            print(f"[judge] {twin.clip_id}: outcome={verdict['outcome']!r} "
                  f"counter={verdict['counter']!r} "
                  f"(expected {twin.outcome_question['expected']!r}/no)", flush=True)
    finally:
        judge.close()
    print(f"[judge] shard {args.shard}/{args.nshards}: {n_judged} judged, "
          f"{n_skipped} skipped, {n_failed} failed (backend={args.judge})")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
