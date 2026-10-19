"""Wan2.2-I2V-A14B 常驻批量 i2v: 832x480, 81帧@16fps(≈5s), 40步 unipc, seed=0。
单卡;逐条断点续跑;用法: python wan22_batch.py [--only CASE-SIDEMODE]
环境变量: CODEBASE_ROOT(含 wan2.2/ 代码), MODELS_ROOT(含 Wan-AI/Wan2.2-I2V-A14B/ 权重),
见 baseline/env.sh。"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts_common import load_manifest, done  # noqa: E402

CODEBASE = Path(os.environ.get("CODEBASE_ROOT", "")) / "wan2.2"
MODELS = Path(os.environ.get("MODELS_ROOT", "")) / "Wan-AI" / "Wan2.2-I2V-A14B"
if not CODEBASE.is_dir() or not MODELS.is_dir():
    raise SystemExit(
        "Set CODEBASE_ROOT (containing wan2.2/ code) and MODELS_ROOT "
        "(containing Wan-AI/Wan2.2-I2V-A14B/ weights) — see baseline/env.sh")
sys.path.insert(0, str(CODEBASE))
from PIL import Image  # noqa: E402
import wan  # noqa: E402
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402

MODEL = "wan22"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑匹配的条目(子串, 如 B01-A.raw)")
    args = ap.parse_args()

    entries = load_manifest(MODEL)
    if args.only:
        entries = [e for e in entries if args.only in f"{e['case']}-{e['side']}.{e['mode']}"]

    cfg = WAN_CONFIGS["i2v-A14B"]
    t0 = time.time()
    engine = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(MODELS),
        device_id=0, rank=0,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    print(f"[wan22] model loaded in {time.time()-t0:.0f}s", flush=True)

    todo = [e for e in entries if not done(e)]
    print(f"[wan22] total={len(entries)} todo={len(todo)}", flush=True)
    t_all = time.time()
    for i, e in enumerate(todo, 1):
        Path(e["out"]).parent.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        video = engine.generate(
            e["prompt"],
            Image.open(e["image"]).convert("RGB"),
            max_area=MAX_AREA_CONFIGS["832*480"],
            frame_num=81,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=40,
            guide_scale=(3.5, 3.5),
            seed=0,
            offload_model=True,
        )
        save_video(tensor=video[None], save_file=e["out"], fps=cfg.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        el = time.time() - t1
        print(f"[wan22] {i}/{len(todo)} {e['case']}-{e['side']}.{e['mode']} "
              f"{el:.0f}s avg={(time.time()-t_all)/i:.0f}s eta={(time.time()-t_all)/i*(len(todo)-i)/3600:.1f}h",
              flush=True)
    print("[wan22] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
