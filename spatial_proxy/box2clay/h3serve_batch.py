#!/usr/bin/env python3
"""经 ref2va 常驻服务批量生成(逐条调 socket,断点续跑:已有产物跳过)。

用法:
    python3 h3serve_batch.py --manifest <含 prompt 三件套的目录> \
        [--subject-root results/subject_v2] [--duration 5] [--seed 42]
读 <manifest>/prompts_struct8.json(key=<det_dir>_<stem>)与 images_struct8.json
(视频绝对路径→case 原图),逐条 ref2va_serve.generate。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="含 prompts_struct8.json/images_struct8.json 的目录")
    ap.add_argument("--subject-root", default=str(HERE / "results" / "subject_v2"),
                    help="控制视频根目录(默认 subject_v2=新几何)")
    ap.add_argument("--duration", type=int, default=5, choices=(5, 10, 15))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=1200.0)
    args = ap.parse_args()
    root = Path(args.manifest)
    prompts = json.loads((root / "prompts_struct8.json").read_text())
    images = json.loads((root / "images_struct8.json").read_text())  # 视频绝对路径→原图

    vid2img = {str(Path(k).resolve()): v for k, v in images.items()}
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for key, prompt in prompts.items():
        det_dir, stem = key.rsplit("_walk_", 1)
        act = stem.replace("_", "+") if stem != "W" else "W"
        video = Path(args.subject_root) / det_dir / f"walk_{stem}.mp4"
        out = root / f"{key}.mp4"
        if out.exists() and out.stat().st_size > 0:
            n_skip += 1
            continue
        img = vid2img.get(str(video.resolve()), "")
        cmd = [PY, str(HERE / "ref2va_serve.py"), "generate",
               "--prompt", prompt, "--video", str(video),
               "--out", str(out), "--duration", str(args.duration),
               "--seed", str(args.seed), "--timeout", str(args.timeout)]
        if img:
            cmd += ["--image", img]
        proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        ok = proc.returncode == 0 and '"ok": true' in proc.stdout
        if ok:
            n_ok += 1
            print(f"[{n_ok + n_skip + n_fail}/{len(prompts)}] {key} ok "
                  f"({time.time() - t0:.0f}s 累计)", flush=True)
        else:
            n_fail += 1
            print(f"[FAIL] {key}: {proc.stdout.strip() or proc.stderr.strip()[:300]}",
                  flush=True)
    print(f"完成: ok={n_ok} skip={n_skip} fail={n_fail} 用时 {(time.time() - t0) / 60:.1f}min")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
