"""控制视频平滑度自检:逐帧下采样后算相邻帧平均绝对差,报告突刺(跳变)。

用法: python check_jumps.py a.mp4 b.mp4 ...(需 numpy 与 ffprobe)
突刺判据:某帧差 > 中位数 × 6 + 2(绝对底线),列出帧号。相机匀速运动下差分应近似平稳。
"""
import subprocess
import sys

import numpy as np


def frame_diffs(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=160:90",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160).astype(np.float32)
    return np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))


for p in sys.argv[1:]:
    d = frame_diffs(p)
    med = float(np.median(d))
    thr = med * 6 + 2.0
    bad = [int(i) + 1 for i, v in enumerate(d) if v > thr]
    print(f"{p}: median_diff={med:.2f} max={float(d.max()):.2f} thr={thr:.2f} "
          f"jumps={bad if bad else '无'}")
