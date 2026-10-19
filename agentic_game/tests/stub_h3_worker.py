#!/usr/bin/env python3
"""协议兼容的假 h3 worker:验证 H3SessionRenderer 的会话管理,不需要 torch。

用法: python3 stub_h3_worker.py <mode>
mode:
  ok    - 正常应答(用 ffmpeg 合成真实 mp4,走真实的最后一帧提取路径)
  die   - 收到请求后立即退出(模拟 worker 崩溃)
  slow  - 收到请求后挂起(模拟生成超时)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def make_video(output: Path, first_frame: str | None) -> None:
    """合成一段真实 mp4;若请求带了首帧,另存一份副本供测试断言透传。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    if first_frame:
        shutil.copyfile(first_frame, output.with_suffix(output.suffix + ".first"))
    if shutil.which("ffmpeg") is None:
        output.write_bytes(b"not a real video")
        return
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.5",
         "-r", "24", "-pix_fmt", "yuv420p", str(output)],
        check=True, capture_output=True,
    )


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
    emit({"event": "ready", "world_size": 1})
    while True:
        line = sys.stdin.readline()
        if not line:  # EOF == close
            break
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(request, dict) or request.get("cmd") == "close":
            break
        if mode == "die":
            sys.exit(9)
        if mode == "slow":
            import time
            time.sleep(60)

        output = Path(request["output"])
        make_video(output, request.get("first_frame"))
        # 混入引擎风格的杂散日志行:主进程必须能忽略它
        sys.stdout.write("[h3a800.lazy] video_vae moved to cuda:0 (noise)\n")
        sys.stdout.flush()
        emit({
            "event": "result",
            "ok": True,
            "turn_id": request.get("turn_id", ""),
            "video": str(output),
            "elapsed_s": 0.01,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
