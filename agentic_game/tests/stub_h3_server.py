#!/usr/bin/env python3
"""协议兼容的假 H3 常驻服务:验证 H3RemoteRenderer,不需要 torch。

用法: python3 stub_h3_server.py <socket_path> [mode]
mode:
  ok    - 正常应答(ffmpeg 合成真实 mp4,走真实的最后一帧提取路径)
  die   - 收到渲染请求后立即退出
  slow  - 收到渲染请求后挂起
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def make_video(output: Path, first_frame: str | None) -> None:
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


def handle(request: dict, mode: str) -> dict | None:
    if request.get("cmd") == "ping":
        return {"event": "pong", "world_size": 8, "renders": 0}
    if request.get("cmd") == "shutdown":
        return {"event": "bye", "shutdown": True}
    if not request.get("prompt"):
        return None
    if mode == "die":
        os._exit(9)
    if mode == "slow":
        time.sleep(60)
    output = Path(request["output"])
    make_video(output, request.get("first_frame"))
    return {
        "event": "result",
        "ok": True,
        "turn_id": request.get("turn_id", ""),
        "video": str(output),
        "elapsed_s": 0.01,
    }


def main() -> int:
    socket_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "ok"
    Path(socket_path).unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    shutdown = False
    try:
        while not shutdown:
            conn, _ = server.accept()
            with conn:
                file = conn.makefile("r")
                while True:
                    line = file.readline()
                    if not line:
                        break
                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    response = handle(request, mode)
                    if response is None:
                        continue
                    conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                    if response.get("shutdown"):
                        shutdown = True
                        break
    finally:
        server.close()
        Path(socket_path).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
