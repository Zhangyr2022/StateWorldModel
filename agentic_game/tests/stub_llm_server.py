#!/usr/bin/env python3
"""协议兼容的假 LLM 服务:验证 LLMClient,不需要 transformers。

用法: python3 stub_llm_server.py <socket_path> [mode]
mode:
  ok       - 固定 JSON 回应(ping/shutdown 正常)
  empty    - 回应空文本
  slow     - 挂起(模拟超时)
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

REPLY = '{"reply": "真的吗?人家有点开心。", "affection_delta": 3, "mood": "shy"}'


def handle(request: dict, mode: str) -> dict | None:
    if request.get("cmd") == "ping":
        return {"event": "pong", "calls": 0, "device": "cpu-stub"}
    if request.get("cmd") == "shutdown":
        return {"event": "bye", "shutdown": True}
    if not request.get("messages"):
        return None
    if mode == "slow":
        time.sleep(60)
    if mode == "empty":
        return {"event": "chat", "ok": True, "text": "  ", "elapsed_s": 0.0}
    return {"event": "chat", "ok": True, "text": REPLY, "elapsed_s": 0.01}


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
