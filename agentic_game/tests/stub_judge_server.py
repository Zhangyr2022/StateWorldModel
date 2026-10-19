#!/usr/bin/env python3
"""协议兼容的假 VLM 判定服务:验证 ResidentQwenJudge,不需要 transformers。

用法: python3 stub_judge_server.py <socket_path> [mode]
mode:
  ok       - 固定 verdict JSON 回应(接受 images 字段,ping/shutdown 正常)
  badjson  - 回应缺必需字段(测试 parse_verdict 的严格校验)
  garbage  - 回应无 JSON 的纯文本
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

REPLY = (
    '{"effect_shown": {"verdict": true, "evidence": "frame 2: effect visible"}, '
    '"rejection_shown": {"verdict": false, "evidence": "frame 1: no refusal"}, '
    '"state_change": {"npc.mira.affinity": "90"}, '
    '"eta_fields": {"npc.mira.affinity": "correct"}, '
    '"hallucinated_change": {"verdict": false, "evidence": "frame 1: none"}, '
    '"continuity_ok": {"verdict": true, "evidence": "frame 1: same scene as Picture 1"}}'
)


def handle(request: dict, mode: str) -> dict | None:
    if request.get("cmd") == "ping":
        return {"event": "pong", "calls": 0, "device": "cpu-stub"}
    if request.get("cmd") == "shutdown":
        return {"event": "bye", "shutdown": True}
    if not request.get("messages"):
        return None
    # VL 通道:images 非空时服务端应注入图片块;stub 只确认收到路径。
    assert request.get("images") is None or isinstance(request["images"], list)
    if mode == "slow":
        time.sleep(60)
    if mode == "badjson":
        return {"event": "chat", "ok": True, "text": '{"effect_shown": {}}', "elapsed_s": 0.01}
    if mode == "garbage":
        return {"event": "chat", "ok": True, "text": "the clip looks fine to me",
                "elapsed_s": 0.01}
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
