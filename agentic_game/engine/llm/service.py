#!/usr/bin/env python3
"""LLM 常驻服务管理:把 Qwen 分片驻留在全部 GPU 上,随时调用。

用法:
    python3 -m engine.llm.service start     # 装载模型并驻留
    python3 -m engine.llm.service status
    python3 -m engine.llm.service stop
    python3 -m engine.llm.service restart

与 Minimax-H3 服务(engine.renderer.service)共存:worker 以
device_map="auto" 把模型按层切分到 8 张卡,每卡只占几 GB,轮到谁
谁计算。环境变量可覆盖:AGENTIC_GAME_QWEN_MODEL / AGENTIC_GAME_LLM_SOCKET /
AGENTIC_GAME_QWEN_PYTHON。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from engine.config import conda_bin
from engine.llm.client import LLMClient
from engine.renderer.minimax_h3 import h3_environment  # 同一批必需环境变量

# 模型权重在 workspace 之外,属机器部署路径(AGENTIC_GAME_QWEN_MODEL 可覆盖)
DEFAULT_MODEL = str(Path.home() / "models" / "Qwen3.8-27")
DEFAULT_PYTHON = conda_bin("h3-a800", "python")
WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"
STOP_GRACE_S = 60.0


def default_llm_socket() -> Path:
    return Path(tempfile.gettempdir()) / "agentic-game-qwen27.sock"


class LLMServiceManager:
    def __init__(self, *, model: str, python: str, socket_path: Path,
                 per_gpu_mem: str = "8GiB") -> None:
        self.model = model
        self.python = python
        self.socket_path = Path(socket_path)
        self.per_gpu_mem = per_gpu_mem
        self.pid_file = self.socket_path.with_suffix(self.socket_path.suffix + ".pid")
        self.log_file = self.socket_path.with_suffix(self.socket_path.suffix + ".log")

    def _client(self) -> LLMClient:
        return LLMClient(socket_path=self.socket_path)

    def status(self) -> dict:
        pong = self._client().ping(timeout_s=5.0)
        if pong is None:
            return {"running": False, "socket": str(self.socket_path)}
        return {
            "running": True,
            "socket": str(self.socket_path),
            "device": pong.get("device"),
            "calls": pong.get("calls"),
            "log": str(self.log_file),
        }

    def wait_until_ready(self, timeout_s: float) -> dict:
        self._client().wait_until_ready(timeout_s=timeout_s)
        return {"running": True, "log": str(self.log_file)}

    def start(self, startup_timeout_s: float = 1800) -> dict:
        current = self.status()
        if current["running"]:
            return current
        self.socket_path.unlink(missing_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.python, str(WORKER_SCRIPT),
            "--model", self.model,
            "--socket", str(self.socket_path),
            "--per-gpu-mem", self.per_gpu_mem,
        ]
        env = h3_environment()  # DISABLE_ADDMM_CUDA_LT / HF_HUB_OFFLINE 同样是本机硬约束
        with open(self.log_file, "ab") as log:
            subprocess.Popen(
                command, cwd="/", env=env, stdout=log, stderr=log, start_new_session=True
            )
        return self.wait_until_ready(startup_timeout_s)

    def stop(self, grace_s: float = STOP_GRACE_S) -> dict:
        graceful = False
        if self._client().ping(timeout_s=5.0) is not None:
            try:
                self._client()._round_trip({"cmd": "shutdown"}, timeout_s=15.0)  # noqa: SLF001
                graceful = True
            except Exception:  # noqa: BLE001 - 半死服务走强杀兜底
                pass
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if self._client().ping(timeout_s=3.0) is None:
                break
            time.sleep(1.0)
        else:
            self._kill()
        self.socket_path.unlink(missing_ok=True)
        self.pid_file.unlink(missing_ok=True)
        return {"stopped": True, "graceful": graceful}

    def _kill(self) -> None:
        subprocess.run(
            ["pkill", "-f", f"engine/llm/worker.py.*{self.socket_path.name}"],
            capture_output=True, check=False,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "restart"))
    parser.add_argument("--model", default=os.environ.get("AGENTIC_GAME_QWEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--python", default=os.environ.get("AGENTIC_GAME_QWEN_PYTHON", DEFAULT_PYTHON))
    parser.add_argument("--socket", type=Path,
                        default=Path(socket_env) if (socket_env := os.environ.get("AGENTIC_GAME_LLM_SOCKET", "")) else default_llm_socket())
    # 7GiB 是与 Minimax-H3 共存的上限:H3 每卡峰值 ~69GiB,8GiB 分片会让
    # GPU1-6 只剩 ~68.7GiB 可用,装载末尾差 ~500MiB 直接 OOM。
    parser.add_argument("--per-gpu-mem", default="7GiB")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    args = parser.parse_args(argv)

    manager = LLMServiceManager(model=args.model, python=args.python,
                                socket_path=args.socket, per_gpu_mem=args.per_gpu_mem)
    if args.command == "status":
        print(_format(manager.status()))
        return 0
    if args.command == "stop":
        print(_format(manager.stop(), prefix="stop -> "))
        return 0
    if args.command == "restart":
        manager.stop()
    print(f"starting LLM service ({args.model}) on all GPUs, weights are loading ...", flush=True)
    print(_format(manager.start(startup_timeout_s=args.startup_timeout)))
    return 0


def _format(status: dict, prefix: str = "") -> str:
    if "stopped" in status:  # stop() 的返回没有 running/socket 字段
        return f"{prefix}stopped (graceful={status.get('graceful')})"
    if not status.get("running"):
        return f"{prefix}not running (socket: {status.get('socket')})"
    parts = [f"{prefix}running"]
    if status.get("device"):
        parts.append(f"device={status['device']}")
    if status.get("calls") is not None:
        parts.append(f"calls={status['calls']}")
    parts.append(f"log={status.get('log')}")
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
