"""判定专用 LLM 机群:H3 退役后,8 卡各起一个单卡 Qwen VL worker 并行判定。

约定(2026-09-18 起):评测判定阶段不再与 H3 共存——渲染完成后先 stop H3
释放 8 卡,再每卡起一个 worker(CUDA_VISIBLE_DEVICES 锁单卡、per-gpu-mem
给足以免磁盘 offload),N 路并行把判定耗时除以 N。与游戏共存的常驻服务
(engine.llm.service 的跨卡分片实例)也必须先停,否则占着显存。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from engine.llm.client import LLMClient
from engine.llm.service import DEFAULT_MODEL, DEFAULT_PYTHON, WORKER_SCRIPT
from engine.renderer.minimax_h3 import h3_environment

FLEET_SOCKET_DIR = Path(tempfile.gettempdir())


def fleet_socket(index: int) -> Path:
    return FLEET_SOCKET_DIR / f"casebench-llm-{index}.sock"


def _kill_pattern(socket_name: str) -> None:
    subprocess.run(
        ["pkill", "-f", f"engine/llm/worker.py.*{socket_name}"],
        capture_output=True, check=False,
    )


def retire_services() -> None:
    """判定前清场:停掉 H3 与跨卡分片的共享 LLM(顺序安全:H3 先停)。

    ref2va 常驻实例(--ref2va 渲染用,监听 box2clay socket)同样占满
    8 卡各 ~65GB,且不在 engine.renderer.service 的默认 socket 上,上面
    的 stop 管不到——必须显式停,否则 Qwen worker 每卡只剩 ~15GB 起不来。
    """
    for module in ("engine.renderer.service", "engine.llm.service"):
        subprocess.run([sys.executable, "-m", module, "stop"],
                       capture_output=True, text=True, timeout=600, check=False)
    from evaluation.casebench.render import ref2va_socket_path

    subprocess.run(
        [sys.executable, "-m", "engine.renderer.service", "stop",
         "--socket", str(ref2va_socket_path())],
        capture_output=True, text=True, timeout=600, check=False)


def start_worker(index: int, *, per_gpu_mem: str, model: str | None = None,
                 python: str | None = None) -> Path:
    """在第 index 张卡上起一个单卡 worker,返回它的 socket。

    model/python 缺省取环境变量 AGENTIC_GAME_QWEN_MODEL / AGENTIC_GAME_QWEN_PYTHON
    (迁移到其他机器时用环境变量指向本机路径),再缺省用本仓库默认值。
    """
    model = model or os.environ.get("AGENTIC_GAME_QWEN_MODEL", DEFAULT_MODEL)
    python = python or os.environ.get("AGENTIC_GAME_QWEN_PYTHON", DEFAULT_PYTHON)
    socket_path = fleet_socket(index)
    socket_path.unlink(missing_ok=True)
    log_file = socket_path.with_suffix(socket_path.suffix + ".log")
    env = h3_environment()
    env["CUDA_VISIBLE_DEVICES"] = str(index)
    command = [python, str(WORKER_SCRIPT), "--model", model,
               "--socket", str(socket_path), "--per-gpu-mem", per_gpu_mem]
    with open(log_file, "ab") as log:
        subprocess.Popen(command, cwd="/", env=env, stdout=log, stderr=log,
                         start_new_session=True)
    return socket_path


def wait_ready(sockets: list[Path], timeout_s: float = 1800.0) -> None:
    """等待全部 worker 就绪;任一超时抛 TimeoutError。"""
    deadline = time.monotonic() + timeout_s
    clients = [LLMClient(socket_path=socket) for socket in sockets]
    pending = list(range(len(clients)))
    while pending:
        for index in list(pending):
            if clients[index].ping(timeout_s=5.0) is not None:
                pending.remove(index)
        if not pending:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"fleet workers not ready: {pending}")
        time.sleep(3.0)


def ensure_fleet(n: int, *, per_gpu_mem: str = "60GiB",
                 model: str | None = None) -> list[Path]:
    """清场 + 起 N 个单卡 worker + 等就绪;已在位的 worker 直接复用。"""
    sockets = [fleet_socket(i) for i in range(n)]
    running = [LLMClient(socket_path=s).ping(timeout_s=5.0) is not None for s in sockets]
    if not all(running):
        retire_services()
        for index in range(n):
            if not running[index]:
                _kill_pattern(sockets[index].name)
                start_worker(index, per_gpu_mem=per_gpu_mem, model=model)
        wait_ready(sockets)
    return sockets


def retire_fleet(n: int) -> None:
    """停掉机群释放 8 卡(H3 不自动重启;重渲染需显式 start)。"""
    for index in range(n):
        socket_path = fleet_socket(index)
        client = LLMClient(socket_path=socket_path)
        if client.ping(timeout_s=5.0) is not None:
            try:
                client._round_trip({"cmd": "shutdown"}, timeout_s=15.0)  # noqa: SLF001
            except Exception:  # noqa: BLE001 - 半死 worker 走强杀
                pass
        for _ in range(10):
            if client.ping(timeout_s=3.0) is None:
                break
            time.sleep(1.0)
        else:
            _kill_pattern(socket_path.name)
        socket_path.unlink(missing_ok=True)


def fleet_status(n: int) -> list[bool]:
    return [LLMClient(socket_path=fleet_socket(i)).ping(timeout_s=5.0) is not None
            for i in range(n)]


__all__ = ["ensure_fleet", "fleet_socket", "fleet_status", "retire_fleet",
           "retire_services", "start_worker"]
