#!/usr/bin/env python3
"""H3 常驻服务管理:把本机全部可见 GPU 的引擎长期驻留在后台,随时调用。

用法:
    python3 -m engine.renderer.service start     # 装载引擎并驻留(约 6 分钟)
    python3 -m engine.renderer.service status    # 查看:是否在跑、已渲染次数
    python3 -m engine.renderer.service stop      # 优雅退出
    python3 -m engine.renderer.service restart

服务 = torchrun 启动的 h3_worker.py --serve(Unix socket 应答)。
游戏进程用 H3RemoteRenderer 直连 socket 渲染,不再各自装载引擎。
socket/pid/log 文件默认放在系统临时目录,名字含卡数与分辨率。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from engine.config import load_config
from engine.renderer.minimax_h3 import DEFAULT_H3_REPO, DEFAULT_TORCHRUN, h3_environment
from engine.renderer.remote import H3RemoteRenderer

WORKER_SCRIPT = Path(__file__).resolve().parent / "h3_worker.py"

STOP_GRACE_S = 120.0

# ref2va 定稿参数(与 box2clay ref2va_serve 一致):MiniMax-H3 底座 +
# ref2va turbo 4-step adapter;ref2va 仅支持 dense attention。
REF2VA_ADAPTER = (
    "checkpoints/Minimax-h3-Turbo/"
    "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
)


def default_socket_path(nproc: int, resolution: str, task: str = "t2v") -> Path:
    import tempfile

    suffix = "-ref2va" if task == "ref2va" else ""
    return Path(tempfile.gettempdir()) / (
        f"agentic-game-h3-{nproc}gpu-{resolution}{suffix}.sock"
    )


def _sidecar(socket_path: Path, suffix: str) -> Path:
    return socket_path.with_suffix(socket_path.suffix + suffix)


class ServiceManager:
    def __init__(self, *, h3_repo: Path, torchrun: Path, nproc: int, resolution: str,
                 attention_backend: str, socket_path: Path, task: str = "t2v") -> None:
        self.h3_repo = Path(h3_repo)
        self.torchrun = Path(torchrun)
        self.nproc = nproc
        self.resolution = resolution
        self.attention_backend = attention_backend
        self.task = task
        self.socket_path = Path(socket_path)
        self.session_tag = f"agentic-game-h3-serve-{self.socket_path.name}"
        self.pid_file = _sidecar(self.socket_path, ".pid")
        self.log_file = _sidecar(self.socket_path, ".log")

    def _client(self) -> H3RemoteRenderer:
        return H3RemoteRenderer(output_dir=self.h3_repo, socket_path=self.socket_path)

    # -- 状态查询 ------------------------------------------------------------

    def status(self) -> dict:
        pong = self._client().ping(timeout_s=5.0)
        if pong is None:
            return {"running": False, "socket": str(self.socket_path)}
        return {
            "running": True,
            "socket": str(self.socket_path),
            "world_size": pong.get("world_size"),
            "renders": pong.get("renders"),
            "pid": self._read_pid(),
            "log": str(self.log_file),
        }

    def wait_until_ready(self, timeout_s: float) -> dict:
        pong = self._client().wait_until_ready(timeout_s=timeout_s)
        return {"running": True, "world_size": pong.get("world_size"),
                "pid": self._read_pid(), "log": str(self.log_file)}

    # -- 生命周期 ------------------------------------------------------------

    def start(self, startup_timeout_s: float) -> dict:
        current = self.status()
        if current["running"]:
            return current
        if self.socket_path.exists():
            # 二次确认:status() 的 ping 可能只是瞬时排队/拒连误报。
            # socket 文件还在时绝不盲目 unlink——unlink 掉活实例的 socket
            # 会把它孤儿化(进程活着但永远不可达),再叠一个新实例占卡。
            time.sleep(3.0)
            if self._client().ping(timeout_s=10.0) is not None:
                return self.status()
        self.socket_path.unlink(missing_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.task == "ref2va":
            adapter = str(self.h3_repo / REF2VA_ADAPTER)
            attention_backend = "dense"
        else:
            adapter = str(
                self.h3_repo / "checkpoints" / "FastH3-4-step-Preview-v1-LoRA" /
                ("vsa-datafree" if self.attention_backend == "vsa" else "dense-datafree") /
                "adapter_model.safetensors"
            )
            attention_backend = self.attention_backend
        command = [
            str(self.torchrun), "--standalone", f"--nproc_per_node={self.nproc}",
            str(WORKER_SCRIPT),
            "--h3-repo", str(self.h3_repo),
            "--model", str(self.h3_repo / "checkpoints" / "MiniMax-H3"),
            "--adapter", adapter,
            "--attention-backend", attention_backend,
            "--task", self.task,
            "--res", self.resolution,
            "--session-tag", self.session_tag,
            "--serve", "--socket", str(self.socket_path),
        ]
        with open(self.log_file, "ab") as log:
            subprocess.Popen(  # noqa: S603 - 固定命令行,非用户输入
                command,
                cwd=str(self.h3_repo),
                env=h3_environment(),
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        return self.wait_until_ready(startup_timeout_s)

    def stop(self, grace_s: float = STOP_GRACE_S) -> dict:
        stopped_via_socket = False
        pong = self._client().ping(timeout_s=5.0)
        if pong is not None:
            try:
                self._client()._round_trip({"cmd": "shutdown"}, timeout_s=15.0)  # noqa: SLF001
                stopped_via_socket = True
            except Exception:  # noqa: BLE001 - 服务半死时走强杀兜底
                pass
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if self._client().ping(timeout_s=3.0) is None:
                break
            time.sleep(2.0)
        else:
            self._kill()
        self._kill_if_pid_stale()
        self.socket_path.unlink(missing_ok=True)
        self.pid_file.unlink(missing_ok=True)
        return {"stopped": True, "graceful": stopped_via_socket}

    def _read_pid(self) -> int | None:
        try:
            return int(self.pid_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def _write_pid(self) -> None:
        pid = self._read_pid()
        if pid is not None:
            return
        # torchrun 自身进程:启动时由 start 写入;这里兜底查 tag
        try:
            out = subprocess.run(
                ["pgrep", "-f", self.session_tag], capture_output=True, text=True, check=False
            )
            pids = [int(p) for p in out.stdout.split()]
            if pids:
                self.pid_file.write_text(str(min(pids)))
        except (OSError, ValueError):
            pass

    def _kill_if_pid_stale(self) -> None:
        pid = self._read_pid()
        if pid is None:
            return
        try:
            proc = subprocess.run(["kill", "-0", str(pid)], capture_output=True, check=False)
            if proc.returncode == 0:
                subprocess.run(["kill", str(pid)], capture_output=True, check=False)
        except OSError:
            pass

    def _kill(self) -> None:
        subprocess.run(["pkill", "-f", self.session_tag], capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "restart"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--res", default=None, choices=("480p", "768p"))
    parser.add_argument("--task", default="t2v", choices=("t2v", "ref2va"),
                        help="t2v=常驻 i2v/t2v 引擎(默认);ref2va=参考引导"
                             "(控制视频+首图;换 ref2va turbo adapter,socket 名"
                             "带 -ref2va 后缀,与 t2v 服务互斥占用 GPU)")
    parser.add_argument("--socket", type=Path, default=None)
    parser.add_argument("--startup-timeout", type=float, default=None,
                        help="等待引擎装载完成的秒数(默认 3600)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    nproc = args.nproc or cfg.renderer.nproc
    resolution = args.res or cfg.renderer.resolution
    socket_path = args.socket or (
        Path(cfg.renderer.socket_path)
        if (cfg.renderer.socket_path and args.task == "t2v")
        else default_socket_path(nproc, resolution, args.task)
    )
    manager = ServiceManager(
        h3_repo=Path(cfg.renderer.h3_repo),
        torchrun=Path(cfg.renderer.torchrun),
        nproc=nproc,
        resolution=resolution,
        attention_backend=cfg.renderer.attention_backend,
        socket_path=socket_path,
        task=args.task,
    )

    if args.command == "status":
        print(_format_status(manager.status()))
        return 0
    if args.command == "stop":
        print(_format_status(manager.stop(), prefix="stop -> "))
        return 0
    if args.command == "restart":
        manager.stop()
    startup_timeout = args.startup_timeout or float(cfg.renderer.startup_timeout_s)
    print(f"starting H3 service on {nproc} GPUs ({resolution}), engine is loading ...", flush=True)
    status = manager.start(startup_timeout_s=startup_timeout)
    if manager._read_pid() is None:  # noqa: SLF001
        manager._write_pid()  # noqa: SLF001
    print(_format_status(status))
    return 0


def _format_status(status: dict, prefix: str = "") -> str:
    if not status.get("running"):
        return f"{prefix}not running (socket: {status.get('socket')})"
    parts = [f"{prefix}running", f"world_size={status.get('world_size')}"]
    if status.get("renders") is not None:
        parts.append(f"renders={status['renders']}")
    if status.get("pid"):
        parts.append(f"pid={status['pid']}")
    parts.append(f"log={status.get('log')}")
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
