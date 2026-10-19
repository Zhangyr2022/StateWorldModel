#!/usr/bin/env python3
"""H3 ref2va 常驻服务(8 卡引擎驻留,随时调试 prompt/种子/时长)。

结构照搬 agentic_game 的 h3_worker/service 三件套,但 task=ref2va + ref2va turbo
adapter,请求协议带 reference video/image(那边的服务只支持 i2v first_frame)。

用法:
    python3 ref2va_serve.py start     # torchrun 拉起 8 卡引擎并驻留(~2-6 min)
    python3 ref2va_serve.py status    # ping:{running, world_size, renders, log}
    python3 ref2va_serve.py stop      # 优雅退出(socket 通知 + 强杀兜底)
    python3 ref2va_serve.py generate --prompt "..." --video a.mp4 --image b.jpg \
        --out c.mp4 [--duration 5] [--seed 42]     # 单条生成(一次一个连接)

请求协议(每连接一行 JSON,断开不退出服务):
    {"cmd":"ping"} → {"event":"pong","world_size":N,"renders":M}
    {"cmd":"shutdown"} → {"event":"bye"} 后整组退出
    {"cmd":"generate","prompt":...,"video":...,"image":...(可省),
     "duration":5|10|15,"seed":...,"out":绝对路径}
        → {"event":"result","ok":true,"out":...,"elapsed_s":...}

SPMD 约定:所有 rank 同调用序列;请求由 rank0 经独立 gloo 组广播,避免与引擎
NCCL 集合通信交错死锁(h3-a800 DESIGN.md §5.3)。socket/日志放 /data
(/tmp 与根分区同盘常满,ENOSPC 无声杀后台——必守约束 9)。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from shutil import which

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]  # box2clay → spatial_proxy → workspace root
# h3-a800 仓库根:H3_REPO 可覆盖;缺省按 workspace 并排布局推导
H3 = Path(os.environ.get("H3_REPO", WORKSPACE / "h3-a800"))
# torchrun:H3_TORCHRUN → PATH(已激活 h3-a800 env) → CONDA_ENVS_ROOT
TORCHRUN = Path(os.environ.get("H3_TORCHRUN") or which("torchrun")
                or Path(os.environ.get("CONDA_ENVS_ROOT",
                                       Path.home() / "miniconda3" / "envs"))
                / "h3-a800" / "bin" / "torchrun")
SOCK = HERE / "results" / "h3_ref2va.sock"
LOG = SOCK.with_suffix(".log")
PID = SOCK.with_suffix(".pid")
# 引擎卡数(8×A800 部署约定);H3_NPROC 可覆盖
NPROC = int(os.environ.get("H3_NPROC", "8"))
START_TIMEOUT_S = 900.0

# ref2va 定稿参数(与 h3ref_batch --first-frame ref2va 一致)
ADAPTER = H3 / "checkpoints" / "Minimax-h3-Turbo" / "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"


# ---------------------------------------------------------------- worker 端

def build_engine():
    """所有 rank 各自装载引擎(cwd=h3-a800,离线 HF,DISABLE_ADDMM_CUDA_LT)。"""
    os.chdir(H3)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
    sys.path.insert(0, str(H3))
    from h3_runtime.engine import MiniMaxH3Inference

    return MiniMaxH3Inference(
        "./checkpoints/MiniMax-H3", str(ADAPTER),
        attention_backend="dense", task="ref2va", width=864, height=480)


class SocketChannel:
    """rank0 的 Unix socket 服务通道:一次一个连接,断开不退出。"""

    def __init__(self, socket_path: Path):
        self.path = socket_path
        self.world_size = 1
        self._server = None
        self._conn = None

    def open(self, world_size: int) -> None:
        self.world_size = world_size
        self.path.unlink(missing_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.path))
        self._server.listen(4)
        _log({"event": "listening", "socket": str(self.path), "world_size": world_size})

    def fetch(self) -> dict | None:
        while True:
            self._accept()
            try:
                buf = b""
                while b"\n" not in buf:
                    chunk = self._conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
            except OSError:
                self._drop()
                continue
            if b"\n" not in buf:
                self._drop()
                continue
            req = json.loads(buf.split(b"\n", 1)[0].decode("utf-8", errors="replace"))
            if req.get("cmd") == "ping":
                self._send({"event": "pong", "world_size": self.world_size,
                            "renders": _render_count()})
                self._drop()
                continue
            if req.get("cmd") == "shutdown":
                self._send({"event": "bye"})
                self._drop()
                return None
            return req

    def deliver(self, result: dict) -> None:
        self._send(result)  # 客户端断开时失败可忽略
        self._drop()

    def close(self) -> None:
        if self._server:
            self._server.close()
        self.path.unlink(missing_ok=True)

    def _accept(self) -> None:
        if self._conn is None:
            self._conn, _ = self._server.accept()

    def _drop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

    def _send(self, payload: dict) -> None:
        if self._conn is None:
            return
        try:
            self._conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError:
            pass


_renders = {"n": 0}


def _render_count() -> int:
    return _renders["n"]


def _log(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_generation(engine, req: dict, rank: int) -> dict:
    """广播后所有 rank 执行;rank0 存盘。"""
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

    prompt = req["prompt"]
    video = str(Path(req["video"]).resolve())
    image = str(Path(req["image"]).resolve()) if req.get("image") else None
    out = Path(req["out"])
    if rank == 0 and not Path(video).exists():
        return {"event": "result", "ok": False, "error": f"video 不存在: {video}"}
    duration = int(req.get("duration", 5))
    seed = int(req.get("seed", 0))
    refs = []
    if image:
        refs.append(MiniMaxH3Reference(image=image))
    refs.append(MiniMaxH3Reference(video=video))
    t0 = time.time()
    media = engine.generate(prompt, duration=duration, seed=seed, references=refs)
    if rank == 0:
        if media is None:
            return {"event": "result", "ok": False, "error": "rank0 未拿到 media"}
        out.parent.mkdir(parents=True, exist_ok=True)
        media.save(str(out))
        _renders["n"] += 1
        return {"event": "result", "ok": True, "out": str(out),
                "elapsed_s": round(time.time() - t0, 1)}
    return {"event": "result", "ok": True, "rank": rank}


def workerServe():
    """SPMD 入口:torchrun --ref2va-worker --socket ..."""
    import torch.distributed as dist
    from datetime import timedelta

    engine = build_engine()
    channel = SocketChannel(SOCK)
    if engine.rank == 0:
        channel.open(engine.world_size)
        PID.write_text(str(os.getpid()))
    # 独立 gloo 组做请求广播(不与引擎 NCCL 交错)
    request_group = dist.new_group(backend="gloo", timeout=timedelta(days=365))
    while True:
        payload = [channel.fetch() if engine.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0, group=request_group)
        req = payload[0]
        if req is None:  # shutdown / 通道关闭
            break
        try:
            result = run_generation(engine, req, engine.rank)
        except Exception as exc:  # 生成失败不退出服务
            result = {"event": "result", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if engine.rank == 0:
            channel.deliver(result)
    channel.close()
    engine.close()
    if engine.rank == 0:
        PID.unlink(missing_ok=True)


# ---------------------------------------------------------------- 管理端

def _ping(timeout_s: float = 5.0) -> dict | None:
    if not SOCK.exists():
        return None
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(timeout_s)
        conn.connect(str(SOCK))
        conn.sendall(b'{"cmd":"ping"}\n')
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        conn.close()
        return json.loads(buf.split(b"\n", 1)[0])
    except (OSError, json.JSONDecodeError):
        return None


def cmd_status() -> int:
    pong = _ping()
    if pong is None:
        print(f"ref2va 服务未运行 (socket={SOCK})")
        return 1
    print(json.dumps({"running": True, "socket": str(SOCK), "log": str(LOG), **pong},
                     ensure_ascii=False))
    return 0


def cmd_start() -> int:
    pong = _ping()
    if pong is not None:
        print(f"已在运行: {json.dumps(pong, ensure_ascii=False)}")
        return 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.touch()
    with open(LOG, "ab") as log:
        subprocess.Popen(  # noqa: S603 - 固定命令行
            [str(TORCHRUN), "--standalone", f"--nproc_per_node={NPROC}",
             str(Path(__file__).resolve()), "--ref2va-worker"],
            cwd=str(H3), stdout=log, stderr=log, start_new_session=True,
            env={**os.environ,
                 "HF_HUB_OFFLINE": "1", "DISABLE_ADDMM_CUDA_LT": "1"})
    t0 = time.time()
    while time.time() - t0 < START_TIMEOUT_S:
        pong = _ping()
        if pong is not None:
            print(f"就绪({time.time() - t0:.0f}s): {json.dumps(pong, ensure_ascii=False)}")
            return 0
        time.sleep(5)
    print(f"启动超时({START_TIMEOUT_S}s),看日志: {LOG}")
    return 1


def cmd_stop() -> int:
    pong = _ping()
    if pong is None:
        print("未在运行")
        return 0
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(15.0)
        conn.connect(str(SOCK))
        conn.sendall(b'{"cmd":"shutdown"}\n')
        conn.close()
    except OSError:
        pass
    time.sleep(5)
    if _ping() is None:
        print("已停止")
        return 0
    if PID.exists():  # 强杀兜底
        subprocess.run(["pkill", "-f", "ref2va_serve.py --ref2va-worker"], check=False)
        PID.unlink(missing_ok=True)
    print("已强杀")
    return 0


def cmd_generate(args) -> int:
    pong = _ping()
    if pong is None:
        print("ref2va 服务未运行,先: python3 ref2va_serve.py start", file=sys.stderr)
        return 1
    req = {"cmd": "generate", "prompt": args.prompt,
           "video": str(Path(args.video).resolve()),
           "duration": args.duration, "seed": args.seed,
           "out": str(Path(args.out).resolve())}
    if args.image:
        req["image"] = str(Path(args.image).resolve())
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(args.timeout)
    conn.connect(str(SOCK))
    conn.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            raise ConnectionError("service closed")
        buf += chunk
    conn.close()
    result = json.loads(buf.split(b"\n", 1)[0])
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref2va-worker", action="store_true",
                    help=argparse.SUPPRESS)  # 仅 torchrun SPMD 内部使用
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("start")
    sub.add_parser("status")
    sub.add_parser("stop")
    g = sub.add_parser("generate")
    g.add_argument("--prompt", required=True)
    g.add_argument("--video", required=True, help="控制视频(ref2va reference)")
    g.add_argument("--image", default=None, help="首帧图(reference 1,可省)")
    g.add_argument("--out", required=True)
    g.add_argument("--duration", type=int, default=5, choices=(5, 10, 15))
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    if args.ref2va_worker:
        workerServe()
        return 0
    if args.cmd == "start":
        return cmd_start()
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "stop":
        return cmd_stop()
    if args.cmd == "generate":
        return cmd_generate(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
