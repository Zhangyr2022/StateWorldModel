#!/usr/bin/env python3
"""Minimax-H3 推理 worker(torchrun SPMD 入口)。

为什么存在:h3-a800 引擎装载 ~141GB 权重需要约 6 分钟。worker 装载一次
引擎,之后每回合只执行一次 engine.generate(5s 480p 约 6~7s)。它只 import
h3-a800 的稳定 Python API,不修改 h3-a800 的任何文件。

两种传输模式:

1. pipe(默认,stdin/stdout JSON Lines)——由 H3SessionRenderer 以
   torchrun 子进程方式启动,生命周期随渲染器。
2. --serve --socket PATH(Unix domain socket)——常驻服务模式:
   由 `python3 -m engine.renderer.service start` 拉起后长期驻留,
   任何进程随时连接调用(H3RemoteRenderer)。一次一个连接,
   请求行协议:
     {"cmd":"ping"}                       → {"event":"pong", ...}
     {"cmd":"shutdown"}                   → {"event":"bye"} 后整组退出
     {"prompt","seed","duration_s","first_frame","output","turn_id"}
                                          → {"event":"result","ok":...}
   连接断开只结束当前连接,worker 继续服务。

SPMD 约定:所有 rank 执行同样的调用序列;请求由 rank 0 经独立 gloo
进程组广播给全部 rank——独立组是为了避免与引擎的 NCCL 集合通信交错
(h3-a800 DESIGN.md §5.3:同组内集合通信交错会死锁)。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3-repo", required=True, help="h3-a800 repository root")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--attention-backend", default="dense", choices=("dense", "vsa"))
    parser.add_argument("--res", default="480p", choices=("480p", "768p"))
    parser.add_argument("--task", default="t2v", choices=("t2v", "ref2va"),
                        help="t2v=文生视频(i2v 经 image 参数);ref2va=参考引导"
                             "(控制视频/图片经 references,需 ref2va adapter)")
    parser.add_argument("--session-tag", default="", help="unique tag for cleanup matching")
    parser.add_argument("--serve", action="store_true",
                        help="常驻服务模式:监听 Unix socket 而非 stdin")
    parser.add_argument("--socket", default="", help="serve 模式的 Unix socket 路径")
    # parse_known_args:容忍启动命令里附加的标记参数
    args, _ = parser.parse_known_args()
    if args.serve and not args.socket:
        parser.error("--serve requires --socket")
    return args


def emit(payload: dict) -> None:
    """rank 0 向日志/主进程写一行 JSON。"""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# pong 标记:客户端据此区分本 worker 与 box2clay 原生 ref2va_serve
# (两者 pong 字段相同;本 worker 支持双协议,原生只支持 box2clay 协议)
SERVICE_TAG = "agentic-engine-h3"


def normalize_request(request: dict) -> dict:
    """协议归一:box2clay ref2va_serve 风格的请求映射为引擎协议。

    box2clay 键:cmd=generate, prompt, video(控制视频), image(首帧,可省),
    duration, seed, out。引擎键:prompt, seed, duration_s, first_frame,
    reference_video, output, turn_id。两套键名不同、语义一致——归一后
    同一生成路径,一个常驻服务同时服务 agentic runner 与 box2clay CLI
    (ref2va_serve.py generate / h3serve_batch.py)。
    """
    if "video" in request:  # box2clay 风格(引擎协议不用裸 "video" 键)
        return {
            "prompt": request.get("prompt", ""),
            "seed": request.get("seed", 0),
            "duration_s": request.get("duration", 5),
            "first_frame": request.get("image"),
            "reference_video": request.get("video"),
            "output": request.get("out", ""),
            "turn_id": request.get("turn_id", ""),
        }
    return request


def read_request_line() -> dict | None:
    """pipe 模式:rank 0 从 stdin 读一条请求;EOF 返回 None(=退出)。"""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class SocketChannel:
    """serve 模式的 rank0 请求通道:一次服务一个连接,断开不退出。

    fetch() 返回待广播的请求;ping 在此直接应答(不涉及其他 rank,
    它们仍在广播点上等待);shutdown 返回 None 让整组退出。
    """

    def __init__(self, socket_path: str) -> None:
        self.renders = 0
        self.world_size = 0
        self._path = Path(socket_path)
        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._file = None

    def open(self, world_size: int) -> None:
        self.world_size = world_size
        self._path.unlink(missing_ok=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self._path))
        # backlog 放宽:多客户端(runner 分片)并发连接时不再被 EAGAIN 弹回,
        # 仍在 accept 前排队;服务端语义不变,一次仍只服务一个连接
        self._server.listen(16)
        emit({"event": "listening", "socket": str(self._path), "world_size": world_size})

    def fetch(self) -> dict | None:
        """读下一条请求;连接断开则等待下一个连接;shutdown 返回 None。"""
        while True:
            if self._conn is None:
                self._accept()
            line = self._file.readline()
            if not line:
                self._drop_connection()
                continue  # 客户端断开,等下一个连接
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(request, dict):
                continue
            if request.get("cmd") == "ping":
                self._send({"event": "pong", "service": SERVICE_TAG,
                            "world_size": self.world_size,
                            "renders": self.renders})
                continue
            if request.get("cmd") == "shutdown":
                self._send({"event": "bye"})
                return None
            if not request.get("prompt"):
                continue
            return request

    def deliver(self, result: dict) -> None:
        self.renders += 1
        self._send(result)  # 客户端已断开时失败可忽略

    def close(self) -> None:
        self._drop_connection()
        if self._server is not None:
            self._server.close()
            self._server = None
        self._path.unlink(missing_ok=True)

    def _accept(self) -> None:
        conn, _ = self._server.accept()
        self._conn = conn
        self._file = conn.makefile("r")

    def _drop_connection(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
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
        except (BrokenPipeError, OSError):
            self._drop_connection()


def run_generation(engine, raw_request: dict, rank: int) -> dict:
    """广播之后的所有 rank 执行:生成 + rank0 存盘。

    请求先经 normalize_request(兼容 box2clay 协议)。带 reference_video
    时走 ref2va(控制视频 + 可选首图作 references);引擎 task=ref2va 而
    请求无控制视频时退化为纯图片 reference(消融/非 navigation 回合)。
    t2v 引擎收到 reference_video 会在 generate 内报错,错误按普通失败上报。
    """
    request = normalize_request(raw_request)
    turn_id = str(request.get("turn_id", ""))
    output = Path(str(request["output"]))
    result: dict = {"event": "result", "turn_id": turn_id}
    try:
        image = request.get("first_frame")
        video = request.get("reference_video")
        if video:
            from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

            references = []
            if image:
                references.append(
                    MiniMaxH3Reference(image=str(Path(str(image)).resolve())))
            references.append(
                MiniMaxH3Reference(video=str(Path(str(video)).resolve())))
            media = engine.generate(
                str(request["prompt"]),
                duration=int(request["duration_s"]),
                seed=int(request["seed"]),
                references=references,
            )
        elif getattr(engine, "task", "t2v") == "ref2va":
            from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

            references = (
                [MiniMaxH3Reference(image=str(Path(str(image)).resolve()))]
                if image else None
            )
            media = engine.generate(
                str(request["prompt"]),
                duration=int(request["duration_s"]),
                seed=int(request["seed"]),
                references=references,
            )
        else:
            media = engine.generate(
                str(request["prompt"]),
                duration=int(request["duration_s"]),
                seed=int(request["seed"]),
                image=image,
            )
        if rank == 0:
            if media is None:  # 理论不可达:rank 0 必有返回
                raise RuntimeError("rank 0 received no media from generate()")
            output.parent.mkdir(parents=True, exist_ok=True)
            media.save(output)
            # video/out 双键:engine 协议读 video,box2clay 协议读 out
            result.update(ok=True, video=str(output), out=str(output),
                          elapsed_s=round(media.elapsed_s, 3))
            if media.stages:  # text/denoise/video_vae/audio_vae 各阶段耗时
                result["stages"] = {k: round(v, 2) for k, v in media.stages.items()}
        else:
            result.update(ok=True)
    except Exception as exc:  # noqa: BLE001 - 错误类型会原样上报给调用方分类
        result.update(ok=False, error_type=type(exc).__name__, message=str(exc)[:2000])
    return result


def serve_loop(engine, fetch, deliver) -> None:
    """所有 rank 的公共循环:rank0 取请求 → 广播 → 全员生成 → 交付结果。

    fetch 返回 None(EOF/shutdown)即整组退出。
    """
    from datetime import timedelta

    import torch.distributed as dist

    rank = engine.rank
    # 常驻语义:rank1+ 会长时间停在 broadcast 上等下一条请求,而 gloo 的
    # 默认 collective 超时是 30 分钟——闲置半小时服务就会整组自杀。
    # 请求等待必须近乎无限(真正的卡死由客户端的渲染超时兜底)。
    request_group = dist.new_group(backend="gloo", timeout=timedelta(days=365))
    while True:
        payload: list[dict | None] = [None]
        if rank == 0:
            payload = [fetch()]
        dist.broadcast_object_list(payload, src=0, group=request_group)
        request = payload[0]
        if request is None or request.get("cmd") == "close":
            break
        result = run_generation(engine, request, rank)
        if rank == 0:
            deliver(result)
    engine.close()


def main() -> int:
    args = parse_args()
    h3_repo = Path(args.h3_repo).resolve()
    sys.path.insert(0, str(h3_repo))

    from h3_runtime.engine import RESOLUTIONS, MiniMaxH3Inference

    engine = MiniMaxH3Inference(
        args.model,
        args.adapter,
        attention_backend=args.attention_backend,
        task=args.task,
        width=RESOLUTIONS[args.res][0],
        height=RESOLUTIONS[args.res][1],
    )

    if args.serve:
        channel = SocketChannel(args.socket)
        if engine.rank == 0:
            channel.open(engine.world_size)
        try:
            serve_loop(engine, fetch=channel.fetch, deliver=channel.deliver)
        finally:
            if engine.rank == 0:
                channel.close()
    else:
        if engine.rank == 0:
            emit({"event": "ready", "world_size": engine.world_size})
        serve_loop(engine, fetch=read_request_line, deliver=emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
