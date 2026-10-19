"""H3RemoteRenderer:连接常驻 H3 服务的渲染客户端。

配合 engine/renderer/service.py 管理的常驻 worker
(engine/renderer/h3_worker.py --serve):引擎在 8 卡上长期驻留,
每次渲染只是一次 socket 调用 + generate,秒级接入。
游戏与主循环仍只依赖 Renderer 抽象,感知不到传输方式。
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

from engine.renderer.base import (
    RenderError,
    RenderFailedError,
    RenderRequest,
    RenderResult,
    RenderTimeoutError,
    Renderer,
)
from engine.renderer.frames import extract_final_frame

_CONNECT_TIMEOUT_S = 10.0
# backlog 满重试上限(多客户端排队属常态,≥2-3 个渲染请求的时长)
_CONNECT_BUSY_DEADLINE_S = 300.0
# 拒连(服务端无监听,可能正在重启)重试窗口
_CONNECT_REFUSED_RETRY_S = 30.0


class H3RemoteRenderer(Renderer):
    """Unix socket 客户端:每次渲染新建连接,服务端一次处理一个连接。"""

    def __init__(
        self,
        output_dir: Path,
        *,
        socket_path: Path,
        render_timeout_s: float = 1800,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.socket_path = Path(socket_path)
        self.render_timeout_s = render_timeout_s
        self.render_count = 0

    # -- 服务探活 ------------------------------------------------------------

    def ping(self, timeout_s: float = 10.0) -> dict | None:
        """向服务发 ping;服务不可达返回 None。"""
        try:
            return self._round_trip({"cmd": "ping"}, timeout_s=timeout_s)
        except RenderError:
            return None

    def wait_until_ready(self, timeout_s: float, poll_interval_s: float = 5.0) -> dict:
        """等待服务就绪(引擎装载完成);超时抛 RenderTimeoutError。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pong = self.ping(timeout_s=poll_interval_s)
            if pong is not None:
                return pong
            time.sleep(1.0)
        raise RenderTimeoutError(
            f"H3 service not ready after {timeout_s:.0f}s (socket {self.socket_path})"
        )

    # -- 渲染 ----------------------------------------------------------------

    def render(self, request: RenderRequest) -> RenderResult:
        request.validate()
        self.render_count += 1
        # 绝对路径:worker 的 cwd 在 h3-a800 仓库,相对路径会写错位置
        video_path = (self.output_dir / "videos" / f"{request.turn_id}.mp4").resolve()
        frame_path = (self.output_dir / "frames" / f"{request.turn_id}.png").resolve()
        first_frame = request.first_frame.resolve() if request.first_frame else None
        reference_video = (
            request.reference_video.resolve() if request.reference_video else None
        )
        message = {
            "prompt": request.prompt,
            "seed": request.seed,
            "duration_s": request.duration_s,
            "first_frame": str(first_frame) if first_frame else None,
            "reference_video": str(reference_video) if reference_video else None,
            "output": str(video_path),
            "turn_id": request.turn_id,
        }
        started = time.monotonic()
        payload = self._round_trip(message, timeout_s=self.render_timeout_s)
        if payload.get("event") != "result":
            raise RenderFailedError(f"unexpected service message: {payload}")
        if not payload.get("ok"):
            raise RenderFailedError(
                f"h3 generate failed for {request.turn_id}: "
                f"{payload.get('error_type')}: {payload.get('message')}"
            )
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RenderFailedError(
                f"service reported success but no video at {video_path} for {request.turn_id}"
            )
        final_frame = extract_final_frame(video_path, frame_path)
        return RenderResult(
            turn_id=request.turn_id,
            video_path=video_path,
            final_frame=final_frame,
            elapsed_s=float(payload.get("elapsed_s", 0.0)) or time.monotonic() - started,
            seed=request.seed,
        )

    # -- socket 基础 ---------------------------------------------------------

    def _connect(self, timeout_s: float) -> socket.socket:
        """连接服务端;backlog 满(EAGAIN)按退避重试。

        服务端一次只服务一个连接且 listen backlog 很小,多客户端并发时
        connect 会得到 BlockingIOError——这是"排队忙"而非服务不可用,
        重试等待即可;拒连(无监听)短重试容忍服务重启窗口;socket 文件
        不存在则立即失败(交给上层 ensure_service 处理)。
        """
        deadline = time.monotonic() + max(timeout_s, _CONNECT_BUSY_DEADLINE_S)
        refused_deadline = time.monotonic() + _CONNECT_REFUSED_RETRY_S
        while True:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(_CONNECT_TIMEOUT_S)
            try:
                conn.connect(str(self.socket_path))
                return conn
            except BlockingIOError as exc:
                conn.close()
                if time.monotonic() >= deadline:
                    raise RenderFailedError(
                        f"H3 service at {self.socket_path} stayed busy for "
                        f"{max(timeout_s, _CONNECT_BUSY_DEADLINE_S):.0f}s ({exc})"
                    ) from exc
                time.sleep(2.0)
            except ConnectionRefusedError as exc:
                conn.close()
                if time.monotonic() >= refused_deadline:
                    raise RenderFailedError(
                        f"H3 service not reachable at {self.socket_path} ({exc}); "
                        "start it with: python3 -m engine.renderer.service start"
                    ) from exc
                time.sleep(2.0)
            except (FileNotFoundError, OSError) as exc:
                conn.close()
                raise RenderFailedError(
                    f"H3 service not reachable at {self.socket_path} ({exc}); "
                    "start it with: python3 -m engine.renderer.service start"
                ) from exc

    def _round_trip(self, message: dict, timeout_s: float) -> dict:
        """连接 → 发一行 → 收一行(带超时)。"""
        conn = self._connect(timeout_s)
        try:
            conn.settimeout(timeout_s)
            conn.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            buffer = b""
            while b"\n" not in buffer:
                try:
                    chunk = conn.recv(65536)
                except socket.timeout as exc:
                    raise RenderTimeoutError(
                        f"H3 service did not answer within {timeout_s:.0f}s"
                    ) from exc
                except ConnectionResetError as exc:
                    raise RenderFailedError(f"H3 service dropped the connection: {exc}") from exc
                if not chunk:
                    raise RenderFailedError("H3 service closed the connection unexpectedly")
                buffer += chunk
        finally:
            conn.close()
        line = buffer.split(b"\n", 1)[0].decode("utf-8", errors="replace")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RenderFailedError(f"unparsable service response: {line[:200]!r}") from exc
        if not isinstance(payload, dict):
            raise RenderFailedError(f"unexpected service response: {line[:200]!r}")
        return payload

    def close(self) -> None:
        """无本地资源;服务由 engine.renderer.service 管理。"""
