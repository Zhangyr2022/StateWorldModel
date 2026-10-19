"""H3SessionRenderer:常驻 worker 模式的 Minimax-H3 渲染器。

与 engine/renderer/h3_worker.py 配对:torchrun 启动 worker(装载一次引擎,
约 8 分钟),之后每个 render() 只是一次 generate() 调用(5s 480p 约 7s),
适合 Phase 4 的连续多回合。单次渲染验证可用 SubprocessH3Renderer。

通信:JSON Lines over torchrun 的 stdin/stdout(torchrun 不重定向它们,
所有 rank 继承;协议约定仅 rank 0 读写)。stdout 上引擎自己的日志行
会被忽略。

超时即视为会话失步:直接关闭会话并抛出 RenderTimeoutError,
下次渲染需要重启会话(由调用方决定)。
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
import uuid
from pathlib import Path

from engine.renderer.base import (
    RenderFailedError,
    RenderRequest,
    RenderResult,
    RenderTimeoutError,
    Renderer,
)
from engine.renderer.frames import extract_final_frame
from engine.renderer.minimax_h3 import (
    DEFAULT_H3_REPO,
    DEFAULT_TORCHRUN,
    h3_environment,
)

WORKER_SCRIPT = Path(__file__).resolve().parent / "h3_worker.py"


class _LineReader:
    """带超时的行读取器(直接用 os.read,避免缓冲 IO 与 select 混用)。"""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""

    def readline(self, timeout_s: float) -> str | None:
        """返回一行(不含换行符);超时抛 RenderTimeoutError;EOF 返回 None。"""
        deadline = time.monotonic() + timeout_s
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RenderTimeoutError(
                    f"timed out after {timeout_s:.0f}s waiting for h3 worker response"
                )
            ready, _, _ = select.select([self._fd], [], [], remaining)
            if not ready:
                raise RenderTimeoutError(
                    f"timed out after {timeout_s:.0f}s waiting for h3 worker response"
                )
            chunk = os.read(self._fd, 65536)
            if not chunk:
                return None
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode("utf-8", errors="replace")


class H3SessionRenderer(Renderer):
    """装载一次、多回合复用的 Minimax-H3 渲染会话。"""

    def __init__(
        self,
        output_dir: Path,
        *,
        h3_repo: Path = DEFAULT_H3_REPO,
        torchrun: Path = DEFAULT_TORCHRUN,
        nproc: int = 8,
        resolution: str = "480p",
        attention_backend: str = "dense",
        model: Path | None = None,
        adapter: Path | None = None,
        worker_script: Path = WORKER_SCRIPT,
        # launch_cmd 供测试注入假 worker;None 时按 torchrun 方式启动真 worker
        launch_cmd: list[str] | None = None,
        startup_timeout_s: float = 3600,
        render_timeout_s: float = 1800,
        close_timeout_s: float = 300,
        worker_log_path: Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.h3_repo = Path(h3_repo)
        self.torchrun = Path(torchrun)
        self.nproc = nproc
        self.resolution = resolution
        self.attention_backend = attention_backend
        self.model = model
        self.adapter = adapter
        self.worker_script = Path(worker_script)
        self.launch_cmd = launch_cmd
        self.startup_timeout_s = startup_timeout_s
        self.render_timeout_s = render_timeout_s
        self.close_timeout_s = close_timeout_s
        self.worker_log_path = worker_log_path or (self.output_dir / "logs" / "h3_worker.log")

        self.session_tag = f"agentic-game-h3-{uuid.uuid4().hex[:8]}"
        self.render_count = 0
        self._proc: subprocess.Popen | None = None
        self._reader: _LineReader | None = None
        self._log_file = None
        self._closed = False

    # -- 会话管理 ------------------------------------------------------------

    def start(self) -> None:
        """启动 worker 并等待引擎装载完成(ready 行)。"""
        if self._proc is not None:
            return
        self.worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.worker_log_path, "ab")
        command = self.launch_cmd or self._build_launch_command()
        try:
            self._proc = subprocess.Popen(
                command,
                cwd=str(self.h3_repo),
                env=h3_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_file,
            )
        except OSError as exc:
            self._log_file.close()
            raise RenderFailedError(f"failed to launch h3 worker: {exc}") from exc
        self._reader = _LineReader(self._proc.stdout.fileno())
        self._wait_for_ready()

    def _build_launch_command(self) -> list[str]:
        command = [
            str(self.torchrun),
            "--standalone",
            f"--nproc_per_node={self.nproc}",
            str(self.worker_script),
            "--h3-repo",
            str(self.h3_repo),
            "--model",
            str(self.model or (self.h3_repo / "checkpoints" / "MiniMax-H3")),
            "--adapter",
            str(
                self.adapter
                or (
                    self.h3_repo
                    / "checkpoints"
                    / "FastH3-4-step-Preview-v1-LoRA"
                    / ("vsa-datafree" if self.attention_backend == "vsa" else "dense-datafree")
                    / "adapter_model.safetensors"
                )
            ),
            "--attention-backend",
            self.attention_backend,
            "--res",
            self.resolution,
            "--session-tag",
            self.session_tag,
        ]
        return command

    def _wait_for_ready(self) -> None:
        try:
            payload = self._next_message(self.startup_timeout_s)
        except RenderTimeoutError:
            self._kill()
            raise
        if payload is None:
            self._kill()
            raise RenderFailedError(
                "h3 worker exited before becoming ready; see "
                f"{self.worker_log_path} for the engine log"
            )
        if payload.get("event") != "ready":
            raise RenderFailedError(f"unexpected worker handshake message: {payload}")

    # -- 渲染 ----------------------------------------------------------------

    def render(self, request: RenderRequest) -> RenderResult:
        request.validate()
        if self._proc is None:
            self.start()
        assert self._proc is not None and self._proc.stdin is not None

        self.render_count += 1
        # 全部传绝对路径:worker 的 cwd 是 h3-a800 仓库,相对路径会写错位置
        video_path = (self.output_dir / "videos" / f"{request.turn_id}.mp4").resolve()
        frame_path = (self.output_dir / "frames" / f"{request.turn_id}.png").resolve()
        first_frame = request.first_frame.resolve() if request.first_frame else None
        message = {
            "prompt": request.prompt,
            "seed": request.seed,
            "duration_s": request.duration_s,
            "first_frame": str(first_frame) if first_frame else None,
            "output": str(video_path),
            "turn_id": request.turn_id,
        }
        started = time.monotonic()
        self._proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

        try:
            payload = self._next_message(self.render_timeout_s)
        except RenderTimeoutError:
            # 会话已失步(结果行稍后才会到达),只能整会话废弃。
            self.close()
            raise
        if payload is None:
            self.close()
            raise RenderFailedError(
                f"h3 worker exited while rendering {request.turn_id}; see "
                f"{self.worker_log_path} for the engine log"
            )
        if payload.get("event") != "result":
            raise RenderFailedError(f"unexpected worker message: {payload}")
        if not payload.get("ok"):
            raise RenderFailedError(
                f"h3 generate failed for {request.turn_id}: "
                f"{payload.get('error_type')}: {payload.get('message')}"
            )
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RenderFailedError(
                f"worker reported success but no video at {video_path} for {request.turn_id}"
            )
        final_frame = extract_final_frame(video_path, frame_path)
        return RenderResult(
            turn_id=request.turn_id,
            video_path=video_path,
            final_frame=final_frame,
            elapsed_s=float(payload.get("elapsed_s", 0.0)),
            seed=request.seed,
        )

    def _next_message(self, timeout_s: float) -> dict | None:
        """读下一条 JSON 协议消息;忽略引擎日志等非协议行。EOF 返回 None。"""
        assert self._reader is not None
        while True:
            line = self._reader.readline(timeout_s)
            if line is None:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # 引擎/编码器的自由格式日志行
            if isinstance(payload, dict):
                return payload

    # -- 清理 ----------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            if self._log_file:
                self._log_file.close()
            return
        try:
            if proc.stdin is not None and proc.poll() is None:
                try:
                    proc.stdin.write(b'{"cmd": "close"}\n')
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    proc.wait(timeout=self.close_timeout_s)
                except subprocess.TimeoutExpired:
                    self._kill()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stdin is not None:
                proc.stdin.close()
            if self._log_file:
                self._log_file.close()
            self._proc = None
            self._reader = None

    def _kill(self) -> None:
        """强制清理:torchrun 的 worker 自成进程组(start_new_session),
        按会话标签 pkill 是可靠的兜底。"""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            subprocess.run(
                ["pkill", "-f", self.session_tag],
                capture_output=True,
                check=False,
            )
            try:
                proc.send_signal(signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass

    def __enter__(self) -> "H3SessionRenderer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
