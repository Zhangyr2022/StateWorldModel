"""H3RemoteRenderer 测试:与 stub socket 服务走真实 Unix socket 协议。

验证常驻服务模式(需求 2):客户端随时连接、ping 探活、渲染、超时、
优雅关闭。不涉及真实 Minimax-H3。
"""

import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from engine.renderer.base import (
    RenderFailedError,
    RenderRequest,
    RenderTimeoutError,
)
from engine.renderer.remote import H3RemoteRenderer

STUB = Path(__file__).resolve().parent / "stub_h3_server.py"
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def make_request(turn_id: str, first_frame: Path | None) -> RenderRequest:
    return RenderRequest(
        turn_id=turn_id,
        prompt="a cinematic test shot",
        seed=1,
        first_frame=first_frame,
        duration_s=5,
    )


class StubService:
    """在后台启动 stub socket 服务,context 退出时清理。"""

    def __init__(self, socket_path: Path, mode: str) -> None:
        self.socket_path = socket_path
        self.mode = mode
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "StubService":
        self.proc = subprocess.Popen(
            [sys.executable, str(STUB), str(self.socket_path), self.mode],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(50):
            if self.socket_path.exists():
                return self
            time.sleep(0.1)
        raise RuntimeError("stub service did not open its socket")

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
        self.socket_path.unlink(missing_ok=True)


@unittest.skipUnless(FFMPEG_AVAILABLE, "stub synthesizes videos with ffmpeg")
class TestRemoteRenderer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.socket_path = self.tmpdir / "h3.sock"

    def test_ping_and_render_with_frame_continuity(self):
        with StubService(self.socket_path, "ok"):
            renderer = H3RemoteRenderer(output_dir=self.tmpdir, socket_path=self.socket_path)
            pong = renderer.ping()
            self.assertIsNotNone(pong)
            self.assertEqual(pong["event"], "pong")

            initial = self.tmpdir / "initial.png"
            initial.write_bytes(b"\x89PNG fake initial frame")
            first = renderer.render(make_request("turn-001", initial))
            self.assertTrue(first.video_path.exists())
            self.assertTrue(first.final_frame.exists())

            second = renderer.render(make_request("turn-002", first.final_frame))
            self.assertTrue(second.video_path.exists())
            # 首帧参数确实被透传给了服务端(stub 留了副本)
            forwarded = second.video_path.with_suffix(".mp4.first")
            self.assertEqual(forwarded.read_bytes(), first.final_frame.read_bytes())
            self.assertEqual(renderer.render_count, 2)

    def test_service_not_running_raises_with_hint(self):
        renderer = H3RemoteRenderer(output_dir=self.tmpdir,
                                    socket_path=self.tmpdir / "nope.sock")
        with self.assertRaises(RenderFailedError) as ctx:
            renderer.render(make_request("turn-001", None))
        self.assertIn("service start", str(ctx.exception))
        self.assertIsNone(renderer.ping())

    def test_timeout_raises(self):
        with StubService(self.socket_path, "slow"):
            renderer = H3RemoteRenderer(output_dir=self.tmpdir, socket_path=self.socket_path,
                                        render_timeout_s=0.5)
            with self.assertRaises(RenderTimeoutError):
                renderer.render(make_request("turn-001", None))

    def test_graceful_shutdown_via_socket(self):
        with StubService(self.socket_path, "ok"):
            renderer = H3RemoteRenderer(output_dir=self.tmpdir, socket_path=self.socket_path)
            payload = renderer._round_trip({"cmd": "shutdown"}, timeout_s=10)
            self.assertEqual(payload["event"], "bye")
            time.sleep(0.3)
            self.assertIsNone(renderer.ping())  # 服务已退出

    def test_wait_until_ready_times_out(self):
        renderer = H3RemoteRenderer(output_dir=self.tmpdir,
                                    socket_path=self.tmpdir / "nope.sock")
        with self.assertRaises(RenderTimeoutError):
            renderer.wait_until_ready(timeout_s=1.0, poll_interval_s=0.2)


if __name__ == "__main__":
    unittest.main()
