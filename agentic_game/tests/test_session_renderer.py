"""H3SessionRenderer 测试:与 stub worker 走真实子进程 + JSON 行协议。

不涉及真实 Minimax-H3(见 AGENTS.md:真实 API 测试与普通
单元测试分开);这里验证会话启动、渲染请求、超时、worker 崩溃与清理。
"""

import tempfile
import unittest
from pathlib import Path

from engine.renderer.base import (
    RenderFailedError,
    RenderRequest,
    RenderTimeoutError,
)
from engine.renderer.session import H3SessionRenderer

STUB = Path(__file__).resolve().parent / "stub_h3_worker.py"
FFMPEG_AVAILABLE = __import__("shutil").which("ffmpeg") is not None


def make_request(turn_id: str, first_frame: Path | None) -> RenderRequest:
    return RenderRequest(
        turn_id=turn_id,
        prompt="a cinematic test shot",
        seed=1,
        first_frame=first_frame,
        duration_s=5,
    )


@unittest.skipUnless(FFMPEG_AVAILABLE, "stub synthesizes videos with ffmpeg")
class TestSessionRenderer(unittest.TestCase):
    def make_renderer(self, tmpdir: Path, mode: str, **kwargs) -> H3SessionRenderer:
        return H3SessionRenderer(
            output_dir=tmpdir,
            h3_repo=Path(tmpdir),  # stub 不读 h3 仓库,任意存在目录即可
            launch_cmd=["false"] if mode == "launch_fail" else
                       [__import__("sys").executable, str(STUB), mode],
            startup_timeout_s=30,
            render_timeout_s=kwargs.pop("render_timeout_s", 30),
            close_timeout_s=10,
            **kwargs,
        )

    def test_render_two_turns_with_frame_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            initial = tmpdir / "initial.png"
            initial.write_bytes(b"\x89PNG fake initial frame")
            renderer = self.make_renderer(tmpdir, "ok")
            renderer.start()
            try:
                first = renderer.render(make_request("turn-001", initial))
                self.assertTrue(first.video_path.exists())
                self.assertTrue(first.final_frame.exists())
                self.assertEqual(first.turn_id, "turn-001")

                second = renderer.render(make_request("turn-002", first.final_frame))
                self.assertTrue(second.video_path.exists())

                # 首帧参数确实被透传给了 worker(stub 留了副本)
                forwarded = second.video_path.with_suffix(".mp4.first")
                self.assertEqual(forwarded.read_bytes(), first.final_frame.read_bytes())
                self.assertEqual(renderer.render_count, 2)
            finally:
                renderer.close()

    def test_worker_crash_raises_render_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            renderer = self.make_renderer(tmpdir, "die")
            renderer.start()
            with self.assertRaises(RenderFailedError):
                renderer.render(make_request("turn-001", None))
            renderer.close()

    def test_timeout_closes_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            renderer = self.make_renderer(tmpdir, "slow", render_timeout_s=0.5)
            renderer.start()
            with self.assertRaises(RenderTimeoutError):
                renderer.render(make_request("turn-001", None))
            # 超时后 render() 已主动关闭(废弃)会话
            self.assertIsNone(renderer._proc)
            self.assertTrue(renderer._closed)

    def test_worker_dying_before_ready_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            renderer = self.make_renderer(tmpdir, "launch_fail")
            with self.assertRaises(RenderFailedError):
                renderer.start()

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            renderer = self.make_renderer(tmpdir, "ok")
            renderer.start()
            renderer.close()
            renderer.close()  # 不应抛异常
            self.assertTrue(renderer._closed)


if __name__ == "__main__":
    unittest.main()
