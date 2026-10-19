"""LLMClient 测试:与 stub socket 服务走真实 Unix socket 协议。"""

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from engine.llm.client import LLMClient, LLMFailedError, LLMTimeoutError

STUB = Path(__file__).resolve().parent / "stub_llm_server.py"


class StubService:
    def __init__(self, socket_path: Path, mode: str = "ok") -> None:
        self.socket_path = socket_path
        self.mode = mode
        self.proc = None

    def __enter__(self) -> "StubService":
        self.proc = subprocess.Popen(
            [sys.executable, str(STUB), str(self.socket_path), self.mode],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(50):
            if self.socket_path.exists():
                return self
            time.sleep(0.1)
        raise RuntimeError("stub LLM service did not open its socket")

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
        self.socket_path.unlink(missing_ok=True)


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.socket_path = self.tmpdir / "llm.sock"

    def test_ping_and_chat(self):
        with StubService(self.socket_path):
            client = LLMClient(socket_path=self.socket_path)
            pong = client.ping()
            self.assertIsNotNone(pong)
            self.assertEqual(pong["event"], "pong")
            text = client.chat([{"role": "user", "content": "你好"}], seed=7)
            self.assertIn("reply", text)
            self.assertEqual(client.call_count, 1)

    def test_not_running_raises_with_hint(self):
        client = LLMClient(socket_path=self.tmpdir / "nope.sock")
        with self.assertRaises(LLMFailedError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("llm.service start", str(ctx.exception))
        self.assertIsNone(client.ping())

    def test_empty_completion_is_failure(self):
        with StubService(self.socket_path, "empty"):
            client = LLMClient(socket_path=self.socket_path)
            with self.assertRaises(LLMFailedError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_timeout_raises(self):
        with StubService(self.socket_path, "slow"):
            client = LLMClient(socket_path=self.socket_path, generate_timeout_s=0.5)
            with self.assertRaises(LLMTimeoutError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_graceful_shutdown(self):
        with StubService(self.socket_path):
            client = LLMClient(socket_path=self.socket_path)
            payload = client._round_trip({"cmd": "shutdown"}, timeout_s=10)  # noqa: SLF001
            self.assertEqual(payload["event"], "bye")
            time.sleep(0.3)
            self.assertIsNone(client.ping())


if __name__ == "__main__":
    unittest.main()
