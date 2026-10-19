"""LLMClient:连接常驻 LLM 服务(engine/llm/worker.py --serve)的客户端。

Unix socket + JSON Lines,与 H3 渲染服务同一套协议风格。游戏逻辑
(如恋爱游戏的对话结算)只依赖这里,不 import transformers。
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

_CONNECT_TIMEOUT_S = 10.0


class LLMError(Exception):
    """LLM 服务错误基类。"""


class LLMTimeoutError(LLMError):
    """服务在时限内没有返回。"""


class LLMFailedError(LLMError):
    """请求失败(服务不可达、生成失败、应答非法等)。"""


class LLMClient:
    def __init__(self, *, socket_path: Path, generate_timeout_s: float = 600.0) -> None:
        self.socket_path = Path(socket_path)
        self.generate_timeout_s = generate_timeout_s
        self.call_count = 0

    # -- 探活 ----------------------------------------------------------------

    def ping(self, timeout_s: float = 10.0) -> dict | None:
        try:
            return self._round_trip({"cmd": "ping"}, timeout_s=timeout_s)
        except LLMError:
            return None

    def wait_until_ready(self, timeout_s: float, poll_interval_s: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pong = self.ping(timeout_s=poll_interval_s)
            if pong is not None:
                return pong
            time.sleep(1.0)
        raise LLMTimeoutError(f"LLM service not ready after {timeout_s:.0f}s ({self.socket_path})")

    # -- 对话 ----------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        seed: int = 0,
        max_tokens: int = 256,
        timeout_s: float | None = None,
        images: list[str | Path] | None = None,
    ) -> str:
        """发送一轮对话,返回模型续写的文本。服务端按贪心解码保证可复现。

        images 非空时启用 VL 模式:图片路径列表会附到最后一条 user
        消息上(供 VLM"观看"当前画面/视频采样帧)。
        """
        self.call_count += 1
        payload = self._round_trip(
            {
                "messages": messages,
                "seed": int(seed),
                "max_tokens": int(max_tokens),
                "images": [str(p) for p in images] if images else None,
            },
            timeout_s=timeout_s or self.generate_timeout_s,
        )
        if not payload.get("ok"):
            raise LLMFailedError(
                f"LLM generate failed: {payload.get('error_type')}: {payload.get('message')}"
            )
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise LLMFailedError("LLM returned an empty completion")
        return text

    # -- 传输 ----------------------------------------------------------------

    def _round_trip(self, message: dict, timeout_s: float) -> dict:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(_CONNECT_TIMEOUT_S)
        try:
            try:
                conn.connect(str(self.socket_path))
            except (ConnectionRefusedError, FileNotFoundError) as exc:
                raise LLMFailedError(
                    f"LLM service not reachable at {self.socket_path} ({exc}); "
                    "start it with: python3 -m engine.llm.service start"
                ) from exc
            conn.settimeout(timeout_s)
            conn.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            buffer = b""
            while b"\n" not in buffer:
                try:
                    chunk = conn.recv(65536)
                except socket.timeout as exc:
                    raise LLMTimeoutError(
                        f"LLM service did not answer within {timeout_s:.0f}s"
                    ) from exc
                except (ConnectionResetError, OSError) as exc:
                    raise LLMFailedError(f"LLM service connection failed: {exc}") from exc
                if not chunk:
                    raise LLMFailedError("LLM service closed the connection unexpectedly")
                buffer += chunk
        finally:
            conn.close()
        line = buffer.split(b"\n", 1)[0].decode("utf-8", errors="replace")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LLMFailedError(f"unparsable LLM response: {line[:200]!r}") from exc
        if not isinstance(payload, dict):
            raise LLMFailedError(f"unexpected LLM response: {line[:200]!r}")
        return payload
