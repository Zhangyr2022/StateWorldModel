"""Renderer 基础层:游戏代码访问视觉生成的唯一出口(Phase 1)。

游戏代码不直接调用 Minimax-H3(见 AGENTS.md「Engine engineering rules」);所有 HTTP/子进程、
API key、超时、重试、最后一帧提取都收敛在这里。测试用 FakeRenderer 替换。
"""

from engine.renderer.base import (
    FrameExtractionError,
    InvalidRenderRequestError,
    RenderError,
    RenderFailedError,
    RenderRequest,
    RenderResult,
    RenderTimeoutError,
    Renderer,
)
from engine.renderer.frames import create_placeholder_frame, extract_final_frame
from engine.renderer.fake import FakeRenderer

__all__ = [
    "FrameExtractionError",
    "InvalidRenderRequestError",
    "RenderError",
    "RenderFailedError",
    "RenderRequest",
    "RenderResult",
    "RenderTimeoutError",
    "Renderer",
    "FakeRenderer",
    "create_placeholder_frame",
    "extract_final_frame",
]
