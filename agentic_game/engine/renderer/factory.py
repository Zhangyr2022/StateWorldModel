"""Renderer 装配:backend + mode + 配置 → Renderer 实例。

run_demo 与 run_ui(Gradio)共用同一份装配逻辑,入口只负责解析自己的
参数形态;h3 的三种调用方式(remote/session/subprocess)与 socket 细节
都收敛在这里。
"""

from __future__ import annotations

from pathlib import Path

from engine.config import AppConfig
from engine.renderer.base import Renderer
from engine.renderer.fake import FakeRenderer
from engine.renderer.minimax_h3 import SubprocessH3Renderer
from engine.renderer.remote import H3RemoteRenderer
from engine.renderer.service import default_socket_path
from engine.renderer.session import H3SessionRenderer


def build_renderer(
    cfg: AppConfig,
    output_dir: Path,
    *,
    backend: str | None = None,
    mode: str | None = None,
    socket_path: Path | None = None,
) -> Renderer:
    """按配置装配渲染后端;backend/mode/socket 缺省取 cfg.renderer。"""
    backend = backend or cfg.renderer.backend
    if backend == "fake":
        return FakeRenderer(output_dir=output_dir)
    if backend != "h3":
        raise ValueError(f"unknown renderer backend: {backend}")

    mode = mode or cfg.renderer.mode
    common = {
        "h3_repo": Path(cfg.renderer.h3_repo),
        "nproc": cfg.renderer.nproc,
        "resolution": cfg.renderer.resolution,
        "attention_backend": cfg.renderer.attention_backend,
    }
    if mode == "remote":
        socket = socket_path or (
            Path(cfg.renderer.socket_path)
            if cfg.renderer.socket_path
            else default_socket_path(nproc=cfg.renderer.nproc, resolution=cfg.renderer.resolution)
        )
        return H3RemoteRenderer(output_dir=output_dir, socket_path=socket,
                                render_timeout_s=cfg.renderer.render_timeout_s)
    if mode == "subprocess":
        return SubprocessH3Renderer(
            output_dir=output_dir,
            timeout_s=cfg.renderer.request_timeout_s,
            **common,
        )
    if mode == "session":
        return H3SessionRenderer(
            output_dir=output_dir,
            torchrun=Path(cfg.renderer.torchrun),
            startup_timeout_s=cfg.renderer.startup_timeout_s,
            render_timeout_s=cfg.renderer.render_timeout_s,
            **common,
        )
    raise ValueError(f"unknown h3 renderer mode: {mode}")


__all__ = ["build_renderer"]
