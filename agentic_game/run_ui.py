#!/usr/bin/env python3
"""Gradio player console entry: one shared interface for all games; everything
the player does is a choice.

Usage:
    # Get it running offline first (fake renderer; no GPU / LLM service needed):
    python3 run_ui.py                       # fake backend by default
    python3 run_ui.py --renderer fake

    # Real rendering and dialogue (start the two resident services first):
    python3 -m engine.renderer.service start
    python3 -m engine.llm.service start
    python3 run_ui.py --renderer h3 --mode remote --port 7860

Open http://127.0.0.1:7860 in a browser; two screens:
the "lobby" (pick a game from the cover wall / categories → choose backend and
seed → start; the screen switches automatically) → the "match" (pick an action
each turn; adv/openworld accept free text) → watch the turn's clip →
at the ending, generate the full captioned replay with one click.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.config import load_config
from engine.llm.service import default_llm_socket
from engine.renderer.service import default_socket_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="listen address (use 0.0.0.0 to serve others)")
    parser.add_argument("--port", type=int, default=7860, help="listen port")
    parser.add_argument("--share", action="store_true", help="create a public gradio link")
    parser.add_argument("--renderer", choices=("fake", "h3"), help="default renderer backend (from config)")
    parser.add_argument("--mode", choices=("remote", "session", "subprocess"),
                        help="how to invoke the h3 backend (from config)")
    parser.add_argument("--config", type=Path, help="path to config.toml (default ./config.toml)")
    parser.add_argument("--socket", type=Path, help="h3 remote service socket path")
    parser.add_argument("--llm-socket", type=Path, help="LLM service socket path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.renderer:
        cfg.renderer.backend = args.renderer
    if args.mode:
        cfg.renderer.mode = args.mode

    import gradio as gr  # noqa: F401 - 提前失败:未安装 gradio 时给出明确报错

    try:
        from engine.ui.app import build_app
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing dependency: {exc.name}; run pip install gradio first") from exc

    renderer_socket = args.socket or (
        Path(cfg.renderer.socket_path) if cfg.renderer.socket_path
        else default_socket_path(nproc=cfg.renderer.nproc, resolution=cfg.renderer.resolution)
    )
    demo = build_app(args.config, llm_socket=args.llm_socket or default_llm_socket(),
                     renderer_socket=renderer_socket,
                     renderer_backend=args.renderer, renderer_mode=args.mode)
    demo.queue(default_concurrency_limit=2)  # 开局/回合各占一个槽;渲染期间仍能翻界面
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                show_error=True, inbrowser=False)  # 阻塞直到 Ctrl+C
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
