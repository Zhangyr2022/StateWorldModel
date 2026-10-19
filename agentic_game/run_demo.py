#!/usr/bin/env python3
"""Minimal turn-based demo entry: plays the full loop end to end —
player choice → deterministic resolution → VisualScript → video → next turn.

The game catalog lives in game/registry.py; this entry parses arguments,
assembles the renderer backend and the player agent, then lets GameLoop play
one full game in one go (for turn-by-turn play / on-screen play, use run_ui.py).

Usage:
    # Offline (fake renderer, no GPU dependency) — verifies the full loop:
    python3 run_demo.py --renderer fake --turns 8 --agent auto

    # Real Minimax-H3 (connect to the resident service, recommended; start it first):
    #   python3 -m engine.renderer.service start      # loads onto 8 GPUs and stays (~6 min)
    python3 run_demo.py --renderer h3 --mode remote --turns 3 --agent cli

    # Real Minimax-H3 (ephemeral session: the renderer starts/stops its own worker):
    python3 run_demo.py --renderer h3 --mode session --turns 3

    # Real Minimax-H3 (subprocess mode: reload the engine every turn; single-turn checks only):
    python3 run_demo.py --renderer h3 --mode subprocess --turns 1

Afterwards all turn videos + the ending animation are assembled by default into
one captioned replay (game_montage.mp4 in the same directory); --no-montage disables this.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from engine.agents import CliAgent, PlayerAgent, RandomAgent, ScriptedAgent
from engine.config import load_config
from engine.llm.client import LLMClient
from engine.llm.service import default_llm_socket
from engine.loop import GameLoop, RENDER_FAILURE_ABORT, RENDER_FAILURE_SKIP
from engine.montage import build_summary_montage
from engine.parts import GameParts, resolve_frame
from engine.renderer.factory import build_renderer
from engine.renderer.frames import create_placeholder_frame
from game.registry import GAMES, GAME_IDS, get

logger = logging.getLogger("run_demo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=GAME_IDS, default="duel_oracle_machine_epic",
                        help="game to run (see game/registry.py and game/skins/*.json): "
                             "elemental combat / story dialogue / open world / themed skins")
    parser.add_argument("--renderer", choices=("fake", "h3"), help="renderer backend (default from config)")
    parser.add_argument("--mode", choices=("remote", "session", "subprocess"), help="how to invoke the h3 backend")
    parser.add_argument("--agent", choices=("auto", "random", "cli"), default="auto",
                        help="auto = scripted/LLM agent, random = random, cli = interactive "
                             "command line; for on-screen play use run_ui.py")
    parser.add_argument("--script", type=str,
                        default="fireball,frostbolt,ice_shield,arcane,fireball,fireball",
                        help="action sequence for auto mode (scripted agent; comma-separated "
                             "action ids, the last one repeats once exhausted)")
    parser.add_argument("--seed", type=int, help="random seed (default from config)")
    parser.add_argument("--turns", type=int, help="max turns (default from config)")
    parser.add_argument("--res", choices=("480p", "768p"), help="h3 resolution")
    parser.add_argument("--nproc", type=int, help="h3 GPU process count (default: auto-detect visible GPUs)")
    parser.add_argument("--output", type=Path, help="output root directory (default from config)")
    parser.add_argument("--config", type=Path, help="path to config.toml (default ./config.toml)")
    parser.add_argument("--render-failure", choices=(RENDER_FAILURE_SKIP, RENDER_FAILURE_ABORT),
                        help="render-failure policy: skip = continue from the previous frame (default), abort = stop")
    parser.add_argument("--initial-frame", type=Path, help="game's starting first frame (default: as declared "
                                                           "by the game; a placeholder is generated if missing)")
    parser.add_argument("--no-montage", action="store_true", help="skip assembling the full replay video at the end")
    parser.add_argument("--socket", type=Path, help="resident-service socket path for remote mode")
    parser.add_argument("--llm-socket", type=Path, help="LLM service socket path")
    # 游戏级选项(如 openworld 的 --scene)来自注册表:先拿 --game,再补定义后严格解析
    pre, _ = parser.parse_known_args(argv)
    for option in get(pre.game).options:
        parser.add_argument(f"--{option.name}", choices=list(option.choices),
                            default=option.default, help=option.label)
    return parser.parse_args(argv)


def build_agent(args: argparse.Namespace, parts: GameParts, llm: LLMClient | None) -> PlayerAgent:
    """玩家代理 = 可替换的选择策略:cli / random / 脚本或 LLM 代理(auto)。"""
    if args.agent == "cli":
        if parts.cli_agent_factory is not None:
            return parts.cli_agent_factory(llm)
        return CliAgent(describe_state=parts.describe)
    if args.agent == "random":
        return RandomAgent(seed=(args.seed if args.seed is not None else 0) + 1)  # 与游戏随机源分开
    # auto:LLM 参与选择的代理(romance/openworld),否则脚本表达"意图",
    # fallback=True:动作不可用(如 MP 不足)时退回第一个可用动作
    if parts.auto_agent_factory is not None:
        return parts.auto_agent_factory(llm)
    return ScriptedAgent([part.strip() for part in args.script.split(",") if part.strip()],
                         fallback=True)


def resolve_initial_frame(args: argparse.Namespace, parts: GameParts,
                          output_dir: Path, seed: int, backend: str) -> Path:
    """初始首帧:游戏开始时还没有任何视频,之后的每一帧都来自上一段视频的最后一帧。"""
    if args.initial_frame is not None:
        frame = args.initial_frame
    else:
        frame = resolve_frame(parts.frame)
    if frame is None or not frame.exists():
        # 皮肤声明了 frame 但文件缺失必须大声告警:占位图会让整局的
        # i2v 链从一张假帧长出来,静默降级等于跑废一局(2026-09-30 教训)。
        logger.warning(
            "The first-frame image declared by the game is unavailable (%s); "
            "falling back to a text placeholder %s. The whole run will be "
            "rendered from this baseline — check the skin's frame path!",
            parts.frame or "(not declared)", frame)
        frame = create_placeholder_frame(
            output_dir / "frames" / "initial.png",
            lines=(args.game.upper(), "initial frame", f"seed={seed} renderer={backend}"),
        )
    return frame


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    entry = get(args.game)

    base_seed = args.seed if args.seed is not None else cfg.loop.base_seed
    # 渲染装配参数:CLI 覆盖 config(工厂从 cfg 读取)
    if args.nproc:
        cfg.renderer.nproc = args.nproc
    if args.res:
        cfg.renderer.resolution = args.res
    # openworld 的结局最早出现在第 12 个交互幕,留出 AUTO 探索幕的余量
    default_turns = {"romance": 10, "openworld": 20, "adv": 25}.get(
        entry.category, cfg.loop.max_turns)
    max_turns = args.turns if args.turns is not None else default_turns
    session_name = time.strftime("run-%Y%m%d-%H%M%S")
    output_dir = (args.output or Path(cfg.loop.output_dir)) / session_name
    backend = args.renderer or cfg.renderer.backend

    # 需要 LLM 的游戏(romance/openworld)先确认常驻服务就绪(快速失败并提示)
    llm: LLMClient | None = None
    if entry.needs_llm:
        llm = LLMClient(socket_path=args.llm_socket or default_llm_socket(),
                        generate_timeout_s=cfg.renderer.render_timeout_s)
        if llm.ping() is None:
            raise SystemExit(
                f"LLM service not ready ({llm.socket_path}); start it first: "
                "python3 -m engine.llm.service start")

    # 组装游戏部件(注册表唯一清单,入口不含任何游戏逻辑)
    parts: GameParts = entry.build(llm, **{o.name: getattr(args, o.name) for o in entry.options})
    parts.visual_state.current_frame = resolve_initial_frame(args, parts, output_dir,
                                                             base_seed, backend)
    logger.info("initial first frame: %s", parts.visual_state.current_frame)

    renderer = build_renderer(cfg, output_dir, backend=backend,
                              mode=args.mode or cfg.renderer.mode,
                              socket_path=args.socket)
    loop = GameLoop(
        runtime=parts.runtime,
        agent=build_agent(args, parts, llm),
        visual_director=parts.visual_director,
        renderer=renderer,
        visual_state=parts.visual_state,
        output_dir=output_dir,
        base_seed=base_seed,
        max_turns=max_turns,
        render_retries=cfg.loop.render_retries,
        render_failure_policy=args.render_failure or cfg.loop.render_failure_policy,
        ending_director=parts.ending_director,
        post_render=parts.post_render,
        hud_of=parts.hud_of,
    )

    logger.info("session dir: %s", output_dir)
    started = time.monotonic()
    try:
        summary = loop.run()
    finally:
        renderer.close()
    wall_s = time.monotonic() - started

    montage_path = None
    if not args.no_montage:
        try:
            montage_path = build_summary_montage(
                summary, output_dir / "game_montage.mp4",
                scene_labels=parts.scene_labels,
                ending_captions=parts.ending_captions,
            )
        except Exception as exc:  # noqa: BLE001 - 拼接是表现层,失败不影响对局结果
            logger.warning("montage failed: %s", exc)

    print("\n===== game over =====")
    print(f"game: {args.game}  ending: {summary.final_status}  reason: {summary.finished_reason}"
          f"  turns: {summary.turns_played}")
    print(f"wall time: {wall_s:.1f}s  output dir: {output_dir}")
    for record in summary.records:
        marker = "OK " if record.render_error is None else "ERR"
        video = Path(record.video_path).name if record.video_path else "-"
        print(f"  [{marker}] {record.turn_id}  action={record.caption or record.action_id:<24} "
              f"status={record.status:<8} video={video}"
              + (f"  error={record.render_error}" if record.render_error else ""))
    if summary.ending is not None:
        marker = "OK " if summary.ending.render_error is None else "ERR"
        print(f"  [{marker}] {summary.ending.turn_id} (ending animation)"
              + (f"  error={summary.ending.render_error}" if summary.ending.render_error else ""))
    if montage_path is not None:
        print(f"full replay (with state captions): {montage_path}")
    if summary.final_status == "won":
        print(f"\nVictory! Full recording in {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
