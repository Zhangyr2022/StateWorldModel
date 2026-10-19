"""InteractiveGame:交互式(真人玩家)会话驱动器——Gradio 界面的引擎侧。

GameLoop.run() 一次性跑完整局,适合自动代理;真人玩家需要"走一步、看
一步":结算 → 看视频 → 再选择。本驱动器复用 GameLoop 的回合机制
(run_turn / run_ending / RunSummary / 日志落盘),只把"选动作"的时机
交还给界面:每调用一次 step(action) 执行一个回合,选择完全由调用方
(UI)决定——Agent-Mediated Interface 的 UI 形态,规则层依然无感知。

不 import gradio:界面框架只是本驱动器的一个客户端。
"""

from __future__ import annotations

from pathlib import Path

from engine.loop import RENDER_FAILURE_SKIP, GameLoop, RunSummary, TurnRecord
from engine.parts import GameParts
from engine.renderer.base import Renderer
from engine.runtime import STATUS_ONGOING, Action


class GameFinishedError(Exception):
    """游戏已结束时还试图执行回合。"""


class InteractiveGame:
    """一局进行中的交互式游戏:逐步结算 + 逐步渲染 + 落盘。"""

    def __init__(
        self,
        parts: GameParts,
        renderer: Renderer,
        *,
        output_dir,
        base_seed: int = 0,
        max_turns: int = 32,
        render_retries: int = 1,
        render_failure_policy: str = RENDER_FAILURE_SKIP,
    ) -> None:
        self.parts = parts
        self.renderer = renderer
        self.output_dir = output_dir
        # GameLoop.run() 开局会建输出目录,交互式驱动没有 run(),在这里补上:
        # 首回合渲染失败也要能写 turn_log.jsonl(§16:失败可追溯,状态不受影响)。
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.loop = GameLoop(
            runtime=parts.runtime,
            agent=_NeverAskedAgent(),  # 选择由 step() 直接给出,代理不参与
            visual_director=parts.visual_director,
            renderer=renderer,
            visual_state=parts.visual_state,
            output_dir=output_dir,
            base_seed=base_seed,
            max_turns=max_turns,
            render_retries=render_retries,
            render_failure_policy=render_failure_policy,
            ending_director=parts.ending_director,
            post_render=parts.post_render,
            hud_of=parts.hud_of,
        )
        self.summary = RunSummary(session_dir=output_dir)
        self.state = parts.runtime.initial_state()
        self._finalize_if_over()

    # -- 玩家界面需要的查询 ----------------------------------------------------

    @property
    def status(self) -> str:
        return self.parts.runtime.status_of(self.state)

    @property
    def finished(self) -> bool:
        return self.summary.finished_reason != ""

    @property
    def turns_played(self) -> int:
        return len(self.summary.records)

    @property
    def max_turns(self) -> int:
        return self.loop.max_turns

    def describe(self) -> str:
        """给玩家看的状态摘要。"""
        return self.parts.describe(self.state)

    def available_actions(self) -> list[Action]:
        """当前可选行为(openworld 的 VLM 提议会在首次调用时生成并缓存)。"""
        return self.parts.runtime.available_actions(self.state)

    def history_lines(self) -> list[str]:
        """逐回合的一行摘要(动作 → 结果),供 UI 展示。

        caption 的格式是"{动作} | {结算后状态}",由 GameLoop 写入。
        """
        lines = []
        for record in self.summary.records:
            marker = "✓" if record.render_error is None else "✗ 渲染失败"
            lines.append(f"R{record.index + 1} · {record.caption} [{marker}]")
        if self.summary.ending is not None:
            lines.append(f"◆ 终局:{self.summary.ending.status}")
        return lines

    # -- 回合执行 ---------------------------------------------------------------

    def step(self, action: Action) -> TurnRecord:
        """执行一个回合:结算 → Visual → 渲染 → 记录,返回回合记录。

        结算失败(非法动作、LLM 结算失败等)会抛异常:游戏状态不变,
        玩家可以重新选择;渲染失败按策略跳过,状态同样不受影响。
        """
        if self.finished:
            raise GameFinishedError("this game is already over")
        record, state = self.loop.run_turn(self.state, action, index=self.turns_played)
        self.summary.records.append(record)
        self.state = state
        self._finalize_if_over()
        return record

    def close(self) -> None:
        """释放渲染后端资源(重复调用安全)。"""
        try:
            self.renderer.close()
        except Exception:  # noqa: BLE001 - 关闭失败不影响已结束的对局
            pass

    # -- 内部 --------------------------------------------------------------------

    def _finalize_if_over(self) -> None:
        """镜像 GameLoop.run 的三种结束条件,必要时渲染结束动画。"""
        if self.finished:
            return
        status = self.status
        if status != STATUS_ONGOING:
            self.summary.finished_reason = "game_over"
            self.summary.final_status = status
        elif len(self.summary.records) >= self.loop.max_turns:
            self.summary.finished_reason = "max_turns"
            self.summary.final_status = STATUS_ONGOING
        elif not self.available_actions():
            self.summary.finished_reason = "no_actions"
            self.summary.final_status = STATUS_ONGOING
        else:
            return
        self.summary.turns_played = len(self.summary.records)
        if self.summary.finished_reason == "game_over" and self.loop.ending_director is not None:
            self.summary.ending = self.loop.run_ending(self.summary.final_status)


class _NeverAskedAgent:
    """占位代理:InteractiveGame 直接调 run_turn,永远不会询问它。"""

    def choose(self, state, actions):
        raise AssertionError("InteractiveGame does not consult an agent")
