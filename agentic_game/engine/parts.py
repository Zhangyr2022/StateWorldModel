"""GameParts:游戏层交给入口层(run_demo / run_ui)的统一装配契约。

一个游戏 = 一份 GameParts:实现 GameRuntime 协议的结算器 + 视觉映射 +
可选的结束动画与渲染后钩子,外加 UI 需要的少量声明(初始首帧、自由输入
动作、场景标签、终局字幕、代理工厂)。run_demo 与 Gradio 界面都只面向
这份契约工作,不 import 任何具体游戏——新增游戏零改动入口(论文贡献 2:
Generalizable Game Runtime 在装配层的体现)。

字段约定:
- runtime/visual_director/visual_state/ending_director/post_render:
  与 GameLoop 构造参数一一对应。
- frame:初始首帧路径(相对仓库根或绝对路径);None 表示没有首帧图
  (GameLoop 会退化为占位首帧或 t2v)。
- free_text_action:自由输入类玩法(恋爱"说话"、开放世界"自由行动")
  的动作模板;UI 据此显示文本输入框,玩家输入经 dataclasses.replace
  写进 label。None 表示本游戏只从 available_actions 里点选。
- auto_action:声明式自动行动(如开放世界 AUTO 探索回合);当它是当前
  唯一可选行动时,UI 不弹选择直接执行。None 表示本游戏没有自动行动。
- auto_agent_factory / cli_agent_factory:run_demo --agent auto/cli 用的
  玩家代理工厂(UI 模式不使用代理,玩家直接在界面上选择)。
- scene_labels:场景 id → 中文名(回放拼接时插入黑屏过渡并标注)。
- ending_captions:终局状态 → 回放字幕(如 won → "◆ 战斗胜利 ◆")。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from engine.agents import PlayerAgent
from engine.loop import EndingDirector, PostRenderHook, VisualDirector
from engine.runtime import Action, GameRuntime
from engine.visual.state import VisualState

# auto 代理工厂:输入 LLMClient 或 None,输出本局的玩家代理。
AgentFactory = Callable[[object | None], PlayerAgent]

# 仓库根(engine/parts.py 的上一级):皮肤/游戏声明里的 frame 相对路径以此为基准。
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_frame(frame: str | Path | None) -> Path | None:
    """GameParts.frame → 绝对路径;None 原样返回(本游戏没有首帧图)。"""
    if frame is None:
        return None
    path = Path(frame)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class GameParts:
    """run_demo / run_ui 共用的游戏装配结果。"""

    runtime: GameRuntime
    visual_director: VisualDirector
    visual_state: VisualState
    ending_director: EndingDirector | None = None
    post_render: PostRenderHook | None = None
    # 可选 HUD 快照:state → {"left"/"right"/"action"...};回放视频烧录用,
    # 不进结算。None 表示本游戏不提供(回放无 HUD)。
    hud_of: Callable[[object], dict] | None = None

    # -- UI / 入口所需的声明 -------------------------------------------------
    frame: str | Path | None = None
    describe_state: Callable[[object], str] | None = None
    free_text_action: Action | None = None
    # 声明式自动行动(如开放世界的 AUTO 探索回合):当它是当前唯一可选
    # 行动时,UI 不等玩家点选、直接执行并播放这一回合;这种回合不连续
    # 出现的约束(如 AUTO 后必须回到交互)由游戏规则负责。None 表示本
    # 游戏没有自动行动,界面行为不变。
    auto_action: Action | None = None
    scene_labels: dict[str, str] = field(default_factory=dict)
    ending_captions: dict[str, str] = field(default_factory=dict)
    auto_agent_factory: AgentFactory | None = None
    cli_agent_factory: AgentFactory | None = None

    def describe(self, state: object) -> str:
        """状态摘要(优先用游戏自带实现,回退 runtime.describe_state)。"""
        if self.describe_state is not None:
            return self.describe_state(state)
        return self.runtime.describe_state(state)


__all__ = ["AgentFactory", "GameParts"]
