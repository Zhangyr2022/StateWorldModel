"""开放世界 Demo 的玩家行为:从 VLM 提议中选择、自由行动,或自动探索。

label 承载行动内容;id 为 "act" / "custom" / "auto"(自动探索回合)。
"""

from __future__ import annotations

from engine.runtime import Action

ACTION_ACT = Action(id="act", label="行动", description="按 VLM 提议行动(label=提议内容)")
ACTION_CUSTOM = Action(id="custom", label="自由行动", description="玩家自定义行动(label=内容)")
# 自动探索回合(AUTO 模式):画面缺乏有意义的交互
# 机会时,AI 推进一幕,不弹出选择;规则层保证 AUTO 之后的下一回合必须回到
# 交互。运行期 description 会被替换为 VLM 的判定理由,展示给玩家。
ACTION_AUTO = Action(id="auto", label="▶ 自动探索",
                     description="画面暂无有意义的交互机会,本回合由 AI 推进场景")

ALL_ACTIONS: tuple[Action, ...] = (ACTION_ACT, ACTION_CUSTOM, ACTION_AUTO)
