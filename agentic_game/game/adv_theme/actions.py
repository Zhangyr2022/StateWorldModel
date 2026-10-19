"""adv 类游戏的玩家动作:唯一的动作是"说一句话"(自由文本输入)。

台词来源可替换(人/脚本/LLM),规则层只关心 Action.label 里的一句话。
"""

from __future__ import annotations

from engine.runtime import Action

ACTION_TALK = Action(id="talk", label="说话",
                     description="以男主角的身份对她说一句话(自由输入)")

__all__ = ["ACTION_TALK"]
