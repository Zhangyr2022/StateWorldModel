"""GameEvent:一次结算产生的最小事实单元。

基础层只定义事件结构,不定义任何具体游戏的事件类型;类型字符串由游戏层
自行约定(如 "damage" / "heal")。Visual Script 由游戏层根据事件列表生成,
Engine 不解释事件的语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GameEvent:
    """单个游戏事实,例如"石像巨人受到 14 点伤害"。

    一个 Action 的结算通常产生 1 到多个事件;事件的顺序即发生顺序。
    """

    type: str
    subject: str = ""
    target: str = ""
    amount: int | None = None
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "subject": self.subject,
            "target": self.target,
            "amount": self.amount,
            "message": self.message,
            "detail": dict(self.detail),
        }
