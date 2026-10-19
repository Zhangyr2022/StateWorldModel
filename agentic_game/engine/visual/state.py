"""VisualState:画面连续性状态,独立于 Game State。

Game State 表示游戏事实(HP、胜负),Visual State 只服务表现层的连续性:
当前场景、角色外观、以及"上一段视频的最后一帧"——它是下一段视频的
首帧输入(末帧链接约定)。Game State 永远拥有更高优先级,
不允许从 Visual State 反推游戏事实。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VisualState:
    """跨回合维护的视觉连续性状态。"""

    scene: str = ""
    style: str = ""
    # 角色名 → 外观描述。外观描述由游戏层维护,用于保持角色一致性。
    appearances: dict[str, str] = field(default_factory=dict)
    # 上一段视频的最后一帧;None 表示还没有画面(第一回合的 t2v 或离线模式)。
    # 所有模式(含 ref2va 控制视频)统一末帧链接:每轮以它为首帧,
    # 控制视频生产方据此重新检测/生成(蓝盒锚定当前帧主体)。
    current_frame: Path | None = None
    turn_index: int = 0
    # 游戏私有元数据(如当前场景 id),engine 不解释其内容。
    meta: dict = field(default_factory=dict)
