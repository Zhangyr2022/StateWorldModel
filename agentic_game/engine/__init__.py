"""生成式回合制游戏基础层(Engine)= 论文的 Generalizable Game Runtime。

本包是"Game = Game Definition + Generic Runtime + Generative Renderer"
中的后两项;游戏定义永远在 game/ 层(Stage 0 由 coding agent 生成,
不修改本包)。模块与论文贡献的对应:

- 贡献 1  Rule-Grounded Architecture:
    runtime.py(确定性结算协议)+ loop.py(结算先于渲染,渲染失败
    不影响 Game State)
- 贡献 2  Generalizable Game Runtime:
    runtime.py + loop.py(Turn System)+ 本包全部协议——不同游戏 =
    不同 Game Definition + 本层原样复用
- 贡献 3  Agent-Mediated Interface:
    agents.py(玩家决策入口)+ llm/(LLM 参与)+ visual/(游戏语义
    → 视觉语义的结构化桥)
- 贡献 4  Training-Free Semantic Rendering:
    visual/script.py + visual/compiler.py + visual/standard.py(按
    NEWSKILL 规范编译)+ renderer/(现成视频模型,零训练)

依赖方向(见 AGENTS.md「Engine engineering rules」):Game Layer → Engine → Renderer Adapter。
Engine 不 import 任何具体游戏代码;Renderer 是唯一的视觉生成出口。

核心循环:玩家选择 → 确定性结算 → Visual Script → Prompt → 视频
→ 最后一帧 → 下一回合。
"""

from engine.loop import GameLoop, RunSummary, TurnRecord
from engine.runtime import (
    STATUS_DRAW,
    STATUS_LOST,
    STATUS_ONGOING,
    STATUS_WON,
    Action,
    Rng,
    TurnOutcome,
    TurnResult,
)

__all__ = [
    "GameLoop",
    "RunSummary",
    "TurnRecord",
    "STATUS_DRAW",
    "STATUS_LOST",
    "STATUS_ONGOING",
    "STATUS_WON",
    "Action",
    "Rng",
    "TurnOutcome",
    "TurnResult",
]
