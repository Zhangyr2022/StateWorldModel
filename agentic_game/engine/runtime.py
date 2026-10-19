"""Turn Runtime 基础接口:动作、随机源、回合结果与游戏层需实现的协议。

这是论文贡献 2(Generalizable Game Runtime)的核心:游戏层只实现本模块
定义的最小协议,即获得整套基础层(Turn System / Visual Pipeline /
Renderer)。不同游戏 = 不同 Game Definition(游戏层代码)+ 本协议原样
复用;Stage 0 的 coding agent 也只需面向这 5 个方法工作。

核心约束(贡献 1,Rule-Grounded):Game Runtime 是世界唯一可信的状态
来源,所有游戏结果必须在视觉生成之前确定。本模块只定义接口与结果容器,
不包含任何具体游戏规则。

Engine 的其余部分(runtime 之外)只通过 GameRuntime 协议访问游戏,
因此游戏逻辑可以在 Renderer 完全关闭的情况下独立运行与测试。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from engine.events import GameEvent

# 回合终局状态。游戏层通过 GameRuntime.status_of 返回其中之一。
STATUS_ONGOING = "ongoing"
STATUS_WON = "won"
STATUS_LOST = "lost"
STATUS_DRAW = "draw"

FINAL_STATUSES = (STATUS_WON, STATUS_LOST, STATUS_DRAW)


@dataclass(frozen=True)
class Action:
    """玩家可选择的一个行为。id 是稳定的标识符,UI 与测试都通过 id 引用。"""

    id: str
    label: str
    description: str = ""


class Rng:
    """基础层唯一随机源。

    游戏逻辑不得自行创建随机生成器(见 AGENTS.md「Engine engineering rules」):所有随机
    都经由 GameLoop 持有的同一个 Rng 实例传入,从而保证相同
    (initial state, action 序列, seed) 得到完全相同的结果。
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._random = random.Random(seed)

    def randint(self, low: int, high: int) -> int:
        """闭区间 [low, high] 内的随机整数。"""
        return self._random.randint(low, high)

    def chance(self, probability: float) -> bool:
        """以给定概率返回 True(0.0~1.0)。"""
        return self._random.random() < probability

    def random(self) -> float:
        return self._random.random()


@dataclass(frozen=True)
class TurnOutcome:
    """resolve_turn 的返回值:这次结算产生的事件、终局状态与新状态。

    state_after 必须是一个新对象(resolve_turn 不得修改传入的 state),
    主循环用它作为下一回合的 state_before,保证状态变化可追溯。
    """

    events: tuple[GameEvent, ...]
    status: str = STATUS_ONGOING
    state_after: Any = None


@dataclass(frozen=True)
class TurnResult:
    """一个完整回合的记录:动作前状态 → 玩家选择 → 事件 → 动作后状态。

    turn_id 在整个游戏过程中唯一,用于关联事件、Visual Script、Prompt、
    视频与日志(见 AGENTS.md「Engine engineering rules」)。
    """

    turn_id: str
    index: int
    action: Action
    state_before: Any
    state_after: Any
    outcome: TurnOutcome

    @property
    def events(self) -> tuple[GameEvent, ...]:
        return self.outcome.events

    @property
    def status(self) -> str:
        return self.outcome.status


class GameRuntime(Protocol):
    """游戏层需要实现的最小接口。

    约定:
    - resolve_turn 必须是确定性的,且不得修改传入的 state(游戏层应先
      复制再修改),这样 state_before / state_after 天然可追溯。
    - status_of 返回 STATUS_* 之一;返回非 ongoing 时主循环结束。
    """

    def initial_state(self) -> Any:
        """创建游戏的初始状态。"""
        ...

    def available_actions(self, state: Any) -> list[Action]:
        """当前状态下玩家可以执行的行为(已按游戏规则过滤)。"""
        ...

    def resolve_turn(self, state: Any, action: Action, rng: Rng) -> TurnOutcome:
        """确定性结算:应用 action,返回事件与终局状态。"""
        ...

    def status_of(self, state: Any) -> str:
        """当前游戏是否结束(STATUS_* 之一)。"""
        ...

    def describe_state(self, state: Any) -> str:
        """给玩家/日志看的一段状态摘要;UI 与 Runtime 解耦的展示接口。"""
        ...
