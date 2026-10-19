"""PlayerAgent:玩家如何做选择(论文贡献 3 的 Stage 1 玩家侧入口)。

Agent-Mediated Interface:自然语言意图经 agent 转换为标准化 Action 后
才进入系统——CLI 输入、脚本、随机或 LLM 代理(见 game/*/agents.py)
都只是可替换的选择策略,Runtime 与规则层对"谁在选"完全无感知。

UI 与 Runtime 解耦(见 AGENTS.md「Engine engineering rules」):
Runtime 只接收标准化 Action,选择逻辑可以来自命令行、脚本、随机或
将来的 Web UI,游戏规则完全无感知。
"""

from __future__ import annotations

import random
from typing import Any, Protocol

from engine.runtime import Action


class PlayerAgent(Protocol):
    def choose(self, state: Any, actions: list[Action]) -> Action:
        """从当前可用行为中选择一个;必须返回 actions 中的元素。"""


class ScriptedAgent:
    """按给定 action id 序列选择,序列用尽后重复最后一个。

    用于自动演示与测试:输入确定 ⇒ 选择确定,配合 Rng 的 seed 即可
    完整复现一局游戏。

    fallback=True 时,脚本动作当前不可用(如 MP 不足)则退回第一个
    可用动作——脚本模拟的是"意图",可用性仍由游戏规则裁决;
    fallback=False 保持严格模式,不可用直接报错(用于可用性测试)。
    """

    def __init__(self, action_ids: list[str] | tuple[str, ...], fallback: bool = False) -> None:
        if not action_ids:
            raise ValueError("ScriptedAgent needs at least one action id")
        self.action_ids = list(action_ids)
        self.fallback = fallback
        self.cursor = 0

    def choose(self, state: Any, actions: list[Action]) -> Action:
        action_id = self.action_ids[min(self.cursor, len(self.action_ids) - 1)]
        self.cursor += 1
        by_id = {action.id: action for action in actions}
        if action_id not in by_id:
            if self.fallback and actions:
                return actions[0]
            raise ValueError(
                f"scripted action {action_id!r} is not available now "
                f"(available: {sorted(by_id)})"
            )
        return by_id[action_id]


class RandomAgent:
    """带 seed 的随机选择:用于压力测试与随机演示,不影响规则确定性。"""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)

    def choose(self, state: Any, actions: list[Action]) -> Action:
        return self._random.choice(actions)


class CliAgent:
    """命令行交互:展示状态摘要与行为列表,玩家输入序号或 id。"""

    def __init__(self, describe_state: Any = None, input_fn=input, print_fn=print) -> None:
        self.describe_state = describe_state
        self._input = input_fn
        self._print = print_fn

    def choose(self, state: Any, actions: list[Action]) -> Action:
        while True:
            if self.describe_state is not None:
                self._print(f"\n{self.describe_state(state)}")
            for i, action in enumerate(actions, start=1):
                self._print(f"  [{i}] {action.label}  {action.description}")
            raw = self._input("选择行动(序号或 id)> ").strip()
            choice = self._parse(raw, actions)
            if choice is not None:
                return choice
            self._print(f"无效输入: {raw!r}")

    @staticmethod
    def _parse(raw: str, actions: list[Action]) -> Action | None:
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(actions):
                return actions[index]
            return None
        for action in actions:
            if action.id == raw:
                return action
        return None
