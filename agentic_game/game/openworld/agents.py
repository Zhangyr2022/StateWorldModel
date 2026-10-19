"""开放世界 Demo 的玩家代理:从 VLM 提议中选择,或输入自定义行动。

自动探索回合(见 game/openworld/ 模块 docstring)只有一个 AUTO 行动:
不询问玩家,直接执行——"不弹出选择"在 CLI 代理上的体现。
"""

from __future__ import annotations

from engine.runtime import Action
from game.openworld.actions import ACTION_AUTO, ACTION_CUSTOM


class OpenworldCliAgent:
    """命令行:显示 VLM 提议的选项,选序号或直接输入自定义行动。"""

    def __init__(self, input_fn=input, print_fn=print) -> None:
        self._input = input_fn
        self._print = print_fn

    def choose(self, state, actions: list[Action]) -> Action:
        if len(actions) == 1 and actions[0].id == ACTION_AUTO.id:
            self._print(f"\n{state.summary()}")
            self._print(f"  [自动] {actions[0].label} —— {actions[0].description}")
            return actions[0]
        self._print(f"\n{state.summary()}")
        for index, action in enumerate(actions, start=1):
            self._print(f"  [{index}] {action.label}")
        self._print("  [直接输入文字] = 自由行动")
        while True:
            raw = self._input("你的选择 > ").strip()
            if not raw:
                continue
            if raw.isdigit() and 1 <= int(raw) <= len(actions):
                return Action(id="act", label=actions[int(raw) - 1].label)
            return Action(id=ACTION_CUSTOM.id, label=raw[:40])


class OpenworldAutoAgent:
    """自动游玩:每回合取 VLM 提议的第一个行动(AUTO 回合即唯一行动)。"""

    def choose(self, state, actions: list[Action]) -> Action:
        return actions[0]
