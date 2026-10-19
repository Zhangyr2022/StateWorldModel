"""WBench case 的 GameRuntime 协议实现:确定性的脚本化回合。

状态 = 案例字段(场景/主体/视角) + 游标 + 视角现值。
resolve_turn 是纯函数:校验 action 与游标对应,推进游标,
perspective_switch 翻转 perspective_now(影响后续导航轮的镜头措辞)。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from engine.events import GameEvent
from engine.runtime import (STATUS_ONGOING, Action, GameRuntime, TurnOutcome)

# WBench 视角切换 token → 视角现值
_FLIP = {"fp_to_tp": "third_person", "tp_to_fp": "first_person"}


def interaction_perspectives(case: dict) -> list[str]:
    """逐 interaction 返回生效视角(perspective_switch 翻转后续轮)。

    控制视频预热用它确定每个 navigation 轮该生成 tp 还是 fp 白模。
    """
    persp = case.get("settings", {}).get("perspective", "first_person")
    out = []
    for inter in case.get("interactions", []):
        if inter.get("type") == "perspective_switch":
            persp = _FLIP.get(inter.get("action", ""), persp)
        out.append(persp)
    return out


@dataclass(frozen=True)
class WBenchCaseState:
    case_id: str
    interactions: tuple          # tuple[dict]: {"type": ..., "action": ...}
    cursor: int
    perspective_now: str         # 当前生效视角(navigation 词典据此选择)
    environment_prompt: str
    character_prompt: str
    subject_desc: str = ""       # settings.subject.desc(纯外观;ref prompt 的 {desc} 槽)
    perspective_prompt: str = ""  # 官方视角句(ref prompt 的 {camera} 槽)

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.interactions)

    @property
    def current(self) -> dict:
        return self.interactions[self.cursor]


class WBenchRuntime(GameRuntime):
    """一个 case 一局;行动脚本即 interactions,结算为纯推进。"""

    def __init__(self, case: dict):
        self.case = case
        settings = case.get("settings", {})
        self._initial = WBenchCaseState(
            case_id=str(case.get("id", "")),
            interactions=tuple(case.get("interactions", [])),
            cursor=0,
            perspective_now=settings.get("perspective", "first_person"),
            environment_prompt=case.get("environment_prompt", ""),
            character_prompt=case.get("character_prompt", ""),
            subject_desc=(settings.get("subject") or {}).get("desc", ""),
            perspective_prompt=case.get("perspective_prompt", ""),
        )

    # -- GameRuntime 协议 ----------------------------------------------------

    def initial_state(self) -> WBenchCaseState:
        return self._initial

    def status_of(self, state: WBenchCaseState) -> str:
        return STATUS_ONGOING if not state.done else "win"

    def available_actions(self, state: WBenchCaseState) -> list[Action]:
        if state.done:
            return []
        inter = state.current
        return [Action(id=str(state.cursor), label=inter.get("action", ""),
                       description=inter.get("type", ""))]

    def resolve_turn(self, state: WBenchCaseState, action: Action, rng) -> TurnOutcome:
        inter = state.interactions[int(action.id)]  # 脚本局:action.id 即游标
        persp = state.perspective_now
        if inter.get("type") == "perspective_switch":
            persp = _FLIP.get(inter.get("action", ""), persp)
        new_state = replace(state, cursor=state.cursor + 1, perspective_now=persp)
        done = new_state.done
        return TurnOutcome(events=(), status="win" if done else STATUS_ONGOING,
                           state_after=new_state)

    def describe_state(self, state: WBenchCaseState) -> str:
        return (f"case={state.case_id} turn={state.cursor} "
                f"perspective={state.perspective_now}")


class ScriptedAgent:
    """行动脚本已定:永远执行当前游标指向的 interaction。"""

    def choose(self, state, actions):
        if not actions:
            raise RuntimeError("no scripted action available")
        return actions[0]
