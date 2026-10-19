"""machine 主题模板的结算规则:玩家与 Boss 完全对称的状态机解释器。

一个回合的六个阶段(与设计师约定一致):
  ① Boss 选择动作   —— 代码控制:pre 过滤 + 权重掷选(唯一与玩家的差异)
  ② 玩家选择动作   —— 由玩家(人/脚本/LLM)选择
  ③ Boss 行动结算   —— 匹配 boss.transitions,应用效果与伤害
  ④ 玩家行动结算   —— 匹配 transitions,记录 outcome、应用效果与伤害
  ⑤ 耦合结算       —— 双方通过共享 vars 耦合:转移的 conditions/effects/
                      outcome_expr 可读写对方阶段留下的变量(如 boss_shield、
                      windup),无需独立阶段
  ⑥ 更新状态       —— 两条状态轴(player.current / boss_state)各自迁移;
                      endings 条件满足即终局

随机只来自传入的 Rng;resolve_turn 深拷贝后修改,不改入参。
"""

from __future__ import annotations

import copy
import random
import zlib

from engine.events import GameEvent
from engine.runtime import (
    STATUS_LOST,
    STATUS_ONGOING,
    STATUS_WON,
    Action,
    Rng,
    TurnOutcome,
)
from game.machine_theme.expr import evaluate
from game.machine_theme.skin import CONDITION_OPS, EFFECT_OPS, MachineSkin


class MachineState:
    """纯数据游戏状态:双方状态轴 + 变量表 + 回合计数 + 结算索引。"""

    def __init__(self, current: str, vars: dict, turn: int = 0,
                 boss_state: str = "", last_boss_action: str = "",
                 last_outcome: str = "") -> None:
        self.current = current                    # 玩家状态轴
        self.vars = dict(vars)
        self.turn = turn
        self.ending = ""
        self.boss_state = boss_state              # Boss 姿态轴
        self.last_boss_action = last_boss_action  # 供演出层读取
        self.last_outcome = last_outcome          # 玩家动作的结算结果档位

    def summary(self, label: str, var_display: tuple = (),
                boss_label: str = "") -> str:
        keys = var_display or tuple(self.vars)
        parts = [f"第{self.turn}回合", label]
        parts += [f"{k}={self.vars[k]}" for k in keys if k in self.vars]
        if boss_label and self.boss_state:
            parts.append(f"boss:{self.boss_state}")
        return " | ".join(parts)

    def to_snapshot(self) -> dict:
        """回合日志用的结构化快照(rebuild_replay_hud 等回放工具读取;
        经 engine.loop._to_jsonable 的 to_snapshot 钩子自动落盘)。"""
        return {
            "current": self.current,
            "vars": dict(self.vars),
            "turn": self.turn,
            "ending": self.ending,
            "boss_state": self.boss_state,
            "last_boss_action": self.last_boss_action,
            "last_outcome": self.last_outcome,
        }


class MachineRuntime:
    """实现 GameRuntime 协议;不依赖任何 Renderer 或 UI。"""

    def __init__(self, skin: MachineSkin) -> None:
        self.skin = skin
        # 调试接管:非 None 时本回合 Boss 直接执行该动作(一次性,结算后清空)。
        # 由 UI 的 debug 面板写入;常规玩法保持 None,走权重自动选择。
        self.forced_boss_action: str | None = None
        # 两侧转移表:(from|*, action) -> [TransitionSpec...];专属优先于通配
        self._player_by_key: dict[tuple[str, str], list] = {}
        for tr in skin.transitions:
            self._player_by_key.setdefault((tr.from_state, tr.action), []).append(tr)
        self._boss_by_key: dict[tuple[str, str], list] = {}
        if skin.boss is not None:
            for tr in skin.boss.transitions:
                self._by_key_add(self._boss_by_key, tr)

    @staticmethod
    def _by_key_add(table: dict, tr) -> None:
        table.setdefault((tr.from_state, tr.action), []).append(tr)

    # -- 引擎协议 ------------------------------------------------------------

    def initial_state(self) -> MachineState:
        boss = self.skin.boss
        return MachineState(
            self.skin.initial_state, dict(self.skin.vars),
            boss_state=boss.initial_state if boss is not None else "")

    def _env(self, state: MachineState) -> dict:
        """表达式环境:vars + boss_state(Boss 姿态对表达式可见)。"""
        env = dict(state.vars)
        env.setdefault("boss_state", state.boss_state)
        return env

    def _player_transitions(self, state: MachineState, action_id: str) -> list:
        return (self._player_by_key.get((state.current, action_id), [])
                + self._player_by_key.get(("*", action_id), []))

    def _boss_transitions(self, state: MachineState, action_id: str) -> list:
        return (self._boss_by_key.get((state.boss_state, action_id), [])
                + self._boss_by_key.get(("*", action_id), []))

    def _usable_actions(self, state: MachineState) -> list[str]:
        node = self.skin.state(state.current)
        return list(node.actions or tuple(a.id for a in self.skin.actions))

    def available_actions(self, state: MachineState) -> list[Action]:
        if self.status_of(state) != STATUS_ONGOING:
            return []
        usable = set(self._usable_actions(state))
        result = []
        for a in self.skin.actions:
            if a.id not in usable or not self._pre_ok(state, a.id):
                continue
            result.append(Action(id=a.id, label=a.label, description=a.description))
        return result

    def _pre_ok(self, state: MachineState, action_id: str) -> bool:
        """动作可用性:存在至少一条 (玩家状态|*, 动作) 转移,其 pre 为真。"""
        for tr in self._player_transitions(state, action_id):
            if not tr.pre:
                return True
            if evaluate(tr.pre, self._env(state), self._probe_rng(state, action_id)):
                return True
        return False

    def _status_probe(self):
        """终局条件等纯确定性求值用:固定种子,保证 status_of 是纯函数。"""
        return random.Random(0)

    def _probe_rng(self, state: MachineState, action_id: str = ""):
        """pre 求值用:不消耗正式随机流,但按 (seed, 回合, 动作) 派生,
        保证同 seed 同局面的可复现性。"""
        key = f"{self.skin.game_id}:{state.turn}:{action_id}:{state.current}"
        return random.Random(zlib.crc32(key.encode()) & 0xFFFFFFFF)

    def status_of(self, state: MachineState) -> str:
        for status in ("won", "lost"):
            ending = self.skin.endings.get(status)
            if ending and all(self._check(self._env(state), c) for c in ending.conditions):
                return STATUS_WON if status == "won" else STATUS_LOST
        return STATUS_ONGOING

    def describe_state(self, state: MachineState) -> str:
        node = self.skin.state(state.current)
        parts = [f"第{state.turn}回合", node.label]
        keys = self.skin.var_display or tuple(state.vars)
        parts += [f"{k}={state.vars[k]}" for k in keys if k in state.vars]
        if state.boss_state and self.skin.boss is not None \
                and state.boss_state != self.skin.boss.initial_state:
            parts.append(f"boss:{state.boss_state}")
        return " | ".join(parts)

    # -- 回合流水线(六阶段) ---------------------------------------------------

    def resolve_turn(self, state: MachineState, action: Action, rng: Rng) -> TurnOutcome:
        if action.id not in self._usable_actions(state):
            raise ValueError(f"action not available in current state: {action.id}")
        new_state = copy.deepcopy(state)
        new_state.turn += 1

        events: list[GameEvent] = [
            GameEvent(type="action", subject=new_state.current, detail={"action": action.id})
        ]

        # ① 玩家行动结算(先于 Boss:防御护盾当回合即生效——不可侵犯之环/
        #    终末之门必须能挡住本回合的 Boss 攻击;沿用旧 duel_theme 骨架的设计)
        _, player_events = self._player_resolve(new_state, action, rng)
        events.extend(player_events)

        # ② Boss 选择动作(代码:pre 过滤 + 权重掷选)+ 行动结算
        #    (玩家已分出胜负则跳过:击杀即胜利,无反击)
        if self.status_of(new_state) == STATUS_ONGOING:
            boss_action_id = self._boss_choose(new_state, rng)
            if boss_action_id is not None:
                events.extend(self._boss_resolve(new_state, boss_action_id, rng))

        # ③ 耦合结算:两侧效果通过共享 vars 天然耦合(无需独立阶段)——
        #    护盾/举盾都持久化在 vars 里:本回合立的盾挡本回合(Boss)与
        #    下一回合(双方)的同侧攻击,消耗后清空。

        # ④ 更新状态:两条状态轴已在各自结算中迁移;这里做终局判定
        status = self.status_of(new_state)
        if status != STATUS_ONGOING:
            ending = self.skin.endings[status]
            new_state.ending = status
            events.append(GameEvent(
                type="victory" if status == STATUS_WON else "defeat",
                message=ending.caption or status,
            ))
        return TurnOutcome(events=tuple(events), status=status, state_after=new_state)

    # -- Boss 侧(①选择 + ③结算) ----------------------------------------------

    def boss_action_menu(self, state: MachineState) -> list[Action]:
        """调试菜单:Boss 的全部动作(含前置条件说明),供 UI 强制指定。

        与 available_actions 不同,这里不按 pre 过滤——调试的意义就在于
        能主动触发平时凑不齐前置的演出;前置是否满足写进 description
        (不满足时强制执行可能不产生结算事件,只演出空转)。
        """
        boss = self.skin.boss
        if boss is None:
            return []
        env = self._env(state)
        menu: list[Action] = []
        for spec in boss.actions:
            usable = (not spec.pre
                      or evaluate(spec.pre, env, self._probe_rng(state, spec.id)))
            note = "" if usable else "(当前前置不满足,强制执行可能不产生结算)"
            pre_note = f" 前置:{spec.pre} " if spec.pre else " "
            menu.append(Action(id=spec.id, label=spec.label,
                               description=f"{spec.description}{pre_note}{note}"))
        return menu

    def _boss_choose(self, state: MachineState, rng: Rng) -> str | None:
        """Boss 动作选择:pre 通过且有可用转移的动作里按权重掷选;
        forced_boss_action 非 None 时直接接管(调试模式,一次性)。"""
        forced, self.forced_boss_action = self.forced_boss_action, None
        boss = self.skin.boss
        if boss is None:
            return None
        if forced:
            return forced
        eligible = []
        for a in boss.actions:
            if a.pre and not evaluate(a.pre, self._env(state),
                                      self._probe_rng(state, a.id)):
                continue
            if self._boss_transitions(state, a.id):
                eligible.append(a)
        if not eligible:
            return None
        total = sum(a.weight for a in eligible)
        pick = rng.random() * total if total > 0 else 0
        acc = 0.0
        chosen = eligible[-1]
        for a in eligible:
            acc += a.weight
            if pick <= acc:
                chosen = a
                break
        return chosen.id

    def _boss_resolve(self, state: MachineState, action_id: str,
                      rng: Rng) -> list[GameEvent]:
        """Boss 行动结算:匹配 boss.transitions 并应用(效果/伤害/姿态迁移)。"""
        state.last_boss_action = action_id
        events: list[GameEvent] = []
        moved = None
        for tr in self._boss_transitions(state, action_id):
            if not all(self._check(self._env(state), c, rng) for c in tr.conditions):
                continue
            if tr.probability < 1.0 and not rng.chance(tr.probability):
                continue
            moved = tr
            break
        if moved is None:
            events.append(GameEvent(
                type="state_change", subject=state.boss_state,
                detail={"boss_state": state.boss_state, "action": action_id,
                        "stayed": True},
            ))
            return events

        for eff in moved.effects:
            self._apply(state, eff, rng, action_id)
        if moved.damage:
            events.extend(self._apply_damage(state, "boss", moved, rng))
        state.boss_state = moved.to_state or state.boss_state
        events.append(GameEvent(
            type="state_change", subject=state.boss_state,
            detail={"boss_state": state.boss_state, "action": action_id},
        ))
        for custom in moved.events:
            events.append(GameEvent(
                type=custom.get("type", "custom"),
                subject=state.boss_state,
                message=custom.get("message", ""),
                detail=dict(custom.get("detail", {})),
            ))
        return events

    # -- 玩家行动结算(④) ------------------------------------------------------

    def _player_resolve(self, state: MachineState, action: Action,
                        rng: Rng) -> tuple[bool, list[GameEvent]]:
        """玩家行动结算:匹配 transitions 并应用;记录 outcome 档位。"""
        events: list[GameEvent] = []
        moved = None
        for tr in self._player_transitions(state, action.id):
            if not all(self._check(self._env(state), c, rng) for c in tr.conditions):
                continue
            if tr.probability < 1.0 and not rng.chance(tr.probability):
                continue
            moved = tr
            break

        if moved is None:
            state.last_outcome = ""
            events.append(GameEvent(
                type="state_change", subject=state.current,
                detail={"from": state.current, "to": state.current,
                        "action": action.id, "stayed": True},
            ))
            return False, events

        if moved.outcome or moved.outcome_expr:
            # 结果档位先于效果求值:表达式读的是结算前的变量
            state.last_outcome = (
                moved.outcome if moved.outcome
                else str(evaluate(moved.outcome_expr, self._env(state), rng)))
        for eff in moved.effects:
            self._apply(state, eff, rng, action.id)
        if moved.damage:
            events.extend(self._apply_damage(state, "player", moved, rng))
        state.current = moved.to_state or state.current
        events.append(GameEvent(
            type="state_change", subject=state.current,
            detail={"from": state.current, "to": state.current,
                    "action": action.id},
        ))
        for custom in moved.events:
            events.append(GameEvent(
                type=custom.get("type", "custom"),
                subject=state.current,
                message=custom.get("message", ""),
                detail=dict(custom.get("detail", {})),
            ))
        return True, events

    # -- 结构化伤害(两侧通用;盾交互在伤害时结算) ------------------------------

    def _apply_damage(self, state: MachineState, side: str, moved, rng: Rng) -> list[GameEvent]:
        """按转移的 damage 规格结算伤害与盾交互。

        damage: {target(默认对方 HP 变量), roll:[base,var] / expr / value,
                 block:[盾种类]=完全格挡(仅溅射 chip), shatter:[盾种类]=击碎(全额),
                 half:[盾种类]=半伤, 三种都消耗盾, chip=溅射值(默认 0)}
        """
        dmg = moved.damage
        if side == "boss":
            target = dmg.get("target", "player_hp")
            shield_var = dmg.get("shield_var", "player_shield")
            label = self._label_of("boss", moved.action)
        else:
            target = dmg.get("target", "boss_hp")
            shield_var = dmg.get("shield_var", "boss_shield")
            label = self._label_of("player", moved.action)

        if "expr" in dmg:
            raw = evaluate(dmg["expr"], self._env(state), rng)
        elif "roll" in dmg:
            base, variance = dmg["roll"]
            raw = base + (rng.randint(0, variance) if variance else 0)
        else:
            raw = int(dmg.get("value", 0))

        shield = state.vars.get(shield_var, "")
        events: list[GameEvent] = []
        blocked_shields = list(dmg.get("block") or ())
        shattered_shields = list(dmg.get("shatter") or ())
        half_shields = list(dmg.get("half") or ())

        if shield in shattered_shields:
            # 击碎:盾崩解,全额伤害
            state.vars[shield_var] = ""
            state.vars[target] = max(0, state.vars.get(target, 0) - raw)
            events.append(GameEvent(
                type="shield_break", subject=label, target=target,
                message=f"{label}击碎了{shield}护盾!(全额 {raw} 伤害)",
                detail={"shield": shield, "by": "attack", "outcome": "shatter",
                        "amount": raw}))
        elif shield in blocked_shields:
            # 完全格挡:只受溅射,盾崩解
            chip = int(dmg.get("chip", 0))
            state.vars[shield_var] = ""
            state.vars[target] = max(0, state.vars.get(target, 0) - chip)
            events.append(GameEvent(
                type="shield_break", subject=label, target=target,
                message=f"{shield}护盾挡下了{label},随后崩解(溅射 {chip} 伤害)",
                detail={"shield": shield, "by": "attack", "outcome": "blocked",
                        "amount": chip}))
        elif shield in half_shields:
            # 半伤:盾消耗
            raw = (raw + 1) // 2
            state.vars[shield_var] = ""
            state.vars[target] = max(0, state.vars.get(target, 0) - raw)
            events.append(GameEvent(
                type="half_blocked", subject=label, target=target, amount=raw,
                message=f"{shield}盾被{label}打穿,只挡住一半(受 {raw} 伤害)",
                detail={"shield": shield, "outcome": "half", "amount": raw}))
        else:
            state.vars[target] = max(0, state.vars.get(target, 0) - raw)
        # 命中(或半伤/溅射)都有伤害事实
        if raw > 0 or shield:
            events.append(GameEvent(
                type="damage", subject=label, target=target, amount=raw,
                message=f"{label}造成 {raw} 点伤害",
                detail={"outcome": "hit", "amount": raw}))
        return events

    def _label_of(self, side: str, action_id: str) -> str:
        """动作的中文名(事件文案用)。"""
        if side == "boss" and self.skin.boss is not None:
            try:
                return self.skin.boss.action(action_id).label
            except KeyError:
                return action_id
        for a in self.skin.actions:
            if a.id == action_id:
                return a.label
        return action_id

    # -- 条件与效果求值 ---------------------------------------------------------

    def _check(self, vars: dict, cond: dict, rng: Rng | None = None) -> bool:
        if "expr" in cond:
            # 终局条件等无 rng 上下文的求值应保持纯确定(勿在条件里用 roll)
            stream = rng if rng is not None else self._status_probe()
            return bool(evaluate(cond["expr"], vars, stream))
        op = cond["op"]
        if op not in CONDITION_OPS:
            raise ValueError(f"unknown condition op: {op}")
        left = vars.get(cond["var"])
        right = cond["value"]
        return {"==": left == right, "!=": left != right, ">=": left >= right,
                "<=": left <= right, ">": left > right, "<": left < right}[op]

    def _apply(self, state: MachineState, eff: dict, rng: Rng,
               action_id: str = "") -> None:
        env = self._env(state)
        if eff.get("op") == "expr":
            state.vars[eff["var"]] = evaluate(eff["expr"], env, rng)
            return
        op, var = eff["op"], eff["var"]
        # add/sub 可携带表达式(如 {"op":"add","expr":"-roll(12,4)"})
        value = evaluate(eff["expr"], env, rng) if "expr" in eff else eff["value"]
        if op not in EFFECT_OPS:
            raise ValueError(f"unknown effect op: {op}")
        if op == "set":
            state.vars[var] = value
        elif op == "add":
            state.vars[var] = state.vars.get(var, 0) + value
        else:
            state.vars[var] = state.vars.get(var, 0) - value


__all__ = ["MachineRuntime", "MachineState"]
