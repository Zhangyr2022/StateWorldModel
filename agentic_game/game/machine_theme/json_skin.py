"""JSON 皮肤加载器(状态机类):一个 .json 定义全部状态、转移与演出 prompt。

命令行自检:
    python3 -m game.machine_theme.json_skin game/skins/my_machine.json

字段规范见 AGENTS.md「Skin JSON reference」。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from game.machine_theme.expr import compile_expr
from game.machine_theme.skin import (
    BossActionSpec,
    BossSide,
    BossStateSpec,
    CONDITION_OPS,
    EFFECT_OPS,
    ActionSpec,
    EndingSpec,
    MachineSkin,
    StateSpec,
    TransitionSpec,
)
from game.skin_validation import SkinValidationError

_TURN_STEPS = ("player", "boss")


def _err(path: str, why: str) -> SkinValidationError:
    return SkinValidationError(f"[{path}] {why}")


def _need(data: dict, key: str, path: str):
    if key not in data or data[key] in (None, ""):
        raise _err(f"{path}.{key}", "缺少必填字段")
    return data[key]


def _str_tuple(value, path: str) -> tuple:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _err(path, "必须是字符串数组")
    return tuple(value)


def _conditions(items, path: str) -> tuple:
    out = []
    for i, cond in enumerate(items or ()):
        if "expr" in cond:
            compile_expr(str(cond["expr"]))  # 语法/白名单校验
            out.append({"expr": str(cond["expr"])})
            continue
        if cond.get("op") not in CONDITION_OPS:
            raise _err(f"{path}[{i}].op",
                       f"必须是 {', '.join(CONDITION_OPS)} 之一")
        if not cond.get("var"):
            raise _err(f"{path}[{i}].var", "缺少变量名")
        out.append(dict(cond))
    return tuple(out)


def _effects(items, path: str) -> tuple:
    out = []
    for i, eff in enumerate(items or ()):
        if eff.get("op") == "expr":
            if not eff.get("var"):
                raise _err(f"{path}[{i}].var", "缺少变量名")
            compile_expr(str(eff["expr"]))
            out.append({"var": str(eff["var"]), "op": "expr",
                        "expr": str(eff["expr"])})
            continue
        if eff.get("op") not in EFFECT_OPS:
            raise _err(f"{path}[{i}].op",
                       f"必须是 {', '.join(EFFECT_OPS)} 之一或 expr")
        if not eff.get("var"):
            raise _err(f"{path}[{i}].var", "缺少变量名")
        out.append(dict(eff))
    return tuple(out)


def skin_from_dict(data: dict) -> MachineSkin:
    """加载并归一化:玩家侧字段可写在顶层(旧格式)或 "player" 节(新格式,
    与 boss 节对称)。归一后统一从 player_data 读取。"""
    if not isinstance(data, dict):
        raise _err("$", "顶层必须是 JSON 对象")

    player = data.get("player")
    if isinstance(player, dict):
        player_data = {**data, **player}   # 顶层公共字段 + player 节覆盖
        if "states" not in player:
            raise _err("player.states", "player 节必须声明 states")
        if "actions" not in player:
            raise _err("player.actions", "player 节必须声明 actions")
        if "transitions" not in player:
            raise _err("player.transitions", "player 节必须声明 transitions")
    else:
        player_data = data   # 旧格式:玩家侧字段在顶层

    vars_data = data.get("vars") or {}
    if not isinstance(vars_data, dict) or not all(
            isinstance(k, str) for k in vars_data):
        raise _err("vars", "必须是对象: 变量名 -> 初始值")

    states_data = _need(player_data, "states", "player")
    if not isinstance(states_data, dict) or not states_data:
        raise _err("states", "必须是非空对象: 状态id -> 状态描述")
    states = {}
    for state_id, node in states_data.items():
        if not isinstance(node, dict) or not node.get("label"):
            raise _err(f"states.{state_id}", "必须是有 label 的对象")
        states[state_id] = StateSpec(
            id=state_id,
            label=str(node["label"]),
            action=str(node.get("action", "")),
            end_state=str(node.get("end_state", "")),
            effects=_str_tuple(node.get("effects"), f"states.{state_id}.effects"),
            actions=_str_tuple(node.get("actions"), f"states.{state_id}.actions"),
        )

    initial = str(_need(player_data, "initial_state", "player"))
    if initial not in states:
        raise _err("initial_state", f"必须是 states 中的 id,当前: {initial}")

    actions = []
    for i, item in enumerate(_need(player_data, "actions", "player")):
        outcomes = item.get("outcomes") or {}
        if not isinstance(outcomes, dict) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in outcomes.items()):
            raise _err(f"actions[{i}].outcomes", "必须是对象: 结果档位 -> 英文演出短语")
        actions.append(ActionSpec(
            id=str(_need(item, "id", f"actions[{i}]")),
            label=str(_need(item, "label", f"actions[{i}]")),
            description=str(item.get("description", "")),
            phrase=str(item.get("phrase", "")),
            outcomes={str(k): str(v) for k, v in outcomes.items()},
            effects=_str_tuple(item.get("effects"), f"actions[{i}].effects"),
        ))
    known_actions = {a.id for a in actions}

    transitions = []
    for i, item in enumerate(_need(player_data, "transitions", "player")):
        from_state = str(_need(item, "from", f"transitions[{i}]"))
        action_id = str(_need(item, "action", f"transitions[{i}]"))
        to_state = str(item.get("to", ""))  # 空 = 原地停留
        if from_state not in ("*",) and from_state not in states:
            raise _err(f"transitions[{i}].from", f"未知状态 {from_state}")
        if to_state and to_state not in states:
            raise _err(f"transitions[{i}].to", f"未知状态 {to_state}")
        if action_id not in known_actions:
            raise _err(f"transitions[{i}].action", f"未知动作 {action_id}")
        pre = str(item.get("pre", ""))
        if pre:
            compile_expr(pre)
        outcome = str(item.get("outcome", ""))
        outcome_expr = str(item.get("outcome_expr", ""))
        if outcome_expr:
            compile_expr(outcome_expr)
        if outcome and outcome_expr:
            raise _err(f"transitions[{i}]", "outcome 与 outcome_expr 只能二选一")
        prob = float(item.get("probability", 1.0))
        if not 0.0 <= prob <= 1.0:
            raise _err(f"transitions[{i}].probability", f"必须在 0~1,当前 {prob}")
        custom_events = []
        for j, ev in enumerate(item.get("events") or ()):
            custom_events.append({
                "type": str(_need(ev, "type", f"transitions[{i}].events[{j}]")),
                "message": str(ev.get("message", "")),
                "detail": dict(ev.get("detail", {})),
            })
        damage = item.get("damage") or {}
        if not isinstance(damage, dict):
            raise _err(f"transitions[{i}].damage", "必须是对象")
        if "expr" in damage:
            compile_expr(str(damage["expr"]))
        elif "roll" in damage and not (
                isinstance(damage["roll"], list) and len(damage["roll"]) == 2):
            raise _err(f"transitions[{i}].damage.roll", "必须是 [base, variance]")
        transitions.append(TransitionSpec(
            from_state=from_state,
            action=action_id,
            to_state=to_state,
            conditions=_conditions(item.get("conditions"), f"transitions[{i}].conditions"),
            probability=prob,
            effects=_effects(item.get("effects"), f"transitions[{i}].effects"),
            events=tuple(custom_events),
            pre=pre,
            outcome=outcome,
            outcome_expr=outcome_expr,
            damage=dict(damage),
        ))

    endings = {}
    for status in ("won", "lost"):
        item = (data.get("endings") or {}).get(status)
        if item is None:
            continue
        endings[status] = EndingSpec(
            status=status,
            conditions=_conditions(item.get("condition"), f"endings.{status}.condition"),
            action=str(item.get("action", "")),
            effects=_str_tuple(item.get("effects"), f"endings.{status}.effects"),
            end_state=str(item.get("end_state", "")),
            caption=str(item.get("caption", status)),
        )

    turn_flow = tuple(data.get("turn_flow") or ("player",))
    for step in turn_flow:
        if step not in _TURN_STEPS:
            raise _err("turn_flow", f"未知阶段 {step!r}(允许: {', '.join(_TURN_STEPS)})")

    boss = None
    boss_data = data.get("boss")
    if boss_data is not None:
        if not isinstance(boss_data, dict):
            raise _err("boss", "必须是对象")
        boss_states = {}
        for state_id, node in (boss_data.get("states") or {}).items():
            if not isinstance(node, dict) or not node.get("label"):
                raise _err(f"boss.states.{state_id}", "必须是有 label 的对象")
            boss_states[state_id] = BossStateSpec(
                id=state_id,
                label=str(node["label"]),
                phrase=str(node.get("phrase", "")),
                end_state=str(node.get("end_state", "")),
                effects=_str_tuple(node.get("effects"), f"boss.states.{state_id}.effects"),
            )
        boss_initial = str(boss_data.get("initial_state", ""))
        if boss_states and boss_initial not in boss_states:
            raise _err("boss.initial_state",
                       f"必须是 boss.states 中的 id,当前: {boss_initial}")

        boss_actions = []
        for i, item in enumerate(boss_data.get("actions") or ()):
            path = f"boss.actions[{i}]"
            pre = str(item.get("pre", ""))
            if pre:
                compile_expr(pre)
            weight = float(item.get("weight", 1.0))
            if weight < 0:
                raise _err(f"{path}.weight", "权重不能为负")
            boss_actions.append(BossActionSpec(
                id=str(_need(item, "id", path)),
                label=str(_need(item, "label", path)),
                weight=weight,
                pre=pre,
                description=str(item.get("description", "")),
                phrase=str(item.get("phrase", "")),
            ))

        boss_transitions = []
        for i, item in enumerate(boss_data.get("transitions") or ()):
            path = f"boss.transitions[{i}]"
            from_state = str(_need(item, "from", path))
            action_id = str(_need(item, "action", path))
            to_state = str(item.get("to", ""))
            if boss_states and from_state not in ("*",) and from_state not in boss_states:
                raise _err(f"{path}.from", f"未知 Boss 状态 {from_state}")
            if to_state and to_state not in boss_states:
                raise _err(f"{path}.to", f"未知 Boss 状态 {to_state}")
            if boss_actions and action_id not in {a.id for a in boss_actions}:
                raise _err(f"{path}.action", f"未知 Boss 动作 {action_id}")
            prob = float(item.get("probability", 1.0))
            if not 0.0 <= prob <= 1.0:
                raise _err(f"{path}.probability", f"必须在 0~1,当前 {prob}")
            custom_events = []
            for j, ev in enumerate(item.get("events") or ()):
                custom_events.append({
                    "type": str(_need(ev, "type", f"{path}.events[{j}]")),
                    "message": str(ev.get("message", "")),
                    "detail": dict(ev.get("detail", {})),
                })
            damage = item.get("damage") or {}
            if not isinstance(damage, dict):
                raise _err(f"{path}.damage", "必须是对象")
            if "expr" in damage:
                compile_expr(str(damage["expr"]))
            elif "roll" in damage and not (
                    isinstance(damage["roll"], list) and len(damage["roll"]) == 2):
                raise _err(f"{path}.damage.roll", "必须是 [base, variance]")
            boss_transitions.append(TransitionSpec(
                from_state=from_state,
                action=action_id,
                to_state=to_state,
                conditions=_conditions(item.get("conditions"), f"{path}.conditions"),
                probability=prob,
                effects=_effects(item.get("effects"), f"{path}.effects"),
                events=tuple(custom_events),
                damage=dict(damage),
            ))

        boss = BossSide(
            name=str(_need(boss_data, "name", "boss")),
            label=str(_need(boss_data, "label", "boss")),
            initial_state=boss_initial,
            states=boss_states,
            actions=tuple(boss_actions),
            transitions=tuple(boss_transitions),
        )

    if "phase_split" in data:
        raw = data["phase_split"]
        if (not isinstance(raw, list) or len(raw) != 4
                or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in raw)):
            raise _err("phase_split", "必须是 4 个 0~1 的数(四段时长占比,和为 1)")
        if abs(sum(raw) - 1.0) > 0.01:
            raise _err("phase_split", f"四段占比之和应为 1,当前 {sum(raw):.2f}")

    return MachineSkin(
        game_id=str(_need(data, "game_id", "$")),
        frame=str(_need(data, "frame", "$")),
        player_name=str(_need(data, "player_name", "$")),
        player_label=str(_need(data, "player_label", "$")),
        player_appearance=str(_need(data, "player_appearance", "$")),
        scene_description=str(_need(data, "scene_description", "$")),
        style=str(data.get("style", "")),
        camera=str(data.get("camera", "")),
        soundscape=str(data.get("soundscape", "")),
        music=str(data.get("music", "")),
        first_frame_anchor=str(data.get("first_frame_anchor", "")),
        stability=_str_tuple(data.get("stability"), "stability"),
        closing_state=str(data.get("closing_state", "")),
        initial_state=initial,
        states=states,
        actions=tuple(actions),
        transitions=tuple(transitions),
        vars=dict(vars_data),
        mp_name=str(data.get("mp_name", "MP")),
        mp_max=int(data.get("mp_max", 0)),
        var_display=_str_tuple(data.get("var_display"), "var_display"),
        endings=endings,
        turn_flow=turn_flow,
        boss=boss,
        phase_split=tuple(float(x) for x in data.get("phase_split")
                          or (0.25, 0.35, 0.2, 0.2)),
        scene_lines=tuple(
            (str(item["expr"]), str(item["text"]))
            for item in data.get("scene_lines", [])
        ),
        scene_phases=tuple(
            (str(item["expr"]), str(item["scene"]))
            for item in data.get("scene_phases", [])
        ),
    )


def load_json_skin(path: str | Path) -> MachineSkin:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkinValidationError(
            f"{path}: 不是合法 JSON(第 {exc.lineno} 行: {exc.msg})") from exc
    return skin_from_dict(data)


def main(argv: list[str]) -> int:
    if not argv:
        print("用法: python3 -m game.machine_theme.json_skin <skin.json> [...]")
        return 2
    failed = 0
    for arg in argv:
        try:
            skin = load_json_skin(arg)
        except SkinValidationError as exc:
            print(f"✗ {arg}\n  {exc}")
            failed += 1
            continue
        print(f"✓ {arg}: {skin.game_id} — {len(skin.states)} 状态 / "
              f"{len(skin.transitions)} 转移 / {len(skin.actions)} 动作")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
