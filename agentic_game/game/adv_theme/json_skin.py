"""JSON 皮肤加载器(adv 类):设计师交付一个 .json 即可获得一个剧情 ADV 游戏。

用法:
    from game.adv_theme.json_skin import load_json_skin
    skin = load_json_skin("game/skins/my_story.json")

命令行校验:
    python3 -m game.adv_theme.json_skin game/skins/my_story.json

字段约定见本文件 load 代码(逐字段;机制与 AGENTS.md 的 adv 说明一致)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from game.adv_theme.skin import ENDING_STATUSES
from game.adv_theme.skin import AdvSkin, EndingSpec, NodeSpec
from game.skin_validation import SkinValidationError


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


def _nodes_from_list(items, max_turns: int) -> tuple:
    if not isinstance(items, list) or not items:
        raise _err("nodes", "必须是非空对象数组(剧情节点按 start_turn 升序)")
    nodes = []
    repo_root = Path(__file__).resolve().parents[2]
    for i, item in enumerate(items):
        start = item.get("start_turn")
        if not isinstance(start, int) or start < 1:
            raise _err(f"nodes[{i}].start_turn", "必须是正整数")
        if start > max_turns:
            raise _err(f"nodes[{i}].start_turn",
                       f"不能大于 max_turns({max_turns})")
        frame = str(item.get("frame", ""))
        if frame:
            frame_path = Path(frame)
            if not frame_path.is_absolute():
                frame_path = repo_root / frame
            if not frame_path.is_file():
                # 声明了节点首帧但文件缺失必须当场报错:静默降级会让该节点
                # 退化成无图 t2v,整段场景从文本"想象"出来(2026-09-30 教训)
                raise _err(f"nodes[{i}].frame", f"首帧图不存在: {frame}")
        nodes.append(NodeSpec(
            id=str(_need(item, "id", f"nodes[{i}]")),
            label=str(_need(item, "label", f"nodes[{i}]")),
            start_turn=start,
            event=str(_need(item, "event", f"nodes[{i}]")),
            scene=str(item.get("scene", "")),
            frame=str(item.get("frame", "")),
            girl_appearance=str(item.get("girl_appearance", "")),
            style=str(item.get("style", "")),
            arrival=str(item.get("arrival", "")),
            mood_hint=str(item.get("mood_hint", "")),
            camera=str(item.get("camera", "")),
        ))
    if nodes[0].start_turn != 1:
        raise _err("nodes[0].start_turn", "第一个节点必须从第 1 回合开始")
    for prev, curr in zip(nodes, nodes[1:]):
        if curr.start_turn <= prev.start_turn:
            raise _err("nodes", f"start_turn 必须严格递增({prev.id} → {curr.id})")
    return tuple(nodes)


def _endings_from_list(items) -> tuple:
    if not isinstance(items, list) or not items:
        raise _err("endings", "必须是非空对象数组(按判定顺序排列)")
    endings = []
    ids = set()
    for i, item in enumerate(items):
        status = str(_need(item, "status", f"endings[{i}]"))
        if status not in ENDING_STATUSES:
            raise _err(f"endings[{i}].status",
                       f"必须是 {', '.join(ENDING_STATUSES)} 之一")
        ending_id = str(_need(item, "id", f"endings[{i}]"))
        if ending_id in ids:
            raise _err(f"endings[{i}].id", f"结局 id 重复: {ending_id}")
        ids.add(ending_id)
        min_rel = item.get("min_relationship", 0)
        if not isinstance(min_rel, int) or min_rel < 0:
            raise _err(f"endings[{i}].min_relationship", "必须是非负整数")
        endings.append(EndingSpec(
            id=ending_id,
            status=status,
            min_relationship=min_rel,
            requires_intent=bool(item.get("requires_intent", False)),
            caption=str(item.get("caption", "")),
            line=str(item.get("line", "")),
            action=str(item.get("action", "")),
            effects=_str_tuple(item.get("effects"), f"endings[{i}].effects"),
            end_state=str(item.get("end_state", "")),
            soundscape=str(item.get("soundscape", "")),
            music=str(item.get("music", "")),
        ))
    if not any(e.min_relationship <= 0 and not e.requires_intent for e in endings):
        raise _err("endings", "必须提供一条不要求意图且 min_relationship<=0 的兜底结局")
    return tuple(endings)


def skin_from_dict(data: dict) -> AdvSkin:
    if not isinstance(data, dict):
        raise _err("$", "顶层必须是 JSON 对象")

    max_turns = data.get("max_turns", 25)
    if not isinstance(max_turns, int) or max_turns < 1:
        raise _err("max_turns", "必须是正整数")

    moods = _str_tuple(data.get("moods"), "moods")
    if not moods:
        raise _err("moods", "必须是非空字符串数组(LLM 从中选当前心情)")
    mood_actions = data.get("mood_actions") or {}
    if not isinstance(mood_actions, dict):
        raise _err("mood_actions", "必须是对象: 心情 -> 英文演出短语")
    missing = [m for m in moods if not mood_actions.get(m)]
    if missing:
        raise _err("mood_actions", f"缺少心情的演出短语: {', '.join(missing)}")

    rel_min = data.get("relationship_min", 0)
    rel_max = data.get("relationship_max", 100)
    rel_initial = data.get("relationship_initial", 50)
    if not (isinstance(rel_min, int) and isinstance(rel_max, int)
            and isinstance(rel_initial, int)):
        raise _err("relationship_*", "必须是整数")
    if rel_min >= rel_max:
        raise _err("relationship_min/max", "必须满足 relationship_min < relationship_max")
    if not rel_min <= rel_initial <= rel_max:
        raise _err("relationship_initial", "必须落在 [min, max] 内")

    relationship_states = []
    for i, band in enumerate(data.get("relationship_states") or ()):
        if (not isinstance(band, list) or len(band) != 2
                or not isinstance(band[0], int) or not isinstance(band[1], str)):
            raise _err(f"relationship_states[{i}]", "必须是 [阈值(整数), 描述(字符串)]")
        relationship_states.append((band[0], band[1]))
    relationship_states.sort(key=lambda band: band[0], reverse=True)

    nodes = _nodes_from_list(data.get("nodes"), max_turns)
    endings = _endings_from_list(data.get("endings"))

    intent_from_turn = data.get("intent_from_turn", 1)
    if not isinstance(intent_from_turn, int) or intent_from_turn < 1:
        raise _err("intent_from_turn", "必须是正整数")

    omit = _str_tuple(data.get("omit_prompt_sections"), "omit_prompt_sections")
    _VALID_OMIT = {"anchor", "arrival", "end_state"}
    unknown = [o for o in omit if o not in _VALID_OMIT]
    if unknown:
        raise _err("omit_prompt_sections",
                   f"未知段落 {unknown};允许值: {sorted(_VALID_OMIT)}")

    language = str(data.get("language", "zh") or "zh").strip().lower()
    if language not in ("zh", "en"):
        raise _err("language", f"必须是 zh/en 之一,当前 {language!r}")

    return AdvSkin(
        game_id=str(_need(data, "game_id", "$")),
        title=str(data.get("title", "")),
        frame=str(_need(data, "frame", "$")),
        language=language,
        player_name=str(data.get("player_name", "你" if language == "zh" else "You")),
        girl_name=str(_need(data, "girl_name", "$")),
        girl_label=str(_need(data, "girl_label", "$")),
        girl_persona=str(_need(data, "girl_persona", "$")),
        girl_constraints=_str_tuple(data.get("girl_constraints"), "girl_constraints"),
        player_persona=str(data.get("player_persona", "")),
        player_scenario=str(data.get("player_scenario", "")),
        mood_notes={str(k): str(v) for k, v in (data.get("mood_notes") or {}).items()},
        effects=_str_tuple(data.get("effects"), "effects"),
        # 外观锚/场景文本可选:节点级字段才是主承载,全局键仅作兜底(可为空)
        girl_appearance=str(data.get("girl_appearance", "")),
        scene_description=str(data.get("scene_description", "")),
        style=str(data.get("style", "")),
        camera=str(data.get("camera", "")),
        soundscape=str(data.get("soundscape", "")),
        music=str(data.get("music", "")),
        stability=_str_tuple(data.get("stability"), "stability"),
        closing_state=str(data.get("closing_state", "")),
        max_turns=max_turns,
        nodes=nodes,
        relationship_initial=rel_initial,
        relationship_min=rel_min,
        relationship_max=rel_max,
        max_abs_delta=int(data.get("max_abs_delta", 3)),
        show_relationship=bool(data.get("show_relationship", False)),
        relationship_states=tuple(relationship_states),
        moods=moods,
        mood_actions={str(k): str(v) for k, v in mood_actions.items()},
        intent_keywords=_str_tuple(data.get("intent_keywords"), "intent_keywords"),
        intent_from_turn=intent_from_turn,
        endings=endings,
        omit_prompt_sections=omit,
    )


def load_json_skin(path: str | Path) -> AdvSkin:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkinValidationError(f"{path}: 不是合法 JSON(第 {exc.lineno} 行: {exc.msg})") from exc
    return skin_from_dict(data)


def main(argv: list[str]) -> int:
    if not argv:
        print("用法: python3 -m game.adv_theme.json_skin <skin.json> [...]")
        return 2
    failed = 0
    for arg in argv:
        try:
            skin = load_json_skin(arg)
        except SkinValidationError as exc:
            print(f"✗ {arg}\n  {exc}")
            failed += 1
            continue
        print(f"✓ {arg}: {skin.game_id} — {skin.girl_name}"
              f" ({len(skin.nodes)} 个剧情节点, {len(skin.endings)} 个结局,"
              f" 共 {skin.max_turns} 回合)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
