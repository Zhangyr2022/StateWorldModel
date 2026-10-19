"""adv 主题模板:固定剧情骨架 + 自由输入 + LLM 女主回应,皮肤可注入。

新 adv 类游戏 = 一份 JSON 皮肤(game/skins/<id>.json, category: "adv")
+ 本模板 + 常驻 LLM 服务,engine 零改动。build_adv_game 把皮肤组装成
GameParts,run_demo 与 run_ui(Gradio)都只面向这份契约工作。

与 machine 主题同范式:JSON 皮肤承载全部内容(剧情节点/关系规则/结局/
演出文本),模板只做解释——之后做同类游戏扩展时只需新增一份皮肤。
"""

from __future__ import annotations

from engine.parts import GameParts
from game.adv_theme.agents import AdvCliAgent, AdvLLMAgent
from game.adv_theme.actions import ACTION_TALK
from game.adv_theme.runtime import AdvRuntime, AdvState
from game.adv_theme.skin import AdvSkin, EndingSpec, NodeSpec
from game.adv_theme.visual import AdvVisualDirector

# 皮肤未声明终局字幕时的回放缺省(Noto Sans CJK 中有字形);
# 实际判定到哪个结局后由视觉导演在终局时改写为该结局的 caption。
ENDING_CAPTIONS = {"won": "◆ 故事落幕 ◆", "lost": "◆ 故事落幕 ◆", "draw": "◆ 故事落幕 ◆"}
ENDING_CAPTIONS_EN = {"won": "◆ The Story Ends ◆", "lost": "◆ The Story Ends ◆",
                      "draw": "◆ The Story Ends ◆"}


def _adv_hud(skin: AdvSkin):
    """回放 HUD 快照工厂:右上她的状态,底部玩家台词与关系变化。

    措辞随皮肤语言(zh/en);英文局不再依赖 montage 的 tr() 兜底翻译。
    """

    def hud(state: AdvState, events=()) -> dict:
        delta = 0
        for e in events or []:
            etype = e.get("type") if isinstance(e, dict) else getattr(e, "type", "")
            if etype == "relationship_change":
                amount = (e.get("amount", 0) if isinstance(e, dict)
                          else getattr(e, "amount", 0))
                delta = amount or 0
        node = skin.node_for_turn(max(state.turn, 1))
        rel_visible = skin.show_relationship
        a = state.relationship
        if skin.is_english:
            rel_row = ((f"Relationship {a} / {skin.relationship_max}",)
                       if rel_visible else (f"Relationship: {_rel_word(delta, en=True)}",))
            rows = []
            if rel_visible:
                rows.append(("bar", a / max(1, skin.relationship_max), "relationship",
                             rel_row[0]))
            else:
                rows.append(("text", rel_row[0]))
            rows.append(("location", f"Story: {node.label}"))
            enemy = f"Her mood: {state.mood}"
            if rel_visible and delta:
                enemy += f" (relationship {delta:+d})"
            return {
                "theme": "adv",
                "tag": f"Turn {state.turn} · {node.label}",
                "right_name": state.girl_name,
                "right_rows": rows,
                "action_desc": state.last_player_line,
                "enemy_action": enemy,
            }
        rows = []
        if rel_visible:
            rows.append(("bar", a / max(1, skin.relationship_max), "relationship",
                         f"关系 {a} / {skin.relationship_max}"))
        else:
            rows.append(("text", f"关系:{_rel_word(delta)}"))
        rows.append(("location", f"剧情:{node.label}"))
        return {
            "theme": "adv",
            "tag": f"第{state.turn}回合 · {node.label}",
            "right_name": state.girl_name,
            "right_rows": rows,
            "action_desc": state.last_player_line,
            "enemy_action": (f"她的心情:{state.mood}"
                             + (f"(关系 {delta:+d})" if rel_visible and delta else "")),
        }

    return hud


def _rel_word(delta: int, en: bool = False) -> str:
    if en:
        if delta > 0:
            return "seems a little closer"
        if delta < 0:
            return "seems a little more distant"
        return "unchanged"
    if delta > 0:
        return "似乎更近了一些"
    if delta < 0:
        return "似乎疏远了一些"
    return "维持原样"


def build_adv_game(skin: AdvSkin, llm=None) -> GameParts:
    """按皮肤组装全部游戏部件(llm:engine.llm.LLMClient,常驻服务)。"""
    if llm is None:
        raise ValueError(f"{skin.game_id} 需要常驻 LLM 服务(先启动 engine.llm.service)")
    captions = dict(ENDING_CAPTIONS_EN if skin.is_english else ENDING_CAPTIONS)
    runtime = AdvRuntime(skin, llm)
    director = AdvVisualDirector(skin, captions=captions)
    return GameParts(
        runtime=runtime,
        visual_director=director.build,
        visual_state=director.initial_visual_state(),
        ending_director=director.ending_script,
        frame=skin.frame,
        describe_state=runtime.describe_state,
        free_text_action=ACTION_TALK,
        scene_labels={node.id: node.label for node in skin.nodes},
        ending_captions=captions,
        auto_agent_factory=lambda _llm=None: AdvLLMAgent(llm, skin),
        hud_of=_adv_hud(skin),
        cli_agent_factory=lambda _llm=None: AdvCliAgent(describe_state=runtime.describe_state),
    )


__all__ = [
    "ACTION_TALK",
    "AdvCliAgent",
    "AdvLLMAgent",
    "AdvRuntime",
    "AdvSkin",
    "AdvState",
    "AdvVisualDirector",
    "EndingSpec",
    "NodeSpec",
    "build_adv_game",
]
