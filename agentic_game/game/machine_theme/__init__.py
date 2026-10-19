"""machine 主题模板:状态、转移、演出 prompt 全部来自 JSON,引擎只做解释。

build_machine_game(skin) 把 MachineSkin 组装成 GameParts,注册表以
category="machine" 挂载(本地结算,不需要 LLM)。
"""

from engine.agents import CliAgent
from engine.parts import GameParts
from game.machine_theme.runtime import MachineRuntime
from game.machine_theme.skin import MachineSkin
from game.machine_theme.visual import MachineVisualDirector

# 皮肤未声明终局字幕时的回放缺省(Noto Sans CJK 中有字形;原在已归档的 duel_theme)
ENDING_CAPTIONS = {"won": "◆ 战斗胜利 ◆", "lost": "◆ 战斗失败 ◆"}


def _machine_hud(skin: MachineSkin):
    """回放/界面 HUD 快照:双方血蓝条 + Boss 招式与状态标记。

    与 scripts/rebuild_replay_hud.py 的 hud_machine 同一形状(overlay 面板
    直接消费);那里从 turn_log 快照重建,这里从活状态直接取。
    """

    def hud(state, events=()) -> dict:
        v = state.vars
        init = skin.vars or {}
        p_hp, p_mp = v.get("player_hp", 0), v.get("player_mp", 0)
        p_hp_max = init.get("player_hp") or p_hp or 1
        # MP 上限:皮肤声明 mp_max(现役 15)优先;否则按初始 MP 封顶
        p_mp_max = getattr(skin, "mp_max", 0) or init.get("player_mp") or 1
        mp_label = getattr(skin, "mp_name", "") or "MP"
        b_hp = v.get("boss_hp")
        b_hp_max = init.get("boss_hp") or b_hp or 1
        marks = [text for text, on in (("Boss盾", v.get("boss_shield")),
                                       ("狂暴", v.get("boss_enraged")),
                                       ("蓄力中", v.get("windup"))) if on]
        boss = skin.boss
        boss_action = ""
        if boss is not None and state.last_boss_action:
            try:
                boss_action = boss.action(state.last_boss_action).label
            except KeyError:
                boss_action = str(state.last_boss_action)
        enemy_line = f"Boss 招式:{boss_action}" if boss_action else "Boss 按兵不动"
        if marks:
            enemy_line += f"({','.join(marks)})"
        return {
            "left_name": skin.player_name,
            "left_stats": (min(p_hp / p_hp_max, 1.0) if p_hp_max else 1.0,
                           min(p_mp / p_mp_max, 1.0) if p_mp_max else 1.0,
                           f"{p_hp} / {p_hp_max}",
                           f"{mp_label} {p_mp} / {p_mp_max}"),
            "right_name": boss.name if boss is not None else "Boss",
            "right_stats": (min(b_hp / b_hp_max, 1.0) if b_hp_max else 0.0,
                            f"{b_hp} / {b_hp_max}"),
            "enemy_action": enemy_line,
        }

    return hud


def build_machine_game(skin: MachineSkin) -> GameParts:
    """按皮肤组装全部游戏部件(机制本地结算,无需 LLM 服务)。"""
    runtime = MachineRuntime(skin)
    director = MachineVisualDirector(skin)
    captions = {status: ending.caption
                for status, ending in skin.endings.items() if ending.caption}
    return GameParts(
        runtime=runtime,
        visual_director=director.build,
        visual_state=director.initial_visual_state(),
        ending_director=director.ending_script,
        frame=skin.frame,
        describe_state=runtime.describe_state,
        hud_of=_machine_hud(skin),
        ending_captions=captions or dict(ENDING_CAPTIONS),
        cli_agent_factory=lambda _llm=None: CliAgent(describe_state=runtime.describe_state),
    )


__all__ = ["MachineSkin", "MachineRuntime", "MachineVisualDirector",
           "build_machine_game"]
