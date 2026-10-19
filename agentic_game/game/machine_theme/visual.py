"""machine 主题模板的视觉导演:回合事实 → 两段式 VisualScript。

两段式演出(2026-09-30 定稿;2026-10-08 按 gallery 时代行为恢复——10-04
曾短暂改为一拍一句的四拍,复现对比见 generated/duel_gallery_rerun/):
Boss/玩家动作不再各自占一条时间轴,结算与收尾也不再分段——只渲染两条
"At ..." 时间轴:
  [行动段]     turn_flow 顺序的各方动作合并(如 ["boss","player"] →
               Boss 短语在前、玩家短语在后,同一时段内先后发生);
  [结算收尾段] 结果演出(命中/被挡/击碎……)+ 双方收尾合并。
时长按 skin.phase_split 前两段之和 / 后两段之和分配(和为 1,默认
0.25+0.35=0.6 / 0.2+0.2=0.4 → 00:00-00:06 / 00:06-00:10);
全局字段(style/first_frame_anchor/camera/soundscape/music/stability/
closing_state)由 PromptCompiler 按 NEWSKILL 顺序排在时间轴前后。
注意:VisualScript.first_frame_anchor 恒取 skin.first_frame_anchor——v8
曾让场景段落文本占用 anchor 槽,导致 duel 皮肤的锚定句被顶出 prompt,
场景文本现只服务 scene_changed → use_first_frame 的切景机制。
"""

from __future__ import annotations

from engine.runtime import TurnResult
from engine.visual.script import Phase, VisualScript
from engine.visual.state import VisualState
from game.machine_theme.skin import MachineSkin

DURATION_S = 10.0  # 一个回合的演出时长(秒);phase_split 按时长占比切分(与 duel_gallery 成片一致)


_PRONOUN_STARTS = ("its ", "his ", "her ", "their ")


def _with_subject(subject: str, text: str) -> str:
    """给裸动词短语补主语;补不了的(独立成句)确保句首大写。

    短语已以主语开头时不重复加;以物主代词开头(如 "its chest ...")时也
    不加——代词回指前句刚出现的主语,补了反而成病句。只有这两种独立成句
    情况才大写首字母;正常补主语时保持小写,句首大写交给 PromptCompiler。
    """
    stripped = " ".join(text.split())
    standalone = (not subject or stripped.lower().startswith(subject.lower())
                  or stripped.lower().startswith(_PRONOUN_STARTS))
    if standalone:
        if stripped:
            stripped = stripped[0].upper() + stripped[1:]
        return stripped
    return f"{subject} {stripped}"


class MachineVisualDirector:
    """实现 VisualDirector 协议((TurnResult, VisualState) -> VisualScript)。"""

    def __init__(self, skin: MachineSkin) -> None:
        self.skin = skin

    def initial_visual_state(self) -> VisualState:
        return VisualState(scene=self.skin.scene_description)

    # -- 主组装 ---------------------------------------------------------------

    def build(self, turn_result: TurnResult, visual_state: VisualState) -> VisualScript:
        state = turn_result.state_after
        node = self.skin.state(state.current)
        spec = self.skin.action(turn_result.action.id)
        boss = self.skin.boss

        # 行动段:turn_flow 顺序的各方动作合并进同一时段(空文本跳过);
        # 每段都带主语(H3 对裸动词短语容易张冠李戴):玩家段 = 身份 + 外观,
        # Boss 段 = 英文视觉标识。
        # 场景状态句:向视频模型声明"现在场上有几只、站在哪、坑里有什么",
        # 防止数量/位置漂移(复数措辞会让模型脑补尸潮)
        from game.machine_theme import expr as _expr
        import random as _random
        _rng = _random.Random(0)
        try:
            svars = dict(state.vars)
        except AttributeError:
            svars = {}
        # 场景演进:表达式选中的场景段落与上一回合不同 → 本回合脱开首帧链
        # (t2v 渲染新场景),并在 montage 里自动插场景转场
        scene_text = self.skin.scene_description
        for phase_expr, phase_scene in self.skin.scene_phases:
            try:
                if _expr.evaluate(phase_expr, svars, _rng):
                    scene_text = phase_scene
                    break
            except Exception:
                continue
        scene_changed = bool(scene_text) and scene_text != visual_state.scene

        inventory_texts = []
        for line_expr, line_text in self.skin.scene_lines:
            try:
                if _expr.evaluate(line_expr, svars, _rng):
                    inventory_texts.append(
                        line_text.format(**{k: str(v) for k, v in svars.items()}))
            except Exception:
                continue

        action_texts: list[str] = []
        for step in self.skin.turn_flow:
            if step == "boss":
                phrase = self._boss_phrase(state)
                if phrase:
                    action_texts.append(_with_subject(
                        f"{boss.label} shown in <Picture 1>",
                        boss.format(phrase, self.skin.player_label)))
            elif step == "player":
                phrase = spec.phrase or node.action
                action_texts.append(_with_subject(
                    f"{self.skin.player_label} shown in <Picture 1>, "
                    f"{self.skin.player_appearance},",
                    self.skin.format(phrase)))

        # 结算结果:同一动作,不同结果 = 不同演出(命中/被挡/击碎……)
        outcome_text = ""
        if state.last_outcome:
            outcome_phrase = spec.outcomes.get(state.last_outcome)
            if outcome_phrase:
                outcome_text = self.skin.format(outcome_phrase)

        # 双方收尾:场景节点收尾 + Boss 姿态收尾(各自带主语)
        labeled_ends = [_with_subject(
            f"{self.skin.player_label} shown in <Picture 1>",
            self.skin.format(node.end_state))]
        if boss is not None:
            posture = boss.states.get(state.boss_state)
            if posture is not None and posture.end_state:
                labeled_ends.append(_with_subject(
                    f"{boss.label} shown in <Picture 1>",
                    self.skin.format(posture.end_state,
                                     self.skin.player_label)))
        end_text = ". ".join(labeled_ends)

        # 两拍时间线(gallery 时代行为,2026-10-08 恢复):
        #   [行动段]     turn_flow 顺序的各方动作合并(如 ["boss","player"]
        #               → Boss 短语在前、玩家短语在后,同一时段内先后发生);
        #   [结算收尾段] 结果演出(命中/被挡/击碎……)+ 双方收尾合并。
        # 时长按 skin.phase_split 前两段之和 / 后两段之和分配(默认 0.6/0.4)。
        split = self.skin.phase_split
        segments = [
            (split[0] + split[1], "; ".join(inventory_texts + action_texts)),
            (split[2] + split[3],
             "; ".join(t for t in (outcome_text, end_text) if t)),
        ]

        present = [seg for seg in segments if seg[1]]  # 空段不占时间轴
        script = VisualScript(
            style=self.skin.style,
            actor=self.skin.player_label,
            actor_appearance=self.skin.player_appearance,
            action="; ".join(text for _, text in present),   # 全回合摘要
            end_state=self.skin.closing_state or end_text,
            effects=self._combined_effects(state, spec, boss),
            camera=self.skin.camera,
            soundscape=self.skin.soundscape,
            music=self.skin.music,
            first_frame_anchor=self.skin.first_frame_anchor or scene_text,
            stability=self.skin.stability,
            duration_s=DURATION_S,
            phases=self._build_phases(present),
        )
        if scene_changed:
            script.use_first_frame = False
            visual_state.scene = scene_text
            script.meta["scene"] = f"scene-{abs(hash(scene_text)) % 10000}"
        script.meta["turn_id"] = turn_result.turn_id
        script.meta["machine_state"] = state.current
        if state.boss_state:
            script.meta["boss_state"] = state.boss_state
        return script

    def ending_script(self, status: str, visual_state: VisualState) -> VisualScript | None:
        ending = self.skin.endings.get(status)
        if ending is None or not ending.action:
            return None
        return VisualScript(
            style=self.skin.style,
            actor=self.skin.player_label,
            actor_appearance=self.skin.player_appearance,
            action=self.skin.format(ending.action),
            end_state=self.skin.format(ending.end_state),
            effects=tuple(self.skin.format(e) for e in ending.effects),
            camera=self.skin.camera,
            soundscape=self.skin.soundscape,
            music=self.skin.music,
            duration_s=DURATION_S,
            meta={"ending": status},
        )

    # -- 内部工具 --------------------------------------------------------------

    def _boss_phrase(self, state) -> str:
        """Boss 本回合动作的演出短语;无记录时回落当前姿态短语。"""
        boss = self.skin.boss
        if boss is None:
            return ""
        try:
            return boss.action(state.last_boss_action).phrase
        except KeyError:
            posture = boss.states.get(state.boss_state)
            return posture.phrase if posture is not None else ""

    def _combined_effects(self, state, spec, boss) -> tuple:
        """场景节点特效 + 玩家动作特效 + Boss 姿态特效。"""
        parts: list[str] = []
        node = self.skin.state(state.current)
        parts += [self.skin.format(e) for e in node.effects]
        parts += [self.skin.format(e) for e in spec.effects]
        if boss is not None:
            posture = boss.states.get(state.boss_state)
            if posture is not None:
                parts += [self.skin.format(e) for e in posture.effects]
        return tuple(parts)

    @staticmethod
    def _build_phases(segments) -> tuple[Phase, ...]:
        """按各段权重把非空段落铺到时长上(调用方已滤掉空段)。"""
        phases = []
        cursor = 0.0
        for weight, text in segments:
            span = DURATION_S * weight
            phases.append(Phase(start_s=round(cursor, 3),
                                end_s=round(cursor + span, 3), text=text))
            cursor += span
        return tuple(phases)


__all__ = ["MachineVisualDirector"]
