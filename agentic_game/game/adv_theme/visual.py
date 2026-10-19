"""adv 主题模板的视觉映射:TurnResult → VisualScript,皮肤注入。

剧情 ADV 的演出是"她的回应":固定场景里的对话演出(台词经 NEWSKILL
对白规范逐字渲染),心情决定肢体短语,进入新剧情节点(场景文本变化)
时脱开首帧链走 t2v——同一位角色(外观锚不变)出现在新阶段的时间流里,
渲染出的最后一帧成为新阶段的 i2v 链起点。终局动画按结局 id 演出。

prompt 语法由 PromptCompiler 统一负责,本模块只填结构化字段。
"""

from __future__ import annotations

from engine.runtime import TurnResult
from engine.visual.script import VisualScript
from engine.visual.standard import Dialogue
from engine.visual.state import VisualState
from game.adv_theme.runtime import MOOD_CUES
from game.adv_theme.skin import AdvSkin, EndingSpec

DURATION_S = 5


class AdvVisualDirector:
    """一个 adv 皮肤对应一个视觉映射器实例。"""

    def __init__(self, skin: AdvSkin, captions: dict | None = None) -> None:
        self.skin = skin
        # 回放终局字幕(状态 → 中文),按实际判定的结局在终局时改写;
        # build_summary_montage 在整局结束后才读它,因此可以动态更新。
        self.captions = captions if captions is not None else {}
        self._last_ending_id = ""     # 终局回合由 build 记录,供 ending_script 用

    def initial_visual_state(self) -> VisualState:
        return VisualState(
            scene=self.skin.scene_description,
            style=self.skin.style,
            appearances={self.skin.girl_name: self.skin.girl_appearance},
        )

    # -- 主组装 ---------------------------------------------------------------

    def build(self, turn_result: TurnResult, visual_state: VisualState) -> VisualScript:
        skin = self.skin
        state = turn_result.state_after
        node = skin.node_for_turn(max(state.turn, 1))
        scene_text = node.scene or skin.scene_description
        scene_changed = bool(scene_text) and scene_text != visual_state.scene
        if scene_text:
            visual_state.scene = scene_text
        visual_state.meta["node"] = node.id

        node_entered = any(e.type == "node_event" for e in turn_result.events)
        # 第 1 回合不触发 node_event(initial_state 已预设首节点),但首帧图
        # (parts.frame)就是该节点的 frame——同样按"节点首帧"处理
        node_first = state.turn == node.start_turn

        # 节点首帧:进入节点(或本回合即节点首回合)把首帧链切到该节点的图
        # (i2v 锚定新场景)——以节点进入为触发,而不是场景文本变化:皮肤不
        # 声明 scene 文本时也必须能换帧。末帧链渲染成功后由引擎自然接续。
        node_frame = None
        if (node_entered or node_first) and node.frame:
            from engine.parts import resolve_frame
            node_frame = resolve_frame(node.frame)
            if node_frame is not None and node_frame.is_file():
                visual_state.current_frame = node_frame
        mood = state.mood if state.mood in skin.mood_actions else (
            skin.moods[0] if skin.moods else "")
        # 神态演出:一律取心情档位短语(皮肤 mood_actions;
        # LLM 只决定回应/心情/打分,不生成动作——2026-10-11 定稿)
        body = (state.last_action or "").strip() or skin.mood_actions.get(mood, "")
        mood_reaction = body + (
            "; her lips move in clear speech as she says her reply"
            if state.last_reply else "")

        # 台词:结算层产出的逐字回应(语言随皮肤),经对白规范渲染为
        # <d>[Chinese|English] …</d>
        dialogue: tuple[Dialogue, ...] = ()
        reply = (state.last_reply or "").strip()
        if reply:
            cue = MOOD_CUES.get(state.mood, "says softly")
            dialogue = (Dialogue(speaker_id="S1",
                                 language="English" if skin.is_english else "Chinese",
                                 text=reply, cue=cue),)

        parts = []
        if node_entered and node.arrival and not skin.omits("arrival"):
            parts.append(f"{node.arrival}, {skin.girl_label} shown in <Picture 1>")
        if mood_reaction:
            parts.append(mood_reaction)
        action = "; ".join(parts)

        if state.ending_id:
            self._last_ending_id = state.ending_id

        frame_switched = node_frame is not None
        # prompt 极简化:皮肤可声明省略锚定句/收尾句(场景内容完全交给首帧图)
        anchor = ("" if skin.omits("anchor")
                  else self._anchor(visual_state, scene_changed,
                                    frame_switched, node))
        end_state = ("" if skin.omits("end_state")
                     else skin.closing_state or self._default_end_state(state, mood))
        return VisualScript(
            style=node.style or skin.style,
            first_frame_anchor=anchor,
            camera=node.camera or skin.camera,
            actor=skin.girl_label,
            actor_appearance=node.girl_appearance or skin.girl_appearance,
            action=action,
            dialogue=dialogue,
            stability=skin.stability or (
                "she keeps the same face, hairstyle and school uniform",
                "no on-screen text or subtitle appears",
            ),
            effects=skin.effects,
            end_state=end_state,
            soundscape=skin.soundscape,
            music=skin.music,
            meta={"node": node.id, "turn": state.turn},
            # 换节点且有真图 → i2v 锚定新图;换节点无图 → 退回 t2v(旧行为)
            use_first_frame=not scene_changed or frame_switched,
            duration_s=DURATION_S,
        )

    def ending_script(self, status: str,
                      visual_state: VisualState) -> VisualScript | None:
        """终局动画:按实际判定的结局(而非仅 won/lost 二值)演出。"""
        skin = self.skin
        ending = self._ending_for(status)
        if ending is None or not ending.action:
            return None
        if ending.caption:
            self.captions[status] = ending.caption
        # 结局台词:皮肤逐字规定(2026-10-12 定稿),终局动画里她照稿说出
        dialogue: tuple[Dialogue, ...] = ()
        if ending.line:
            dialogue = (Dialogue(speaker_id="S1",
                                 language="English" if skin.is_english else "Chinese",
                                 text=ending.line, cue="says softly"),)
        return VisualScript(
            style=skin.style,
            first_frame_anchor=self._anchor(
                visual_state, scene_changed=False, frame_switched=False,
                node=self.skin.node_for_turn(self.skin.max_turns)),
            camera=(f"The camera stays locked-off with no cut, holding still "
                    f"as the scene settles"),
            actor=skin.girl_label,
            actor_appearance=skin.girl_appearance,
            action=ending.action,
            dialogue=dialogue,
            stability=skin.stability or (
                "she keeps the same face, hairstyle and school uniform",
                "no on-screen text or subtitle appears",
            ),
            effects=ending.effects,
            end_state=ending.end_state,
            soundscape=ending.soundscape or skin.soundscape,
            music=ending.music or skin.music,
            meta={"ending": ending.id},
            duration_s=DURATION_S,
        )

    # -- 内部 ------------------------------------------------------------------

    def _ending_for(self, status: str) -> EndingSpec | None:
        """终局回合记录的结局 id 优先;否则按状态取第一条匹配结局。"""
        skin = self.skin
        ending = skin.ending_by_id(self._last_ending_id)
        if ending is not None and ending.status == status:
            return ending
        for candidate in skin.endings:
            if candidate.status == status:
                return candidate
        return None

    def _anchor(self, visual_state: VisualState, scene_changed: bool,
                frame_switched: bool, node) -> str:
        """锚定句一律由当前节点的 appearance + scene 构建(与节点首帧图对应),
        空字段自动略去;first_frame_anchor 字段已废弃,同名键会被忽略。"""
        skin = self.skin
        appearance = node.girl_appearance or skin.girl_appearance
        scene_text = node.scene or skin.scene_description
        if scene_changed and not frame_switched:
            # 换节点但该节点没有首帧图:退回 t2v,只声明"同一人、时间前进"
            return (f"the same girl from earlier ({appearance}), "
                    "unchanged in appearance and outfit, in the same place as "
                    "time has moved forward")
        parts = [f"{skin.girl_label} shown in <Picture 1>"]
        if appearance.strip():
            parts.append(appearance.strip())
        if scene_text.strip():
            parts.append(scene_text.strip())
        return ", ".join(parts)

    @staticmethod
    def _default_end_state(state, mood: str) -> str:
        return (f"she settles back with a {mood or 'calm'} expression, "
                "still in the same spot, the camera locked-off and everything "
                "else in the scene unchanged")


__all__ = ["AdvVisualDirector", "DURATION_S"]
