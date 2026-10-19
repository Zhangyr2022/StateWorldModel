"""H3 prompt 标准(NEWSKILL)的结构化实现:对白、声画上限与稳定性约束。

规范原文见 engine/visual/NEWSKILL.md(Minimax H3 提示词标准化器)。本模块
把规范中可机器检查/可确定性渲染的部分落成代码,供 PromptCompiler 使用:

- **对白**:说话者用稳定 ID((S1) / (S2)…),台词逐字放进
  ``<d>[语言] …</d>``,说话者身份写在标签外;画外音必须用
  "says in an off-screen voiceover" 并紧接 "while ... lips remain
  completely closed"(口型同步约束)。
- **声画上限**:overall_soundscape ≤ 4 句、non_diegetic_music ≤ 3 句;
  对白/画面内音乐不得重复出现在 soundscape 里。
- **稳定性约束(防错约束)**:只添加与场景相关的限制(身份漂移、服装
  变化、动作瞬移、未经要求的屏幕文字等),作为描述的最后一句渲染。

游戏层只声明结构化字段(VisualScript.dialogue / VisualScript.stability),
拼接格式集中在这里——与 PromptCompiler 的分工一致:游戏写"说什么",
基础层定"怎么排"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# overall_soundscape / non_diegetic_music 的句数上限(NEWSKILL 声画规则)
MAX_SOUNDSCAPE_SENTENCES = 4
MAX_MUSIC_SENTENCES = 3

# 画外音的固定措辞与口型同步约束(NEWSKILL 对白规则)
_OFF_SCREEN_CUE = "says in an off-screen voiceover"
_OFF_SCREEN_LIPS = "while their lips remain completely closed"

_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")


@dataclass(frozen=True)
class Dialogue:
    """一句台词:稳定说话者 ID + 逐字文本(不得在游戏层改写/翻译)。

    speaker_id 形如 "S1";同一说话者跨镜头保持同一 ID。on_screen=False
    表示画外旁白,渲染时自动加口型同步约束。
    """

    speaker_id: str
    language: str  # "Chinese" / "English" / ...
    text: str
    cue: str = "says"  # 说话方式: says / shouts / whispers / says softly ...
    on_screen: bool = True

    def render(self, actor: str) -> str:
        """按 NEWSKILL 格式渲染一句对白;actor 是说话者的视觉身份描述。"""
        if not self.text.strip():
            raise ValueError("dialogue text must be verbatim and non-empty")
        cue = self.cue.strip() or "says"
        if not self.on_screen:
            cue = _OFF_SCREEN_CUE
        line = f"{actor} ({self.speaker_id}) {cue}: <d>[{self.language}] {self.text.strip()}</d>"
        if not self.on_screen:
            line += f" {_OFF_SCREEN_LIPS}"
        return line


def count_sentences(text: str) -> int:
    """按句末标点统计句数(NEWSKILL 的 1–4 句 / 1–3 句上限用)。"""
    text = text.strip()
    if not text:
        return 0
    parts = [p for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return len(parts) or 1


def validate_standard(script) -> list[str]:
    """检查一条 VisualScript 是否符合 NEWSKILL 的可机器检查规则。

    返回违规描述列表(空列表 = 合规)。对白文本非空是硬约束,
    由 Dialogue.render 自己抛错;这里检查声画句数与说话者 ID 唯一性。
    """
    problems: list[str] = []
    soundscape_sentences = count_sentences(script.soundscape)
    if soundscape_sentences > MAX_SOUNDSCAPE_SENTENCES:
        problems.append(
            f"overall_soundscape has {soundscape_sentences} sentences "
            f"(max {MAX_SOUNDSCAPE_SENTENCES})"
        )
    music_sentences = count_sentences(script.music)
    if music_sentences > MAX_MUSIC_SENTENCES:
        problems.append(
            f"non_diegetic_music has {music_sentences} sentences "
            f"(max {MAX_MUSIC_SENTENCES})"
        )
    speaker_ids = [d.speaker_id for d in script.dialogue]
    if len(set(speaker_ids)) != len(speaker_ids):
        problems.append(f"speaker ids must be unique per clip, got {speaker_ids}")
    return problems


def render_stability(stability) -> str:
    """把稳定性约束元组渲染成描述末尾的一句话(无约束时返回空串)。

    条目末尾的句号统一剥离(连接符与末句句号负责标点),避免 ".;"/".." 残渣。
    """
    items = [item.strip().rstrip(".") for item in stability if item.strip()]
    if not items:
        return ""
    return "Throughout the clip, " + "; ".join(items) + "."


__all__ = [
    "Dialogue",
    "MAX_MUSIC_SENTENCES",
    "MAX_SOUNDSCAPE_SENTENCES",
    "count_sentences",
    "render_stability",
    "validate_standard",
]
