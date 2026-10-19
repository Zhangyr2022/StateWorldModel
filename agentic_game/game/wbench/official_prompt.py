"""实验 D 的裸官方 prompt 编译器:控制视频 + WBench 官方 prompt(2026-09-24)。

prompt 族(用户定稿,逐字):
    <首帧锚定句> <Video 1> 场景描述句> <官方 build_turn_prompt 句>
- 锚定句与 <Picture 1>/<Video 1> 引用句置首(H3 官方指南 "Write the
  first-frame instruction first";与已验证的 reference overlay 同位),
  官方动作句随后——与裸 ref2va 消融(h3-ref2va,锚定句首行)同约定。
- <Video 1> 句逐字取自用户定稿(tp)/reference overlay 的 fp 等价句
  (fp previz 无蓝盒,蓝盒句对 fp 事实上不成立;两版均止于
  "no other objects",不带 {previz_motion}/跟随句/反残留句——那正是
  本实验要隔离的"框架 prompt 工程")。
- 无控制视频回合(非 navigation/词表外/生产失败降级)输出 = 锚定句 +
  官方句,与裸 ref2va 消融的 prompt 逐字节一致。

排版职责在本模块(实验专属 prompt 族);VisualScript 结构化字段仍由
director 填写,官方句经 script.meta["official_prompt"] 传递。
"""
from __future__ import annotations

from engine.visual.compiler import PromptCompiler
from engine.visual.script import VisualScript

FIRST_FRAME_ANCHOR = (
    "<Picture 1> is the exact first frame of the target video: begin the "
    "video on <Picture 1>, reproducing it faithfully, and keep its style "
    "and lighting for the whole shot."
)

# 用户定稿 tp 句(= engine/visual/reference.py _TP_OPENING 去掉尾随的
# " — {previz_motion}" 槽);fp 句 = _FP_OPENING 同规则等价句
TP_VIDEO_SENTENCE = (
    "<Video 1> is the motion and camera reference: a minimal previz "
    "containing only a blue box standing in for <Subject 1> on an infinite "
    "light-gray ground with a 1-meter grid and two static red reference "
    "boxes near the subject — no other objects."
)
FP_VIDEO_SENTENCE = (
    "<Video 1> is the motion and camera reference: a minimal first-person "
    "previz with only an infinite light-gray ground with a 1-meter grid "
    "and two static red reference boxes ahead — no other objects."
)


class OfficialRef2VACompiler(PromptCompiler):
    """实验 D:meta["official_prompt"] + 首帧锚定 + 可选 <Video 1> 句。

    刻意不走父类 NEWSKILL 路径——本实验的变量就是 prompt 族本身。
    """

    def compile(self, script: VisualScript, *, has_first_frame: bool,
                has_reference_video: bool = False) -> str:
        parts = [FIRST_FRAME_ANCHOR]
        if has_reference_video and script.reference is not None:
            parts.append(
                TP_VIDEO_SENTENCE if script.reference.perspective == "third_person"
                else FP_VIDEO_SENTENCE)
        official = str(script.meta.get("official_prompt", "")).strip()
        if official:
            parts.append(official)
        prompt = " ".join(parts)
        if not has_first_frame:
            # 与 engine.visual.reference.degrade_first_frame 同规则
            prompt = (prompt.replace("<Picture 1>", "the scene")
                            .replace("<Video 1>", "the reference video"))
        return prompt
