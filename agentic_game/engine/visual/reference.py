"""Reference overlay(控制视频/ref2va)的句式库:叠加在标准 NEWSKILL 输出上。

设计(与独立 prompt 族的旧方案不同,2026-09-24 定稿):系统只有**一个**
prompt 族——PromptCompiler 的 NEWSKILL 格式。当某回合实际携带控制视频
(has_reference_video)时,在标准输出上叠加本模块的 reference overlay:

  1. 对齐行换成 ref2va 语义:<Picture 1> 首帧锚定 + <Video 1> 运动参考
     (含 previz 场景描述与运动句);
  2. 描述内的主体句换成"主体跟随控制视频"句(tp 用 <Subject 1> 跟随句);
  3. 描述末尾追加反残留句(蓝盒/红柱/网格必须被真实场景替换)。

无控制视频(消融 first_frame、非 navigation 回合、生产失败降级、off 模式)
不叠加任何内容,prompt 与基线逐字节一致——消融只隔离控制视频一个变量。

句式来源:叠加句逐字取自 box2clay wbench_prompt.py 的 flat 模板
(17 条成片验证;ref2va 的 reference 主导性极强,反残留必须逐元素显式)。
"""

from __future__ import annotations

from engine.visual.script import MotionReference, VisualScript

# <Picture 1> 首帧锚定 + <Video 1> 运动参考(替换默认 I2VA 对齐行)
_TP_OPENING = (
    "<Picture 1> is the exact first frame of the target video: begin the video on "
    "<Picture 1>, reproducing it faithfully, and keep its style and lighting for the "
    "whole shot. <Video 1> is the motion and camera reference: a minimal previz "
    "containing only a blue box standing in for <Subject 1> on an infinite "
    "light-gray ground with a 1-meter grid and two static red reference boxes near "
    "the subject — no other objects — {previz_motion}"
)
_FP_OPENING = (
    "<Picture 1> is the exact first frame of the target video: begin the video on "
    "<Picture 1>, reproducing it faithfully, and keep its style and lighting for the "
    "whole shot. <Video 1> is the motion and camera reference: a minimal first-person "
    "previz with only an infinite light-gray ground with a 1-meter grid and two static "
    "red reference boxes ahead — no other objects — {previz_motion}"
)

# tp 主体跟随句对(appearance 槽位 = VisualScript.actor_appearance;
# official_action = 官方动作句全文,插在外观句之后)
_TP_FOLLOW = (
    "<Subject 1> is {appearance} in <Picture 1>. {official_action} "
    "<Subject 1> {subject_motion}, "
    "while keeping the exact appearance and screen size from <Picture 1>."
)
# fp 相机跟随句(相机即主体,无 <Subject 1>):官方动作句 + 跟随语义,
# 拼成一句("The camera pushes forward, follows the movement, …")。
# {fp_motion} = MotionReference.previz_motion(官方 FIRST_PERSON_ACTIONS 原文,去尾句号)
_FP_FOLLOW = (
    "{fp_motion}, follows the movement, framing and timing of <Video 1> "
    "exactly, in continuous first-person view."
)
# previz_motion 缺失时的自含主语兜底
_FP_FOLLOW_FALLBACK = (
    "The camera follows the movement, framing and timing of <Video 1> exactly, "
    "in continuous first-person view."
)

# 反残留句(描述末尾;previz 几何不得出现在成片里)
_TP_RESIDUE = (
    "Replace the blue box, the red reference boxes and the empty grid floor with "
    "the real scene from <Picture 1>: do not preserve the blue box's placeholder "
    "appearance, never show the two red reference boxes (they are previz-only "
    "aids), and let no previz geometry remain."
)
_FP_RESIDUE = (
    "Replace the previz grid ground and the two red reference boxes with the real "
    "scene from <Picture 1>: never show the two red reference boxes (they are previz-only "
    "aids), and let no previz geometry remain."
)


def reference_opening(reference: MotionReference) -> str:
    """对齐行:首帧锚定 + <Video 1> 场景/运动说明(带尾句号,单行)。"""
    motion = reference.previz_motion.strip().rstrip(".")
    if reference.perspective == "third_person":
        return _TP_OPENING.format(previz_motion=motion) + "."
    if reference.perspective == "first_person":
        return _FP_OPENING.format(previz_motion=motion) + "."
    raise ValueError(
        f"MotionReference.perspective must be third_person/first_person, "
        f"got {reference.perspective!r}"
    )


def reference_actor_sentences(script: VisualScript) -> str:
    """描述内的主体句(替换标准输出的 actor/appearance/action 句)。

    tp:"<Subject 1> is {外观} ... <Subject 1> {跟随句} ..."——跟随句来自
    MotionReference.subject_motion(控制视频措辞);fp:固定相机跟随句。
    """
    ref = script.reference
    if ref.perspective == "third_person":
        appearance = script.actor_appearance.strip()
        follow = ref.subject_motion.strip()
        if not appearance or not follow:
            raise ValueError(
                "reference overlay (third_person) requires non-empty "
                "actor_appearance and MotionReference.subject_motion"
            )
        official = ref.official_action.strip()
        if official and not official.endswith((".", "!", "?")):
            official += "."
        return _TP_FOLLOW.format(appearance=appearance,
                                 official_action=official,
                                 subject_motion=follow)
    motion = ref.previz_motion.strip().rstrip(".")
    if not motion:
        return _FP_FOLLOW_FALLBACK
    return _FP_FOLLOW.format(fp_motion=motion)


def reference_residue_sentence(reference: MotionReference) -> str:
    """反残留句(描述末尾追加)。"""
    return _TP_RESIDUE if reference.perspective == "third_person" else _FP_RESIDUE


def degrade_first_frame(prompt: str) -> str:
    """use_first_frame=False 时降级 <Picture 1> 引用(与默认路径同规则)。"""
    return prompt.replace("<Picture 1>", "the scene")


__all__ = [
    "degrade_first_frame",
    "reference_actor_sentences",
    "reference_opening",
    "reference_residue_sentence",
]
