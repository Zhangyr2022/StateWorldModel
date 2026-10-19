"""PromptCompiler:VisualScript → Minimax-H3 指令(论文 §7 / 贡献 4)。

Semantic Bridge 的最后一跳:整个系统里唯一允许把视觉语义排版成
视频模型指令的地方(见 AGENTS.md——prompt 组织集中在基础层,
游戏代码只提供结构化描述,不允许散落 prompt 片段)。

格式完全遵循两份规范:
- h3-a800 官方 prompt 指南(随 MiniMax-H3 权重分发,本仓不含其文本):
  带首帧(I2VA)时第一行是首帧对齐指令,随后空一行;单 [Shot 1];
  三个核心字段 integrated_multimodal_description / overall_soundscape /
  non_diegetic_music;
- engine/visual/NEWSKILL.md(提示词标准化器):
  描述内按"主体→场景连续性→时间线→镜头→风格→对白→特效→防错约束→
  收尾状态"的顺序组织;对白用稳定说话者 ID + <d>[语言] …</d> 逐字标签;
  overall_soundscape ≤ 4 句、non_diegetic_music ≤ 3 句;稳定性约束渲染为
  描述末句。句数与说话者 ID 由 engine.visual.standard.validate_standard
  机器检查,违规直接拒绝编译。

Reference overlay(2026-09-24 融合定稿):系统只有一个 prompt 族。当回合
实际携带控制视频(has_reference_video=True,script.reference 非 None 且
生产成功)时,在**同一 NEWSKILL 骨架**上叠加 overlay——对齐行换成
<Picture 1>+<Video 1> 语义、主体句换成跟随句、末尾追加反残留句
(engine/visual/reference)。无控制视频时不叠加,prompt 与基线逐字节
一致(消融只隔离控制视频一个变量)。
"""

from __future__ import annotations

from engine.visual.reference import (
    reference_actor_sentences,
    reference_opening,
    reference_residue_sentence,
)
from engine.visual.standard import render_stability, validate_standard
from engine.visual.script import VisualScript

# I2VA 首帧对齐指令(指南 §2.1),必须是 prompt 第一行,后接一个空行。
_I2VA_ALIGNMENT_LINE = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)

_DESCRIPTION_HEADER = "integrated_multimodal_description: [Shot 1]"


class PromptCompiler:
    """把 VisualScript 确定性地编译成一条视频生成指令。"""

    def compile(self, script: VisualScript, *, has_first_frame: bool,
                has_reference_video: bool = False) -> str:
        self._validate(script)
        if has_reference_video and script.reference is None:
            raise ValueError(
                "has_reference_video=True requires script.reference (MotionReference)"
            )
        # NEWSKILL 可机器检查的规则:句数上限、说话者 ID 唯一性。
        problems = validate_standard(script)
        if problems:
            raise ValueError("VisualScript violates the H3 prompt standard: " + "; ".join(problems))

        description = self._description(script, with_reference=has_reference_video)
        if has_reference_video:
            opening = reference_opening(script.reference)
        else:
            opening = _I2VA_ALIGNMENT_LINE

        lines = [
            opening,
            "",
            description,
            f"overall_soundscape: {script.soundscape or 'N/A'}",
            f"non_diegetic_music: {script.music or 'N/A'}",
        ]
        prompt = "\n".join(lines)
        if not has_first_frame:
            # T2VA(指南 §2.1):无首帧时默认路径删对齐行;reference 路径
            # 保留(还要承载 <Video 1> 句),只降级 <Picture 1> 引用。
            if not has_reference_video:
                prompt = prompt.replace(_I2VA_ALIGNMENT_LINE + "\n\n", "")
            prompt = prompt.replace("<Picture 1>", "the scene")
        return prompt

    def _description(self, script: VisualScript, *, with_reference: bool) -> str:
        """NEWSKILL 描述:风格 → 首帧锚定 → 镜头 → 动作 → 反应 → 对白 →
        特效 → 多镜头 → 防错约束 → 收尾状态;with_reference 时主体句换成
        控制视频跟随句、末尾追加反残留句(其余字段与基线完全一致)。

        动作/反应文本由游戏层写成完整短语(含必要的 toward 等介词),
        基础层只负责拼接顺序与格式,不自动补写内容。
        """
        parts: list[str] = []
        if script.style:
            parts.append(_sentence(script.style))
        if script.phases:
            # 时间轴路径:全局句(风格 → 首帧锚定 → 镜头)在前,每个时段一行
            # "At 00:00.000 - 00:03.000, ...";细节(主体/反应)已在段内,
            # 防错约束与收尾状态照 NEWSKILL 顺序渲染在时间轴之后。
            if script.first_frame_anchor:
                parts.append(_sentence(script.first_frame_anchor))
            if script.camera:
                parts.append(_sentence(script.camera))
            for phase in script.phases:
                text = phase.text.rstrip(".!?")
                parts.append(f"At {_phase_range(phase, script.duration_s)}, {text}.")
            stability_line = render_stability(script.stability)
            if stability_line:
                parts.append(_sentence(stability_line))
            if with_reference:
                parts.append(_sentence(reference_residue_sentence(script.reference)))
            if script.end_state:
                parts.append(_sentence(f"The clip ends as {script.end_state}"))
            return " ".join(parts)
        if script.first_frame_anchor:
            parts.append(_sentence(script.first_frame_anchor))
        actor_text = reference_actor_sentences(script) if with_reference else None
        if script.camera:
            camera_sentence = _sentence(script.camera)
            # control 的官方动作句全文(official_action)已含同款镜头句时
            # 不再单独渲染,避免重复
            if actor_text is None or camera_sentence.rstrip(".") not in actor_text:
                parts.append(camera_sentence)
        if actor_text is not None:
            # 主体句 = 外观句 + 官方动作句 + 跟随句(<Subject 1> …);
            # 外观/跟随/官方句语义来自游戏层字段,措辞排版在此。
            parts.append(_sentence(actor_text))
        else:
            subject = ", ".join(x for x in (script.actor, script.actor_appearance)
                                if x.strip())
            parts.append(_sentence(f"{subject}, {script.action}"))
        if script.target_reaction:
            # 反应句有 target 时拼成"target + 反应"(如"巨人举起土盾"承接玩家
            # 攻击);没有 target 时反应句自含主语(如盾/治疗回合的巨人阶段),
            # 单独成句,避免把 None 拼进 prompt。
            if script.target:
                parts.append(_sentence(f"{script.target} {script.target_reaction}"))
            else:
                parts.append(_sentence(script.target_reaction))
        # 对白:逐字台词 + 稳定说话者 ID(NEWSKILL 对白规则);台词属于
        # 画面内声音,只出现在描述里,不得重复写进 overall_soundscape。
        for line in script.dialogue:
            parts.append(_sentence(line.render(script.actor)))
        if script.dialogue:
            # 对白紧贴正面约束:台词只以语音存在,压制模型把台词画成字幕的偏置
            parts.append(_sentence("spoken aloud only, never displayed as text"))
        if script.effects:
            parts.append(_sentence(", ".join(script.effects)))
        # 多 shot:后续镜头按指南 §4.2 生成切镜时间戳([Shot 2] At 00:0S.SSS...)
        for index, shot in enumerate(script.shots, start=2):
            parts.append(
                f"[Shot {index}] At {_format_cut_time(shot.cut_at_s, script.duration_s)}, "
                f"the camera cuts to {_sentence(shot.text)}"
            )
        # 防错约束(NEWSKILL 稳定性规则):只约束本场景相关的失败模式,
        # 统一渲染成描述末尾一句话,游戏层不手拼 prompt 文本。
        stability_line = render_stability(script.stability)
        if stability_line:
            parts.append(_sentence(stability_line))
        if script.end_state:
            parts.append(_sentence(f"The clip ends as {script.end_state}"))
        if with_reference:
            # 反残留:控制视频的 previz 几何必须被真实场景替换。
            parts.append(_sentence(reference_residue_sentence(script.reference)))
        return _DESCRIPTION_HEADER + " " + " ".join(parts)

    @staticmethod
    def _validate(script: VisualScript) -> None:
        if not script.action.strip():
            raise ValueError("VisualScript.action is required to compile a prompt")
        if not script.actor.strip():
            raise ValueError("VisualScript.actor is required to compile a prompt")
        # end_state 可为空:皮肤可声明省略收尾句(prompt 极简化),编译器跳过该段
        if "[" in script.action or "]" in script.action:
            # 单 shot 约束:游戏描述里不允许自带 shot 标记;多镜头只能
            # 通过 script.shots 声明,时间戳由编译器生成。
            raise ValueError(
                "VisualScript fields must not contain shot markers ('[...]'); "
                "declare extra shots via VisualScript.shots instead"
            )


def _phase_range(phase, duration_s: int) -> str:
    """渲染 "00:00.000 - 00:03.000" 时间区间(不钳制,设计声明的区间)。"""
    minutes_a, seconds_a = int(phase.start_s // 60), phase.start_s % 60
    minutes_b, seconds_b = int(phase.end_s // 60), phase.end_s % 60
    return (f"{minutes_a:02d}:{seconds_a:06.3f}"
            f" - {minutes_b:02d}:{seconds_b:06.3f}")


def _format_cut_time(cut_at_s: float, duration_s: int) -> str:
    """切镜时间格式化为指南要求的 00:0S.SSS(钳制在片长内,留 0.5s 收尾)。"""
    clamped = max(0.5, min(float(duration_s) - 0.5, float(cut_at_s)))
    minutes = int(clamped // 60)
    seconds = clamped - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def _sentence(text: str) -> str:
    """压掉多余空白、句首大写并确保以句号结尾(拼成连续散文)。"""
    text = " ".join(text.split())
    if text:
        text = text[0].upper() + text[1:]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text
