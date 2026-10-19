"""VisualScript:游戏语义 → 视频语义之间的结构化桥梁(论文 §7 / 贡献 3、4)。

它是 Agent-Mediated Interface 的数据面:游戏层(或游戏内的 LLM/VLM 导演)
只填这份结构化表示,永不接触 prompt 语法;格式知识全部集中在基础层
(PromptCompiler + engine/visual/standard.py),从而让 Renderer 与游戏
代码彻底解耦(Training-Free Semantic Rendering)。

设计约束(见 AGENTS.md「Engine engineering rules」):只描述"结果"——谁、做了什么、
对谁、发生什么、结尾什么状态;不做镜头语言 DSL 与动画 Timeline。

字段按 engine/visual/NEWSKILL.md(h3 官方 prompt 规范)组织(游戏每回合
一段连续演出):
- style:               画面风格,如 "3D CG, cinematic, dark fantasy game"
- first_frame_anchor:  与首帧画面一致的锚点描述(主体/构图/场景),
                       I2VA 要求先锚定首帧再展开动作;可自然引用 <Picture 1>
- camera:              镜头运动,按规范词表写成自然英文句(类型+幅度+速度)
- actor/action/target: 这一拍"谁做了什么、对谁"
- target_reaction:     目标的反应
- dialogue:            逐字台词(稳定说话者 ID + <d> 标签,见 standard.Dialogue)
- effects:             特效短语
- stability:           防错约束(身份/服装/位置不变等,描述末尾统一渲染)
- end_state:           这段视频结尾必须落到的状态(下一段视频的首帧由此而来)
- soundscape/music:    环境声与非剧情音乐(模型同时生成音频;句数有上限)

多 shot:默认整段只有一个 [Shot 1](单镜头是游戏演出的默认约束);
开放世界等需要硬切镜头的游戏通过 shots 字段显式声明后续镜头
(切镜时间 + 内容),由 PromptCompiler 按规范格式生成时间戳。

meta:游戏私有的产出期元数据(如当前场景 id),随 TurnRecord 落盘,
供回放拼装等表现层使用;engine 不解释其内容。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.visual.standard import Dialogue


@dataclass(frozen=True)
class Shot:
    """一个后续镜头:cut_at_s 为相对本段视频开头的切镜时间(秒)。"""

    cut_at_s: float
    text: str


@dataclass(frozen=True)
class MotionReference:
    """控制视频(白模 previz)参考声明:本回合的运动/镜头由一段控制视频承载。

    这是 Reference-Guided Realization 的数据面:游戏层只声明"要哪个动作的
    控制视频 + 控制视频里发生了什么 + 主体应如何跟随",不写 prompt 语法;
    控制视频的生产(检测/白模渲染)由游戏层注入的 reference_producer 完成,
    prompt 排版由基础层(PromptCompiler + engine.visual.reference)完成——
    职责分工与既有约定一致(见 AGENTS.md「Engine engineering rules」)。

    字段:
    - perspective:    "third_person" | "first_person"(决定 overlay 句式与
                       previz 渲染器;tp=蓝盒场景,fp=相机即主体)
    - action_token:   原始动作 token("W" / "W+A" / "left" ...),生产方
                       据此选择/生成控制视频文件
    - previz_motion:  控制视频内发生的运动描述句(进入 prompt 的 <Video 1> 句)
    - subject_motion: 主体跟随句(tp:"moves forward exactly as the blue box
                       does, matching ..."——控制视频措辞,主体跟随语义);
                       fp 为空(相机即主体)
    - official_action: 官方动作句全文(tp 用;WBench THIRD_PERSON_ACTIONS[token],
                       如 "The subject moves forward. The camera follows,
                       maintaining distance and framing.")。插在外观句之后,
                       使 control prompt 同样携带基线的官方动作措辞;其中
                       的镜头句与 camera 字段去重。fp 为空(fp 主体句直接用
                       官方动作句拼接,见 reference._FP_FOLLOW)
    """

    perspective: str
    action_token: str
    previz_motion: str
    subject_motion: str = ""
    official_action: str = ""


@dataclass(frozen=True)
class Phase:
    """一个带时间区间的演出段:[start_s, end_s) 秒内发生 text 描述的内容。"""

    start_s: float
    end_s: float
    text: str


@dataclass
class VisualScript:
    """把一个 Turn 表现成一段约 5 秒视频所需的最小描述。"""

    style: str = ""
    first_frame_anchor: str = ""
    camera: str = ""
    actor: str = ""
    actor_appearance: str = ""
    action: str = ""
    target: str | None = None
    target_reaction: str | None = None
    dialogue: tuple[Dialogue, ...] = field(default_factory=tuple)
    effects: tuple[str, ...] = field(default_factory=tuple)
    stability: tuple[str, ...] = field(default_factory=tuple)
    end_state: str = ""
    soundscape: str = ""
    music: str = ""
    shots: tuple[Shot, ...] = field(default_factory=tuple)
    phases: tuple[Phase, ...] = field(default_factory=tuple)
    meta: dict = field(default_factory=dict)
    # 是否用上一段视频的最后一帧作首帧(默认 True,i2v 连续链)。
    # 场景切换等"从文本重新开始"的回合设为 False(t2v),渲染产出的
    # 最后一帧自然成为新场景的 i2v 链起点。
    use_first_frame: bool = True
    duration_s: int = 5
    # 控制视频参考声明:None = 本回合不带控制视频(prompt 即标准
    # NEWSKILL 输出)。声明且生产成功时,prompt 在标准输出上叠加
    # reference overlay(锚定句 + <Video 1> 跟随句 + 反残留,见
    # engine.visual.reference)——同一 prompt 族,非独立分支。
    reference: MotionReference | None = None

    def to_dict(self) -> dict:
        return {
            "style": self.style,
            "first_frame_anchor": self.first_frame_anchor,
            "camera": self.camera,
            "actor": self.actor,
            "actor_appearance": self.actor_appearance,
            "action": self.action,
            "target": self.target,
            "target_reaction": self.target_reaction,
            "dialogue": [
                {
                    "speaker_id": d.speaker_id,
                    "language": d.language,
                    "text": d.text,
                    "cue": d.cue,
                    "on_screen": d.on_screen,
                }
                for d in self.dialogue
            ],
            "effects": list(self.effects),
            "stability": list(self.stability),
            "end_state": self.end_state,
            "soundscape": self.soundscape,
            "music": self.music,
            "shots": [{"cut_at_s": shot.cut_at_s, "text": shot.text} for shot in self.shots],
            "meta": dict(self.meta),
            "use_first_frame": self.use_first_frame,
            "duration_s": self.duration_s,
            "reference": None if self.reference is None else {
                "perspective": self.reference.perspective,
                "action_token": self.reference.action_token,
                "previz_motion": self.reference.previz_motion,
                "subject_motion": self.reference.subject_motion,
                "official_action": self.reference.official_action,
            },
        }
