"""AdvSkin:剧情 ADV 类游戏的声明式 Game Definition。

《雨停之前》式玩法的模板化(设计说明:本文件与 runtime.py 的 docstring):
固定剧情骨架 + 玩家自由输入 + LLM 女主回应。游戏 = 剧情节点表 + 关系值
规则 + 结局判定表 + 每节点的视觉语义,全部来自 JSON;结算引擎(runtime)
只做解释——节点按回合数推进(时间不由玩家暂停)、关系值由有界 LLM 打分
驱动、结局在最终回合按关系值+意图命中判定。

与 machine 主题同范式:LLM 的打分只是有界输入,规则层集中做夹取与终局
裁决,结算仍确定、有界、可审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 终局状态(engine.runtime.STATUS_* 的子集;结局映射到 won/lost/draw)
ENDING_STATUSES = ("won", "lost", "draw")


@dataclass(frozen=True)
class NodeSpec:
    """一个剧情节点:固定事件 + 该阶段的视觉与心理语义。

    start_turn 为 1-based 起始回合;节点按 start_turn 升序排列,
    "当前节点" = start_turn <= 当前回合的最后一个节点。
    """

    id: str
    label: str                 # 中文名(HUD/回放字幕/状态摘要)
    start_turn: int            # 本节点从第几回合开始
    event: str                 # 【当前剧情】固定事件(注入 LLM prompt,中文)
    scene: str = ""            # 本节点场景文本(与节点首帧图对应;空 = 全局)
    frame: str = ""            # 本节点首帧图(相对仓库根;进入节点时换首帧 i2v,
                               #  空 = 沿用上一段视频末帧链)
    girl_appearance: str = ""  # 本节点外观锚(与首帧图逐像素对应;空 = 全局)
    style: str = ""            # 本节点画风(含该节点的光线/色调;空 = 全局)
    arrival: str = ""          # 进入本节点第一回合的英文演出短语(空 = 无)
    mood_hint: str = ""        # 【当前情绪】补充引导(中文,注入 LLM prompt)
    camera: str = ""           # 节点级机位(空 = 全局)


@dataclass(frozen=True)
class EndingSpec:
    """一个结局:判定条件(关系值阈值 + 是否要求后期恋爱意图)+ 终局演出。

    结局按声明顺序判定,第一条满足的生效;皮肤必须提供一条
    min_relationship<=0 且不要求意图的结局作兜底。
    """

    id: str
    status: str                # "won" | "lost" | "draw"
    min_relationship: int = 0
    requires_intent: bool = False   # True = 还需玩家在 intent_from_turn 后
                                    # 说过 intent_keywords 之一(如明确告白)
    caption: str = ""          # 回放终局字幕(中文)
    line: str = ""             # 结局台词(中文;终局动画里她逐字说出,空 = 无台词)
    action: str = ""           # 终局动画短语(英文;空 = 不生成终局动画)
    effects: tuple = ()
    end_state: str = ""
    soundscape: str = ""
    music: str = ""


@dataclass(frozen=True)
class AdvSkin:
    # -- 身份与场景 --------------------------------------------------------
    game_id: str
    title: str = ""
    frame: str = ""                   # 初始首帧图(委托方提供;缺失时入口 loudly 降级占位图)
    player_name: str = "你"
    girl_name: str = ""
    girl_label: str = ""              # 英文视觉标识(prompt 侧)
    girl_persona: str = ""            # 【角色】人设(注入 LLM system prompt)
    girl_constraints: tuple = ()      # 角色红线(如"不要替玩家说话")
    player_persona: str = ""          # 玩家角色人设(注入玩家代理 system prompt)
    player_scenario: str = ""         # 玩家与女主的共同处境(同上;游戏专属内容)
    mood_notes: dict = field(default_factory=dict)  # 心情 -> 语义注释(注入 prompt)
    effects: tuple = field(default_factory=tuple)   # 装饰性特效短语(注入 H3 prompt)
    girl_appearance: str = ""         # 外观锚(与首帧一致,跨场景锁定身份)
    scene_description: str = ""       # 全局英文场景文本(节点未声明 scene 时兜底)
    style: str = ""                   # 全局画风(节点未声明 style 时兜底)
    camera: str = ""
    soundscape: str = ""
    music: str = ""
    stability: tuple = ()             # 防错约束句组
    closing_state: str = ""           # 全局收尾状态(空 = 用心情收尾兜底)
    language: str = "zh"              # 对白/结算文案语言:"zh"(默认)或 "en";
                                      # 决定 LLM prompt 模板、台词语言标签与
                                      # 事件/摘要措辞(en 皮肤 = 全英文对白局)

    # -- 剧情骨架与回合数 ---------------------------------------------------
    max_turns: int = 25
    nodes: tuple = field(default_factory=tuple)   # NodeSpec,按 start_turn 升序

    # -- 关系系统(隐藏数值) -------------------------------------------------
    relationship_initial: int = 50
    relationship_min: int = 0
    relationship_max: int = 100
    max_abs_delta: int = 3            # 单回合关系值变化上限(设计文档 §15 分档)
    show_relationship: bool = False   # False = 对玩家隐藏数值(设计文档 §15)
    relationship_states: tuple = ()   # [(阈值, 心理状态描述)],按阈值降序;
                                      # 注入【当前关系】,让 LLM 表现出阶段感

    # -- 心情(演出分档;LLM 输出,规则层校验) --------------------------------
    moods: tuple = ()                 # 允许的心情 id
    mood_actions: dict = field(default_factory=dict)   # 心情 -> 英文演出短语

    # -- 意图与结局 ----------------------------------------------------------
    intent_keywords: tuple = ()       # 恋爱意图关键词(玩家台词命中即记录)
    intent_from_turn: int = 1         # 只统计该回合及以后的关键词命中
    endings: tuple = field(default_factory=tuple)  # EndingSpec,按判定顺序

    # -- prompt 极简化开关(可选):省略指定段落 -------------------------------
    # 可选值:"anchor"(首帧锚定句)/ "arrival"(节点过渡句)/ "end_state"(收尾句)
    # 省略 anchor 时场景内容完全由首帧图承载(i2v);省略 end_state 时 prompt
    # 不再有 "The clip ends as …" 收尾。
    omit_prompt_sections: tuple = field(default_factory=tuple)

    def omits(self, section: str) -> bool:
        return section in self.omit_prompt_sections

    @property
    def is_english(self) -> bool:
        return self.language.lower().startswith("en")

    # -- 查询 -----------------------------------------------------------------

    def node_for_turn(self, turn: int) -> NodeSpec:
        """当前回合所在节点:start_turn <= turn 的最后一个节点。"""
        current = self.nodes[0]
        for node in self.nodes:
            if node.start_turn <= turn:
                current = node
            else:
                break
        return current

    def relationship_state(self, relationship: int) -> str:
        """当前关系值对应的心理状态描述(取阈值 <= 关系值的最高档)。"""
        for threshold, text in self.relationship_states:
            if relationship >= threshold:
                return text
        return ""

    def select_ending(self, relationship: int, intent_hit: bool) -> EndingSpec:
        """按声明顺序取第一条满足条件的结局(皮肤兜底结局保证必有结果)。"""
        for ending in self.endings:
            if relationship < ending.min_relationship:
                continue
            if ending.requires_intent and not intent_hit:
                continue
            return ending
        return self.endings[-1]

    def ending_by_id(self, ending_id: str) -> EndingSpec | None:
        for ending in self.endings:
            if ending.id == ending_id:
                return ending
        return None


__all__ = ["AdvSkin", "EndingSpec", "NodeSpec", "ENDING_STATUSES"]
