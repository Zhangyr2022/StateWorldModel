"""MachineSkin:状态机类游戏的声明式 Game Definition。

游戏 = 有限状态 + 转移表 + 每状态的视觉 prompt:
- states:状态节点,每个节点声明该状态下的演出模板(action/end_state/effects);
- actions:玩家可做的动作(全部状态通用,可用 states[].actions 收窄);
- transitions:转移表 —— (from, action) 命中后按 conditions 过滤、按 probability
  掷骰,命中即迁移到 to 并执行 effects(修改变量)与 events(自定义事件);
- vars:游戏变量(数值/布尔),被 conditions 读取、effects 修改;
- endings:win/lose 条件与终局动画。

结算完全确定:随机只来自传入的 Rng;resolve_turn 深拷贝后修改。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# conditions/effects 支持的算子(白名单;JSON 校验与 runtime 共用)
CONDITION_OPS = ("==", "!=", ">=", "<=", ">", "<")
EFFECT_OPS = ("set", "add", "sub")


@dataclass(frozen=True)
class StateSpec:
    """一个状态节点:展示名 + 该状态下的演出模板。"""

    id: str
    label: str                 # 中文名(HUD/状态摘要)
    action: str = ""           # 该状态的画面主体动作(英文演出短语)
    end_state: str = ""        # 画面收尾状态(英文)
    effects: tuple = ()        # 固定特效短语组(英文)
    actions: tuple = ()        # 本状态可用的动作 id;空 = 全部可用


@dataclass(frozen=True)
class ActionSpec:
    """一个玩家动作:id 稳定,phrase 为演出短语。"""

    id: str
    label: str
    description: str = ""
    phrase: str = ""           # 执行动作时的英文演出短语(起手段)
    outcomes: dict = field(default_factory=dict)
    # outcomes: {结果档位: 英文演出短语} —— 同一动作不同结算结果的演出分档
    # (如 hit/blocked/shatter);运行时按转移声明的 outcome 选择,未命中用 phrase。
    effects: tuple = ()


@dataclass(frozen=True)
class TransitionSpec:
    """一条转移:from 状态下执行 action 时,满足条件则迁移并执行效果。

    from_state 支持 "*" 通配(所有状态);to_state 为空 = 原地停留。
    effects 支持结构化形式({var, op, value},op 含 set/add/sub)与
    表达式形式({var, op: "expr", expr: "..."},可写 roll/算术/if)。
    pre 为可用性表达式:为假时该 (状态,动作) 视为不可用(从清单隐藏)。
    """

    from_state: str
    action: str
    to_state: str = ""         # "" = 原地停留
    conditions: tuple = ()     # ({var, op, value}, ...);全部满足才可命中
    probability: float = 1.0   # 条件满足时的迁移概率;未命中则尝试下一条同 (from, action) 转移
    effects: tuple = ()        # 结构化效果或 {"var", "op": "expr", "expr"}
    events: tuple = ()         # 迁移时额外发出的自定义事件 ({type, message}, ...)
    pre: str = ""              # 可用性表达式(空 = 恒可用)
    outcome: str = ""          # 结果档位(对齐动作的 outcomes 键;空 = 无档位)
    outcome_expr: str = ""     # 结果档位表达式(如 ifelse(boss_shield, "blocked", "hit"))
    damage: dict = field(default_factory=dict)
    # 结构化伤害: {target(变量,默认 player_hp), roll:[base,var] / expr / value,
    #   block:[盾种类]=完全格挡, half:[盾种类]=半伤, shield_var=盾所在变量,
    #   chip=被完全挡下时的溅射值} —— 盾均消耗;两侧转移通用


@dataclass(frozen=True)
class EndingSpec:
    """终局:结束条件 + 终局动画与字幕。"""

    status: str                # "won" | "lost"
    conditions: tuple = ()
    action: str = ""           # 终局动画短语(英文)
    effects: tuple = ()
    end_state: str = ""
    caption: str = ""          # 回放终局字幕(中文)


@dataclass(frozen=True)
class BossStateSpec:
    """Boss 的一个姿态状态:该姿态下的演出素材(与全局状态轴平行)。"""

    id: str
    label: str                 # 中文名(HUD)
    phrase: str = ""           # 该姿态下的英文演出短语(无专属动作时使用)
    end_state: str = ""        # 英文收尾描述
    effects: tuple = ()


@dataclass(frozen=True)
class BossActionSpec:
    """一个 Boss 动作:选择元数据(权重/前置) + 演出短语。

    与玩家动作对称;结算内容由 boss.transitions 承载。
    """

    id: str
    label: str                 # 中文名(HUD/事件)
    weight: float = 1.0        # 选择权重(pre 通过的动作里按权重掷选)
    pre: str = ""              # 可用性表达式(空 = 恒可用)
    description: str = ""
    phrase: str = ""           # 英文演出短语(Boss 本回合做什么)


@dataclass(frozen=True)
class BossSide:
    """Boss 侧:与玩家对齐的状态轴 + 动作表;动作选择由代码(权重策略)控制。"""

    name: str                  # 中文名(HUD)
    label: str                 # 英文视觉标识(短语占位 {enemy})
    initial_state: str = ""
    states: dict = field(default_factory=dict)   # id -> BossStateSpec
    actions: tuple = ()                          # BossActionSpec,按声明顺序
    transitions: tuple = ()                      # Boss 转移表(结构同玩家转移;
                                                 # from/to 引用 boss.states,
                                                 # action 引用 boss.actions)

    def state(self, state_id: str) -> BossStateSpec:
        return self.states[state_id]

    def action(self, action_id: str) -> BossActionSpec:
        for a in self.actions:
            if a.id == action_id:
                return a
        raise KeyError(action_id)

    def format(self, text: str, player_label: str = "") -> str:
        """{enemy}/{player} 占位符替换(与 DuelSkin.format 同约定)。"""
        return text.format(enemy=self.label, player=player_label or self.label)


@dataclass(frozen=True)
class MachineSkin:
    # -- 身份与场景 --------------------------------------------------------
    game_id: str
    frame: str
    player_name: str
    player_label: str
    player_appearance: str
    scene_description: str
    style: str = ""
    camera: str = ""
    soundscape: str = ""
    music: str = ""
    # 首帧锚定句(可引用 <Picture 1>):与皮肤首帧画面一致的主体/构图描述,
    # 编译器渲染在时间轴之前;空 = 不渲染
    first_frame_anchor: str = ""
    # 防错约束句组:渲染为描述末尾的 "Throughout the clip, ..." 一句
    stability: tuple = ()
    # 全局收尾状态:渲染为 "The clip ends as ...";空 = 回退用各段收尾拼接
    closing_state: str = ""

    # -- 状态机 ------------------------------------------------------------
    # MP 资源显示名(每款皮肤自己的叫法,如"剑气"/"潮息")与上限(0 = 无上限,
    # HUD 回退按初始 MP 封顶);2026-10-14 起现役皮肤统一 mp_max: 15
    mp_name: str = "MP"
    mp_max: int = 0
    initial_state: str = ""
    states: dict = field(default_factory=dict)        # id -> StateSpec
    actions: tuple = ()                               # ActionSpec,按 UI 展示顺序
    transitions: tuple = ()                           # TransitionSpec
    vars: dict = field(default_factory=dict)          # 初始变量表(纯数据)
    var_display: tuple = ()                           # HUD/摘要要展示的变量名;空 = 全部

    # -- 终局 --------------------------------------------------------------
    endings: dict = field(default_factory=dict)       # "won"/"lost" -> EndingSpec

    # -- 回合流水线与 Boss 行为表(可选) ------------------------------------
    # turn_flow: 回合由哪些阶段组成;"boss" 阶段读取 boss_ai 的权重行为表。
    # 行为表项: {id, weight, pre(表达式), set/add 效果, to_state, event};
    #          damage: {target, roll:[base,var] 或 expr, block:[盾种类], half:[盾种类]}
    #          —— 玩家盾在 block 名单=完全格挡(盾消耗),half 名单=半伤(盾消耗)。
    turn_flow: tuple = ("player",)
    boss: BossSide | None = None                      # Boss 侧(状态轴+动作表);无 = 无 Boss

    # -- 演出时段配比(四段:boss 动作 / 玩家动作 / 结算结果 / 双方收尾) ------
    # 按顺序对应 turn_flow 的各阶段 + outcome 段 + end 段;占比和应为 1。
    phase_split: tuple = (0.25, 0.35, 0.2, 0.2)
    # 场景状态句(可选):[(expr, text), ...];导演按当前 vars 求值,
    # 命中的句子注入时间轴首拍,向视频模型声明"场上现在有什么、在哪"
    scene_lines: tuple = ()
    # 场景演进(可选):[(expr, scene_text), ...];表达式按优先级求值,
    # 命中的场景段落与上一回合不同时,本回合脱开首帧链走 t2v 并切场景
    scene_phases: tuple = ()

    def state(self, state_id: str) -> StateSpec:
        return self.states[state_id]

    def format(self, text: str, player_label: str = "") -> str:
        """{player}/{enemy} 占位符替换;enemy 取 Boss 侧定义(无 Boss 保留原文)。"""
        enemy = self.boss.label if self.boss is not None else "{enemy}"
        return text.format(player=player_label or self.player_label,
                           enemy=enemy)

    def action(self, action_id: str) -> ActionSpec:
        for a in self.actions:
            if a.id == action_id:
                return a
        raise KeyError(action_id)


__all__ = ["ActionSpec", "BossActionSpec", "BossSide", "BossStateSpec",
           "CONDITION_OPS", "EFFECT_OPS", "EndingSpec",
           "MachineSkin", "StateSpec", "TransitionSpec"]
