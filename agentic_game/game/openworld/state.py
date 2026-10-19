"""开放世界探索 Demo 的 Game State。

pending_actions(VLM 提议的行动)与本回合的模式判定(pending_mode)是
游戏事实的一部分,随状态被携带;而视频采样帧等表现层产物放在 runtime
的运行期缓存里,不进 Game State。

回合模式(INTERACTIVE/AUTO 双模式):
- INTERACTIVE:当前存在有意义的行动机会,展示三个选项并保留自由输入;
- AUTO:画面缺乏有效交互机会时,AI 自动推进一幕,不弹出选择;
- 硬性规则:AUTO 之后的下一回合必须是 INTERACTIVE,不得连续 AUTO。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 回合模式取值(游戏层共享词汇;规则层结算、表现层演出都按它分流)
MODE_INTERACTIVE = "interactive"  # 交互回合:展示行动选项,重大决定属于玩家
MODE_AUTO = "auto"                # 自动探索回合:AI 推进一幕,尽量创造下一决策点


@dataclass
class ExplorationState:
    scene_id: str = "western"
    scene_name: str = "边陲小镇"
    # VLM 提议的玩家行动(每回合刷新,3 个左右;AUTO 回合为空列表)
    pending_actions: list[str] = field(default_factory=list)
    # 本回合提议时判定的模式(available_actions 生成,resolve_turn 消费清空)
    pending_mode: str = ""
    # VLM 对"是否有意义行动机会"的判断说明(英文短句;AUTO 行动的展示理由)
    mode_note: str = ""
    # 上一回合的模式(""=尚未开始;为 AUTO 时下一回合强制 INTERACTIVE)
    last_mode: str = ""
    # 最近一次行动的后果(英文画面描述,由导演 VLM 产出,直接喂给视频模型)
    last_consequence: str = ""
    # 剧情旁白(中文),按幕累积
    story: list[str] = field(default_factory=list)
    # 玩家行动历史 [(行动, 后果)]
    history: list[tuple[str, str]] = field(default_factory=list)
    # VLM 构思的结局旁白(plot_complete 时写入)
    climax_text: str = ""
    # 主线任务进度:当前完成到的阶段下标(下一阶段 = quest_stage + 1)
    quest_stage: int = 0
    # 主线节奏只数交互幕(docs §7:AUTO 产生视频幕数,但不替玩家推进主线)
    interactive_turns: int = 0  # 全局交互幕数(结局最早幕数的判定依据)
    stage_interactive: int = 0  # 当前主线阶段内的交互幕数(阶段推进时清零)
    # 体力:每幕行动消耗;清空后进入力竭(只能缓慢走动,见 rules)
    stamina: int = 40
    exhausted_turns: int = 0
    turn: int = 0
    ending: str = ""

    def summary(self) -> str:
        from game.openworld.quests import quest_for

        scene = f"[{self.scene_name}]"
        quest = quest_for(self.scene_id)
        stage = min(self.quest_stage, quest.total - 1)
        stamina_note = f"体力 {self.stamina}(力竭剩余 {self.exhausted_turns} 幕)" \
            if self.exhausted_turns else f"体力 {self.stamina}"
        quest_line = (f"主线《{quest.title}》({stage + 1}/{quest.total}):"
                      f"{quest.stage_text(stage)} | {stamina_note}")
        if self.last_mode == MODE_AUTO:
            quest_line += " | 上幕自动探索"
        if self.story:
            return f"第{self.turn}幕 {scene} {quest_line} | {self.story[-1]}"
        return f"第{self.turn}幕 {scene} {quest_line} | 冒险刚刚开始。"
