"""开放世界的主线任务定义:每个世界一条分阶段主线。

阶段(stage)是导演 VLM 与规则层之间的契约:
- 导演提示词会收到"当前阶段"的描述,要求剧情向该阶段推进;
- 导演判断本段演出已清楚呈现该阶段达成时,输出 stage_complete=true;
- 规则层据此推进 quest_stage(状态转移),全部阶段完成即胜利(结局动画)。

每个阶段描述都要"可在一段约 5 秒视频里被清楚呈现"(易于呈现约束)。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestSpec:
    title: str          # 主线名(中文,用于状态摘要与字幕)
    brief: str          # 一句话背景(注入导演提示词)
    stages: tuple       # 阶段描述(中文,逐段递进,最后一段即终局)

    @property
    def total(self) -> int:
        return len(self.stages)

    def stage_text(self, index: int) -> str:
        return self.stages[min(index, len(self.stages) - 1)]


QUESTS: dict[str, QuestSpec] = {
    "western": QuestSpec(
        title="通缉犯黑荆",
        brief="镇上的通缉犯黑荆抢走了矿场的薪水,你接下了悬赏令",
        stages=(
            "在酒馆里从醉汉口中套听到黑荆藏身峡谷的线索",
            "沿着峡谷马蹄印追踪,发现黑荆的营地篝火",
            "与黑荆在峡谷尽头正面对峙,将他绳之以法",
        ),
    ),
    "anime": QuestSpec(
        title="消失的街头偶像",
        brief="每晚在街头唱歌的偶像少女突然消失,你决定找到她",
        stages=(
            "在公告栏的演出海报前找到她留下的记号",
            "循着记号进入涩谷地下的秘密 Livehouse",
            "登上 Livehouse 的舞台,陪她完成复出演唱会",
        ),
    ),
    "chinese": QuestSpec(
        title="失传的古卷",
        brief="镇上老掌柜托付你寻找一卷失传百年的医书古卷",
        stages=(
            "拜访镇口老掌柜,得到半张藏宝图残页",
            "按残页指引在山中古庙解开石门机关",
            "在古庙深处请出古卷,带回交予老掌柜",
        ),
    ),
    "japan": QuestSpec(
        title="狐狸的谢礼",
        brief="你救过一只白狐,它留下一枚稻荷钱币引你入山",
        stages=(
            "在山脚神社向稻荷神祈愿,钱币泛起微光",
            "循着夜路上的狐火上山,穿过千本鸟居",
            "在山顶神域见到白狐真身,接受它的谢礼",
        ),
    ),
    "roman": QuestSpec(
        title="角斗场的黎明",
        brief="你是一名即将登上大竞技场的角斗士,渴望赢回自由",
        stages=(
            "在武器匠铺取得父亲留下的短剑",
            "在竞技场练兵场完成入场操演,赢得观众欢呼",
            "在大竞技场的正午决斗中胜出,获得自由",
        ),
    ),
    "gta": QuestSpec(
        title="罪恶都市首夜",
        brief="你是初到罪恶之城的年轻人,今晚要在霓虹海岸线站稳脚跟",
        stages=(
            "在滨海大道拦下一辆亮着尾灯的超跑,证明你玩得起这座城市的游戏",
            "甩开警车的追截,把车停进霓虹巷的地下车库",
            "在顶层夜店的天台见到线人,接过属于你的第一单活",
        ),
    ),
}

DEFAULT_WORLD = "western"


def quest_for(world_id: str) -> QuestSpec:
    return QUESTS.get(world_id, QUESTS[DEFAULT_WORLD])


__all__ = ["DEFAULT_WORLD", "QUESTS", "QuestSpec", "quest_for"]
