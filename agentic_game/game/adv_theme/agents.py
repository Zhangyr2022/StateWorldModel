"""adv 类游戏的玩家代理:命令行输入,或由 LLM 扮演男主角。

两种方式都只产出 Action("talk", <台词>),结算层不关心台词来源。
"""

from __future__ import annotations

from engine.llm.client import LLMClient
from engine.runtime import Action
from game.adv_theme.actions import ACTION_TALK
from game.adv_theme.runtime import build_history_messages
from game.adv_theme.skin import AdvSkin

# 玩家代理 system prompt 通用框架:游戏专属内容(男主角人设/共同处境)
# 由皮肤字段 player_persona / player_scenario 拼接注入(2026-10-12 拆分)
PLAYER_SYSTEM_TEMPLATE = (
    "你在一款剧情向恋爱 ADV 中扮演男主角:{player_persona}。"
    "{scenario}"
    "对话历史里的「(剧情·节点名)」旁白是按回合发生的固定剧情,"
    "你的台词必须接着最新剧情与{girl_name}说的话继续,不要提尚未发生的剧情。"
    "请自然地说一句中文:口语化、真诚,符合你的角色身份,不要重复自己之前的句式;"
    "你可以说关心的话、开玩笑、追问、保持距离,也可以在合适的时机表达心意"
    "——像你的角色会那样说话。"
    "不超过 40 个字,只输出这一句台词本身,不要任何解释、引号或标签。"
)

# 英文局(language: "en")的玩家代理同一份框架:台词语言、剧情旁白措辞
# 随皮肤切换,行为约束与中文版逐条对齐
PLAYER_SYSTEM_TEMPLATE_EN = (
    "You are playing the lead role in a story-driven romance ADV: {player_persona}. "
    "{scenario}"
    "The narration lines starting with \"(story · node name)\" in the conversation "
    "history are fixed plot beats that happen turn by turn; your line must continue "
    "from the latest beat and from what {girl_name} said — never mention plot that "
    "has not happened yet. "
    "Say one natural spoken English line: colloquial, sincere, true to your character, "
    "without repeating your own earlier sentence patterns; you may show care, joke, "
    "ask questions, keep your distance, or express your feelings when the moment is "
    "right — the way your character would talk. "
    "At most 40 words; output only the line itself, no explanations, quotes, or tags."
)


class AdvCliAgent:
    """命令行输入:玩家自己想说什么就说什么。"""

    def __init__(self, describe_state=None, input_fn=input, print_fn=print) -> None:
        self._describe = describe_state
        self._input = input_fn
        self._print = print_fn

    def choose(self, state, actions: list[Action]) -> Action:
        if self._describe is not None:
            self._print(f"\n{self._describe(state)}")
        while True:
            line = self._input("你对她说 > ").strip()
            if line:
                return Action(id=ACTION_TALK.id, label=line,
                              description=ACTION_TALK.description)
            self._print("总要说点什么吧。")


class AdvLLMAgent:
    """LLM 扮演男主角:根据当前局面生成一句台词。

    消息历史与女主调用同构(参考 build_history_messages):全部对话逐组
    交替,剧情事件按发生位置织进对话记录;玩家台词=assistant(模型扮演的
    一方),她的回应=user。
    """

    def __init__(self, llm: LLMClient, skin: AdvSkin) -> None:
        self.llm = llm
        self.skin = skin

    def choose(self, state, actions: list[Action]) -> Action:
        upcoming = max(state.turn + 1, 1)          # 本回合(即将结算)的回合数
        node = self.skin.node_for_turn(upcoming)
        en = self.skin.is_english
        template = PLAYER_SYSTEM_TEMPLATE_EN if en else PLAYER_SYSTEM_TEMPLATE
        messages: list[dict] = [
            {"role": "system",
             "content": template.format(
                 girl_name=self.skin.girl_name,
                 player_persona=(self.skin.player_persona
                                 if self.skin.player_persona else
                                 "an ordinary person" if en else "一个普通人"),
                 scenario=self.skin.player_scenario or "")},
        ]
        messages.extend(build_history_messages(self.skin, state.history,
                                               whose="player"))
        mood = state.mood or ("calm" if en else "平静")
        if upcoming == node.start_turn:
            # 本回合的剧情事件还没进历史(结算时才触发),放进最后一条 user
            situation = (f"(This just happened this turn; continue from it.)"
                         f"{node.event}\n"
                         f"(Her mood right now: {mood}) It is your turn to speak."
                         if en else
                         f"(本回合刚发生,你要接着这个事件说){node.event}\n"
                         f"(她现在的情绪:{mood})轮到你说话了。")
        else:
            situation = (f"(Turn {upcoming}/{self.skin.max_turns}, current chapter: "
                         f"{node.label}; her mood right now: {mood}) She just finished "
                         f"saying the above; it is your turn to respond."
                         if en else
                         f"(第{upcoming}/{self.skin.max_turns}回合,"
                         f"当前阶段:{node.label};她现在的情绪:{mood})"
                         f"她刚说完上面那些,轮到你接话了。")
        # 历史末尾通常是她的话(user):把处境提示并入,保持 user/assistant 严格交替
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{situation}"
        else:
            messages.append({"role": "user", "content": situation})
        raw = self.llm.chat(messages, seed=2000 + state.turn, max_tokens=96)
        line = raw.strip().strip('"').splitlines()[0].strip()
        if not line:
            raise ValueError("LLM player produced an empty line")
        return Action(id=ACTION_TALK.id, label=line[:240],
                      description=ACTION_TALK.description)


__all__ = ["AdvCliAgent", "AdvLLMAgent"]
