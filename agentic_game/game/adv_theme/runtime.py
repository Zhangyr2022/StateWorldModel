"""adv 主题模板的结算规则:固定剧情骨架 + 自由输入 + LLM 女主回应。

一回合三步(见本文件 resolve 的阶段说明):
  ① 节点裁决    当前回合落在哪个剧情节点;进入新节点即触发固定事件
                (玩家拥有对话自由,但没有暂停世界时间的能力)
  ② LLM 女主    按角色+当前剧情+当前关系+当前情绪生成回应与关系值打分
                (JSON 契约;贪心解码+重试;契约破坏/服务失败按 romance
                同款降级,绝不静默编造)
  ③ 关系与结局  LLM 打分夹取到 ±max_abs_delta 后更新关系值;最终回合按
                关系值+意图命中判定结局(固定的是事件,不固定的是反应)

随机只来自传入的 Rng(仅用作 LLM 采样种子);resolve_turn 不修改入参。
"""

from __future__ import annotations

import json
import re

from engine.events import GameEvent
from engine.llm.client import LLMClient, LLMError
from engine.runtime import (
    STATUS_DRAW,
    STATUS_LOST,
    STATUS_ONGOING,
    STATUS_WON,
    Action,
    Rng,
    TurnOutcome,
)
from game.adv_theme.actions import ACTION_TALK
from game.adv_theme.skin import AdvSkin, EndingSpec, NodeSpec

# 心情 → 台词方式(演出层 cue;与 mood_actions 的肢体短语互补)
MOOD_CUES = {
    "calm": "says calmly",
    "warm": "says softly",
    "guarded": "says quietly",
    "tired": "says wearily",
    "sad": "says quietly",
    "flustered": "says",
}

SYSTEM_PROMPT_TEMPLATE = (
    "【角色】\n"
    "你是{girl_name}:{persona}\n"
    "{constraints}\n"
    "\n"
    "【当前关系】\n"
    "Relationship:{relationship}(0~100,隐藏数值,不要在回应里提数字)\n"
    "{relationship_state}\n"
    "\n"
    "【当前情绪】\n"
    "{mood_line}\n"
    "\n"
    "剧情事件与全部对话都记录在下面的对话历史里(以「(剧情·节点名)」开头的"
    "旁白消息就是刚发生/曾发生的固定剧情),请结合它们回应玩家最新的话。"
    "回应必须使用中文(简体),并只输出"
    "一个 JSON 对象(不要输出任何其他文字;必须先写 delta_reason 再写数字):\n"
    '{{"reply": "<口语化的中文回应,符合角色,不超过 60 字、最多三句>", '
    '"delta_reason": "<不超过 15 字,说明玩家这句话做了什么(如\'主动关心我\')>", '
    '"relationship_delta": <整数,-{max_delta}到{max_delta}之间,表示这句话让你们关系的变化:'
    '明显拉近 +{max_delta},良好 +2,轻微 +1,普通 0,轻微不适 -1,明显伤害 -2,严重伤害 -{max_delta}>, '
    '"mood": "<{mood_choices} 中选一个>"}}\n'
    "打分标准(必须逐回合独立判断,禁止默认给 0):\n"
    "- 玩家主动关心你、记住你说过的话、注意到你的状态、帮你分担、给你带东西"
    " → delta 必须大于 0(轻微 +1,真诚走心 +2~+{max_delta})\n"
    "- 只有纯粹的事务性应答(只谈工作交接、礼节性回应)才是 0\n"
    "- 敷衍、开过分的玩笑、强行推进、忽视你的拒绝 → -1~-{max_delta}\n"
    "重要:打分与心情只由玩家这一轮的言行决定,与你之前回应的长短、语气无关;"
    "即使前面的对话很简短,玩家真诚的关心依然至少 +1,你的回应也依然要自然地把"
    "话说完(不要因为历史而越来越省略)。\n"
    "结合你当前的真实心态({relationship_state}),这些善意会让你有细微的触动,"
    "mood 也应随之起伏,禁止整局只用 calm。\n"
    "剧情红线:不要提前进入尚未发生的剧情节点,不要突然透露完整过去,"
    "不要自行解决核心矛盾,不要替玩家说话。"
)

# 英文局(language: "en")的同一份契约:台词、事件措辞、守卫方向全部换到英文,
# 结算规则与打分分档与中文版完全一致(皮肤内容本身由英文皮肤 JSON 提供)。
SYSTEM_PROMPT_TEMPLATE_EN = (
    "[Role]\n"
    "You are {girl_name}: {persona}\n"
    "{constraints}\n"
    "\n"
    "[Current relationship]\n"
    "Relationship: {relationship} (0-100, hidden number; never mention it in your reply)\n"
    "{relationship_state}\n"
    "\n"
    "[Current mood]\n"
    "{mood_line}\n"
    "\n"
    "The plot events and the entire conversation are recorded in the history below "
    "(narration lines starting with \"(story · node name)\" are fixed plot beats that "
    "just happened / happened earlier). Respond to the player's latest line in light "
    "of them. Your reply MUST be natural spoken English, and you must output only "
    "one JSON object (no other text; write delta_reason before the number):\n"
    '{{"reply": "<colloquial English reply, in character, at most 40 words / three sentences>", '
    '"delta_reason": "<at most 10 words: what the player\'s line just did (e.g. \'cared about me\')>", '
    '"relationship_delta": <integer between -{max_delta} and {max_delta}: how this line changed your relationship: '
    'clearly closer +{max_delta}, good +2, slight +1, neutral 0, slight discomfort -1, clearly hurtful -2, seriously hurtful -{max_delta}>, '
    '"mood": "<pick one from {mood_choices}>"}}\n'
    "Scoring standard (judge each turn independently; defaulting to 0 is forbidden):\n"
    "- If the player proactively cares about you, remembers what you said, notices your state, "
    "shares your burden, or brings you something → delta must be > 0 (slight +1, sincere +2~+{max_delta})\n"
    "- Only purely transactional replies (work handover only, polite acknowledgment) are 0\n"
    "- Perfunctory, overly teasing, pushy, or ignoring your refusal → -1~-{max_delta}\n"
    "Important: score and mood are decided only by the player's words and actions this turn, "
    "regardless of the length or tone of your previous replies; even after brief exchanges, "
    "the player's sincere care is still worth at least +1, and your reply must still finish "
    "naturally (do not grow terser because of history).\n"
    "Combined with your current state of mind ({relationship_state}), these kindnesses touch "
    "you subtly, and your mood should rise and fall with them; using only calm all game is forbidden.\n"
    "Plot red lines: do not jump ahead to plot nodes that have not happened yet, do not suddenly "
    "reveal your entire past, do not resolve your own core conflict by yourself, do not speak for the player."
)

FIRST_TURN_EVENT_HEADER = "【刚发生的剧情事件——你的回应要以它为中心】"
FIRST_TURN_EVENT_HEADER_EN = "[The plot event that just happened — center your reply on it]"
FIRST_TURN_EVENT_NOTE = (
    "(事件里也包含你说的话:先接住事件里你的言行与玩家对此的回应,再谈别的;"
    "不要顺着之前的对话继续。)"
)
FIRST_TURN_EVENT_NOTE_EN = (
    "(The event also contains your own words: first pick up what you said or did in it "
    "and the player's response to that, then move on to other things; do not drift along "
    "the earlier small talk.)"
)
# 节点固定事件织入对话记录时的旁白前缀(历史剧情也是对话的一部分)
PLOT_NARRATION_PREFIX = "(剧情·{label})"
PLOT_NARRATION_PREFIX_EN = "(story · {label})"

MAX_ATTEMPTS = 3

# 追加到最后一条 user 消息末尾的提醒:多轮历史里 assistant 都是纯中文,
# 模型会顺着做"纯对话续写"而丢掉 JSON 契约(2026-10-09 真实服务实测);
# 放在末尾靠近生成位置,比 system 里的远距离指令有效(worker 的 chat
# template 也不允许 system 出现在中间)。同槽位还带句式禁令:模型会锚定
# 自己第 1 回合定型的模板(「……」开头+「不过,既然你都这么说了」+「勉为其难」,
# 2026-10-09 实测 15/25 句同模板),必须在生成位置反复打破。
_JSON_REMINDER_TEMPLATE = (
    "\n\n(输出格式再次强调:只输出一个 JSON 对象,不要输出任何其他文字:"
    '{{"reply": "<中文回应>", "delta_reason": "<≤15字,玩家这句话做了什么>", '
    '"relationship_delta": <整数,-{max_delta}到{max_delta};关心/带东西/邀约/记住'
    '你的话 → 至少 +1>, "mood": "<{mood_choices} 中选一个>"}})\n'
    "【反重复硬规则】你已经说过 {n_replies} 句回应,回应对相似度是本游戏的第一红线。\n"
    "以下开头全部用过,新回应禁止以它们或任何近似写法开头:\n"
    "{banned_openings}\n"
    "以下首字也全部用过,新回应的第一个字禁止与它们相同:\n"
    "{banned_first}\n"
    "你的最近两句:\n上上句:「{prev_reply}」\n上一句:「{last_reply}」\n"
    "这一句的开头、句式、措辞、切入点都必须与它们完全不同。\n"
    "句式要求(逐条严格遵守):\n"
    "①以一个全新的、上面两个清单之外的方式开场;\n"
    "②句式结构必须与历史所有句子不同,旧句式一律弃用;\n"
    "③同一个意思(口是心非地接受/嘴硬否认/别扭道谢)每次都要用全新的措辞;\n"
    "④历史回应里任何超过 6 个字的短语都不许再次出现,必须换词重说;\n"
    "⑤本回合只允许选一种形状:反问/短叹/先应后驳/只答不问/以半拍停顿起句,"
    "且该形状不得与你最近三句相同;\n"
    "⑥连续两回合的回应放在一起读,必须像两个不同场景说的话,不允许续写感。"
)

# 英文局的同槽位提醒(契约/反重复规则逐条对齐中文版;禁重复粒度从"字"放宽到
# "词",因为英文按词起句)
_JSON_REMINDER_TEMPLATE_EN = (
    "\n\n(Reminder of the output format once more: output only one JSON object, no other text: "
    '{{"reply": "<English reply>", "delta_reason": "<at most 10 words: what the player\'s line did>", '
    '"relationship_delta": <integer, -{max_delta} to {max_delta}; care/gifts/invitations/remembering '
    'what you said → at least +1>, "mood": "<pick one from {mood_choices}"}})\n'
    "[Anti-repetition hard rules] You have already given {n_replies} replies; similarity between "
    "replies is this game's first red line.\n"
    "All of the following openings have been used; a new reply must NOT start with them or any "
    "near-copy:\n{banned_openings}\n"
    "The following first words have also been used; the first word of a new reply must not be "
    "one of them:\n{banned_first}\n"
    "Your two most recent replies:\ntwo ago: \"{prev_reply}\"\nlast: \"{last_reply}\"\n"
    "This reply's opening, sentence structure, wording, and angle must all be completely "
    "different from them.\n"
    "Sentence-pattern requirements (follow each strictly):\n"
    "① Open in a brand-new way, outside both lists above;\n"
    "② The sentence structure must differ from every sentence in the history; discard all old patterns;\n"
    "③ The same meaning (reluctantly accepting / stubbornly denying / awkward thanks) must be "
    "reworded completely fresh each time;\n"
    "④ No phrase longer than 4 words from any earlier reply may appear again; say it in new words;\n"
    "⑤ This turn only one shape is allowed: a rhetorical question / a short sigh / agree-then-rebut / "
    "answer without asking / start on a half-beat pause, and it must not match your last three replies;\n"
    "⑥ Any two consecutive replies, read together, must sound like two different scenes — no "
    "continuation feel."
)

CJK_RE = re.compile(r"[一-鿿]")


def _has_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s or ""))


def build_history_messages(skin: AdvSkin, history: list, *, whose: str) -> list[dict]:
    """把 (玩家台词, 她的回应) 历史重建为消息序列,剧情事件按发生位置织入。

    - whose="girl":玩家台词=user、她的回应=assistant(她是要模拟的一方);
    - whose="player":玩家台词=assistant、她的回应=user;
    - 每个节点的固定事件以旁白身份织入它开始的回合之前:girl 视角并入该回合
      的玩家消息前缀(事件先发生、玩家再说话);player 视角追加在上一回合她
      的消息之后,首个节点单独一条开场旁白。剧情与历史剧情因此都是**对话记录
      的一部分**,不单独挂在 system(2026-10-10 定稿)。
    """
    prefix = (PLOT_NARRATION_PREFIX_EN if skin.is_english
              else PLOT_NARRATION_PREFIX)
    msgs: list[dict] = []
    active: str | None = None
    for i, (player_line, girl_reply) in enumerate(history):
        node = skin.node_for_turn(i + 1)
        event = None
        if node.id != active:
            active = node.id
            event = f"{prefix.format(label=node.label)}{node.event}"
        if whose == "girl":
            user_line = f"{event}\n{player_line}" if event else player_line
            msgs.append({"role": "user", "content": user_line})
            msgs.append({"role": "assistant", "content": girl_reply})
        else:
            if event and msgs:
                # 追加到上一条她的消息后(时间顺序:她说完话,剧情事件发生)
                msgs[-1]["content"] = f"{msgs[-1]['content']}\n{event}"
            elif event:
                msgs.append({"role": "user", "content": event})
            msgs.append({"role": "assistant", "content": player_line})
            msgs.append({"role": "user", "content": girl_reply})
    return msgs


class AdvState:
    """纯数据游戏状态:关系值 + 回合 + 节点 + 对话历史 + 终局。"""

    def __init__(self, girl_name: str, relationship: int, turn: int = 0,
                 node: str = "", mood: str = "",
                 history: list | None = None,
                 last_player_line: str = "", last_reply: str = "",
                 last_action: str = "",
                 intent_turn: int = 0, ending_id: str = "") -> None:
        self.girl_name = girl_name
        self.relationship = relationship
        self.turn = turn
        self.node = node              # 当前剧情节点 id
        self.mood = mood              # 她上一回合的心情(演出与情绪连续性)
        self.history = list(history or [])   # [(player_line, reply), ...]
        self.last_player_line = last_player_line
        self.last_reply = last_reply
        self.last_action = last_action    # 她上一回合的神态/小动作(LLM 生成)
        self.intent_turn = intent_turn    # 最后一次恋爱意图命中的回合;0 = 未命中
        self.ending_id = ending_id        # 非空 = 已终局

    def to_snapshot(self) -> dict:
        """结构化快照(turn_log.jsonl / rebuild_replay_hud 读取)。"""
        return {
            "relationship": self.relationship,
            "turn": self.turn,
            "node": self.node,
            "mood": self.mood,
            "last_action": self.last_action,
            "intent_turn": self.intent_turn,
            "ending_id": self.ending_id,
            "history": list(self.history),
        }


class AdvRuntime:
    """实现 GameRuntime 协议;LLM 客户端经构造注入(测试可换桩)。"""

    def __init__(self, skin: AdvSkin, llm: LLMClient) -> None:
        if llm is None:
            raise ValueError(f"{skin.game_id} 需要常驻 LLM 服务(先启动 engine.llm.service)")
        self.skin = skin
        self.llm = llm

    # -- 引擎协议 ------------------------------------------------------------

    def initial_state(self) -> AdvState:
        first = self.skin.node_for_turn(1)
        return AdvState(
            girl_name=self.skin.girl_name,
            relationship=self.skin.relationship_initial,
            node=first.id,
            mood=self.skin.moods[0] if self.skin.moods else "",
        )

    def available_actions(self, state: AdvState) -> list[Action]:
        if self.status_of(state) != STATUS_ONGOING:
            return []
        return [ACTION_TALK]

    def status_of(self, state: AdvState) -> str:
        if state.ending_id:
            ending = self.skin.ending_by_id(state.ending_id)
            return ending.status if ending is not None else STATUS_ONGOING
        if state.turn >= self.skin.max_turns:
            return self.select_ending(state).status
        return STATUS_ONGOING

    def select_ending(self, state: AdvState) -> EndingSpec:
        return self.skin.select_ending(
            state.relationship, self._intent_hit(state))

    def resolve_turn(self, state: AdvState, action: Action, rng: Rng) -> TurnOutcome:
        if self.status_of(state) != STATUS_ONGOING:
            raise ValueError(f"action not available in current state: {action.id}")
        player_line = (action.label or "").strip()
        if not player_line or action.label == ACTION_TALK.label:
            raise ValueError("player line must be carried in Action.label")

        skin = self.skin
        new_state = AdvState(
            girl_name=state.girl_name,
            relationship=state.relationship,
            turn=state.turn + 1,
            node=state.node,
            mood=state.mood,
            history=list(state.history),
            intent_turn=state.intent_turn,
        )
        events: list[GameEvent] = [
            GameEvent(type="player_line", subject=skin.player_name, message=player_line)
        ]

        # ① 节点裁决:固定事件按回合数发生(剧情骨架不受玩家话题影响)
        node = skin.node_for_turn(new_state.turn)
        node_entered = node.id != new_state.node
        if node_entered:
            new_state.node = node.id
            events.append(GameEvent(
                type="node_event", subject=new_state.girl_name,
                message=f"—— {node.label} ——\n{node.event}",
                detail={"node": node.id, "turn": node.start_turn},
            ))

        # ② LLM 女主回应 + 关系打分(动作一律走 mood_actions 档位)
        reply, delta, mood = self._ask_girl(new_state, player_line, node, rng)
        if node_entered and node.arrival:
            events.append(GameEvent(
                type="node_arrival", subject=new_state.girl_name,
                detail={"node": node.id},
            ))

        # ③ 关系结算:LLM 打分只是有界输入,规则层集中夹取(结算确定可审计)
        bounded = max(-skin.max_abs_delta, min(skin.max_abs_delta, delta))
        clamped = max(skin.relationship_min,
                      min(skin.relationship_max, new_state.relationship + bounded))
        actual = clamped - new_state.relationship
        new_state.relationship = clamped
        new_state.last_player_line = player_line
        new_state.last_reply = reply
        new_state.mood = mood if mood in skin.moods else (skin.moods[0] if skin.moods else "")
        # 动作:一律由皮肤 mood_actions 按心情档位给出(神态-only);
        # LLM 只决定回应/心情/打分,不生成动作
        new_state.last_action = skin.mood_actions.get(new_state.mood, "")
        new_state.history.append((player_line, reply))
        events.append(GameEvent(
            type="girl_reply", subject=new_state.girl_name, message=reply,
            detail={"mood": new_state.mood},
        ))
        events.append(GameEvent(
            type="relationship_change", subject=new_state.girl_name,
            amount=actual,
            message=self._relationship_message(actual, clamped),
            detail={"requested_delta": delta, "bounded_delta": bounded},
        ))

        # 恋爱意图:玩家后期明确表达(如告白)→ 记录命中回合(结局 A 的门槛)
        if new_state.turn >= skin.intent_from_turn:
            hit = next((kw for kw in skin.intent_keywords if kw in player_line), None)
            if hit:
                new_state.intent_turn = new_state.turn
                events.append(GameEvent(
                    type="intent_hint", subject=skin.player_name,
                    detail={"keyword": hit, "turn": new_state.turn},
                ))

        # ④ 终局:骨架走完才判定(固定回合推进;结局由全程互动共同形成)
        status = STATUS_ONGOING
        if new_state.turn >= skin.max_turns:
            ending = self.select_ending(new_state)
            new_state.ending_id = ending.id
            status = {"won": STATUS_WON, "lost": STATUS_LOST,
                      "draw": STATUS_DRAW}[ending.status]
            events.append(GameEvent(
                type="victory" if ending.status == "won"
                else "defeat" if ending.status == "lost" else "ending",
                subject=new_state.girl_name,
                message=ending.caption or ending.id,
                detail={"ending": ending.id, "relationship": new_state.relationship},
            ))
        return TurnOutcome(events=tuple(events), status=status, state_after=new_state)

    def chapter_card(self, state: AdvState) -> dict | None:
        """即将进行的回合若开启一个剧情节点,返回该节点的章节卡;否则 None。

        开局(第 1 回合 = 首个节点)与之后每次进入新节点时出现,让玩家在
        开口之前先读到本节点的固定事件(结算时同一事件织入 LLM 对话,见
        resolve_turn ①)。界面经特性检测调用(无此方法的游戏不显示章节卡)。
        返回 {"title": 节点名, "text": 事件旁白, "index": 第几章(1 起), "total": 章数}。
        """
        if self.status_of(state) != STATUS_ONGOING:
            return None
        upcoming = state.turn + 1
        node = self.skin.node_for_turn(upcoming)
        if state.turn > 0 and node.id == state.node:
            return None
        index = next(i for i, n in enumerate(self.skin.nodes) if n.id == node.id)
        return {"title": node.label, "text": node.event,
                "index": index + 1, "total": len(self.skin.nodes)}

    def describe_state(self, state: AdvState) -> str:
        node = self.skin.node_for_turn(max(state.turn, 1))
        mood = state.mood
        if self.skin.is_english:
            rel = (f" | Relationship {state.relationship}/{self.skin.relationship_max}"
                   if self.skin.show_relationship else "")
            reply = state.last_reply or "(waiting for you to speak)"
            return (f"Turn {state.turn} [{node.label}]{rel} | "
                    f"{state.girl_name}: {reply} ({mood})")
        rel = f" | 关系 {state.relationship}/{self.skin.relationship_max}" \
            if self.skin.show_relationship else ""
        reply = state.last_reply or "(等你开口)"
        return f"第{state.turn}回合 [{node.label}]{rel} | {state.girl_name}:{reply}({mood})"

    # -- 内部 ------------------------------------------------------------------

    def _relationship_message(self, actual: int, clamped: int) -> str:
        """relationship_change 事件的展示措辞(中文局中文、英文局英文)。"""
        skin = self.skin
        if skin.show_relationship:
            return (f"Relationship {actual:+d} → {clamped}/{skin.relationship_max}"
                    if skin.is_english else
                    f"关系 {actual:+d} → {clamped}/{skin.relationship_max}")
        if skin.is_english:
            if actual > 0:
                return "she seems to have warmed a little"
            if actual < 0:
                return "she seems to have cooled a little"
            return "her attitude seems unchanged"
        return (f"她的态度似乎{'亲近' if actual > 0 else '疏远' if actual < 0 else '没有变化'}了一些")

    def _intent_hit(self, state: AdvState) -> bool:
        return state.intent_turn >= self.skin.intent_from_turn

    def _build_system_prompt(self, state: AdvState, node: NodeSpec) -> str:
        skin = self.skin
        en = skin.is_english
        constraints = "\n".join(
            (f"Never {c}" if en else f"不要{c}") for c in skin.girl_constraints)
        relationship_state = skin.relationship_state(state.relationship)
        mood_line = (f"Her mood last turn: {state.mood} (reference only; "
                     "re-judge from the player's words and actions this turn). "
                     if state.mood and en else
                     f"她上一回合的心情:{state.mood}(只作参考,"
                     f"本回合请按玩家这轮的言行重新判断)。" if state.mood else "")
        if node.mood_hint:
            mood_line = f"{mood_line}{node.mood_hint}"
        # 心情语义注释来自皮肤(mood_notes),通用模板不硬编码任何游戏的心情含义
        mood_choices = "|".join(
            f"{m}({skin.mood_notes[m]})" if skin.mood_notes.get(m) else m
            for m in skin.moods)
        template = SYSTEM_PROMPT_TEMPLATE_EN if en else SYSTEM_PROMPT_TEMPLATE
        return template.format(
            girl_name=skin.girl_name,
            persona=skin.girl_persona,
            constraints=constraints,
            relationship=state.relationship,
            relationship_state=relationship_state,
            mood_line=mood_line,
            max_delta=skin.max_abs_delta,
            mood_choices=mood_choices,
        )

    def _ask_girl(self, state: AdvState, player_line: str,
                  node: NodeSpec, rng: Rng) -> tuple[str, int, str]:
        """问女主:返回 (reply, raw_delta, mood)。

        LLM 只决定回应/心情/关系打分;动作由皮肤 mood_actions 按心情档位
        给出(2026-10-11 定稿,LLM 不再生成动作短语)。

        失败处理与 romance 同款(结算失败有界、可审计;契约语言随皮肤,
        守卫方向随语言翻转:中文局要求回应含中文,英文局要求回应无中文):
        - JSON 契约破坏:把模型真实文本当回应,打分 0,不静默编造数值;
        - 模型违规输出错误语言的文本(对白标签会念错):弃用并走中性兜底;
        - 服务瞬态错误:重试;持续失败长局降级中性回应,开局即失败才上抛。
        """
        en = self.skin.is_english
        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt(state, node)}
        ]
        # 对话历史(含织入的剧情旁白):剧情与历史剧情都是对话记录的一部分
        messages.extend(build_history_messages(self.skin, state.history, whose="girl"))
        # last_reply 用占位符替换而非 format:回应原文可能含花括号等字符
        notes = self.skin.mood_notes or {}
        mood_choices = "|".join(
            f"{m}({notes[m]})" if notes.get(m) else m for m in self.skin.moods)
        replies = [r.strip() for _, r in state.history if r and r.strip()]
        # 英文按头 3 个词起句,中文按头 6 字(反重复粒度与语言习惯一致)
        def _head(r: str) -> str:
            return " ".join(r.split()[:3]) if en else r[:6]
        openings: list[str] = []
        for r in replies:
            head = _head(r)
            if head and head not in openings:
                openings.append(head)
        if en:
            banned = ("、".join(f"\"{o}\"" for o in openings)
                      if openings else
                      "(This is the first line; no history openings yet.)")
            banned_first = ("、".join(sorted({r.split()[0] for r in replies
                                             if r.split()}))
                            if replies else "(none)")
            no_prev, no_last = "(none)", "(you have not spoken yet)"
        else:
            banned = ("、".join(f"「{o}」" for o in openings)
                      if openings else "(这是第一句,没有历史开头)")
            banned_first = ("、".join(sorted({r[0] for r in replies}))
                            if replies else "(无)")
            no_prev, no_last = "(无)", "(你还没有说过话)"
        # 上一句/上上句从 history 取:new_state.last_reply 在此刻还未写入
        # (resolve_turn 在 _ask_girl 之后才赋值),取它会永远拿到空串
        prev_reply = replies[-2] if len(replies) >= 2 else no_prev
        last_reply = replies[-1] if replies else no_last
        template = (_JSON_REMINDER_TEMPLATE_EN if en
                    else _JSON_REMINDER_TEMPLATE)
        reminder = template.format(
            max_delta=self.skin.max_abs_delta,
            mood_choices=mood_choices,
            n_replies=len(replies),
            banned_openings=banned,
            banned_first=banned_first,
            prev_reply=prev_reply[:60],
            last_reply="__LAST_REPLY__")
        reminder = reminder.replace(
            "__LAST_REPLY__", last_reply)
        if state.turn == node.start_turn:
            # 节点首回合:本回合的事件还没进历史,把它与玩家台词一起放在
            # 最后一条 user(紧贴生成位置)——近因优势给剧情事件,她才会
            # 接事件里自己刚说的话,而不是顺着之前的闲聊(2026-10-10 实测)
            if en:
                final_user = (f"{FIRST_TURN_EVENT_HEADER_EN}\n{node.event}\n"
                              f"{FIRST_TURN_EVENT_NOTE_EN}\n\n"
                              f"[The player just said] {player_line}")
            else:
                final_user = (f"{FIRST_TURN_EVENT_HEADER}\n{node.event}\n"
                              f"{FIRST_TURN_EVENT_NOTE}\n\n"
                              f"【玩家刚刚说】{player_line}")
        else:
            final_user = player_line
        messages.append({"role": "user", "content": final_user + reminder})

        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            try:
                raw = self.llm.chat(messages, seed=rng.randint(0, 2**31 - 1),
                                    max_tokens=256)
            except LLMError as exc:
                last_error = exc
                continue
            parsed = _parse_reply(raw, self.skin.moods)
            # 语言守卫针对最终回应文本(契约内/外一视同仁):
            # 中文局要求含中文,英文局要求无中文(否则对白标签会念错语言)
            reply_text = parsed[0] if parsed is not None else raw.strip()
            if not reply_text or (
                    (not _has_cjk(reply_text)) if not en else _has_cjk(reply_text)):
                return self._neutral_reply(state)
            if parsed is not None:
                return parsed
            return (raw.strip()[:240], 0,
                    self.skin.moods[0] if self.skin.moods else "")
        if state.turn > 1:
            return self._neutral_reply(state)
        raise LLMError(f"girl reply failed after {MAX_ATTEMPTS} attempts: {last_error}")

    def _neutral_reply(self, state: AdvState) -> tuple[str, int, str]:
        """中性兜底回应(随皮肤语言;打分 0,心情回落到第一档)。"""
        if self.skin.is_english:
            return (f"{state.girl_name} looks at you quietly, then gives a small "
                    "nod after a moment.",
                    0, self.skin.moods[0] if self.skin.moods else "")
        return (f"{state.girl_name}安静地看着你,片刻后轻轻点了点头。",
                0, self.skin.moods[0] if self.skin.moods else "")


def _parse_reply(raw: str, moods: tuple) -> tuple[str, int, str] | None:
    """从模型输出提取 reply/delta/mood;容错 markdown 代码块。

    动作不由 LLM 决定(2026-10-11 定稿):输出里即使带 action 字段也忽略,
    演出短语一律由皮肤 mood_actions 按心情档位给出。
    """
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    reply = data.get("reply")
    delta = data.get("relationship_delta", 0)
    mood = data.get("mood", "")
    if not isinstance(reply, str) or not reply.strip():
        return None
    try:
        delta = int(delta)
    except (TypeError, ValueError):
        delta = 0
    if not isinstance(mood, str) or mood not in moods:
        mood = moods[0] if moods else ""
    return reply.strip()[:240], int(delta), mood


__all__ = ["AdvRuntime", "AdvState", "MOOD_CUES"]
