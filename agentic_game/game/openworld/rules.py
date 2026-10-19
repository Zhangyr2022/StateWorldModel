"""开放世界 Demo 的规则:双回合模式的探索循环(见本模块与 game/openworld/state.py 的模式说明)。

两个 LLM 接触点(都走 engine.llm 常驻服务,VL 模式读图):
1. 评估与提议(available_actions 内):VLM 看当前画面(首图或上一段视频的
   头/中/尾 3 帧),先判断是否存在"有意义的行动机会"(§3.1),再提出 3 个
   对应不同意图的玩家行动(§3.2);
2. 导演(resolve_turn):按本回合模式选用提示词——
   - INTERACTIVE:玩家已选定行动,VLM 输出剧情旁白(中文)与下一段视频的
     画面描述(英文,可含第二镜头),后者经 VisualScript 直接成为视频模型
     的输入文本;
   - AUTO:画面平淡(且上一回合不是 AUTO)时不弹出选择,VLM 按自动探索
     规则推进一幕,在结尾留下一个尚未被处理的机会(§5),不替玩家做重大
     决定、不推进主线。

硬性规则:AUTO 之后的下一回合必须 INTERACTIVE(§3);主线节奏(最低幕数/
最早结局/保底推进)只数交互幕——AUTO 产生视频幕数,但主线不由幕数自动完成
(§7);导演调用失败按已定义的兜底降级(VLM 不可用时保守回到交互),绝不
阻断游戏循环。
"""

from __future__ import annotations

import json
import re

from engine.events import GameEvent
from engine.llm.client import LLMClient, LLMError
from engine.runtime import (
    STATUS_ONGOING,
    STATUS_WON,
    Action,
    Rng,
    TurnOutcome,
)
from game.openworld.actions import ACTION_AUTO
from game.openworld.quests import DEFAULT_WORLD, quest_for
from game.openworld.state import MODE_AUTO, MODE_INTERACTIVE, ExplorationState

# -- 提示词 -----------------------------------------------------------------

# 导演的输出 JSON 契约(交互/自动两种模式共用)。
DIRECTOR_JSON_SPEC = (
    '{"story": "<English narration, at most 40 words, second person or '
    "narrator voice>\", "
    '"consequence": "<English visual description of what happens in the next '
    "shot, third person, present tense, up to 60 words, describing the scene, "
    "the explorer's action and its visible result>\", "
    '"shot2": "<optional English description for a second cut within the same '
    'clip, or empty string>", '
    '"cut_at_s": <number, seconds into the clip for the second cut, e.g. 2.5>, '
    '"stage_complete": <true only when this clip clearly shows the current quest '
    "stage being achieved>, "
    '"plot_complete": <true only when the story should reach its climax now>, '
    '"climax": "<English ending narration, required when plot_complete is true, '
    'otherwise empty string>"}'
)

DIRECTOR_SYSTEM_PROMPT = (
    "你是第三人称开放世界冒险游戏的导演。玩家扮演一位年轻的旅人,正在一片"
    "未知大陆上探索,身上有一条主线任务。本回合是交互回合:玩家已经做出了"
    "选择。请阅读当前画面(若干帧)、主线任务进度与剧情历史,对玩家的选择"
    "做出回应,并只输出一个 JSON 对象(不要输出任何其他文字):\n"
    + DIRECTOR_JSON_SPEC + "\n"
    "要求:consequence 必须与当前画面无缝衔接(同一角色、同一环境),可以引入"
    "新的人物、建筑或事件来推进剧情;剧情要服务于主线任务的当前阶段,情节要有"
    "悬念与进展,避免原地踏步。stage_complete 的判定标准:只要本幕剧情让该阶段"
    "取得了实质进展(关键线索到手 / 抵达关键地点 / 关键事件发生 / 关键人物出现),"
    "就应当为 true——不要等待完美条件,每个阶段通常 1~2 幕即可完成。"
)

AUTO_DIRECTOR_SYSTEM_PROMPT = (
    "你是第三人称开放世界冒险游戏的导演。玩家扮演一位年轻的旅人。本回合是"
    "自动探索回合:当前画面缺乏值得玩家选择的行动机会,由你推进一幕,目标"
    "是为下一次有意义的选择创造条件。请阅读当前画面(若干帧)、主线任务进度"
    "与剧情历史,并只输出一个 JSON 对象(不要输出任何其他文字):\n"
    + DIRECTOR_JSON_SPEC + "\n"
    "自动探索规则:\n"
    "- 主动推进:让角色沿既定方向前进接近新地点、自然观察四周发现线索、声音"
    "或异常、让天气/光线/路况/附近人物产生合理变化,或引入符合世界设定与当前"
    "剧情的新地点、人物、物体或事件;\n"
    "- 视频结尾应停留在一个明确但尚未被角色处理的机会面前(如亮着灯的废弃"
    "小屋门口、远处招手求助的人、一条岔路、一个带奇怪标记的箱子),为下一"
    "回合创造决策点——目标是创造决策点,不是替玩家完成决策;\n"
    "- 禁止:替玩家攻击、偷窃、交易、承诺、接受任务或消耗重要物品;自动完成"
    "需要玩家表态的关键任务节点;无依据地瞬移到遥远地点,或突然改变角色身份"
    "与环境设定;为了强行制造选择而生成敌人、陷阱或突发危机;原地重复走路、"
    "重复镜头或重复已知信息而没有实质推进;\n"
    "- consequence 必须与当前画面无缝衔接(同一角色、同一环境),保持人物外观"
    "与摄影风格一致;\n"
    "- 自动探索回合不推进主线:stage_complete 与 plot_complete 一律输出 false。"
)

PROPOSER_SYSTEM_PROMPT = (
    "你在看第三人称开放世界冒险游戏的当前画面(多帧按时间顺序排列),"
    "站在玩家视角评估并提议下一步行动。只输出一个 JSON 对象:\n"
    '{"has_opportunity": <true/false, 当前是否存在有意义的行动机会>, '
    '"reason": "<one short English sentence explaining the judgment>", '
    '"actions": ["<English action, at most 8 words>", "<...>", "<...>"], '
    '"location_note": "<one short English sentence describing the current location>"}\n'
    "“有意义的行动机会”指至少满足以下一条:"
    "探索(出现可进入的地点/可检查的物品/值得追踪的线索)、"
    "人物(出现可交流/可帮助/可回避或值得观察的角色)、"
    "风险(出现危险/冲突/需要判断的紧急情况)、"
    "方向(玩家可以在不同路线/目标/策略之间做出实质选择)、"
    "主线(出现与当前任务阶段有关的可推进事件)。"
    "单纯的“继续走/看看周围/再等一下”若没有实质区别,不算有机会。"
    "有机会时提出 3 个行动:对应不同意图或结果方向,不是同一件事换措辞;"
    "必须基于画面真实可见的内容与已有剧情,不得虚构画面中不存在的物体;"
    "可保留一条“离开/继续前进”的退出路径;不得提前泄露未知结果"
    "(例如不直接说某个箱子里一定有宝物)。没有机会时 actions 给出空数组。"
)

# -- 节奏与数值 ---------------------------------------------------------------

MAX_ATTEMPTS = 3        # 导演调用的最大尝试次数
STAMINA_COST = 12       # 每幕行动消耗(交互/AUTO 都消耗,都是真实的一幕)
EXHAUSTED_TURNS = 2     # 体力清空后只能缓慢走动的幕数
SLOW_WALK_RECOVER = 30  # 力竭缓慢走动期间每幕恢复的体力

# 主线节奏只数交互幕(docs §7):每个主线阶段的最少交互幕数(导演判定
# 达成的门槛);三个阶段 ≈ 15 个交互幕,并给体力机制留出演出差错空间。
MIN_INTERACTIVE_PER_STAGE = 5
STAGE_BACKSTOP_EXTRA = 2  # 交互演满最少幕数 +2 仍无导演判定 → 保底推进(节奏兜底)
MIN_INTERACTIVE_BEFORE_CLIMAX = 12  # 结局最早出现的交互幕数(保证剧情充分展开)

FALLBACK_ACTIONS = ["沿着大路继续向前探索", "仔细观察四周的环境", "走进附近显眼的建筑一探究竟"]


class OpenworldRuntime:
    """实现 GameRuntime 协议;LLM 客户端与帧缓存经构造注入。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        scene_image: str,
        scene_name: str = "未知大陆",
        world_id: str = DEFAULT_WORLD,
        frame_provider: dict[int, list[str]] | None = None,
    ) -> None:
        self.llm = llm
        self.initial_scene_image = scene_image
        self.scene_name = scene_name
        self.world_id = world_id
        self.quest = quest_for(world_id)
        # turn_index → 上一段视频的采样帧路径(由 post_render 钩子写入)
        self._frames_by_next_turn: dict[int, list[str]] = frame_provider or {}

    # -- 引擎协议 ------------------------------------------------------------

    def initial_state(self) -> ExplorationState:
        return ExplorationState(scene_id=self.world_id, scene_name=self.scene_name)

    def available_actions(self, state: ExplorationState) -> list[Action]:
        if self.status_of(state) != STATUS_ONGOING:
            return []
        if not state.pending_mode:
            self._propose_turn(state)
        if state.pending_mode == MODE_AUTO:
            # 自动探索回合:只有一个行动,UI 据此直接播放、不弹出选择
            return [Action(id=ACTION_AUTO.id, label=ACTION_AUTO.label,
                           description=state.mode_note or ACTION_AUTO.description)]
        return [
            Action(id=f"act-{index}", label=text, description="VLM 提议的行动")
            for index, text in enumerate(state.pending_actions)
        ]

    def resolve_turn(self, state: ExplorationState, action: Action, rng: Rng) -> TurnOutcome:
        if self.status_of(state) != STATUS_ONGOING:
            raise ValueError(f"action not available in current state: {action.id}")
        is_auto = action.id == ACTION_AUTO.id
        if is_auto and state.last_mode == MODE_AUTO:
            raise ValueError("AUTO 后的下一回合必须是交互回合(docs §3 硬性规则)")
        chosen = (action.label or "").strip()
        if not is_auto and not chosen:
            raise ValueError("player action must be carried in Action.label")
        mode = MODE_AUTO if is_auto else MODE_INTERACTIVE

        new_state = ExplorationState(
            scene_id=state.scene_id,
            scene_name=state.scene_name,
            pending_actions=[],
            pending_mode="",
            mode_note="",
            last_mode=mode,
            last_consequence=state.last_consequence,
            story=list(state.story),
            history=list(state.history),
            climax_text=state.climax_text,
            quest_stage=state.quest_stage,
            interactive_turns=state.interactive_turns + (0 if is_auto else 1),
            stage_interactive=state.stage_interactive + (0 if is_auto else 1),
            stamina=state.stamina,
            exhausted_turns=state.exhausted_turns,
            turn=state.turn + 1,
        )
        if is_auto:
            events: list[GameEvent] = [GameEvent(
                type="auto_explore", subject="AI",
                message=state.mode_note or "画面暂无有意义的交互机会,自动推进一幕")]
        else:
            events = [GameEvent(type="player_action", subject="玩家", message=chosen)]

        # 导演与评估者必须看同一份"当前画面":帧缓存按"将在该 turn 被消费"
        # 键入(post_render 写入 index+1),要用结算前的旧 state 取帧;用
        # new_state.turn 会向后错一拍,导致导演永远只看到开局首图。
        observation_images = self._images_for(state)
        plan = self._direct(new_state, chosen, mode, images=observation_images)
        new_state.last_consequence = plan["consequence"]
        new_state.story.append(plan["story"])
        new_state.history.append(("自动探索" if is_auto else chosen, plan["consequence"]))

        # 体力结算(两种模式同规则):力竭幕只能缓慢走动(导演已收到指令)
        # 并恢复体力;普通幕扣体力,清空则进入力竭
        if new_state.exhausted_turns > 0:
            new_state.exhausted_turns -= 1
            new_state.stamina = min(100, new_state.stamina + SLOW_WALK_RECOVER)
            left = new_state.exhausted_turns
            events.append(GameEvent(
                type="exhausted",
                message="体力透支:本幕只能缓慢走动"
                        + (f"(还需缓慢走动 {left} 幕)" if left else "(已恢复行动力)"),
                detail={"slow_walk": True,
                        "stamina": new_state.stamina},
            ))
        else:
            new_state.stamina = max(0, new_state.stamina - STAMINA_COST)
            if new_state.stamina == 0:
                new_state.exhausted_turns = EXHAUSTED_TURNS
                events.append(GameEvent(
                    type="stamina_depleted",
                    message=f"体力清空:接下来 {EXHAUSTED_TURNS} 幕只能缓慢走动"))

        events.append(GameEvent(type="story_beat", message=plan["story"]))
        events.append(
            GameEvent(
                type="consequence",
                message="下一幕画面已由导演确定",
                detail={
                    "consequence": plan["consequence"],
                    "shot2": plan["shot2"],
                    "cut_at_s": plan["cut_at_s"],
                    "mode": mode,
                },
            )
        )

        # 主线结算(docs §7):关键阶段由导演提出判断、规则层按既定条件结算;
        # 节奏兜底(最低幕数/最早结局/超时推进)只数交互幕——AUTO 也产生
        # 视频幕数,但主线不因玩家缺少选择而凭幕数自动完成。
        stage_complete = bool(plan["stage_complete"]) and not is_auto
        plot_complete = bool(plan["plot_complete"]) and not is_auto

        if plot_complete and new_state.interactive_turns >= MIN_INTERACTIVE_BEFORE_CLIMAX:
            new_state.ending = "climax"
            new_state.climax_text = plan["climax"] or plan["story"]
            events.append(GameEvent(type="victory", message="这段旅程迎来了它的结局"))
            return TurnOutcome(events=tuple(events), status=STATUS_WON, state_after=new_state)

        # 保底(节奏兜底):该阶段交互演满最少幕数 +2 仍无导演判定 → 视为达成;
        # 达成还需该阶段已演出足够交互幕数,防止导演误判导致节奏过快。
        if new_state.stage_interactive >= MIN_INTERACTIVE_PER_STAGE + STAGE_BACKSTOP_EXTRA:
            stage_complete = True
        if stage_complete and new_state.stage_interactive >= MIN_INTERACTIVE_PER_STAGE:
            if new_state.quest_stage < self.quest.total - 1:
                new_state.quest_stage += 1
                new_state.stage_interactive = 0
                events.append(
                    GameEvent(
                        type="quest_stage",
                        subject="主线",
                        message=f"《{self.quest.title}》推进至第 {new_state.quest_stage + 1} 阶段:"
                                f"{self.quest.stage_text(new_state.quest_stage)}",
                        detail={"stage": new_state.quest_stage},
                    )
                )
            else:
                new_state.ending = "climax"
                new_state.climax_text = plan["climax"] or (
                    f"主线《{self.quest.title}》完成:{self.quest.stage_text(self.quest.total - 1)}")
                events.append(
                    GameEvent(type="quest_complete", subject="主线",
                              message=f"主线《{self.quest.title}》完成!")
                )
                events.append(GameEvent(type="victory", message="这段旅程迎来了它的结局"))
                return TurnOutcome(events=tuple(events), status=STATUS_WON, state_after=new_state)
        return TurnOutcome(events=tuple(events), status=STATUS_ONGOING, state_after=new_state)

    def status_of(self, state: ExplorationState) -> str:
        if state.ending == "climax":
            return STATUS_WON
        return STATUS_ONGOING

    def describe_state(self, state: ExplorationState) -> str:
        return state.summary()

    # -- 渲染帧缓存(post_render 钩子写入) ----------------------------------

    def on_video_rendered(self, turn_index: int, video_path: str) -> None:
        """GameLoop.post_render 回调:缓存刚渲染视频的采样帧,
        供下一回合的评估提议 / 导演使用(§4:以实际画面为观察基础)。"""
        from pathlib import Path

        from engine.renderer.frames import extract_sample_frames

        frames = extract_sample_frames(Path(video_path), self._frames_dir(), n=3)
        self._frames_by_next_turn[turn_index + 1] = [str(path) for path in frames]

    def _frames_dir(self):
        from pathlib import Path

        cache = getattr(self, "_frames_dir_cache", None)
        if cache is None:
            cache = Path("/tmp") / "openworld-frames"
            cache.mkdir(parents=True, exist_ok=True)
            self._frames_dir_cache = cache
        return cache

    # -- VLM 调用 ------------------------------------------------------------

    def _images_for(self, state: ExplorationState) -> list[str]:
        return self._frames_by_next_turn.get(state.turn, [self.initial_scene_image])

    def _propose_turn(self, state: ExplorationState) -> None:
        """评估当前画面并把本回合的模式与行动提议写入 state(供缓存)。

        模式判定(§3/§4):
        - 上一回合是 AUTO → 强制 INTERACTIVE(硬性规则,画面再平淡也要
          给出三种说得通的探索策略,§5.4);
        - VLM 明确判定无有意义机会(has_opportunity=false 且未提出行动)
          → AUTO,本回合自动探索;
        - VLM 不可用 / 输出不可解析 → 保守降级为 INTERACTIVE + 通用探索
          选项,绝不静默进入 AUTO。
        """
        forced_interactive = state.last_mode == MODE_AUTO
        stage = min(state.quest_stage, self.quest.total - 1)
        quest_text = (
            f"主线任务《{self.quest.title}》当前阶段({stage + 1}/{self.quest.total}):"
            f"{self.quest.stage_text(stage)}。判断与提议都应服务于这一阶段。\n"
        )
        forced_text = (
            "上一回合已自动探索,本回合必须由玩家选择行动:即使画面平淡,也要基于"
            "实际画面与已有剧情给出三种真正不同、在当前情境说得通的探索策略"
            "(例如:沿远处的道路继续寻找聚落 / 停下观察地面痕迹寻找方向线索 / "
            "返回刚经过的岔路换一个方向),has_opportunity 必须为 true。\n"
        ) if forced_interactive else ""
        messages = [
            {"role": "system", "content": PROPOSER_SYSTEM_PROMPT},
            {"role": "user", "content":
                f"{quest_text}{forced_text}"
                f"剧情到目前的进展:{'; '.join(state.story) or '冒险尚未开始'}"},
        ]

        def _set(mode: str, actions: list[str], note: str = "") -> None:
            state.pending_mode = mode
            state.pending_actions = actions
            state.mode_note = note

        try:
            raw = self.llm.chat(messages, seed=state.turn * 17 + 5, max_tokens=256,
                                images=self._images_for(state))
        except LLMError:
            _set(MODE_INTERACTIVE, list(FALLBACK_ACTIONS))  # VLM 暂不可用:交互兜底,游戏不断
            return
        parsed = _parse_json(raw)
        if not isinstance(parsed, dict):
            _set(MODE_INTERACTIVE, list(FALLBACK_ACTIONS))
            return
        note = parsed.get("location_note")
        if isinstance(note, str) and note.strip():
            state.scene_name = note.strip()[:20]
        reason = parsed.get("reason")
        mode_note = reason.strip()[:120] if isinstance(reason, str) else ""
        raw_actions = parsed.get("actions")
        # 40 字符上限与自由输入一致:提议是英文短句,20 字符会把导演收到的
        # 玩家行动截断成半句话(真实运行中观察到 "Inspect the horses o")
        actions = [str(a).strip()[:40] for a in raw_actions if str(a).strip()][:4] \
            if isinstance(raw_actions, list) else []
        if not forced_interactive and parsed.get("has_opportunity") is False and not actions:
            _set(MODE_AUTO, [], mode_note)
            return
        _set(MODE_INTERACTIVE, actions or list(FALLBACK_ACTIONS), mode_note)

    def _direct(self, state: ExplorationState, chosen: str, mode: str,
                images: list[str] | None = None) -> dict:
        if images is None:
            images = self._images_for(state)
        history_text = "\n".join(f"- {story}" for story in state.story) or "(尚未开始)"
        stage = min(state.quest_stage, self.quest.total - 1)
        quest_text = (
            f"主线任务《{self.quest.title}》:{self.quest.brief}。\n"
            f"当前阶段({stage + 1}/{self.quest.total}):{self.quest.stage_text(stage)}\n"
            "请让本幕剧情向这一阶段推进;若画面清楚呈现该阶段达成,stage_complete=true。\n"
        )
        exhausted_clause = (
            f"Explorer stamina: {state.stamina}/100"
            + (f", exhausted_turns_remaining: {state.exhausted_turns}"
               if state.exhausted_turns else "")
            + ("\nIMPORTANT: the explorer's stamina is empty — in this clip "
               "they can ONLY walk slowly and tiredly (heavy slow steps, "
               "short of breath, occasionally pausing). No running, no "
               "energetic movement."
               if state.exhausted_turns else "")
        )
        if mode == MODE_AUTO:
            system_prompt = AUTO_DIRECTOR_SYSTEM_PROMPT
            directive = ("\n本回合为自动探索:请按自动探索规则推进一幕,并在视频"
                         "结尾留下一个明确但尚未被角色处理的机会。")
        else:
            system_prompt = DIRECTOR_SYSTEM_PROMPT
            directive = f"\n玩家选择的行动:{chosen}\n请生成下一幕。"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{quest_text}\n剧情历史:\n{history_text}\n\n"
                           f"{exhausted_clause}{directive}",
            },
        ]
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            try:
                raw = self.llm.chat(messages, seed=state.turn * 31 + 7, max_tokens=512,
                                    images=images)
            except LLMError as exc:
                last_error = exc
                continue
            parsed = _parse_json(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("consequence"), str) \
                    and parsed["consequence"].strip():
                story = str(parsed.get("story", "")).strip()[:80] or (
                    "旅途自动向前推进。" if mode == MODE_AUTO else chosen)
                return {
                    "story": story,
                    "consequence": parsed["consequence"].strip()[:400],
                    "shot2": str(parsed.get("shot2", "")).strip()[:400],
                    "cut_at_s": _as_float(parsed.get("cut_at_s"), 2.5),
                    "stage_complete": bool(parsed.get("stage_complete")),
                    "plot_complete": bool(parsed.get("plot_complete")),
                    "climax": str(parsed.get("climax", "")).strip()[:120],
                }
            last_error = ValueError(f"unparsable director output: {raw[:200]!r}")
        # 兜底:导演暂时失声,给出"继续探索"的保守画面,游戏继续
        story = "画面平淡,旅途自动向前推进。" if mode == MODE_AUTO \
            else f"你选择了「{chosen}」,旅途继续。"
        return {
            "story": story,
            "consequence": (
                "The young explorer presses on through the same landscape, "
                "camera tracking behind as new details of the surroundings "
                "come into view."
            ),
            "shot2": "",
            "cut_at_s": 2.5,
            "stage_complete": False,
            "plot_complete": False,
            "climax": "",
        }


def _parse_json(raw: str) -> dict | None:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["OpenworldRuntime", "ExplorationState"]
