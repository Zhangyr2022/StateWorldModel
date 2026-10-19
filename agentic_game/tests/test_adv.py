"""adv 主题模板测试:节点推进、关系结算、结局判定、契约容错与完整循环。

LLM 用进程内 FakeLLM 替换(不依赖常驻服务);渲染用 FakeRenderer——
游戏逻辑与 Visual Pipeline 的测试与 GPU 完全无关(见 AGENTS.md「Engine engineering rules」)。
"""

import base64
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from engine.llm.client import LLMError
from engine.loop import GameLoop
from engine.renderer.fake import FakeRenderer
from engine.runtime import (
    STATUS_DRAW,
    STATUS_LOST,
    STATUS_ONGOING,
    STATUS_WON,
    Action,
    Rng,
    TurnResult,
)
from game.adv_theme import build_adv_game
from game.adv_theme.actions import ACTION_TALK
from game.adv_theme.agents import AdvCliAgent, AdvLLMAgent
from game.adv_theme.json_skin import load_json_skin
from game.adv_theme.runtime import AdvRuntime, AdvState
from game.adv_theme.visual import AdvVisualDirector
from game.registry import GAMES

SKIN_PATH = Path(__file__).resolve().parents[1] / "game" / "skins" / "before_rain_mio.json"

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class FakeLLM:
    """进程内 LLM 桩:按脚本依次返回预设文本。"""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, seed=0, max_tokens=256, timeout_s=None):
        self.calls.append({"messages": messages, "seed": seed})
        if not self.responses:
            raise LLMError("fake LLM exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def girl_json(reply="……嗯。你今天也很吵。", delta=0, mood="calm",
              action=None) -> str:
    data = {"reply": reply, "relationship_delta": delta, "mood": mood}
    if action is not None:
        data["action"] = action
    return json.dumps(data, ensure_ascii=False)


def talk(line: str) -> Action:
    return Action(id=ACTION_TALK.id, label=line)


class ScriptedTalkAgent:
    """自由文本脚本代理:按给定台词序列说话(用尽后重复最后一句)。"""

    def __init__(self, lines: list[str]) -> None:
        if not lines:
            raise ValueError("need at least one line")
        self.lines = list(lines)
        self.cursor = 0

    def choose(self, state, actions):
        line = self.lines[min(self.cursor, len(self.lines) - 1)]
        self.cursor += 1
        return Action(id=ACTION_TALK.id, label=line)


def load_skin():
    return load_json_skin(SKIN_PATH)


def line_at(turn: int, intent: bool = True) -> str:
    """第 turn 回合的玩家台词;intent=True 时第 21 回合起带恋爱意图。"""
    if turn >= 21 and intent:
        return "我喜欢你。毕业以后,我也没打算消失。"
    return ["今天照片好多啊。", "你觉得哪张最好看?", "累的话就休息一下吧。",
            "这本纪念册做完要多久?", "放学后一起留下吧。"][turn % 5]


def make_loop_responses(deltas, moods=None, replies=None):
    return [girl_json(delta=deltas[i % len(deltas)],
                      mood=(moods[i % len(moods)] if moods else "calm"))
            for i in range(max(len(deltas), 30))]


class TestSkinAndRegistry(unittest.TestCase):
    def test_registered_as_adv_needs_llm(self):
        entry = GAMES["before_rain_mio"]
        self.assertEqual(entry.category, "adv")
        self.assertTrue(entry.needs_llm)

    def test_node_progression(self):
        skin = load_skin()
        self.assertEqual(skin.node_for_turn(1).id, "meet")
        self.assertEqual(skin.node_for_turn(5).id, "meet")
        self.assertEqual(skin.node_for_turn(6).id, "rain")
        self.assertEqual(skin.node_for_turn(11).id, "discovered")
        self.assertEqual(skin.node_for_turn(16).id, "understood")
        self.assertEqual(skin.node_for_turn(21).id, "choice")
        self.assertEqual(skin.node_for_turn(25).id, "choice")


class TestAdvRuntime(unittest.TestCase):
    def setUp(self):
        self.skin = load_skin()
        self.runtime = AdvRuntime(self.skin, FakeLLM([girl_json(delta=2, mood="warm")]))
        self.state = self.runtime.initial_state()

    def test_initial_state(self):
        self.assertEqual(self.state.relationship, 50)
        self.assertEqual(self.state.node, "meet")
        self.assertEqual(self.state.turn, 0)
        self.assertEqual([a.id for a in self.runtime.available_actions(self.state)],
                         [ACTION_TALK.id])

    def test_chapter_card_at_start_and_each_new_node(self):
        """章节卡:开局给出首个节点;之后仅在下一回合进入新节点时出现。"""
        nodes = self.skin.nodes
        card = self.runtime.chapter_card(self.state)
        self.assertEqual(card["title"], nodes[0].label)
        self.assertEqual(card["text"], nodes[0].event)
        self.assertEqual((card["index"], card["total"]), (1, len(nodes)))

        runtime = AdvRuntime(self.skin, FakeLLM([girl_json()] * 40))
        state = runtime.initial_state()
        shown = {}
        while state.turn + 1 <= nodes[-1].start_turn:
            card = runtime.chapter_card(state)
            if card is not None:
                shown[state.turn + 1] = card["title"]
            state = runtime.resolve_turn(state, talk("嗯"), Rng(state.turn)).state_after
        # 每个节点恰好在它的 start_turn 之前出现一次
        self.assertEqual(shown, {n.start_turn: n.label for n in nodes})

    def test_resolve_updates_relationship_and_history(self):
        outcome = self.runtime.resolve_turn(self.state, talk("今天照片好多啊"), Rng(1))
        after = outcome.state_after
        self.assertEqual(after.relationship, 52)  # 50 + clamp(2)
        self.assertEqual(after.turn, 1)
        self.assertEqual(after.mood, "warm")
        self.assertEqual(after.history, [("今天照片好多啊", "……嗯。你今天也很吵。")])
        self.assertEqual([e.type for e in outcome.events],
                         ["player_line", "girl_reply", "relationship_change"])
        self.assertEqual(outcome.status, STATUS_ONGOING)
        # 输入 state 不得被修改
        self.assertEqual(self.state.relationship, 50)
        self.assertEqual(self.state.history, [])

    def test_delta_clamped_to_max_abs(self):
        self.runtime = AdvRuntime(self.skin, FakeLLM([girl_json(delta=10)]))
        after = self.runtime.resolve_turn(self.state, talk("超级开心"), Rng(2)).state_after
        self.assertEqual(after.relationship, 53)  # 50 + clamp(10→3)

    def test_relationship_clamped_to_bounds(self):
        state = AdvState(girl_name="雨宫澪", relationship=99, turn=3)
        self.runtime = AdvRuntime(self.skin, FakeLLM([girl_json(delta=3)]))
        after = self.runtime.resolve_turn(state, talk("晚安"), Rng(3)).state_after
        self.assertEqual(after.relationship, 100)

    def test_node_event_fires_on_stage_change(self):
        state = AdvState(girl_name="雨宫澪", relationship=55, turn=5, node="meet")
        outcome = self.runtime.resolve_turn(state, talk("外面下雨了"), Rng(4))
        types = [e.type for e in outcome.events]
        self.assertIn("node_event", types)
        self.assertEqual(outcome.state_after.node, "rain")
        node_event = next(e for e in outcome.events if e.type == "node_event")
        self.assertEqual(node_event.detail["node"], "rain")
        self.assertIn("大雨", node_event.message)

    def test_no_node_event_inside_stage(self):
        state = AdvState(girl_name="雨宫澪", relationship=55, turn=6, node="rain")
        outcome = self.runtime.resolve_turn(state, talk("雨好大"), Rng(5))
        self.assertNotIn("node_event", [e.type for e in outcome.events])

    def test_intent_recorded_only_after_window(self):
        # 第 20 回合:即使说了"喜欢"也不计入(窗口从 21 开始)
        state = AdvState(girl_name="雨宫澪", relationship=60, turn=19, node="understood")
        outcome = self.runtime.resolve_turn(state, talk("我喜欢你"), Rng(6))
        self.assertEqual(outcome.state_after.intent_turn, 0)
        # 第 21 回合:命中并记录
        state = AdvState(girl_name="雨宫澪", relationship=60, turn=20, node="choice")
        outcome = self.runtime.resolve_turn(state, talk("我喜欢你"), Rng(7))
        self.assertEqual(outcome.state_after.intent_turn, 21)
        self.assertIn("intent_hint", [e.type for e in outcome.events])

    def _play_to_end(self, deltas, intent=True,
                     last_line="我喜欢你。做我女朋友吧。"):
        runtime = AdvRuntime(self.skin, FakeLLM(make_loop_responses(deltas)))
        state = runtime.initial_state()
        agent = ScriptedTalkAgent(
            [line_at(t + 1, intent=intent) for t in range(24)] + [last_line])
        status = STATUS_ONGOING
        for _ in range(25):
            actions = runtime.available_actions(state)
            outcome = runtime.resolve_turn(state, agent.choose(state, actions), Rng(11))
            state = outcome.state_after
            status = outcome.status
        return runtime, state, status

    def test_ending_a_love_requires_relationship_and_intent(self):
        runtime, state, status = self._play_to_end([3])
        self.assertEqual(status, STATUS_WON)
        self.assertEqual(state.ending_id, "love")
        self.assertGreaterEqual(state.intent_turn, 21)  # 意图窗口内命中(记录最后一次)
        self.assertEqual(runtime.available_actions(state), [])

    def test_ending_b_special_without_intent(self):
        # 关系足够高但从未告白:全程普通玩家台词(不命中意图关键词)
        runtime, state, status = self._play_to_end(
            [3], intent=False, last_line="纪念册就交给你了,保重。")
        self.assertEqual(status, STATUS_WON)
        self.assertEqual(state.ending_id, "special")
        self.assertEqual(state.intent_turn, 0)

    def test_ending_c_friend(self):
        runtime, state, status = self._play_to_end(
            [0], last_line="那明天见。")
        self.assertEqual(status, STATUS_DRAW)
        self.assertEqual(state.ending_id, "friend")

    def test_ending_d_part(self):
        runtime, state, status = self._play_to_end(
            [-3], last_line="随便你吧。")
        self.assertEqual(status, STATUS_LOST)
        self.assertEqual(state.ending_id, "part")

    def test_game_not_over_before_final_turn(self):
        state = AdvState(girl_name="雨宫澪", relationship=90, turn=24,
                         node="choice", intent_turn=21)
        self.assertEqual(self.runtime.status_of(state), STATUS_ONGOING)
        state.turn = 25
        self.assertEqual(self.runtime.status_of(state), STATUS_WON)

    def test_malformed_json_keeps_raw_text(self):
        # 契约破坏但输出确是中文:模型真实文本直接当回应,打分 0(不编造数值)
        self.runtime = AdvRuntime(
            self.skin, FakeLLM(["(她低头继续整理照片,没有回答。)"]))
        outcome = self.runtime.resolve_turn(self.state, talk("在吗"), Rng(8))
        after = outcome.state_after
        self.assertEqual(after.last_reply, "(她低头继续整理照片,没有回答。)")
        self.assertEqual(after.relationship, 50)  # 打分未知 → 0

    def test_non_chinese_contract_break_falls_back_to_chinese(self):
        # 契约要求中文,模型却输出纯英文:英文挂在 Chinese 对白标签会被念错,
        # 弃用并走中文兜底(守卫方向与 romance 的英文契约相反)
        self.runtime = AdvRuntime(
            self.skin, FakeLLM(["She looks away and keeps sorting the photos."]))
        outcome = self.runtime.resolve_turn(self.state, talk("在想什么"), Rng(9))
        self.assertEqual(
            outcome.state_after.last_reply,
            "雨宫澪安静地看着你,片刻后轻轻点了点头。")

    def test_llm_failure_first_turn_propagates(self):
        self.runtime = AdvRuntime(self.skin, FakeLLM([LLMError("down")] * 3))
        with self.assertRaises(LLMError):
            self.runtime.resolve_turn(self.state, talk("在吗"), Rng(10))

    def test_llm_failure_midgame_degrades(self):
        runtime = AdvRuntime(self.skin, FakeLLM([girl_json(delta=2)] + [LLMError("x")] * 6))
        first = runtime.resolve_turn(self.state, talk("你好"), Rng(12))
        second = runtime.resolve_turn(first.state_after, talk("在吗"), Rng(13))
        self.assertEqual(second.status, STATUS_ONGOING)
        self.assertEqual(second.state_after.relationship, 52)  # 降级回应打分 0

    def test_empty_line_rejected(self):
        with self.assertRaises(ValueError):
            self.runtime.resolve_turn(self.state, Action(id=ACTION_TALK.id, label=""), Rng(14))
        with self.assertRaises(ValueError):
            self.runtime.resolve_turn(self.state, ACTION_TALK, Rng(14))  # label 还是默认"说话"

    def test_system_prompt_carries_story_state(self):
        llm = FakeLLM([girl_json()])
        runtime = AdvRuntime(self.skin, llm)
        # 从第 5 回合结算进第 6 回合:正是节点二(rain)的第一回合
        state = AdvState(girl_name="雨宫澪", relationship=52, turn=5, node="meet",
                         mood="warm", history=[("第一句", "嗯。")])
        runtime.resolve_turn(state, talk("第二句"), Rng(16))
        messages = llm.calls[-1]["messages"]
        system = messages[0]["content"]
        # system 只挂角色/关系/情绪与契约;剧情改由对话记录承载
        self.assertIn("雨宫澪", system)
        self.assertNotIn("【当前剧情】", system)
        self.assertIn("Relationship:52", system)
        self.assertIn("relationship_delta", system)
        self.assertIn("回应必须使用中文", system)
        # 历史里的对话逐组交替;第 1 回合是节点一首回合 → 玩家消息带剧情旁白
        self.assertTrue(messages[1]["content"].startswith("(剧情·节点一 · 相识)"))
        self.assertIn("第一次两个人留下来", messages[1]["content"])
        self.assertEqual(messages[2], {"role": "assistant", "content": "嗯。"})
        # 第 6 回合 = rain 首回合:本回合事件与玩家台词一起在最后一条 user
        last = messages[-1]["content"]
        self.assertTrue(last.startswith("【刚发生的剧情事件"))
        self.assertIn("下起大雨", last)
        self.assertIn("第二句", last)
        self.assertIn("输出格式再次强调", last)
        self.assertIn("relationship_delta", last)

    def test_history_weaves_past_node_events(self):
        """历史剧情也织进对话记录:第 6 回合的玩家消息带节点二旁白前缀。"""
        llm = FakeLLM([girl_json()])
        runtime = AdvRuntime(self.skin, llm)
        history = [(f"第{t}回合的话", f"她的第{t}句") for t in range(1, 7)]
        state = AdvState(girl_name="雨宫澪", relationship=58, turn=6, node="rain",
                         history=history)
        runtime.resolve_turn(state, talk("雨还在下"), Rng(17))
        messages = llm.calls[-1]["messages"][1:]  # 去 system
        # 7 条 = 6 组历史 + 玩家本句;第 6 组(user)带节点二事件旁白
        self.assertEqual(len(messages), 13)
        self.assertTrue(messages[0]["content"].startswith("(剧情·节点一 · 相识)"))
        self.assertTrue(messages[10]["content"].startswith("(剧情·节点二 · 靠近)"))
        self.assertIn("下起大雨", messages[10]["content"])
        self.assertIn("第6回合的话", messages[10]["content"])
        # 节点中段:最后一条 user 就是玩家本句,没有事件重排块
        self.assertTrue(messages[-1]["content"].startswith("雨还在下"))
        self.assertNotIn("刚发生的剧情事件", messages[-1]["content"])


    def test_llm_action_field_ignored_mood_actions_used(self):
        """LLM 只决定 reply/mood/Δ;动作一律取心情档位短语(即使契约带 action)。"""
        skin = load_skin()
        llm = FakeLLM([girl_json(delta=2, mood="warm",
                                 action="pressing the photo print flat with both palms, "
                                        "eyes softening as she leans a little closer")])
        runtime = AdvRuntime(skin, llm)
        director = AdvVisualDirector(skin)
        state = runtime.initial_state()
        outcome = runtime.resolve_turn(state, talk("这张送你了"), Rng(30))
        self.assertEqual(outcome.state_after.mood, "warm")
        self.assertEqual(outcome.state_after.last_action,
                         skin.mood_actions["warm"])
        self.assertNotIn("pressing the photo print",
                         outcome.state_after.last_action)
        result = TurnResult(turn_id="turn-001", index=0, action=talk("这张送你了"),
                            state_before=state, state_after=outcome.state_after,
                            outcome=outcome)
        script = director.build(result, director.initial_visual_state())
        self.assertIn(skin.mood_actions["warm"], script.action)  # 档位短语入演出

    def test_action_missing_falls_back_to_mood_phrase(self):
        """契约里没有 action(旧桩/降级)→ 回退心情档位短语,不产生空句子。"""
        skin = load_skin()
        llm = FakeLLM([girl_json(delta=1, mood="flustered")])  # 无 action 字段
        runtime = AdvRuntime(skin, llm)
        director = AdvVisualDirector(skin)
        state = runtime.initial_state()
        outcome = runtime.resolve_turn(state, talk("在吗"), Rng(31))
        self.assertEqual(outcome.state_after.last_action,
                         skin.mood_actions["flustered"])
        result = TurnResult(turn_id="turn-001", index=0, action=talk("在吗"),
                            state_before=state, state_after=outcome.state_after,
                            outcome=outcome)
        script = director.build(result, director.initial_visual_state())
        self.assertIn(skin.mood_actions["flustered"], script.action)


class TestAdvVisual(unittest.TestCase):
    def setUp(self):
        # 本类验证完整 prompt 组装(锚定句/收尾句都在);
        # 最终皮肤 before_rain_mio 自身省略这三段,见 TestPromptSplit
        self.skin = dataclasses.replace(load_skin(), omit_prompt_sections=())

    def _build(self, deltas=(2,), moods=("warm",), lines=("今天照片好多啊",)):
        captions = {}
        llm = FakeLLM([girl_json(delta=deltas[0], mood=moods[0])])
        runtime = AdvRuntime(self.skin, llm)
        director = AdvVisualDirector(self.skin, captions=captions)
        state = runtime.initial_state()
        outcome = runtime.resolve_turn(state, talk(lines[0]), Rng(20))
        result = TurnResult(
            turn_id="turn-001", index=0, action=talk(lines[0]),
            state_before=state, state_after=outcome.state_after, outcome=outcome,
        )
        return director, result, captions, outcome

    def test_dialogue_verbatim_and_anchor(self):
        director, result, captions, _ = self._build()
        script = director.build(result, director.initial_visual_state())
        self.assertEqual(len(script.dialogue), 1)
        # 台词逐字来自结算层(视觉层不得改写)
        self.assertEqual(script.dialogue[0].text, "……嗯。你今天也很吵。")
        self.assertEqual(script.dialogue[0].speaker_id, "S1")
        self.assertEqual(script.dialogue[0].language, "Chinese")
        self.assertIn("<Picture 1>", script.first_frame_anchor)
        self.assertTrue(script.use_first_frame)
        self.assertEqual(script.meta["node"], "meet")

    def test_node_scene_change_switches_to_node_frame(self):
        """换节点:皮肤声明了节点首帧 → 首帧链切到该图(i2v),锚定句对应新图。"""
        captions = {}
        llm = FakeLLM([girl_json(delta=1, mood="calm")])
        runtime = AdvRuntime(self.skin, llm)
        director = AdvVisualDirector(self.skin, captions=captions)
        visual_state = director.initial_visual_state()
        old_frame = visual_state.current_frame  # None(未渲染)
        state = AdvState(girl_name="雨宫澪", relationship=55, turn=5, node="meet")
        outcome = runtime.resolve_turn(state, talk("外面下大雨了"), Rng(21))
        result = TurnResult(turn_id="turn-006", index=5, action=talk("外面下大雨了"),
                            state_before=state, state_after=outcome.state_after,
                            outcome=outcome)
        script = director.build(result, visual_state)
        self.assertTrue(script.use_first_frame)      # 有节点首帧 → 仍走 i2v
        self.assertEqual(script.meta["node"], "rain")
        # 首帧链切到节点二的首帧图
        self.assertIsNotNone(visual_state.current_frame)
        self.assertNotEqual(visual_state.current_frame, old_frame)
        self.assertIn("mio2", str(visual_state.current_frame))
        # 锚定句对应新图(<Picture 1> + 节点场景)
        self.assertIn("<Picture 1>", script.first_frame_anchor)
        self.assertIn("heavy rain", script.first_frame_anchor)
        self.assertIn("rain", script.action)         # 到达演出的开场短语

    def test_ending_script_uses_actual_ending(self):
        captions = {}
        llm = FakeLLM([girl_json(delta=3, mood="warm")])
        runtime = AdvRuntime(self.skin, llm)
        director = AdvVisualDirector(self.skin, captions=captions)
        visual_state = director.initial_visual_state()
        state = AdvState(girl_name="雨宫澪", relationship=99, turn=24, node="choice",
                         intent_turn=21)
        outcome = runtime.resolve_turn(state, talk("我喜欢你。等我。"), Rng(22))
        self.assertEqual(outcome.state_after.ending_id, "love")
        result = TurnResult(turn_id="turn-025", index=24, action=talk("我喜欢你。等我。"),
                            state_before=state, state_after=outcome.state_after,
                            outcome=outcome)
        director.build(result, visual_state)
        script = director.ending_script(STATUS_WON, visual_state)
        self.assertIsNotNone(script)
        self.assertEqual(script.meta["ending"], "love")
        self.assertEqual(captions[STATUS_WON], "◆ Ending A · 恋爱 ◆")


class TestAdvLoop(unittest.TestCase):
    """完整闭环:FakeLLM 结算 + FakeRenderer 演出,25 回合跑通一局。"""

    def test_full_loop_fake_renderer_love_ending(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            initial = tmpdir / "classroom.png"
            initial.write_bytes(TINY_PNG)

            skin = load_skin()
            llm = FakeLLM(make_loop_responses([3]))
            parts = build_adv_game(skin, llm)
            parts.visual_state.current_frame = initial

            lines = [line_at(t + 1) for t in range(24)] + ["我喜欢你。做我女朋友吧。"]
            loop = GameLoop(
                runtime=parts.runtime,
                agent=ScriptedTalkAgent(lines),
                visual_director=parts.visual_director,
                renderer=FakeRenderer(output_dir=tmpdir),
                visual_state=parts.visual_state,
                output_dir=tmpdir / "session",
                base_seed=44,
                max_turns=25,
                ending_director=parts.ending_director,
                hud_of=parts.hud_of,
            )
            summary = loop.run()

            self.assertEqual(summary.finished_reason, "game_over")
            self.assertEqual(summary.turns_played, 25)
            self.assertEqual(summary.final_status, STATUS_WON)
            self.assertIsNotNone(summary.ending)          # Ending A 结束动画
            self.assertTrue(Path(summary.ending.video_path).exists())
            # 节点推进:第 6/11/16/21 回合换了剧情节点
            nodes = [record.meta.get("node") for record in summary.records]
            self.assertEqual(nodes[0], "meet")
            self.assertEqual(nodes[5], "rain")
            self.assertEqual(nodes[10], "discovered")
            self.assertEqual(nodes[15], "understood")
            self.assertEqual(nodes[20], "choice")
            # 每个回合都有视频与 HUD;caption 带回合状态摘要
            for record in summary.records:
                self.assertTrue(Path(record.video_path).exists())
                self.assertIn("第", record.caption)
            # 场景演进:节点二起走 t2v,首帧链被 fake renderer 延续
            self.assertTrue(all(record.render_error is None for record in summary.records))
            # 终局字幕被实际结局改写
            self.assertEqual(parts.ending_captions[STATUS_WON], "◆ Ending A · 恋爱 ◆")

    def test_full_loop_part_ending_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            skin = load_skin()
            llm = FakeLLM(make_loop_responses([-3]))
            parts = build_adv_game(skin, llm)
            loop = GameLoop(
                runtime=parts.runtime,
                agent=ScriptedTalkAgent([line_at(t + 1) for t in range(25)]),
                visual_director=parts.visual_director,
                renderer=FakeRenderer(output_dir=tmpdir),
                visual_state=parts.visual_state,
                output_dir=tmpdir / "session",
                base_seed=45,
                max_turns=25,
                ending_director=parts.ending_director,
            )
            summary = loop.run()
            self.assertEqual(summary.final_status, STATUS_LOST)
            self.assertEqual(summary.records[-1].status, STATUS_LOST)


class TestAdvAgents(unittest.TestCase):
    def test_cli_input_becomes_action(self):
        skin = load_skin()
        agent = AdvCliAgent(input_fn=lambda _: "你会舍不得毕业吗?",
                            print_fn=lambda *_: None)
        action = agent.choose(AdvState(girl_name="雨宫澪", relationship=50), [ACTION_TALK])
        self.assertEqual(action.id, ACTION_TALK.id)
        self.assertEqual(action.label, "你会舍不得毕业吗?")

    def test_llm_agent_builds_line(self):
        agent = AdvLLMAgent(FakeLLM(['"你会来毕业典礼吗?"']), load_skin())
        action = agent.choose(AdvState(girl_name="雨宫澪", relationship=50, turn=3), [ACTION_TALK])
        self.assertEqual(action.label, "你会来毕业典礼吗?")

    def test_llm_agent_reacts_to_node_event(self):
        """玩家代理的消息历史与女主同构:对话交替 + 剧情事件织入。"""
        skin = load_skin()
        llm = FakeLLM(['"外面雨这么大,今天谁都走不了了吧。"',
                       '"那把伞放门口吧,免得明天又忘。"'])
        agent = AdvLLMAgent(llm, skin)
        # 第 5 回合结算后(turn=5),即将进入第 6 回合 = rain 节点首回合:
        # 本回合事件放进最后一条 user;历史里已有节点一的旁白
        state = AdvState(girl_name="雨宫澪", relationship=55, turn=5, node="meet",
                         mood="calm", history=[("上句", "她的回答")])
        agent.choose(state, [ACTION_TALK])
        messages = llm.calls[-1]["messages"]
        self.assertTrue(messages[1]["content"].startswith("(剧情·节点一 · 相识)"))
        self.assertEqual(messages[1]["role"], "user")     # 开场旁白
        self.assertEqual(messages[2], {"role": "assistant", "content": "上句"})
        # 她的回答(user)末尾并入了处境提示(保持 user/assistant 严格交替)
        self.assertTrue(messages[3]["content"].startswith("她的回答"))
        situation = messages[3]["content"]
        self.assertIn("本回合刚发生", situation)          # 节点首回合 → 事件
        self.assertIn("下起大雨", situation)
        self.assertIn("轮到你说话", situation)
        # 中段(turn=7 → 第 8 回合,已在 rain 内):rain 事件已织进历史,
        # 最后一条 user 只是"轮到你接话"的处境提示
        history = [(f"话{t}", f"答{t}") for t in range(1, 8)]
        agent.choose(AdvState(girl_name="雨宫澪", relationship=55, turn=7,
                              node="rain", mood="calm", history=history),
                     [ACTION_TALK])
        messages = llm.calls[-1]["messages"]
        woven = [m for m in messages if "剧情·节点二" in m["content"]]
        self.assertEqual(len(woven), 1)                   # rain 事件织入一次
        self.assertIn("下起大雨", woven[0]["content"])
        situation = messages[-1]["content"]
        self.assertIn("当前阶段:节点二", situation)
        self.assertNotIn("本回合刚发生", situation)


class TestPromptSplit(unittest.TestCase):
    """prompt 拆分:游戏专属内容在皮肤,通用框架在代码;历史带说话人前缀。"""

    def test_skin_prompt_fields_parsed(self):
        skin = load_skin()
        self.assertEqual(skin.player_persona, "一名普通的高三男生")
        self.assertIn("毕业纪念册", skin.player_scenario)
        self.assertEqual(skin.mood_notes.get("calm"), "平静,仅限纯事务应答")

    def test_omit_prompt_sections_on_final_skin(self):
        """最终皮肤省略锚定句/过渡句/收尾句:prompt 不含这三段。"""
        from engine.visual.compiler import PromptCompiler

        skin = load_skin()  # before_rain_mio(omit: anchor/arrival/end_state)
        self.assertTrue(skin.omits("anchor"))
        llm = FakeLLM([girl_json(delta=1, mood="flustered")])
        runtime = AdvRuntime(skin, llm)
        director = AdvVisualDirector(skin)
        state = runtime.initial_state()
        outcome = runtime.resolve_turn(state, talk("在吗"), Rng(31))
        result = TurnResult(turn_id="turn-001", index=0, action=talk("在吗"),
                            state_before=state, state_after=outcome.state_after,
                            outcome=outcome)
        script = director.build(result, director.initial_visual_state())
        self.assertEqual(script.first_frame_anchor, "")   # 锚定句省略
        self.assertEqual(script.end_state, "")            # 收尾句省略
        prompt = PromptCompiler().compile(script, has_first_frame=True)
        self.assertNotIn("The clip ends as", prompt)      # 收尾句不再渲染
        self.assertIn(skin.mood_actions["flustered"], prompt)  # 神态动作仍在

    def test_player_system_prompt_uses_skin_content(self):
        skin = load_skin()
        llm = FakeLLM([girl_json()])
        agent = AdvLLMAgent(llm, skin)
        state = AdvState(girl_name=skin.girl_name, relationship=55, turn=5,
                         node="meet", mood="calm", history=[("上句", "她的回答")])
        agent.choose(state, [ACTION_TALK])
        system = llm.calls[-1]["messages"][0]["content"]
        self.assertIn(skin.player_persona, system)       # 皮肤注入男主角人设
        self.assertIn(skin.player_scenario, system)      # 皮肤注入共同处境


if __name__ == "__main__":
    unittest.main()
