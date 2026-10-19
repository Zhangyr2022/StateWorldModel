"""开放世界 Demo 测试:双回合模式(INTERACTIVE/AUTO)、导演输出、主线节奏。

规则契约见 game/openworld/ 模块说明:有值得选择的事就让玩家选择;
没有就让 AI 自动探索一幕;AUTO 后必须回到交互;主线节奏只数交互幕。
"""

import base64
import json
import tempfile
import unittest
from pathlib import Path

from engine.llm.client import LLMError
from engine.loop import GameLoop
from engine.renderer.fake import FakeRenderer
from engine.runtime import STATUS_ONGOING, STATUS_WON, Action, Rng, TurnResult
from game.openworld import (
    OpenworldAutoAgent,
    OpenworldCliAgent,
    OpenworldRuntime,
    build_openworld_visual_script,
    clip_hud_text,
    ending_openworld_script,
    initial_openworld_visual_state,
)
from game.openworld.rules import (
    MIN_INTERACTIVE_BEFORE_CLIMAX,
    MIN_INTERACTIVE_PER_STAGE,
    STAGE_BACKSTOP_EXTRA,
)

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

PROPOSER_JSON = json.dumps({
    "has_opportunity": True,
    "reason": "a tavern door stands open on the main street",
    "actions": ["走进酒馆打听消息", "检查驿站的布告栏", "爬上水塔眺望全镇"],
    "location_note": "西部小镇主街",
}, ensure_ascii=False)

NO_OPP_JSON = json.dumps({
    "has_opportunity": False,
    "reason": "an empty highway with no landmarks around",
    "actions": [],
    "location_note": "空旷公路",
}, ensure_ascii=False)

DIRECTOR_JSON = json.dumps({
    "story": "酒馆老板听到镇名后神色一变,压低声音说起矿场的传闻。",
    "consequence": (
        "Inside the dim tavern, the explorer leans on the counter as the "
        "keeper's face darkens; lamplight swings and a wanted poster flutters."
    ),
    "shot2": "The keeper slides a rusted key across the counter.",
    "cut_at_s": 2.5,
    "plot_complete": False,
    "climax": "",
}, ensure_ascii=False)

# 导演在自动探索回合输出 stage_complete=true 也必须被规则层忽略(§5.2:
# AUTO 不自动完成关键任务节点)。
AUTO_DIRECTOR_JSON = json.dumps({
    "story": "旅人沿公路前行,转过弯后看见一座亮着灯的废弃加油站。",
    "consequence": (
        "The explorer walks along the empty highway, rounds a bend and slows "
        "before a derelict gas station, its lone window glowing at dusk."
    ),
    "shot2": "",
    "cut_at_s": 2.5,
    "stage_complete": True,
    "plot_complete": False,
    "climax": "",
}, ensure_ascii=False)

DIRECTOR_STAGE_JSON = json.dumps({
    "story": "醉汉吐露了黑荆藏身峡谷的线索。",
    "consequence": (
        "The drunk leans close and whispers the name of the canyon; the "
        "explorer memorizes it as the fire gutters low."
    ),
    "shot2": "",
    "cut_at_s": 2.5,
    "stage_complete": True,
    "plot_complete": False,
    "climax": "",
}, ensure_ascii=False)

FINALE_JSON = json.dumps({
    "story": "钥匙打开矿场大门,所有线索汇聚,冒险抵达终点。",
    "consequence": (
        "The mine gate swings open in a shower of rust as golden light spills "
        "over the explorer's stunned face."
    ),
    "shot2": "",
    "cut_at_s": 2.5,
    "plot_complete": True,
    "climax": "门后不是矿藏,而是整座小镇失落的历史。",
}, ensure_ascii=False)


class FakeLLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, seed=0, max_tokens=256, timeout_s=None, images=None):
        self.calls.append({"messages": messages, "seed": seed, "images": images})
        if not self.responses:
            raise LLMError("fake LLM exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _placeholder_scene() -> str:
    """真实场景画在 HF 资产仓(assets 缺失时生成占位图,测试保持零资产行)。"""
    candidate = Path(__file__).resolve().parents[1] / "assets" / "openworld" / "western.png"
    if candidate.is_file():
        return str(candidate)
    import tempfile
    from PIL import Image

    fd, path = tempfile.mkstemp(suffix=".png")
    Path(path).unlink(missing_ok=True)
    Image.new("RGB", (1280, 720), (120, 110, 100)).save(path)
    return path


def make_runtime(llm, scene_image: str | None = None, **kwargs):
    scene_image = scene_image or _placeholder_scene()
    return OpenworldRuntime(llm, scene_image=scene_image, **kwargs)


def _resolve(runtime, state, llm, rng, *, action=None):
    """available_actions + resolve_turn 的一步(测试里最常用的组合)。"""
    if action is None:
        action = runtime.available_actions(state)[0]
    return runtime.resolve_turn(state, action, rng)


class TestOpenworldRuntime(unittest.TestCase):
    def test_propose_actions_via_vlm(self):
        llm = FakeLLM([PROPOSER_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        actions = runtime.available_actions(state)
        self.assertEqual(len(actions), 3)
        self.assertIn("酒馆", actions[0].label)
        self.assertNotIn(state.pending_mode, ("", "auto"))  # 有机会 → 交互回合
        # VLM 确实收到了图像(首图)
        self.assertEqual(llm.calls[0]["images"], [runtime.initial_scene_image])
        self.assertEqual(state.scene_name, "西部小镇主街")

    def test_fallback_actions_when_llm_down(self):
        # VLM 不可用时保守回到交互 + 通用探索选项,绝不静默进入 AUTO
        runtime = make_runtime(FakeLLM([LLMError("down")]))
        state = runtime.initial_state()
        actions = runtime.available_actions(state)
        self.assertEqual(len(actions), 3)
        self.assertEqual(state.pending_mode, "interactive")

    def test_resolve_produces_consequence_and_multishot(self):
        llm = FakeLLM([PROPOSER_JSON, DIRECTOR_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        actions = runtime.available_actions(state)

        outcome = runtime.resolve_turn(state, actions[0], Rng(1))
        after = outcome.state_after

        types = [e.type for e in outcome.events]
        self.assertIn("story_beat", types)
        self.assertIn("consequence", types)
        self.assertEqual(after.story, ["酒馆老板听到镇名后神色一变,压低声音说起矿场的传闻。"])
        self.assertIn("tavern", after.last_consequence)
        self.assertEqual(after.last_mode, "interactive")
        self.assertEqual(after.interactive_turns, 1)
        detail = outcome.events[-1].detail
        self.assertTrue(detail["shot2"])  # 第二镜头
        self.assertEqual(detail["cut_at_s"], 2.5)
        self.assertEqual(detail["mode"], "interactive")
        self.assertEqual(outcome.status, STATUS_ONGOING)

    def test_plot_complete_wins_at_twelve_interactive_turns(self):
        # 响应按真实消耗顺序排列:propose/direct 交替。
        # 结局最早出现在 MIN_INTERACTIVE_BEFORE_CLIMAX 个交互幕:前 11 幕喂
        # 普通推进,第 12 个交互幕导演返回 FINALE(plot_complete=true)→ 好结局。
        llm = FakeLLM([PROPOSER_JSON, DIRECTOR_JSON] * 11
                      + [PROPOSER_JSON, FINALE_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        rng = Rng(3)
        for _ in range(12):
            outcome = _resolve(runtime, state, llm, rng)
            state = outcome.state_after
        self.assertEqual(state.interactive_turns, MIN_INTERACTIVE_BEFORE_CLIMAX)
        self.assertEqual(outcome.status, STATUS_WON)
        self.assertEqual(outcome.state_after.ending, "climax")
        self.assertTrue(outcome.state_after.climax_text)

    def test_plot_complete_too_early_is_ignored(self):
        # 第 1 个交互幕就宣布 plot_complete:未达最早结局幕数,继续探索
        llm = FakeLLM([PROPOSER_JSON, FINALE_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        outcome = _resolve(runtime, state, llm, Rng(9))
        self.assertEqual(outcome.status, STATUS_ONGOING)
        self.assertEqual(outcome.state_after.ending, "")

    def test_status_never_lost_midway(self):
        runtime = make_runtime(FakeLLM([]))
        self.assertEqual(runtime.status_of(runtime.initial_state()), "ongoing")


class TestOpenworldAutoMode(unittest.TestCase):
    """双回合模式契约(docs §3/§5/§7)。"""

    def test_no_opportunity_proposes_single_auto(self):
        llm = FakeLLM([NO_OPP_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        actions = runtime.available_actions(state)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].id, "auto")
        self.assertEqual(actions[0].description, "an empty highway with no landmarks around")
        self.assertEqual(state.pending_mode, "auto")
        self.assertEqual(state.scene_name, "空旷公路")

    def test_auto_turn_ignores_director_stage_flags(self):
        # AUTO 不替玩家推进主线:导演即使输出 stage_complete=true 也被忽略
        llm = FakeLLM([NO_OPP_JSON, AUTO_DIRECTOR_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        outcome = _resolve(runtime, state, llm, Rng(2))
        after = outcome.state_after
        self.assertEqual(outcome.status, STATUS_ONGOING)
        self.assertEqual(after.last_mode, "auto")
        self.assertEqual(after.quest_stage, 0)
        self.assertEqual(after.stage_interactive, 0)
        self.assertEqual(after.interactive_turns, 0)
        self.assertEqual(after.stamina, 40 - 12)  # AUTO 也是真实的一幕,消耗体力
        self.assertTrue(after.story)              # 剧情照常累积
        types = [e.type for e in outcome.events]
        self.assertIn("auto_explore", types)
        self.assertNotIn("player_action", types)
        self.assertEqual(outcome.events[-1].detail["mode"], "auto")

    def test_auto_forces_interactive_next_turn(self):
        # 硬性规则:AUTO 后的下一回合必须 INTERACTIVE——即使 VLM 仍判定
        # 无机会,也要给出兜底的三种探索策略(§5.4)
        llm = FakeLLM([NO_OPP_JSON, AUTO_DIRECTOR_JSON, NO_OPP_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        outcome = _resolve(runtime, state, llm, Rng(4))
        state = outcome.state_after
        self.assertEqual(state.last_mode, "auto")

        actions = runtime.available_actions(state)
        self.assertEqual(len(actions), 3)
        self.assertEqual(state.pending_mode, "interactive")
        self.assertEqual([a.label for a in actions],
                         ["沿着大路继续向前探索", "仔细观察四周的环境",
                          "走进附近显眼的建筑一探究竟"])

    def test_double_auto_rejected(self):
        llm = FakeLLM([NO_OPP_JSON, AUTO_DIRECTOR_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        outcome = _resolve(runtime, state, llm, Rng(6))
        state = outcome.state_after
        with self.assertRaises(ValueError):
            runtime.resolve_turn(state, Action(id="auto", label="▶ 自动探索"), Rng(6))

    def test_custom_action_overrides_pending_auto(self):
        # 玩家自由输入可以否决 AUTO:重大决定始终属于玩家(§2.5)
        llm = FakeLLM([NO_OPP_JSON, DIRECTOR_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        runtime.available_actions(state)  # 判定为 AUTO
        outcome = runtime.resolve_turn(
            state, Action(id="custom", label="蹲下检查路面的车辙痕迹"), Rng(8))
        after = outcome.state_after
        self.assertEqual(after.last_mode, "interactive")
        self.assertEqual(after.interactive_turns, 1)
        self.assertEqual(after.history[0][0], "蹲下检查路面的车辙痕迹")

    def test_stage_advance_needs_min_interactive_turns(self):
        # 导演每幕都宣布 stage_complete,也要演满最少交互幕数才推进(节奏)
        llm = FakeLLM([PROPOSER_JSON, DIRECTOR_STAGE_JSON] * MIN_INTERACTIVE_PER_STAGE)
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        rng = Rng(11)
        for i in range(MIN_INTERACTIVE_PER_STAGE - 1):
            outcome = _resolve(runtime, state, llm, rng)
            state = outcome.state_after
            self.assertEqual(state.quest_stage, 0)
        outcome = _resolve(runtime, state, llm, rng)
        self.assertEqual(outcome.state_after.quest_stage, 1)
        self.assertEqual(outcome.state_after.stage_interactive, 0)  # 推进后清零

    def test_stage_backstop_after_extra_interactive_turns(self):
        # 保底:导演迟迟不宣布,该阶段交互演满最少幕数 +2 也视为达成
        total = MIN_INTERACTIVE_PER_STAGE + STAGE_BACKSTOP_EXTRA
        llm = FakeLLM([PROPOSER_JSON, DIRECTOR_JSON] * total)
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        rng = Rng(13)
        for _ in range(total - 1):
            outcome = _resolve(runtime, state, llm, rng)
            state = outcome.state_after
            self.assertEqual(state.quest_stage, 0)
        outcome = _resolve(runtime, state, llm, rng)
        self.assertEqual(outcome.state_after.quest_stage, 1)

    def test_auto_turns_do_not_count_toward_pacing(self):
        # 交叠 AUTO 的 5 幕只有 3 个交互幕:即使导演每幕都宣布达成,
        # 主线也不得凭幕数(含 AUTO)推进(§7)
        script = [PROPOSER_JSON, DIRECTOR_STAGE_JSON, NO_OPP_JSON, AUTO_DIRECTOR_JSON] * 2 \
            + [PROPOSER_JSON, DIRECTOR_STAGE_JSON]
        llm = FakeLLM(script)
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        rng = Rng(15)
        for _ in range(5):
            outcome = _resolve(runtime, state, llm, rng)
            state = outcome.state_after
        self.assertEqual(state.turn, 5)
        self.assertEqual(state.interactive_turns, 3)
        self.assertEqual(state.stage_interactive, 3)
        self.assertEqual(state.quest_stage, 0)


class TestOpenworldVisual(unittest.TestCase):
    def make_script(self, director_json=DIRECTOR_JSON, proposer_json=PROPOSER_JSON,
                    action=None):
        llm = FakeLLM([proposer_json, director_json])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        resolved_action = action or runtime.available_actions(state)[0]
        outcome = runtime.resolve_turn(state, resolved_action, Rng(5))
        visual_state = initial_openworld_visual_state("western")
        turn_result = TurnResult(
            turn_id="turn-001", index=0, action=resolved_action,
            state_before=state, state_after=outcome.state_after, outcome=outcome,
        )
        return build_openworld_visual_script(turn_result, visual_state)

    def test_consequence_becomes_action_text(self):
        script = self.make_script()
        self.assertIn("tavern", script.action)  # VLM 的画面描述即视频输入文本
        self.assertIn("<Picture 1>", script.first_frame_anchor)
        self.assertIn("vantage point", script.end_state)  # 交互回合的收尾

    def test_auto_script_ends_on_unresolved_opportunity(self):
        # AUTO 演出的结尾落在"明确但尚未被处理的机会"上(§5.3)
        script = self.make_script(director_json=AUTO_DIRECTOR_JSON,
                                  proposer_json=NO_OPP_JSON,
                                  action=Action(id="auto", label="▶ 自动探索"))
        self.assertIn("gas station", script.action)
        self.assertIn("not yet acted on", script.end_state)

    def test_multishot_compiles_with_timestamp(self):
        from engine.visual.compiler import PromptCompiler

        script = self.make_script()
        self.assertEqual(len(script.shots), 1)
        prompt = PromptCompiler().compile(script, has_first_frame=True)
        self.assertIn("[Shot 2] At 00:02.500, the camera cuts to", prompt)

    def test_ending_script_uses_climax(self):
        llm = FakeLLM([PROPOSER_JSON, DIRECTOR_JSON,
                       PROPOSER_JSON, DIRECTOR_JSON,
                       PROPOSER_JSON, FINALE_JSON])
        runtime = make_runtime(llm)
        state = runtime.initial_state()
        rng = Rng(6)
        for _ in range(3):
            outcome = _resolve(runtime, state, llm, rng)
            state = outcome.state_after
        visual_state = initial_openworld_visual_state("western")
        visual_state.meta["exploration_state"] = state
        ending = ending_openworld_script(STATUS_WON, visual_state)
        self.assertIsNotNone(ending)
        self.assertIn("final moment", ending.action)
        self.assertIsNone(ending_openworld_script("ongoing", visual_state))


class TestOpenworldAgents(unittest.TestCase):
    def test_cli_agent_auto_picks_single_auto_action(self):
        printed = []

        def _no_input(_prompt=""):
            raise AssertionError("AUTO 回合不应询问玩家")

        runtime = make_runtime(FakeLLM([NO_OPP_JSON]))
        state = runtime.initial_state()
        actions = runtime.available_actions(state)
        agent = OpenworldCliAgent(input_fn=_no_input, print_fn=printed.append)
        action = agent.choose(state, actions)
        self.assertEqual(action.id, "auto")
        self.assertTrue(any("自动" in line for line in printed))

    def test_cli_agent_still_offers_choices(self):
        def input_fn(_prompt=""):
            return "2"

        runtime = make_runtime(FakeLLM([PROPOSER_JSON]))
        state = runtime.initial_state()
        actions = runtime.available_actions(state)
        agent = OpenworldCliAgent(input_fn=input_fn, print_fn=lambda line: None)
        action = agent.choose(state, actions)
        self.assertEqual(action.label, "检查驿站的布告栏")


class TestOpenworldHudText(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(clip_hud_text("You enter the saloon."), "You enter the saloon.")

    def test_long_english_cut_at_word_boundary(self):
        text = 'You lean in, your voice low and steady. "Heard the name Blackthorn. You seen him'
        clipped = clip_hud_text(text)
        self.assertEqual(clipped, 'You lean in, your voice low and steady. "Heard the name...')
        self.assertLessEqual(len(clipped), 64)

    def test_long_cjk_counts_double_width(self):
        clipped = clip_hud_text("旅途自动向前推进" * 8)
        self.assertTrue(clipped.endswith("..."))
        self.assertEqual(len(clipped), 30 + 3)


class TestOpenworldLoop(unittest.TestCase):
    def test_full_loop_with_fake_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            initial = tmpdir / "scene.png"
            initial.write_bytes(TINY_PNG)
            # 结局最早出现在第 MIN_INTERACTIVE_BEFORE_CLIMAX 个交互幕
            llm = FakeLLM([PROPOSER_JSON, DIRECTOR_JSON] * 11
                          + [PROPOSER_JSON, FINALE_JSON])
            runtime = make_runtime(llm, scene_image=str(initial))
            visual_state = initial_openworld_visual_state()
            visual_state.current_frame = initial

            loop = GameLoop(
                runtime=runtime,
                agent=OpenworldAutoAgent(),
                visual_director=build_openworld_visual_script,
                renderer=FakeRenderer(output_dir=tmpdir),
                visual_state=visual_state,
                output_dir=tmpdir / "session",
                base_seed=7,
                max_turns=14,
                ending_director=ending_openworld_script,
                post_render=lambda record: runtime.on_video_rendered(
                    record.index, record.video_path),
            )
            summary = loop.run()
            self.assertEqual(summary.final_status, "won")
            self.assertIsNotNone(summary.ending)
            self.assertTrue(Path(summary.ending.video_path).exists())
            self.assertTrue(summary.records[0].caption)

    def test_loop_never_plays_consecutive_auto_turns(self):
        # 整局循环验证 §3 硬性规则:AUTO 永不连续;AUTO 后一定回到交互
        script = [NO_OPP_JSON, AUTO_DIRECTOR_JSON,   # T1:AUTO
                  NO_OPP_JSON, DIRECTOR_JSON,        # T2:强制交互(VLM 无选项 → 兜底)
                  PROPOSER_JSON, DIRECTOR_JSON,      # T3:交互
                  NO_OPP_JSON, AUTO_DIRECTOR_JSON,   # T4:AUTO
                  PROPOSER_JSON, DIRECTOR_JSON]      # T5:交互
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            initial = tmpdir / "scene.png"
            initial.write_bytes(TINY_PNG)
            runtime = make_runtime(FakeLLM(script), scene_image=str(initial))
            visual_state = initial_openworld_visual_state()
            visual_state.current_frame = initial
            loop = GameLoop(
                runtime=runtime,
                agent=OpenworldAutoAgent(),
                visual_director=build_openworld_visual_script,
                renderer=FakeRenderer(output_dir=tmpdir),
                visual_state=visual_state,
                output_dir=tmpdir / "session",
                base_seed=7,
                max_turns=5,
            )
            summary = loop.run()
            self.assertEqual(summary.turns_played, 5)
            ids = [record.action_id for record in summary.records]
            self.assertEqual(ids, ["auto", "act-0", "act-0", "auto", "act-0"])
            for prev, cur in zip(ids, ids[1:]):
                self.assertFalse(prev == "auto" and cur == "auto",
                                 f"consecutive AUTO turns: {ids}")


if __name__ == "__main__":
    unittest.main()
