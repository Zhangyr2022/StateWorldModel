"""GameLoop 测试:多回合串联、首帧链、渲染失败不影响游戏状态(§16/§25)。"""

import base64
import json
import tempfile
import unittest
from pathlib import Path

from engine.agents import ScriptedAgent
from engine.loop import RENDER_FAILURE_ABORT, RENDER_FAILURE_SKIP, GameLoop
from engine.renderer.base import (
    InvalidRenderRequestError,
    RenderFailedError,
    RenderRequest,
    RenderResult,
    Renderer,
)
from engine.renderer.fake import FakeRenderer
from game.registry import GAMES

# 载具:duel_oracle_machine_epic(状态机战斗皮肤)——引擎级测试与具体游戏解耦
_DUEL_PARTS = GAMES["duel_oracle_machine_epic"].build()


def _fresh_visual_state():
    return GAMES["duel_oracle_machine_epic"].build().visual_state


SCRIPT = ("fireball", "frostbolt", "ice_shield", "arcane", "fireball", "fireball")


def _t2v_stub_script():
    """use_first_frame=False 的最小 VisualScript(模拟恋爱场景切换回合)。"""
    from engine.visual.script import VisualScript

    return VisualScript(
        style="test",
        actor="someone",
        action="arrives at a new place and settles in",
        end_state="standing still in the new place",
        use_first_frame=False,
    )

# 1x1 的合法 PNG(测试里不能用坏图:ffmpeg 会耗满超时才放弃)
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class RecordingRenderer(Renderer):
    """包装 FakeRenderer,记录每次请求以验证首帧链。"""

    def __init__(self, inner: Renderer) -> None:
        self.inner = inner
        self.requests: list[RenderRequest] = []

    def render(self, request: RenderRequest) -> RenderResult:
        self.requests.append(request)
        return self.inner.render(request)


class FailingRenderer(Renderer):
    """模拟 Minimax-H3 完全不可用。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def render(self, request: RenderRequest) -> RenderResult:
        raise RenderFailedError("simulated h3 outage")


class CounterAgent:
    """会反制的测试代理:Boss 蓄力时举冰盾、盾态用冰枪击碎、低血治疗,否则火球。"""

    def __init__(self) -> None:
        pass

    def choose(self, state, actions):
        ids = {a.id for a in actions}
        if state.boss_state == "windup" and "ice_shield" in ids:
            return next(a for a in actions if a.id == "ice_shield")
        if state.vars.get("boss_shield") and "frostbolt" in ids:
            return next(a for a in actions if a.id == "frostbolt")
        if state.vars.get("player_hp", 90) <= 40 and "heal" in ids:
            return next(a for a in actions if a.id == "heal")
        for slot in ("fireball", "arcane"):
            if slot in ids:
                return next(a for a in actions if a.id == slot)
        return actions[0]


class GameLoopTestBase(unittest.TestCase):
    def make_loop(self, tmpdir: Path, renderer: Renderer, **kwargs) -> GameLoop:
        initial_frame = tmpdir / "initial.png"
        initial_frame.write_bytes(TINY_PNG)
        visual_state = _fresh_visual_state()
        visual_state.current_frame = initial_frame
        defaults = dict(
            runtime=_DUEL_PARTS.runtime,
            agent=CounterAgent(),
            visual_director=_DUEL_PARTS.visual_director,
            renderer=renderer,
            visual_state=visual_state,
            output_dir=tmpdir / "session",
            base_seed=7,
            max_turns=12,
        )
        defaults.update(kwargs)
        return GameLoop(**defaults)


class TestHappyPath(GameLoopTestBase):
    def test_full_loop_produces_chain_of_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            recording = RecordingRenderer(FakeRenderer(output_dir=tmpdir))
            loop = self.make_loop(tmpdir, recording)
            summary = loop.run()

            self.assertTrue(summary.turns_played > 0)
            # 每回合产出视频与最后一帧
            for record in summary.records:
                self.assertIsNone(record.render_error)
                self.assertTrue(Path(record.video_path).exists())
                self.assertTrue(Path(record.final_frame).exists())
                self.assertTrue(record.prompt)

            # 首帧链:turn N 的请求首帧 == turn N-1 的最后一帧
            self.assertEqual(recording.requests[0].first_frame, tmpdir / "initial.png")
            for prev, curr in zip(recording.requests, recording.requests[1:]):
                prev_frame = tmpdir / "frames" / f"{prev.turn_id}.png"
                self.assertEqual(curr.first_frame, prev_frame)

            # 每个 seed 递增
            for index, request in enumerate(recording.requests):
                self.assertEqual(request.seed, 7 + index)

            # 回合日志:一行一回合,字段齐全
            log_lines = (tmpdir / "session" / "turn_log.jsonl").read_text().splitlines()
            self.assertEqual(len(log_lines), summary.turns_played)
            entry = json.loads(log_lines[0])
            for key in ("turn_id", "state_before", "state_after", "events", "visual_script", "prompt", "render"):
                self.assertIn(key, entry)

    def test_game_can_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            loop = self.make_loop(tmpdir, FakeRenderer(output_dir=tmpdir),
                                  agent=ScriptedAgent(["fireball", "frostbolt", "heal"], fallback=True),
                                  max_turns=40)
            summary = loop.run()
            self.assertEqual(summary.finished_reason, "game_over")
            self.assertEqual(summary.final_status, "won")

    def test_turn_records_carry_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            loop = self.make_loop(tmpdir, FakeRenderer(output_dir=tmpdir))
            summary = loop.run()
            for record in summary.records:
                self.assertIn("hp", record.caption.lower())  # describe_state 的状态摘要
                self.assertNotIn("回合", record.caption.split("|")[0])  # 回合号只出现一次


class TestRenderFailure(GameLoopTestBase):
    def test_skip_policy_game_state_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            # 同 seed 下,渲染全失败的游戏进程应与正常渲染完全一致(§16)
            (tmpdir / "f").mkdir()
            failing_loop = self.make_loop(tmpdir / "f", FailingRenderer(tmpdir),
                                          render_failure_policy=RENDER_FAILURE_SKIP)
            failing_summary = failing_loop.run()

            (tmpdir / "n").mkdir()
            normal_loop = self.make_loop(tmpdir / "n", FakeRenderer(output_dir=tmpdir))
            normal_summary = normal_loop.run()

            self.assertGreater(failing_summary.turns_played, 0)
            self.assertEqual(
                [(record.turn_id, record.action_id, record.status) for record in failing_summary.records],
                [(record.turn_id, record.action_id, record.status) for record in normal_summary.records],
            )
            self.assertTrue(all(record.render_error for record in failing_summary.records))
            self.assertFalse(failing_summary.ok)

    def test_abort_policy_raises_after_state_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            loop = self.make_loop(tmpdir, FailingRenderer(tmpdir),
                                  render_failure_policy=RENDER_FAILURE_ABORT)
            with self.assertRaises(RenderFailedError):
                loop.run()
            # 结算已完成的回合日志仍在:状态先于渲染确定
            log_lines = (tmpdir / "session" / "turn_log.jsonl").read_text().splitlines()
            self.assertEqual(len(log_lines), 1)

    def test_invalid_request_is_not_retried(self):
        """请求本身非法(重试也不会成功)时立即抛错,不消耗重试次数。"""

        class InvalidRequestRenderer(Renderer):
            attempts = 0

            def render(self, request: RenderRequest) -> RenderResult:
                self.attempts += 1
                raise InvalidRenderRequestError("always invalid")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            renderer = InvalidRequestRenderer()
            loop = self.make_loop(tmpdir, renderer, render_retries=3)
            with self.assertRaises(InvalidRenderRequestError):
                loop.run()
            self.assertEqual(renderer.attempts, 1)

    def test_t2v_request_gets_synthesized_frame(self):
        """t2v(无首帧,如场景切换回合)在 fake 渲染器下也能闭环。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            recording = RecordingRenderer(FakeRenderer(output_dir=tmpdir))
            loop = GameLoop(
                runtime=_DUEL_PARTS.runtime,  # 运行时只用事件通路,脚本由桩生成
                agent=ScriptedAgent(["arcane"]),
                visual_director=lambda result, state: _t2v_stub_script(),
                renderer=recording,
                visual_state=_fresh_visual_state(),  # current_frame=None
                output_dir=tmpdir / "session",
                max_turns=1,
            )
            summary = loop.run()
            self.assertEqual(summary.turns_played, 1)
            self.assertIsNone(summary.records[0].render_error)
            self.assertTrue(Path(summary.records[0].video_path).exists())
            self.assertIsNone(recording.requests[0].first_frame)  # 请求确实是 t2v


class TestEndingAnimation(GameLoopTestBase):
    def test_ending_rendered_on_game_over(self):
        ending_director = _DUEL_PARTS.ending_director

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            loop = self.make_loop(tmpdir, FakeRenderer(output_dir=tmpdir),
                                  agent=CounterAgent(),
                                  max_turns=40,
                                  ending_director=ending_director)
            summary = loop.run()
            self.assertEqual(summary.finished_reason, "game_over")
            self.assertIn(summary.final_status, ("won", "lost"))
            self.assertIsNotNone(summary.ending)
            self.assertEqual(summary.ending.turn_id, f"ending-{summary.final_status}")
            self.assertTrue(Path(summary.ending.video_path).exists())
            self.assertTrue(Path(summary.ending.final_frame).exists())
            # 结束动画不占用回合记录
            self.assertEqual(len(summary.records), summary.turns_played)
            # 日志里也有 ending 行
            log_text = (tmpdir / "session" / "turn_log.jsonl").read_text()
            self.assertIn(f"ending-{summary.final_status}", log_text)

    def test_ending_render_failure_keeps_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "f").mkdir()

            class EndingOnlyFails(Renderer):
                def render(self, request):
                    if request.turn_id.startswith("ending"):
                        raise RenderFailedError("ending outage")
                    return FakeRenderer(output_dir=tmpdir).render(request)

            ending_director = _DUEL_PARTS.ending_director
            loop = self.make_loop(tmpdir / "f", EndingOnlyFails(),
                                  agent=CounterAgent(),
                                  max_turns=40,
                                  ending_director=ending_director)
            summary = loop.run()
            # 游戏结果不受渲染失败影响(§16);终局可能是任一结果
            self.assertIn(summary.final_status, ("won", "lost"))
            self.assertIsNotNone(summary.ending)
            self.assertIsNotNone(summary.ending.render_error)

    def test_no_ending_director_no_ending(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            loop = self.make_loop(tmpdir, FakeRenderer(output_dir=tmpdir),
                                  agent=ScriptedAgent(["fireball", "frostbolt", "heal"], fallback=True),
                                  max_turns=25)
            summary = loop.run()
            self.assertIsNone(summary.ending)


if __name__ == "__main__":
    unittest.main()
