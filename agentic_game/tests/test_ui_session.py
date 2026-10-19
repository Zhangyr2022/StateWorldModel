"""交互式会话测试:Gradio 界面的引擎侧(逐步结算 / 非法动作 / 终局 / 回放)。

全部离线:FakeRenderer + 仓库自带首帧,不依赖 GPU / LLM / gradio。
"""

import contextlib
import shutil
import tempfile
import unittest
from pathlib import Path

from engine.montage import build_summary_montage
from engine.parts import resolve_frame
from engine.renderer.fake import FakeRenderer
from engine.runtime import STATUS_ONGOING, Action
from engine.ui.session import GameFinishedError, InteractiveGame
from game.registry import get

FFMPEG = shutil.which("ffmpeg") is not None


def _new_session(output_dir: Path, *, game_id: str = "duel_oracle_machine_epic", max_turns: int = 30,
                 seed: int = 7) -> InteractiveGame:
    """组装一个离线交互式会话(与 run_ui.on_start 同一条装配路径)。"""
    parts = get(game_id).build(None)
    frame = resolve_frame(parts.frame)
    parts.visual_state.current_frame = frame  # 仓库自带首帧,FakeRenderer 要求非空
    renderer = FakeRenderer(output_dir=output_dir)
    return InteractiveGame(parts, renderer, output_dir=output_dir,
                           base_seed=seed, max_turns=max_turns)


class InteractiveSessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmp.name)

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            self._tmp.cleanup()

    def test_step_advances_state_and_records_turns(self):
        session = _new_session(self.output_dir, max_turns=2)
        self.assertEqual(session.status, STATUS_ONGOING)
        self.assertFalse(session.finished)

        first = session.step(session.available_actions()[0])
        self.assertEqual(first.index, 0)
        self.assertTrue(Path(first.video_path).exists())
        self.assertEqual(session.turns_played, 1)
        self.assertIsNot(session.state, first)  # state_before/after 分离

        second = session.step(session.available_actions()[0])
        self.assertEqual(second.index, 1)
        # max_turns=2:两回合后自动结束,无结束动画
        self.assertTrue(session.finished)
        self.assertEqual(session.summary.finished_reason, "max_turns")
        self.assertIsNone(session.summary.ending)
        with self.assertRaises(GameFinishedError):
            session.step(session.available_actions() or [Action(id="x", label="x")])

    def test_illegal_action_keeps_state_and_allows_retry(self):
        session = _new_session(self.output_dir)
        before = session.state
        with self.assertRaises(ValueError):
            session.step(Action(id="not-a-skill", label="不存在的行为"))
        self.assertIs(session.state, before)  # 状态未被污染
        self.assertEqual(session.turns_played, 0)
        # 之后仍可正常出招
        record = session.step(session.available_actions()[0])
        self.assertEqual(session.turns_played, 1)

    def test_game_over_renders_ending_animation(self):
        session = _new_session(self.output_dir, seed=7)
        # 直接把玩家打到濒死:下一回合敌人任何攻击都会终结比赛
        session.state.vars["player_hp"] = 1  # 直接濒死:Boss 的攻击行为会终结比赛
        record = None
        for _ in range(30):  # Boss 当回合可能按兵不动,循环出招直到分出胜负
            record = session.step(session.available_actions()[0])
            if session.finished:
                break
        self.assertEqual(session.summary.finished_reason, "game_over")
        # 濒死循环出招:Boss 先手可能先杀死玩家(lost),玩家火球也可能先收下胜利(won)
        self.assertIn(session.summary.final_status, ("won", "lost"))
        self.assertGreaterEqual(session.summary.turns_played, 1)
        self.assertIsNotNone(session.summary.ending)
        self.assertEqual(session.summary.ending.status, "lost")
        self.assertTrue(Path(session.summary.ending.video_path).exists())

    def test_history_lines_and_describe(self):
        session = _new_session(self.output_dir, max_turns=1)
        self.assertTrue(session.describe())
        session.step(session.available_actions()[0])
        lines = session.history_lines()
        self.assertEqual(len(lines), 1)
        self.assertIn("R1", lines[0])

    @unittest.skipUnless(FFMPEG, "montage needs ffmpeg")
    def test_montage_from_summary(self):
        session = _new_session(self.output_dir, max_turns=2)
        while not session.finished:
            session.step(session.available_actions()[0])
        path = build_summary_montage(
            session.summary, self.output_dir / "game_montage.mp4",
            scene_labels=session.parts.scene_labels,
            ending_captions=session.parts.ending_captions,
        )
        self.assertIsNotNone(path)
        self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
