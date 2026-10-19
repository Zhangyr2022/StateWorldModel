"""回合画面 replay-HUD 化测试:machine hud_of 快照 + overlay 片段合成。

全部离线:FakeRenderer + ffmpeg 合成小视频,不依赖 GPU / LLM / gradio。
"""

import contextlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from engine.parts import resolve_frame
from engine.renderer.fake import FakeRenderer
from engine.ui.hud_clip import compose_hud_clip
from game.registry import get

FFMPEG = shutil.which("ffmpeg") is not None


def _new_machine_session(output_dir: Path, max_turns: int = 2):
    """组装一个离线 machine 交互会话(与 run_ui.on_start 同一条装配路径)。"""
    from engine.ui.session import InteractiveGame

    parts = get("duel_oracle_machine_epic").build(None)
    frame = resolve_frame(parts.frame)
    parts.visual_state.current_frame = frame
    return InteractiveGame(parts, FakeRenderer(output_dir=output_dir),
                           output_dir=output_dir, base_seed=7, max_turns=max_turns)


class MachineHudSnapshotTest(unittest.TestCase):
    """machine 主题的 hud_of:overlay 面板所需的键一个不能少。"""

    def test_hud_of_after_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = _new_machine_session(Path(tmp))
            record = session.step(session.available_actions()[0])
        hud = (record.meta or {}).get("hud")
        self.assertIsInstance(hud, dict)
        for key in ("left_name", "left_stats", "right_name", "right_stats",
                    "enemy_action", "tag", "action"):
            self.assertIn(key, hud, key)
        self.assertTrue(hud["left_name"])
        self.assertTrue(hud["right_name"])
        hp_pct, mp_pct, hp_text, mp_text = hud["left_stats"]
        self.assertTrue(0.0 <= hp_pct <= 1.0)
        self.assertIn("/", hp_text)
        # 打出去一招:Boss 行动行不再按兵不动(状态机的 Boss 必然响应)
        self.assertTrue(hud["enemy_action"])


class ComposeHudClipTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)
        self.source = self.out_dir / "videos" / "turn-001.mp4"
        self.source.parent.mkdir(parents=True, exist_ok=True)
        # 2 秒纯色小视频(无音频,与 FakeRenderer 产物同构的最小用例)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=0x0b1226:s=864x480:r=24:d=2",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             str(self.source)],
            check=True)

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            self._tmp.cleanup()

    @unittest.skipUnless(FFMPEG, "需要 ffmpeg")
    def test_compose_renders_hud_clip(self):
        hud = {"tag": "第1回合 · 测试", "left_name": "测试剑士",
               "left_stats": (0.8, 0.5, "80 / 100", "MP 5 / 15"),
               "right_name": "测试魔王", "right_stats": (0.9, "90 / 100"),
               "enemy_action": "Boss 招式:测试冲击",
               "action": "测试技能", "action_desc": "无消耗的稳定输出"}
        out = compose_hud_clip(self.out_dir, str(self.source), hud=hud)
        self.assertIsNotNone(out)
        produced = Path(out)
        self.assertEqual(produced.name, "turn-001-hud.mp4")
        self.assertEqual(produced.parent.name, "clips")
        self.assertGreater(produced.stat().st_size, 10_000)  # 有真实画面内容

    @unittest.skipUnless(FFMPEG, "需要 ffmpeg")
    def test_cache_reuse(self):
        first = compose_hud_clip(self.out_dir, str(self.source), hud={"tag": "x"})
        stamp = Path(first).stat().st_mtime_ns
        again = compose_hud_clip(self.out_dir, str(self.source), hud={"tag": "x"})
        self.assertEqual(Path(first).stat().st_mtime_ns, stamp)  # 命中缓存不重编

    @unittest.skipUnless(FFMPEG, "需要 ffmpeg")
    def test_lang_cached_separately(self):
        """界面切语言:同一回合的 zh/en 片段各自缓存,互不覆盖。"""
        hud = {"tag": "第1回合 · 测试", "enemy_action": "敌方行动"}
        zh = compose_hud_clip(self.out_dir, str(self.source), hud=hud, lang="zh")
        en = compose_hud_clip(self.out_dir, str(self.source), hud=hud, lang="en")
        self.assertEqual(Path(zh).name, "turn-001-hud-zh.mp4")
        self.assertEqual(Path(en).name, "turn-001-hud-en.mp4")

    def test_missing_video_returns_none(self):
        self.assertIsNone(compose_hud_clip(self.out_dir, "", hud={}))
        self.assertIsNone(compose_hud_clip(self.out_dir, str(
            self.out_dir / "videos" / "nope.mp4"), hud={"tag": "x"}))


if __name__ == "__main__":
    unittest.main()
