"""Montage 测试:回合视频拼接 + 字幕带 + 顺序组装。"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from engine.montage import (
    ClipSpec,
    MontageError,
    _font,
    _wrap_by_width,
    build_game_montage,
)

FFMPEG = shutil.which("ffmpeg") is not None
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def make_color_clip(path: Path, color: str, seconds: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c={color}:s=320x180:d={seconds}:r=24",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@unittest.skipUnless(FFMPEG, "montage needs ffmpeg")
@unittest.skipUnless(FONT.exists(), "montage captions need a CJK font")
class TestMontage(unittest.TestCase):
    def test_concat_with_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            clips = [
                ClipSpec(make_color_clip(tmpdir / "a.mp4", "navy", 0.5), caption="回合1 · 火球术 | 莉娜 HP 90/90 | 巨人 HP 144/160"),
                ClipSpec(make_color_clip(tmpdir / "b.mp4", "darkred", 0.5), caption="回合2 · 冰霜护盾 | 完全格挡了熔岩喷火!"),
                ClipSpec(make_color_clip(tmpdir / "c.mp4", "goldenrod", 0.5), caption=""),  # 无字幕
            ]
            out = build_game_montage(clips, tmpdir / "montage.mp4")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)
            # 时长 ≈ 各段之和(0.5×3 = 1.5s,给编码容差)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
                check=True, capture_output=True, text=True,
            )
            duration = float(probe.stdout.strip())
            self.assertAlmostEqual(duration, 1.5, delta=0.4)

    def test_missing_clip_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MontageError):
                build_game_montage([ClipSpec(video=Path(tmp) / "missing.mp4", caption="x")],
                                   Path(tmp) / "out.mp4")

    def test_missing_font_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = ClipSpec(make_color_clip(Path(tmp) / "a.mp4", "navy", 0.3), caption="x")
            with self.assertRaises(MontageError):
                build_game_montage([clip], Path(tmp) / "out.mp4", font_path="/no/such/font.ttf")


@unittest.skipUnless(FONT.exists(), "montage captions need a CJK font")
class TestCaptionWrap(unittest.TestCase):
    """字幕折行:按像素宽度折行(_wrap_by_width),不再按字符数。"""

    @classmethod
    def setUpClass(cls):
        cls.draw = ImageDraw.Draw(Image.new("RGB", (640, 200)))
        cls.font = _font(str(FONT), 22)

    def test_wraps_long_caption_within_limit(self):
        caption = "回合3 · 火球术 | 魔法师莉娜 HP 90/90 MP 8/15 | 石像巨人(狂暴) HP 96/160 | 巨人正在积蓄熔岩"
        lines = _wrap_by_width(self.draw, caption, self.font, 220, max_lines=6)
        self.assertLessEqual(len(lines), 6)  # 左侧栏默认最多 6 行
        self.assertTrue(all(line for line in lines))
        # 除末行(可能带省略号)外,每行都应放得下
        for line in lines[:-1]:
            self.assertLessEqual(self.draw.textlength(line, font=self.font), 220)

    def test_two_lines_are_balanced(self):
        # 两行时做均衡:两行宽度尽量接近,而不是第一行塞满
        lines = _wrap_by_width(
            self.draw, "fireball frost shield lava burst", self.font, 150, max_lines=2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(lines))

    def test_truncates_overlong_caption(self):
        lines = _wrap_by_width(self.draw, "长" * 200, self.font, 120, max_lines=6)
        self.assertLessEqual(len(lines), 6)
        self.assertTrue("".join(lines).endswith("…"))


if __name__ == "__main__":
    unittest.main()
