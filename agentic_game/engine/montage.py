"""Montage:把一局游戏的回合视频拼成一部完整录像,画面上叠加 RPG 风 HUD。

视觉层(纯表现,不影响结算):每个回合片段上叠一层半透明 HUD 贴片——
左上玩家状态(HP/MP 条)、右上 Boss 状态、左下技能信息面板、左上角小型
战斗状态标签;贴片整体 200ms alpha 淡入。HUD 贴片由 PIL 离线渲染成
RGBA PNG,再经 ffmpeg overlay 合成;无 HUD 的片段(场景过渡/终局)只画
居中标签。产物失败只影响回放,不影响对局。

hud 快照键约定(由游戏层 hud_of / 回放重建脚本提供):
    tag         小型状态标签文本(如"第3回合 · 神谕·第一句")
    left_name   玩家显示名         right_name  Boss 显示名
    left_stats  (hp_pct, mp_pct, "hp 文本", "mp 文本")
    right_stats (hp_pct, "hp 文本")

中文需要 CJK 字体,默认 Noto Sans CJK;缺失时调用方可显式指定。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from engine.hud_i18n import tr

DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# 场景切换黑屏过渡时长(秒)
SCENE_TRANSITION_S = 0.8

# 输出统一尺寸(偶数,H.264 要求);HUD 按该分辨率绘制,条宽用百分比换算
OUT_W, OUT_H = 864, 480
# HUD 状态带:不叠在画面上,额外拼在视频下方(用户可见区域保持纯净)
HUD_BAND_H = 190

# HUD 淡入时长(秒)
HUD_FADE_S = 0.2

# 色板:暗色玻璃 + 金色描边 + 奥术青
GOLD = (222, 186, 96, 220)
GOLD_DIM = (196, 164, 92, 150)
PANEL_BG = (10, 13, 22, 158)
PANEL_EDGE = (208, 176, 100, 190)
TXT_MAIN = (238, 240, 246, 235)
TXT_SUB = (198, 204, 216, 220)
CYAN = (130, 226, 206, 255)
BOSS_GOLD = (233, 200, 116, 255)
HP_FILL = (178, 34, 52, 255)
HP_HI = (226, 84, 72, 255)
MP_FILL = (44, 84, 206, 255)
MP_HI = (98, 142, 236, 255)
TRACK = (22, 26, 38, 215)
# 类别主题色:duel 金 / romance 玫瑰 / openworld 苔绿
ROSE = (242, 167, 179, 255)
ROSE_FILL = (214, 74, 118, 255)
ROSE_HI = (248, 138, 168, 255)
ROSE_EDGE = (226, 150, 160, 190)
LEAF = (168, 214, 138, 255)
LEAF_EDGE = (150, 196, 122, 190)
QUEST_GOLD = (238, 208, 138, 255)

_FFMPEG_TIMEOUT_S = 300


class MontageError(Exception):
    """Montage 构建失败(ffmpeg 错误、字体缺失等)。"""


@dataclass
class ClipSpec:
    """一个待拼接片段。

    video 为 None 时表示黑屏过渡(时长 black_s 秒,可带字幕)。
    hud 为回放 HUD 快照,提供时片段上叠加 RPG 风状态面板,
    caption 退化为左上角小标签文本。
    """

    video: Path | None
    caption: str = ""
    black_s: float = 0.0
    hud: dict | None = None

    @staticmethod
    def black(duration_s: float, caption: str = "") -> "ClipSpec":
        return ClipSpec(video=None, caption=caption, black_s=duration_s)


def build_game_montage(
    clips: list[ClipSpec],
    output_path: Path,
    *,
    work_dir: Path | None = None,
    font_path: str = DEFAULT_FONT,
    hud_style: str = "band",
) -> Path:
    """把 clips 依序拼成一部带 HUD 叠层的录像,写到 output_path。

    hud_style:"band"(默认)= HUD 渲染进视频下方的专用状态带(864x670,
    画面零遮挡);"overlay" = 半透明面板叠在画面上(864x480,gallery 成片的
    replay_hud 风格)。
    """
    if hud_style not in ("band", "overlay"):
        raise MontageError(f"unknown hud_style: {hud_style!r} (band/overlay)")
    if not clips:
        raise MontageError("no clips to montage")
    if not Path(font_path).exists():
        raise MontageError(
            f"font not found: {font_path} (字幕需要字体文件;可经 font_path 指定 CJK 字体)"
        )
    for spec in clips:
        if spec.video is not None and not spec.video.exists():
            raise MontageError(f"clip video missing: {spec.video}")
        if spec.video is None and spec.black_s <= 0:
            raise MontageError("black clip needs positive black_s")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir) if work_dir is not None else output_path.parent / "montage_parts"
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized: list[Path] = []
    list_file = work_dir / "concat.txt"
    try:
        for index, spec in enumerate(clips):
            part = work_dir / f"part-{index:03d}.mp4"
            _compose_clip(spec, part, font_path, hud_style)
            normalized.append(part)

        list_file.write_text(
            "".join(f"file '{part.resolve()}'\n" for part in normalized), encoding="utf-8"
        )
        _run_ffmpeg(
            [
                "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output_path),
            ]
        )
    finally:
        for part in normalized:
            part.unlink(missing_ok=True)
        list_file.unlink(missing_ok=True)
        if work_dir.exists() and not any(work_dir.iterdir()):
            work_dir.rmdir()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise MontageError(f"montage produced no output at {output_path}")
    return output_path


def build_summary_montage(
    summary,
    output_path: Path,
    *,
    scene_labels: dict[str, str] | None = None,
    ending_captions: dict[str, str] | None = None,
) -> Path | None:
    """把一局(GameLoop/交互式会话产出)的回合视频拼成完整回放。

    - 渲染失败的回合没有视频,自动跳过;
    - 回合 meta 里带 scene 且与上一回合不同时,插入黑屏过渡并标注新场景;
    - 结束动画放最后,标签取 ending_captions[终局状态],缺省用状态名。

    没有任何可用片段时返回 None(不视为错误)。
    """
    scene_labels = scene_labels or {}
    ending_captions = ending_captions or {}
    clips: list[ClipSpec] = []
    previous_scene = None
    for record in summary.records:
        if record.video_path is None:
            continue  # 渲染失败的回合没有视频可拼
        scene_id = record.meta.get("scene")
        if scene_id is not None and previous_scene is not None and scene_id != previous_scene:
            clips.append(ClipSpec.black(
                SCENE_TRANSITION_S, caption=f"—— {scene_labels.get(scene_id, scene_id)} ——"))
        clips.append(ClipSpec(video=Path(record.video_path),
                              caption=record.caption, hud=record.meta.get("hud")))
        if scene_id is not None:
            previous_scene = scene_id
    ending = summary.ending
    if ending is not None and ending.video_path is not None:
        clips.append(ClipSpec(
            video=Path(ending.video_path),
            caption=ending_captions.get(ending.status, ending.status),
        ))
    if not clips:
        return None
    return build_game_montage(clips, Path(output_path))


# ---------------------------------------------------------------------------
# 单片段合成:归一化转码 + HUD 贴层(带淡入)
# ---------------------------------------------------------------------------

def _compose_clip(spec: ClipSpec, out_path: Path, font_path: str,
                  hud_style: str = "band") -> None:
    """归一化尺寸/编码;HUD 按风格渲染——band=视频下方专用状态带(默认,
    画面零遮挡),overlay=半透明面板叠在画面上(gallery 时代 replay_hud)。"""
    overlay = hud_style == "overlay"
    overlay_png: Path | None = None
    full_h = OUT_H if overlay else OUT_H + HUD_BAND_H
    pad_y = "(oh-ih)/2" if overlay else "0"
    scale = ("scale=%d:%d:force_original_aspect_ratio=decrease,"
             "pad=%d:%d:(ow-iw)/2:%s:color=0x0b0e16" % (OUT_W, OUT_H, OUT_W, full_h, pad_y))
    command = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    if spec.video is None:
        command += ["-f", "lavfi", "-i",
                    f"color=c=0x0b0e16:s={OUT_W}x{full_h}:d={spec.black_s}"]
        audio_map = "2:a"  # 黑屏片段的静音轨在 2 号输入(见下)
    else:
        command += ["-i", str(spec.video)]
        audio_map = "0:a?"
    if spec.video is None and not spec.hud and not spec.caption.strip():
        # 无字幕黑屏也要有静音轨(1 号输入),否则 -map audio_map 指向不存在的
        # 输入(2026-10-13 章节转场卡首次走到该路径时暴露)
        command += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
        audio_map = "1:a"

    if spec.hud or spec.caption.strip():
        overlay_png = out_path.with_suffix(".hud.png")
        renderer = _render_hud_overlay_png if overlay else _render_hud_png
        renderer(overlay_png, font_path, caption=spec.caption, hud=spec.hud)
        command += ["-loop", "1", "-i", str(overlay_png)]  # PNG 恒为 1 号输入
        if spec.video is None:
            command += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
        overlay_y = 0 if overlay else OUT_H
        command += [
            "-filter_complex",
            f"[0:v]{scale}[base];"
            f"[1:v]fade=t=in:st=0:d={HUD_FADE_S:.2f}:alpha=1[hud];"
            f"[base][hud]overlay=0:{overlay_y}:shortest=1[v]",
            "-map", "[v]", "-map", audio_map,
        ]
    else:
        command += ["-vf", scale, "-map", "0:v", "-map", audio_map]

    command += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
    ]
    if spec.video is None:
        command += ["-shortest"]
    command += [str(out_path)]
    try:
        _run_ffmpeg(command)
    finally:
        if overlay_png is not None:
            overlay_png.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# HUD 贴片渲染(PIL)
# ---------------------------------------------------------------------------

def _font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


def _fit_text(d: ImageDraw.ImageDraw, text: str, font_path: str, size: int,
              max_w: float, min_size: int = 12) -> tuple[ImageFont.FreeTypeFont, str]:
    """面板名过长时逐级缩小字号;仍放不下再截断加省略号(英文皮肤名较长)。"""
    font = _font(font_path, size)
    while size > min_size and d.textlength(text, font=font) > max_w:
        size -= 1
        font = _font(font_path, size)
    if d.textlength(text, font=font) > max_w:
        while text and d.textlength(text + "…", font=font) > max_w:
            text = text[:-1]
        text += "…"
    return font, text


def _bar(d: ImageDraw.ImageDraw, x, y, w, h, pct, fill, hi, font, value_text,
         right_edge: float | None = None):
    """一条 RPG 进度条:暗色槽 + 金边 + 渐变填充(上半亮色)+ 右侧数值。

    right_edge:面板右缘;英文数值文本(如 "Sword Qi 13 / 15")比中文宽,
    给出时压缩条宽,保证 条 + 数值 都落在面板内(2026-10-15 英文 HUD)。
    """
    if right_edge is not None and value_text:
        w = max(24, min(w, right_edge - (x + 10) - d.textlength(value_text, font=font)))
    _rounded = d.rounded_rectangle
    _rounded((x, y, x + w, y + h), h // 2, TRACK, outline=GOLD_DIM, width=1)
    pad = 2
    fill_w = max(h - pad * 2, int((w - pad * 2) * max(0.0, min(1.0, pct))))
    _rounded((x + pad, y + pad, x + pad + fill_w, y + h - pad),
             (h - pad * 2) // 2, fill)
    _rounded((x + pad, y + pad, x + pad + fill_w, y + pad + max(2, (h - pad * 2) // 3)),
             2, hi)
    d.text((x + w + 10, y + h // 2), value_text, font=font, fill=TXT_SUB, anchor="lm")


def _glow_text(layer: Image.Image, xy, text, font, color, blur=3, anchor="la"):
    """轻微发光的文字:模糊底两层 + 锐利面一层。"""
    glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).text(xy, text, font=font, fill=color, anchor=anchor)
    glow = glow.filter(ImageFilter.GaussianBlur(blur))
    layer.alpha_composite(glow)
    layer.alpha_composite(glow)
    ImageDraw.Draw(layer).text(xy, text, font=font, fill=color, anchor=anchor)


def _rune_icon(layer: Image.Image, cx, cy, r):
    """无图片资源的符文占位:金色双圆环 + 青色菱形符文,带微光。"""
    d = ImageDraw.Draw(layer)
    glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=GOLD, width=2)
    d.ellipse((cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4), outline=GOLD_DIM, width=1)
    gd.regular_polygon((cx, cy, r - 7), 4, rotation=45, fill=CYAN)
    gd.line((cx - (r - 10), cy, cx + (r - 10), cy), fill=CYAN, width=2)
    layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(2)))
    ImageDraw.Draw(layer).regular_polygon((cx, cy, r - 7), 4, rotation=45, fill=CYAN)
    ImageDraw.Draw(layer).line((cx - (r - 10), cy, cx + (r - 10), cy), fill=CYAN, width=2)


def _heart(layer: Image.Image, cx, cy, r):
    """玫瑰爱心图标:两圆一三角,带微光。"""
    d = ImageDraw.Draw(layer)
    glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for im in (gd, d):
        im.ellipse((cx - r, cy - r, cx + r * 0.1, cy + r * 0.1), fill=ROSE_FILL)
        im.ellipse((cx - r * 0.1, cy - r, cx + r, cy + r * 0.1), fill=ROSE_FILL)
        im.polygon([(cx - r, cy - r * 0.25), (cx + r, cy - r * 0.25),
                    (cx, cy + r)], fill=ROSE_FILL)
    layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(2)))
    d.ellipse((cx - r * 0.55, cy - r * 0.6, cx - r * 0.15, cy - r * 0.2),
              fill=(255, 214, 224, 200))


def _compass_icon(layer: Image.Image, cx, cy, r):
    """探险罗盘图标:金环 + 苔绿四芒星,带微光。"""
    d = ImageDraw.Draw(layer)
    glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=GOLD, width=2)
    gd.regular_polygon((cx, cy, r - 6), 4, rotation=0, fill=LEAF)
    layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(2)))
    ImageDraw.Draw(layer).regular_polygon((cx, cy, r - 6), 4, rotation=0, fill=LEAF)
    d.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=GOLD)


def _tokens_by_width(draw, text, font, max_w):
    """把文本切成 token:英文按词、超长无空格串(中文/URL)按字符。"""
    tokens: list[str] = []
    for word in text.split(" "):
        if not word:
            continue
        buf = ""
        for ch in word:
            if draw.textlength(buf + ch, font=font) > max_w and buf:
                tokens.append(buf)
                buf = ch
            else:
                buf += ch
        tokens.append(buf)
    return tokens


def _greedy_lines(draw, tokens, font, max_w, max_lines):
    """贪心装填:按顺序尽量填满每一行。

    返回 (lines, overflowed):overflowed 表示装满 max_lines 行后仍有放不下的内容。
    """
    lines: list[str] = []
    current = ""
    overflowed = False
    for tok in tokens:
        trial = (current + " " + tok) if current else tok
        if draw.textlength(trial, font=font) > max_w and current:
            lines.append(current)
            current = tok
            if len(lines) == max_lines:
                overflowed = True  # current 及其后的 token 都放不下了
                break
        else:
            current = trial
    else:
        if current:
            lines.append(current)
    return lines, overflowed


def _wrap_by_width(draw, text, font, max_w, max_lines=2):
    """按像素宽度折行(英文按词、中文按字),最多 max_lines 行。

    两行时做均衡(两行宽度尽量接近,避免第二行只剩一两个词);
    仍放不下的溢出部分在末行加省略号。
    """
    tokens = _tokens_by_width(draw, text, font, max_w)
    if not tokens:
        return [""]
    if max_lines == 2 and len(tokens) > 1:
        widths = [draw.textlength(tok, font=font) for tok in tokens]
        best = None
        for cut in range(1, len(tokens)):
            w1 = draw.textlength(" ".join(tokens[:cut]), font=font)
            w2 = draw.textlength(" ".join(tokens[cut:]), font=font)
            if w1 <= max_w and w2 <= max_w:
                cost = abs(w1 - w2)
                if best is None or cost < best[0]:
                    best = (cost, [" ".join(tokens[:cut]), " ".join(tokens[cut:])])
        if best is not None:
            return best[1]
    lines, overflowed = _greedy_lines(draw, tokens, font, max_w, max_lines)
    if overflowed:
        # 溢出部分在末行加省略号(为省略号腾出宽度)
        while lines[-1] and draw.textlength(lines[-1] + "…", font=font) > max_w:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    return lines


def _render_hud_overlay_png(path: Path, font_path: str, *, caption: str,
                            hud: dict | None) -> None:
    """渲染整帧透明 HUD 贴片(OUT_W x OUT_H)并落盘 PNG。

    gallery 成片(2026-10-01)的 replay_hud 布局:半透明面板直接叠在画面上——
    左上回合标签药丸 + 玩家面板、右上 Boss/角色面板、左下技能信息面板
    (符文图标 + 发光技能名 + 描述 + 敌方行动)。自 engine/montage.py 的
    状态带版本(2026-10-03)移植保留,供建 hud_style="overlay" 时使用。
    """
    img = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f_name = _font(font_path, 20)
    f_tag = _font(font_path, 16)
    f_val = _font(font_path, 14)
    f_skill = _font(font_path, 20)
    f_desc = _font(font_path, 15)

    if hud:
        # -- 左上:小型战斗状态标签(半透明药丸,细金边) ------------------
        tag = tr(hud.get("tag") or caption or "")
        if tag:
            tw = d.textlength(tag, font=f_tag)
            d.rounded_rectangle((14, 12, 14 + tw + 24, 40), 14,
                                fill=(8, 10, 16, 150), outline=GOLD_DIM, width=1)
            d.text((26, 26), tag, font=f_tag, fill=TXT_MAIN, anchor="lm")

        # -- 玩家面板(左上;romance/openworld 无玩家数值则跳过) --------
        if hud.get("left_name"):
            px, py, pw, ph = 14, 50, 300, 92
            d.rounded_rectangle((px, py, px + pw, py + ph), 10, fill=PANEL_BG,
                                outline=PANEL_EDGE, width=1)
            d.line((px + 8, py + 1, px + pw - 8, py + 1), fill=(255, 244, 214, 60),
                   width=1)
            l_font, l_name = _fit_text(d, tr(hud.get("left_name", "")), font_path,
                                       20, pw - 28)
            d.text((px + 14, py + 10), l_name, font=l_font,
                   fill=TXT_MAIN, anchor="la")
            hp_pct, mp_pct, hp_text, mp_text = hud.get(
                "left_stats", (1.0, 1.0, "", ""))
            _bar(d, px + 14, py + 40, pw - 110, 11, hp_pct, HP_FILL, HP_HI, f_val,
                 tr(hp_text), right_edge=px + pw - 10)
            _bar(d, px + 14, py + 62, pw - 110, 11, mp_pct, MP_FILL, MP_HI, f_val,
                 tr(mp_text), right_edge=px + pw - 10)

        # -- 右上面板:优先通用行格式(right_rows),否则按旧类别分支 ------
        theme = hud.get("theme", "duel")
        bw = 360
        rows = hud.get("right_rows")
        bh = (52 + 24 * len(rows)) if rows else 70
        bx, by = OUT_W - bw - 14, 14
        _rounded = d.rounded_rectangle
        _rounded((bx, by, bx + bw, by + bh), 10, fill=PANEL_BG,
                 outline=PANEL_EDGE, width=1)
        d.line((bx + 8, by + 1, bx + bw - 8, by + 1), fill=(255, 244, 214, 60), width=1)
        name_color = {"romance": ROSE, "openworld": QUEST_GOLD}.get(theme, BOSS_GOLD)
        r_font, r_name = _fit_text(d, tr(hud.get("right_name", "")), font_path,
                                   20, bw - 28)
        d.text((bx + 14, by + 8), r_name, font=r_font,
               fill=name_color, anchor="la")
        if rows:
            _BAR_STYLES = {"hp": (HP_FILL, HP_HI), "affection": (ROSE_FILL, ROSE_HI),
                           "stamina": ((92, 178, 96, 255), (150, 218, 128, 255))}
            y = by + 40
            for row in rows:
                kind = row[0]
                if kind == "bar":
                    _, pct, style, text = row
                    fill, hi = _BAR_STYLES.get(style, (HP_FILL, HP_HI))
                    _bar(d, bx + 14, y, bw - 130, 10, pct, fill, hi, f_val, tr(text),
                         right_edge=bx + bw - 14)
                elif kind == "pips":
                    _, done, total, text = row
                    pip_w, gap = 16, 6
                    for k in range(total):
                        fx = bx + 14 + k * (pip_w + gap)
                        _rounded((fx, y, fx + pip_w, y + 12), 3,
                                 (126, 178, 102, 255) if k < done else TRACK,
                                 outline=LEAF_EDGE, width=1)
                    d.text((bx + bw - 14, y + 6), tr(text), font=f_val, fill=LEAF,
                           anchor="rm")
                else:
                    # 垂直居中于行(与 pips 文字同基线),避免压到下一行的条
                    d.text((bx + 14, y + 6), tr(row[1]), font=f_val,
                           fill=TXT_SUB, anchor="lm")
                y += 24
        elif theme == "romance":
            d.text((bx + 14, by + 8), tr(hud.get("right_name", "")), font=f_name,
                   fill=ROSE, anchor="la")
            mood = tr(hud.get("right_sub", ""))
            d.text((bx + bw - 14, by + 12), mood, font=f_val, fill=ROSE,
                   anchor="ra")
            a_pct, a_text = hud.get("right_stats", (1.0, ""))
            _bar(d, bx + 14, by + 40, bw - 130, 12, a_pct, ROSE_FILL, ROSE_HI,
                 f_val, a_text)
        elif theme == "openworld":
            done, total, stage_text = hud.get("right_stats", (0, 1, ""))
            pip_w, gap = 16, 6
            for k in range(total):
                filled = k < done
                _rounded((bx + 14 + k * (pip_w + gap), by + 42,
                          bx + 14 + k * (pip_w + gap) + pip_w, by + 54), 3,
                         (126, 178, 102, 255) if filled else TRACK,
                         outline=LEAF_EDGE, width=1)
            d.text((bx + bw - 14, by + 47), stage_text, font=f_val, fill=LEAF,
                   anchor="rm")
        else:
            e_hp_pct, e_hp_text = hud.get("right_stats", (1.0, ""))
            _bar(d, bx + 14, by + 40, bw - 130, 12, e_hp_pct, HP_FILL, HP_HI,
                 f_val, tr(e_hp_text), right_edge=bx + bw - 14)

        # -- 技能信息面板(左下:符文 + 发光技能名 + 描述 + 敌方行动) --
        # 面板高度随内容行数自适应;台词完整显示,溢出打省略号
        enemy_action = hud.get("enemy_action", "")
        sw = 620
        action_text = tr(hud.get("action", ""))
        action_lines = (_wrap_by_width(d, action_text, f_skill, sw - 84 - 20,
                                       max_lines=99) if action_text else [])
        desc = tr(hud.get("action_desc", ""))
        if desc.strip() == hud.get("action", "").strip():
            desc = ""  # 自由输入类游戏(如恋爱"说话")label 即台词,避免重复
        desc_lines = (_wrap_by_width(d, desc, f_desc, sw - 84 - 46, max_lines=99)
                      if desc else [])
        enemy_lines = (_wrap_by_width(d, tr(enemy_action), f_desc,
                                      sw - 84 - 96, max_lines=1)
                       if enemy_action else [])
        sh = 20 + 26 * len(action_lines) + 20 * len(desc_lines) + \
            (34 if enemy_lines else 0) + 14
        sx, sy = 14, OUT_H - sh - 14
        edge = {"romance": ROSE_EDGE, "openworld": LEAF_EDGE}.get(theme, PANEL_EDGE)
        inner = {"romance": (242, 167, 179, 90),
                 "openworld": (168, 214, 138, 90)}.get(theme, (126, 226, 206, 90))
        name_color = {"romance": ROSE, "openworld": LEAF}.get(theme, CYAN)
        d.rounded_rectangle((sx, sy, sx + sw, sy + sh), 12, fill=PANEL_BG,
                            outline=edge, width=1)
        d.rounded_rectangle((sx + 3, sy + 3, sx + sw - 3, sy + sh - 3), 9,
                            outline=inner, width=1)
        d.line((sx + 10, sy + 1, sx + sw - 10, sy + 1), fill=(255, 244, 214, 60), width=1)
        mid_icon = sy + (20 + 26 * len(action_lines) + 20 * len(desc_lines)) // 2
        if theme == "romance":
            _heart(img, sx + 44, mid_icon, 22)
        elif theme == "openworld":
            _compass_icon(img, sx + 44, mid_icon, 22)
        else:
            _rune_icon(img, sx + 44, mid_icon, 22)
        y = sy + 14
        for line in action_lines:
            _glow_text(img, (sx + 84, y), line, f_skill, name_color)
            y += 26
        for line in desc_lines:
            d.text((sx + 84, y), line, font=f_desc, fill=TXT_SUB, anchor="la")
            y += 20
        if enemy_lines:
            d.line((sx + 72, y + 1, sx + sw - 16, y + 1), fill=GOLD_DIM, width=1)
            label = tr({"romance": "心动变化", "openworld": "冒险见闻"}.get(theme, "敌方行动"))
            d.text((sx + 84, y + 8), label, font=f_tag, fill=BOSS_GOLD, anchor="la")
            lw = d.textlength(label, font=f_tag)
            d.text((sx + 84 + lw + 12, y + 9), enemy_lines[0], font=f_desc,
                   fill=(238, 178, 158, 235), anchor="la")
    elif caption.strip():
        # 场景过渡 / 终局:居中小标签
        text = tr(caption.strip())
        tw = d.textlength(text, font=f_tag)
        d.rounded_rectangle(((OUT_W - tw) / 2 - 18, OUT_H / 2 - 18,
                             (OUT_W + tw) / 2 + 18, OUT_H / 2 + 18), 14,
                            fill=(8, 10, 16, 160), outline=GOLD_DIM, width=1)
        d.text((OUT_W / 2, OUT_H / 2), text, font=f_tag, fill=TXT_MAIN,
               anchor="mm")

    img.save(path)


def _render_hud_png(path: Path, font_path: str, *, caption: str, hud: dict | None) -> None:
    """渲染底部状态带(OUT_W x HUD_BAND_H)并落盘 PNG;画面区域零遮挡。"""
    img = Image.new("RGBA", (OUT_W, HUD_BAND_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, OUT_W, HUD_BAND_H), fill=(8, 10, 16, 210))
    d.line((0, 0, OUT_W, 0), fill=(255, 244, 214, 70), width=1)
    f_name = _font(font_path, 19)
    f_tag = _font(font_path, 15)
    f_val = _font(font_path, 13)
    f_skill = _font(font_path, 17)
    f_desc = _font(font_path, 13)

    if hud:
        # -- 左上:回合状态标签药丸 ----------------------------------------
        tag = tr(hud.get("tag") or "")
        if tag:
            tw = d.textlength(tag, font=f_tag)
            d.rounded_rectangle((14, 10, 14 + tw + 22, 36), 13,
                                fill=(8, 10, 16, 150), outline=GOLD_DIM, width=1)
            d.text((25, 23), tag, font=f_tag, fill=TXT_MAIN, anchor="lm")

        # -- 左下:玩家面板(名称 + HP/MP 条) -----------------------------
        px, py, pw, ph = 14, 44, 320, 104
        d.rounded_rectangle((px, py, px + pw, py + ph), 10, fill=PANEL_BG,
                            outline=PANEL_EDGE, width=1)
        d.line((px + 8, py + 1, px + pw - 8, py + 1), fill=(255, 244, 214, 60), width=1)
        l_font, l_name = _fit_text(d, tr(hud.get("left_name", "")), font_path,
                                   19, pw - 28)
        d.text((px + 14, py + 8), l_name, font=l_font,
               fill=TXT_MAIN, anchor="la")
        hp_pct, mp_pct, hp_text, mp_text = hud.get("left_stats", (1.0, 1.0, "", ""))
        _bar(d, px + 14, py + 42, pw - 116, 11, hp_pct, HP_FILL, HP_HI, f_val,
             tr(hp_text), right_edge=px + pw - 10)
        _bar(d, px + 14, py + 70, pw - 116, 11, mp_pct, MP_FILL, MP_HI, f_val,
             tr(mp_text), right_edge=px + pw - 10)
        act = tr(hud.get("action", ""))
        if act:
            lines = _wrap_by_width(d, act, f_skill, OUT_W - 28, max_lines=1)
            d.text((px + 14, py + 92), lines[0], font=f_skill, fill=CYAN, anchor="la")

        # -- 右侧:Boss 面板(名称 + 条/行,最多两行) ---------------------
        theme = hud.get("theme", "duel")
        bw = 420
        bx, by = OUT_W - bw - 14, 44
        rows = hud.get("right_rows") or []
        bh = 40 + 26 * min(len(rows), 2) + 10 if rows else 82
        d.rounded_rectangle((bx, by, bx + bw, by + bh), 10, fill=PANEL_BG,
                            outline=PANEL_EDGE, width=1)
        d.line((bx + 8, by + 1, bx + bw - 8, by + 1), fill=(255, 244, 214, 60), width=1)
        name_color = {"romance": ROSE, "openworld": QUEST_GOLD}.get(theme, BOSS_GOLD)
        r_font, r_name = _fit_text(d, tr(hud.get("right_name", "")), font_path,
                                   19, bw - 28)
        d.text((bx + 14, by + 8), r_name, font=r_font,
               fill=name_color, anchor="la")
        if rows:
            _BAR_STYLES = {"hp": (HP_FILL, HP_HI), "affection": (ROSE_FILL, ROSE_HI),
                           "stamina": ((92, 178, 96, 255), (150, 218, 128, 255))}
            y = by + 42
            for row in rows[:3]:
                kind = row[0]
                if kind == "bar":
                    _, pct, style, text = row
                    fill, hi = _BAR_STYLES.get(style, (HP_FILL, HP_HI))
                    _bar(d, bx + 14, y, bw - 130, 10, pct, fill, hi, f_val, tr(text),
                         right_edge=bx + bw - 14)
                elif kind == "pips":
                    _, done, total, text = row
                    pip_w, gap = 14, 5
                    for k in range(total):
                        fx = bx + 14 + k * (pip_w + gap)
                        d.rounded_rectangle((fx, y, fx + pip_w, y + 11), 3,
                                            (126, 178, 102, 255) if k < done else TRACK,
                                            outline=LEAF_EDGE, width=1)
                    d.text((bx + bw - 14, y + 5), tr(text), font=f_val, fill=LEAF,
                           anchor="rm")
                else:
                    d.text((bx + 14, y + 4), tr(row[1]), font=f_val,
                           fill=TXT_SUB, anchor="la")
                y += 26
        else:
            e_hp_pct, e_hp_text = hud.get("right_stats", (1.0, ""))
            _bar(d, bx + 14, by + 42, bw - 130, 12, e_hp_pct, HP_FILL, HP_HI,
                 f_val, tr(e_hp_text), right_edge=bx + bw - 14)
            mood = hud.get("right_sub", "")
            if mood:
                d.text((bx + bw - 14, by + 10), tr(mood), font=f_val, fill=name_color,
                       anchor="ra")

        # -- 底部:敌方行动一行 -------------------------------------------
        enemy_action = tr(hud.get("enemy_action", ""))
        if enemy_action:
            label = tr({"romance": "心动变化", "openworld": "冒险见闻"}.get(theme, "敌方行动"))
            d.text((14, 158), label, font=f_tag, fill=BOSS_GOLD, anchor="la")
            lw = d.textlength(label, font=f_tag)
            eline = _wrap_by_width(d, enemy_action, f_desc, OUT_W - 28 - lw - 12,
                                   max_lines=1)
            d.text((14 + lw + 12, 159), eline[0], font=f_desc,
                   fill=(238, 178, 158, 235), anchor="la")
    elif caption.strip():
        # 场景过渡 / 终局:状态带内居中小标签
        text = tr(caption.strip())
        tw = d.textlength(text, font=f_tag)
        d.rounded_rectangle(((OUT_W - tw) / 2 - 18, HUD_BAND_H / 2 - 18,
                             (OUT_W + tw) / 2 + 18, HUD_BAND_H / 2 + 18), 14,
                            fill=(8, 10, 16, 160), outline=GOLD_DIM, width=1)
        d.text((OUT_W / 2, HUD_BAND_H / 2), text, font=f_tag, fill=TXT_MAIN,
               anchor="mm")

    img.save(path)


def _run_ffmpeg(command: list[str]) -> None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise MontageError("ffmpeg timed out during montage step") from exc
    except OSError as exc:
        raise MontageError(f"failed to launch ffmpeg: {exc}")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise MontageError("ffmpeg failed during montage: " + " | ".join(tail))
