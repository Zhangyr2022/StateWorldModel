#!/usr/bin/env python3
"""把旧会话的回合视频重拼为带主题 HUD 的完整回放(duel/machine/romance/adv/openworld 通用)。

用法:python3 scripts/rebuild_replay_hud.py <会话目录> [输出文件名]

会话目录需含 turn_log.jsonl(每行一条回合记录,带 state_after/events 快照)
与 videos/turn-*.mp4。HUD 数据从快照精确重建;产物默认写到
<会话目录>/replay_hud.mp4。

章节转场(adv):skin 可解析时,开场与每个 node_event 回合前插入一张
galgame 式章节卡(节点题 + nodes[].event 剧情旁白,金色饰线 + 衬线排版,
节奏:淡入 0.6s → 停留(中文 22 字/秒 3s 封顶 / 英文 3 词/秒 5.5s 封顶)
→ 淡出 0.6s 退净 → 0.2s 黑场 → 下一段首帧 freeze 0.3s(无声)→ 正片),
过渡块渲染见 _transition_clip(单文件连续时间线,不经 concat -c copy)。

兼容性:machine(状态机)主题的 state_after 自 2026-09-30 起落盘为结构化
快照(to_snapshot 钩子);更早的会话该字段是对象 repr 字符串,无法重建
面板——这些回合退化为仅字幕(不再崩溃),终局动画照常居中字幕。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.hud_i18n import hud_lang, tr  # noqa: E402
from engine.montage import (  # noqa: E402
    DEFAULT_FONT,
    ClipSpec,
    _run_ffmpeg,
    _wrap_by_width,
    build_game_montage,
)


def _events_of(rec):
    return rec.get("events") or []


def _event_get(e, key, default=None):
    if isinstance(e, dict):
        return e.get(key, default)
    return getattr(e, key, default)


def hud_duel(state, rec):
    p, e = state.get("player", {}), state.get("enemy", {})
    return {
        "left_name": p.get("name", ""),
        "left_stats": (p.get("hp", 1) / p.get("max_hp", 1) if p.get("max_hp") else 1.0,
                       p.get("mp", 1) / p.get("max_mp", 1) if p.get("max_mp") else 1.0,
                       f"{p.get('hp', '?')} / {p.get('max_hp', '?')}",
                       f"{p.get('mp', '?')} / {p.get('max_mp', '?')}"),
        "right_name": e.get("name", ""),
        "right_stats": (e.get("hp", 1) / e.get("max_hp", 1) if e.get("max_hp") else 1.0,
                        f"{e.get('hp', '?')} / {e.get('max_hp', '?')}"),
    }


def hud_machine(state, rec, skin):
    """machine(状态机)HUD:数值从 vars 快照重建,上限取皮肤初始值。"""
    v = state.get("vars", {})
    init = (skin.vars if skin is not None else None) or {}
    boss = skin.boss if skin is not None else None
    p_hp, p_mp = v.get("player_hp", 0), v.get("player_mp", 0)
    p_hp_max = init.get("player_hp") or p_hp or 1
    # MP 上限:皮肤声明 mp_max(现役 15)优先;否则按初始 MP 封顶
    p_mp_max = getattr(skin, "mp_max", 0) or init.get("player_mp") or 1
    mp_label = getattr(skin, "mp_name", "") or "MP"
    b_hp = v.get("boss_hp")
    b_hp_max = init.get("boss_hp") or b_hp or 1
    marks = [t for t, on in (("Boss盾", v.get("boss_shield")),
                             ("狂暴", v.get("boss_enraged")),
                             ("蓄力中", v.get("windup"))) if on]
    boss_action = ""
    last = state.get("last_boss_action")
    if boss is not None and last:
        try:
            boss_action = boss.action(last).label
        except KeyError:
            boss_action = str(last)
    enemy_line = f"Boss 招式:{boss_action}" if boss_action else "Boss 按兵不动"
    if marks:
        enemy_line += f"({','.join(marks)})"
    return {
        "left_name": skin.player_name if skin is not None else "玩家",
        "left_stats": (min(p_hp / p_hp_max, 1.0) if p_hp_max else 1.0,
                       min(p_mp / p_mp_max, 1.0) if p_mp_max else 1.0,
                       f"{p_hp} / {p_hp_max}",
                       f"{mp_label} {p_mp} / {p_mp_max}"),
        "right_name": boss.name if boss is not None else "Boss",
        "right_stats": (min(b_hp / b_hp_max, 1.0) if b_hp else 0.0,
                        f"{b_hp} / {b_hp_max}"),
        "enemy_action": enemy_line,
    }


def hud_romance(state, rec):
    from game.romance.state import MOOD_LABELS
    from game.romance.visual import SCENES

    delta = 0
    for e in _events_of(rec):
        if _event_get(e, "type") == "affection_change":
            delta = _event_get(e, "amount", 0) or 0
    mood = MOOD_LABELS.get(state.get("mood", ""), state.get("mood", ""))
    scene = SCENES.get(state.get("scene", ""), {}).get("label", state.get("scene", ""))
    a = state.get("affection", 0)
    if a >= 75:   rel = "恋人"
    elif a >= 60: rel = "暧昧"
    elif a >= 40: rel = "朋友"
    elif a >= 15: rel = "初识"
    else:         rel = "陌生疏远"
    return {
        "theme": "romance",
        "tag": f"第{state.get('turn', '?')}回合 · {scene}",
        "right_name": state.get("girl_name", ""),
        "right_rows": [
            ("bar", a / 100.0, "affection", f"好感 {a} / 100"),
            ("text", f"关系:{rel} · 心情:{mood}"),
            ("location", f"地点:{scene}"),
        ],
        "action_desc": state.get("last_player_line", ""),
        "enemy_action": (f"好感 {delta:+d} · 她看起来{mood}" if delta
                         else f"好感持平 · 她看起来{mood}"),
    }


_ADV_MOOD_LABELS = {"calm": "平静", "warm": "温软", "guarded": "戒备",
                    "tired": "疲惫", "sad": "低落", "flustered": "慌乱"}


def _load_adv_skin(game_id: str):
    from game.adv_theme.json_skin import load_json_skin
    return load_json_skin(REPO_ROOT / "game" / "skins" / f"{game_id}.json")


def hud_adv(state, rec, skin):
    """adv(剧情 ADV)HUD:关系值条 + 剧情节点 + 心情措辞。

    关系值在游戏内对玩家隐藏(show_relationship=false),但 replay HUD 是
    回看/评审工具,数值可见是有意为之(与 turn_log 快照一致)。
    """
    delta = 0
    for e in _events_of(rec):
        if _event_get(e, "type") == "relationship_change":
            delta = _event_get(e, "amount", 0) or 0
    rel_max = getattr(skin, "relationship_max", 100) if skin is not None else 100
    rel = state.get("relationship", 0)
    node_id = state.get("node", "")
    node_label = node_id
    if skin is not None:
        try:
            node_label = skin.node_for_turn(max(int(state.get("turn", 1)), 1)).label
        except Exception:  # noqa: BLE001 - 快照越界时退回节点 id
            pass
    mood = _ADV_MOOD_LABELS.get(state.get("mood", ""), state.get("mood", ""))
    if rel >= 75:   rel_word = "心意接近摊牌"
    elif rel >= 55: rel_word = "开始习惯你"
    elif rel >= 35: rel_word = "礼貌的同学"
    else:           rel_word = "疏远客气"
    rows = [
        ("bar", min(rel / max(rel_max, 1), 1.0), "relationship",
         f"关系 {rel}/{rel_max}"),
        ("text", f"关系:{rel_word} · 心情:{mood}"),
        ("location", f"剧情:{node_label}"),
    ]
    if state.get("intent_turn"):
        rows.append(("text", f"恋爱意图@第{state['intent_turn']}回合"))
    if state.get("ending_id"):
        rows.append(("text", f"结局:{state['ending_id']}"))
    girl_name = getattr(skin, "girl_name", "") or "她"
    return {
        "theme": "romance",  # 复用玫瑰面板配色(剧情向恋爱,语义一致)
        "tag": f"第{state.get('turn', '?')}回合 · {node_label}",
        "right_name": girl_name,
        "right_rows": rows,
        "action_desc": (state.get("history") or [("", "")])[-1][0],
        "enemy_action": (f"态度 {delta:+d} · {girl_name}看起来{mood}" if delta
                         else f"态度持平 · {girl_name}看起来{mood}"),
    }


def hud_openworld(state, rec):
    from game.openworld import clip_hud_text
    from game.openworld.quests import quest_for
    from game.openworld.state import MODE_AUTO

    quest = quest_for(state.get("scene_id", "western"))
    stage = min(state.get("quest_stage", 0), quest.total - 1)
    story = state.get("story") or []
    location = clip_hud_text(state.get("scene_name"), 26)
    auto = state.get("last_mode") == MODE_AUTO  # 旧会话无该字段则视为交互
    rows = [("text", f"地点:{location or quest.title}")]
    if auto:
        rows.append(("text", "本幕自动探索 · 下一回合由你决定"))
    if "stamina" in state:  # 旧会话无体力字段则省略
        st = state["stamina"]
        rows.append(("bar", st / 100.0, "stamina", f"体力 {st}/100"))
    rows.append(("pips", min(state.get("quest_stage", 0), quest.total), quest.total,
                 f"主线《{quest.title}》 {stage + 1}/{quest.total}"))
    return {
        "theme": "openworld",
        "tag": f"第{state.get('turn', '?')}幕 · {state.get('scene_name', '')}"
               + (" · 自动探索" if auto else ""),
        "right_name": "旅行者",
        "right_rows": rows,
        "action_desc": clip_hud_text(story[-1]) if story else "冒险刚刚开始",
    }


def _detect_kind(state):
    if not isinstance(state, dict):
        return "duel"
    if "vars" in state and "current" in state:
        return "machine"
    if "relationship" in state and "node" in state:
        return "adv"
    if "affection" in state:
        return "romance"
    if "quest_stage" in state or "scene_name" in state:
        return "openworld"
    return "duel"


def _log_prefers_english(log) -> bool:
    """按日志文本判断会话语言:首回合动作/事件无中文即视为英文局。

    zh 与 en 皮肤的 girl_label 相同(如 "the schoolgirl"),反查皮肤时用
    该标记在同名皮肤里挑对语言。
    """
    for rec in (log or [])[:3]:
        label = (rec.get("action") or {}).get("label", "")
        if label and not _has_cjk(label):
            return True
        for e in _events_of(rec):
            msg = _event_get(e, "message") or ""
            if msg and not _has_cjk(msg):
                return True
            if msg:
                break
    return False


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s or "")


def _find_skin(vs, prefer_lang: str = "zh"):
    """按 visual_script 的玩家英文标识反查注册表中的皮肤(machine/duel 通用)。

    machine 的 visual_script meta 不带 game_id,用 player_label 匹配最稳;
    查不到返回 None(HUD 退化为无皮肤信息,数值上限按当前值)。
    adv 的 zh/en 皮肤共用 girl_label,按 prefer_lang(会话语言)挑选。
    """
    if not isinstance(vs, dict):
        return None
    actor = (vs.get("actor") or "").strip()
    if not actor:
        return None
    from game.registry import GAMES

    adv_match = None
    for entry in GAMES.values():
        try:
            if entry.category == "adv":
                # adv 组装需要 LLM 客户端,直接从皮肤文件加载(反查用 girl_label)
                skin = _load_adv_skin(entry.game_id)
                if getattr(skin, "girl_label", "") == actor:
                    lang = getattr(skin, "language", "zh")
                    if lang == prefer_lang:
                        return skin
                    adv_match = adv_match or skin
                continue
            runtime = entry.build(None).runtime
        except Exception:  # noqa: BLE001 - 个别游戏构建失败不该拖垮重拼
            continue
        skin = getattr(runtime, "skin", None)
        if skin is not None and getattr(skin, "player_label", "") == actor:
            return skin
    return adv_match


def _descriptions_of(skin):
    """皮肤的动作说明表(duel = skills,machine = actions;其余为空)。"""
    skills = getattr(skin, "skills", None)
    if skills:
        return {s.id: s.description for s in skills.values()}
    return {a.id: a.description for a in getattr(skin, "actions", ()) or ()}


# ---------------------------------------------------------------------------
# 章节转场卡(adv):节点开始的回合前插入一张 galgame 式章节页,
# 展示 nodes[].event 的那段剧情旁白。纯表现层:只影响 replay_hud 成片。
# ---------------------------------------------------------------------------

_SERIF_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
_SERIF_BOLD = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
# 淡入/淡出时长与黑场缓冲(秒)——章节卡节奏:淡入 → 停留 → 淡出退净 →
# 0.2s 黑场 → 下一段首帧 freeze 0.3s(无声)→ 正片(2026-10-15 按反馈定稿)
CHAPTER_FADE_S = 0.6
CHAPTER_GAP_S = 0.2
FREEZE_S = 0.3


def _chapter_card_png(path: Path, label: str, text: str, font_path: str) -> None:
    """渲染一张章节卡:暗底渐晕 + 金色饰线 + 节点题(衬线粗体)+ 事件段落。

    排版即 galgame 章节页惯例:题头居中、段落居中、大量留白;配色沿用
    montage 的金色系,与 HUD 面板同宗。text 为空时只渲染题头(duel 的
    XX v.s. XX 开场卡走这个形态,见 _vs_card_png)。
    """
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = 864, 480
    card = Image.new("RGB", (w, h), (10, 13, 22))
    # 中央略亮的径向渐晕:给暗底一点纵深,避免死黑
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((w * 0.08, h * 0.02, w * 0.92, h * 0.98), fill=30)
    card = Image.composite(Image.new("RGB", (w, h), (24, 29, 46)), card,
                           mask.filter(ImageFilter.GaussianBlur(70)))
    d = ImageDraw.Draw(card, "RGBA")

    gold, body, sub = (222, 186, 96), (234, 236, 242), (168, 174, 188)
    serif = _SERIF_FONT if Path(_SERIF_FONT).exists() else font_path
    serif_bold = _SERIF_BOLD if Path(_SERIF_BOLD).exists() else serif
    f_label = ImageFont.truetype(serif_bold, 31)
    f_body = ImageFont.truetype(serif, 21)

    cx, y = w // 2, 92
    # 饰线:线 — 菱形 — 线
    d.line((cx - 132, y, cx - 20, y), fill=gold + (170,), width=1)
    d.line((cx + 20, y, cx + 132, y), fill=gold + (170,), width=1)
    d.regular_polygon((cx, y, 5), 4, rotation=45, outline=gold)

    # 节点题(加字距,居中)
    track = 8
    widths = [d.textlength(ch, font=f_label) for ch in label]
    total = sum(widths) + track * (len(label) - 1)
    x = cx - total / 2
    for ch, cw in zip(label, widths):
        d.text((x, y + 22), ch, font=f_label, fill=gold + (255,), anchor="la")
        x += cw + track

    # 事件段落:按宽度折行、居中,行距 1.6;整块在题头下方竖直居中(text 空 = 无段落)
    lines = []
    if text.strip():
        max_w, line_h = 700, 34
        lines = _wrap_by_width(d, text, f_body, max_w, max_lines=9)
        if len(lines) >= 7:  # 长段落换小字号,避免顶到饰线
            f_body = ImageFont.truetype(serif, 19)
            line_h = 30
            lines = _wrap_by_width(d, text, f_body, max_w, max_lines=10)
        block_h = line_h * len(lines)
        ty = max(y + 74, (h - block_h) // 2)
        for line in lines:
            d.text((cx, ty), line, font=f_body, fill=body + (238,), anchor="ma")
            ty += line_h

    # 底部小菱形收尾
    d.regular_polygon((cx, h - 46, 4), 4, rotation=45, fill=gold + (200,))
    card.save(path)


def _card_hold(text: str) -> float:
    """章节卡停留时长:中文按 22 字/秒(上限 3s);拉丁文按 3 词/秒
    (上限 5.5s——英文章节段约为中文 2~3 倍长,按词数给足阅读时间)。"""
    if any("一" <= ch <= "鿿" for ch in text):
        return min(3.0, max(1.2, len(text) / 22.0))
    return min(5.5, max(1.2, len(text.split()) / 3.0))


def _transition_clip(work_dir: Path, index: int, card_png: Path, hold: float,
                     next_video: Path) -> Path:
    """过渡块:章节卡(淡入·停留·淡出)→ 0.2s 黑场 → 下一段首帧 freeze
    0.3s(无声)→ 交回正片。三段在同一次 ffmpeg 编码里 concat,时间线
    连续,整块作为单一零件进入 montage。

    不再预渲染卡片 mp4 再 concat -c copy:旧法用 `-loop 1 -t X` + 输出端
    `-r 24`,25fps→24fps 重采样使视频流比音频流长 ~0.6s,concat 按流各自
    偏移,黑场期间会漏出下一段声音、正片开头被卡的静止尾巴占据(2026-10-15
    adv_eng 实测)。图像输入一律走 `-framerate 24`(demuxer 侧定帧率,无
    重采样),视频/音频时长精确一致。
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    frame = work_dir / f"intro-{index:02d}-frame.png"
    _run_ffmpeg(["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                 "-i", str(next_video), "-frames:v", "1", str(frame)])
    card_s = hold + 2 * CHAPTER_FADE_S
    out = work_dir / f"intro-{index:02d}.mp4"
    _run_ffmpeg([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-framerate", "24", "-loop", "1", "-t", f"{card_s:.3f}", "-i", str(card_png),
        "-f", "lavfi", "-t", f"{CHAPTER_GAP_S:.3f}", "-i",
        "color=c=0x0b0e16:s=864x480:r=24",
        "-framerate", "24", "-loop", "1", "-t", f"{FREEZE_S:.3f}", "-i", str(frame),
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-filter_complex",
        (f"[0:v]fade=t=in:st=0:d={CHAPTER_FADE_S:.2f},"
         f"fade=t=out:st={hold:.2f}:d={CHAPTER_FADE_S:.2f},"
         f"format=yuv420p,setsar=1[c];"
         f"[2:v]scale=864:480,format=yuv420p,setsar=1[f];"
         f"[c][1:v][f]concat=n=3:v=1:a=0[v]"),
        "-map", "[v]", "-map", "3:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-shortest", str(out),
    ])
    frame.unlink(missing_ok=True)
    return out


def _vs_card_clip(work_dir: Path, name_a: str, name_b: str,
                  font_path: str, next_video: Path) -> Path:
    """VS 开场卡过渡块:淡入·停留·淡出 → 黑场 → 下一段首帧 freeze。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    png = work_dir / "vs.png"
    _vs_card_png(png, name_a, name_b, font_path)
    clip = _transition_clip(work_dir, 0, png, 2.2, next_video)
    png.unlink(missing_ok=True)
    return clip


def _vs_card_png(path: Path, name_a: str, name_b: str, font_path: str) -> None:
    """duel 开场卡:双方名号对峙(上 A 下 B,金色 v.s. 分隔),同章节卡视觉。"""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = 864, 480
    card = Image.new("RGB", (w, h), (10, 13, 22))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((w * 0.08, h * 0.02, w * 0.92, h * 0.98), fill=30)
    card = Image.composite(Image.new("RGB", (w, h), (24, 29, 46)), card,
                           mask.filter(ImageFilter.GaussianBlur(70)))
    d = ImageDraw.Draw(card, "RGBA")
    gold, body, sub = (222, 186, 96), (234, 236, 242), (168, 174, 188)
    serif = _SERIF_FONT if Path(_SERIF_FONT).exists() else font_path
    serif_bold = _SERIF_BOLD if Path(_SERIF_BOLD).exists() else serif
    f_vs = ImageFont.truetype(font_path, 18)

    def _name_font(text, base=34):
        """名号过长(英文名普遍较长)逐级缩号 + 收字距,保证不出画幅。"""
        size, track = base, 6
        while size > 14:
            f = ImageFont.truetype(serif_bold, size)
            total = (sum(d.textlength(ch, font=f) for ch in text)
                     + track * (len(text) - 1))
            if total <= w - 60:
                return f, track
            size -= 2
            track = max(1, track - 1)
        return ImageFont.truetype(serif_bold, 14), 1

    def _center(text, y, font, color, track=6):
        """水平居中 + 垂直中点锚(mm):y = 字形视觉中心,保证上下对称。"""
        widths = [d.textlength(ch, font=font) for ch in text]
        x = w / 2 - (sum(widths) + track * (len(text) - 1)) / 2
        for ch, cw in zip(text, widths):
            d.text((x, y), ch, font=font, fill=color, anchor="lm")
            x += cw + track

    cx, cy = w // 2, h // 2
    # 三段对称:名 A(上)— v.s. + 饰线(正中)— 名 B(下),整体以画面正中为轴
    f_a, t_a = _name_font(name_a)
    f_b, t_b = _name_font(name_b)
    _center(name_a, cy - 78, f_a, body + (246,), track=t_a)
    d.line((cx - 96, cy, cx - 26, cy), fill=gold + (150,), width=1)
    d.line((cx + 26, cy, cx + 96, cy), fill=gold + (150,), width=1)
    _center("v.s.", cy, f_vs, gold + (230,), track=3)
    _center(name_b, cy + 78, f_b, body + (246,), track=t_b)
    d.regular_polygon((cx, h - 46, 4), 4, rotation=45, fill=gold + (200,))
    card.save(path)


def main(argv: list[str]) -> int:
    session = Path(argv[1]) if len(argv) > 1 else None
    if session is None or not (session / "turn_log.jsonl").exists():
        print(__doc__)
        return 2
    out_name = argv[2] if len(argv) > 2 else "replay_hud.mp4"

    log = [json.loads(line) for line in
           (session / "turn_log.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
    # 会话目录可能含多次运行追加的记录:从"最后一次 turn-001"起读新局,
    # 顺序读到第一个 won/lost(终局)为止,丢弃更早运行的历史残留。
    starts = [i for i, r in enumerate(log) if r.get("turn_id") == "turn-001"]
    log = log[starts[-1]:] if starts else log

    skin = _find_skin(log[0].get("visual_script") if log else {},
                      prefer_lang="en" if _log_prefers_english(log) else "zh")
    descriptions = _descriptions_of(skin)

    ENDING_CAPTIONS = {
        "romance": {"won": "◆ 告白成功 ◆", "lost": "◆ 她离开了 ◆"},
        "openworld": {"won": "◆ 剧终 ◆", "lost": "◆ 剧终 ◆"},
        "duel": {"won": "◆ 战斗胜利 ◆", "lost": "◆ 战斗失败 ◆"},
        "machine": {"won": "◆ 战斗胜利 ◆", "lost": "◆ 战斗失败 ◆"},
        "adv": {"won": "◆ 故事落幕 ◆", "lost": "◆ 故事落幕 ◆", "draw": "◆ 故事落幕 ◆"},
    }
    clips: list[ClipSpec] = []
    last_kind = "duel"
    # 过渡块(开场卡/章节卡/VS 卡):卡退净 → 0.2s 黑场 → 下一段首帧
    # freeze 0.3s → 正片;freeze 需要下一段视频的首帧,先找到第一个
    # 实际存在的回合视频。
    first_video: Path | None = None
    for rec in log:
        if rec["turn_id"].startswith("ending"):
            continue
        candidate = session / "videos" / f"{rec['turn_id']}.mp4"
        if candidate.exists():
            first_video = candidate
            break
    # 章节转场(adv):节点开始时插入剧情旁白章节卡;节点一没有 node_event
    # 事件,开场卡从皮肤 nodes[0] 取文案——五段剧情因此都有章节页。
    cards_dir = session / "chapter_cards"
    chapter_no = 0
    opening_card: Path | None = None
    if (skin is not None and hasattr(skin, "node_for_turn") and log
            and first_video is not None):
        node = skin.node_for_turn(1)
        if getattr(node, "event", ""):
            cards_dir.mkdir(parents=True, exist_ok=True)
            png = cards_dir / "chapter-opening.png"
            _chapter_card_png(png, node.label, node.event, DEFAULT_FONT)
            opening_card = _transition_clip(
                cards_dir, 1, png, _card_hold(node.event), first_video)
            chapter_no = 1
    # 开场卡(duel/machine):XX v.s. XX 双方名号对峙,与 adv 章节卡同视觉
    # (名号经 tr():英文 HUD 时开场卡同为英文名)
    vs_card: Path | None = None
    if (first_video is not None
            and _detect_kind(log[0].get("state_after") or {}) == "machine"
            and skin is not None and getattr(skin, "boss", None) is not None):
        vs_card = _vs_card_clip(cards_dir, tr(skin.player_name),
                                tr(skin.boss.name), DEFAULT_FONT, first_video)
    for i, rec in enumerate(log):
        video = session / "videos" / f"{rec['turn_id']}.mp4"
        if not video.exists():
            continue
        # 节点事件 → 过渡块(章节卡退净 → 黑场 → 本段首帧 freeze)再进正片
        # (node_event.message = "—— label ——\nevent")
        node_ev = next((e for e in _events_of(rec)
                        if _event_get(e, "type") == "node_event"), None)
        if node_ev is not None:
            raw = (_event_get(node_ev, "message") or "").strip()
            label, _, text = raw.partition("\n")
            label = label.strip("—－-—~～ ").strip()
            text = text.strip()
            if label and text:
                chapter_no += 1
                cards_dir.mkdir(parents=True, exist_ok=True)
                png = cards_dir / f"chapter-{chapter_no:02d}.png"
                _chapter_card_png(png, label, text, DEFAULT_FONT)
                clips.append(ClipSpec(video=_transition_clip(
                    cards_dir, chapter_no, png, _card_hold(text), video)))
        state = rec.get("state_after") or rec.get("state_before") or {}
        action = rec.get("action") or {}
        label = action.get("label", "")
        if rec["turn_id"].startswith("ending"):
            # 终局动画:无 HUD,只放居中终局字幕(此前被误画成空 duel 面板)
            caption = ENDING_CAPTIONS[last_kind].get(rec.get("status", ""), "")
            clips.append(ClipSpec(video=video, caption=caption))
            break
        if not isinstance(state, dict) or not state:
            # 旧格式会话(如 2026-09-30 前的 machine 日志,state 为 repr 字符串):
            # 无法重建面板,退化为仅字幕,不再崩溃。
            clips.append(ClipSpec(video=video, caption=f"第{i + 1}回合 · {label}"))
            continue
        kind = _detect_kind(state)
        last_kind = kind
        hud = {"tag": f"第{i + 1}回合 · {label}"
               if kind in ("duel", "machine") else None,
               "action": label}
        if kind == "machine":
            hud.update(hud_machine(state, rec, skin))
            hud["action_desc"] = descriptions.get(action.get("id", ""), "")
        elif kind == "adv":
            hud.update(hud_adv(state, rec, skin))
        elif kind == "romance":
            hud.update(hud_romance(state, rec))
        elif kind == "openworld":
            hud.update(hud_openworld(state, rec))
        else:
            hud.update(hud_duel(state, rec))
            hud["action_desc"] = descriptions.get(action.get("id", ""), "")
        if not hud.get("tag"):
            hud["tag"] = f"第{i + 1}回合 · {label}"
        clips.append(ClipSpec(video=video, caption=hud["tag"], hud=hud))

    if not clips:
        print("没有可用回合视频")
        return 1
    if opening_card is not None:
        clips.insert(0, ClipSpec(video=opening_card))
    if vs_card is not None:
        clips.insert(0, ClipSpec(video=vs_card))
    output = session / out_name
    # replay_hud 用 gallery 成片时代的 overlay 面板风格(864x480,面板叠在
    # 画面上);game_montage 仍是状态带风格,见 engine/montage.py hud_style。
    # 开放世界只支持英文:HUD 一律英文,不受 AGENTIC_GAME_HUD_LANG 影响
    with hud_lang("en" if last_kind == "openworld" else None):
        build_game_montage(clips, output, hud_style="overlay")
    shutil.rmtree(cards_dir, ignore_errors=True)  # 章节卡零件已拼入成片
    print(f"完成:{output}({len(clips)} 个片段,含 {chapter_no} 张章节卡)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
