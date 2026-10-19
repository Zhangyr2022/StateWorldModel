"""界面文案中英对照(纯数据,不依赖 gradio)。

界面语言 lang ∈ {"en", "zh"};默认跟随 AGENTIC_GAME_HUD_LANG(engine.hud_i18n.LANG),
与回放 HUD 用同一个开关。游戏内容(剧情、技能名)不在这里——英文界面下
由 engine.hud_i18n.tr 翻译,词表未收录的原样透传。
"""

from __future__ import annotations

from engine.hud_i18n import LANG as _ENV_LANG

DEFAULT_LANG = "zh" if _ENV_LANG == "zh" else "en"

# 语言切换单选:(展示, 取值)
LANG_CHOICES = (("EN", "en"), ("中文", "zh"))

_TEXT = {
    # -- 顶栏 / 标签页 ---------------------------------------------------
    "app_title": ("生成式回合游戏台", "Generative Turn-Based Arcade"),
    "app_sub": ("AI-RENDERED · TURN-BASED · 每一幕都由 AI 现场生成",
                "AI-RENDERED · TURN-BASED · EVERY SCENE GENERATED LIVE"),
    "tab_lobby": ("🎮 游戏大厅", "🎮 Lobby"),
    "tab_play": ("🎬 对局", "🎬 Play"),
    # -- 大厅 ------------------------------------------------------------
    "sec_pick": ("选择你的冒险", "Choose Your Adventure"),
    "sec_pick_sub": ("点击封面查看详情", "Click a cover for details"),
    "no_blurb": ("*这个游戏还没有简介。*", "*No description yet.*"),
    "start": ("▶  开始游戏", "▶  START GAME"),
    "advanced": ("⚙  高级设置", "⚙  Advanced Settings"),
    "backend": ("渲染后端(h3 渲染一段约 1~3 分钟)",
                "Render backend (h3 takes ~1-3 min per scene)"),
    "seed": ("随机种子", "Random seed"),
    "max_turns": ("最大回合数", "Max turns"),
    "scene": ("开场场景", "Opening scene"),
    "turns_n": ("{n} 回合", "{n} turns"),
    "turns_any": ("回合数随开局设置", "Turns set at start"),
    "needs_llm": ("需要对话服务", "Needs LLM service"),
    "how_to_play": ("玩法", "HOW TO PLAY"),
    "lang_zh": ("中文", "Chinese"),
    "lang_en": ("EN", "English"),
    "lang_en_only": ("仅英文", "English only"),
    # -- 对局 ------------------------------------------------------------
    "chapter": ("剧情事件 · {i} / {n}", "STORY EVENT · {i} / {n}"),
    "sec_status": ("当前局势", "Situation"),
    "sec_actions": ("你的行动", "Your Move"),
    "idle_status": ("先在大厅选择游戏,点击「开始游戏」。",
                    "Pick a game in the lobby and press START GAME."),
    "turn_video": ("回合演出", "Scene"),
    "montage_btn": ("🎬 生成完整回放", "🎬 Build Full Replay"),
    "montage_video": ("完整回放(带状态字幕)", "Full replay (with HUD)"),
    "action_label": ("本回合行动", "Action this turn"),
    "step": ("⚔  执行行动", "⚔  TAKE ACTION"),
    "debug": ("🐞 Debug:接管 Boss 行动", "🐞 Debug: control the Boss"),
    "boss_pick": ("强制本回合 Boss 行动", "Force the Boss action this turn"),
    "log": ("📜  战况记录", "📜  Battle Log"),
    "log_title": ("**战况记录**", "**Battle Log**"),
    "log_empty": ("还没有进行任何回合。", "No turns played yet."),
    "back": ("←  返回大厅", "←  Back to Lobby"),
    "ongoing": ("● 进行中", "● In progress"),
    "free_text": ("自由输入", "Free input"),
    # -- 终局 / 状态 -----------------------------------------------------
    "status_ongoing": ("进行中", "In progress"),
    "status_won": ("🏆 胜利", "🏆 Victory"),
    "status_lost": ("💀 失败", "💀 Defeat"),
    "status_draw": ("🤝 平局", "🤝 Draw"),
    "out_of_turns": ("回合数用尽,未分出胜负", "Out of turns, no winner"),
    "no_actions": ("没有可选行动了", "No actions left"),
    # -- 提示 / 错误 -----------------------------------------------------
    "turn_failed": ("本回合未执行:{err}", "Turn not played: {err}"),
    "err_pick_game": ("请先在大厅选择一个游戏", "Pick a game in the lobby first"),
    "err_unknown_game": ("游戏不存在或已下线:{id}", "Game not found: {id}"),
    "err_start": ("开局失败:{err}", "Failed to start: {err}"),
    "err_no_session": ("请先点击「开始游戏」", "Press START GAME first"),
    "err_no_game": ("请先开始一局游戏", "Start a game first"),
    "err_montage": ("回放拼接失败:{err}", "Replay build failed: {err}"),
    "err_no_free_text": ("这个游戏不支持自由输入,请点选行动",
                         "This game has no free input; pick an action"),
    "err_action_stale": ("行动已失效,请重新选择:{label}",
                         "That action is no longer available: {label}"),
    "err_pick_action": ("请先选择一个行动(或输入台词)",
                        "Pick an action (or type a line) first"),
    "err_boss_stale": ("Boss 行动已失效,请重新选择:{label}",
                       "That Boss action is no longer available: {label}"),
    "err_no_video": ("没有可拼接的视频(渲染失败的回合不会进入回放)",
                     "No videos to stitch (failed renders are skipped)"),
}

# 渲染后端单选:取值保持中文原文(脚本 / gradio_client 按值调用),只换展示
BACKEND_LABELS_EN = {
    "fake · 离线占位(无需 GPU)": "fake · offline placeholder (no GPU)",
    "h3 remote · 常驻渲染服务": "h3 remote · resident render service",
    "h3 session · 临时渲染会话": "h3 session · temporary render session",
}

CATEGORY_HINT = {
    "zh": {
        "duel": "点选一个技能作为本回合行动。",
        "machine": "点选一个技能作为本回合行动。",
        "romance": "在输入框写下你想说的话,点击「执行行动」。",
        "adv": "在输入框写下你想说的话,点击「执行行动」。剧情会随回合推进。",
        "openworld": "有值得选择的事时:点选行动或自由输入;画面平淡时本回合自动探索"
                     "(不弹选择直接播放),AUTO 后必定回到交互。仅支持英文。",
    },
    "en": {
        "duel": "Pick a skill as your action this turn.",
        "machine": "Pick a skill as your action this turn.",
        "romance": "Type what you want to say, then press TAKE ACTION.",
        "adv": "Type what you want to say, then press TAKE ACTION. "
               "The story advances each turn.",
        "openworld": "When there's a real choice, pick an action or type your own; "
                     "quiet moments auto-explore (plays without asking) and always "
                     "hand control back to you next turn. English only.",
    },
}

FREE_TEXT_LABEL = {
    "zh": {"romance": "💬 对她说", "adv": "💬 对她说",
           "openworld": "✍️ 自定义行动(可选)"},
    "en": {"romance": "💬 Say to her", "adv": "💬 Say to her",
           "openworld": "✍️ Custom action (optional)"},
}

# registry 选项标签(中文)→ 英文
OPTION_LABELS_EN = {"开场场景": "Opening scene"}


def normalize(lang: str | None) -> str:
    return "zh" if (lang or "").strip().lower() == "zh" else "en"


def t(lang: str | None, key: str, **fmt) -> str:
    """取界面文案;未知 key 原样返回(便于发现漏译)。"""
    pair = _TEXT.get(key)
    if pair is None:
        return key
    text = pair[0] if normalize(lang) == "zh" else pair[1]
    return text.format(**fmt) if fmt else text


def backend_choices(choices, lang: str | None) -> list[tuple[str, str]]:
    """渲染后端 Radio 的 (展示, 取值) 列表。"""
    if normalize(lang) == "zh":
        return [(c, c) for c in choices]
    return [(BACKEND_LABELS_EN.get(c, c), c) for c in choices]


def status_label(status: str, lang: str | None) -> str:
    key = f"status_{status}"
    return t(lang, key) if key in _TEXT else status


def hint(category: str, lang: str | None) -> str:
    return CATEGORY_HINT[normalize(lang)].get(category, "")


def free_text_label(category: str, lang: str | None) -> str:
    return FREE_TEXT_LABEL[normalize(lang)].get(category, t(lang, "free_text"))


def option_label(label: str, lang: str | None) -> str:
    return OPTION_LABELS_EN.get(label, label) if normalize(lang) == "en" else label
