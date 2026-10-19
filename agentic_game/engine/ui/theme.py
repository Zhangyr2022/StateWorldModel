"""游戏客户端风主题:界面用到的常量、HTML 片段构造器、gradio 主题与 CSS。

视觉方向:深色奇幻游戏客户端——顶栏 logo + 菜单式标签、封面卡墙、
"精选游戏"详情面板带大号开始按钮;对局页是带金色角饰的影院画面 +
RPG 指令菜单。

稳健性约定:
- 配色经 gradio 主题变量(GRADIO_THEME_VARS)下发,明/暗两套取同一组值——不依赖浏览器是否
  处于 dark 模式(旧版只覆盖 CSS 变量,浅色模式下成了白底白字);
- 字体只用本地字体栈,不拉 Google Fonts(内网环境会卡住页面加载);
- CSS 只挂在自家 elem_id/elem_classes 与 gradio 稳定的语义钩子上
  (role="tab"、label.selected、.thumbnail-item、.caption-label、
  data-testid="block-info"),任何一条失配都只降级回主题色,不破坏布局。
所有动态文本经 html.escape。本模块不依赖 gradio(gr.themes 对象在 app.py 组装)。
"""

from __future__ import annotations

import html

from engine.ui.i18n import t
from engine.ui.meta import category_label

# 两个"屏幕"的 tab id(gr.Tab(id=...) / gr.Tabs(selected=...) 必须一致)。
TAB_LOBBY = "tab-lobby"
TAB_PLAY = "tab-play"

# 类别 → chip 配色 class 后缀(CSS 里 .chip-<accent> 定义颜色)。
CATEGORY_ACCENT = {"adv": "adv", "machine": "machine", "openworld": "openworld",
                   "romance": "romance"}

# 调色板(gradio 主题变量与 CSS 共用)
BG_0 = "#06070d"
PANEL_SOLID = "#12152a"
GOLD = "#f0c060"
GOLD_DEEP = "#c8902c"
TEXT = "#ece8dc"
TEXT_DIM = "#9a9cb4"

FONT_STACK = ("Noto Sans CJK SC", "PingFang SC", "Hiragino Sans GB",
              "Microsoft YaHei", "Segoe UI", "system-ui", "sans-serif")
MONO_STACK = ("ui-monospace", "Menlo", "Consolas", "monospace")


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def _chip(text: str, cls: str = "") -> str:
    return f'<span class="chip {cls}">{_esc(text)}</span>'


# -- 顶栏 ----------------------------------------------------------------------

def topbar_html(lang: str = "zh") -> str:
    return (
        '<div class="topbar">'
        '<div class="logo-mark">✦</div>'
        '<div class="logo-text">'
        f'<div class="logo-title">{_esc(t(lang, "app_title"))}</div>'
        f'<div class="logo-sub">{_esc(t(lang, "app_sub"))}</div>'
        '</div>'
        '</div>'
    )


def section_title_html(title: str, sub: str = "") -> str:
    sub_html = f'<span class="sec-sub">{_esc(sub)}</span>' if sub else ""
    return f'<div class="sec-title"><span>{_esc(title)}</span>{sub_html}</div>'


# -- 大厅 ----------------------------------------------------------------------

def detail_title_html(title: str, category: str, lang: str = "zh") -> str:
    accent = CATEGORY_ACCENT.get(category, "romance")
    label = category_label(category, lang)
    return (f'<div class="detail-title">'
            f'<div class="detail-kicker">{_chip(label, f"chip-{accent}")}</div>'
            f'<div class="detail-name">{_esc(title)}</div></div>')


def detail_meta_html(language: str, category: str, max_turns: int | None,
                     needs_llm: bool, hint: str, lang: str = "zh") -> str:
    """language:游戏内容语言("中文"/"EN");开放世界标"仅英文",
    中文皮肤的 DUEL 在英文界面经词表出英文,按界面语言标注。"""
    if category == "openworld":
        lang_text = t(lang, "lang_en_only")
    elif category == "machine":
        lang_text = t(lang, "lang_en" if lang == "en" else "lang_zh")
    else:
        lang_text = t(lang, "lang_en" if language == "EN" else "lang_zh")
    turns = t(lang, "turns_n", n=max_turns) if max_turns else t(lang, "turns_any")
    llm = _chip(t(lang, "needs_llm"), "chip-llm") if needs_llm else ""
    hint_line = (f'<div class="detail-hint"><b>{_esc(t(lang, "how_to_play"))}</b>'
                 f'{_esc(hint)}</div>' if hint else "")
    return (
        '<div class="detail-meta">'
        f'{_chip(lang_text, "chip-lang")}'
        f'{_chip(turns, "chip-lang")}{llm}'
        f'</div>{hint_line}'
    )


def cover_placeholder_html(title: str) -> str:
    """封面缺失时的占位卡(纯 CSS,不依赖图片文件)。"""
    return (f'<div class="cover-fallback"><span>{_esc(title)}</span></div>')


# -- 对局 ----------------------------------------------------------------------

def chapter_card_html(card: dict | None, lang: str = "zh") -> str:
    """剧情章节卡(galgame 式):章序 + 节点名 + 事件旁白;card 为空返回 ""。"""
    if not card:
        return ""
    kicker = t(lang, "chapter", i=card.get("index", 1), n=card.get("total", 1))
    text = _esc(card.get("text", "")).replace("\n", "<br>")
    return (
        '<div class="chapter-card">'
        f'<div class="cc-kicker">{_esc(kicker)}</div>'
        f'<div class="cc-title">{_esc(card.get("title", ""))}</div>'
        '<div class="cc-rule"><span>◆</span></div>'
        f'<div class="cc-text">{text}</div>'
        '</div>'
    )


def play_header_html(title: str, category: str, turn: int, max_turns: int,
                     status: str = "ongoing", banner: str = "", game_id: str = "",
                     lang: str = "zh") -> str:
    """对局页 HUD 顶栏:标题 + 类别 + 回合进度条 + 终局/进行中状态。

    game_id 以暗色小字附在标题后(便于排查;ui_smoke 也按它确认开局)。
    """
    accent = CATEGORY_ACCENT.get(category, "romance")
    label = category_label(category, lang)
    if banner:
        status_chip = _chip(banner, f"chip-status-{status}")
        turn_block = ""
    else:
        status_chip = _chip(t(lang, "ongoing"), "chip-status-ongoing")
        if max_turns:
            pct = max(0.0, min(1.0, (turn - 1) / max_turns)) * 100
            turn_block = (
                '<div class="ph-turn">'
                f'<span class="ph-turn-label">TURN <b>{turn}</b>/{max_turns}</span>'
                f'<span class="ph-track"><span class="ph-fill" style="width:{pct:.0f}%"></span></span>'
                '</div>')
        else:
            turn_block = (f'<div class="ph-turn"><span class="ph-turn-label">'
                          f'TURN <b>{turn}</b></span></div>')
    return (
        '<div class="play-header">'
        '<div class="ph-left">'
        f'{_chip(label, f"chip-{accent}")}'
        f'<span class="ph-title">{_esc(title)}</span>'
        + (f'<span class="ph-id">{_esc(game_id)}</span>' if game_id else "")
        + '</div>'
        f'<div class="ph-right">{turn_block}{status_chip}</div>'
        '</div>'
    )


# -- gradio 主题变量 --------------------------------------------------------------
# 由 app.py 套到 gr.themes.Base(...).set() 上(本模块不依赖 gradio);
# 有 _dark 变体的变量两套取同值,主题与浏览器明暗偏好无关。

GRADIO_THEME_VARS = dict(
        body_background_fill=BG_0,
        body_text_color=TEXT,
        body_text_color_subdued=TEXT_DIM,
        background_fill_primary=PANEL_SOLID,
        background_fill_secondary="#0f1224",
        border_color_primary="rgba(255,255,255,.10)",
        border_color_accent=GOLD,
        color_accent=GOLD,
        color_accent_soft="rgba(240,192,96,.16)",
        link_text_color=GOLD,
        block_background_fill="transparent",
        block_border_color="transparent",
        block_border_width="0px",
        block_shadow="none",
        block_label_background_fill="transparent",
        block_label_text_color=TEXT_DIM,
        block_title_text_color=TEXT_DIM,
        block_info_text_color=TEXT_DIM,
        panel_background_fill="transparent",
        panel_border_color="transparent",
        input_background_fill="rgba(255,255,255,.05)",
        input_background_fill_focus="rgba(255,255,255,.08)",
        input_border_color="rgba(255,255,255,.12)",
        input_border_color_focus=GOLD,
        input_placeholder_color="#6e7190",
        input_shadow="none",
        input_shadow_focus="0 0 0 3px rgba(240,192,96,.18)",
        checkbox_background_color="rgba(255,255,255,.06)",
        checkbox_background_color_selected=GOLD,
        checkbox_border_color="rgba(255,255,255,.25)",
        checkbox_border_color_selected=GOLD,
        checkbox_label_background_fill="rgba(255,255,255,.04)",
        checkbox_label_background_fill_hover="rgba(255,255,255,.08)",
        checkbox_label_background_fill_selected="rgba(240,192,96,.14)",
        checkbox_label_border_color="rgba(255,255,255,.10)",
        checkbox_label_border_color_selected=GOLD,
        checkbox_label_text_color=TEXT,
        checkbox_label_text_color_selected=GOLD,
        button_primary_background_fill=f"linear-gradient(180deg,#ffd77a 0%,{GOLD} 45%,{GOLD_DEEP} 100%)",
        button_primary_background_fill_hover=f"linear-gradient(180deg,#ffe39c 0%,#f6cb72 45%,{GOLD} 100%)",
        button_primary_text_color="#2a1a02",
        button_primary_border_color="#ffe1a0",
        button_secondary_background_fill="rgba(255,255,255,.05)",
        button_secondary_background_fill_hover="rgba(255,255,255,.10)",
        button_secondary_border_color="rgba(255,255,255,.16)",
        button_secondary_text_color=TEXT,
        button_secondary_text_color_hover=GOLD,
        table_even_background_fill="rgba(255,255,255,.03)",
        table_odd_background_fill="transparent",
        code_background_fill="rgba(255,255,255,.06)",
        shadow_drop="none",
        shadow_drop_lg="none",
)


# 页面加载即切到 dark(gradio 少数组件只认 .dark 类);主题本身已明暗一致。
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


# -- 自定义样式 ------------------------------------------------------------------

CUSTOM_CSS = """
/* ── 全局:深空背景 + 星点 ───────────────────────────────────── */
body, .gradio-container {
  background:
    radial-gradient(1200px 600px at 15% -10%, rgba(88,96,255,.20), transparent 60%),
    radial-gradient(900px 520px at 105% 0%, rgba(240,160,64,.14), transparent 55%),
    radial-gradient(1000px 700px at 50% 120%, rgba(150,60,200,.14), transparent 60%),
    linear-gradient(180deg, #0d1020 0%, #06070d 100%) !important;
  background-attachment:fixed !important;
}
.gradio-container {max-width:1480px !important; padding-top:10px !important;}
.gradio-container::before {
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.55;
  background-image:
    radial-gradient(1px 1px at 12% 22%, #fff8 50%, transparent 51%),
    radial-gradient(1px 1px at 72% 12%, #fff6 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 42% 68%, #ffe7a855 50%, transparent 51%),
    radial-gradient(1px 1px at 88% 58%, #fff5 50%, transparent 51%),
    radial-gradient(1px 1px at 26% 86%, #fff4 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 60% 36%, #c8d0ff44 50%, transparent 51%);
}
footer {display:none !important;}
.gradio-container .prose, .gradio-container .prose * {color:inherit;}

/* ── 顶栏 ─────────────────────────────────────────────────────── */
.topbar {
  display:flex; align-items:center; gap:14px; padding:6px 4px 2px;
}
.logo-mark {
  width:46px; height:46px; border-radius:12px; flex:none;
  display:flex; align-items:center; justify-content:center;
  font-size:24px; color:#2a1a02;
  background:linear-gradient(145deg,#ffe39c,#f0c060 50%,#b07a20);
  box-shadow:0 0 0 1px #ffe1a066, 0 6px 22px rgba(240,192,96,.45);
}
.logo-title {
  font-size:1.55em; font-weight:900; letter-spacing:.12em; line-height:1.15;
  background:linear-gradient(180deg,#fff3cf,#f0c060 60%,#c8902c);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  text-shadow:0 0 24px rgba(240,192,96,.18);
}
.logo-sub {font-size:.72em; letter-spacing:.22em; color:#8d90aa; margin-top:2px;}

/* 语言切换(顶栏右侧) */
.topbar-row {align-items:center !important;}
#lang-switch {flex:none !important; min-width:0 !important; margin-left:auto;}
#lang-switch .wrap {gap:0 !important; border:1px solid rgba(240,192,96,.35); border-radius:999px; padding:3px; background:rgba(0,0,0,.25);}
#lang-switch label {padding:5px 16px !important; border:none !important; font-weight:800; letter-spacing:.08em;}

/* ── 菜单式标签页 ─────────────────────────────────────────────── */
#cin-tabs {background:transparent; border:none;}
#cin-tabs [role="tablist"] {
  gap:8px; border-bottom:1px solid rgba(240,192,96,.16) !important;
  padding-bottom:10px; margin-bottom:14px;
}
#cin-tabs [role="tab"] {
  color:#9a9cb4 !important; font-weight:700; letter-spacing:.14em;
  padding:9px 22px !important; border-radius:10px !important;
  border:1px solid rgba(255,255,255,.08) !important;
  background:rgba(255,255,255,.03) !important;
  transition:all .18s ease;
}
#cin-tabs [role="tab"]:hover {color:#ece8dc !important; border-color:rgba(240,192,96,.35) !important;}
#cin-tabs [role="tab"].selected, #cin-tabs [role="tab"][aria-selected="true"] {
  color:#2a1a02 !important; border-color:#ffe1a0 !important;
  background:linear-gradient(180deg,#ffd77a,#f0c060 50%,#c8902c) !important;
  box-shadow:0 4px 18px rgba(240,192,96,.35);
}
#cin-tabs [role="tab"].selected::after {display:none !important;}
#cin-tabs > div[role="tabpanel"], #cin-tabs .tabitem {
  background:transparent !important; border:none !important; padding:0 !important;
}

/* ── 面板(游戏 UI 窗口) ───────────────────────────────────────── */
.glass {
  position:relative;
  background:linear-gradient(180deg, rgba(22,26,48,.86), rgba(12,14,28,.90)) !important;
  border:1px solid rgba(240,192,96,.20) !important; border-radius:18px !important;
  box-shadow:0 18px 50px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.06);
  padding:16px !important;
  backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
}
.glass::before {
  content:""; position:absolute; left:18px; right:18px; top:0; height:1px;
  background:linear-gradient(90deg, transparent, rgba(255,226,160,.7), transparent);
}
.sec-title {
  display:flex; align-items:baseline; gap:10px; margin:2px 2px 4px;
  font-weight:800; letter-spacing:.14em; color:#ece8dc; font-size:1.02em;
}
.sec-title > span:first-child::before {
  content:"◆"; color:#f0c060; margin-right:8px; font-size:.8em;
  text-shadow:0 0 10px rgba(240,192,96,.8);
}
.sec-sub {color:#7f829c; font-weight:500; letter-spacing:.06em; font-size:.82em;}

/* gradio 表单容器默认拿边框色当底(输入框之间的分隔线),深色主题下去掉 */
.gradio-container .form {background:transparent !important; border:none !important; gap:12px !important;}

/* 表单标签 */
.gradio-container [data-testid="block-info"] {
  color:#a7a9c2 !important; font-weight:700; letter-spacing:.08em; font-size:.82em;
}

/* ── 胶囊单选(大厅筛选 / 渲染后端) ───────────────────────────── */
.pill-radio, .action-radio {background:transparent !important; border:none !important; padding:0 !important; box-shadow:none !important;}
.pill-radio .wrap {display:flex; gap:8px; flex-wrap:wrap;}
.pill-radio label {
  border:1px solid rgba(255,255,255,.12) !important; border-radius:999px !important;
  background:rgba(255,255,255,.04) !important; padding:7px 16px !important;
  color:#c9cbe0 !important; font-weight:600; cursor:pointer;
  transition:all .15s ease; margin:0 !important; box-shadow:none !important;
}
.pill-radio label input {display:none !important;}
.pill-radio label:hover {border-color:rgba(240,192,96,.5) !important; color:#fff !important;}
.pill-radio label.selected {
  border-color:#f0c060 !important; color:#2a1a02 !important;
  background:linear-gradient(180deg,#ffd77a,#f0c060) !important;
  box-shadow:0 0 16px rgba(240,192,96,.35) !important;
}
.pill-radio label.selected span {color:#2a1a02 !important;}

/* ── 封面卡墙(gr.Gallery) ────────────────────────────────────── */
#lobby-gallery, #lobby-gallery .grid-wrap {background:transparent !important; border:none !important;}
#lobby-gallery .grid-wrap {padding:4px 6px 4px 2px !important;}
#lobby-gallery .grid-container {gap:14px !important;}
#lobby-gallery .thumbnail-item {
  position:relative; border-radius:14px !important; overflow:hidden;
  border:1px solid rgba(255,255,255,.10) !important;
  box-shadow:0 8px 22px rgba(0,0,0,.5) !important;
  transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
  aspect-ratio:16/10;
}
#lobby-gallery .thumbnail-item img {
  object-fit:cover !important; width:100%; height:100%;
  transition:transform .35s ease, filter .25s ease; filter:saturate(.92) brightness(.92);
}
#lobby-gallery .thumbnail-item:hover {
  transform:translateY(-4px);
  border-color:rgba(240,192,96,.65) !important;
  box-shadow:0 14px 34px rgba(0,0,0,.6), 0 0 0 1px rgba(240,192,96,.35), 0 0 26px rgba(240,192,96,.25) !important;
}
#lobby-gallery .thumbnail-item:hover img {transform:scale(1.06); filter:saturate(1.08) brightness(1.02);}
#lobby-gallery .thumbnail-item.selected {
  border-color:#f0c060 !important;
  box-shadow:0 0 0 2px #f0c060, 0 0 30px rgba(240,192,96,.45) !important;
}
#lobby-gallery .caption-label {
  position:absolute; left:0; right:0; bottom:0; margin:0 !important; max-width:none !important;
  padding:26px 12px 9px !important; border-radius:0 !important; border:none !important;
  background:linear-gradient(180deg, transparent, rgba(6,7,13,.92) 70%) !important;
  color:#f3efe2 !important; font-weight:700; font-size:.86em !important;
  letter-spacing:.03em; text-align:left; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; text-shadow:0 1px 3px #000;
}
#lobby-gallery .thumbnail-item:hover .caption-label {opacity:1 !important;}

/* ── 详情面板(精选游戏) ────────────────────────────────────────── */
.detail-panel {gap:10px !important;}
.detail-art {position:relative;}
.detail-art img {
  border-radius:14px !important; object-fit:cover !important; width:100%;
  border:1px solid rgba(240,192,96,.25);
  box-shadow:0 14px 40px rgba(0,0,0,.6);
  -webkit-mask-image:linear-gradient(180deg,#000 72%,transparent 100%);
          mask-image:linear-gradient(180deg,#000 72%,transparent 100%);
}
.detail-art .icon-button-wrapper {display:none !important;}
.detail-art button {cursor:default;}
.detail-title {margin:-34px 4px 0; position:relative; z-index:2;}
.detail-kicker {margin-bottom:6px;}
.detail-name {
  font-size:1.65em; font-weight:900; letter-spacing:.06em; line-height:1.2;
  color:#fff7e2; text-shadow:0 2px 14px rgba(0,0,0,.9), 0 0 22px rgba(240,192,96,.25);
}
.detail-blurb {color:#c3c5da !important; line-height:1.7; margin:0 4px !important;}
.detail-blurb p {margin:0 !important;}
.detail-meta {display:flex; gap:6px; flex-wrap:wrap; margin:0 4px;}
.detail-hint {
  color:#a9abc4; font-size:.86em; line-height:1.6; margin:10px 4px 0;
  padding:10px 12px; border-radius:10px;
  background:rgba(255,255,255,.035); border-left:3px solid #f0c060;
}
.detail-hint b {color:#f0c060; margin-right:8px; letter-spacing:.1em;}
.cover-fallback {
  height:120px; border-radius:14px; border:1px dashed rgba(255,255,255,.18);
  display:flex; align-items:center; justify-content:center; color:#9a9cb4;
}

/* ── 徽章 ─────────────────────────────────────────────────────── */
.chip {
  display:inline-block; padding:3px 11px; border-radius:6px;
  border:1px solid rgba(255,255,255,.18); font-size:.78em; font-weight:700;
  letter-spacing:.06em; color:#b8bad0; white-space:nowrap;
  background:rgba(255,255,255,.04);
}
.chip-adv      {color:#ff8fab; border-color:rgba(255,143,171,.5); background:rgba(255,143,171,.12);}
.chip-machine  {color:#ffcf6e; border-color:rgba(255,207,110,.5); background:rgba(255,207,110,.12);}
.chip-openworld{color:#6fe3d8; border-color:rgba(111,227,216,.5); background:rgba(111,227,216,.12);}
.chip-romance  {color:#c4a0ff; border-color:rgba(196,160,255,.5); background:rgba(196,160,255,.12);}
.chip-llm      {color:#8fb8ff; border-color:rgba(143,184,255,.45); background:rgba(143,184,255,.10);}
.chip-status-ongoing {color:#7ee0a0; border-color:rgba(126,224,160,.45); background:rgba(126,224,160,.10);}
.chip-status-won  {color:#ffe08a; border-color:#ffe08a; background:rgba(255,224,138,.16); box-shadow:0 0 16px rgba(255,224,138,.35);}
.chip-status-lost {color:#ff8a8a; border-color:rgba(255,138,138,.6); background:rgba(255,138,138,.12);}
.chip-status-draw {color:#d9d2a6; border-color:rgba(217,210,166,.5); background:rgba(217,210,166,.10);}

/* ── 按钮 ─────────────────────────────────────────────────────── */
.btn-gold, .btn-start {
  font-weight:900 !important; letter-spacing:.2em !important; font-size:1.05em !important;
  border:1px solid #ffe1a0 !important; border-radius:12px !important;
  text-shadow:0 1px 0 rgba(255,255,255,.4);
  box-shadow:0 8px 26px rgba(240,192,96,.35), inset 0 1px 0 rgba(255,255,255,.55), inset 0 -3px 0 rgba(120,70,0,.35) !important;
  transition:transform .15s ease, box-shadow .15s ease, filter .15s ease !important;
}
.btn-start {min-height:58px !important; font-size:1.25em !important;}
.btn-gold:hover, .btn-start:hover {
  transform:translateY(-2px);
  box-shadow:0 12px 34px rgba(240,192,96,.5), inset 0 1px 0 rgba(255,255,255,.6), inset 0 -3px 0 rgba(120,70,0,.35) !important;
}
.btn-gold:active, .btn-start:active {transform:translateY(1px);}
.btn-gold:disabled, .btn-start:disabled {filter:grayscale(.8) brightness(.6); box-shadow:none !important; transform:none;}
.btn-start {animation:startPulse 2.6s ease-in-out infinite;}
@keyframes startPulse {
  0%,100% {box-shadow:0 8px 26px rgba(240,192,96,.30), inset 0 1px 0 rgba(255,255,255,.55), inset 0 -3px 0 rgba(120,70,0,.35);}
  50%     {box-shadow:0 8px 40px rgba(240,192,96,.62), inset 0 1px 0 rgba(255,255,255,.55), inset 0 -3px 0 rgba(120,70,0,.35);}
}
.btn-ghost {
  background:rgba(255,255,255,.03) !important; border:1px solid rgba(255,255,255,.14) !important;
  color:#c9cbe0 !important; font-weight:700 !important; letter-spacing:.1em;
}
.btn-ghost:hover {border-color:rgba(240,192,96,.55) !important; color:#f0c060 !important;}

/* ── 手风琴(高级设置 / 战况记录) ──────────────────────────────── */
.game-acc {
  background:rgba(255,255,255,.025) !important; border:1px solid rgba(255,255,255,.08) !important;
  border-radius:12px !important; padding:4px 10px !important;
}
.game-acc .label-wrap, .game-acc button.label-wrap {color:#b8bad0 !important; font-weight:700; letter-spacing:.1em;}
.game-acc .label-wrap:hover {color:#f0c060 !important;}

/* ── 对局:HUD 顶栏 ───────────────────────────────────────────── */
.play-header {
  display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;
  padding:12px 16px; border-radius:14px;
  background:linear-gradient(90deg, rgba(240,192,96,.10), rgba(255,255,255,.02) 50%, rgba(240,192,96,.06));
  border:1px solid rgba(240,192,96,.22);
}
.ph-left, .ph-right {display:flex; align-items:center; gap:12px; flex-wrap:wrap;}
.play-header .ph-title {font-weight:900; font-size:1.2em; letter-spacing:.06em; color:#fff7e2;}
.play-header .ph-id {color:#5f627c; font-size:.74em; font-family:ui-monospace,Menlo,Consolas,monospace;}
.ph-turn {display:flex; align-items:center; gap:10px;}
.ph-turn-label {color:#9a9cb4; font-size:.8em; letter-spacing:.2em; font-weight:700;}
.ph-turn-label b {color:#f0c060; font-size:1.35em;}
.ph-track {
  display:inline-block; width:120px; height:7px; border-radius:4px;
  background:rgba(255,255,255,.08); overflow:hidden; border:1px solid rgba(255,255,255,.08);
}
.ph-fill {display:block; height:100%; background:linear-gradient(90deg,#c8902c,#ffd77a);
          box-shadow:0 0 10px rgba(240,192,96,.7);}

/* ── 对局:影院画面 ───────────────────────────────────────────── */
.stage-frame {
  position:relative; padding:10px !important; border-radius:18px !important;
  background:radial-gradient(ellipse at center, #0b0d18 0%, #04050a 100%) !important;
  border:1px solid rgba(240,192,96,.30) !important;
  box-shadow:0 22px 60px rgba(0,0,0,.7), inset 0 0 40px rgba(0,0,0,.8);
}
.stage-frame::before, .stage-frame::after {
  content:""; position:absolute; width:26px; height:26px; pointer-events:none; z-index:5;
  border-color:#f0c060; border-style:solid;
}
.stage-frame::before {top:-1px; left:-1px; border-width:2px 0 0 2px; border-radius:18px 0 0 0;}
.stage-frame::after  {bottom:-1px; right:-1px; border-width:0 2px 2px 0; border-radius:0 0 18px 0;}
#stage-video, .montage-video {background:transparent !important; border:none !important;}
#stage-video video, .montage-video video {border-radius:12px; background:#000;}
#stage-video [data-testid="block-label"], .montage-video [data-testid="block-label"] {
  background:rgba(6,7,13,.75) !important; color:#f0c060 !important; border:1px solid rgba(240,192,96,.35) !important;
  border-radius:8px !important; letter-spacing:.12em; font-weight:700;
}

/* ── 对局:剧情章节卡(ADV 节点事件) ─────────────────────────────── */
.chapter-card {
  position:relative; padding:18px 20px 16px; border-radius:14px; text-align:center;
  background:
    radial-gradient(ellipse at 50% 0%, rgba(240,192,96,.16), transparent 70%),
    linear-gradient(180deg, rgba(30,24,44,.95), rgba(14,12,26,.95));
  border:1px solid rgba(240,192,96,.45);
  box-shadow:0 0 0 1px rgba(240,192,96,.12) inset, 0 10px 30px rgba(0,0,0,.5), 0 0 28px rgba(240,192,96,.15);
  animation:chapterIn .7s ease-out both;
}
@keyframes chapterIn {from {opacity:0; transform:translateY(-6px);} to {opacity:1; transform:none;}}
.cc-kicker {color:#c8a35a; font-size:.72em; letter-spacing:.32em; font-weight:800;}
.cc-title {
  margin:6px 0 4px; font-size:1.35em; font-weight:900; letter-spacing:.12em;
  font-family:"Noto Serif CJK SC","Songti SC","Source Han Serif SC",Georgia,serif;
  background:linear-gradient(180deg,#fff3cf,#f0c060 65%,#c8902c);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.cc-rule {display:flex; align-items:center; gap:10px; margin:6px auto 10px; max-width:70%;
          color:#f0c060; font-size:.7em;}
.cc-rule::before, .cc-rule::after {content:""; flex:1; height:1px;
  background:linear-gradient(90deg, transparent, rgba(240,192,96,.7), transparent);}
.cc-text {
  color:#e6e1d2; line-height:1.85; font-size:.95em; text-align:left;
  font-family:"Noto Serif CJK SC","Songti SC","Source Han Serif SC",Georgia,serif;
}

/* ── 对局:指令面板 ───────────────────────────────────────────── */
.control-panel {gap:12px !important;}
.block.status-box {
  padding:12px 14px !important; border-radius:12px !important;
  background:linear-gradient(180deg, rgba(240,192,96,.07), rgba(255,255,255,.02)) !important;
  border:1px solid rgba(240,192,96,.20) !important;
  color:#dcdaf0 !important; line-height:1.7; font-size:.94em;
}
.prose.status-box {border:none !important; background:none !important; padding:0 !important;}
.status-box blockquote {
  border-left:3px solid #ff8a8a !important; color:#ffb4b4 !important;
  background:rgba(255,138,138,.08); padding:6px 10px !important; margin:8px 0 0 !important; border-radius:6px;
}
.action-radio .wrap {display:flex !important; flex-direction:column !important; gap:8px !important;}
.action-radio label {
  width:100% !important; justify-content:flex-start !important;
  border:1px solid rgba(255,255,255,.10) !important; border-radius:10px !important;
  background:linear-gradient(90deg, rgba(255,255,255,.05), rgba(255,255,255,.015)) !important;
  padding:11px 14px 11px 34px !important; position:relative; cursor:pointer;
  color:#dcdaf0 !important; font-weight:700; letter-spacing:.04em;
  transition:all .15s ease; box-shadow:none !important; margin:0 !important;
}
.action-radio label input {display:none !important;}
.action-radio label::before {
  content:"◇"; position:absolute; left:13px; top:50%; transform:translateY(-50%);
  color:#6e7190; font-size:.9em; transition:all .15s ease;
}
.action-radio label:hover {
  border-color:rgba(240,192,96,.55) !important; transform:translateX(3px);
  background:linear-gradient(90deg, rgba(240,192,96,.12), rgba(255,255,255,.02)) !important;
}
.action-radio label:hover::before {color:#f0c060;}
.action-radio label.selected {
  border-color:#f0c060 !important; color:#fff3cf !important;
  background:linear-gradient(90deg, rgba(240,192,96,.26), rgba(240,192,96,.04)) !important;
  box-shadow:0 0 0 1px rgba(240,192,96,.4), 0 0 20px rgba(240,192,96,.2) !important;
}
.action-radio label.selected::before {content:"▶"; color:#f0c060; text-shadow:0 0 8px #f0c060;}
.action-radio label.selected span {color:#fff3cf !important;}
.action-info {color:#a9abc4 !important; font-size:.88em; line-height:1.65;}
.action-info strong {color:#f0c060 !important;}
.free-text textarea, .free-text input {
  font-size:1em !important; border-radius:10px !important;
  background:rgba(0,0,0,.35) !important;
}

/* ── 滚动条 ──────────────────────────────────────────────────── */
.gradio-container ::-webkit-scrollbar {width:8px; height:8px;}
.gradio-container ::-webkit-scrollbar-thumb {background:rgba(240,192,96,.35); border-radius:4px;}
.gradio-container ::-webkit-scrollbar-track {background:transparent;}

/* 加载遮罩:深色化,避免白色闪屏 */
.gradio-container .wrap.default.full {background:rgba(6,7,13,.55) !important;}
.gradio-container .eta-bar {background:rgba(240,192,96,.18) !important;}
"""
