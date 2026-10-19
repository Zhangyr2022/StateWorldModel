"""Gradio 界面:游戏客户端风的玩家操作台——封面卡墙大厅 + 对局两屏。

通用性边界(与 run_demo 同一契约,与旧版界面一致):
- 游戏列表来自 game.registry(封面墙自动枚举,新游戏零改动);
- 界面只面向 GameParts + InteractiveGame(engine/ui/session.py)工作,
  不 import 任何具体游戏,也不知道任何游戏规则;
- 选择策略 = 玩家本人:点选 available_actions 里的行为,自由输入类玩法
  (adv"说话"、开放世界"自定义行动")用文本框,不再有自动代理。

大厅 → 对局的呈现分两页(gr.Tabs):大厅用皮肤 frame 封面做卡片墙
(engine/ui/meta.py 提供展示元数据),开局成功经 start_btn.success 切到
对局页;开局失败(gr.Error)自动留在大厅。渲染仍在 GameLoop 内部进行:
结算永远先于渲染,渲染失败不影响游戏状态。
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from pathlib import Path

import gradio as gr

from engine.config import load_config
from engine.llm.client import LLMClient
from engine.llm.service import default_llm_socket
from engine.montage import build_summary_montage
from engine.parts import resolve_frame
from engine.renderer.factory import build_renderer
from engine.renderer.fake import _make_static_video
from engine.renderer.frames import create_placeholder_frame
from engine.runtime import Action
from engine.hud_i18n import hud_lang, tr
from engine.ui import i18n, theme
from engine.ui import meta as meta_mod
from engine.ui.hud_clip import compose_hud_clip
from engine.ui.session import GameFinishedError, InteractiveGame

logger = logging.getLogger("engine.ui")

# 渲染后端选项:取值 = 中文原文(脚本 / gradio_client 按值调用);英文界面只换展示
BACKEND_CHOICES = (
    "fake · 离线占位(无需 GPU)",
    "h3 remote · 常驻渲染服务",
    "h3 session · 临时渲染会话",
)
BACKEND_MAP = {
    BACKEND_CHOICES[0]: ("fake", None),
    BACKEND_CHOICES[1]: ("h3", "remote"),
    BACKEND_CHOICES[2]: ("h3", "session"),
}


class UiState:
    """一个浏览器会话的界面状态:界面语言 + 游戏 id + 交互式驱动器。

    lang 随顶栏语言开关实时变化:界面文案、游戏文本显示与回放 HUD 都跟随它
    (开放世界只支持英文,HUD 恒为英文,见 hud_lang)。
    """

    def __init__(self, lang: str = i18n.DEFAULT_LANG) -> None:
        self.lang = lang
        self.game_id = ""
        self.category = ""
        self.scene: str | None = None  # 场景类游戏开局所选场景
        self.session: InteractiveGame | None = None

    @property
    def hud_lang(self) -> str:
        return "en" if self.category == "openworld" else self.lang

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


def build_app(config_path=None, *, llm_socket: Path | None = None,
              renderer_socket: Path | None = None,
              renderer_backend: str | None = None,
              renderer_mode: str | None = None) -> gr.Blocks:
    """组装 Gradio 界面;socket 覆盖项缺省取 config / 服务默认路径。

    renderer_backend / renderer_mode:命令行覆盖(run_ui --renderer/--mode),
    决定大厅"渲染后端"的默认选项;缺省取 config。
    """
    from engine.ui.meta import FILTER_CHOICES, gallery_items as _gallery_items
    from engine.ui.meta import load_game_meta as _load_metas

    cfg = load_config(config_path)
    if renderer_backend:
        cfg.renderer.backend = renderer_backend
    if renderer_mode:
        cfg.renderer.mode = renderer_mode
    llm_socket = llm_socket or default_llm_socket()
    renderer_socket = renderer_socket or (
        Path(cfg.renderer.socket_path) if cfg.renderer.socket_path else None)
    output_root = cfg.loop.output_dir  # 每局一个 ui-run-* 目录,与 run-* 同构
    default_backend = (
        BACKEND_CHOICES[0] if cfg.renderer.backend == "fake"
        else (BACKEND_CHOICES[1] if cfg.renderer.mode == "remote" else BACKEND_CHOICES[2])
    )
    L0 = i18n.DEFAULT_LANG  # 页面初始语言

    # 大厅展示元数据(封面/简介/语言,来自皮肤 frame 等只读字段)。
    metas = _load_metas()
    all_images, all_ids = _gallery_items(metas, "全部", L0)
    first_id = all_ids[0]["game_id"] if all_ids else ""
    first_scene = all_ids[0].get("scene") if all_ids else None

    def _new_output_dir() -> str:
        return f"{output_root}/ui-run-{time.strftime('%Y%m%d-%H%M%S')}"

    def _filter_choices(lang: str) -> list[tuple[str, str]]:
        return [(meta_mod.filter_label(key, lang), key) for key in FILTER_CHOICES]

    def _option_choices(option, lang: str) -> list[tuple[str, str]]:
        """开场场景下拉:(场景展示名, 场景 id)。"""
        return [(meta_mod.scene_title(str(c), lang), c) for c in option.choices]

    def _cover_for(game_id: str, scene: str | None) -> str | None:
        """详情大图:场景类游戏用所选场景的画面(未指定场景取默认场景)。"""
        meta = metas.get(game_id)
        if meta is None:
            return None
        if meta.scene_choices:
            entry = _entry(game_id)
            choice = scene or (entry.options[0].default if entry.options else None)
            cover = meta_mod.scene_cover(str(choice)) if choice else None
            if cover is not None:
                return str(cover)
        return str(meta.cover) if meta.cover else None

    def _show_detail(game_id: str, scene: str | None = None, lang: str = L0):
        """选中一个游戏 → 详情面板更新(卡片点选与下拉选择共用)。

        scene 非空(场景类卡片点选)时把开场选项预置为该场景,下拉仍可改。
        返回 (封面, 标题, 简介, 元信息徽章, 开场选项下拉) 的组件更新。
        """
        meta = metas.get(game_id)
        if meta is None:
            return (gr.update(value=None), "", "", "", gr.update(visible=False))
        entry = _entry(game_id)
        option_update = gr.update(visible=False)
        if entry.options:
            option = entry.options[0]
            scene = scene or option.default
            option_update = gr.update(visible=True,
                                      label=i18n.option_label(option.label, lang),
                                      choices=_option_choices(option, lang),
                                      value=scene)
        return (gr.update(value=_cover_for(game_id, scene)),
                theme.detail_title_html(meta_mod.display_title(meta, lang, scene),
                                        meta.category, lang),
                meta_mod.display_blurb(meta, lang) or i18n.t(lang, "no_blurb"),
                theme.detail_meta_html(meta.language, meta.category, meta.max_turns,
                                       meta.needs_llm, i18n.hint(entry.category, lang),
                                       lang),
                option_update)

    first_detail = _show_detail(first_id, first_scene, L0)

    # ------------------------------------------------------------------ 界面

    with gr.Blocks(title=i18n.t(L0, "app_title"),
                   css=theme.CUSTOM_CSS,
                   theme=_build_theme(),
                   js=theme.FORCE_DARK_JS) as demo:
        with gr.Row(elem_classes=["topbar-row"]):
            topbar_md = gr.HTML(theme.topbar_html(L0))
            lang_radio = gr.Radio(choices=list(i18n.LANG_CHOICES), value=L0,
                                  show_label=False, interactive=True, scale=0,
                                  min_width=170, elem_id="lang-switch",
                                  elem_classes=["pill-radio"])
        ui_state = gr.State(UiState(L0))
        selected_game_id = gr.State(first_id)
        lobby_ids = gr.State(all_ids)

        # 不上屏的游戏选择器:只为脚本/gradio_client 保留 /on_pick_dropdown 端点
        # (gr.State 不进 API 面,scripts/ui_smoke.py 靠它选游戏);玩家用封面墙选。
        game_dd = gr.Dropdown(choices=[(gid, gid) for gid in metas], value=first_id,
                              interactive=True, visible=False)

        with gr.Tabs(selected=theme.TAB_LOBBY, elem_id="cin-tabs") as tabs:
            # ============================ 大厅 ============================
            with gr.Tab(i18n.t(L0, "tab_lobby"), id=theme.TAB_LOBBY) as lobby_tab:
                with gr.Row(equal_height=False):
                    # 左:封面卡墙
                    with gr.Column(scale=3, elem_classes=["glass", "lobby-left"]):
                        pick_title_md = gr.HTML(theme.section_title_html(
                            i18n.t(L0, "sec_pick"), i18n.t(L0, "sec_pick_sub")))
                        filter_radio = gr.Radio(
                            choices=_filter_choices(L0), value="全部",
                            show_label=False, interactive=True,
                            elem_id="lobby-filter", elem_classes=["pill-radio"])
                        gallery = gr.Gallery(
                            value=all_images, columns=3, height=640,
                            object_fit="cover", allow_preview=False,
                            show_label=False, interactive=False,
                            elem_id="lobby-gallery")
                    # 右:精选游戏详情 + 开始按钮 + 高级设置
                    with gr.Column(scale=2, elem_classes=["glass", "detail-panel"]):
                        detail_art_img = gr.Image(
                            value=_cover_for(first_id, first_scene),
                            show_label=False, interactive=False, height=300,
                            show_download_button=False, show_fullscreen_button=False,
                            elem_classes=["detail-art"])
                        detail_title_md = gr.HTML(first_detail[1])
                        detail_blurb_md = gr.Markdown(first_detail[2],
                                                      elem_classes=["detail-blurb"])
                        detail_meta_md = gr.HTML(first_detail[3])
                        first_entry = _entry(first_id) if first_id else None
                        option_dd = gr.Dropdown(
                            label=(i18n.option_label(first_entry.options[0].label, L0)
                                   if first_entry and first_entry.options
                                   else i18n.t(L0, "scene")),
                            choices=(_option_choices(first_entry.options[0], L0)
                                     if first_entry and first_entry.options else []),
                            value=first_scene, visible=first_scene is not None,
                            interactive=True)
                        start_btn = gr.Button(i18n.t(L0, "start"), variant="primary",
                                              size="lg", elem_classes=["btn-start"])
                        with gr.Accordion(i18n.t(L0, "advanced"), open=False,
                                          elem_classes=["game-acc"]) as adv_acc:
                            backend_radio = gr.Radio(
                                choices=i18n.backend_choices(BACKEND_CHOICES, L0),
                                value=default_backend,
                                label=i18n.t(L0, "backend"),
                                elem_classes=["pill-radio"])
                            with gr.Row():
                                seed_num = gr.Number(value=cfg.loop.base_seed, precision=0,
                                                     label=i18n.t(L0, "seed"))
                                turns_num = gr.Number(value=cfg.loop.max_turns, precision=0,
                                                      label=i18n.t(L0, "max_turns"))

            # ============================ 对局 ============================
            with gr.Tab(i18n.t(L0, "tab_play"), id=theme.TAB_PLAY) as play_tab:
                play_header_md = gr.HTML("")
                with gr.Row(equal_height=False):
                    # 左:影院画面 + 回放
                    with gr.Column(scale=3):
                        with gr.Group(elem_classes=["stage-frame"]):
                            video = gr.Video(label=i18n.t(L0, "turn_video"), autoplay=True,
                                             height=440, elem_id="stage-video")
                        montage_btn = gr.Button(i18n.t(L0, "montage_btn"), visible=False,
                                                elem_classes=["btn-ghost"])
                        montage_video = gr.Video(label=i18n.t(L0, "montage_video"),
                                                 visible=False, height=400,
                                                 elem_classes=["montage-video"])
                    # 右:RPG 指令面板
                    with gr.Column(scale=2, elem_classes=["glass", "control-panel"]):
                        # 剧情章节卡:开局与进入新节点前出现(运行时提供 chapter_card 时)
                        chapter_md = gr.HTML("", visible=False)
                        status_title_md = gr.HTML(
                            theme.section_title_html(i18n.t(L0, "sec_status")))
                        status_md = gr.Markdown(i18n.t(L0, "idle_status"),
                                                elem_classes=["status-box"])
                        actions_title_md = gr.HTML(
                            theme.section_title_html(i18n.t(L0, "sec_actions")))
                        action_radio = gr.Radio(label=i18n.t(L0, "action_label"),
                                                show_label=False, choices=[],
                                                visible=False, interactive=True,
                                                elem_classes=["action-radio"])
                        action_info_md = gr.Markdown("", elem_classes=["action-info"])
                        free_text = gr.Textbox(label="", visible=False, placeholder="",
                                               interactive=True, lines=2,
                                               elem_classes=["free-text"])
                        step_btn = gr.Button(i18n.t(L0, "step"), variant="primary",
                                             size="lg", interactive=False,
                                             elem_classes=["btn-gold"])
                        # Debug 面板:运行时提供 boss_action_menu(machine 主题)
                        # 时出现,可强制指定本回合 Boss 行动(一次性,用后即清)。
                        debug_cb = gr.Checkbox(label=i18n.t(L0, "debug"),
                                               value=False, visible=False)
                        boss_radio = gr.Radio(label=i18n.t(L0, "boss_pick"), choices=[],
                                              visible=False, interactive=True,
                                              elem_classes=["pill-radio"])
                        boss_info_md = gr.Markdown("")
                        with gr.Accordion(i18n.t(L0, "log"), open=False,
                                          elem_classes=["game-acc"]) as log_acc:
                            log_md = gr.Markdown("")
                        back_btn = gr.Button(i18n.t(L0, "back"), elem_classes=["btn-ghost"])

        # ---------------------------------------------------------- 回调

        def on_filter(filter_key: str, ui: UiState):
            """切换筛选:重建封面墙,并自动选中筛选结果的第一个游戏(下拉同步)。"""
            images, ids = _gallery_items(metas, filter_key, ui.lang)
            first = ids[0] if ids else {"game_id": "", "scene": None}
            detail = _show_detail(first["game_id"], first.get("scene"), ui.lang)
            return (gr.update(value=images), ids, first["game_id"], *detail)

        def on_gallery_select(evt: gr.SelectData, ids: list[dict], ui: UiState):
            """点选封面卡:更新详情面板与已选游戏(下标越界静默忽略)。"""
            entry = (ids[evt.index]
                     if evt is not None and isinstance(getattr(evt, "index", None), int)
                     and 0 <= evt.index < len(ids) and isinstance(ids[evt.index], dict)
                     else {"game_id": "", "scene": None})
            game_id, scene = entry["game_id"], entry.get("scene")
            detail = _show_detail(game_id, scene, ui.lang)
            return (game_id, *detail)

        def on_pick_dropdown(game_id: str, ui: UiState):
            """从下拉选择游戏:与卡片点选同一详情逻辑(不写回下拉自身)。"""
            detail = _show_detail(game_id or "", None, ui.lang)
            return (game_id or "", *detail)

        def on_scene(scene: str, game_id: str, ids: list[dict], ui: UiState):
            """玩家改开场场景:详情大图、标题与封面墙高亮跟着换成该场景。"""
            meta = metas.get(game_id)
            title = (theme.detail_title_html(meta_mod.display_title(meta, ui.lang, scene),
                                             meta.category, ui.lang)
                     if meta else gr.update())
            index = next((i for i, e in enumerate(ids or [])
                          if e.get("game_id") == game_id and e.get("scene") == scene), None)
            highlight = gr.update(selected_index=index) if index is not None else gr.update()
            return gr.update(value=_cover_for(game_id, scene)), title, highlight

        def on_lang(lang: str, filter_key: str, game_id: str, scene: str, ui: UiState):
            """切换界面语言:重排大厅(ADV 换成该语言版本)+ 全部静态文案;
            有进行中的对局时按新语言重绘对局面板(含 HUD 片段,按语言缓存)。"""
            lang = i18n.normalize(lang)
            ui.lang = lang
            images, ids = _gallery_items(metas, filter_key or "全部", lang)
            game_id = meta_mod.counterpart_id(game_id, metas, lang)
            if not game_id and ids:
                game_id, scene = ids[0]["game_id"], ids[0].get("scene")
            meta = metas.get(game_id)
            detail = _show_detail(game_id, scene if meta and meta.scene_choices else None,
                                  lang)
            lobby = (
                theme.topbar_html(lang),
                gr.Tab(label=i18n.t(lang, "tab_lobby")),
                gr.Tab(label=i18n.t(lang, "tab_play")),
                theme.section_title_html(i18n.t(lang, "sec_pick"),
                                         i18n.t(lang, "sec_pick_sub")),
                gr.update(choices=_filter_choices(lang)),
                gr.update(value=images), ids, game_id, *detail,
                gr.update(value=i18n.t(lang, "start")),
                gr.Accordion(label=i18n.t(lang, "advanced")),
                gr.update(choices=i18n.backend_choices(BACKEND_CHOICES, lang),
                          label=i18n.t(lang, "backend")),
                gr.update(label=i18n.t(lang, "seed")),
                gr.update(label=i18n.t(lang, "max_turns")),
                theme.section_title_html(i18n.t(lang, "sec_status")),
                theme.section_title_html(i18n.t(lang, "sec_actions")),
                gr.update(value=i18n.t(lang, "back")),
                gr.Accordion(label=i18n.t(lang, "log")),
            )
            # 对局面板:与 REFRESH_OUTPUTS 同序;无对局时只换文案
            if ui.session is not None:
                play = list(_refresh(ui.session, _entry(ui.game_id), ui,
                                     _title_of(ui, lang)))
                play[9] = gr.update()  # 保留已生成的完整回放(只换标签)
            else:
                play = [gr.update() for _ in REFRESH_OUTPUTS]
                play[1] = i18n.t(lang, "idle_status")
            play[6] = (gr.update(value=play[6], label=i18n.t(lang, "turn_video"))
                       if isinstance(play[6], str) else
                       gr.update(label=i18n.t(lang, "turn_video")))
            play[5] = _merge(play[5], value=i18n.t(lang, "step"))
            play[8] = _merge(play[8], value=i18n.t(lang, "montage_btn"))
            play[9] = _merge(play[9], label=i18n.t(lang, "montage_video"))
            play[10] = _merge(play[10], label=i18n.t(lang, "debug"))
            play[11] = _merge(play[11], label=i18n.t(lang, "boss_pick"))
            return (*lobby, *play, ui)

        def on_start(game_id, option_value, backend_label, seed, max_turns, ui: UiState):
            """开局:确认服务 → 组装 GameParts → 建交互式会话。"""
            lang = ui.lang
            if not game_id:
                raise gr.Error(i18n.t(lang, "err_pick_game"))
            ui.close()
            try:
                entry = _entry(game_id)
            except KeyError as exc:
                raise gr.Error(i18n.t(lang, "err_unknown_game", id=game_id)) from exc
            backend, mode = BACKEND_MAP[backend_label]
            output_dir = _new_output_dir()
            try:
                parts = _build_parts(entry, option_value, llm_socket)
                _attach_initial_frame(parts, output_dir)
                renderer = build_renderer(cfg, output_dir, backend=backend, mode=mode,
                                          socket_path=renderer_socket)
                try:
                    session = InteractiveGame(
                        parts, renderer,
                        output_dir=output_dir,
                        base_seed=int(seed),
                        max_turns=int(max_turns),
                        render_retries=cfg.loop.render_retries,
                        render_failure_policy=cfg.loop.render_failure_policy,
                    )
                except Exception:
                    renderer.close()  # 组装失败不泄漏渲染后端(如 h3 worker)
                    raise
            except Exception as exc:  # noqa: BLE001 - 开局失败要原样告诉玩家
                logger.exception("start game failed")
                raise gr.Error(i18n.t(lang, "err_start", err=_disp(str(exc), lang))) from exc
            ui.game_id, ui.category, ui.session = entry.game_id, entry.category, session
            ui.scene = option_value if entry.options else None
            # State 输出必须返回"要存的值",不能返回 gr.State 组件本身
            return (*_refresh(session, entry, ui, _title_of(ui, lang)), ui)

        def _title_of(ui: UiState, lang: str) -> str:
            """对局标题:开放世界用开局所选场景名,其余用游戏展示名。"""
            meta = metas.get(ui.game_id)
            return meta_mod.display_title(meta, lang, ui.scene) if meta else ui.game_id

        def on_step(radio_label, free_line, boss_label, debug_on, ui: UiState):
            """执行一个回合:玩家选择(可选:强制 Boss 行动)→ 结算 → 渲染 → 展示。"""
            session = ui.session
            if session is None:
                raise gr.Error(i18n.t(ui.lang, "err_no_session"))
            entry = _entry(ui.game_id)
            title = _title_of(ui, ui.lang)
            try:
                action = _collect_action(session, radio_label, free_line, ui.lang)
                if debug_on and boss_label:
                    menu = _boss_menu(session) or []
                    forced = next((a.id for a in menu if a.label == boss_label), None)
                    if forced is None:
                        raise ValueError(i18n.t(ui.lang, "err_boss_stale", label=boss_label))
                    session.parts.runtime.forced_boss_action = forced
                session.step(action)
                _auto_followup(session)  # 声明式自动行动:如开放世界的 AUTO 探索回合
            except (GameFinishedError, ValueError) as exc:
                # 选择无效/游戏已结束:状态不变,玩家重新选择
                return (*_refresh(session, entry, ui, title,
                                  error=i18n.t(ui.lang, "turn_failed",
                                               err=_disp(str(exc), ui.lang))), ui)
            except Exception as exc:  # noqa: BLE001 - 结算失败不影响已落盘的状态
                logger.exception("turn failed")
                return (*_refresh(session, entry, ui, title,
                                  error=i18n.t(ui.lang, "turn_failed",
                                               err=f"{type(exc).__name__}: "
                                                   f"{_disp(str(exc), ui.lang)}")), ui)
            return (*_refresh(session, entry, ui, title), ui)

        def on_montage(ui: UiState):
            """把整局视频拼成带状态字幕的完整回放(HUD 语言 = 当前界面语言)。"""
            session = ui.session
            if session is None:
                raise gr.Error(i18n.t(ui.lang, "err_no_game"))
            try:
                with hud_lang(ui.hud_lang):
                    path = build_summary_montage(
                        session.summary,
                        Path(session.output_dir) / "game_montage.mp4",
                        scene_labels=session.parts.scene_labels,
                        ending_captions=session.parts.ending_captions,
                    )
            except Exception as exc:  # noqa: BLE001 - 拼接失败不影响对局
                logger.exception("montage failed")
                raise gr.Error(i18n.t(ui.lang, "err_montage", err=_disp(str(exc), ui.lang))) from exc
            if path is None:
                raise gr.Error(i18n.t(ui.lang, "err_no_video"))
            return gr.update(value=str(path), visible=True)

        # ---------------------------------------------------------- 事件绑定

        # 新增输出只追加在末尾:scripts/ui_smoke.py 按下标读 API 返回元组
        REFRESH_OUTPUTS = [play_header_md, status_md, action_radio, action_info_md,
                           free_text, step_btn, video, log_md, montage_btn,
                           montage_video, debug_cb, boss_radio, boss_info_md,
                           chapter_md]
        DETAIL_OUTPUTS = [detail_art_img, detail_title_md, detail_blurb_md,
                          detail_meta_md, option_dd]

        filter_radio.change(
            on_filter, inputs=[filter_radio, ui_state],
            outputs=[gallery, lobby_ids, selected_game_id, *DETAIL_OUTPUTS],
            queue=False, show_progress="hidden")
        gallery.select(
            on_gallery_select, inputs=[lobby_ids, ui_state],
            outputs=[selected_game_id, *DETAIL_OUTPUTS],
            queue=False, show_progress="hidden")
        # .input 只响应玩家亲手选择;筛选/点卡片回写下拉时不触发,
        # 否则会把卡片预置的开场场景重置成默认值。
        game_dd.input(
            on_pick_dropdown, inputs=[game_dd, ui_state],
            outputs=[selected_game_id, *DETAIL_OUTPUTS],
            queue=False, show_progress="hidden")
        option_dd.input(on_scene, inputs=[option_dd, selected_game_id, lobby_ids, ui_state],
                        outputs=[detail_art_img, detail_title_md, gallery],
                        queue=False, show_progress="hidden")
        # 语言切换:静态文案 + 对局面板(REFRESH_OUTPUTS 内的组件只出现一次,
        # 其文案合并进对应的 update,见 on_lang)。
        lang_radio.change(
            on_lang, inputs=[lang_radio, filter_radio, selected_game_id, option_dd, ui_state],
            outputs=[topbar_md, lobby_tab, play_tab, pick_title_md, filter_radio,
                     gallery, lobby_ids, selected_game_id, *DETAIL_OUTPUTS,
                     start_btn, adv_acc, backend_radio, seed_num, turns_num,
                     status_title_md, actions_title_md, back_btn, log_acc,
                     *REFRESH_OUTPUTS, ui_state],
            queue=False, show_progress="hidden")
        start_evt = start_btn.click(
            on_start,
            inputs=[selected_game_id, option_dd, backend_radio, seed_num,
                    turns_num, ui_state],
            outputs=REFRESH_OUTPUTS + [ui_state], api_name="start")
        start_evt.success(  # 开局成功才切屏;gr.Error 会让玩家留在大厅
            lambda: gr.Tabs(selected=theme.TAB_PLAY),
            inputs=None, outputs=[tabs], queue=False, show_progress="hidden")
        step_btn.click(
            on_step, inputs=[action_radio, free_text, boss_radio, debug_cb, ui_state],
            outputs=REFRESH_OUTPUTS + [ui_state], api_name="step")
        action_radio.change(_selected_action_info, inputs=[ui_state, action_radio],
                            outputs=[action_info_md], show_progress="hidden")
        debug_cb.change(_toggle_boss_radio, inputs=[debug_cb, ui_state],
                        outputs=[boss_radio], show_progress="hidden")
        boss_radio.change(_selected_boss_info, inputs=[ui_state, boss_radio],
                          outputs=[boss_info_md], show_progress="hidden")
        montage_btn.click(on_montage, inputs=[ui_state], outputs=[montage_video],
                          api_name="montage")
        back_btn.click(lambda: gr.Tabs(selected=theme.TAB_LOBBY),
                       inputs=None, outputs=[tabs],
                       queue=False, show_progress="hidden")

    return demo


# ---------------------------------------------------------------------- 辅助

def _build_theme() -> gr.themes.Base:
    """深色游戏主题:theme.GRADIO_THEME_VARS 明/暗两套同值,只用本地字体栈。"""
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=theme.FONT_STACK,
        font_mono=theme.MONO_STACK,
    )
    values = dict(theme.GRADIO_THEME_VARS)
    # 不是每个变量都有 _dark 版本,只补存在的
    values.update({f"{key}_dark": value for key, value in theme.GRADIO_THEME_VARS.items()
                   if hasattr(base, f"{key}_dark")})
    return base.set(**values)


def _entry(game_id: str):
    from game.registry import get

    return get(game_id)


def _build_parts(entry, option_value, llm_socket: Path):
    """按注册项组装 GameParts;需要 LLM 的游戏先确认服务就绪。"""
    llm = None
    options = {option.name: option_value for option in entry.options if option_value}
    if entry.needs_llm:
        llm = LLMClient(socket_path=llm_socket)
        if llm.ping() is None:
            raise RuntimeError(
                f"LLM 服务未就绪({llm_socket});请先运行 python3 -m engine.llm.service start")
    return entry.build(llm, **options)


def _attach_initial_frame(parts, output_dir: str) -> None:
    """把游戏声明的首帧挂到 VisualState;缺失时生成占位图(仅影响画面)。"""
    frame = resolve_frame(parts.frame)
    if frame is None or not frame.exists():
        # 静默降级会让整局 i2v 链从一张假帧长出来,必须大声告警。
        logger.warning(
            "游戏声明的首帧图不可用(%s),退化为文字占位图;请检查皮肤 frame 路径!",
            parts.frame or "(未声明)")
        frame = create_placeholder_frame(
            Path(output_dir) / "frames" / "initial.png",
            lines=(parts.runtime.__class__.__name__, "initial frame"),
        )
    parts.visual_state.current_frame = frame


def _collect_action(session: InteractiveGame, radio_label: str, free_line: str,
                    lang: str = i18n.DEFAULT_LANG) -> Action:
    """把玩家的界面输入转换成一个 Action(错误提示用界面语言)。"""
    free_line = (free_line or "").strip()
    free_action = session.parts.free_text_action
    if free_line:
        if free_action is None:
            raise ValueError(i18n.t(lang, "err_no_free_text"))
        return dataclasses.replace(free_action, label=free_line[:80])
    if radio_label:
        for action in session.available_actions():
            if action.label == radio_label:
                return action
        raise ValueError(i18n.t(lang, "err_action_stale", label=radio_label))
    raise ValueError(i18n.t(lang, "err_pick_action"))


def _auto_followup(session: InteractiveGame) -> None:
    """执行游戏声明的自动行动(openworld AUTO 回合的 UI 形态)。

    界面约定:当可用行动只剩 parts.auto_action 一个时,本回合不需要玩家
    选择——直接执行并播放视频,不弹出行动列表;这种回合不连续出现的约束
    (如 AUTO 后必须回到交互)由游戏规则负责,因此这里至多自动执行一回合。
    自动回合失败只记日志:已完成的回合与状态保持一致,界面照常刷新。
    """
    auto = session.parts.auto_action
    if auto is None or session.finished:
        return
    try:
        actions = session.available_actions()
        if len(actions) == 1 and actions[0].id == auto.id:
            session.step(actions[0])
    except Exception:  # noqa: BLE001 - 自动回合是增益,失败不影响已完成的回合
        logger.exception("auto action turn failed")


def _merge(update, **changes):
    """在一个 gr.update(...) 字典上追加字段(非 update 值先包成 value 更新)。"""
    if not isinstance(update, dict):
        update = gr.update(value=update)
    return {**update, **changes}


_CJK_RUN = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]+")


def _disp(text: str, lang: str, *, drop: bool = False) -> str:
    """游戏文本按界面语言显示。

    中文界面原样显示(夹带英文可以接受);英文界面经 HUD 词表翻译,且
    绝不允许残留中文——词表未收录的部分:drop=True(说明类文字)整段
    不显示,否则剔除中文片段只留英文部分(标签/状态行)。
    """
    if lang != "en" or not text:
        return text
    out = tr(text, "en")
    if not _CJK_RUN.search(out):
        return out
    if drop:
        return ""
    out = _CJK_RUN.sub(" ", out)
    out = re.sub(r"\s+([,.;:!?)])", r"\1", re.sub(r"\s{2,}", " ", out))
    return out.strip(" ,.;:|·-—") or ""


def _refresh(session: InteractiveGame, entry, ui: UiState, title: str, *,
             error: str | None = None) -> tuple:
    """开局/回合后刷新全部界面组件(顺序 = REFRESH_OUTPUTS 列表)。

    界面文案用 ui.lang;回放 HUD 片段用 ui.hud_lang(开放世界恒为英文)。
    """
    lang = ui.lang
    parts = session.parts
    finished = session.finished

    # 顶栏:标题 + 回合徽章;终局时以结局文案作状态徽章。
    if finished:
        reason = session.summary.finished_reason
        if reason == "game_over":
            caption = parts.ending_captions.get(session.status)
            banner = (_disp(caption, lang) if caption
                      else i18n.status_label(session.status, lang))
        elif reason == "max_turns":
            banner = i18n.t(lang, "out_of_turns")
        else:
            banner = i18n.t(lang, "no_actions")
    else:
        banner = ""
    header = theme.play_header_html(title, entry.category,
                                    session.turns_played + 1, session.max_turns,
                                    status=session.status, banner=banner,
                                    game_id=entry.game_id, lang=lang)

    status_text = _disp(session.describe(), lang)
    if error:
        status_text += f"\n\n> ⚠️ {error}"

    actions = [] if finished else session.available_actions()
    free_action = parts.free_text_action
    text_only = (free_action is not None and len(actions) == 1
                 and actions[0].id == free_action.id)
    # (展示, 取值):取值保持原始 label,_collect_action 按它匹配
    radio_update = gr.update(choices=[(_disp(a.label, lang), a.label) for a in actions],
                             value=None, visible=bool(actions) and not text_only)
    text_update = gr.update(
        visible=free_action is not None and not finished,
        label=i18n.free_text_label(entry.category, lang),
        placeholder=_disp(free_action.description, lang, drop=True) if free_action else "",
        value="",
    )
    step_update = gr.update(interactive=not finished)

    # 展示最新一段视频:优先结束动画,否则最后一个有视频的回合;
    # 画面统一转成 replay-HUD overlay 风格(合成失败回退原始,不阻断游玩);
    # 一段都没有(如刚开局)时把当前首帧循环成静帧短片,让第 0 回合有画面
    video_path = None
    if session.summary.ending is not None and session.summary.ending.video_path:
        ending_caption = parts.ending_captions.get(
            session.summary.final_status or session.status, "")
        video_path = (_hud_clip(session, session.summary.ending.video_path,
                                caption=ending_caption, lang=ui.hud_lang)
                      or session.summary.ending.video_path)
    else:
        for record in reversed(session.summary.records):
            if record.video_path is not None:
                video_path = (_hud_clip(session, record.video_path,
                                        hud=(record.meta or {}).get("hud"),
                                        lang=ui.hud_lang)
                              or record.video_path)
                break
    if video_path is None:
        video_path = _initial_clip(session)

    log_lines = [_disp(line, lang) for line in session.history_lines()]
    log_update = (i18n.t(lang, "log_title") + "\n\n"
                  + "\n".join(f"- {line}" for line in log_lines)
                  if log_lines else i18n.t(lang, "log_empty"))
    info = "" if text_only else _actions_info(session, lang)

    # Debug 面板:运行时提供 boss_action_menu 才出现;选项每回合刷新,
    # 可见性跟随复选框(此处不动 value/visible,避免覆盖勾选状态)。
    boss_menu = [] if finished else (_boss_menu(session) or [])
    debug_update = gr.update(visible=bool(boss_menu))
    boss_update = gr.update(choices=[(_disp(a.label, lang), a.label) for a in boss_menu],
                            value=None)
    boss_info = ""

    card = None if finished else _chapter_card(session)
    if card is not None:
        card = {**card, "title": _disp(card.get("title", ""), lang),
                "text": _disp(card.get("text", ""), lang, drop=True)}
    chapter_update = gr.update(value=theme.chapter_card_html(card, lang),
                               visible=card is not None)

    return (header, status_text, radio_update, info, text_update, step_update,
            video_path, log_update, gr.update(visible=finished),
            gr.update(value=None, visible=False),
            debug_update, boss_update, boss_info, chapter_update)


def _chapter_card(session) -> dict | None:
    """剧情章节卡(运行时特性检测,同 boss_action_menu);失败不影响游玩。"""
    fn = getattr(session.parts.runtime, "chapter_card", None)
    if fn is None:
        return None
    try:
        return fn(session.state)
    except Exception:  # noqa: BLE001 - 章节卡是展示增益
        logger.exception("chapter card failed")
        return None


def _hud_clip(session: InteractiveGame, video: str, hud: dict | None = None,
              caption: str = "", lang: str | None = None) -> str | None:
    """replay-HUD overlay 片段(按语言缓存于会话目录);失败返回 None 走原始视频。"""
    return compose_hud_clip(session.output_dir, video, hud=hud, caption=caption,
                            lang=lang)


def _initial_clip(session: InteractiveGame) -> str | None:
    """开局静帧短片:把 visual_state.current_frame 循环成 3s mp4(带缓存)。"""
    frame = session.parts.visual_state.current_frame
    if frame is None or not Path(frame).exists():
        return None
    clip = Path(session.output_dir) / "clips" / "initial.mp4"
    if not clip.exists():
        clip.parent.mkdir(parents=True, exist_ok=True)
        if not _make_static_video(Path(frame), clip, 3):
            return None
    return str(clip)


def _actions_info(session: InteractiveGame, lang: str) -> str:
    """点选模式下展示每个行为的说明。"""
    return "\n".join(
        f"- **{_disp(action.label, lang)}**"
        + (f" — {desc}" if (desc := _disp(action.description, lang, drop=True)) else "")
        for action in session.available_actions() if action.description
    )


def _selected_action_info(ui: UiState, radio_label: str) -> str:
    """单选变化时展示所选行为的说明。"""
    session = ui.session
    if session is None or not radio_label:
        return ""
    for action in session.available_actions():
        if action.label == radio_label:
            desc = _disp(action.description, ui.lang, drop=True)
            return f"*{desc}*" if desc else ""
    return ""


def _boss_menu(session) -> list:
    """Boss 行动调试菜单(运行时特性检测);不支持的游戏返回空列表。"""
    menu = getattr(session.parts.runtime, "boss_action_menu", None)
    if menu is None:
        return []
    try:
        return menu(session.state)
    except Exception:  # noqa: BLE001 - 菜单失败不影响正常游玩
        return []


def _toggle_boss_radio(debug_on: bool, ui: UiState):
    """勾选 Debug 时展开 Boss 行动单选(仅支持该特性的游戏)。"""
    has_menu = ui.session is not None and bool(_boss_menu(ui.session))
    return gr.update(visible=bool(debug_on) and has_menu)


def _selected_boss_info(ui: UiState, boss_label: str) -> str:
    """Boss 行动单选变化时展示其说明(含前置条件)。"""
    session = ui.session
    if session is None or not boss_label:
        return ""
    for action in _boss_menu(session):
        if action.label == boss_label:
            return (f"*{_disp(action.description, ui.lang)}*"
                    if action.description else "")
    return ""
