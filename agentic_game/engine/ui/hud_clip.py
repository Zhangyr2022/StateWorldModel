"""回合视频的 replay-HUD overlay 合成(纯表现层,不依赖 gradio)。

把一段原始回合视频经 engine.montage 的 overlay 面板(replay_hud 同款)
重编码为带 HUD 的展示片段;失败一律返回 None,由调用方回退原始视频——
合成只是观感增益,绝不影响对局状态。
"""

from __future__ import annotations

import logging
from pathlib import Path

from engine.hud_i18n import hud_lang
from engine.montage import ClipSpec, build_game_montage

logger = logging.getLogger("engine.ui")


def compose_hud_clip(output_dir, video: str, hud: dict | None = None,
                     caption: str = "", lang: str | None = None) -> str | None:
    """把 video 合成带 HUD 面板的片段,缓存到 <output_dir>/clips/。

    lang:HUD 文本语言("en"/"zh";None = 跟随 AGENTIC_GAME_HUD_LANG)。
    输出名 = <原文件名>-hud.mp4,指定 lang 时为 -hud-<lang>.mp4(已存在
    直接复用,不同语言互不覆盖);任何异常返回 None。
    """
    if not video:
        return None
    source = Path(video)
    suffix = f"-hud-{lang}" if lang else "-hud"
    out = Path(output_dir) / "clips" / f"{source.stem}{suffix}.mp4"
    if out.exists():
        return str(out)
    try:
        work = out.parent / "hud_parts"
        with hud_lang(lang):
            build_game_montage([ClipSpec(video=source, caption=caption, hud=hud)],
                               out, work_dir=work, hud_style="overlay")
        for part in work.glob("part-*.mp4"):
            part.unlink(missing_ok=True)
        return str(out) if out.exists() else None
    except Exception:  # noqa: BLE001 - 表现层失败回退原始视频
        logger.exception("hud clip compose failed: %s", video)
        return None
