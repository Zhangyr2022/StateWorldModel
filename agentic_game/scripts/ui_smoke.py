#!/usr/bin/env python3
"""UI 冒烟:用 gradio_client 驱动本机 run_ui.py(fake 后端)完整打一局。

先起服务再跑本脚本(手动运行,不进 CI;gradio_client 随 gradio 一起安装):

    python3 run_ui.py --renderer fake --port 7861 &
    python3 scripts/ui_smoke.py http://127.0.0.1:7861

gr.State 不进 gradio API 面:程序化客户端先用 /on_pick_dropdown 选定游戏,
再调 /start(服务端会用同一会话里最后一次 State 更新)。动作标签从 /start
返回的 Radio update payload 里读,读不到时回退 game.registry。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # 以脚本方式运行时仍可 import game/ 仓库包

DEFAULT_GAME = "duel_oracle_machine_epic"
BACKEND_FAKE = "fake · 离线占位(无需 GPU)"
# gradio_client 可见的输出顺序(REFRESH_OUTPUTS 里纯 gr.update 的按钮类输出
# 不会出现在 API 返回中,State 也不出现):
HEADER_OUT, ACTION_RADIO_OUT, VIDEO_OUT, LOG_OUT = 0, 2, 5, 6


def main(url: str, game_id: str = DEFAULT_GAME) -> int:
    from gradio_client import Client

    client = Client(url)
    client.view_api()  # 端点形状变化时第一时间在这里暴露

    print(f"[smoke] pick game {game_id}")
    client.predict(game_id=game_id, api_name="/on_pick_dropdown")

    print(f"[smoke] start (max_turns=3, backend=fake)")
    start = client.predict(option_value=None, backend_label=BACKEND_FAKE,
                           seed=7, max_turns=3, api_name="/start")
    _expect(len(start) >= 11, f"start 输出应 ≥11 元组,实际 {len(start)}")
    _expect(game_id in str(start[0]), f"顶栏应是 {game_id}:{_one_line(start[0], 80)}")
    _check_video("start", start[VIDEO_OUT])
    labels = _radio_choices(start[ACTION_RADIO_OUT])
    _expect(bool(labels), "start 应给出可选行动")
    print(f"[smoke] actions: {labels[:3]}")

    for turn in range(1, 4):
        result = client.predict(radio_label=labels[0], free_line="", boss_label=None,
                                debug_on=False, api_name="/step")
        _expect(len(result) >= 11, f"step#{turn} 输出应 ≥11 元组,实际 {len(result)}")
        _check_video(f"step#{turn}", result[VIDEO_OUT])
        _expect(f"R{turn} " in str(result[LOG_OUT]),
                f"step#{turn} 后战况记录应含 R{turn}:{_one_line(result[LOG_OUT], 80)}")
        choices = _radio_choices(result[ACTION_RADIO_OUT])
        if choices:
            labels = choices  # 菜单每回合刷新,下一回合用新菜单
        print(f"[smoke] step#{turn} ok · log={_one_line(result[LOG_OUT], 90)}")

    # 横幅文案随界面语言(默认跟随 AGENTIC_GAME_HUD_LANG),两种语言都认
    from engine.ui.i18n import t
    _expect(any(t(lang, "out_of_turns") in str(result[HEADER_OUT]) for lang in ("zh", "en")),
            "3 回合打满后顶栏应显示终局横幅")

    print("[smoke] montage")
    montage = client.predict(api_name="/montage")
    path = _as_path(montage)
    _expect(bool(path) and Path(path).exists(), f"回放文件应存在:{montage}")
    print(f"[smoke] montage file: {path}")

    print("\n[smoke] PASS ✓")
    return 0


def _radio_choices(payload) -> list[str]:
    """Radio 的 gr.update payload → 选项取值列表;格式不符返回空。

    choices 是 (展示, 取值) 对:展示随界面语言变(英文界面经词表翻译),
    提交给 /step 的必须是取值(游戏原始 label)。
    """
    if not isinstance(payload, dict):
        return []
    choices = payload.get("choices") or []
    labels = []
    for item in choices:
        if isinstance(item, (list, tuple)) and item:
            labels.append(str(item[-1]))
        elif isinstance(item, str):
            labels.append(item)
    return labels


def _check_video(stage: str, value) -> None:
    path = _as_path(value)
    if path is None:
        print(f"[smoke] {stage}: 无视频(无 ffmpeg 时属预期)")  # 首帧短片需要 ffmpeg
        return
    _expect(Path(path).exists(), f"{stage} 的视频文件应存在:{path}")


def _as_path(value):
    # gradio 的 Video 返回可能是 {video:..} / {value:{video:..}} / str,逐层拆包
    while isinstance(value, dict):
        value = value.get("video") or value.get("value")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, (str, Path)) and str(value):
        return Path(str(value).replace("file=", ""))
    return None


def _one_line(text, limit):
    line = " ".join(str(text).split())
    return line[:limit] + ("…" if len(line) > limit else "")


def _expect(cond: bool, message: str) -> None:
    if not cond:
        print(f"\n[smoke] FAIL ✗ — {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7861"
    game = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GAME
    raise SystemExit(main(url, game))
