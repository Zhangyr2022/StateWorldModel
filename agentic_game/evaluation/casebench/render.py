"""渲染:instruction → H3 I2VA/ref2va prompt → 常驻服务(或 fake 干跑)。

数据侧 instruction 就是给视频模型的完整指令(卡片声明 "Inputs to the
video model"),评测不改写一个字,只按 H3 官方 I2VA
信封包装:首帧对齐行 + integrated_multimodal_description + N/A 声场/音乐。
时长固定 5s(H3 只接受 5/10/15;数据约束 chunk=2-5s 取上界)。

ref2va 首帧模式(--ref2va,不开白模):引擎换 ref2va adapter,input.png
不再是"锁定的首帧"而是纯图片 reference——ref2va 架构上 reference 不锁
帧 0,所以对齐行必须换成 ref2va 首帧锚定句(显式要求"从该图开始播"),
描述主体不动(apply_ref2va_anchor)。服务用 box2clay socket 上的 engine
ref2va 常驻实例(与 wbench runner 同一份 ensure_service)。
"""

from __future__ import annotations

from pathlib import Path

from engine.renderer.base import RenderRequest
from engine.renderer.remote import H3RemoteRenderer

from evaluation.casebench.dataset import TwinSpec

DURATION_S = 5

# I2VA 首帧对齐指令(指南 §2.1),必须是 prompt 第一行,后接一个空行。
_ALIGNMENT_LINE = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)

# ref2va 首帧锚定句(--ref2va)。与 I2VA 对齐行的语义差:i2v 的 image
# 参数就是帧 0(对齐行只声明"引用");ref2va 的 reference 不锁首帧,必须
# 显式命令"在 <Picture 1> 上开局、忠实复现、再从那里动起来、保持风格"。
# 措辞沿用 engine/visual/reference 的首帧锚定从句 + box2clay h3ref_batch
# 的动画从句(17 条成片验证过的 ref2va 表达)。
_REF2VA_OPENING_LINE = (
    "<Picture 1> is the exact first frame of the target video: begin the video "
    "on <Picture 1>, reproducing it faithfully, then animate naturally from "
    "there while keeping its exact visual style and lighting."
)

# ref2va 常驻服务统一监听 box2clay socket(2026-09-24 定稿):engine worker
# 讲双协议,agentic runner / box2clay CLI 共用同一实例,不再互切重载。
from game.wbench.previz import BOX2CLAY_ROOT  # noqa: E402  (单一事实来源)


def ref2va_socket_path() -> Path:
    """engine ref2va 常驻服务的 socket(与 game/wbench/run_wbench 一致)。"""
    return BOX2CLAY_ROOT / "results" / "h3_ref2va.sock"


def apply_ref2va_anchor(prompt: str) -> str:
    """i2v prompt → ref2va 首帧 prompt(描述主体逐字节不动)。

    信封/remain/framework 的第一行是 I2VA 对齐行 → 原位换成 ref2va 锚定句;
    raw/raw_state 没有对齐行 → 锚定句 + 空行加在开头(ref2va 下裸原文
    没有"从该图开始"的指令,模型会凭文本自由开局,首帧不复现)。
    """
    if prompt.startswith(_ALIGNMENT_LINE):
        return prompt.replace(_ALIGNMENT_LINE, _REF2VA_OPENING_LINE, 1)
    return f"{_REF2VA_OPENING_LINE}\n\n{prompt}"


def build_prompt(instruction: str) -> str:
    """instruction 原文 → H3 I2VA 信封(不增删改任何语义内容)。"""
    text = " ".join(instruction.split())
    if not text:
        raise ValueError("instruction is empty; nothing to render")
    return "\n".join([
        _ALIGNMENT_LINE,
        "",
        f"integrated_multimodal_description: [Shot 1] {text}",
        "overall_soundscape: N/A",
        "non_diegetic_music: N/A",
    ])


def video_path_for(run_dir: Path, twin: TwinSpec) -> Path:
    return Path(run_dir) / "videos" / f"{twin.clip_id}.mp4"


def prompt_path_for(run_dir: Path, twin: TwinSpec) -> Path:
    """该 clip 实际发送给 H3 的 prompt 存档(失败报告按此复现)。"""
    return Path(run_dir) / "prompts" / f"{twin.clip_id}.txt"


def _dump_prompt(run_dir: Path, twin: TwinSpec, prompt: str) -> None:
    path = prompt_path_for(run_dir, twin)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def final_frame_for(run_dir: Path, twin: TwinSpec) -> Path:
    """渲染器抽好的末帧(H3RemoteRenderer/FakeRenderer 都写这个路径)。"""
    return Path(run_dir) / "frames" / f"{twin.clip_id}.png"


def build_raw_state_prompt(twin: TwinSpec) -> str:
    """baseline 告知版:raw 原文 + 追加 gold 终态的表现短语。"""
    phrases = "; ".join(twin.gold_end_phrases)
    if not phrases:
        raise ValueError(f"{twin.clip_id}: no gold end phrases from visual_vocab")
    return (f"{twin.baseline_text}\n\nThe clip must end as: {phrases}. "
            "Hold this final state for about the last second.")


def render_twin(renderer, twin: TwinSpec, run_dir: Path, *,
                seed: int = 0, duration_s: int = DURATION_S, force: bool = False,
                prompt_mode: str = "envelope",
                ref2va: bool = False) -> tuple[Path, bool]:
    """渲染单个孪生 clip;已有产物且非 force 时跳过。返回 (视频路径, 是否新渲)。

    prompt_mode:
    - "envelope" instruction 原文包 H3 I2VA 信封(数据未改动);
    - "remain"   主流程适配:一律摘除点名反结局主动态的 must-NOT 句
      (twin.render_instruction),反分支由 "Instead, 保持初始状态" 表达;
    - "raw"      baseline(单独 H3):card txt「Inputs to the video model」
      整段原文直接作 prompt,不做任何包装(仍带 input.png 首帧);
    - "raw_state" baseline 告知版:raw 原文 + 追加 gold 终态的表现短语。

    ref2va=True 时 prompt 再过 apply_ref2va_anchor(对齐行 → ref2va 首帧
    锚定句),渲染请求走 ref2va 引擎(input.png 作纯图片 reference)。
    """
    if prompt_mode not in ("envelope", "remain", "raw", "raw_state"):
        raise ValueError(f"unknown prompt_mode: {prompt_mode!r}")
    video_path = video_path_for(run_dir, twin)
    if video_path.is_file() and video_path.stat().st_size > 0 and not force:
        return video_path, False
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if prompt_mode == "raw":
        prompt = twin.baseline_text
    elif prompt_mode == "raw_state":
        prompt = build_raw_state_prompt(twin)
    elif prompt_mode == "remain":
        prompt = build_prompt(twin.render_instruction)
    else:
        prompt = build_prompt(twin.instruction)
    if ref2va:
        prompt = apply_ref2va_anchor(prompt)
    _dump_prompt(run_dir, twin, prompt)
    renderer.render(RenderRequest(
        turn_id=twin.clip_id,
        prompt=prompt,
        seed=seed,
        first_frame=twin.input_image,
        duration_s=duration_s,
    ))
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"render reported success but no video at {video_path}")
    return video_path, True


def build_renderer(run_dir: Path, *, backend: str, mode: str | None,
                   socket_path: Path | None, config_path: Path | None = None):
    """装配渲染后端(remote 常驻服务为批量默认;fake 供无 GPU 干跑)。"""
    from engine.config import load_config
    from engine.renderer.factory import build_renderer as _build

    cfg = load_config(config_path)
    return _build(cfg, Path(run_dir), backend=backend, mode=mode, socket_path=socket_path)


__all__ = ["DURATION_S", "H3RemoteRenderer", "apply_ref2va_anchor",
           "build_prompt", "build_renderer", "final_frame_for",
           "ref2va_socket_path", "render_twin", "video_path_for"]
