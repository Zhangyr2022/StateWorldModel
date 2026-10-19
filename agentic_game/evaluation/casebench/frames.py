"""判定帧采样:视频 → 按固定帧率抽帧(评测侧;不动 engine/renderer/frames.py)。

fps=3 对 5s 片段产出约 15 帧,加渲染时抽好的末帧,给判定 VLM 完整的
时间覆盖(outcome 问题只看结尾,末帧必须在场)。长边缩放到与常驻 VL
worker 相同的 896px,控制图像 token 成本。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from engine.renderer.base import FrameExtractionError

# 与常驻 VL worker(engine/llm/worker.py)一致的图像长边上限。
IMAGE_MAX_SIDE = 896

# 单次判定送 VLM 的帧数上限:超过则均匀降采样(安全阀,防上下文爆炸)。
MAX_FRAMES_PER_JUDGE = 32


def extract_frames_fps(
    video_path: Path,
    out_dir: Path,
    fps: float = 3.0,
    size: int = IMAGE_MAX_SIDE,
    max_frames: int = MAX_FRAMES_PER_JUDGE,
) -> list[Path]:
    """按 fps 提取全部帧;超过 max_frames 时均匀降采样。失败抛 FrameExtractionError。"""
    if shutil.which("ffmpeg") is None:
        raise FrameExtractionError("ffmpeg not available")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={size}:-2:flags=bicubic",
        str(out_dir / "f-%04d.png"),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise FrameExtractionError(f"ffmpeg failed: {exc}") from exc
    if proc.returncode != 0:
        raise FrameExtractionError(f"ffmpeg failed: {proc.stderr[-200:]}")
    frames = sorted(out_dir.glob("f-*.png"))
    if not frames:
        raise FrameExtractionError(f"no frames extracted from {video_path}")
    if max_frames and len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[min(len(frames) - 1, int(i * step))] for i in range(max_frames)]
    return frames


def judge_images(video_path: Path, final_frame: Path, out_dir: Path, *,
                 fps: float = 3.0, max_frames: int = MAX_FRAMES_PER_JUDGE) -> list[Path]:
    """送 VLM 的图像序列:采样帧 + 末帧(去重,末帧恒在最后)。"""
    images = extract_frames_fps(video_path, out_dir, fps=fps, max_frames=max_frames)
    if final_frame.is_file() and final_frame not in images:
        images.append(final_frame)
    return images


def extract_final_frame(video_path: Path, out_png: Path) -> Path:
    """抽视频最后一帧存 PNG(外部提交没有渲染器代抽的末帧时用)。"""
    if shutil.which("ffmpeg") is None:
        raise FrameExtractionError("ffmpeg not available")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-sseof", "-0.2", "-i", str(video_path), "-frames:v", "1", str(out_png),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise FrameExtractionError(f"ffmpeg failed: {exc}") from exc
    if proc.returncode != 0 or not out_png.is_file() or out_png.stat().st_size == 0:
        raise FrameExtractionError(f"final frame extraction failed: {proc.stderr[-200:]}")
    return out_png


__all__ = ["IMAGE_MAX_SIDE", "MAX_FRAMES_PER_JUDGE", "extract_final_frame",
           "extract_frames_fps", "judge_images"]
