"""帧工具:最后一帧提取与占位首帧。

最后一帧是视频连续性的关键(末帧链接约定):上一段视频的最后一帧
就是下一段视频的首帧输入。提取统一走 ffmpeg(本机已安装),
不引入 Python 视频解码依赖。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from engine.renderer.base import FrameExtractionError, InvalidRenderRequestError

_FFMPEG_TIMEOUT_S = 120


def extract_final_frame(video_path: Path, out_path: Path) -> Path:
    """从视频末尾附近取一帧作为 final frame。

    -sseof -0.1:从文件尾倒数 0.1s 开始定位,取第一帧即最后一帧。
    """
    if not video_path.exists():
        raise FrameExtractionError(f"video does not exist: {video_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(out_path),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise FrameExtractionError(f"ffmpeg timed out extracting final frame of {video_path}") from exc
    except OSError as exc:
        raise FrameExtractionError(f"failed to launch ffmpeg: {exc}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise FrameExtractionError(
            f"ffmpeg failed (exit {proc.returncode}) extracting final frame of {video_path}: "
            + " | ".join(tail)
        )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise FrameExtractionError(f"ffmpeg produced no frame at {out_path}")
    return out_path


def extract_sample_frames(video_path: Path, out_dir: Path, n: int = 3,
                          size: int = 896) -> list[Path]:
    """从视频中均匀采样 n 帧(头/中/尾),缩放到长边 size,返回帧路径列表。

    供 VLM"观看"一段视频使用:头中尾三帧足以概括镜头的起承与结尾。
    """
    if not video_path.exists():
        raise FrameExtractionError(f"video does not exist: {video_path}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S,
    )
    if probe.returncode != 0:
        raise FrameExtractionError(f"ffprobe failed on {video_path}")
    try:
        duration = float(probe.stdout.strip())
    except ValueError as exc:
        raise FrameExtractionError(f"cannot parse duration of {video_path}") from exc

    duration = max(duration, 0.5)
    offsets = [duration * (i + 0.5) / n for i in range(n)]
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, seconds in enumerate(offsets):
        frame_path = out_dir / f"frame-{index}.png"
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-ss", f"{seconds:.2f}", "-i", str(video_path),
             "-frames:v", "1", "-vf", f"scale={size}:-1", str(frame_path)],
            capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S,
        )
        if result.returncode != 0 or not frame_path.exists() or frame_path.stat().st_size == 0:
            raise FrameExtractionError(f"failed to sample frame at {seconds:.2f}s of {video_path}")
        frames.append(frame_path)
    return frames


def create_placeholder_frame(
    out_path: Path,
    size: tuple[int, int] = (864, 480),
    lines: tuple[str, ...] = (),
) -> Path:
    """生成一张占位首帧(demo 起始画面)。

    只在 demo/离线演示时使用;需要 Pillow。测试与游戏逻辑不依赖这里。
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - 环境缺依赖时给出可操作的提示
        raise InvalidRenderRequestError(
            "create_placeholder_frame needs Pillow (pip install pillow), "
            "or provide an explicit first_frame image"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(18, 24, 38))
    draw = ImageDraw.Draw(image)
    y = size[1] // 2 - 20 * len(lines)
    for line in lines:
        draw.text((40, y), line, fill=(220, 226, 240))
        y += 28
    image.save(out_path)
    return out_path
