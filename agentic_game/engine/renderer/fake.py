"""FakeRenderer:不调用任何模型的离线渲染器。

用途(见 AGENTS.md「Engine engineering rules」):游戏逻辑与 Visual Pipeline 的测试和
无 GPU 环境下的演示。行为:
- final_frame = 首帧拷贝(画面"静止",符合 fake 语义);
- 视频文件:有 ffmpeg 时用首帧合成一段真实的 5s 静止视频(演示时可播放),
  否则直接拷贝首帧字节;
- t2v 请求(first_frame=None,如场景切换回合)自动合成占位首帧,闭环不中断。
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

from engine.renderer.base import (
    InvalidRenderRequestError,
    RenderRequest,
    RenderResult,
    Renderer,
)
from engine.renderer.frames import create_placeholder_frame


class FakeRenderer(Renderer):
    def __init__(self, output_dir: Path, duration_s: int = 5) -> None:
        self.output_dir = Path(output_dir)
        self.duration_s = duration_s
        self.render_count = 0

    def render(self, request: RenderRequest) -> RenderResult:
        request.validate()
        # t2v(如恋爱/开放世界的场景切换回合,use_first_frame=False)也是
        # 合法请求:fake 语义下直接生成一张占位首帧,闭环不中断。
        if request.first_frame is None:
            request = _with_placeholder_frame(request, self.output_dir / "frames")
        if not Path(request.first_frame).exists():
            raise InvalidRenderRequestError(f"first_frame not found: {request.first_frame}")

        self.render_count += 1
        stem = request.turn_id
        video_path = self.output_dir / "videos" / f"{stem}.mp4"
        frame_path = self.output_dir / "frames" / f"{stem}.png"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.parent.mkdir(parents=True, exist_ok=True)

        if _make_static_video(request.first_frame, video_path, self.duration_s):
            pass
        else:
            # 无 ffmpeg 时退化为字节拷贝:仅供存在性检查,不可被下游
            # 当作真实视频 probe(帧提取等)。
            shutil.copyfile(request.first_frame, video_path)
        shutil.copyfile(request.first_frame, frame_path)
        return RenderResult(
            turn_id=request.turn_id,
            video_path=video_path,
            final_frame=frame_path,
            elapsed_s=0.0,
            seed=request.seed,
        )


def _with_placeholder_frame(request: RenderRequest, frames_dir: Path) -> RenderRequest:
    """给 t2v 请求合成一张占位首帧,返回新的 RenderRequest。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame = create_placeholder_frame(
        frames_dir / f"{request.turn_id}-t2v-seed{request.seed}.png",
        lines=(request.turn_id, "t2v (no first frame)", f"seed={request.seed}"),
    )
    return dataclasses.replace(request, first_frame=frame)


def _make_static_video(first_frame: Path, video_path: Path, duration_s: int) -> bool:
    """用 ffmpeg 把首帧循环成静止视频;ffmpeg 不可用/失败时返回 False。

    -nostdin:ffmpeg 默认会读 stdin 交互,挂在这里绝不能吞掉管道数据。
    scale+pad 到偶数尺寸:源图(如 1672x941 的首帧)奇数边会让
    yuv420p 的 libx264 直接报错,必须显式归一化到 864x480 画布。
    """
    if shutil.which("ffmpeg") is None:
        return False
    video_filter = (
        "scale=864:480:force_original_aspect_ratio=decrease,"
        "pad=864:480:(ow-iw)/2:(oh-ih)/2"
    )
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        "24",
        "-i",
        str(first_frame),
        "-t",
        str(duration_s),
        "-vf",
        video_filter,
        "-r",
        "24",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0 or not video_path.exists():
        return False
    return True
