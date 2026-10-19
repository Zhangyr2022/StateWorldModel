"""Renderer 接口与错误分类(论文贡献 4:Training-Free Semantic Rendering)。

游戏与主循环只依赖本抽象,不依赖任何具体视频模型——现成模型
(Minimax-H3)零训练直接接入,换模型只换 adapter,不动 Runtime 与游戏层
("Cross-game adaptation happens in program space")。

关键约束(见 AGENTS.md「Engine engineering rules」):
- Renderer 只表现已经确定的游戏结果,失败属于表现层失败,
  不允许影响 Game State。
- 错误必须分类(超时/失败/帧提取失败/请求非法),上层据此决定
  retry、跳过视觉或中止,而不是裸 except。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class RenderError(Exception):
    """Renderer 错误基类。"""


class InvalidRenderRequestError(RenderError):
    """请求本身非法(空 prompt、缺首帧等),重试也不会成功。"""


class RenderTimeoutError(RenderError):
    """外部系统在时限内没有返回。"""


class RenderFailedError(RenderError):
    """视频生成失败(API 错误、非零退出、输出缺失等)。"""


class FrameExtractionError(RenderError):
    """视频已生成但最后一帧提取失败。"""


@dataclass(frozen=True)
class RenderRequest:
    """一次视频生成请求。

    first_frame 为 None 表示文本生成视频(t2v),通常只发生在游戏开始;
    之后每个回合都应携带上一段视频的最后一帧(i2v)以保持画面连续。

    reference_video 非 None 时为参考引导生成(ref2va):控制视频承载
    本回合的运动/镜头,first_frame 作为外观与场景锚点;二者一起进入
    模型的 references。不支持该模式的 Renderer 应在 render() 里抛
    InvalidRenderRequestError,而不是静默忽略。
    """

    turn_id: str
    prompt: str
    seed: int
    first_frame: Path | None = None
    duration_s: int = 5
    reference_video: Path | None = None

    def validate(self) -> None:
        if not self.turn_id:
            raise InvalidRenderRequestError("turn_id is required")
        if not self.prompt.strip():
            raise InvalidRenderRequestError("prompt must be a non-empty string")
        if self.duration_s <= 0:
            raise InvalidRenderRequestError("duration_s must be positive")
        if self.reference_video is not None and not self.reference_video.is_file():
            raise InvalidRenderRequestError(
                f"reference_video does not exist: {self.reference_video}"
            )


@dataclass(frozen=True)
class RenderResult:
    """一次成功的渲染:视频文件与它的最后一帧。"""

    turn_id: str
    video_path: Path
    final_frame: Path
    elapsed_s: float = 0.0
    seed: int = 0


class Renderer(ABC):
    """统一渲染接口。游戏与主循环只依赖这个抽象,不依赖具体后端。"""

    @abstractmethod
    def render(self, request: RenderRequest) -> RenderResult:
        """渲染一段约 duration_s 秒的视频并返回其最后一帧。

        实现必须自行处理超时、重试粒度以下的错误分类;这里不吞异常。
        """

    def close(self) -> None:
        """释放后端资源(子进程、会话等)。默认无操作。"""
