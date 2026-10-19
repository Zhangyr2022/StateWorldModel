"""MinimaxH3Renderer:通过 h3-a800 仓库的稳定入口 run.sh 生成视频。

两种模式(同一 Renderer 接口,游戏代码无感知):

- SubprocessH3Renderer:每次请求启动一次 run.sh(torchrun 8 卡)。
  简单可靠,但每次都要重新装载 ~141GB 权重(~8 分钟),适合单次验证。
- H3SessionRenderer(engine.renderer.session):worker 常驻,装载一次,
  每回合只付 ~7s 生成时间,适合连续多回合(Phase 4)。

本模块不修改 h3-a800 的任何文件,只调用它的命令行接口
(`./run.sh <nproc> <out.mp4> [infer.py 参数]`),prompt 经 --prompt-file
传递以避免 shell 转义问题。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

from engine.config import WORKSPACE_ROOT, conda_bin
from engine.renderer.base import (
    RenderError,
    RenderFailedError,
    RenderRequest,
    RenderResult,
    RenderTimeoutError,
    Renderer,
)
from engine.renderer.frames import extract_final_frame

# h3-a800 的固定约定(其 run.sh 内置同名变量):HF 离线防 hub 探测挂起,
# DISABLE_ADDMM_CUDA_LT 绕开本机 cuBLASLt 兼容性 bug。
_H3_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "DISABLE_ADDMM_CUDA_LT": "1",
    "TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING": "false",
}

DEFAULT_H3_REPO = WORKSPACE_ROOT / "h3-a800"
# PATH 上有 torchrun(已激活 h3-a800 环境)优先;否则用部署约定 conda 路径
DEFAULT_TORCHRUN = Path(which("torchrun") or conda_bin("h3-a800", "torchrun"))


def h3_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else __import__("os").environ)
    # default-set 而非覆盖:目标机可用外部环境变量关闭本机 cuBLASLt
    # workaround 或改 HF 镜像(HF_HUB_OFFLINE / DISABLE_ADDMM_CUDA_LT 等)
    for key, value in _H3_ENV.items():
        env.setdefault(key, value)
    return env


class SubprocessH3Renderer(Renderer):
    """每次渲染调用一次 run.sh 的最简实现。"""

    def __init__(
        self,
        output_dir: Path,
        *,
        h3_repo: Path = DEFAULT_H3_REPO,
        torchrun: Path = DEFAULT_TORCHRUN,
        nproc: int = 8,
        resolution: str = "480p",
        attention_backend: str = "dense",
        model: Path | None = None,
        adapter: Path | None = None,
        timeout_s: int = 3600,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.h3_repo = Path(h3_repo)
        self.torchrun = Path(torchrun)
        self.nproc = nproc
        self.resolution = resolution
        self.attention_backend = attention_backend
        self.model = model
        self.adapter = adapter
        self.timeout_s = timeout_s
        self.render_count = 0

    def render(self, request: RenderRequest) -> RenderResult:
        request.validate()
        self.render_count += 1
        # 全部传绝对路径:run.sh 会 cd 到 h3-a800 仓库,相对路径会写错位置
        video_path = (self.output_dir / "videos" / f"{request.turn_id}.mp4").resolve()
        frame_path = (self.output_dir / "frames" / f"{request.turn_id}.png").resolve()
        prompt_file = (self.output_dir / "tmp" / f"{request.turn_id}.prompt.txt").resolve()
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(request.prompt, encoding="utf-8")

        first_frame = request.first_frame.resolve() if request.first_frame else None
        command = [
            "bash",
            str(self.h3_repo / "run.sh"),
            str(self.nproc),
            str(video_path),
            "--task",
            "i2v" if first_frame is not None else "t2v",
            "--duration",
            str(request.duration_s),
            "--res",
            self.resolution,
            "--attention-backend",
            self.attention_backend,
            "--seed",
            str(request.seed),
            "--prompt-file",
            str(prompt_file),
        ]
        if self.model is not None:
            command += ["--model", str(self.model)]
        if self.adapter is not None:
            command += ["--adapter", str(self.adapter)]
        if first_frame is not None:
            command += ["--image", str(first_frame)]

        try:
            proc = subprocess.run(
                command,
                cwd=self.h3_repo,
                env=h3_environment(),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderTimeoutError(
                f"h3 run.sh timed out after {self.timeout_s}s for {request.turn_id}"
            ) from exc
        except OSError as exc:
            raise RenderFailedError(f"failed to launch h3 run.sh: {exc}") from exc

        if proc.returncode != 0:
            raise RenderFailedError(
                f"h3 run.sh exited with {proc.returncode} for {request.turn_id}: "
                + _stderr_tail(proc.stderr)
            )
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RenderFailedError(
                f"h3 run.sh reported success but produced no video for {request.turn_id}: "
                + _stderr_tail(proc.stderr)
            )

        final_frame = extract_final_frame(video_path, frame_path)
        return RenderResult(
            turn_id=request.turn_id,
            video_path=video_path,
            final_frame=final_frame,
            elapsed_s=0.0,
            seed=request.seed,
        )


def _stderr_tail(stderr: str | None, lines: int = 8) -> str:
    if not stderr:
        return "(no stderr)"
    tail = [line for line in stderr.strip().splitlines() if line.strip()][-lines:]
    return " | ".join(tail)


__all__ = [
    "DEFAULT_H3_REPO",
    "DEFAULT_TORCHRUN",
    "SubprocessH3Renderer",
    "RenderError",
    "h3_environment",
]
