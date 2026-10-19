"""配置:默认值 ← config.toml ← 环境变量(后者优先)。

环境相关配置不硬编码在业务代码里(见 AGENTS.md「Engine engineering rules」);
敏感信息(API key 等)只经环境变量进入,不落入仓库。
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which

CONFIG_FILE = Path("config.toml")

# ── 部署拓扑约定 ────────────────────────────────────────────────────
# 4 个项目并排存放(agentic_game / worldstate_bench / spatial_proxy / h3-a800),
# 跨项目默认路径一律由本文件位置向上推导 workspace 根,不写死机器绝对路径;
# 布局不同时用 WORKSPACE_ROOT 环境变量覆盖。conda 环境根用 CONDA_ENVS_ROOT。
WORKSPACE_ROOT = Path(os.environ.get(
    "WORKSPACE_ROOT", Path(__file__).resolve().parents[2]))
CONDA_ENVS_ROOT = Path(os.environ.get(
    "CONDA_ENVS_ROOT", Path.home() / "miniconda3" / "envs"))


def conda_bin(env_name: str, exe: str) -> str:
    """部署约定的 conda 环境可执行文件路径(CONDA_ENVS_ROOT 可覆盖)。"""
    return str(CONDA_ENVS_ROOT / env_name / "bin" / exe)


# PATH 上有 torchrun(已激活 h3-a800 环境)就用它,否则回落部署约定路径。
_DEFAULT_TORCHRUN = which("torchrun") or conda_bin("h3-a800", "torchrun")


@dataclass
class RendererConfig:
    backend: str = "fake"        # fake | h3
    mode: str = "session"        # session | remote | subprocess(h3 后端时有效)
    h3_repo: str = str(WORKSPACE_ROOT / "h3-a800")
    torchrun: str = _DEFAULT_TORCHRUN
    # h3 GPU 进程数:<=0 = 自动探测本机可见 GPU 数(默认);正数 = 固定卡数。
    # load_config 会把它解析成有效卡数,服务端与客户端据此派生同一个 socket 名。
    nproc: int = 0
    resolution: str = "480p"     # 480p | 768p
    attention_backend: str = "dense"
    startup_timeout_s: int = 3600
    render_timeout_s: int = 1800
    request_timeout_s: int = 3600  # subprocess 模式的单次请求超时
    socket_path: str = ""          # remote 模式连接的常驻服务 socket;空 = service 默认路径


@dataclass
class LoopConfig:
    base_seed: int = 7
    max_turns: int = 12
    render_retries: int = 1
    render_failure_policy: str = "skip"  # skip | abort
    output_dir: str = "generated"


@dataclass
class AppConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)


def detect_gpu_count() -> int:
    """本机可见的 NVIDIA GPU 数;探测不到返回 0。

    先尊重 CUDA_VISIBLE_DEVICES(与 torch 的可见集语义一致:设了空串
    = 0 卡),未设置才问 nvidia-smi。主进程不 import torch(基础层约束),
    探测只用环境变量 + 子进程。
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([token for token in visible.split(",") if token.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"],
                             capture_output=True, text=True, timeout=10.0, check=False)
        if out.returncode == 0:
            return len([line for line in out.stdout.splitlines() if line.strip()])
    except (OSError, subprocess.TimeoutExpired):
        pass
    return 0


def load_config(path: Path | None = None) -> AppConfig:
    """读取配置文件与环境变量;文件不存在时全部走默认值。"""
    config = AppConfig()
    source = Path(path) if path is not None else CONFIG_FILE
    if source.exists():
        data = tomllib.loads(source.read_text(encoding="utf-8"))
        _apply(config.loop, data.get("loop", {}))
        _apply(config.renderer, data.get("renderer", {}))

    env = os.environ
    if value := env.get("AGENTIC_GAME_RENDERER"):
        config.renderer.backend = value
    if value := env.get("AGENTIC_GAME_H3_REPO"):
        config.renderer.h3_repo = value
    if value := env.get("AGENTIC_GAME_TORCHRUN"):
        config.renderer.torchrun = value
    if value := env.get("AGENTIC_GAME_NPROC"):
        config.renderer.nproc = int(value)
    if value := env.get("AGENTIC_GAME_SEED"):
        config.loop.base_seed = int(value)
    if value := env.get("AGENTIC_GAME_MAX_TURNS"):
        config.loop.max_turns = int(value)

    # nproc <= 0(默认)= 常驻服务在本机有多少可见卡就用多少卡;
    # 一张卡都探测不到时回落 1,让 torchrun 给出明确的失败而不是静默多占。
    if config.renderer.nproc <= 0:
        config.renderer.nproc = detect_gpu_count() or 1
    return config


def _apply(target: object, data: dict) -> None:
    """把 toml 字典合并进 dataclass;未知键忽略并告警式打印。"""
    for key, value in data.items():
        if hasattr(target, key):
            current = getattr(target, key)
            if isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, str):
                value = str(value)
            setattr(target, key, value)
