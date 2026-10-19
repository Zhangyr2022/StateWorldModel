"""配置加载：config.yaml 是路径与协议参数的唯一来源。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _dig(d: dict, *keys, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


@dataclass
class Config:
    raw: dict
    root: Path
    overrides: dict = field(default_factory=dict)

    def _path(self, key: str, default: str) -> Path:
        if key in self.overrides:
            return Path(self.overrides[key])
        return self.root / _dig(self.raw, "paths", key, default=default)

    # ---- paths ----
    @property
    def specs_dir(self) -> Path:
        return self._path("specs_dir", "specs")

    @property
    def out_yaml(self) -> Path:
        return self._path("out_yaml", "out/yaml")

    @property
    def out_frames(self) -> Path:
        return self._path("out_frames", "out/frames")

    @property
    def out_cases(self) -> Path:
        return self._path("out_cases", "out/cases")

    @property
    def data_cases(self) -> Path:
        """随仓库分发的打包数据集快照(package 后刷新,评测与 baseline 默认读这里)。"""
        return self._path("data_cases", "data/cases")

    @property
    def out_work(self) -> Path:
        return self._path("out_work", "out/work")

    # ---- envs ----
    @property
    def build_python(self) -> str:
        return _dig(self.raw, "envs", "build_python",
                    default="/usr/bin/python3")

    @property
    def render_python(self) -> str:
        # 缺键即报错,不再静默回落源机路径(那只会让子进程在别处莫名失败)
        value = _dig(self.raw, "envs", "render_python")
        if not value:
            raise RuntimeError("config.yaml 缺少 envs.render_python")
        return str(value)

    @property
    def review_python(self) -> str:
        value = _dig(self.raw, "envs", "review_python")
        if not value:
            raise RuntimeError("config.yaml 缺少 envs.review_python")
        return str(value)

    # ---- render ----
    @property
    def render(self) -> dict:
        return self.raw.get("render", {})

    @property
    def gpu_pairs(self) -> list[str]:
        return [str(p) for p in _dig(self.raw, "render", "gpu_pairs",
                                     default=["0,1", "2,3", "4,5", "6,7"])]

    # ---- review ----
    @property
    def review(self) -> dict:
        return self.raw.get("review", {})

    @property
    def review_gpus(self) -> list[int]:
        return [int(g) for g in _dig(self.raw, "review", "gpus",
                                      default=[0, 1, 2, 3])]

    # ---- cycle ----
    @property
    def cycle(self) -> dict:
        return self.raw.get("cycle", {})


def load_config(path: Path | None = None) -> Config:
    p = path or (ROOT / "config.yaml")
    raw = yaml.safe_load(open(p, encoding="utf-8")) if p.exists() else {}
    return Config(raw=raw, root=p.resolve().parent)
