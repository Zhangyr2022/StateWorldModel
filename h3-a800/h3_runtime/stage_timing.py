"""Per-stage wall-clock accounting for the MiniMax-H3 pipeline on A800.

Wraps the modular pipeline's step classes (`text encoder`, `denoise loop`,
`video decode`, `audio decode`) with synchronize-bounded timers and reports
the breakdown of the last request. Stage boundaries are coarse — four
synchronizations per request — so the measurement itself is cheap.
"""

from __future__ import annotations

import time

import torch


_STAGES = ("text", "denoise", "video_vae", "audio_vae")
_current: dict[str, float] | None = None


def activate() -> None:
    global _current
    _current = {stage: 0.0 for stage in _STAGES}


def reset() -> None:
    """Zero the accumulators so one request's report is not polluted by warmups."""
    global _current
    if _current is None:
        activate()
    else:
        for stage in _STAGES:
            _current[stage] = 0.0


def add(stage: str, seconds: float) -> None:
    if _current is not None:
        _current[stage] = _current.get(stage, 0.0) + seconds


def collect() -> dict[str, float]:
    return dict(_current) if _current is not None else {}


def wrap_step(cls, stage: str) -> None:
    """Wrap one modular step class's `__call__` with a stage timer."""
    original = cls.__call__

    def timed_call(self, components, state, *args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return original(self, components, state, *args, **kwargs)
        finally:
            torch.cuda.synchronize()
            add(stage, time.perf_counter() - started)

    cls.__call__ = timed_call


def install() -> None:
    from diffusers.modular_pipelines.minimax_h3.decoders import (
        MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep,
    )
    from diffusers.modular_pipelines.minimax_h3.denoise import (
        MiniMaxH3DenoiseStep, MiniMaxH3Ref2VADenoiseStep,
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VATextEncoderStep, MiniMaxH3TextEncoderStep,
    )

    wrap_step(MiniMaxH3TextEncoderStep, "text")
    wrap_step(MiniMaxH3Ref2VATextEncoderStep, "text")
    wrap_step(MiniMaxH3DenoiseStep, "denoise")
    wrap_step(MiniMaxH3Ref2VADenoiseStep, "denoise")
    wrap_step(MiniMaxH3VideoDecodeStep, "video_vae")
    wrap_step(MiniMaxH3AudioDecodeStep, "audio_vae")
