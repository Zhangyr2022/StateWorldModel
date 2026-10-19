#!/usr/bin/env python3
"""Run one case from bench/sol-h3-prompts.json through the optimized stack.

Defaults to the session's recommended config (whatever the env carries):
H3_ULYSSES_COMM_DTYPE=int8_qkv + H3_VAE_COMPILE=max-autotune-no-cudagraphs
on the sol backend. Warms each resolution's shapes first (sol kernel T-key
JIT + per-shape VAE compile), then generates the case at its official seed
and records stage timings, the frame-std canary, and sample frames.

Run:
  H3_ULYSSES_COMM_DTYPE=int8_qkv H3_VAE_COMPILE=max-autotune-no-cudagraphs \
  torchrun --standalone --nproc_per_node=8 bench/showcase_case.py --case 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESOLUTIONS = {"480p": (864, 480), "768p": (1344, 768)}
WARMUP_PROMPT = "A continuous cinematic shot of morning light moving across a quiet room."


def frame_stats(video: torch.Tensor) -> dict:
    """Per-frame spatial std on the 0-255 scale (DESIGN §7.1 canary)."""
    if video.ndim == 5:
        video = video[0]
    std = video.float().std(dim=(1, 2, 3)) * 255.0
    return {"frame_std_mean": round(float(std.mean()), 2),
            "frame_std_min": round(float(std.min()), 2),
            "frame_std_max": round(float(std.max()), 2)}


def save_frames(video: torch.Tensor, prefix: Path, indices) -> None:
    from PIL import Image

    if video.ndim == 5:
        video = video[0]
    for index in indices:
        if index >= video.shape[0]:
            continue
        frame = (video[index].float().mul(255.0).round()).to(torch.uint8)
        Image.fromarray(frame.cpu().permute(1, 2, 0).numpy()).save(
            prefix.with_name(f"{prefix.name}_f{index}.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=1, help="1-based case index")
    ap.add_argument("--resolutions", default="480p,768p")
    ap.add_argument("--durations", default="10", help="seconds per resolution")
    args = ap.parse_args()

    cases = json.loads((ROOT / "bench/sol-h3-prompts.json").read_text())["cases"]
    case = cases[args.case - 1]
    duration = int(args.durations.split(",")[0])

    from h3_runtime.engine import MiniMaxH3Inference

    engine = MiniMaxH3Inference(
        str(ROOT / "checkpoints/MiniMax-H3"),
        str(ROOT / "checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"),
        attention_backend="sol",
        width=864, height=480,
    )
    rank0 = engine.is_rank_zero
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    results = []

    for res_label in args.resolutions.split(","):
        width, height = RESOLUTIONS[res_label]
        # per-shape warmup at the CASE's own duration: the sol kernel autotunes
        # on key=["T"] and torch.compile(dynamic=False) recompiles per tile-batch
        # shape — a 5s warmup before a 10s case pays ~155 s of JIT inside the
        # measurement (measured: 20 s kernel JIT + 135 s VAE recompile).
        engine.generate(WARMUP_PROMPT, duration=duration, seed=0, width=width, height=height)
        if rank0:
            print(f"[case{args.case}] {res_label} warmup done", flush=True)
        t0 = time.perf_counter()
        media = engine.generate(case["prompt"], duration=duration, seed=int(case["seed"]),
                                width=width, height=height)
        wall = time.perf_counter() - t0
        if not rank0:
            continue
        entry = {
            "case": case["id"], "res": res_label, "duration_s": duration,
            "seed": int(case["seed"]),
            "wire": __import__("os").environ.get("H3_ULYSSES_COMM_DTYPE", "bf16"),
            "vae_compile": __import__("os").environ.get("H3_VAE_COMPILE", ""),
            "wall_s": round(wall, 2),
            "inference_s": round(media.elapsed_s, 2),
            "stages_s": {k: round(v, 2) for k, v in (media.stages or {}).items()},
            **frame_stats(media.video),
        }
        print("CASE " + json.dumps(entry, ensure_ascii=False), flush=True)
        results.append(entry)
        stem = f"showcase_case{args.case}_{res_label}"
        save_frames(media.video, out_dir / stem, (30, 120, 240))
        media.save(out_dir / f"{stem}.mp4")
        del media

    if rank0:
        out = out_dir / f"showcase_case{args.case}.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"CASE COMPLETE -> {out}", flush=True)
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
