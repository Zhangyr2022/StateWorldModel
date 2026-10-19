#!/usr/bin/env python3
"""E2E A/B of the Ulysses QKV wire dtype (bf16 vs int8) on the sol backend.

`_qkv_wire_dtype` reads H3_ULYSSES_COMM_DTYPE per attention call, so one
engine load can measure both wires: warmup (bf16), bf16 baseline per duration,
one unmeasured int8 generation (JIT of the pack/decode kernels), int8
measured, and a bf16 re-measure as a drift control.

Per generation this records the stage timings, the sparse-attention stats
(confirms the wire actually engaged), the per-frame std stats — the
gray-collapse canary of DESIGN §7.1 (broken backend: std ≈ 2.3; healthy:
≈ 53-58) — and dumps sample frames for visual inspection.

Run (from the repo root, with run.sh's env):
  torchrun --standalone --nproc_per_node=8 bench/int8_wire_ab.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPT_FILE = ROOT / "prompts/orbital_original.txt"
SEED = 42


def frame_stats(video: torch.Tensor) -> dict:
    """Per-frame spatial std on the 0-255 scale — DESIGN §7.1's gray-collapse canary
    (broken backend ≈ 2.3, healthy ≈ 53-58). Layout is (B?, T, C, H, W) in [0, 1],
    per encoding.py's `clamp01(TCHW) * 255` contract."""
    if video.ndim == 5:
        video = video[0]
    per_frame_std = video.float().std(dim=(1, 2, 3)) * 255.0   # (T,)
    per_frame_mean = video.float().mean(dim=(1, 2, 3))
    return {
        "frame_std_mean": round(float(per_frame_std.mean()), 3),
        "frame_std_min": round(float(per_frame_std.min()), 3),
        "frame_std_max": round(float(per_frame_std.max()), 3),
        "frame_mean_abs_dev": round(float((per_frame_mean - per_frame_mean.mean()).abs().mean()), 5),
    }


def save_frames(video: torch.Tensor, prefix: Path, indices=(30, 60, 90)) -> None:
    from PIL import Image

    if video.ndim == 5:
        video = video[0]
    for index in indices:
        if index >= video.shape[0]:
            continue
        frame = (video[index].float().mul(255.0).round()).to(torch.uint8)
        frame = frame.cpu().permute(1, 2, 0).numpy()
        Image.fromarray(frame).save(prefix.with_name(f"{prefix.name}_f{index}.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", default="5,15")
    ap.add_argument("--wires", default="bf16,int8_qkv",
                    help="ordered list; each gets its own warmup before its first measured run")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    durations = tuple(int(d) for d in args.durations.split(","))
    wires = [w.strip() for w in args.wires.split(",")]
    for wire in wires:
        if wire not in ("bf16", "int8_qkv"):
            raise SystemExit(f"--wires entries must be bf16 or int8_qkv, got {wire!r}")

    from h3_runtime.engine import MiniMaxH3Inference

    engine = MiniMaxH3Inference(
        str(ROOT / "checkpoints/MiniMax-H3"),
        str(ROOT / "checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"),
        attention_backend="sol",
        width=864, height=480,
    )
    rank0 = engine.is_rank_zero
    prompt = PROMPT_FILE.read_text()
    out_dir = ROOT / "output"
    if rank0:
        out_dir.mkdir(exist_ok=True)

    def generate(wire: str, duration: int, label: str, measured: bool) -> dict | None:
        os.environ["H3_ULYSSES_COMM_DTYPE"] = wire
        t0 = time.perf_counter()
        media = engine.generate(prompt, duration=duration, seed=SEED)
        wall = time.perf_counter() - t0
        if not rank0:
            return None
        entry = {
            "label": label, "wire": wire, "duration_s": duration, "seed": SEED,
            "wall_s": round(wall, 3),
            "inference_s": round(media.elapsed_s, 3),
            "stages_s": {k: round(v, 3) for k, v in (media.stages or {}).items()},
            **frame_stats(media.video),
        }
        sparse = engine.sparse_attention.stats() if engine.sparse_attention else None
        if sparse:
            entry["sol"] = {k: sparse.get(k) for k in (
                "sparse_calls", "dense_calls", "sparse_fraction",
                "qkv_wire_dtype", "qkv_wire_ratio", "route_density",
                "sink_tokens", "sequence_length")}
        print("WIREAB " + json.dumps(entry), flush=True)
        if measured:
            # frames before media.save(): save() hands the buffers to the
            # encoder and nulls them out
            stem = f"int8ab_{wire}_{duration}s{('_' + args.tag) if args.tag else ''}"
            save_frames(media.video, out_dir / stem)
            media.save(out_dir / f"{stem}.mp4")
        return entry

    results = []
    for wire in wires:
        # Warm up at the first duration: the sol kernel autotunes on key=["T"],
        # so a 15s shape that never ran 15s-warmup pays its JIT inside the
        # measurement (the first 15s bf16 run without this measured denoise
        # 33.9 s vs a clean 13.8 s — DESIGN §7.1's "must --warmup" lesson).
        generate(wire, durations[0], f"{wire}-warmup", measured=False)
        for d in durations:
            results.append(generate(wire, d, f"{wire}-{d}s", measured=True))
    # drift control: repeat the first wire's shortest duration at the end
    generate(wires[0], durations[0], f"{wires[0]}-{durations[0]}s-repeat", measured=False)

    if rank0:
        out = out_dir / f"int8ab{('-' + args.tag) if args.tag else ''}_{engine.world_size}gpu.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"WIREAB COMPLETE -> {out}", flush=True)
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
