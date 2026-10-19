#!/usr/bin/env python3
"""Full benchmark matrix for one (backend, world_size) process.

Warmups every (task, duration) shape once (absorbs Triton JIT + cuDNN
autotune + AdaLN/VAE first-use), then measures t2v and i2v at 5/10/15s.
Rank 0 writes output/matrix_<backend>_<world>.json.
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

DURATIONS = (5, 10, 15)
FRAMES = {5: 124, 10: 243, 15: 362}
WIDTH, HEIGHT = 864, 480  # 480p profile (overridable via --width/--height)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("dense", "vsa", "sol"), required=True)
    ap.add_argument("--durations", default="5,10,15")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--tag", default="", help="output filename suffix")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    durations = tuple(int(d) for d in args.durations.split(","))

    adapter = ROOT / "checkpoints/FastH3-4-step-Preview-v1-LoRA" / (
        "vsa-datafree" if args.backend == "vsa" else "dense-datafree"
    ) / "adapter_model.safetensors"

    from h3_runtime.engine import MiniMaxH3Inference

    engine = MiniMaxH3Inference(
        str(ROOT / "checkpoints/MiniMax-H3"),
        str(adapter),
        attention_backend=args.backend,
        width=args.width, height=args.height,
    )
    rank0 = engine.is_rank_zero
    keyframe = ROOT / "bench/i2v_keyframe.png"
    if rank0 and not keyframe.exists():
        raise SystemExit(f"missing {keyframe}: committed i2v first-frame asset")
    dist.barrier()

    # warmup: every (task, duration) shape once — first calls carry the Triton
    # JIT compile (~2.6s per geometry) and VAE first-use; uncounted.
    if not args.no_warmup:
        for task in ("t2v", "i2v"):
            for d in durations:
                h3_vsa_reset(engine)
                t0 = time.perf_counter()
                engine.generate(PROMPT, duration=d, seed=42, width=args.width, height=args.height,
                                image=keyframe if task == "i2v" else None)
                if rank0:
                    print(f"WARMUP {task} {d}s done in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for task in ("t2v", "i2v"):
        for d in durations:
            h3_vsa_reset(engine)
            t0 = time.perf_counter()
            media = engine.generate(PROMPT, duration=d, seed=42, width=args.width, height=args.height,
                                    image=keyframe if task == "i2v" else None)
            wall = time.perf_counter() - t0
            if not rank0:
                continue
            entry = {
                "backend": args.backend, "task": task, "duration_s": d,
                "frames": FRAMES[d], "width": args.width, "height": args.height,
                "wall_s": round(wall, 3),
                "inference_s": round(media.elapsed_s, 3),
                "stages_s": {k: round(v, 3) for k, v in (media.stages or {}).items()},
            }
            results.append(entry)
            print("MATRIX " + json.dumps(entry), flush=True)
            media.save(ROOT / "output" / f"matrix_{args.backend}_{task}_{d}s.mp4")

    if rank0:
        (ROOT / "output").mkdir(exist_ok=True)
        suffix = f"-{args.tag}" if args.tag else ""
        out = ROOT / "output" / f"matrix_{args.backend}{suffix}_{engine.world_size}gpu.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"MATRIX COMPLETE ({args.backend}, {engine.world_size} GPU) -> {out}", flush=True)
    engine.close()
    return 0


def h3_vsa_reset(engine) -> None:
    """Reset the per-request VSA clock if the vsa backend is installed."""
    try:
        from h3_runtime.vsa import h3_vsa
        h3_vsa.reset_clock()
    except Exception:
        pass


PROMPT = (
    "A cinematic wide shot inside a colossal orbital shipyard above a luminous blue "
    "planet. A sleek silver exploration starship slowly launches from a glowing "
    "circular docking ring as coherent blue plasma thrusters ignite; small maintenance "
    "drones drift past while the camera performs a smooth forward dolly."
)

if __name__ == "__main__":
    raise SystemExit(main())
