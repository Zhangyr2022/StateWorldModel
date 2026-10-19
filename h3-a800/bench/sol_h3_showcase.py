#!/usr/bin/env python3
"""Run every Sol-H3 showcase case: 10s, 768p, dense, 8 GPU.

Loads the engine once and iterates the 13 curated prompts (each with its own
seed from the showcase JSON). Rank 0 writes output/showcase_results.json and
output/showcase_<id>.mp4 per case.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CASES = json.loads((ROOT / "bench/sol-h3-prompts.json").read_text(encoding="utf-8"))["cases"]
DURATION = 10


def main() -> int:
    from h3_runtime.engine import MiniMaxH3Inference

    engine = MiniMaxH3Inference(
        str(ROOT / "checkpoints/MiniMax-H3"),
        str(ROOT / "checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"),
        attention_backend="dense",
        width=1344, height=768,
    )
    rank0 = engine.is_rank_zero
    results = []
    for case in CASES:
        cid = case["id"]
        t0 = time.perf_counter()
        media = engine.generate(case["prompt"], duration=DURATION, seed=int(case["seed"]),
                                width=1344, height=768)
        wall = time.perf_counter() - t0
        if not rank0:
            continue
        entry = {
            "id": cid, "title": case["title"], "category": case["category"],
            "language": case["language"], "seed": int(case["seed"]),
            "wall_s": round(wall, 3),
            "inference_s": round(media.elapsed_s, 3),
            "stages_s": {k: round(v, 3) for k, v in (media.stages or {}).items()},
        }
        results.append(entry)
        print("SHOWCASE " + json.dumps(entry, ensure_ascii=False), flush=True)
        (ROOT / "output").mkdir(exist_ok=True)
        media.save(str(ROOT / "output" / f"showcase_{cid}.mp4"))

    if rank0:
        out = ROOT / "output/showcase_results.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        total = sum(r["wall_s"] for r in results)
        print(f"SHOWCASE COMPLETE: {len(results)} cases, {total:.1f}s total wall -> {out}",
              flush=True)
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
