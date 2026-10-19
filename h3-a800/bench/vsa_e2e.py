#!/usr/bin/env python3
"""VSA end-to-end benchmark: 480p and 768p at 5s on the vsa-datafree adapter.

Run: torchrun --standalone --nproc_per_node=8 bench/vsa_e2e.py
"""

import sys
from pathlib import Path

import json
import time

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from h3_runtime.engine import MiniMaxH3Inference

PROMPT = "A cinematic wide shot inside a colossal orbital shipyard above a luminous blue planet. A sleek silver exploration starship slowly launches from a glowing circular docking ring as coherent blue plasma thrusters ignite; small maintenance drones drift past while the camera performs a smooth forward dolly. Realistic metallic reflections, stable spacecraft geometry, volumetric light, deep space in the background, rich blue and amber color grading, highly detailed, smooth coherent motion, no cuts, no dialogue, no subtitles, no text, no watermark."

engine = MiniMaxH3Inference(
    ROOT / "checkpoints/MiniMax-H3",
    ROOT / "checkpoints/FastH3-4-step-Preview-v1-LoRA/vsa-datafree/adapter_model.safetensors",
    attention_backend="vsa",
    width=864, height=480,
)
rank0 = engine.is_rank_zero
for name, case in [("vsa_480p_5s", dict(duration=5, width=864, height=480)),
                   ("vsa_768p_5s", dict(duration=5, width=1344, height=768))]:
    t0 = time.perf_counter()
    media = engine.generate(PROMPT, seed=42, **case)
    wall = time.perf_counter() - t0
    if not rank0:
        continue
    print("VSA " + json.dumps({
        "case": name, "wall_s": round(wall, 3),
        "stages_s": {k: round(v, 3) for k, v in (media.stages or {}).items()},
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1)}), flush=True)
    (ROOT / "output").mkdir(exist_ok=True)
    media.save(ROOT / "output" / f"{name}.mp4")
if rank0:
    print("VSA E2E COMPLETE", flush=True)
engine.close()
