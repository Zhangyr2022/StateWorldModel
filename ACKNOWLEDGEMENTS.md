# Acknowledgements

This monorepo contains four projects. Each keeps its own detailed attribution
file — this document is the consolidated view. External **weights** are used
at runtime and are never redistributed; third-party **libraries** are fetched
from upstream at install time, and adapted third-party code keeps its original
license notices.

| Project | Attribution | Summary |
|---|---|---|
| `agentic_game/` (StatefulHarness) | all original code | Builds on MiniMax-H3, FastH3 LoRA, Qwen3.8-27B (weights only); Gradio + Pillow on the UI side |
| `worldstate_bench/` (WorldState-Bench) | core pipeline + judging original | Baseline wrappers adapted from HunyuanVideo-1.5 / LongCat-Video / Wan2.2 upstreams (details below) |
| `h3-a800/` (MiniMax-H3 runtime) | fork of NVIDIA's Sol-H3 package | Provenance summary in [DESIGN.md → Migration](h3-a800/DESIGN.md); Sol-Attn fetched from upstream (not redistributed); FastVideo VSA semantics port |
| `spatial_proxy/` (clay-model preview) | WildDet3D, cloned at install time (not redistributed) | Allen AI (SAM License; itself vendors a SAM 3 fork and lingbot-depth); Blender/bpy rendering — details in its README → Acknowledgements |

## Borrowed / adapted code

| Component | Original project | Source | Usage | License |
|---|---|---|---|---|
| H3 inference engine (9 files modified, 8 identical) | Sol-H3 runtime in [NVlabs/Sana](https://github.com/NVlabs/Sana) | `sol-engine` branch | Forked, substantially modified for SM80 | Apache-2.0 |
| `h3-a800/h3_runtime/third_party/sol_attn/` | Sol-Attn kernel package in [NVlabs/Sana](https://github.com/NVlabs/Sana) (`sol-engine` @ `8e0db4fa562d`) | Not redistributed: `h3-a800/fetch_sol_attn.sh` fetches it and applies one patch (autotune `num_stages` pinned) | Fetched | Apache-2.0 (nested FlashAttention BSD-3, cuDNN frontend Apache-2.0 — see its `THIRD_PARTY_NOTICES.md`) |
| `h3-a800/h3_runtime/vsa/` | [FastVideo](https://github.com/hao-ai-lab/FastVideo) video sparse attention | Semantics port, rewritten framework-free | Adapted | Apache-2.0 |
| `h3-a800/h3_runtime/vsa/block_sparse_attn_triton.py` | Triton FlashAttention-v2 tutorial | Tutorial kernel + block-sparse wrapper | Modified | MIT |
| `worldstate_bench/baseline/hy15_batch.py` | [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | Batch/i2v loop adapted; Tencent license header preserved verbatim | Adapted | Tencent Hunyuan Community License |
| `worldstate_bench/baseline/{longcat,wan22}_batch.py` | [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) · [Wan2.2](https://github.com/Wan-Video/Wan2.2) | Thin wrappers importing upstream packages (upstream repos are cloned separately, not vendored) | Thin wrapper | upstream licenses — **[author review]: confirm exact upstream licenses** |
| `spatial_proxy/WildDet3D/` | [WildDet3D](https://github.com/allenai/WildDet3D) (Allen Institute for AI) | Cloned from upstream at install time (not redistributed); used via its inference API | External dependency | SAM License |

## External models (weights only, not redistributed)

| Model | Source | License | Used by |
|---|---|---|---|
| MiniMax-H3 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | MiniMax H3 Community License | the frozen video renderer everywhere |
| FastVideo-FastH3-4-step-Preview-v1-LoRA | [FastVideo](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA) | Apache-2.0 | fast t2v/i2v sampling |
| Minimax-h3-Turbo | [lightx2v](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | inherits MiniMax H3 Community License | ref2va rendering |
| Qwen3.8-27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Apache-2.0 | LLM player/kernel, visual observer, benchmark judge & auditor |
| FLUX.2-dev | [black-forest-labs](https://huggingface.co/black-forest-labs/FLUX.2-dev) (gated) | FLUX.2 [dev] Non-Commercial License | benchmark first frames (released dataset images are FLUX.2 outputs — see license note in worldstate_bench ACKNOWLEDGEMENTS) |
| WildDet3D ckpt | [allenai/WildDet3D](https://huggingface.co/allenai/WildDet3D) | SAM License | clay-model detection stage |

## Libraries

diffusers, transformers (Apache-2.0) · PyTorch / torchvision (BSD-3-Clause) ·
Gradio (Apache-2.0) · Pillow (HPND / MIT-CMU) · PyYAML (MIT) ·
Blender / `bpy` 4.2 (GPL; used via the pip module as a separate rendering
runtime).

## External benchmark cited in the paper

[WBench](https://meituan-longcat.github.io/WBench/) (Fudan University & Meituan
LongCat, arXiv:2605.25874, MIT) — the multi-turn interactive-video benchmark
used for the paper's external evaluation (Table 2) and the spatial-preview
channel ablation (Table 8). Its repository and dataset are **not** included in
this release; obtain them from the official sources and point
`WBENCH_ROOT` / `WBENCH_DATA` at them (see `agentic_game/game/wbench/` and
`spatial_proxy/box2clay/`).

## Open attribution items (author review)

- `worldstate_bench/baseline/`: confirm the exact upstream licenses of
  LongCat-Video and Wan2.2.
- FLUX.2: the owner confirmed the FLUX.2 [dev] Non-Commercial License terms
  are acceptable for distributing the generated first frames
  (HF dataset `Louis0411/StateWorldModel`).
- If benchmark design methodology draws on prior work, add conceptual
  citations.
