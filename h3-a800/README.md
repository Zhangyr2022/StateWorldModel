# h3-a800 — MiniMax-H3 Inference on 8×A800 80GB

Inference deployment for **MiniMax-H3** video generation (T2V, I2V first-frame
conditioning, and Ref2VA multi-reference conditioning) targeting
**8×A800 80GB PCIe (SM80)**. It forks NVIDIA's official Sol-H3 8×B300 (SM103)
inference package and rewrites the model-loading and attention paths for SM80
memory and topology. This is the frozen video renderer used by the companion
**StatefulHarness** system (`../agentic_game/`) and WorldState-Bench
(`../worldstate_bench/`) in "State World Models: Execute the Rules, Generate
the Pixels"; see [Citation](#citation).

- **Sharded text encoder.** A full Qwen3-VL conditioner (~60 GB in BF16) does
  not fit on an 80 GB card, so its first 50 decoder layers are spread across
  ranks with hidden states relayed down the rank chain.
- **Three attention backends.** `dense` (SDPA baseline), `vsa` (FastVideo
  FastH3 video-tile top-k sparse attention, ported), and `sol` (the upstream
  Triton Sol-Attn path re-wired for SM80, including a fix for a silent
  strided-input misread in its Triton kernel — fastest at long sequences; the
  Sol-Attn package is fetched from upstream by `fetch_sol_attn.sh`).
- **FastH3 4-step distillation adapter** fused into the weights: 5 s / 480p
  end-to-end in **~5.7 s** on 8 GPUs (dense), **~4.4 s** with the sol backend,
  int8 QKV communication and compiled VAE decode.
- **Ref2VA**: image / video / audio references (dense attention, ref2v turbo
  4-step LoRA).

## Overview

MiniMax-H3 is a 33B DiT video+audio generation model. Each request encodes the
prompt with a Qwen3-VL conditioner (reading `hidden_states[50]`), runs five
scheduler points / four transformer forwards at 8-way Ulysses sequence
parallelism, then decodes video and audio VAEs in parallel and muxes an fMP4
(H.264 + AAC) on rank 0.

Deploying this on 80 GB A800s (vs. the 288 GB B300s the upstream package
targets) required three changes, which are this repository's contributions:

1. **Memory staging** — the text encoder is sharded layerwise across ranks
   (`h3_runtime/text_shard.py`), the DiT loads first and its AdaLN projection
   (~26 GB/rank) is evicted to CPU after precompute, and the VAEs move to the
   GPU lazily at first use. Steady state is ~48–55 GB/rank.
2. **Attention backends** — the upstream `sol_bsa` backend bottoms out in a
   cuDNN block-sparse kernel that requires SM90+, so SM80 gets `dense` (SDPA),
   a ported FastVideo VSA (`h3_runtime/vsa/`), and the upstream `sol` path
   re-wired onto its Triton fallback kernel (`h3_runtime/sparse_attention.py`),
   including the diagnosis and fix of a stride-blind pointer kernel
   (see [DESIGN.md §7.1](DESIGN.md)).
3. **Communication compression** — an opt-in int8 QKV wire format for the
   Ulysses all-to-all (0.52× bytes, sol backend), worth −10–16% end-to-end at
   480p where the all-to-all is the largest single cost (DESIGN.md §7.3).

All measured numbers in the documentation were taken on 8×A800 80GB PCIe at
locked clocks with seed 42; see [DESIGN.md](DESIGN.md) for the full matrices.

## Installation

Tested environment: **Python 3.12, PyTorch 2.10.0+cu130, Triton 3.6.0,
transformers 5.8.1**, NVIDIA driver 595.45.04 (CUDA 13). One GPU per
process via `torchrun`; the engine rejects world sizes other than 4 and 8
(smaller world sizes do not fit in 80 GB).

```bash
git clone https://github.com/Zhangyr2022/StateWorldModel.git
cd StateWorldModel/h3-a800

conda create -n h3-a800 python=3.12
conda activate h3-a800

# One command: torch 2.10.0+cu130 / torchvision 0.25.0+cu130 come from the CUDA 13 index
pip install -r requirements.txt

# Download weights (~219 GB with both tasks; needs ~230 GB free)
python download_checkpoints.py --output-dir ./checkpoints --task both

# Only for --attention-backend sol: fetch the Sol-Attn kernel package (not redistributed)
bash fetch_sol_attn.sh
```

`run.sh` exports the environment variables this deployment needs
(`DISABLE_ADDMM_CUDA_LT=1`, `HF_HUB_OFFLINE=1`); both are overridable from the
environment. `requirements.txt` pins `diffusers` to the exact commit the
modular MiniMax-H3 pipeline was verified against.

## Repository structure

```text
h3-a800/
├── h3_runtime/            # inference runtime
│   ├── engine.py          # MiniMaxH3Inference: load orchestration, generate(), SPMD guards
│   ├── text_shard.py      # layerwise-sharded Qwen3-VL conditioner + rank relay
│   ├── ulysses.py         # Ulysses all-to-all (packed QKV, int8 wire option)
│   ├── sparse_attention.py# sol backend wiring (routing, sink, correctness gate)
│   ├── vsa/               # FastVideo VSA port (core + Triton block-sparse kernel)
│   ├── fusions.py         # Triton elementwise fusions (RMSNorm/RoPE/SwiGLU/AdaLN)
│   ├── relayout.py, comm_quant.py, quant_kernels.py   # layout + int8 wire kernels
│   ├── vae_parallel.py    # context-parallel VAE decode with global tile batching
│   ├── encoding.py        # CUDA → fragmented-MP4 (GPU quantize + pinned-memory chunks)
│   ├── adaln.py, lora.py, cp_plan.py, ...             # loading / fusion / parallelism
│   └── third_party/       # Sol-Attn kernel package, fetched by fetch_sol_attn.sh (not in git)
├── bench/                 # benchmark matrix, VSA E2E, topology probe, showcase runner
├── tests/                 # numerical regression tests (fusions, VSA, sol, int8 wire, relay)
├── prompts/               # prompt assets used by bench and examples
├── infer.py               # CLI entry point (run via run.sh)
├── run.sh                 # torchrun launcher with validated env baked in
├── download_checkpoints.py
├── fetch_sol_attn.sh      # Sol-Attn from NVlabs/Sana (pinned commit) + patches/ (num_stages pin)
└── DESIGN.md              # full architecture, memory layout, measured matrices, env vars
```


## Data preparation

There is no training dataset: this is an inference-only release. The only
"data" step is provisioning model weights (table below; download and
verification commands in the root [README → Models and data](../README.md#models-and-data)).

## Pretrained models and checkpoints

| Checkpoint | Required for | Size |
|---|---|---:|
| `MiniMaxAI/MiniMax-H3` (MiniMax H3 Community License) | everything | ~210 GB |
| `FastVideo-FastH3-4-step-Preview-v1-LoRA` `dense-datafree` | t2v / i2v (dense, sol) | 1.4 GB |
| `FastVideo-FastH3-4-step-Preview-v1-LoRA` `vsa-datafree` | `--attention-backend vsa` | 5.0 GB |
| `lightx2v/Minimax-h3-Turbo` ref2v turbo LoRA | `--task ref2va` | 1.3 GB |

Full instructions, including hash verification:
the root [README → Models and data](../README.md#models-and-data).

## Quick start / inference

`run.sh <nproc> <output.mp4> [infer.py args...]` (nproc: 4 or 8):

```bash
# T2V, 480p, 5 s (dense backend, FastH3 4-step adapter)
./run.sh 8 output/demo_480p.mp4 --task t2v --duration 5 --res 480p --seed 42 \
    --prompt "A cinematic shot of ..."

# I2V: pass a first frame and the request becomes i2v
./run.sh 8 output/demo_i2v.mp4 --task i2v --duration 5 --res 480p --seed 42 \
    --prompt "..." --image first_frame.png

# Ref2VA: --reference is repeatable (image:/video:/audio:); order is meaningful
./run.sh 8 output/demo_ref2va.mp4 --task ref2va --duration 5 --res 480p --seed 42 \
    --prompt "The subject speaks warmly to the camera." \
    --reference image:subject.png --reference audio:voice.wav

# VSA backend (requires the vsa-datafree adapter)
./run.sh 8 output/demo_vsa.mp4 --attention-backend vsa \
    --adapter ./checkpoints/FastH3-4-step-Preview-v1-LoRA/vsa-datafree/adapter_model.safetensors \
    --task t2v --duration 5 --res 768p --seed 42 --prompt-file prompts/orbital_original.txt

# sol backend (Triton Sol-Attn; fastest at long sequences; --warmup absorbs JIT)
./run.sh 8 output/demo_sol.mp4 --attention-backend sol \
    --task t2v --duration 5 --res 480p --seed 42 --prompt-file prompts/orbital_original.txt --warmup

# sol + int8 QKV wire + compiled VAE decode: fastest measured profile (DESIGN §7.3)
H3_ULYSSES_COMM_DTYPE=int8_qkv H3_VAE_COMPILE=max-autotune-no-cudagraphs \
    ./run.sh 8 output/demo_sol_int8.mp4 --attention-backend sol \
    --task t2v --duration 5 --res 480p --seed 42 --prompt-file prompts/orbital_original.txt --warmup
```

Key `infer.py` arguments: `--task t2v|i2v|ref2va`, `--duration 5|10|15`,
`--res 480p|768p` (or `--width/--height` in multiples of 32),
`--attention-backend dense|vsa|sol`, `--adapter`, `--prompt/--prompt-file`,
`--image`, `--reference`, `--seed`, `--warmup`. The full list with semantics is
in [DESIGN.md §3](DESIGN.md).

Python API (SPMD: launch under `torchrun`; every rank must issue identical
`generate()` calls — arguments are CRC-checked across ranks):

```python
from h3_runtime.engine import MiniMaxH3Inference

engine = MiniMaxH3Inference(
    "./checkpoints/MiniMax-H3",
    "./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors",
    attention_backend="dense",   # or "vsa" / "sol"
    width=864, height=480,
)
media = engine.generate("prompt", duration=5, seed=42, image="first.png")  # image optional
media.save("out.mp4")            # non-zero ranks return None
engine.close()
```

## Benchmarks

Headline end-to-end numbers (8×A800, locked 1410 MHz, seed 42, 4-step adapter;
full matrices and 4-GPU numbers in [DESIGN.md §4](DESIGN.md)):

| Configuration | t2v 5s (480p) | t2v 15s (480p) | t2v 5s (768p) |
|---|---:|---:|---:|
| dense | 5.66 s | 20.43 s | 15.86 s |
| sol (Triton) | 5.06 s | 16.83 s | 12.58 s |
| sol + int8 wire + VAE compile | **4.38 s** | **14.82 s** | not measured |

Reproduce with the benchmark scripts (each loads the engine once, warms up,
then measures; `DESIGN.md` documents the exact protocols and caveats):

```bash
torchrun --standalone --nproc_per_node=8 bench/full_matrix.py --backend dense
torchrun --standalone --nproc_per_node=8 bench/full_matrix.py --backend sol
torchrun --standalone --nproc_per_node=8 bench/vsa_e2e.py
torchrun --standalone --nproc_per_node=8 bench/int8_wire_ab.py
```

Numerical regression tests (single GPU unless noted; no checkpoints needed):

```bash
python tests/test_fusions_sm80.py
python tests/test_vsa_kernel_sm80.py
python tests/test_vsa_core.py
python tests/test_sol_triton_sm80.py
torchrun --standalone --nproc_per_node=2 tests/test_int8_wire_sm80.py
torchrun --standalone --nproc_per_node=2 tests/test_text_shard_relay.py
```

## Training

This repository is inference-only; no training code is included. The 4-step
distillation adapters are downloaded as trained artifacts (see the root [README → Models and data](../README.md#models-and-data)).

## Reproducing the measured results

1. Provision weights per the root [README → Models and data](../README.md#models-and-data) and verify hashes.
2. Lock GPU clocks for stable numbers: `nvidia-smi -lgc 1410,1410 -i <0..7>`.
3. Run the `bench/` script for the configuration of interest — every script
   warms up before measuring; single-shot runs without warmup are invalid on
   this stack (JIT/autotune dominates cold requests; see DESIGN.md §7.1).
4. Expected numbers are tabulated in [DESIGN.md §4](DESIGN.md) together with
   the per-stage breakdowns (`text / denoise / video_vae / audio_vae`).

Known reproduction boundaries are documented in DESIGN.md §6: the sol backend
requires `H3_SOL_SYNC=1` (default) and `H3_SOL_PACKED_INPUT=0` (default); VSA ×
768p long-duration decode OOMs; `bench/full_matrix.py` measures each case once
(warm), without median-of-N.

## Acknowledgements

Forked from NVIDIA's Sol-H3 package in [NVlabs/Sana](https://github.com/NVlabs/Sana)
(sol-engine branch); VSA semantics from
[FastVideo](https://github.com/hao-ai-lab/FastVideo); the `sol_attn` kernel
package (fetched from NVlabs/Sana, not redistributed) nests FlashAttention
(BSD-3) and cuDNN frontend (Apache-2.0) derived code. Per-file provenance is documented in
DESIGN.md §Migration (per-file provenance was verified by byte-level
comparison against the upstream tree); license notes in the root
[ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md).

## Citation

This runtime accompanies the paper "State World Models: Execute the Rules, Generate the Pixels"; please cite:

```bibtex
@misc{stateworldmodels2026,
  title        = {State World Models: Execute the Rules, Generate the Pixels},
  author       = {{Chan Yan Ki} and Yanran Zhang and Wenzhao Zheng and Xiaofeng Wang and
                  Zheng Zhu and Lei Chen and Jie Zhou and Jiwen Lu},
  year         = {2026},
  howpublished = {\url{https://ivg-yanranzhang.github.io/StateWorldModel/}}
}
```

If you use it, please also credit the upstream projects above and the
MiniMax-H3 model authors.

## License

This repository is released under [Apache-2.0](LICENSE). It ships no model
weights; the MiniMax-H3 weights this deployment downloads are governed by the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
Third-party code keeps its original notices — see
the root [ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md).
