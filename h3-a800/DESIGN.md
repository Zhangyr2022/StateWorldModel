# h3-a800 Design Document

MiniMax-H3 (FL2VA partition, T2V/I2V/Ref2VA) inference on **8 × A800 80GB
PCIe (SM80)**. Forked from NVIDIA's official Sol-H3 8×B300 inference package
(NVlabs/Sana, `sol-engine` branch, `models/minimax_h3/Sol-H3`), with the
loading and attention paths rewritten for SM80 memory and topology.

- Upstream baseline: `Sol-H3/h3_runtime` (validated on B300/SM103;
  dense-datafree + SOL/BSA + int8/fp8 wire)
- This fork's core changes: layerwise text-encoder sharding (80 GB cannot hold
  the full model), attention backend replacement (**dense / VSA / sol** —
  `sol` is the upstream Triton Sol-Attn path re-wired for SM80, with a fix for
  the `triton_ref` kernel's strided-input bug; fastest at long sequences,
  see §7.1; `sol_bsa` remains refused), resolution parameterization, and a
  complete port of VSA (FastVideo FastH3 90pct/tile64)
- Task scope: **T2V + I2V** (first-frame conditioning) + **Ref2VA**
  (migrated 2026-09-19: `transformer_ref` partition, image/video/audio
  references, dense attention, ref2v turbo 4-step LoRA; see the Ref2VA migration notes below)

---

## 1. System Architecture

### 1.1 Pipeline

```
prompt (+ first-frame image)
  → text conditioning: Qwen3-VL conditional encoder, reading hidden_states[50]
      (layerwise sharding: each rank holds 6–7 layers; hidden states relay
      along the rank chain via send/recv + final-rank broadcast)
  → DiT denoising: 33B transformer, 5 schedule points / 4 forwards
      (FastH3 4-step distillation adapter fused into the weights)
      8-rank Ulysses sequence parallelism (one varlen all-to-all for Q/K/V
      and one for the gate)
  → video VAE decode (tile sharding + global batching) ‖ audio VAE decode
  → fMP4 muxing (rank 0: GPU quantization + chunked H.264 + AAC)
```

### 1.2 Memory layout (per rank, BF16)

| Component | Resident | Notes |
|---|---:|---|
| DiT (after AdaLN precompute) | ~37 GB | `adaln_proj` (26 GB) is evicted to CPU right after loading and moved back block-by-block during precompute; modulation table 0.22 GB |
| Text-encoder slice | 6.3–10.2 GB | the first 50 layers split contiguously across ranks; rank 0 additionally holds `embed_tokens` + the vision tower |
| VSA gate Linear (`vsa` backend only) | ~3.9 GB | 50 × Linear(5376→7168); created on CPU, driven onto the GPU block by block, then weights copied in |
| Video/audio VAE | lazy | moved to the GPU at first decode/encode (after the AdaLN eviction frees headroom) |
| Steady-state total | ~48–55 GB | safe margin on 80 GB; measured peak (incl. activations) 71.1 GB |

### 1.3 Attention backends

| Backend | Description | Status |
|---|---|---|
| `dense` (default) | PyTorch SDPA (flash backend), exact attention over the full sequence | ✅ baseline path; every measured matrix covers it |
| `vsa` | FastVideo FastH3 VSA: video-tile top-k (default keep 10%, tile=64), prefix-key exemption, learned gate compression branch, Triton block-sparse kernel | ✅ E2E working; quality on par with dense; speed in §3.2 (note: §4.1) |
| `sol` | Triton Sol-Attn (the `sol_attn` package, fetched by `fetch_sol_attn.sh`, auto-falls-back to its `triton_ref` kernel on SM80): threshold routing + KV sink + full re-dense of prefix queries; wiring identical to upstream (`sparse_attention.py`). Fixed the `triton_ref` strided-input bug (§7.1); this fork materializes contiguous inputs by default | ✅ usable: wins over dense across the whole 480p matrix (denoise −4.6% to −16.1%); visual quality on par with dense (§7.1); **must keep `H3_SOL_SYNC=1` (default) and `H3_SOL_PACKED_INPUT=0` (default)** |
| ~~sol_bsa~~ | cuDNN BSA **requires SM90–SM120** (verified with nvidia-cudnn-frontend 1.28.0 installed: it raises directly on SM80; the API has an architecture guard). SM80 is covered by `sol`'s Triton path. Selecting it raises explicitly | excluded (architectural, not workaroundable) |

Key VSA design points:
- **Prefix rows (text|cond|audio) are semantically dense** and go straight to
  flash SDPA (in the packed order the prefix rows are naturally contiguous);
  only video rows take the tile scatter + block-sparse kernel — the kernel's
  longest program drops from 544 blocks to ~56
- Video queries: full prefix-key exemption + per-tile top-k on pooled-Q/K
  scores (`ceil(0.1 × n_video)` video tiles)
- The gate compression branch acts on **all rows** (prefix included), matching
  FastVideo semantics; zero gates degenerate to pure sparse
- Selection and execution happen after the Ulysses all-to-all (full sequence ×
  this rank's heads): the topology is per-head, no cross-rank agreement needed
- `dense_first` policy: step 1 and the first 2 layers are forced fully dense
  (`H3_VSA_DENSE_FIRST_STEPS/LAYERS`)

### 1.4 Text-encoder sharding

- Only the first 50 layers of the Qwen3-VL conditional encoder participate
  (`hidden_states[50]`, the output of layer 49); the layer tail and the
  lm_head are never loaded
- Contiguous layer segments `[(0,7),(7,13),...]` live on the 8 ranks; rank 0
  additionally holds `embed_tokens` + the vision tower (i2v needs it)
- Everything besides the weights (token ids, causal mask, M-RoPE, vision
  merge) is recomputed locally on every rank from identical inputs; hidden
  states and deepstack features relay along a dedicated process group chain
  (separate from WORLD, avoiding the NCCL eager-mode P2P×collective deadlock)
- Load-time calibration gate (default on): full-model GPU reference forward
  vs. sharded relay, both text and vision paths must match with max_abs=0

### 1.5 SPMD contract

Every pipeline block runs **locally on all ranks** with the same inputs
(setup, keyframe VAE, token construction are all local); only encoder tensors
are broadcast internally. `generate()` enforces an all_gather signature
(duration/seed/width/height/has_image + prompt CRC + image CRC) — any rank
seeing a different python-level request raises immediately, preventing
"one rank missed the image → local layout row count ≠ broadcast embeds row
count" from exploding as an `index_copy` misalignment deep inside the DiT.

---

## 2. Environment

| Item | Value |
|---|---|
| Python / PyTorch | 3.12, torch 2.10.0+cu130, triton 3.6.0, transformers 5.8.1, av, safetensors (`requirements.txt`) |
| diffusers | pinned at `@abc5e9bf` in `requirements.txt` (the MiniMax-H3 modular pipeline) |
| checkpoints | `checkpoints/MiniMax-H3` (~210 GB) + `checkpoints/FastH3-4-step-Preview-v1-LoRA/{dense-datafree, vsa-datafree}` + `checkpoints/Minimax-h3-Turbo`; see the root [README → Checkpoints](../README.md#checkpoints-and-models) |
| driver | 595.45.04 (CUDA 13); lock clocks before benchmarking: `nvidia-smi -lgc 1410,1410 -i <0..7>` |

### Required environment variables (`run.sh` bakes them in)

| Variable | Reason |
|---|---|
| `DISABLE_ADDMM_CUDA_LT=1` | this stack's torch cu130 cuBLASLt fails every bias-carrying `F.linear` (`h3_runtime/__init__.py` also sets it as a fallback) |
| `HF_HUB_OFFLINE=1` | everything is local; offline mode avoids hub probing hangs (direct huggingface.co may be unreachable). To go online, `HF_ENDPOINT=https://hf-mirror.com` works as a mirror |

---

## 3. Usage

### 3.1 Command line (recommended)

```bash
# from the repository root

# T2V, 480p, 5 s (default dense backend, default adapter = dense-datafree)
./run.sh 8 output/demo_480p.mp4 \
    --task t2v --duration 5 --res 480p --seed 42 \
    --prompt "A cinematic shot of ..."

# I2V: first-frame conditioning (--image drives it; --task i2v is a hint for arg checks)
./run.sh 8 output/demo_i2v.mp4 --task i2v --duration 5 --res 480p --seed 42 \
    --prompt "..." --image first_frame.png

# VSA backend (must pair with the vsa-datafree adapter)
./run.sh 8 output/demo_vsa.mp4 --attention-backend vsa \
    --adapter ./checkpoints/FastH3-4-step-Preview-v1-LoRA/vsa-datafree/adapter_model.safetensors \
    --task t2v --duration 5 --res 768p --seed 42 --prompt-file prompts/orbital_original.txt

# SOL backend (Triton Sol-Attn; fastest at long sequences; --warmup absorbs JIT;
# H3_SOL_SYNC=1 and H3_SOL_PACKED_INPUT=0 are the defaults — do not change them, see §7.1)
./run.sh 8 output/demo_sol.mp4 --attention-backend sol \
    --task t2v --duration 5 --res 480p --seed 42 --prompt-file prompts/orbital_original.txt --warmup

# sol + int8 QKV communication compression (480p favorite: denoise another −13~16%, §7.3;
# lossy but triple-verified to the same quality tier as bf16). Stacked with VAE compile
# (decode −30%), 480p@5s E2E reaches 4.38 s (−19% vs baseline)
H3_ULYSSES_COMM_DTYPE=int8_qkv H3_VAE_COMPILE=max-autotune-no-cudagraphs \
    ./run.sh 8 output/demo_sol_int8.mp4 --attention-backend sol \
    --task t2v --duration 5 --res 480p --seed 42 --prompt-file prompts/orbital_original.txt --warmup

# 768p (native resolution)
./run.sh 8 output/demo_768p.mp4 --task t2v --duration 5 --res 768p --seed 42 --prompt "..."
```

`run.sh <nproc> <output.mp4> [infer.py args...]`: nproc only accepts 4/8;
`--model`/`--adapter`/`--output` have built-in defaults — an explicitly passed
flag always wins.

### 3.2 `infer.py` arguments

| Argument | Values | Notes |
|---|---|---|
| `--task` | t2v / i2v / ref2va | i2v requires `--image`; ref2va requires `--reference` |
| `--duration` | 5 / 10 / 15 s | 124 / 243 / 362 frames @24 fps |
| `--res` | 480p / 768p | 864×480 or 1344×768; or `--width/--height` (multiples of 32, > 0) |
| `--seed` | integer | same seed + same request reproduces |
| `--attention-backend` | dense / vsa / sol | vsa must pair with the vsa-datafree adapter (missing gates raise explicitly); sol pairs with dense-datafree; ref2va is dense-only |
| `--adapter` | safetensors path | dense-datafree (dense/sol) or vsa-datafree (vsa) or ref2v turbo (ref2va) |
| `--reference-image-resize-mode` | match / diffusers | ref2va image sizing: `match` (never upscale, area capped to the canvas; default) or `diffusers` (official 2048-short-edge parity) |
| `--prompt / --prompt-file / --image / --reference / --warmup` | | mutual exclusion and existence validated before loading |

### 3.3 Python API

```python
from h3_runtime.engine import MiniMaxH3Inference

engine = MiniMaxH3Inference(
    "./checkpoints/MiniMax-H3",
    "./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors",
    attention_backend="dense",   # or "vsa" (needs the vsa-datafree adapter) / "sol"
    width=864, height=480,
)
media = engine.generate("prompt", duration=5, seed=42, image="first.png")  # image optional
media.save("out.mp4")            # non-zero ranks return None
engine.close()
```

All `generate()` arguments (including prompt text and image content) are
enforced consistent across ranks (CRC-signature all_gather check);
mismatches raise immediately, and a single rank's pre-check failure is
aligned so every rank raises in sync (no collective hangs).

### 3.4 Environment variables (all defaulted; none required)

| Variable | Default | Notes |
|---|---|---|
| `H3_VSA_SPARSITY` | 0.9 | video-tile keep rate = `ceil((1-sparsity) × n_video_tiles)` |
| `H3_VSA_EXEMPT` | 1 | prefix-key exemption (0 = compete: prefix and video keys share the budget) |
| `H3_VSA_DENSE_FIRST_STEPS` | 1 | first N denoising steps forced fully dense |
| `H3_VSA_DENSE_LAYERS` | 2 | first N layers forced fully dense |
| `H3_VAE_COMPILE` | off | torch.compile for video decode (not bitwise-identical) |
| `H3_A800_TEXT_VERIFY` | 1 | load-time sharding calibration gate |
| `H3_VSA_PROFILE` | 0 | per-segment timing of `vsa_attention` (stdout) |
| `H3_A800_STAGGERED_TEXT` | 0 | rank-by-rank serial text-encoder loading (saves memory, slower) |
| `H3_A800_DISABLE_FUSIONS` | 0 | disable all Triton fusions (troubleshooting) |
| `H3_FUSION_WARPS_BASE/2048/8192` | 4/8/16 | fusion-kernel warp tiers (B300-tuned values; recalibrated on A800) |
| `H3_VAE_BATCHED / H3_VAE_GLOBAL_BATCH` | 1/1 | VAE tile-batching switches |
| `H3_SOL_SYNC` | **1** | `cuda.synchronize` after every sol attention call. Keep it on: in async mode ~15 ms of CPU overhead per call (dense's inline SDPA is ~0.3 ms) slows the 8-rank pipeline to ~2.5× dense (§7.1). Set 0 only for experiments |
| `H3_SOL_PACKED_INPUT` | **0 (changed in this fork)** | the triton_ref pointer kernel's address math ignores strides; feeding it a packed [Q\|K\|V] view silently reads the wrong data (rel_l2 ≈ 1.2, §7.1). Default materializes contiguous copies (~1–2 ms/call). Set 1 only for the CuTe path (SM100+) |
| `H3_SOL_DUMP_QKV` | off | export real QKV (gate + two deep calls, rank 0) for offline tau/threshold sweeps |
| `H3_SOL_CALL_STATS` | 0 | per-call CPU-side timing, printed at `close()` (diagnostics) |
| `H3_SOL_TAU / H3_SOL_SINK_MODE / SOL_ATTN_FIRST_DENSE_*` | 1.0 / prefix / 1,2 | SOL routing policy (matches the upstream-validated A100 configuration) |
| `H3_ULYSSES_COMM_DTYPE` | bf16 | Ulysses QKV wire format. `int8_qkv` (**sol backend only**, unlocked and verified 2026-09-12): the all-to-all QKV record shrinks 768 B→400 B per row·head (wire ratio 0.52); the receive side decodes straight into three contiguous tensors via `dequantize_qkv_split` (incidentally replacing the bf16 path's 3× contiguous copies). Measured at 480p: denoise −13~−16.5%, E2E −10~12% (§7.3); lossy (wire noise ≈ ⅓ of the routing approximation), quality triple-verified to the same tier as bf16. dense/vsa backends still refuse it |
| `H3_A800_ALLOW_SOL` | 0 | no consumer (historical leftover) |

### 3.5 Tests and benchmarks

```bash
python tests/test_fusions_sm80.py        # 4 Triton fusion kernels vs fp32 reference
python tests/test_vsa_kernel_sm80.py     # vendored block-sparse kernel numerics + throughput
python tests/test_vsa_core.py            # VSA attention core (tile/mask/gate/untile)
python tests/test_sol_triton_sm80.py     # SOL Triton kernel: dispatch/numerics/sink/route-density/throughput gates
torchrun --standalone --nproc_per_node=2 tests/test_int8_wire_sm80.py    # int8 QKV wire
torchrun --standalone --nproc_per_node=2 tests/test_text_shard_relay.py  # sharding relay calibration
torchrun --standalone --nproc_per_node=8 bench/topo_probe.py              # NCCL topology bandwidth
torchrun --standalone --nproc_per_node=8 bench/full_matrix.py --backend dense   # benchmark matrix (t2v+i2v × 5/10/15s, warmed)
torchrun --standalone --nproc_per_node=8 bench/full_matrix.py --backend sol     # sol benchmark matrix
torchrun --standalone --nproc_per_node=8 bench/vsa_e2e.py                # VSA benchmark (480p+768p)
torchrun --standalone --nproc_per_node=8 bench/int8_wire_ab.py           # int8 wire A/B (both wires, one process + frame-std canary)
```

---

## 4. Measured results (8×A800, locked 1410 MHz, seed 42, FastH3 4-step)

### 4.1 Full benchmark matrix (t2v/i2v × dense/vsa × 5/10/15s × 8/4 GPUs, `bench/full_matrix.py`)

Each (backend, GPU-count) process loads once, runs 6 warmups (absorbing
Triton JIT / cuDNN / AdaLN / first VAE use), then measures 6 cases.
480p = 864×480; the 4-GPU rows use per-tile VAE decode
(`H3_VAE_GLOBAL_BATCH=0` — global batching OOMs at 4 GPUs).

**8 GPUs (Ulysses degree 8)**

| Case | dense E2E | dense denoise | vsa E2E | vsa denoise |
|---|---:|---:|---:|---:|
| t2v 5s | **5.66s** | 4.18 | 6.80s | 5.35 |
| t2v 10s | **12.22s** | 9.54 | 13.38s | 10.69 |
| t2v 15s | **20.43s** | 16.48 | 21.44s | 17.50 |
| i2v 5s | **6.28s** | 4.42 | 7.73s | 5.87 |
| i2v 10s | **12.98s** | 9.92 | 14.65s | 11.58 |
| i2v 15s | **21.42s** | 17.08 | 22.90s | 18.57 |

**4 GPUs (Ulysses degree 4, per-tile VAE decode)**

| Case | dense E2E | dense denoise | vsa E2E | vsa denoise |
|---|---:|---:|---:|---:|
| t2v 5s | **9.04s** | 6.41 | 10.30s | 7.68 |
| t2v 10s | **20.87s** | 15.74 | 21.66s | 16.51 |
| t2v 15s | **36.10s** | 28.45 | 35.85s | 28.21 |
| i2v 5s | **9.80s** | 6.77 | 11.42s | 8.39 |
| i2v 10s | **21.83s** | 16.31 | 23.05s | 17.54 |
| i2v 15s | **37.37s** | 29.35 | 37.64s | 29.62 |

Readings:
- 8→4 GPU scaling (8→4 cards): dense 5s 1.6×, 15s 1.77× — communication and
  decode-sharding gains decay as sequences lengthen
- VSA vs dense: +1.1~1.5 s at 8 GPUs, +0.3~1.3 s at 4 (VSA wins by 0.25 s on
  15s t2v) — the sparse kernel's 3× attention win is diluted by the
  non-attention GEMMs (~9 s, unchanged by sparsity) and the per-layer
  tile/gather/map overhead (~2–3 ms × 200 calls); the longer the sequence
  (the higher attention's share), the smaller VSA's gap
- i2v vs t2v: +0.5~1.4 s (vision tower + conditioning rows + keyframe encode)
- 768p re-measured (post-fix regression): **t2v 5s median 15.52s** (denoise
  13.02s), matching the pre-fix 15.47s
- **The sol backend's comparison matrix is in §7.1** (after the triton_ref
  strided-input fix: fastest at every duration, visual quality on par with
  dense, now a supported default — but keep `H3_SOL_SYNC=1` and
  `H3_SOL_PACKED_INPUT=0`)

**768p (1344×768) dense full rows** (8 GPUs; `H3_VAE_GLOBAL_BATCH=0`
per-clip decode — global-batch decoding OOMs at 768p for 10s/15s, and
per-clip `video_vae` costs ~2× the global batch):

| Case | E2E | text | denoise | video_vae | audio_vae |
|---|---:|---:|---:|---:|---:|
| t2v 5s | **15.86s** | 0.08 | 13.11 | 2.55 | 0.08 |
| t2v 10s | **40.93s** | 0.08 | 35.56 | 5.06 | 0.14 |
| t2v 15s | **76.42s** | 0.09 | 68.33 | 7.58 | 0.20 |
| i2v 5s | **17.74s** | 0.46 | 14.15 | 2.53 | 0.08 |
| i2v 10s | **43.48s** | 0.45 | 37.28 | 5.05 | 0.14 |
| i2v 15s | **79.48s** | 0.45 | 70.62 | 7.60 | 0.20 |

- 768p attention grows quadratically: 15s (91 latent frames, ~92k tokens)
  denoise is 5.2× the 5s (38k tokens) figure
- vs global batching: per-clip is ~1.3 s slower at 5s (2.55 vs 1.26), but the
  global batch OOMs at 10s/15s — 768p ≥10s must decode per-clip (or shrink
  the global batch's tile grouping; not done)
- i2v 768p 10s across all 13 official showcase prompts: §4.2

### 4.2 Showcase sweep (Sol-H3 official 13 prompts, 768p 10s, dense, 8 GPUs)

The 13 cases in `bench/sol-h3-prompts.json` (each with its official seed,
covering seven languages — zh/en/ar/de/ko/es/it — and photoreal/fantasy/anime
styles), run back-to-back with one load by `bench/sol_h3_showcase.py`:

- Total wall **552s** (mean 42.5 s/case incl. blocking H.264 encode);
  `output/showcase_<id>.mp4` × 13
- Details in `output/showcase_results.json`

### 4.3 VSA component performance (torch.profiler + segmented timing, locked clocks)

| Item | Time |
|---|---:|
| sparse kernel (video rows only, 10% keep, steady state) | 4.6–6.3 ms |
| sparsity sweep | keep 1.0 → 35.7 ms; 0.5 → 18.3 ms; 0.1 → 4.62 ms |
| prefix-row flash SDPA (985 rows × 33241 keys) | ~0.7 ms |
| video tile ×3 + pool/scores/mask + map_to_index + untile + gate | ~3 ms |
| Triton JIT first compile | ~2.6s (480p) / ~6.9s (768p), once per process per geometry |
| text-shard relay (whole request) | ≤0.28s |

### 4.4 Topology bandwidth (`bench/topo_probe.py`, world-8)

| Path | Bandwidth |
|---|---:|
| intra-pair NVLink (0↔1 etc.) all-to-all | 152 GB/s/rank |
| cross-pair quad (0,1,2,7) all-to-all | 36.7 GB/s/rank |
| world-8 all-to-all | 11.6 GB/s/rank |
| world-8 all-gather | 66 ms / 196 MB |

Serving corollary: two 4-GPU instances (one NVLink pair each) outperform one
8-GPU instance in throughput.

---

## 5. Design Constraints and Notes

1. **SPMD**: all ranks must pass the identical python-level request (prompt/
   image content included in the CRC); encoder tensors broadcast internally,
   everything else recomputed locally.
2. **Load order is fixed**: the text shim registers first (blocking the 60 GB
   full load) → DiT to GPU → AdaLN fuse → AdaLN evict → gate create/copy
   (VSA) — only the post-eviction memory fits the i2v keyframe encode.
3. **NCCL**: the text relay uses a dedicated process group (avoids the eager
   P2P×collective deadlock); a single rank's pre-check failure is aligned via
   all_gather so all ranks raise together.
4. **Component indexes resolve through `--model`**: the engine passes
   `pretrained_model_name_or_path=model_path` to `load_components`, so a
   checkpoint tree whose `model_index.json`/`modular_model_index.json` still
   embed absolute paths from a previous machine loads correctly from its new
   location.
5. **world_size is 4 or 8 only**: 1/2 GPUs cannot fit (DiT + text slice
   > 80 GB); the constructor rejects explicitly.
6. **sol_bsa is architecturally excluded; sol is fixed and usable** (§7.1):
   sol_bsa's cuDNN BSA API verifiably requires SM90–SM120 (frontend installed;
   raises on SM80); `sol` works via its Triton kernel and is fastest at long
   sequences, but **two defaults must not change**: `H3_SOL_SYNC=1` (async CPU
   overhead drags the pipeline) and `H3_SOL_PACKED_INPUT=0` (the triton_ref
   pointer kernel ignores strides; packed views silently produce wrong
   attention). Extra optional deps: `nvidia-cudnn-frontend==1.28.0` +
   `nvidia-cutlass-dsl==4.7.1` (installable from mirrors; they only turn
   sol_bsa's error from ModuleNotFoundError into the explicit architecture
   message).
7. **VSA JIT warmup**: first call ~2.6–6.9s (once per process per geometry);
   serving should use `--warmup`.

## 6. Known Boundaries

- Ref2VA migrated (2026-09-19) but dense-attention only: its sparse routing
  (`text_audio` sink) was only validated on cuDNN BSA, unreachable on SM80;
  the 480p/768p long durations and multi-reference combinations have not run
  a full matrix.
- VSA 480p/10s and higher-sparsity steady-state benchmarks incomplete; VSA's
  quality claim rests on single-case visual inspection + FastVideo's official
  evaluation; no quantitative SSIM suite exists.
- `bench/full_matrix.py` measures each case once (after warmup), no
  median-of-N.
- `map_to_index`'s `.item()` sync currently keeps VSA out of CUDA graph /
  torch.compile.
- **VSA × 768p: denoise runs at every duration; video decode OOMs at long
  durations (recorded bug, unfixed)**. With per-clip decode
  (`H3_VAE_GLOBAL_BATCH=0`), t2v 5/10/15s and i2v 5/10s run through
  (warm denoise 42.7/59.7/116.4s incl. JIT); **i2v 15s OOMs in decode** —
  a 4.18 GiB contiguous allocation fails with 7.99 GB reserved-but-unallocated
  (fragmentation), the VSA resident set (gate Linear 3.9 GB + tile scratch
  ~180 MB) squeezing the decode headroom. Candidate fixes (unverified):
  `PYTORCH_ALLOC_CONF=expandable_segments:True`; release h3_vsa tile scratch
  (`RequestState._buffers`) or temporarily unpin the gate Linear before
  decode (§7.2).
- sol × 768p@15s not re-measured (an allocator-fragmentation OOM was recorded
  once: 68.6 GB used + 7.6 GB reserved-unused); likely needs
  `expandable_segments` + `H3_VAE_GLOBAL_BATCH=0` together.
- sol's per-layer routing approximation error is ~8% (tau=1.0; acceptable in
  E2E visual checks); quality-sensitive use can lower `H3_SOL_TAU` for denser
  routing (speed win shrinks accordingly).
- sol diagnostics (real-data gates / route-deviation comparison /
  `H3_SOL_DUMP_QKV`) are kept.

## 7. Future Extensions (unscheduled; the criteria here decide whether to do them)

### 7.1 SOL/Triton attention backend

**Origin and starting state.** NVlabs/Sana's `sol-engine` branch carries
`techniques/sparse_backends/sol_attn/` as an installable `sol_attn` package:
CuTe-specialized kernels for sm100/sm90/sm120/sm89, and **every other
capability ≥ 8 device falls back to `triton_ref/`** (`fwd.py`, 475 lines of
Triton flash + `preprocess.py`; SM80 has no TMA and takes the pointer path),
dispatched by `interface.get_sol_attn_backend()` on
`torch.cuda.get_device_capability`.

**Upstream had already validated Triton Sol-Attn on 4×A100 (SM80)**:
`models/minimax_h3/A100/SOURCE_SNAPSHOT.json` records
`"sm": "80", "sol_attn_backend": "triton"`; the README documents tau=1.0,
exact threshold, full-prefix KV sink, dense prefix queries, first 10 steps /
first 2 blocks fully dense. Its 768p@5s 217→61s (3.55×) is the **sum** of
SGLang FSDP + fused kernels + FirstBlockCache + Sol-Attn; the repo has no
sol-only ablation. Correction of this fork's earlier reasoning: the "cuDNN BSA
unverified below SM103" claim that removed sol from the menu only holds for
the BSA path; the Triton path had upstream precedent on SM80. This fork's
`sparse_attention.py` wiring (routing, sink, centroid-residual merge,
correctness gate) was fully preserved — only the kernel package itself was
missing.

**Expected-win model** (why this can beat VSA, and where it stops winning):

- Attention's share of per-rank denoise compute scales with S (attention
  FLOPs grow as S², GEMMs as S): 480p@5s (S≈51k rows) ≈10% → 480p@15s ≈20% →
  768p@15s (S≈367k) ≈30–35%. §4.1's VSA measurements (480p@5s +28% →
  parity at 480p@15s/4 GPUs) fit that model — **sparse attention can only net
  a win in the long-sequence regime**; at ≤10s/480p, dense is the end state.
- SOL's structural overhead advantages over VSA: no gate branch (VSA's gate
  GEMM 5376→7168 at long S ≈ 5–6% of denoise — by itself nearly VSA's entire
  deficit); 1D 64-token blocks consume packed QKV directly (no tile scatter /
  untile / scratch zeroing / `.item()` sync); preprocess pooling is cheap;
  the centroid-residual merge ≈ 1.5% of dense attention (kc/vc ~16 MB stay
  L2-resident).
- Two limits: (a) triton_ref kernel efficiency on SM80 estimated at 50–70% of
  cuDNN flash; (b) hence the **break-even condition is roughly route density
  ρ < 55–60%** (ρ = KV-block fraction kept at the tau=1.0 threshold). ρ was
  unmeasured on this model's real weights/shapes — the single decisive
  unknown; upstream prints route density every run, and it was the first
  thing to look at after porting.

**Porting checklist** (half-day scale, in order):

1. Export `techniques/sparse_backends/sol_attn/` from the Sana tree
   (`_vendor/`, `mps/` optional) into `h3_runtime/third_party/sol_attn/`,
   replacing the vendored 0.5.0 (back it up first) — the existing
   `_RELEASE_ROOT` mechanism then just works.
2. Interface alignment: `compile_bucket_size` parameter, `preprocess.prepare`
   signature, `sol_attn()` return contract — check each call site in
   `sparse_attention.py` (the two versions differ).
3. Numerics gate first: `sparse_attention.py`'s existing correctness gate
   (tau=-1000 full routing vs SDPA) must pass on real QKV before any
   performance talk.
4. A/B: 768p@10s/15s (8 GPUs) dense vs sol-triton; adapter unchanged
   (dense-datafree; upstream B300 also pairs dense-datafree + SOL).
5. **Stop rule**: if measured ρ > 60%, or 768p@15s denoise shows no ≥5% net
   win, abandon.

**Actual porting record (2026-09-12; steps 1–3)**: a per-file diff confirmed
the vendored package already matched the Sana tree `origin/sol-engine`
(`triton_ref`, `get_sol_attn_backend` both present), **no package swap
needed**; the only adaptation was `compile_bucket_size` — the interface
raises on non-SM100 backends, and `sparse_attention.py`'s default is now
per-backend (sol → 0, no bucketing; sol_bsa → 4096). Probe
`tests/test_sol_triton_sm80.py` passed all gates (GPU 7, real shapes
S=32k/93k/38k, H=7/14):

| Item | Measured | Criterion |
|---|---|---|
| dispatch | SM80 → `triton` | ✅ |
| full-route numerics (tau=-1000 vs bf16 SDPA) | max_abs=0.0005 / mean_abs=3e-5 | 1/160 of the 0.08/0.002 gate |
| route density ρ (tau=1.0, incl. sink + local band) | **0.16–0.19**, consistent across shapes | < 0.60 ✅ (⅓ of the stop line) |
| per-call throughput vs dense SDPA | **0.36–0.37×** (≈2.7× faster), consistent | — |
| 100%-density degradation | 1.64× slower than dense | Triton ≈ 61% of cuDNN, as estimated |
| JIT+autotune first compile | 5.9–7.9 s/shape/process | serving absorbs with `--warmup` |

Extrapolation (from §4.1 dense decomposition × probe ratios): 768p@15s
denoise 68.3s ≈ attention 32s → save ~21s; 768p@5s denoise 13.1s → ~2.5s.
**E2E and visual quality still unmeasured** (step 4).

**E2E record (2026-09-12, 8 GPUs, 768p@5s, seed 42, `prompts/orbital_original.txt`)
— the gray-collapse incident and root cause**:

| Configuration | denoise | E2E | Notes |
|---|---:|---:|---|
| dense (warmed, re-measured) | 13.06s | 15.53s | matches §4.1's historical 13.11s |
| sol async (no `H3_SOL_SYNC`) | 38.4s | 45.7s | CPU-overhead pathology, below |
| sol + `H3_SOL_SYNC=1` (warmed, **bug era**) | 10.13s | 12.58s | speed real, **content all-gray** |

The first run passed on speed alone (denoise −22%); pixel-level inspection
then found **every sol output was uniform gray collapse** (frame std ≈2.3 vs
dense ≈53–58; the audio track was fine — audio tokens sit in the exact-prefix
region, the failure was in the video-row routing path). After layer-by-layer
elimination (route density 0.21–0.27 normal, full-route numerics normal,
tau/threshold sweeps, kernel-vs-mask dense re-implementation, stages pinned),
the root cause was pinned with **real data in the production strided layout**:

> **The triton_ref pointer kernel hard-codes a dense (T·H·D) layout in its
> address math and ignores actual strides.** The production Ulysses path feeds
> a packed [Q|K|V] interleaved view (row stride 3D); the kernel silently reads
> the wrong data — measured on real QKV: contiguous input rel_l2=0.0008–0.0012
> (correct), strided input rel_l2=1.0–1.4 (completely wrong).

Three accompanying facts explain the long invisibility: (1) the engine
correctness gate and the offline probes both `.contiguous()` first — they
only ever tested contiguous inputs; (2) upstream `_validate_inputs` explicitly
permits packed zero-copy views (designed for the CuTe TMA path; the Triton
pointer path cannot honor them); (3) B300 production runs CuTe/cuDNN and never
touches this kernel — upstream's validation and our failure do not conflict.
Also fixed: `num_stages` pinned to 1 (the data-dependent exact-scan loop
produces wrong output on some processes at stages ≥ 2, and autotune
time-selects configs, producing cross-process non-determinism).

**Fix**: `H3_SOL_PACKED_INPUT` defaults to 0 in this fork (materialize
contiguous copies, ~1–2 ms/call; the switch is upstream's own), and
`num_stages` pinned to 1 (two autotune config blocks in `triton_ref/fwd.py`).

**Post-fix 480p full matrix** (8 GPUs, `bench/full_matrix.py`, warmed,
`H3_VAE_COMPILE=default` on both sides, seed 42) — content recovered
(frame std ≈55 vs dense ≈58, no visible difference from dense):

| t2v | dense denoise | sol denoise | Δ | dense E2E | sol E2E | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 5s | 4.14 | **3.95** | −4.6% | 5.27 | **5.06** | −3.9% |
| 10s | 9.47 | **8.34** | −12.0% | 11.51 | **10.37** | −9.9% |
| 15s | 16.45 | **13.80** | −16.1% | 19.39 | **16.83** | −13.2% |

i2v shows the same trend (denoise −4.2% / −9.6% / −14.5%). The win grows
monotonically with duration, matching the attention-share model; the −9~−24%
speed figures from before the fix (strided-bug era) are invalid (measured on
wrong routing + zero-copy). JSON:
`output/matrix_dense-dense_compile_8gpu.json`,
`output/matrix_sol-sol_fixed_8gpu.json`. Frame comparisons:
`output/frame_sol_fixed_f{30,60,90}.png` vs `frame_dense_f{30,60,90}.png`.

**Path A conclusion (the "is triton_ref routing equivalent to CuTe"
discriminating experiment)**: taking the Sol-H3 vendored code as the
authority, item by item:

| Comparison | Conclusion |
|---|---|
| threshold computation (`_diag_threshold_kernel`, both preprocesses) | **same math** (q̄·kc, log2 domain, global K statistics + variance; only TMA/pointer load APIs differ) |
| routing predicate (`sol_attn_route_is_exact` vs fwd.py:97-99) | **verbatim equivalent**: `(column_mean > threshold) or \|q_block−kv_block\| ≤ 1` |
| score normalization (CuTe mainloop `col_mean` vs triton `tl.sum/q_len`) | **same** (per-query-row mean, log2-scaled domain) |
| sink contract | **same** (KV-only; the caller re-denses prefix queries) |
| approximate-correction path | exists on both (CuTe mainloop header comment; triton's exp2(score−max)·v̄ centroid branch) |

I.e. **triton_ref's routing semantics are a faithful re-implementation of the
CuTe ones**; the contiguous-input full-route numerics (0.0008–0.0012) also
prove the kernel arithmetic is correct. The remaining ~8% per-layer routing
approximation (tau=1.0, ~24% density) is inherent to the Sol policy and
measured **survivable** (E2E visual checks show no difference from dense).
The tau sweep (contiguous real QKV): tau 1.0→0→−1 moves deviation
7.7%→2.9%→1.1%; quality-sensitive use can trade `H3_SOL_TAU` for quality
(speed win shrinks accordingly).

Methodology lessons (three; every future backend bring-up must follow them):

1. **Always `--warmup`**: a single unwarmed request bills fusion-kernel JIT /
   cuDNN autotune / memory-pool growth into the denoise steps — the unwarmed
   dense re-measure hit 28.9s (13.06s warmed); any unwarmed comparison is
   invalid.
2. **Speed conclusions require pixel checks**: this section's first −22%
   "speedup" was measured on a content-corrupted backend; every new backend
   passes frame-std/visual inspection before any performance claim.
3. **Numerics verification must use production input layouts**: the gate and
   offline probes defaulted to `.contiguous()`, hiding the strided-view
   kernel bug through two review rounds. Keep the production stride/alignment
   properties in verification, or at least add a strided case.

### 7.2 VSA prerequisite fixes and measurement gaps

- **Decode-OOM fix**: `h3_vsa.RequestState._buffers` tile scratch stays
  resident through decode; proactively freeing it before VAE decode/encode
  (or temporarily unpinning the gate Linear) restores the headroom. Only then
  can VSA×768p comparison numbers exist — which is also §7.1's control group.
- **VSA's own speed candidates** (long S): remove `map_to_index`'s `.item()`
  sync (§6; also the prerequisite for CUDA graph / torch.compile), switch
  scratch reuse to write-full-use-without-zeroing, pool in bf16.
- **Decision rule**: if §7.1 lands with acceptable quality, VSA becomes the
  backup; the two are judged by same-shape denoise time + visual quality, not
  maintained in parallel.

### 7.3 int8 QKV communication compression (the 480p battleground; landed 2026-09-12)

**Why communicate, not attend.** 480p S is too small: attention is only
10–24% of denoise (S² growth cannot catch the linear GEMM/communication).
Per-bucket measurement (`bench/int8_wire_ab.py` stage timing + an independent
topology probe): at 480p the **Ulysses all-to-all is 33–40% of denoise, the
single largest item** — world-8 PCIe all-to-all is only 12.2 GB/s/rank
(`NCCL_ALGO` auto/Ring/Tree indistinguishable; topology-bound), and each rank
moves ~206 MB of QKV + ~68 MB of output per block (15s tier). The only way to
speed up communication is to move fewer bytes.

**Implementation** (three places, all reusing upstream facilities):

1. `ulysses._qkv_wire_dtype`: the int8 gate widened from "sol_bsa native
   consumers only" to "any attention_fn that exposes an `int8_qkv` entry" —
   sol consumes it via decoding; dense/vsa (bare functions, no gate
   encoding) still refuse.
2. `comm_quant.dequantize_qkv_split` (new): decodes into **three independent
   contiguous tensors** rather than the interleaved-384 layout — the
   triton_ref pointer kernel needs contiguous inputs anyway, so the split
   decode simultaneously replaces the bf16 path's 3× `.contiguous()` copies
   (one job, two wins).
3. `sparse_attention.int8_qkv` (sol branch): a declined call de-duplicates
   quantization and goes dense SDPA; a sparse call quantizes and enters
   `_sol_forward` (shared execution body with `__call__`, `_import_kernel`
   cached).

Send side unchanged: `relayout.qknorm_rope_pack_qkv_destination_major`'s int8
encoding (group-32 UE5M3 scale + int8 values, 400 B/row·head) already existed,
fused into the RoPE+pack kernel.

**Verification chain** (all three §7.1 disciplines observed):

| Layer | Result |
|---|---|
| 2-GPU wire unit test (`tests/test_int8_wire_sm80.py`) | Q/K/V round trip rel_l2 0.006–0.008 (inherent quantization noise); split == padded decode bitwise; decoded output dense-contiguous; full-route attention deviation 0.026 (random tensors — the most softmax-hostile pessimistic bound, **below the accepted 8% routing approximation**) |
| 8-GPU E2E 480p (`bench/int8_wire_ab.py`, locked clocks, fully warmed, both wires in one process + end-of-run drift control) | 5s: denoise 3.95→**3.30 (−16.5%)**, E2E 5.42→**4.76 (−12.2%)**; 15s: denoise 13.84→**12.04 (−13.0%)**, E2E 17.80→**16.00 (−10.1%)**. The bf16 baseline matches the §7.1 archive exactly (13.84 vs 13.80); re-run drift <0.2% |
| quality (frame-std canary + content) | frame std 55–62 (0–255 scale), same tier as bf16, no gray collapse; deviation from dense reference frames (mean_abs 41.0 / corr 0.408) on the same order as the **already-accepted** bf16-sol deviation (41.9 / 0.388); the bf16 path reproduces the §7.1 archived frames bitwise; VLM content comparison shows both wires produce the same scene/composition without artifacts |

**Usage**: `H3_ULYSSES_COMM_DTYPE=int8_qkv` (opt-in; bf16 remains the default)
+ `--warmup`. The full stack with `H3_VAE_COMPILE=max-autotune-no-cudagraphs`
(480p, 8 GPUs, locked clocks, seed 42, orbital prompt; `bench/int8_wire_ab.py`;
JSON: `output/int8ab-{base,clean15,vaecomp,vaecomp15}_8gpu.json`):

| Configuration | 5s E2E / denoise / video_vae | 15s E2E / denoise / video_vae |
|---|---|---|
| sol bf16 wire (baseline) | 5.42 / 3.95 / 1.25 | 17.80 / 13.84 / 3.59 |
| + int8 wire | 4.76 / 3.30 / 1.25 | 16.00 / 12.04 / 3.61 |
| + int8 wire + VAE compile | **4.38 / 3.29 / 0.88** | **14.82 / 11.98 / 2.47** |

I.e. 5s cumulative **−19.2%**, 15s cumulative **−16.7%** (the VAE compile
alone −30~31% on decode). Note: compile specializes per tile-batch shape
(`dynamic=False`), each new shape JITs ~10–140s on first use — serving must
pre-warm per shape.

**Boundaries**: the output-direction wire is still bf16 (the
`H3_ULYSSES_OUTPUT_DTYPE` int8/fp8 paths are unvalidated on sol); wire noise
stacks with the routing approximation (both independent small perturbations,
no cross-amplification evidence; E2E frames verified); the 15s tier's relative
win is slightly smaller (−13% vs 5s's −16.5%), matching the communication-
share model.


## Migration (provenance summary)

The code's origin was verified by file-level byte comparison against the
source trees. Summary of the accounting (full per-file tables were in the
release's MIGRATION notes, now consolidated here):

- **Primary source**: NVIDIA's Sol-H3 MiniMax-H3 inference package
  ([NVlabs/Sana](https://github.com/NVlabs/Sana), `sol-engine` branch,
  `models/minimax_h3/Sol-H3`) — forked as the base and retargeted from
  8×B300 (SM103) to 8×A800 80GB (SM80).
- **Copied verbatim (58 files, byte-identical)**: 8 runtime-core files
  (`cp_plan`, `encoding`, `relayout`, `parallel_hooks`, `vae_parallel`,
  `quant_kernels`, `sol_residual`, `bsa_metadata`). The Sol-Attn kernel
  package is no longer redistributed: `fetch_sol_attn.sh` fetches it from
  NVlabs/Sana `sol-engine` at commit `8e0db4fa562d`
  (`models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn`, 51 files) into
  `h3_runtime/third_party/sol_attn/` and applies
  `patches/sol_attn-pin-num-stages.patch` (the only change: autotune
  `num_stages` pinned to 1 in `triton_ref/fwd.py`, see §7.1).
- **Rewritten / modified (12 files)**: `engine.py` (SM80 staged loading,
  backend wiring, CRC generation checks), `sparse_attention.py` (per-backend
  bucketing, `H3_SOL_SYNC` / `H3_SOL_PACKED_INPUT` defaults, int8 wire for
  sol), `ulysses.py` (VSA gate stream, MXFP8 removal), `lora.py` (VSA gate
  adapter), `infer.py` (CLI rewrite), `comm_quant.py`, `fusion_install.py`,
  `fusions.py` (env-tunable warp tiers), `adaln.py`, `__init__.py`,
  `bench/full_matrix.py`, `requirements.txt`.
- **Removed vs. upstream**: the MXFP8 DiT compute path (`mxfp8.py`,
  `compute_quant.py`) — SM80 has no FP8; the `sol_bsa` backend
  (architecturally SM90+ only).
- **Built here (no upstream equivalent)**: `text_shard.py` (layerwise-sharded
  Qwen3-VL conditioner with rank relay), the `vsa/` port, the SM80 sol
  re-wiring and strided-input kernel fix (§7.1), the int8 QKV wire for the
  Triton sol path (§7.3), the Ref2VA SM80 migration, all tests and bench
  scripts.
- **Methodology only, no code**: the earlier single-GPU (SM86) study
  `minimax-h3-desktop` informed benchmarking discipline; a cross-identifier
  search confirmed no code transfer in either direction.
- Code volume: upstream `h3_runtime` ≈ 3.9k lines → ≈ 5.6k lines here (plus
  original VSA ~1.3k, text_shard ~0.6k, bench/tests/launchers).
