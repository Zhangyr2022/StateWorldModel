# WorldState-Bench — Usage: dataset generation & evaluation

This is the operational walkthrough: environments → generating the dataset →
dataset format → evaluating clips → baseline backends → adding cases.
Concepts and metric definitions: [README.md](README.md).

```
generation/specs (50 case programs)
        │  run.py build → render → review → cycle → package
        ▼
data/cases/  (50 pairs, downloaded via download_data.py)   ← the released dataset
        │  baseline/prompts_common.py  (Rule-only / Rule+end prompts)
        ▼
runs/<model>/{raw,raw_state}/*.mp4          ← clips from any video backend
        │  evaluation/run_eval.py  (headless → materialize → judge → report)
        ▼
runs/<model>-{raw,raw-state}/{metrics.json, report.md, verdicts/}
```

## 1. Environments

Three environments: a tiny **build** env (runs `build`/`package` in-process)
and two GPU envs used as subprocess interpreters for **render** (FLUX.2) and
**review** (Qwen3.8). The review env also runs the evaluation judge.

```bash
pip install -r requirements.txt                                 # build env (pyyaml)
bash generation/ops/envs/rebuild_envs.sh <conda-prefix>         # render + review envs
```

The script never touches existing environments: if `flux2` or `kairos` already exists it stops
without changes. Pick other names with `FLUX2_ENV=… KAIROS_ENV=…`, or pass `--force` to delete and
rebuild them. flash-attn comes from a local wheel in `generation/ops/envs/wheels/` when present,
otherwise from the upstream prebuilt wheel (falling back to a source build).

| Env | Python | Key pins (verified) | GPU memory |
|---|---|---|---|
| `flux2` (render) | 3.11 | torch 2.14.0 · diffusers 0.40.0 · torchvision 0.29.0 | 2×24 GB+ per render shard |
| `kairos` (review + judge) | 3.10 | torch 2.6.0+cu126 · transformers 5.2.0 · flash-attn 2.7.4.post1 | ~52 GB bf16 per GPU (one model copy per shard GPU) |

Evaluation additionally needs system **ffmpeg**.

**Weights** (download commands: root [README → Models and data](../README.md#models-and-data)):

| Model | Purpose | Size |
|---|---|---|
| FLUX.2-dev (gated) | first-frame rendering | ~110 GB bf16 |
| Qwen3.8-27B | first-frame audit + clip judge (`ops/dl_qwen38.sh <dir>`) | ~52 GB |
| FasterRCNN-MobileNetV3-Large-320-FPN | person-count gate | auto-downloads on first render (pre-place at `~/.cache/torch/hub/checkpoints/` to stay offline) |

**Configuration** — `config.yaml` is the single source of truth. The five
machine-specific values:

```yaml
envs:
  build_python: /usr/bin/python3                       # any python with pyyaml
  render_python: <conda-prefix>/envs/flux2/bin/python
  review_python: <conda-prefix>/envs/kairos/bin/python
render:
  model: <path-to>/FLUX.2-dev
review:
  model: <path-to>/Qwen3.8-27B
```

GPU topology (defaults target an 8-GPU node): `render.gpu_pairs` — one render
shard per GPU pair (`["0,1","2,3","4,5","6,7"]`); `review.gpus` — one review
worker per listed GPU. Wrong interpreter paths fail fast with an actionable
message.

**Capacity**: render shards are FLUX.2 bf16 sharded across a GPU *pair*;
review loads one full Qwen3.8 copy per GPU; ~250 GB disk for weights +
outputs; render/review run with `HF_HUB_OFFLINE=1`.

**Verification sequence** (each step exercises the next dependency):

```bash
python3 generation/run.py selftest              # build chain, temp dirs, no GPU
python3 generation/run.py render --limit 1      # render chain (FLUX.2 + gate)
python3 generation/run.py review --only B01 --gpus 0   # review chain (Qwen3.8)
python3 generation/run.py package               # packaging; compare data/cases/
```

## 2. Generating the dataset

```bash
python3 generation/run.py selftest   # smoke-test the build chain (temp dirs only)
python3 generation/run.py build      # spec → out/yaml/*.yaml (100) + out/work/render.csv (50 rows)
python3 generation/run.py render     # first-frame render (one shard proc per GPU pair)
python3 generation/run.py review     # VLM frame audit (auto-starts a new round)
python3 generation/run.py cycle      # full-auto: render→audit→revise→re-render until all pass
python3 generation/run.py package    # → out/cases/ + refreshes the tracked dataset data/cases/
```

What each stage does:

- **build** — evaluates the 50 case specs in `generation/specs/specs_b*.py`
  into twin YAMLs through hard validations **V1–V10** (grammar, branch
  divergence driven by the hidden value, non-empty gold delta on both sides,
  neutral first-frame prompt, discriminative judge options, action
  neutrality). Gold answers are derived by **mechanically simulating** the
  rule DSL (`worldstate_bench/dsl.py`) — no human-written gold.
- **render** — FLUX.2 renders each case's shared first frame (1280×720,
  protocol in `config.yaml`: 30 steps, guidance 4.0, seed ladder 42–47); a
  FasterRCNN person-count gate enforces the frame's cast. Resume: existing
  frames (>10 kB) are skipped.
- **review** — a six-dimension VLM audit of every rendered frame (sanity,
  entity/person count, initial predicates, action-ready composition,
  premature-terminal, terminal-room). On failure a second VLM call proposes a
  revised render prompt, which must re-pass `builder.validate_prompt` before
  reuse.
- **cycle** — loops render→audit→revise→re-render (same seed, revised
  wording) until every frame passes; the revision journal
  (`out/work/prompt_fixes.json`) is re-applied after every `build`, so a
  rebuild never loses audit-driven fixes.
- **package** — emits `out/cases/<CASE>/{input.png, case_{A,B}.json,
  card_{A,B}.txt}` **and refreshes the committed dataset at `data/cases/`**
  (the default dataset root for everything downstream).

Useful variants:

```bash
python3 generation/run.py render --limit 25              # first 25 cases only (demo)
python3 generation/run.py render --only B13 --seeds 43   # one case, explicit seed
python3 generation/run.py review --only B01,B04 --gpus 0 # subset on one GPU
python3 generation/run.py cycle --dry-run                # print the plan without executing
```

**Never invoke `worldstate_bench/render.py` directly** — the revision-journal
overlay lives in `generation/run.py`.

## 3. Evaluating clips

First fetch the dataset if you have not already:

```bash
python download_data.py          # → data/cases/ (50 cases; --repo/$SWM_DATASET_REPO to override)
```

The evaluator judges **already-generated clips** against the dataset's
pre-written judge questions. Clips are expected at:

```
runs/<model>/raw/<CLIP>.mp4        # Rule-only  (paper Track A)
runs/<model>/raw_state/<CLIP>.mp4  # Rule+end   (paper Track B)
```

with `<CLIP>` = `<CASE>-<SIDE>` (e.g. `B01-A`), 100 clips per mode.

```bash
# Full chain: data self-check → materialize → per-GPU shard judging → report
python3 -m evaluation.run_eval --models wan22 --gpus 0,1,2,3

# Zero-GPU dry run (heuristic fills expected answers — NOT a real result; needs the clips in place,
# and exits with an error when no clips are found)
python3 -m evaluation.run_eval --models wan22 --limit 1 --judge heuristic

# Stages (all resumable; verdicts dedupe by existence)
python3 -m evaluation.run_eval --models wan22 --stage materialize|judge|report

# A single materialized run:
python3 -m evaluation.run_eval --run-dir runs/wan22-raw --stage report
```

Each (model, mode) is materialized into a standard run dir (videos are
symlinked, not copied):

```
runs/<model>-raw/  (or -raw-state/)
├── videos/<CLIP>.mp4   symlink → the clip
├── frames/<CLIP>.png   final frame (ffmpeg); frames/<CLIP>/ holds samples
├── prompts/<CLIP>.txt  the prompt actually sent at inference time
├── headless.json       dataset self-consistency check result
├── manifest.json       model/mode/source
└── verdicts/ + metrics.json + report.md + FAILURES.md
```

### Protocol

Frame contract: Picture 1 = first frame, then chronological samples (default
3 fps, 896 px long side, ≤32 frames/clip), last picture = final frame.
Initial questions look at Picture 1 only; the action question at the whole
clip; the outcome question at the end; the counter question at the whole
clip. Answers must be copied **verbatim** from the data-provided options;
unparseable or out-of-vocabulary answers count as wrong (never guessed). The
judge sees the branch-specific expected outcomes but **not** the hidden
value, the rule text, or the other twin's expected outcome.

Judge backends: `local` (default — one worker subprocess per GPU loading
`config.review.model`, greedy decoding, interpreter from
`config.envs.review_python`) and `heuristic` (pipeline wiring only).

Outputs per run: `metrics.json` + human-readable `report.md` +
`FAILURES.md` (each failure embeds the actual inference prompt). **ΔSTRICT**
= STRICT(`-raw-state`) − STRICT(`-raw`) — compare the two run dirs.

Zero-GPU dataset self-check at any time:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from evaluation.dataset import load_cases
from evaluation.headless import run_headless
passed, checked = run_headless(load_cases())
print(f'{passed}/{len(checked)} clips self-consistent')"
```

### Relationship to `../agentic_game/evaluation/`

The harness-side evaluation (GameLoop → VisualScript → PromptCompiler
rendering, H3 I2VA envelope, resident judge fleet) lives in the harness
repository and produces the paper's StatefulHarness rows; this directory
carries only the judging/metric chain bound to the benchmark itself. The data
format (`case_{A,B}.json`) is identical on both sides.

## 4. Baseline backends

`baseline/` compiles both probes from the dataset and drives three video
backends: **LongCat-Video** (distilled), **HunyuanVideo-1.5**,
**Wan2.2-I2V-A14B**.

Prerequisites: the three upstream inference repos cloned under one directory
(`CODEBASE_ROOT`), their weights in place (`MODELS_ROOT`), and conda envs
`{longcat,hunyuan15,wan22}` — then fill in the four paths in
`baseline/env.sh`.

```bash
source baseline/env.sh
python3 baseline/prompts_common.py longcat hy15 wan22   # build manifests from data/cases
bash baseline/smoke3.sh            # optional: one B01-A.raw clip per model on GPUs 0-2
bash baseline/progress.sh          # per-model generation progress
bash baseline/worker.sh 0 longcat hy15   # one-GPU job chain
```

- `raw` prompts = the card's "Inputs to the video model" block verbatim
  (hidden value, rule, shared action, instruction, chunk);
- `raw_state` prompts = raw + the explicit final-state sentence ("The clip
  must end as: …", built from `gold_end_state × visual_vocab`).

Resume: a target mp4 that exists and is >50 kB is skipped; `worker.sh` writes
a `done_<model>` marker after a complete chain (delete to rerun); single-clip
rerun: `wan22_batch.py --only B01-A.raw` (substring match).

Other backends from the paper are driven through their own action interfaces
— extending is instruction-level work: compile the same two probes and write
clips into `runs/<model>/{raw,raw_state}/`, then judge with `run_eval`
(also how **external submissions** are evaluated).

## 5. Adding cases

A case is one Python spec in `generation/specs/specs_bN.py` (10 cases per
module, B01–B50): theme + scene description, hidden variable declaration
(`name`, `value_A`, `value_B`, invisible-evidence notes), action script, and
rule branches in the strict DSL. The builder validates **V1–V10** at `build`
time — branch symmetry, hidden-value-driven divergence, neutral first-frame
prompt, discriminative judge options, action neutrality — so an invalid spec
fails immediately with the offending field. Keep the invariants: hidden value
invisible in the frame prompt; both branches specify their endings with
fixed sentence frames; judge options distinguish the two outcomes. Then run
the §2 pipeline; `package` refreshes `data/cases/` and the whole §3 chain
picks the new case up.
