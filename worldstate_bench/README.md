# WorldState-Bench

The benchmark and evaluation toolkit from **"State World Models: Execute the
Rules, Generate the Pixels"** (under double-blind review, ICLR 2027).

WorldState-Bench is a benchmark of **50 counterfactual hidden-state pairs**
(100 clips): each pair shares one initial frame, one action script, and one
executable rule program, and differs only in a **hidden value** that is never
visible in the frame. A video model must resolve the rule under the hidden
value and render **exactly the required ending — and never the forbidden
one**. The dataset ships as a HuggingFace dataset repo; fetch it with
`python download_data.py` (≈63 MB) into `data/cases/`.

The repository contains everything around that benchmark:

1. **`generation/`** — how the benchmark is generated (spec → validated twins
   → FLUX.2 first frame → VLM-audited → packaged);
2. **`evaluation/`** — the VLM judging code (sampled frames → fixed choice
   questions → ENDFOLLOW / CNTABSENT / STRICT / PAIRFOLLOW / PAIRSTRICT);
3. **`baseline/`** — how the paper's two probe prompts are constructed and
   run on video backends (Rule-only, Track A · Rule+end, Track B).

> **Paper:** TODO (link added on publication) · **Project page:** TODO

**Everything about installing, generating the dataset, and running the
evaluation is in [USAGE.md](USAGE.md).**

## Overview

Each case is a **counterfactual pair** — twins A and B that share:

- the **same initial frame** `I₀` (one FLUX.2-rendered keyframe, 1280×720,
  audited so that no visual cue reveals the hidden value),
- the **same action script** (a neutral body trigger: raise a wand, pull a
  lever, step on a plate …),
- the **same rule program** written in a closed strict DSL and *mechanically
  simulated* to derive gold answers (no human-written gold),

and differ in one **hidden value** `z_A ≠ z_B` that selects different rule
branches, hence different required endings. Identical endings across a pair
necessarily violate the required outcome for at least one instance — the pair
is counted as compliant only when both clips show their own endings
(**Change-to-Comply** protocol: *Change* = a discernible event is produced,
*Comply* = the event satisfies the rule for the given hidden value).

The paper evaluates this benchmark under two instruction probes compiled from
each case specification:

| Probe | Track | Prompt content |
|---|---|---|
| **Rule-only** | A | original case instruction, hidden value, selected rule branch, action |
| **Rule+end** | B | additionally specifies the required terminal outcome in an explicit final-state sentence |

A frozen video model renders each clip; the multimodal **Qwen3.8-27B-VL**
judge answers fixed choice questions from sampled frames (it receives the
expected outcomes for the relevant rule branch, but neither the hidden
variable nor the rule text).

### Metrics

| Level | Metric | Meaning |
|---|---|---|
| clip (100) | **ENDFOLLOW** ↑ | final frames depict the ending specified for that instance |
| clip | **CNTABSENT** ↑ | the forbidden outcome (the other instance's required ending) is absent throughout the clip |
| clip | **STRICT** ↑ (primary) | ENDFOLLOW ∧ CNTABSENT — the mandated outcome is the only salient ending |
| clip | **UNCLEAR** ↓ | unjudgeable output; counted separately, never credited |
| pair (50) | **PAIRFOLLOW** | both twins individually ENDFOLLOW-correct |
| pair | **PAIRSTRICT** (headline) | both twins individually STRICT-correct |
| pair | **TWINCONTRAST** ↑ | judged endings differ across the pair (diagnoses copy-paste behavior) |

Probe contrast: **ΔSTRICT** = STRICT(Rule+end) − STRICT(Rule-only).

Headline results from the paper (50 pairs, 100 clips): the best of **11
unwrapped video backends** reaches 32% PAIRFOLLOW under Rule-only; for
MiniMax-H3, specifying the required ending raises PAIRFOLLOW from 32% to 64%
while PAIRSTRICT stays at 4%; the full **StatefulHarness** framework — which
adds deterministic rule settlement and compiled references on top of the same
frozen model — reaches **92% PAIRFOLLOW / 90% PAIRSTRICT** (see
`../agentic_game/evaluation/`).

### Dataset facts

| Fact | Value |
|---|---|
| Cases | 50 counterfactual pairs (100 clips), 49 hidden variables |
| Scenario themes | 21 |
| Interaction families | trigger_gesture (21) · trigger_mechanism (13) · trigger_cast (10) · trigger_zone (6) |
| People per frame | 41 cases × 1 player, 9 cases × 2 (NPC/guard) |
| First frame | FLUX.2, 1280×720, audited for hidden-value leaks |
| Judge | Qwen3.8-27B-VL, fixed choice questions on sampled frames |

### Dataset layout

```
data/cases/B01/
├── input.png        # shared first frame (EXIF: seed/steps/gate)
├── case_A.json      # full twin-A record (schema in USAGE.md)
├── case_B.json      # full twin-B record
├── card_A.txt       # human/model-readable card: inputs, gold, judges, constraints
└── card_B.txt
```

Key fields: `id/counterpart/genre/interaction_family/initial_scene`,
`hidden_state{variable,A,B}` (never visible), `rule` (DSL text + precondition
+ effects), `shared_action`, `instruction` (video-model prompt with explicit
counter-outcome prohibition), `gold_A/gold_B`, **`terminal_visual_states`**
(explicit occurs / not_occurs pair with an `instead_remains` phrase),
`judge{initial_questions, action_question, outcome_question{expected,
not_expected}}`, `constraints`, plus machine fields (`initial_state`,
`gold_end_state`, `delta`, `visual_vocab`, `prompt_ok`).

## Repository structure

```
worldstate_bench/
├── generation/                      # Part 1 · benchmark generation
│   ├── run.py                       #   entry: build / render / review / cycle / package / selftest
│   ├── specs/                       #   50 case specs in 5 modules (specs_b1..b5 = B01–B50)
│   └── ops/                         #   env rebuild specs + Qwen downloader
├── evaluation/                      # Part 2 · VLM judging of generated clips
│   └── run_eval.py …                #   headless → materialize → judge → report
├── baseline/                        # Part 3 · probe prompts + video-model wrappers
├── worldstate_bench/                # shared core package (config / DSL / builder / render / review / cases)
├── download_data.py                 # fetch the dataset from HF → data/cases/
├── data/cases/                      # the released dataset (downloaded; not committed)
├── config.yaml                      # single source of truth for paths & protocol parameters
├── tests/                           # unit tests (evaluation package)
├── USAGE.md                         # generation + evaluation walkthrough
└── out/, runs/                      # artifacts (gitignored)
```

## Quick start

Zero-GPU smoke of the whole data path:

```bash
pip install -r requirements.txt                      # pyyaml + huggingface_hub
python download_data.py                              # dataset → data/cases/ (~63 MB)
python3 generation/run.py selftest                   # build chain on temp dirs
python3 -m evaluation.run_eval --models <model> --stage headless
```

Full pipeline (needs GPUs + FLUX.2/Qwen weights): see **[USAGE.md](USAGE.md)**.

## Reproducing the paper results

1. **Benchmark regeneration** — `USAGE.md` § Generation, end-to-end with the
   same `config.yaml` protocol values (1280×720, 30 steps, guidance 4.0, seed
   ladder 42–47). Auditing is model-in-the-loop, so byte-identical
   regeneration is not expected; the hard guarantees are the V1–V10
   structural validations and the all-pass audit, deterministic given the
   committed specs.
2. **Probe ablation (paper Table 1 / ΔSTRICT)** — `raw` = Rule-only (Track
   A), `raw_state` = Rule+end (Track B); generate clips per backend
   (`USAGE.md` § Baselines), then judge with `evaluation.run_eval`. ΔSTRICT =
   STRICT(raw_state) − STRICT(raw).
3. There is **no training** here — the repository constructs evaluation data,
   runs probe baselines, and judges results. The StatefulHarness framework
   rows of the paper are produced by `../agentic_game/evaluation/`.

## Citation

```bibtex
@inproceedings{worldstatebench2027,
  title     = {State World Models: Execute the Rules, Generate the Pixels},
  author    = {Anonymous},
  booktitle = {Under review at ICLR 2027},
  year      = {2027},
  note      = {TODO: update with the final author list and venue on acceptance}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). Caveats: the dataset's first frames are
generated by FLUX.2-dev, whose license governs derived images, and
`baseline/hy15_batch.py` retains the Tencent Hunyuan community license
header. Full attribution: the root [ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md).
