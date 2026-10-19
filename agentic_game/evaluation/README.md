# evaluation — StatefulHarness on WorldState-Bench

This package produces the **STATEFULHARNESS row of Table 1** in the paper:
the WorldState-Bench cases (B01–B50, A/B twins = 100 clips) are played
through the full harness — **render** (the settled instruction wrapped in an
H3 I2VA envelope → the resident H3 service, i2v 5 s) → **judge** (a resident
Qwen VL watches the sampled frames and answers the dataset's pre-written
questions verbatim) → **report** (metrics). It also judges **external
submissions** (`run_external.py`), which is how the unwrapped baselines of
Table 1 are scored against the identical probe and judge.

## Dataset

The benchmark dataset is **not duplicated here**. It lives in the benchmark
repository at `../worldstate_bench/data/cases/` (50 case folders), fetched
once with `cd ../worldstate_bench && python download_data.py`; regenerating
the benchmark via `worldstate_bench/generation/run.py package` refreshes that
snapshot automatically. The default cases root resolves to that sibling
folder; it can be overridden with `--cases <DIR>` or the
`WORLDSTATE_BENCH_CASES` environment variable.

Each case directory contains the **shared first frame** (`input.png`), the
complete spec of each twin (`case_A.json` / `case_B.json`: `instruction`,
`hidden_state`, `rule`, `gold_end_state`/`delta`, `judge` — three pre-written
question groups: initial/action/outcome, with options and expected answers),
and human-readable cards (`card_A.txt` / `card_B.txt`, not consumed here).

Twin semantics: same first frame + same action + the same rule, with only the
`hidden_state` value flipped — the video model must resolve, for each side,
the branch ending that matches that side's hidden state. Case construction
(specs → FLUX.2 first frames → VLM audit → packaging) lives in
`worldstate_bench/generation/`.

## Quick start

```bash
# 1) Full harness run: headless checks (zero GPU) → render → judge → report
python3 -m evaluation.run_cases --stage all

# 2) The render stage needs the resident H3 service (the "H3 loads first, LLM starts
#    later" convention applies only to the coexistence mode; the judge stage defaults to
#    --judge parallel, which automatically stops H3 and starts 8 single-GPU LLM workers)
bash evaluation/ensure_services.sh

# 3) Stage by stage / pick cases
python3 -m evaluation.run_cases --stage render --case B01 --case B02
python3 -m evaluation.run_cases --stage judge                # 8-way parallel
python3 -m evaluation.run_cases --stage report --run-id <id>

# 4) Prompt modes: envelope=verbatim-instruction envelope (default) / remain=must-NOT
#    sentences removed / framework=one-turn GameRuntime through the real engine loop /
#    raw=baseline instructions passed through untouched
python3 -m evaluation.run_cases --stage all --prompt-mode remain

# 5) GPU-free dry run (exercises the data flow only; results are not publishable)
python3 -m evaluation.run_cases --stage all --renderer fake --judge heuristic

# 6) ref2va first-frame mode (no white-model preview): input.png is passed as a
#    pure-image reference and the prompt alignment line is swapped for the ref2va
#    first-frame anchoring line (everything else byte-identical); the service is the
#    resident engine ref2va instance on the box2clay socket, mutually exclusive with
#    the t2v service over the 8 GPUs
python3 -m evaluation.run_cases --stage all --ref2va

# 7) Judge external videos (unwrapped baselines / submissions) under both probes:
#    place clips at <output>/runs/<model>/raw/<CLIP>.mp4 and raw/<CLIP>.mp4-style
#    raw_state/<CLIP>.mp4 (100 each, B01–B50 × A/B), then:
python3 -m evaluation.run_external --models wan22,astra
```

Artifacts: `evaluation/output/casebench/runs/<run_id>/` (manifest.json,
videos/, frames/, verdicts/, metrics.json, report.md).

## Metrics (paper ↔ code)

Clip-level metrics require the judge's answer to equal the expected option
**verbatim**; unparseable or out-of-set answers score as incorrect (never
guessed). Unclear outcomes are counted separately and receive no credit.

| Code name | Paper name | Level | Definition |
|---|---|---|---|
| ActionShown | — (probe diagnostic) | clip | The shared action appears in the clip |
| **EndingFollow** | **ENDFOLLOW** | clip | Ending question hits **this side's branch** (hidden state resolved correctly) |
| **CounterAbsent** | **CNTABSENT** | clip | The counter-branch outcome never appears anywhere in the clip |
| **StrictFollow** | **STRICT** (headline) | clip | EndingFollow ∧ CounterAbsent |
| UnclearOutcome | UNCLEAR | clip | Ending answer unclear (counted wrong, reported separately) |
| **PairFollow** | **PAIRFOLLOW** | pair | EndingFollow holds on both A and B |
| **PairStrict** | **PAIRSTRICT** (headline) | pair | StrictFollow holds on both A and B |
| TwinContrast | TWINCONTRAST | pair | A/B judged endings differ (anti-copy-paste diagnostic) |
| BranchGap, DirectionError / LeakageOnly / DoubleError, CopyPaste, AnyUnanswered | Appendix B.1 diagnostics | — | Derived from existing verdicts, zero GPU |

`StartKeep` (all first-frame premise questions correct) and the action
question are recorded but excluded from the primary metrics, matching the
paper's judging protocol. The judge never sees the hidden value, the rule
program, or the other twin's expected outcome — only the branch-specific
expected endings.

### Instruction conditions (probes)

- `raw` = the paper's **Rule-only** probe: the case instruction verbatim.
- `raw_state` = the paper's **Rule+end** probe: raw + "The clip must end as:
  …" with the required terminal state (compiled by
  `worldstate_bench/baseline/prompts_common.py` for external baselines).
- `envelope` / `remain` / `framework` = harness-side variants; the harness
  condition reported in Table 1 settles the outcome before generation and
  compiles the required ending into the instruction (§3.2 of the paper).

## Relationship to `worldstate_bench/`

| | `worldstate_bench/` | `agentic_game/evaluation/` (here) |
|---|---|---|
| Role | The benchmark itself: case generation, dataset, judge, metrics, baseline wrappers | The **harness condition**: render cases through StatefulHarness + judge; also judges external submissions |
| Judge | LocalJudge (per-GPU sharded transformers workers) | ResidentJudge (engine resident Qwen VL service, 8-worker fleet) |
| Metric names | Paper names (`ENDFOLLOW`, `STRICT`, …) | Local names (`EndingFollow`, `StrictFollow`, …) — same definitions |

Both implementations follow the identical judging protocol (same system
prompt, same frame-position contract: Picture 1 = first frame, sampled frames
at 3 fps, ≤32 frames + final frame; answers verbatim from the fixed option
set). Baselines are generated in `worldstate_bench/baseline/` and judged
either there or here via `run_external.py`.

## Code layout (`evaluation/casebench/`)

- `dataset.py` — case loading and pair validation (a structurally invalid
  case raises immediately); default cases root points at
  `../worldstate_bench/data/cases`;
- `render.py` — instruction → H3 I2VA envelope (content unchanged, not one
  character altered) → resident service; with `--ref2va` the alignment line
  is swapped for the ref2va first-frame anchoring line (`apply_ref2va_anchor`);
- `framework.py` — the `framework` prompt mode: each twin runs as a one-turn
  game through the real `GameLoop` + `PromptCompiler` (the actual harness path);
- `headless.py` — zero-GPU self-consistency: the five outcome declarations
  (resolved-rule sentences / terminal_visual_states / gold / gold_end_state /
  judge expected) must cross-reference 100/100 before rendering proceeds;
- `frames.py` / `judge.py` — judging-frame sampling and the VLM judging
  backend (resident/heuristic);
- `fleet.py` — the 8-worker single-GPU judge fleet;
- `metrics.py` / `report.py` — metric rows, aggregation, and reports;
- `run_cases.py` — harness run entry point and all command-line arguments;
- `run_external.py` — external-video judging entry point (materialize →
  judge → report).

Services: `bash evaluation/ensure_services.sh` (H3 load ~5 min, rendering
~8 s/clip; judging ~40 s/clip). Tests: `python3 -m unittest
tests.test_casebench` (self-contained fixtures, no GPU and no dataset
needed).
