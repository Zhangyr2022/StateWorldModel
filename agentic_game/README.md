# agentic_game — Stateful Harness: Turn-Based Games Rendered by a Frozen Video Model

Code for **"State World Models: Execute the Rules, Generate the Pixels"**
(framework name in the paper: *Stateful Harness*).

> Paper: TODO (under review) · Project page: TODO · Authors: TODO

A player makes a decision → the engine **resolves the turn deterministically
against an explicit world state** → the resolved event is compiled into a
structured instruction and **rendered as a short video by a frozen video
model** (MiniMax-H3, I2VA from the current first frame) → the clip's last
frame becomes the next turn's first frame → loop until an ending, then one
click stitches everything into a captioned replay film.

The repository is a **game-design base layer**: `engine/` is stable
infrastructure with no game rules in it. A new game is either a single
designer-authored **state-machine JSON skin** dropped into `game/skins/` (no
code), or — for genuinely new mechanics — five small files under
`game/<your_game>/` (see [AGENTS.md](AGENTS.md)). **20 games ship as JSON
skins**: duel ×9 + story ADV ×10 (5 Chinese + 5 English) + open world ×1.

## Installation

Requirements: **Python ≥ 3.11** (uses `tomllib`; developed on 3.12). The main
process needs no GPU and no PyTorch.

```bash
python -m pip install -r requirements.txt   # pillow + gradio (pinned <6)
python download_assets.py                   # first-frame images (~94 MB, HF dataset repo)
```

The **first-frame assets are required** (skins reference them; the fake demo
needs the real files; the lobby hides cover-less games). `download_assets.py`
reads the repo id from `--repo` / `$SWM_ASSETS_REPO`. Optional: **ffmpeg** on
`PATH` (replay assembly). Real rendering additionally
needs the companion `h3-a800` deployment + a Qwen VL checkpoint — see
[Real rendering](#real-rendering-h3-video--llm-dialogue) and the root
[Checkpoints](../README.md#checkpoints-and-models) section of the root README.

## How to play

### The game flow (what happens each turn)

```
player decision ──▶ rule resolution (deterministic, state + seed) ──▶ VisualScript
      ▲                                                                    │
      │                                                    prompt compiler │
      │                                                                    ▼
last frame ◀── new clip ◀── frozen video model (MiniMax-H3, I2VA) ◀─ instruction
```

1. You observe the current scene (the previous clip, with a **state HUD**
   overlaid: HP / relationship / story progress).
2. You pick or type an action.
3. The engine settles the outcome **before** any generation (events, numeric
   deltas, branch taken, win/lose — fully determined by state + seed).
4. The settled event is compiled into the video model's instruction; the
   model renders ~5 s of video for that outcome.
5. The clip's last frame becomes the next turn's first frame; repeat until an
   ending, then the game's clips can be stitched into one captioned film.

Render failures never touch game state — the turn resolves regardless, and
the failure policy (retry / reuse last anchor / skip) keeps play going.

### Start in 30 seconds (offline, no GPU)

```bash
python3 run_ui.py --renderer fake
```

Open **http://127.0.0.1:7860** → click a cover card in the lobby →
"▶ Start Game". DUEL games are fully playable offline (the state machine is
real; only the picture is a placeholder animation). Story/open-world games
additionally need the LLM service (next section).

### Full experience (LLM conversation + real H3 rendering)

Default deployment = **1+4**: 1 GPU for Qwen (conversation) + 4 GPUs for H3
(rendering):

```bash
scripts/start_services.sh start      # Qwen ~2 min, H3 ~6 min; status/stop/restart
python3 run_ui.py                    # sockets are pinned in config.toml
```

Manual equivalent and GPU-slot overrides:

```bash
QWEN_GPU=0 H3_GPUS=1,2,3,4 QWEN_MODEL=/path/to/Qwen-weights scripts/start_services.sh start
# or step by step:
python3 -m engine.llm.service start --model /path/to/Qwen-weights --per-gpu-mem 70GiB
python3 -m engine.renderer.service start
```

4 GPUs is the empirically tested floor for H3 (FastH3 4-step + 480p, 16–40 s
per turn); more GPUs are more stable.

### The lobby

A dark-cinema-styled single-page UI with two screens (lobby ↔ match):

- **Cover wall** — 25 cards: character-art covers for the 19 duel/adv games +
  6 scene cards for openworld (clicking a scene card pre-selects it);
- **Filter pills** — All / story ADV / DUEL / open world;
- **Details panel** — cover image, category badge, one-line intro, language,
  turn count, whether the conversation service is required;
- **Start bar** — render backend (`fake` / `h3 remote` (recommended) /
  `h3 session`), **random seed** (same seed + same actions = fully
  reproducible run; default 7), **max turns** (cap ends the game in a draw;
  default 24), then "▶ Start Game" (failure keeps you in the lobby, e.g. LLM
  not ready).

### The three categories

| Category | What you do | How it resolves |
|---|---|---|
| **DUEL** ×9 | Each turn, **click one skill** (cost/counter notes shown) | Local state machine + weighted boss AI; deterministic and reproducible |
| **Story ADV** ×10 | **Type into the "💬 Say to her" box**; free-form input | An LLM plays the heroine and responds; relationship/mood advance the chapters |
| **Open world** ×1 | Click one of the VLM-proposed actions, or **define your own** | A VLM director watches the frame and advances the story; quiet turns auto-explore |

### In a match

- **Left**: this turn's performance video (autoplay) with the state HUD
  overlaid; top bar shows game name / turn count / ending badge.
- **Right**: status panel (HP / relationship / her words …), action area,
  🐞 debug panel (duel only: force the boss's action for one turn), 📜 battle
  log, "← Back to lobby".
- Turn flow: pick an action → "▶ Execute action" → resolve + render (~1–3 min
  on real H3) → the video refreshes → next turn. If a render fails, the turn
  is skipped per policy and **the game state is unaffected**.

### Endings and the replay film

At game end (victory / defeat / turn cap) the ending animation plays
automatically. Click **"🎬 Generate full replay"** to stitch the whole game
into **a single film with state captions** and play it in place.

### Artifacts of each run

```
generated/ui-run-<timestamp>/
├── turn_log.jsonl        # per-turn event log (resolution / rendering / captions)
├── clips/initial.mp4     # short clip of the static opening first frame
├── videos/turn-001.mp4 … # each turn's performance
├── ending.mp4?           # ending animation (when the game is decided)
└── game_montage.mp4      # the film produced by "Generate full replay"
```

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "LLM service not ready" at startup | Story / open world need the Qwen service; start it before starting those games |
| "Failed to start: h3 …" | The H3 service is not up or still loading; wait and retry, or switch to the `fake` backend |
| Each turn takes 1–3 minutes | Normal for real H3 rendering; use `fake` if you don't want to wait |
| A turn's video did not update | Render-failure skip policy; game state unaffected, continue to the next turn |
| "Action no longer valid, please choose again" | The turn's menu was refreshed; pick again — the turn is not consumed |
| Want to replay the same game | Same seed + same action sequence = an identical match |

Common `run_ui.py` options: `--port`, `--host 0.0.0.0` (LAN),
`--share` (temporary public link), `--renderer`, `--mode
remote|session|subprocess`, `--config`. Headless CLI:
`python3 run_demo.py --game duel_oracle_machine_epic --renderer fake --agent auto`.

## Repository structure

```text
agentic_game/
├── engine/                    # Base layer (no game rules)
│   ├── runtime.py             #   Action / Rng / TurnResult / GameRuntime protocol
│   ├── loop.py                #   GameLoop: resolve → visual → render → next turn + endings
│   ├── visual/                #   VisualScript / VisualState / PromptCompiler (single prompt outlet)
│   ├── renderer/              #   fake / remote / session / worker + resident service
│   ├── llm/                   #   LLMClient / resident worker / service management
│   ├── ui/                    #   Gradio app + turn-by-turn session driver (bilingual EN/中文)
│   ├── montage.py             #   Replay assembly: caption bar + concat
│   └── config.py              #   defaults ← config.toml ← environment variables
├── game/
│   ├── registry.py            # The single game registry (both entry points read it)
│   ├── skins/*.json           # 20 active skins — one file = one game
│   ├── adv_theme/             # adv template engine (story skeleton / LLM heroine / expressions)
│   ├── machine_theme/         # duel template engine (states / transitions / boss tables)
│   ├── openworld/             # VLM-directed exploration
│   ├── wbench/                # external-WBench evaluation adapter (third-party benchmark, not included)
│   └── skin_validation.py     # Skin validation error types
├── evaluation/                # WorldState-Bench harness evaluation (render → judge → report)
├── assets/                    # First-frame images per category (adv / duel / openworld)
├── tests/                     # unittest suite — no GPU / real API required (230 tests)
├── scripts/                   # start_services.sh · ui_smoke.py · rebuild_replay_hud.py · dev_llm_stub.py
├── run_demo.py                # CLI entry: plays a full game headlessly
├── run_ui.py                  # Gradio entry: the player lobby + match screen
└── AGENTS.md                  # Building a new game + the skin JSON spec + engine rules
```

## Quick start (no GPU, no services, no weights)

```bash
python3 -m unittest discover -s tests      # base-layer suite: 230 tests, ~2 min
python3 run_demo.py --game duel_oracle_machine_epic --renderer fake --agent auto
python3 run_ui.py --renderer fake          # Gradio lobby, offline
```

## Real rendering (H3 video + LLM dialogue)

The production pipeline needs the companion **h3-a800** inference repository
(MiniMax-H3 on multi-GPU) and a Qwen VL checkpoint, deployed side by side
(see the section *Full experience* above). Cross-repo paths derive from
`WORKSPACE_ROOT` (`engine/config.py`) — no absolute paths hardcoded.
Configuration precedence: **built-in defaults ← `config.toml` (template:
`config.example.toml`) ← environment variables** (`AGENTIC_GAME_RENDERER`,
`AGENTIC_GAME_H3_REPO`, `AGENTIC_GAME_TORCHRUN`, `AGENTIC_GAME_NPROC`,
`AGENTIC_GAME_SEED`, `AGENTIC_GAME_MAX_TURNS`). Which weights you need: root
[Checkpoints](../README.md#checkpoints-and-models) section of the root README.

```bash
python3 run_demo.py --game duel_oracle_machine_epic --renderer h3 --mode remote --agent random
python3 run_ui.py --renderer h3 --mode remote       # the UI, now with real video
```

## Evaluation (WorldState-Bench harness)

`evaluation/` runs the paper's benchmark through the full pipeline: headless
data validation (zero GPU) → render against the resident H3 service → a Qwen
VL judge answers pre-written questions per case → metrics report. Dataset and
metric names are aligned with the paper; a `--renderer fake --judge
heuristic` dry run exercises the whole data path without GPUs. Details:
[evaluation/README.md](evaluation/README.md).

## Reproducibility

- **Resolution is exactly reproducible**: the same (initial state, action
  sequence, seed) produces the identical game trajectory. Randomness only
  flows through the injected `Rng(seed)` (`--seed` / `AGENTIC_GAME_SEED`).
- **Rendering is not bit-reproducible**: the video model is generative. The
  *resolved outcome* of every turn is fixed by the seed; the pixels vary.
- The offline test suite, the `fake`-renderer demo, and the evaluation dry
  run are fully reproducible on CPU-only machines.

## Acknowledgements

Built on the [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) video
model (deployed by the companion `h3-a800` repository), the Qwen3.8-27B VLM,
Gradio, and Pillow. Full attribution: root
[ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md). External-benchmark adapters
(`game/wbench/`) target the third-party WBench (Fudan & Meituan LongCat,
arXiv:2605.25874), which is **not** included in this release.

## Citation

```bibtex
@misc{stateworldmodels2026,
  title  = {State World Models: Execute the Rules, Generate the Pixels},
  author = {TODO: authors},
  year   = {2026},
  note   = {Under review. TODO: venue, paper URL, project page}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE). Third-party models keep their own
licenses; see the root [ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md).
