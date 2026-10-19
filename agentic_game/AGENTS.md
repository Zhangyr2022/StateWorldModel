# AGENTS.md — Building a New Game on This Base Layer

This repository is driven by a coding agent to generate games. There are two
routes:

1. **JSON skin (the default, no code)** — one `.json` file into `game/skins/`;
   states / transitions / numeric rules / boss behavior tables / every
   performance prompt live in that file. Spec: [§ Skin JSON reference](#skin-json-reference).
2. **Five-file module** — only when the engine entirely lacks a mechanic
   (an engine extension — discuss first, then implement). Process: [§ The five-file process](#the-five-file-process).

**Engine (`engine/`) is stable infrastructure — do not modify it for a new
game.** The engineering rules for the base layer are in
[§ Engine engineering rules](#engine-engineering-rules).

Reference implementations: `game/adv_theme/` (story games with LLM
resolution), `game/openworld/` (VLM-directed exploration), and the combat
skins (`game/skins/*.json` + the `machine_theme` engine).

---

## The five-file process

A new game = four files plus one registration line:

1. `state.py` — your GameState (pure data, dataclass).
2. `actions.py` — player-selectable Actions (`engine.runtime.Action`, stable
   ids) and numeric parameters. Free-input gameplay can, like adv_theme, use a
   single `speak` action and put the content into `Action.label`.
3. `rules.py` — implement the `GameRuntime` protocol (engine/runtime.py):
   - `initial_state()` → the initial state
   - `available_actions(state)` → currently selectable actions
   - `resolve_turn(state, action, rng)` → `TurnOutcome(events, status, state_after)`
     **Deterministic**; randomness only via the passed-in `rng`; **do not
     mutate the passed-in state** (deepcopy, mutate the copy, return the new
     object). When an LLM is needed, inject `engine.llm.LLMClient` (a resident
     service) via the constructor; constrain outputs with a JSON contract +
     clamping + retries; LLM failure is a resolution failure — raise
     explicitly or degrade by a defined fallback rule, never silently invent.
   - `status_of(state)` → `won / lost / draw / ongoing`
   - `describe_state(state)` → a state summary for the player/logs (also the
     replay caption text)
4. `visual.py` — `build_visual_script(turn_result, visual_state)` →
   `engine.visual.script.VisualScript`: translate this turn into one visual
   performance (duration declared via `duration_s`; default 5 s, current
   combat skins 10 s); plus an `ending_xxx_script(status, visual_state)`
   ending-animation mapping.
   - Video-facing text is English; in-game copy may be Chinese.
   - Appearances are registered in `VisualState.appearances` and persist
     across turns (character consistency).
   - `first_frame_anchor` describes a composition consistent with the previous
     frame (may reference `<Picture 1>`).
   - **Do not** assemble the final prompt, do not write `[Shot N]` markers, do
     not call the Renderer.
5. `__init__.py` — provide `build_game(llm=None, **options) -> GameParts`
   (engine/parts.py): assemble the parts above into the unified contract and
   declare the few things the UI needs (initial first frame, free-text action,
   scene labels, ending captions, agent factory).

Then register one line in `game/registry.py` (game_id + display name +
category + needs_llm) — `run_demo.py` and `run_ui.py` work purely off the
registry, so entry points need zero new code.

### Prohibitions

- Do not modify `engine/` to serve one game's needs; when the engine lacks a
  capability, discuss first.
- Game code must not import `engine.renderer.minimax_h3/session`, must not
  issue HTTP, must not assemble prompt strings; depend only on the `Renderer`
  abstraction and `VisualScript`.
- Do not use self-built random sources, wall-clock time, global state, or any
  other non-deterministic input inside `resolve_turn`.
- Do not reconstruct Game State from Visual State; a render failure must not
  roll back game state.

### Validation checklist (before merging a new game)

```bash
python3 -m unittest discover -s tests   # the existing base-layer tests must not regress
# plus your own:
#  - rules tests: resolution / illegal actions / endings
#  - determinism test: same-seed replay is identical
#  - visual test: every action has a VisualScript mapping
#  - offline run: run_demo --renderer fake end to end
```

---

## Skin JSON reference

The adopted design is the **state machine** (`category: "machine"`): the
designer delivers a single JSON file into `game/skins/`; after a restart it
appears in the game list. States, transitions, numeric values, boss behavior,
and the visual-performance prompt for every case all live in this one file;
the engine (`game/machine_theme/`) only parses and evaluates it.

Quick start:

```bash
# 1. Write your json using an existing example as reference
#    Combat: game/skins/duel_oracle_machine_epic.json
#    Exploration: tests/fixtures/museum_heist.json (mechanism-test sample)
# 2. Self-check (invalid input reports the offending field and the reason)
python3 -m game.machine_theme.json_skin game/skins/my_game.json
# 3. Drop it into game/skins/ and play right away
python3 run_demo.py --game my_game --renderer fake --agent auto
```

### Top-level structure

```json
{
  "category": "machine",              // fixed
  "game_id": "my_game",               // unique id (the --game value)
  "title": "display name",
  "frame": "assets/xx.png",           // initial first frame
  "player_name": "中文名", "player_label": "the english label",
  "player_appearance": "English appearance anchor (locks identity across turns)",
  "scene_description": "scene description (English)", "style": "visual style",
  "camera": "camera constraint (recommended: locked camera / never leave the original position)",
  "first_frame_anchor": "first-frame anchoring sentence (may reference <Picture 1>; optional)",
  "stability": ["anti-drift constraint sentences (optional)"],
  "closing_state": "global closing sentence (optional)",
  "soundscape": "...", "music": "...",
  "turn_flow": ["player", "boss"],    // pure exploration games: ["player"]
  "vars": { }, "initial_state": "...", "states": { },
  "actions": [ ], "transitions": [ ],
  "endings": { "won": {...}, "lost": {...} },
  "var_display": [...]
}
```

### Player and boss sides (structurally symmetric)

Player and boss each have their own states (state axis) and actions; the only
difference is that the player's actions are chosen by the player while the
boss's are chosen by weighted draw (`pre` filtering + `weight`).

```json
"player": {
  "initial_state": "standoff",
  "states": { "standoff": { "label": "对峙", "action": "...", "end_state": "...", "effects": [...] } },
  "actions": [ { "id": "fireball", "label": "禁典·红日坠落",
                 "phrase": "opening performance (English)",
                 "outcomes": { "hit": "hit performance", "blocked": "performance of being stopped" } } ],
  "transitions": [ { "from": "*", "action": "fireball", "to": "",
                     "pre": "player_mp >= 5",
                     "outcome_expr": "ifelse(boss_shield, 'blocked', 'hit')",
                     "effects": [...] } ]
}
```

- `transitions[].from` may be the wildcard `"*"`; omitting `to` = stay put;
- `pre` is the availability expression (false → the action is hidden);
  `conditions` + `probability` control whether the transition fires
  (fall-through = else);
- `outcome` / `outcome_expr` picks which performance from
  `actions[].outcomes` plays — same action + different outcome = a different
  performance; the outcome expression reads the board **before** this turn's
  effects;
- the boss section is symmetric; `boss.actions` carries selection metadata
  (`weight` / `pre` / performance), resolution lives in `boss.transitions`;
  `boss.states` is the boss posture axis (e.g. guard/windup/enraged);
- expression environment: `vars` and `boss_state` (the boss posture) are both
  visible; `boss_state == 'windup'` can replace boolean flags;
- `damage.block/half/shatter` act on the target's shield (all three consume
  it); `description` is shown in the UI debug panel.

### Turn pipeline and timing

- `"turn_flow": ["player", "boss"]` — the player resolves first, the boss
  second; the turn stops as soon as any phase decides the game (defensive
  shields raised by the player take effect the same turn).
- One turn is **10 seconds** rendered as **two** timeline segments
  (`phase_split`, four numbers summing to 1; default 0.6/0.4 → 6 s action /
  4 s resolution+closing): action phrases in `turn_flow` order, then the
  outcome performance + both sides' `end_state`.
- Exploration games without a boss omit the `boss` section and use
  `["player"]`.

### Endings

`won`/`lost` each take `{condition: [...], action, effects, end_state,
caption}`; checked in declaration order after each turn. Ending conditions
**only read vars** — to express "reached a state", `set` a flag variable in a
transition.

### Design constraints (must be followed)

1. Every action must have at least one reachable transition in every state
   (a fallback), otherwise it stalls in place;
2. Probabilistic failure paths must have an exit (e.g. alertness up → crouch
   down), otherwise deadlock is inevitable;
3. Avoid `roll` in ending conditions — ending determination must be purely
   deterministic;
4. Clamp capped variables (HP/MP) with `ifelse` to handle overflow;
5. Performance phrases (`phrase` / `end_state`) are in **English** and follow
   the NEWSKILL spec (`engine/visual/NEWSKILL.md`): one clear subject action +
   one set of visible effects, with direction, a final pose, and a completed
   state.

---

## Engine engineering rules

The base layer's hard invariants (the full style guide was consolidated
here):

1. **One-way dependency**: `game → engine` only; `engine` never imports a
   concrete game (the UI reads the registry lazily).
2. **Games never touch the renderer or prompts**: depend only on the
   `Renderer` abstraction and `VisualScript`; the compiler
   (`engine/visual/compiler.py`) is the single prompt outlet; `engine/loop.py`
   is the single render call site.
3. **Determinism**: `resolve_turn` is a pure function of
   (state, action, `Rng(seed)`); no wall clock, globals, or hidden sources;
   same seed ⇒ same trajectory (enforced by `tests/test_determinism.py`).
4. **Explicit state mutation**: deepcopy-mutate-return; no in-place surprises;
   explicit dataclasses over dicts for state.
5. **Renderer failure never corrupts game state** — failure policy: retry →
   reuse previous anchor → skip/abort; the turn outcome stays settled.
6. **One turn = one fixed-length clip**; VisualScripts describe outcomes, not
   complex directorial detail; visual continuity comes from the anchor chain
   and persistent appearances.
7. **Generated content is separate from source** (`generated/`, gitignored);
   configuration is separate from code (`config.example.toml` → `config.toml`
   → env vars).
8. **Explicit errors and logging**: raise typed errors
   (`SkinValidationError`, `MontageError`, …); per-turn structured records
   (`turn_log.jsonl`) keep playthroughs debuggable.
9. **The video model must be mockable** — the `FakeRenderer` backs the whole
   offline test suite (230 tests) and the UI smoke.

## Lookup guide

- Where are Actions defined? → `game/*/actions.py`
- Where does a turn resolve? → `resolve_turn` in `game/*/rules.py`
- Where is the Visual Script produced? → `game/*/visual.py`
- How are games registered/assembled? → `game/registry.py` + `engine/parts.py` (GameParts)
- Where is the Gradio UI? → `run_ui.py` entry + `engine/ui/` (session.py drives turn by turn,
  app.py is the interface; the UI never imports a concrete game)
- Where are prompts assembled? → `engine/visual/compiler.py` (the single outlet)
- Where is the Renderer invoked? → `engine/loop.py` (the single call site)
- How is the benchmark evaluated? → `evaluation/README.md`
