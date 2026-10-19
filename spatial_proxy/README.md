# spatial_proxy / box2clay — White-Model Control-Video Pipeline

Generate **camera-controllable "previz" control videos** from real images, and use
them to drive **MiniMax-H3 Ref2VA** video generation. Given one WBench case image,
the pipeline detects the subject in 3D, renders a minimal Blender "clay" previz
video (blue subject box, two red reference pillars, infinite grid floor), builds a
prompt triple, and generates the final photorealistic video conditioned on
reference image + control video.

```
WBench case image ──① WildDet3D 3D detection (det.json: box2d + box3d + K)
                    ──② Blender previz control video (bpy 4.2, EEVEE headless + Toon)
                    ──③ QC: 120-frame check, jump check, first-frame alignment check
                    ──④ Prompt builder (flat / structured / first-person)
                    ──⑤ H3 ref2va generation (8×A800, ../h3-a800) ──→ final video
```

Two camera-control modes are supported:

| | third_person (74 WBench cases) | first_person (84 WBench cases) |
|---|---|---|
| Renderer | `clay_subject.py` | `fp_previz.py` |
| Scene | blue subject box + two red pillars (front-left / front-right, black outlines) + infinite grid floor | uniform white model, **no blue box**, grid floor, fixed 1.6 m eye height |
| Motion | subject translates in world space; camera rigid-follows / orbits / vertical arc | camera itself moves (push/pull/truck/pan/pitch) |
| Actions | 16 (aligned with WBench `THIRD_PERSON_ACTIONS`, `S+A` added) | 16 (WBench `FIRST_PERSON_ACTIONS`) |
| Amplitude | `--push 0.70` (W/S/A/D equal; arcs 45°) | `--move 3.2` (4-direction equal; arcs 45°) |
| Anchoring | det.json box; **first-frame blue-box near-face projection = det 2D box (±5 px, constructive guarantee)** | no detection needed; fixed 1.6 m eye height |

The pipeline was built against [WBench](https://meituan-longcat.github.io/WBench/)
(arXiv:2605.25874) navigation cases; all commands below assume the repository
layout with `spatial_proxy/`, `h3-a800/`, and `WBench/` side by side under one parent
directory (override with `WORKSPACE_ROOT`).

## Repository structure

```text
spatial_proxy/
├── box2clay/                  # the pipeline (first-party)
│   ├── detect_boxes.py        # ① WildDet3D open-vocabulary 3D detection → det.json
│   ├── clay_subject.py        # ② third-person previz renderer (16 actions)
│   ├── fp_previz.py           # ② first-person previz renderer
│   ├── boxes2clay.py          # render library (setup_render/make_cam/add_box/edge tubes/toon)
│   ├── _archive/              # retired code snapshots (see _archive/README.md)
│   ├── wbench_prompt.py       # ④ prompt executor (flat | structured | fp), --self-check
│   ├── gen_struct8_prompts.py # ④ batch prompt triple export
│   ├── wbench_template.md     # structured-branch template (single source of truth)
│   ├── wbench_manifest.py     # WBench case scanner → TSV manifest
│   ├── check_jumps.py         # ③ frame-diff spike check on rendered MP4s
│   ├── check_first_frame_align.py  # ③ blue-box first frame vs det box (±5 px)
│   ├── ref2va_serve.py        # ⑤ resident 8-GPU H3 ref2va service (Unix socket)
│   ├── h3serve_batch.py       # ⑤ batch driver over the socket (resumable)
│   ├── h3ref_batch.py         # ⑤ alternative: single-load torchrun batch
│   ├── run_det8.sh            # ① example: 8 cases × 8 GPUs batch detection
│   └── results/               # sample control videos + H3 outputs (see Results)
└── README.md                  # WildDet3D is NOT included — clone it (see Installation)
```

## Installation

Three environments are used (do not share one Python env across them). Set
`CONDA_ENVS_ROOT` to your conda environments root.

```bash
# 1) bpy311 — previz rendering (EEVEE headless; no Blender desktop app needed)
conda create -n bpy311 python=3.11 && conda activate bpy311
pip install bpy==4.2.0 pillow

# 2) wilddet3d — detection stage (clone the upstream repo + compile the CUDA ops)
git clone https://github.com/allenai/WildDet3D WildDet3D      # SAM License
conda create -n wilddet3d python=3.11 && conda activate wilddet3d
pip install -r WildDet3D/requirements.txt
# pull its declared submodules (sam3 fork, lingbot-depth) per WildDet3D/docs,
# build vis4d_cuda_ops, and place its checkpoint (see ../README.md, Checkpoints)

# 3) h3-a800 — generation stage; see ../h3-a800/README.md (Python 3.12, torch cu130)
```

Configuration is env-var driven; defaults derive everything from the file
locations in the side-by-side layout:

| Variable | Purpose | Default |
|---|---|---|
| `WORKSPACE_ROOT` | workspace root anchor for cross-project paths | derived from file location |
| `CONDA_ENVS_ROOT` | conda env root (interpreters) | `~/miniconda3/envs` |
| `WBENCH_DATA` | WBench dataset directory | `<workspace>/WBench/data` |
| `WILDDET3D_ROOT` | WildDet3D checkout | `<workspace>/spatial_proxy/WildDet3D` |
| `H3_REPO` | h3-a800 repository | `<workspace>/h3-a800` |
| `H3_TORCHRUN` | torchrun binary | `$PATH` → `CONDA_ENVS_ROOT/h3-a800/bin/torchrun` |
| `H3_NPROC` | number of GPUs for the ref2va service | `8` |

## Data preparation

WBench dataset (images + case JSONs) is **not** committed; download it once:

```bash
hf download meituan-longcat/WBench --repo-type dataset --local-dir ../WBench/data
```

The third-person pipeline uses the 74 third-person navigation cases
(`nav_cate ∈ {progressive, roundtrip, repeat, loop, l_shape, zigzag}`), all of
whose actions are within the 16-action table; the first-person pipeline uses 84
first-person cases.

## Checkpoints

| Model | Purpose | Required for | Expected path | Download |
|---|---|---|---|---|
| WildDet3D `wilddet3d_alldata_all_prompt_v1.0.pt` (~4.7 GB) | open-vocab 3D detection | ③rd-person stage ① only | `WildDet3D/ckpt/` | `huggingface-cli download allenai/WildDet3D wilddet3d_alldata_all_prompt_v1.0.pt --local-dir WildDet3D/ckpt/` |
| MiniMax-H3 base (~210 GB, gated) | video+audio DiT | stage ⑤ | `../h3-a800/checkpoints/MiniMax-H3/` | see `../README.md#checkpoints-and-models` |
| Minimax-h3-Turbo ref2v LoRA (1.3 GB) | ref2va 4-step adapter | stage ⑤ | `../h3-a800/checkpoints/Minimax-h3-Turbo/` | see `../README.md#checkpoints-and-models` |

First-person mode (① skip: no detection) needs no WildDet3D checkpoint; stages
①–④ need no H3 weights. Nothing downloads at runtime — all loads run with
`HF_HUB_OFFLINE=1`. Weights are gitignored; ~19 GB of other third-party weights
explored during development (IGGT, VGGT-Ω) were **removed** with their repos and
are not needed by this pipeline.

## Pipeline

All commands run from `spatial_proxy/box2clay/`; `<env>` denotes the conda env's
python under `$CONDA_ENVS_ROOT`.

### ① Detection (third-person only; wilddet3d env)

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $CONDA_ENVS_ROOT/wilddet3d/bin/python \
    detect_boxes.py ../WBench/data/images/case_14.jpg \
    --out results/subject --texts "person"
# → results/subject/case_<id>_<class>/det.json  (authoritative: box2d_xyxy + box3d_cam + K)
#   det_vis.png is visualization only; depth_*.npy is decoupled from rendering
```

`run_det8.sh` is a recorded example batch (8 cases across 8 GPUs; edit the case
table or set `WILDDET_PYTHON`).

### ② Control video (bpy311 env; ~15 s per clip, 3–5 concurrent processes per GPU)

```bash
# third-person (red corner pillars on by default; --no-corner-refs to disable)
CUDA_VISIBLE_DEVICES=0 $CONDA_ENVS_ROOT/bpy311/bin/python clay_subject.py \
    results/subject/case_14_person results/subject_v2/case_14_person \
    --subject person --subject-height 1.7 --samples 16 \
    --bare --edge-tubes --action all16 --no-preview --width 864 --push 0.70

# first-person
CUDA_VISIBLE_DEVICES=0 $CONDA_ENVS_ROOT/bpy311/bin/python fp_previz.py \
    results/fp_v1/case_12 --action all16 --move 3.2 --no-preview --width 864
```

Output: `<out_dir>/walk_<action>.mp4` (`+` → `_`), 120 frames at 864 px width.
Without `--action`, only `preview.png` (single frame) is produced for iteration.

### ③ QC (must pass before generation)

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
    -of csv=p=0 <mp4>          # must print 120
python check_jumps.py <mp4...>  # all reports must be jump-free
python check_first_frame_align.py case_14_person   # blue box vs det box ±5 px
```

Known false positive: pitch-type (`up`) clips show sub-pixel late-clip drift that
the median-threshold detector flags; inspect visually.

### ④ Prompts (bpy311 env)

```bash
python gen_struct8_prompts.py \
    --manifest results/wbench_det_v3.json --style flat \
    --subject-root results/subject_v2 --out results/h3_flat_v5
# → prompts_struct8.json (key=<det_dir>_walk_<action>), images_struct8.json
#   (video abs path → case image), videos_struct8.txt, prompts_review.txt
```

Single-prompt debugging / self-check:

```bash
python wbench_prompt.py --case 14 --action W --style flat
python wbench_prompt.py --self-check
```

`--style flat` (default) emits per-action direction sentences plus the three red-box
text mentions; `--style structured` is the action-independent six-section control
branch (template in `wbench_template.md`); `--style fp` is first-person. AB testing
showed the action-independent structured text collapses A/D lateral moves into
forward motion — keep direction sentences explicit.

### ⑤ H3 generation (h3-a800 env; resident service ~38 s per clip)

```bash
# resident service (one-time ~3 min load; 8 GPUs × ~70 GB)
python ref2va_serve.py start | status | stop

# batch over a prompt manifest (resumable: existing outputs are skipped;
# delete old mp4s before re-running with changed prompts)
$CONDA_ENVS_ROOT/h3-a800/bin/python h3serve_batch.py \
    --manifest results/h3_flat_v5 --subject-root results/subject_v2 \
    --duration 5 --seed 42
# → <manifest>/<key>.mp4
```

Alternative without the resident service (single 8-GPU torchrun load per batch,
~2–3 min): `torchrun --standalone --nproc_per_node=8 h3ref_batch.py --videos ...
--prompts-json ... --images-json ... --first-frame ref2va` (`--out-root` must be
absolute).

## Key conventions

1. **Key naming**: `<det_dir>_walk_<action>` (`+` → `_`) is shared by control-video
   filenames, prompt-JSON keys, and H3 output names. A key mismatch in
   `h3ref_batch.py` falls back to a built-in scenario prompt **with a loud
   warning** — never use the fallback for WBench batches.
2. **Absolute paths**: `h3ref_batch.py` `os.chdir`s into `H3_REPO`, so its
   `--out-root` and `--images-json` values must be absolute;
   `h3serve_batch.py --subject-root` resolves relative paths against the CWD.
3. **Prompt slots** (third person): `{desc}` = `settings.subject.desc` (appearance
   only), `{env}` = `environment_prompt` (kept verbatim — its off-screen sentences
   serve scene adherence), `{camera}` = `perspective_prompt`. `character_prompt`,
   `scene_adherence`, `subject_adherence`, `metric_list` never enter the prompt.
4. **First-frame alignment is a hard constraint**: the blue box's near-face
   projection must equal the det 2D box. Four known destroyers, all fixed — do not
   reintroduce: detection yaw (~90° systematic offset, deprecated), VGGT ground
   hcam (adopting it shifts the box), transposed dims axis order (`(w, H, l)`),
   over-tight hcam clamping.
5. **Amplitudes**: third person `--push 0.70`; first person `--move 3.2`; arcs 45°.
   0.35 / 1.6 were measured as too weak and discarded.
6. **Render stack**: EEVEE Next headless + Toon materials + edge-tube cylinders
   (self-emissive pure black; AgX lifts blacks to gray) + emissive black grid +
   direct FFMPEG output (the output `filepath` must end in `.mp4`). Blender's bpy
   mode ships a default scene — clear it. Take `matrix_parent_inverse` **after**
   `view_layer.update()`; avoid TRACK_TO (it rolls 90°) — use analytic Euler.
7. **`/tmp` can be full**: sockets and temp artifacts go under the workspace
   (`results/`); verify the service process is alive after start.

## Results

`box2clay/results/` ships one minimal demo pair (one third-person case + one
first-person case, each showing the control video and the H3 ref2va output it
produced):

| File | Content |
|---|---|
| `sample/control/case_1_dog/walk_W.mp4` | third-person control video (blue box + red pillars, action W) |
| `sample/control/fp_case_100/walk_S.mp4` | first-person control video (action S) |
| `sample/h3/case_1_dog_walk_W.mp4` | H3 ref2va output generated from the control video above |
| `sample/h3/fp_case_100_walk_S.mp4` | first-person H3 ref2va output |
| `wbench_det_v3.json` | example case manifest (12 cases: case_id / det_dir / subject_class / subject_height / actions) |

The full sample batches these were trimmed from remain in git history. To
re-run at scale, regenerate prompt triples with `gen_struct8_prompts.py` after
preparing your own detections.

## Acknowledgements

- [WildDet3D](https://github.com/allenai/WildDet3D) (Allen Institute for AI et
  al., arXiv:2604.08626) — vendored under `WildDet3D/` (SAM License; see its
  `LICENSE` and README). Used as-is via its documented inference API for stage ①;
  it in turn vendors a SAM 3 fork (SAM License) and
  [lingbot-depth](https://github.com/Robbyant/lingbot-depth) (Apache-2.0).
- [WBench](https://meituan-longcat.github.io/WBench/) (Fudan University & Meituan
  LongCat, arXiv:2605.25874, MIT) — external benchmark; case images/JSONs drive
  the pipeline and the 16-action vocabulary is aligned with its
  `prompt_builder.py`.
- [Blender](https://www.blender.org/) / `bpy` 4.2 (GPL) — rendering runtime via
  the pip module. [BlenderProc](https://github.com/DLR-RM/BlenderProc) (GPLv3)
  was previously used only to bootstrap the external `bpy` module
  (`USE_EXTERNAL_BPY_MODULE`); the released code imports `bpy` directly and does
  not use BlenderProc.
- [IGGT](https://lifuguan.github.io/IGGT_official) (arXiv:2510.22706) and
  [VGGT-Ω](https://vggt-omega.github.io/) (arXiv:2605.15195) — explored for box
  production / depth validation in early pipeline versions; both were removed
  from this release (its `boxes.json` renderer is archived at
  `box2clay/_archive/clay_scene.py`).
- MiniMax-H3 weights (MiniMax H3 Community License) and the
  [Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) ref2v
  adapter power stage ⑤ via [../h3-a800](../h3-a800/).

## Citation

Placeholder — paper metadata to be added by the authors.

```bibtex
@misc{box2clay2026,
  title  = {TODO: Paper title},
  author = {TODO},
  year   = {2026},
  note   = {TODO: URL/arXiv id}
}
```

## License

This directory's first-party code (`box2clay/`) is released under
[Apache-2.0](LICENSE), aligned with the workspace convention (`../h3-a800`).
It ships no model weights: the MiniMax-H3 weights used by stage ⑤ are governed
by the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
and vendored third-party code retains its original notices — most notably the
WildDet3D detector under the SAM License (research/educational use); see
Acknowledgements above.
