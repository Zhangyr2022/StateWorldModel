#!/bin/bash
# baseline/ experiment environment — the single place with machine-specific paths.
# Source this before running any baseline script:
#   source baseline/env.sh
#
# All values may be overridden from the calling shell, e.g.:
#   CODEBASE_ROOT=/data/me/codebase MODELS_ROOT=/data/me/models bash baseline/worker.sh 0 longcat

# Project root (this file lives in <project>/baseline/).
export BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(dirname "$BASELINE_ROOT")"

# Directory containing the cloned upstream inference repos, one subdir each:
#   LongCat-Video/  HunyuanVideo-1.5/  wan2.2/
export CODEBASE_ROOT="${CODEBASE_ROOT:-/path/to/upstream-codebases}"

# Directory containing the video-model weights (see ../README.md, Checkpoints):
#   LongCat-Video/                        (LongCat-Video, distilled)
#   Wan-AI/Wan2.2-I2V-A14B/               (Wan2.2 I2V)
#   hub/models--tencent--HunyuanVideo-1.5/snapshots/<rev>/   (HF cache layout)
export MODELS_ROOT="${MODELS_ROOT:-/path/to/models}"

# conda prefix holding the three baseline envs: longcat / hunyuan15 / wan22.
export CONDA_ROOT="${CONDA_ROOT:-/path/to/miniforge3}"

# HuggingFace env (mirror + big-disk cache are optional; adjust or drop).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -n "${HF_HOME_OVERRIDE:-}" ] && export HF_HOME="$HF_HOME_OVERRIDE"

# Baseline run outputs and logs live under the project (gitignored).
export RUNS_ROOT="$PROJECT_ROOT/runs"
export LOGS_ROOT="$BASELINE_ROOT/logs"
