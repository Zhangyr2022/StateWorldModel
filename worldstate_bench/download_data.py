#!/usr/bin/env python3
"""Download the WorldState-Bench dataset (50 counterfactual twin cases) from HuggingFace.

The dataset is not committed to git (it ships in the HF dataset repo
`Louis0411/StateWorldModel`, ~63 MB under data/cases/). It is the default
dataset root for `evaluation/run_eval.py`, `baseline/prompts_common.py`, and
`../agentic_game/evaluation/`.

Usage (from anywhere):
    python3 download_data.py

Only the data/cases subtree is fetched (allow_patterns), so the same repo can
also carry the game assets without duplication on either side.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO = "Louis0411/StateWorldModel"

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "data" / "cases"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("SWM_DATASET_REPO", DEFAULT_REPO),
                        help="HF dataset repo id (default: $SWM_DATASET_REPO or %s)" % DEFAULT_REPO)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the dataset already exists")
    args = parser.parse_args(argv)

    def n_cases() -> int:
        return sum(1 for d in CASES.iterdir() if d.is_dir()) if CASES.is_dir() else 0

    if n_cases() == 50 and not args.force:
        print(f"dataset already present under {CASES} (50 cases; use --force to re-download)")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.\n  pip install -U huggingface_hub")
        return 2

    print(f"downloading {args.repo} [data/cases/*] → {ROOT} ...")
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(ROOT),
                      allow_patterns=["data/cases/*"])
    if n_cases() != 50:
        print(f"ERROR: download finished but {CASES} holds {n_cases()} cases "
              "(expected 50) — check the repo layout (data/cases/ at the top level).")
        return 2
    print(f"OK: 50 cases under {CASES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
