#!/usr/bin/env python3
"""Download the game first-frame assets (adv / duel / openworld) from HuggingFace.

The images are not committed to git (they ship in the HF dataset repo
`Louis0411/StateWorldModel`, ~94 MB under assets/) but they are **required**:
the skins reference them as first frames, the fake-renderer demo needs the
real files on disk, and the Gradio lobby shows only games that have a cover.

Usage (from anywhere):
    python3 download_assets.py

Only the assets/ subtree is fetched (allow_patterns), so the same repo can
also carry the WorldState-Bench dataset without duplication on either side.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO = "Louis0411/StateWorldModel"

ROOT = Path(__file__).resolve().parent
REQUIRED_DIRS = ("assets/adv", "assets/duel", "assets/openworld")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("SWM_ASSETS_REPO", DEFAULT_REPO),
                        help="HF dataset repo id (default: $SWM_ASSETS_REPO or %s)" % DEFAULT_REPO)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the asset folders already exist")
    args = parser.parse_args(argv)

    present = all((ROOT / d).is_dir() and any((ROOT / d).iterdir()) for d in REQUIRED_DIRS)
    if present and not args.force:
        print(f"assets already present under {ROOT / 'assets'} (use --force to re-download)")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.\n  pip install -U huggingface_hub")
        return 2

    print(f"downloading {args.repo} [assets/*] → {ROOT} ...")
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(ROOT),
                      allow_patterns=["assets/*"])
    missing = [d for d in REQUIRED_DIRS if not (ROOT / d).is_dir()]
    if missing:
        print(f"ERROR: download finished but {missing} are missing — "
              "check the repo layout (assets/ must be at the top level).")
        return 2
    n = sum(1 for d in REQUIRED_DIRS for f in (ROOT / d).rglob("*") if f.is_file())
    print(f"OK: {n} asset files under {ROOT / 'assets'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
