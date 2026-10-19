#!/usr/bin/env python3
"""Download the MiniMax-H3 checkpoints and adapters used by this deployment.

Total download is ~219 GB with every task enabled (~143 GB for t2v/i2v only;
the `transformer_ref` Ref2VA partition alone is ~66 GB). All sources are
Hugging Face repos; `MiniMaxAI/MiniMax-H3` is gated, so accept the model terms
in the browser and log in with `hf auth login` first. Interrupted downloads
resume: rerun the same command and finished shards are skipped.

Space is checked up front because a 200+ GB download must not start on a disk
that cannot hold it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

GB = 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--task",
        choices=("t2v", "ref2va", "both"),
        default="both",
        help="Which DiT partition to download: `transformer` (t2v/i2v), "
        "`transformer_ref` (ref2va), or both (default: both)",
    )
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    need = (160 if args.task == "t2v" else 230) * GB
    free = shutil.disk_usage(root).free
    if free < need:
        raise SystemExit(
            f"{root} has {free / GB:.0f} GB free; this download plus headroom needs "
            f"~{need / GB:.0f} GB. Free space or choose another --output-dir."
        )

    model_patterns = [
        "model_index.json",
        "modular_model_index.json",
        "audio_scheduler/*",
        "audio_vae/*",
        "processor/*",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "vae/*",
    ]
    if args.task in {"t2v", "both"}:
        model_patterns.append("transformer/*")
    if args.task in {"ref2va", "both"}:
        model_patterns.append("transformer_ref/*")

    snapshot_download(
        "MiniMaxAI/MiniMax-H3",  # gated: accept the terms on the model page first
        local_dir=root / "MiniMax-H3",
        allow_patterns=model_patterns,
    )
    if args.task in {"t2v", "both"}:
        snapshot_download(
            # Renamed upstream after the Sol-H3 snapshot; the old
            # "FastVideo/FastH3-4-step-Preview-v1-LoRA" id 404s.
            "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
            local_dir=root / "FastH3-4-step-Preview-v1-LoRA",
            allow_patterns=[
                "dense-datafree/adapter_model.safetensors",
                "vsa-datafree/adapter_model.safetensors",  # required by --attention-backend vsa
                "adapter_manifest.json",
            ],
        )
    if args.task in {"ref2va", "both"}:
        snapshot_download(
            "lightx2v/Minimax-h3-Turbo",
            local_dir=root / "Minimax-h3-Turbo",
            allow_patterns=["minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"],
        )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
