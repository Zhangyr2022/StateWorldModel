#!/usr/bin/env python3
"""Command-line entry point for MiniMax-H3 inference on A800."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h3_runtime.engine import RESOLUTIONS, MiniMaxH3Inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MiniMax-H3 checkpoint directory or repo ID")
    parser.add_argument("--adapter", required=True, help="LoRA adapter file to fuse (t2v/i2v: FastH3 "
        "dense-datafree, alpha 64; ref2va: the ref2v turbo LoRA, alpha 8)")
    parser.add_argument(
        "--attention-backend",
        choices=("dense", "vsa", "sol"),
        default="dense",
        help="Attention backend (default: dense; vsa needs the VSA adapter; "
        "sol runs the Triton Sol-Attn kernel; ref2va is dense-only)",
    )
    parser.add_argument(
        "--res",
        choices=tuple(RESOLUTIONS),
        help="Resolution preset; overrides --width/--height (480p = 864x480, 768p = 1344x768)",
    )
    parser.add_argument("--width", type=int, default=864, help="Canvas width, multiple of 32 (default 864)")
    parser.add_argument("--height", type=int, default=480, help="Canvas height, multiple of 32 (default 480)")
    parser.add_argument("--task", choices=("t2v", "i2v", "ref2va"), default="t2v",
                        help="i2v is t2v plus --image; ref2va conditions on --reference media")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--image", type=Path, help="First-frame keyframe for i2v; PNG, JPEG, or WebP")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="TYPE:PATH",
        help="Ordered ref2va input; repeat with image:, video:, or audio:. The order is "
        "meaningful: it labels the references in the prompt presentation and lays "
        "them out on the shared rotary clock",
    )
    parser.add_argument(
        "--reference-image-resize-mode",
        choices=("match", "diffusers"),
        default="match",
        help="ref2va image sizing: fast match mode (never upscale, cap to the canvas "
        "area; default) or official Diffusers 2048-short-edge mode",
    )
    parser.add_argument("--duration", type=int, choices=(5, 10, 15), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def load_references(specs: list[str]):
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

    references = []
    kinds = []
    for spec in specs:
        kind, separator, path = spec.partition(":")
        kind = kind.lower().strip()
        path = path.strip()
        if not separator or kind not in {"image", "video", "audio"} or not path:
            raise SystemExit(
                f"invalid --reference {spec!r}; use image:PATH, video:PATH, or audio:PATH"
            )
        if not Path(path).exists():
            raise SystemExit(f"--reference {spec!r}: {path} does not exist")
        # decoding (and for video/audio, the PyAV demux) happens right here, on
        # every rank, before any collective runs
        references.append(MiniMaxH3Reference(**{kind: path}))
        kinds.append(kind)
    if references and not ({"image", "video"} & set(kinds)):
        raise SystemExit("an audio reference must be paired with an image or video reference")
    return references


def main() -> int:
    args = parse_args()
    # request-level validation up front: a failure here must not cost an
    # ~8-minute engine load first
    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit("pass either --prompt or --prompt-file, not both")
    if bool(args.prompt) == bool(args.prompt_file):
        raise SystemExit("pass exactly one of --prompt or --prompt-file")
    if args.prompt is not None and not args.prompt.strip():
        raise SystemExit("--prompt is empty")
    if args.prompt_file is not None and not args.prompt_file.read_text(encoding="utf-8").strip():
        raise SystemExit(f"--prompt-file {args.prompt_file} is empty")
    if args.image is not None and not args.image.exists():
        raise SystemExit(f"--image {args.image} does not exist")
    if args.task == "i2v" and args.image is None:
        raise SystemExit("--image is required for i2v")
    if args.task == "t2v" and args.image is not None:
        raise SystemExit("pass --task i2v to condition on --image")
    if args.task == "ref2va" and args.image is not None:
        raise SystemExit("use --reference image:PATH for ref2va, not --image")
    if args.task == "ref2va" and not args.reference:
        raise SystemExit("pass at least one --reference for ref2va")
    if args.task != "ref2va" and args.reference:
        raise SystemExit("--reference is only valid for --task ref2va")
    if args.task == "ref2va" and args.attention_backend != "dense":
        raise SystemExit("ref2va supports --attention-backend dense on this deployment")
    width, height = RESOLUTIONS[args.res] if args.res else (args.width, args.height)

    prompt = (
        args.prompt
        if args.prompt is not None
        else args.prompt_file.read_text(encoding="utf-8").strip()
    )
    references = load_references(args.reference) if args.task == "ref2va" else None

    with MiniMaxH3Inference(
        args.model,
        args.adapter,
        attention_backend=args.attention_backend,
        task="ref2va" if args.task == "ref2va" else "t2v",
        width=width,
        height=height,
        reference_image_resize_mode=args.reference_image_resize_mode,
    ) as engine:
        if args.warmup:
            engine.warmup(duration=args.duration, prompt=prompt, image=args.image,
                          references=references)
        result = engine.generate(prompt, duration=args.duration, seed=args.seed,
                                 image=args.image, references=references)
        if result is not None:
            result.save(args.output)
            print(
                json.dumps(
                    {
                        "output": str(args.output.resolve()),
                        "task": args.task,
                        "duration": args.duration,
                        "seed": args.seed,
                        "attention_backend": args.attention_backend,
                        "width": width,
                        "height": height,
                        "reference_image_resize_mode": (
                            args.reference_image_resize_mode if args.task == "ref2va" else None
                        ),
                        "inference_s": round(result.elapsed_s, 3),
                        "stages_s": (
                            {k: round(v, 3) for k, v in result.stages.items()}
                            if result.stages
                            else None
                        ),
                    },
                    ensure_ascii=False,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
