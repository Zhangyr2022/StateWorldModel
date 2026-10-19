"""渲染核心：FLUX.2-dev 共享首帧 + 结构门控 + 断点续渲。

协议（config.render 可调）：1280×720 / 30 步 / guidance 4.0 / seed 阶梯。
每 case 一张首帧（A/B twin 同图，symbol 不入图）。
门控：FasterRCNN person——gate=personN → 恰 N 人（含 NPC/守卫的 case 为 2）；
单人需全身（score≥gate_min_score、面积≥gate_min_area、底边≥gate_min_bottom）。
断点续渲：目标 PNG 存在且 >10KB 跳过（幂等可重跑）。
注意：Flux2Pipeline 不支持 negative_prompt（实测 TypeError），勿加。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from worldstate_bench.config import Config, load_config

RESUME_MIN_BYTES = 10_000


def main(argv: list[str] | None = None, cfg: Config | None = None) -> int:
    ap = argparse.ArgumentParser(prog="worldstate_bench.render")
    ap.add_argument("--csv", type=Path, action="append", required=True)
    ap.add_argument("--frames-root", type=Path, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpus", default=None, help="diffusers device_map 用的卡")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="只渲这个 case，如 F01-01")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seeds", default=None, help="逗号分隔显式 seed 列表")
    args = ap.parse_args(argv)

    cfg = cfg or load_config()  # 子进程模式：读项目根 config.yaml（默认值仅作缺键回退）
    rc = cfg.render
    W = int(rc.get("width", 1280))
    H = int(rc.get("height", 720))
    STEPS = int(rc.get("steps", 30))
    GUIDANCE = float(rc.get("guidance", 4.0))
    SEEDS = [int(s) for s in rc.get("seeds", [42, 43, 44, 45, 46, 47])]
    GATE_PASS = float(rc.get("gate_pass_score", 0.55))
    GATE_MIN_SCORE = float(rc.get("gate_min_score", 0.5))
    GATE_MIN_AREA = float(rc.get("gate_min_area", 0.02))
    GATE_MIN_BOTTOM = float(rc.get("gate_min_bottom", 0.70))
    MODEL = args.model or rc.get("model")
    if not MODEL:
        raise SystemExit("config.yaml 缺少 render.model,且未传 --model")
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds
             else SEEDS)

    rows: list[dict] = []
    for p in args.csv:
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    rows = rows[args.shard::args.nshards]
    if args.only:
        rows = [r for r in rows if r["case"] == args.only]
    if args.limit:
        rows = rows[: args.limit]

    def out_path(row: dict) -> Path:
        return args.frames_root / f"{row['case']}.png"

    todo = [r for r in rows
            if not (out_path(r).exists()
                    and out_path(r).stat().st_size > RESUME_MIN_BYTES)]
    print(f"[render] rows={len(rows)} todo={len(todo)} "
          f"(skipped {len(rows) - len(todo)} existing)", flush=True)
    if not todo:
        return 0

    import os
    import torch
    from PIL import ExifTags, Image

    os.environ.setdefault("CUDA_VISIBLE_DEVICES",
                          args.gpus or ",".join(cfg.gpu_pairs[0:1]))
    from diffusers import Flux2Pipeline

    t0 = time.time()
    pipe = Flux2Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="balanced")
    pipe.set_progress_bar_config(disable=True)
    print(f"[render] pipeline ready ({time.time()-t0:.0f}s)", flush=True)

    gate = None
    try:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )
        gate = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
        gate.eval().to("cuda:0")
        print("[render] gate: person detector loaded", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[render] WARN gate unavailable ({e}); single-seed rendering",
              flush=True)

    def person_boxes(img: "Image.Image") -> list[tuple[float, float]]:
        if gate is None:
            return [(1.0, 1.0)]
        import numpy as np
        t = torch.from_numpy(np.asarray(img.convert("RGB"))).float() \
            .permute(2, 0, 1) / 255.0
        with torch.no_grad():
            out = gate([t.to("cuda:0")])[0]
        boxes = []
        for score, label, box in zip(out["scores"], out["labels"], out["boxes"]):
            if int(label) != 1 or float(score) < GATE_MIN_SCORE:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            if (y2 - y1) * (x2 - x1) >= GATE_MIN_AREA * H * W \
                    and y2 >= GATE_MIN_BOTTOM * H:
                boxes.append((float(score),
                              (y2 - y1) * (x2 - x1) / (H * W)))
        return boxes

    def gate_target(mode: str) -> int | None:
        # personN → 恰 N 人（person0=0 人；含 NPC/守卫的 case 为 person2）；
        # personN_soft → 不做人数门控（如 Case 31 稻草人易被误检），仅记录
        if mode.endswith("_soft"):
            return None
        try:
            return int(mode.removeprefix("person"))
        except ValueError:
            return 0

    log_path = args.frames_root / "_render_log.jsonl"
    args.frames_root.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a", encoding="utf-8")

    n_ok = n_fail = 0
    for n, row in enumerate(todo, 1):
        t0 = time.time()
        out = out_path(row)
        mode = row["gate"]
        want = gate_target(mode)   # None → 软门控：不做人数验收
        status, tries, used_seed = "ok", [], seeds[0]
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            best = None  # (gate_key, img, seed, n_persons)
            for seed in seeds:
                try:
                    gen = torch.Generator("cpu").manual_seed(seed)
                    img = pipe(prompt=row["prompt"],
                               height=H, width=W, num_inference_steps=STEPS,
                               guidance_scale=GUIDANCE,
                               generator=gen).images[0]
                except Exception as e:  # noqa: BLE001
                    tries.append({"seed": seed,
                                  "error": f"{type(e).__name__}: {e}"})
                    continue
                boxes = person_boxes(img)
                n_p = len(boxes)
                top = max((s for s, _ in boxes), default=0.0)
                tries.append({"seed": seed, "n_person": n_p,
                              "top": round(top, 3)})
                if want is None:      # 软门控：首张成功即收，人数仅记录
                    best = (0, img, seed, n_p)
                    break
                key = (0 if n_p == want else 1, -top if want == 1 else -n_p)
                if best is None or key < best[0]:
                    best = (key, img, seed, n_p)
                if n_p == want and (want == 0 or top >= GATE_PASS):
                    break
            if best is None:
                raise RuntimeError("all seed attempts failed")
            _, img, used_seed, n_p = best
            if want is not None and n_p != want:
                status = f"gate_fail:n_person={n_p}(want {want})"
                n_fail += 1
            exif = Image.Exif()
            exif[ExifTags.Base.Software] = "AI generated;flux2-dev"
            exif[ExifTags.Base.ImageDescription] = (
                f"seed={used_seed};steps={STEPS};guidance={GUIDANCE};"
                f"gate={mode};case={row['case']}")
            img.save(out, exif=exif, quality=95, subsampling=0)
            if status == "ok":
                n_ok += 1
        except Exception as e:  # noqa: BLE001
            status = f"{type(e).__name__}: {e}"
            n_fail += 1
            print(f"[render] ERROR {row['case']}: {status}",
                  file=sys.stderr, flush=True)
        logf.write(json.dumps({
            "n": n, "of": len(todo), "case": row["case"],
            "family": row["family"], "gate": mode, "out": str(out),
            "status": status, "sec": round(time.time() - t0, 1),
            "seed": used_seed, "tries": tries, "steps": STEPS,
            "size": f"{W}x{H}",
        }) + "\n")
        logf.flush()
        print(f"[render] {n}/{len(todo)} {row['case']} {status} "
              f"({time.time() - t0:.0f}s)", flush=True)

    logf.close()
    print(f"[render] done. ok={n_ok} fail={n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
