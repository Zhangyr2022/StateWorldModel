"""白模视频 → h3-a800 ref2va 批量生成(单次装载引擎,循环全部视频)。

用法(h3-a800 env;单次装载循环全部视频,替代逐条冷启动的 ./run.sh):
    python h3ref_batch.py --videos v1.mp4 v2.mp4 ... [--out-root OUT] [--duration 5]
    注意:脚本内部 os.chdir(H3_REPO),--out-root 与 --images-json 里的
    路径必须是绝对路径。

每条视频的 prompt 来自 --prompts-json / --prompt-override;两者都缺时回退
内置通用白模模板(响亮告警,勿用于 WBench 批量)。ref2va 条件于白模视频。
"""
import argparse
import json
import os
import sys
from pathlib import Path

# h3-a800 仓库根:H3_REPO 可覆盖;缺省按 workspace 并排布局推导(box2clay→spatial_proxy→workspace)
H3 = Path(os.environ.get(
    "H3_REPO", Path(__file__).resolve().parents[2] / "h3-a800"))
sys.path.insert(0, str(H3))

# 通用白模 prompt(白模 ref 在前,H3 依据它补全真实场景)
DEFAULT_PROMPT = (
    "Reference video: a plain white 3D layout model of the scene with simple box buildings "
    "and blocky figures, camera slowly moving forward. Generate the real cinematic live scene "
    "that matches this exact 3D layout and camera move: same buildings in the same positions, "
    "same people walking at the same spots, fully textured and lit, photorealistic game cinematics, "
    "smooth steady camera."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out-root",
                    default=str(Path(__file__).resolve().parent / "results" / "h3"))
    ap.add_argument("--duration", type=int, default=5, choices=(5, 10, 15))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--prompt-override", default=None,
                    help="覆盖全部场景 prompt(用于对比实验)")
    ap.add_argument("--prompts-json", default=None,
                    help='按视频 stem 覆盖 prompt 的 json 文件:{"<stem>": "<prompt>"}')
    ap.add_argument("--images-json", default=None,
                    help='视频路径→首帧图片 的映射 json(--first-frame 非 none 时必给,'
                         '路径须绝对;如 WBench case)')
    ap.add_argument("--suffix", default="",
                    help="输出文件名后缀(如 _v2)")
    ap.add_argument("--first-frame", choices=("none", "i2v", "ref2va"), default="none",
                    help="i2v=原图作精确首帧(无白模 ref);ref2va=原图作第一 reference + 白模视频")
    args = ap.parse_args()
    overrides = json.loads(Path(args.prompts_json).read_text()) if args.prompts_json else {}
    images_map = json.loads(Path(args.images_json).read_text()) if args.images_json else {}

    os.chdir(H3)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
    from h3_runtime.engine import MiniMaxH3Inference

    ref2va = args.first_frame != "i2v"
    engine = MiniMaxH3Inference(
        "./checkpoints/MiniMax-H3",
        "./checkpoints/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
        if ref2va else
        "./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors",
        attention_backend="dense",
        task="ref2va" if ref2va else "t2v",
        width=args.width, height=args.height,
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    try:
        for v in args.videos:
            v = Path(v).resolve()
            stem = v.parent.name + "_" + v.stem + args.suffix
            out = out_root / f"{stem}.mp4"
            if out.exists():
                print(f"[h3] skip {stem}(已存在)")
                continue
            key = v.parent.name + "_" + v.stem   # 与输出 stem 一致
            prompt = overrides.get(key) or overrides.get(v.stem)
            if not prompt:
                if overrides and not args.prompt_override:
                    # 静默回退是已知陷阱:内置通用模板只适用于白模快速人工测试,
                    # WBench 批量 key 不匹配会生成错误 prompt。
                    print(f"[h3][WARN] {key} 不在 --prompts-json,回退内置通用白模 prompt")
                prompt = args.prompt_override or DEFAULT_PROMPT
            if args.first_frame == "ref2va":
                prompt = (
                    "The first reference is the exact opening frame of the video: begin the video on "
                    "it, reproducing it faithfully, then animate naturally from there while keeping "
                    "its exact visual style. "
                ) + prompt
            asset_img = None
            if args.first_frame != "none":
                asset_img = (Path(images_map[str(v)])
                             if images_map and str(v) in images_map else None)
            if args.first_frame != "none" and (asset_img is None or not Path(asset_img).exists()):
                print(f"[h3] 找不到 {v.parent.name} 的原图,跳过")
                continue
            if args.first_frame == "i2v":
                # i2v:原图精确首帧;orbit 视频提示弧线运镜
                move = ("camera arcs sideways around the scene"
                        if v.stem == "orbit" else "camera pushes slowly forward")
                prompt += f" {move[0].upper() + move[1:]}."
                print(f"[h3] {stem} ← i2v first frame {asset_img.name}")
                media = engine.generate(prompt, duration=args.duration, seed=args.seed,
                                        image=str(asset_img))
            else:
                print(f"[h3] {stem} ← ref {v}" +
                      (f" + first-frame img {asset_img.name}" if asset_img else ""))
                refs = []
                if asset_img is not None:
                    refs.append(MiniMaxH3Reference(image=str(asset_img)))
                refs.append(MiniMaxH3Reference(video=str(v)))
                media = engine.generate(prompt, duration=args.duration, seed=args.seed,
                                        references=refs)
            if media is not None:
                media.save(str(out))
                manifest.append({"ref": str(v), "out": str(out), "prompt": prompt,
                                 "duration": args.duration, "seed": args.seed})
                print(f"[h3] done → {out}")
            else:
                print(f"[h3] rank!=0,跳过保存")
    finally:
        engine.close()
        if manifest:  # 非 0 rank 的 media 为 None,别把 rank 0 的 manifest 覆盖掉
            (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"[h3] 全部完成:{len(manifest)} 条,manifest 在 {out_root}/manifest.json")


if __name__ == "__main__":
    main()
