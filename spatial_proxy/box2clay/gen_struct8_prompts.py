"""nav8 8 case × 现存控制视频 → 结构化 prompt 小批量(与 flat nav8 对比)。

只换 prompt:控制视频直接复用 results/subject/<det_dir>/walk_<动作>.mp4,不重渲。
产物(新文件夹,方便与 results/h3_nav8/ 的 flat 成片对比):
  <out>/prompts_struct8.json   key → 结构化 prompt(h3ref_batch 约定)
  <out>/images_struct8.json    视频(绝对路径)→ case 原图
  <out>/prompts_review.txt     人读版(每条视频用的 prompt)
  <out>/videos_struct8.txt     控制视频绝对路径清单(--videos 直接 cat)
"""
import argparse
import json
import os
from pathlib import Path

from wbench_prompt import build_prompt

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get(
    "WBENCH_DATA", HERE.parents[1] / "WBench" / "data"))  # workspace 并排布局;WBENCH_DATA 可覆盖


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=None,
                    help="case id 子集(如 14 26 120 124);默认全部 8 case")
    ap.add_argument("--manifest", default=str(HERE / "results" / "wbench_det8.json"),
                    help="case 清单 json(wbench_det8.json 模式:case_id/det_dir/subject_class/actions)")
    ap.add_argument("--subject-root", default=str(HERE / "results" / "subject"),
                    help="控制视频所在根目录(新几何=subject_v2)")
    ap.add_argument("--out", default=str(HERE / "results" / "h3_prompts"),
                    help="prompt 三件套输出目录(实际批量均显式传 --out)")
    ap.add_argument("--style", choices=("structured", "flat"), default="flat",
                    help="flat=nav8 定稿扁平+红盒(现行默认,逐动作方向句);"
                         "structured=wbench_template.md 结构化(对照)")
    args = ap.parse_args()
    out = Path(args.out)
    det8 = json.loads(Path(args.manifest).read_text())
    if args.cases:
        det8 = [c for c in det8 if c["case_id"] in set(args.cases)]
    out.mkdir(parents=True, exist_ok=True)
    prompts, images, videos, review = {}, {}, [], []
    for c in det8:
        case = json.loads((BASE / "cases" / f"case_{c['case_id']}.json").read_text())
        for act in dict.fromkeys(c["actions"]):
            stem = f"walk_{act.replace('+', '_')}"
            video = Path(args.subject_root) / c["det_dir"] / f"{stem}.mp4"
            if not video.exists():
                print(f"[skip] 控制视频不存在: {video}")
                continue
            key = f"{c['det_dir']}_{stem}"
            prompts[key] = build_prompt(case, act, style=args.style)
            images[str(video.resolve())] = str((BASE / case["settings"]["initial_image"]).resolve())
            videos.append(str(video.resolve()))
            review.append(f"### {key}  (case_{c['case_id']} / {act})\n{prompts[key]}\n")
    (out / "prompts_struct8.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=1))
    (out / "images_struct8.json").write_text(json.dumps(images, indent=1))
    (out / "prompts_review.txt").write_text("\n".join(review))
    (out / "videos_struct8.txt").write_text("\n".join(videos))
    print(f"{len(prompts)} 条 prompt, {len(videos)} 条视频 → {out}")


if __name__ == "__main__":
    main()
