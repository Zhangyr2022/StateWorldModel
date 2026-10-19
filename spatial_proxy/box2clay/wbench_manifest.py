"""生成 WBench third_person cases 的批量 manifest:
case_id / image / mask(可空)/ subject(COCO 类名或 type)/ height(尺度锚,米)
[/ world_term(YOLO-World 具体类别词,--out 后缀 _v2 时输出)]。

高度优先级:desc 关键词 > 类型默认 > 1.7。用法:
    python wbench_manifest.py --out results/wbench_manifest.tsv
"""
import argparse
import glob
import json
import os
from pathlib import Path

from PIL import Image

# WBench 数据目录:WBENCH_DATA 可覆盖;缺省按 workspace 并排布局推导
BASE = Path(os.environ.get(
    "WBENCH_DATA", Path(__file__).resolve().parents[2] / "WBench" / "data"))
# 类型 → (COCO 类名[YOLO 兜底用], 默认高度米)
TYPE_DEFAULT = {"human": ("person", 1.7), "robot": ("person", 1.7),
                "vehicle": ("car", 1.5), "animal": ("dog", 0.6), "other": ("person", 1.2)}
# desc 关键词 → 高度(先命中先用)
KEYWORD_HEIGHTS = [
    ("giraffe", 5.0), ("elephant", 3.2), ("camel", 2.2), ("horse", 1.8), ("cow", 1.5),
    ("cat", 0.35), ("kitten", 0.3), ("puppy", 0.4), ("dog", 0.6), ("bird", 0.3),
    ("drone", 0.4), ("jeep", 1.8), ("boat", 1.2), ("gnome", 0.4), ("mermaid", 1.5),
    ("fairy", 1.6), ("puppet", 1.2), ("ghost", 1.0), ("child", 1.2), ("kid", 1.2),
    ("robot", 1.7),
]
# desc 关键词 → YOLO-World 具体类别词(开放词表,不限词表;先命中先用)
WORLD_TERMS = [
    "dog", "cat", "camel", "elephant", "cow", "horse", "sheep", "bird", "rabbit",
    "robot", "cyborg", "android", "mermaid", "fairy", "gnome", "puppet", "ghost",
    "dragon", "jeep", "car", "boat", "ship", "drone", "cart", "wagon", "bicycle",
    "samurai", "ninja", "knight", "dancer", "traveler", "wizard", "archer",
    "girl", "boy", "woman", "man", "child", "person",
]


def _has_word(desc, word):
    import re
    return re.search(rf"\b{re.escape(word)}s?\b", desc) is not None  # 允许复数


def guess_height(subj_type, desc):
    d = (desc or "").lower()
    for kw, h in KEYWORD_HEIGHTS:
        if _has_word(d, kw):
            return h
    return TYPE_DEFAULT.get(subj_type, ("person", 1.7))[1]


def guess_world_term(desc, coco):
    d = (desc or "").lower()
    hits = [w for w in WORLD_TERMS if _has_word(d, w)]
    if not hits:
        return coco
    specific = hits[0]
    return specific if specific == coco else f"{specific},{coco}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wbench_manifest.tsv")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(BASE / "cases" / "*.json"))):
        d = json.load(open(f))
        s = d.get("settings", {})
        if s.get("perspective") != "third_person":
            continue
        cid = d["id"]
        img = os.path.join(BASE, s["initial_image"]) if s.get("initial_image") else None
        subj = s.get("subject") or {}
        stype = subj.get("type") or "unknown"
        coco, h_default = TYPE_DEFAULT.get(stype, ("person", 1.7))
        height = guess_height(stype, subj.get("desc"))
        mask_path = None
        if s.get("subject_mask"):
            mp = os.path.join(BASE, s["subject_mask"])
            if os.path.exists(mp):
                m = Image.open(mp)
                if m.getextrema()[1] > 127:  # 有前景像素
                    mask_path = mp
        rows.append((cid, img or "", mask_path or "", stype, coco, height,
                     guess_world_term(subj.get("desc"), coco)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("case_id\timage\tmask\tsubject_type\tcoco\tsubject_height\tworld_term\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    n_mask = sum(1 for r in rows if r[2])
    print(f"third_person cases: {len(rows)}, 有可用 mask: {n_mask}, 无 mask 走 YOLO/兜底: {len(rows) - n_mask}")
    print(f"manifest → {out}")


if __name__ == "__main__":
    main()
