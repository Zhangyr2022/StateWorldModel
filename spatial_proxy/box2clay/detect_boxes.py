"""WildDet3D 文本 prompt 推理:图片 → 人/建筑 3D box(JSON + 可视化 + 深度图)。

输出目录结构(每张图):
    <out>/<stem>/
        det.json      boxes3d + 内参 + 分数(相机坐标系,米)
        det_vis.png   3D wireframe 投影回原图
        depth.npy     模型预测的米制深度(原图分辨率,camera z)
        depth_vis.png 深度伪彩

用法:
    CUDA_VISIBLE_DEVICES=0 <wilddet3d env>/bin/python detect_boxes.py \
        <img_or_dir> [<img_or_dir> ...] --out results/subject [--texts person,building] [--conf 0.2]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# WildDet3D 仓:WILDDET3D_ROOT 可覆盖;缺省按 spatial_proxy/ 并排布局推导
WILD = Path(os.environ.get(
    "WILDDET3D_ROOT", Path(__file__).resolve().parents[1] / "WildDet3D"))
sys.path.insert(0, str(WILD))

from wilddet3d import build_model, preprocess  # noqa: E402
from wilddet3d.vis.visualize import draw_3d_boxes  # noqa: E402

CKPT = WILD / "ckpt" / "wilddet3d_alldata_all_prompt_v1.0.pt"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="图片文件或目录")
    p.add_argument("--out", required=True, help="输出根目录")
    p.add_argument("--texts", default="person,building", help="逗号分隔的检测类别(开放词表)")
    p.add_argument("--vocab-json", default=None,
                   help='按图覆盖词表的 json:{"<stem>": ["person", "temple", ...]}')
    p.add_argument("--conf", type=float, default=0.2, help="2D×3D 组合分数下限")
    p.add_argument("--conf-3d", type=float, default=0.1, help="3D 分数下限")
    return p.parse_args()


def cross_class_nms(objs, iou=0.6):
    """跨类别 2D NMS:不同词表词常命中同一栋楼,按分数贪心去重。"""
    keep = []
    for o in sorted(objs, key=lambda o: -o["score"]):
        x1, y1, x2, y2 = o["box2d_xyxy"]
        drop = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k["box2d_xyxy"]
            ix = max(0, min(x2, kx2) - max(x1, kx1))
            iy = max(0, min(y2, ky2) - max(y1, ky1))
            inter = ix * iy
            union = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
            if union > 0 and inter / union > iou:
                drop = True
                break
        if not drop:
            keep.append(o)
    return keep


def collect_images(inputs):
    imgs = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            imgs += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS)
        else:
            imgs.append(p)
    return imgs


def unpad_resize_depth(depth, padding, original_hw):
    """input_hw(padded) 深度图 → 裁掉 padding → resize 回原图分辨率。"""
    pl, pr, pt, pb = padding
    H, W = depth.shape
    d = depth[pt:H - pb if pb > 0 else H, pl:W - pr if pr > 0 else W]
    t = torch.from_numpy(np.ascontiguousarray(d))[None, None].float()
    t = torch.nn.functional.interpolate(t, size=tuple(original_hw), mode="nearest")
    return t[0, 0].numpy()


def colorize_depth(depth):
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    d = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    import matplotlib
    rgba = matplotlib.colormaps["turbo"](d)  # (H, W, 4)
    return (rgba[..., :3] * 255).astype(np.uint8)


def main():
    args = parse_args()
    texts = [t.strip() for t in args.texts.split(",") if t.strip()]
    vocab = json.loads(Path(args.vocab_json).read_text()) if args.vocab_json else {}
    images = collect_images(args.inputs)
    if not images:
        raise SystemExit(f"没有找到输入图片: {args.inputs}")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[det] {len(images)} 张图, texts={texts}, conf={args.conf}")

    model = build_model(
        checkpoint=str(CKPT),
        score_threshold=args.conf,
        score_3d_threshold=args.conf_3d,
        skip_pretrained=True,
    )

    for img_path in images:
        stem = img_path.stem
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        img = np.array(Image.open(img_path).convert("RGB"))
        data = preprocess(img)  # 默认内参:focal=max(H,W),主点居中

        texts_img = vocab.get(img_path.stem, texts)
        with torch.no_grad():
            boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = model(
                images=data["images"].cuda(),
                intrinsics=data["intrinsics"].cuda()[None],
                input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]],
                padding=[data["padding"]],
                input_texts=texts_img,
            )

        K = data["original_intrinsics"].numpy()
        H, W = data["original_hw"]
        objs = []
        for i in range(len(boxes3d[0])):
            objs.append({
                "class": texts_img[int(class_ids[0][i])],
                "score": float(scores[0][i]),
                "score_2d": float(scores_2d[0][i]),
                "score_3d": float(scores_3d[0][i]),
                "box2d_xyxy": [float(v) for v in boxes[0][i]],
                # 实际槽位 [cx,cy,cz, W,L,H, qw..qz]:竖直高度在下标 5(W<=L 已归一),
                # 以 vis4d boxes3d_to_corners 为准;勿按 (w,h,l) 想当然
                "box3d_cam": [float(v) for v in boxes3d[0][i]],
            })
        if len(texts_img) > 1:
            objs = cross_class_nms(objs, iou=0.6)
        # 同类严去重:开放词表下同一物体常被同类多框(不同深度假设),2D IoU>0.5 保留分高
        deduped, seen = [], []
        for o in sorted(objs, key=lambda o: -o["score"]):
            x1, y1, x2, y2 = o["box2d_xyxy"]
            dup = False
            for k in deduped:
                if k["class"] != o["class"]:
                    continue
                kx1, ky1, kx2, ky2 = k["box2d_xyxy"]
                ix = max(0, min(x2, kx2) - max(x1, kx1))
                iy = max(0, min(y2, ky2) - max(y1, ky1))
                inter = ix * iy
                union = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
                if union > 0 and inter / union > 0.5:
                    dup = True
                    break
            if not dup:
                deduped.append(o)
        objs = deduped
        objs.sort(key=lambda o: -o["score"])

        depth = None
        if depth_maps is not None and depth_maps[0] is not None:
            depth = unpad_resize_depth(
                depth_maps[0][0].float().cpu().numpy(), data["padding"], data["original_hw"])
            np.save(out_dir / "depth.npy", depth.astype(np.float32))
            Image.fromarray(colorize_depth(depth)).save(out_dir / "depth_vis.png")

        record = {
            "image": str(img_path), "W": int(W), "H": int(H),
            "K": K.tolist(), "focal": float(K[0, 0]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "texts": texts_img,
            "depth": "depth.npy" if depth is not None else None,
            "objects": objs,
        }
        (out_dir / "det.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
        draw_3d_boxes(
            image=img,
            boxes3d=boxes3d[0],
            intrinsics=K,
            scores_2d=scores_2d[0],
            scores_3d=scores_3d[0],
            class_ids=class_ids[0],
            class_names=texts_img,
            save_path=str(out_dir / "det_vis.png"),
        )
        n_person = sum(o["class"] == "person" for o in objs)
        n_bldg = len(objs) - n_person
        print(f"[det] {img_path.name}: {len(objs)} objs "
              f"(person={n_person}, other={n_bldg}), depth={'ok' if depth is not None else 'none'}")

    print("[det] 全部完成")


if __name__ == "__main__":
    main()
