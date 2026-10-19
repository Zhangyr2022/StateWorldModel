#!/usr/bin/env python3
"""首帧蓝框 vs det_vis 框对齐验收(2026-09-23 修复后)。

对每个 case:取 walk_W.mp4 首帧,蓝色最大连通域(排除起点标记)与 det.json
box2d_xyxy 归一化对比,并产出 det_vis | 首帧 并排图供目测。
用法:python3 check_first_frame_align.py case_11_elephant case_18_horse ...
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def blue_bbox_main(img):
    """蓝色像素最大连通域的 bbox(排除起点标记等小块)。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (100, 60, 60), (130, 255, 255))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    x, y, w, h, area = stats[1 + np.argmax(stats[1:, 4])]
    return x, y, x + w, y + h


def main() -> None:
    cases = sys.argv[1:] or [p.name for p in (HERE / "results" / "subject").iterdir()
                             if p.is_dir() and not p.name.endswith("_ab")]
    for d in cases:
        det_p = HERE / "results" / "subject" / d / "det.json"
        vid_p = HERE / "results" / "subject" / d / "walk_W.mp4"
        det = json.loads(det_p.read_text())
        # 与渲染一致:取该 case 渲染时的 subject 类(多数=最高分;马 case 用 horse)
        subj_cls = {"case_18_horse": "horse"}.get(d)
        objs = det["objects"]
        o = (max([x for x in objs if x["class"] == subj_cls], key=lambda x: x["score"])
             if subj_cls else max(objs, key=lambda x: x["score"]))
        W, H = det["W"], det["H"]
        x1, y1, x2, y2 = o["box2d_xyxy"]
        import subprocess
        first = HERE / "results" / "subject" / d / "_first_W.png"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(vid_p),
                        "-frames:v", "1", "-update", "1", str(first)], check=True)
        img = cv2.imread(str(first))
        bh, bw = img.shape[:2]
        bb = blue_bbox_main(img)
        if bb is None:
            print(f"{d}: 首帧未检测到蓝框!")
            continue
        bx1, by1, bx2, by2 = bb
        # 渲染分辨率与原图等比(宽 864),归一化对比
        det_n = dict(w=(x2 - x1) / W, h=(y2 - y1) / H, cx=(x1 + x2) / 2 / W,
                     top=y1 / H, bottom=y2 / H)
        ren_n = dict(w=(bx2 - bx1) / bw, h=(by2 - by1) / bh, cx=(bx1 + bx2) / 2 / bw,
                     top=by1 / bh, bottom=by2 / bh)
        err = {k: ren_n[k] - det_n[k] for k in det_n}
        print(f"{d}:")
        print(f"  det  w={det_n['w']:.3f} h={det_n['h']:.3f} cx={det_n['cx']:.3f} "
              f"top={det_n['top']:.3f} bottom={det_n['bottom']:.3f}")
        print(f"  首帧 w={ren_n['w']:.3f} h={ren_n['h']:.3f} cx={ren_n['cx']:.3f} "
              f"top={ren_n['top']:.3f} bottom={ren_n['bottom']:.3f}")
        print(f"  偏差  " + " ".join(f"{k}={v:+.3f}" for k, v in err.items()))
        # 并排图(det_vis | 首帧)供目测
        vis = cv2.imread(str(HERE / "results" / "subject" / d / "det_vis.png"))
        hgt = 480
        vis_r = cv2.resize(vis, (int(vis.shape[1] * hgt / vis.shape[0]), hgt))
        pair = np.hstack([vis_r, cv2.resize(img, (int(bw * hgt / bh), hgt))])
        out = HERE / "results" / "subject" / d / "_align_check.png"
        cv2.imwrite(str(out), pair)
        print(f"  并排图 → {out}")


if __name__ == "__main__":
    main()
