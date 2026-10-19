#!/usr/bin/env python3
"""第一人称 previz 控制视频:统一白模(无蓝盒,相机即主体)。

场景(所有 case 共用,与 case 无关的"统一白模"):
    无限灰地面 + 1m 网格(5m 主线)+ 前方左右两个静止红色参考盒(世界固定,
    为平移/摇镜/俯仰提供视差参照)。相机固定视高 1.6m(眼高),初始朝 -Z。

动作(WBench FIRST_PERSON_ACTIONS 语义,120 帧 ease):
    W 前推 1.6m | S 后拉 1.6m | A/D 左/右横移 1.6m
    left/right 原地 yaw ±45° | up/down 原地 pitch ±45°
    W+left/W+right/W+up/W+down = 前推 + 摇/仰同时

用法(bpy311 env):
    fp_previz.py <out_dir> --action W [--width 864] [--frames 120]
输出:<out_dir>/walk_<动作>.mp4('+'→'_')
"""
import argparse
import importlib.util
import math
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b2c = _load("boxes2clay")

EYE = 1.6            # 统一视高(米)
ARC = 45.0           # 摇镜/俯仰角度
REF_FP = 0.5         # 红柱 footprint
REF_H = 60.0         # 红柱「无限高」:任何俯仰都出画
REF_X, REF_Z = 2.2, 7.0   # 左右红柱位置(前方 7m,左右 ±2.2m)

DIR = {"W": (0.0, -1.0), "S": (0.0, 1.0), "A": (-1.0, 0.0), "D": (1.0, 0.0)}
YAW = {"left": ARC, "right": -ARC}      # Blender yaw:+θ 朝 -X(左)——left 即向左转(2026-09-23 修正曾反号)
PITCH = {"up": ARC, "down": -ARC}       # +θ 抬头


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--action", required=True,
                   help="W|S|A|D|left|right|up|down|W+left|W+right|W+up|W+down|all12")
    p.add_argument("--width", type=int, default=864)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--move", type=float, default=3.2,
                   help="W/S/A/D 平移/推拉行程(米);四向等幅(2026-09-23 定稿 3.2,"
                        "1.6 实测运动幅度偏弱)")
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


def build_scene(args):
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for blk in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for b in list(blk):
            if b.users == 0:
                blk.remove(b)
    W, H = 1376, 768                       # 与原图等比的标称内参基准
    scale = args.width / W
    res_w = int(round(W * scale / 2)) * 2
    res_h = int(round(H * scale / 2)) * 2
    b2c.setup_render(res_w, res_h, args.samples, frames=args.frames, fps=args.fps)
    mat_ground = b2c.clay_material("ClayGround", (0.80, 0.80, 0.80), rough=0.8)
    mat_red = b2c.toon_material("RefRed", (0.78, 0.08, 0.10))
    mat_edge = b2c.flat_material("Edge", (0.0, 0.0, 0.0))
    b2c.add_sun()
    b2c.add_box("Ground", (0, -EYE - 0.05, 0), (4000, 0.1, 4000), None, mat_ground)
    gl, gm, t_line, t_major = 0.0, 0.0, 0.045, 0.09
    mat_line = b2c.flat_material("Grid", (gl, gl, gl))
    mat_major = b2c.flat_material("GridMajor", (gm, gm, gm))
    for i in range(-20, 21):
        mat = mat_major if i % 5 == 0 else mat_line
        t = t_major if i % 5 == 0 else t_line
        b2c.add_box(f"GridX{i:+03d}", (i, -EYE + 0.005, 0), (t, 0.01, 40), None, mat)
        b2c.add_box(f"GridZ{i:+03d}", (0, -EYE + 0.005, i), (40, 0.01, t), None, mat)
    for side in (-1, 1):
        c = (side * REF_X, -EYE + REF_H / 2, -REF_Z)
        b2c.add_box(f"RefRed{'L' if side < 0 else 'R'}",
                    c, (REF_FP, REF_H, REF_FP), None, mat_red)
        # 黑描边(与蓝盒同款棱线圆柱;半径按柱深定标)
        b2c.add_edge_tubes(f"RefEdge{'L' if side < 0 else 'R'}", c,
                           (REF_FP, REF_H, REF_FP), None, mat_edge,
                           radius=2.5 * REF_Z / (args.width * W / 1376))
    return b2c.make_cam("Cam", W, (W, H))   # 焦距=原图长边像素,864 宽时等效 864px


def animate(cam, action, args):
    """相机本体运动:平移(ease 前半程加速)+ 摇/仰同步 ease。"""
    sc = bpy.context.scene
    s, e = sc.frame_start, sc.frame_end
    keys = action.split("+")
    dx, dz = (sum(DIR[k][i] for k in keys if k in DIR) for i in range(2))
    yaw = sum(YAW[k] for k in keys if k in YAW)
    pitch = sum(PITCH[k] for k in keys if k in PITCH)
    for f in range(s, e + 1):
        t = b2c.ease((f - s) / max(e - s, 1))
        cam.location = (dx * args.move * t, 0.0, dz * args.move * t)
        cam.rotation_euler = (math.radians(pitch * t), math.radians(yaw * t), 0.0)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acts = (["W", "S", "A", "D", "left", "right", "up", "down",
             "W+A", "W+D", "S+A", "S+D", "W+left", "W+right", "W+up", "W+down"]
            if args.action == "all16" else [args.action])
    cam = build_scene(args)
    sc = bpy.context.scene
    sc.camera = cam
    if not args.no_preview:
        sc.frame_set(1)
        sc.render.filepath = str(out_dir / "preview.png")
        bpy.ops.render.render(write_still=True)
        print(f"[fp-previz] preview → {out_dir / 'preview.png'}")
    for act in acts:
        animate(cam, act, args)
        mp4 = b2c.render_cam(cam, out_dir, f"walk_{act.replace('+', '_')}")
        print(f"[fp-previz] walk_{act} → {mp4}")


if __name__ == "__main__":
    main()
