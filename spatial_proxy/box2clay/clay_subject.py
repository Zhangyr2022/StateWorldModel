"""WildDet3D det.json → 简化白模控制视频:目标对象蓝色盒体 + 地面。

极简场(2026-09-23 定稿):场景只含——
    1. subject 最高分 1 实例的蓝色盒体(Toon 蓝 + 棱线圆柱黑描边,轴对齐)
    2. 灰色无限地面 + 1m 网格(5m 主线);左前/右前两个静止红色参考长方体
       (--no-corner-refs 关闭;2026-09-23 起替代原蓝色起点标记的锚定职责)
输出 preview.png(单帧)或 --action 的 walk_<动作>.mp4,相机内参与原图逐像素对齐。

主体锚定(2026-09-23 修订:首帧蓝盒近面投影 = det 2D 框,构造性保证):
    距离 z = f·H_real/2D框像素高(针孔);
    盒宽 w = (x2-x1)·z/f(投影拟合 det 像素宽),盒高 = H_real;
    盒深 l = min(检测短边·k, w),k=z/box3d_cam.cz,中心放 z+l/2 使近面贴 z;
    横向 x_cv = (u_center-cx)·z/f;相机高 hcam = (v_bottom-cy)·z/f(针孔底边行)。
    弃用 WildDet3D yaw(四元数有 ~90° 系统偏移,曾致长边朝相机、近端透视放大,
    首帧蓝盒比 det 框偏大且底边出画)与 VGGT 地面高度(采纳会挪动盒子竖直位置,
    破坏逐像素对齐;depth_vggt 仅离线校验用,渲染不读)。
Blender dims 轴序 = (w, H, l):x=屏幕横向, y=竖直, z=深度。

用法(bpy311 env):
    clay_subject.py <det_dir> <out_dir> [--subject person] [--subject-height 1.7]
                    [--raw-3d] [--samples 48] [--width 864]
                    [--no-edges] [--edge-width 2] [--freestyle] [--edge-tubes]
                    [--frames 120] [--fps 24] [--push 0.70]
                    [--action W|S|A|D|left|right|up|down|组合|all16]
视频:WBench third_person 动作,主体世界平移 + 相机解析 euler 跟随/环绕。
(旧跟拍 rig 路线 --video/--path 与逐帧 PNG 路线 --png-frames 已归档:
 已随公开版整理移除)
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import bpy

HERE = Path(__file__).resolve().parent


def _load(name):
    """同目录兄弟模块按路径装载(渲染函数库 boxes2clay)。"""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b2c = _load("boxes2clay")   # 渲染库:setup_render/toon/flat/add_box/edge-tubes/enable_edges/...

# 锚定用的真实高度(米),未知类回落 1.7
SUBJECT_HEIGHTS = {
    "person": 1.7, "bicycle": 1.1, "car": 1.5, "motorcycle": 1.5, "bus": 3.2,
    "train": 3.8, "truck": 3.5, "bird": 0.3, "cat": 0.4, "dog": 0.6, "horse": 1.8,
    "sheep": 1.1, "cow": 1.7, "elephant": 3.2, "bear": 1.5, "zebra": 1.5,
    "giraffe": 5.0, "backpack": 0.5, "umbrella": 0.9, "suitcase": 0.6,
    "sports ball": 0.25, "bottle": 0.3, "chair": 0.9, "couch": 0.85,
    "potted plant": 0.8, "tv": 0.7, "laptop": 0.35, "bench": 0.9, "vase": 0.4,
}
BLUE = (0.12, 0.32, 0.85)  # 加深一档:AgX+白光下 (66,133,244) 会偏淡


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("det_dir", help="detect_boxes.py 输出的单图目录(含 det.json)")
    p.add_argument("out_dir", help="输出目录(preview.png)")
    p.add_argument("--subject", default="person", help="目标类别(det.json 里的 class 名)")
    p.add_argument("--subject-height", type=float, default=None,
                   help="目标对象真实高度(米);不填查内置表,未知类回落 1.7")
    p.add_argument("--raw-3d", action="store_true",
                   help="跳过针孔锚定,直接用 WildDet3D 的 box3d_cam(对照用)")
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--denoise", action="store_true",
                   help="开启 OPTIX 降噪(0.1→0.4s/帧;16spp 无降噪已无明显噪点)")
    p.add_argument("--engine", default="eevee", choices=["eevee", "cycles"],
                   help="eevee=实时引擎+Toon+反转壳描边(默认);cycles=路径追踪")
    p.add_argument("--freestyle", action="store_true",
                   help="bare 场也用 Freestyle 描边(定稿样例观感,~1.2s/帧;A/B 对照用)")
    p.add_argument("--edge-tubes", action="store_true",
                   help="描边改用 12 棱细圆柱+角球(所有 Mesh Edge 黑线,EEVEE 实时)")
    p.add_argument("--width", type=int, default=864,
                   help="渲染宽(高按原图比例;H3 画布 864×480,控制视频作 reference "
                        "进 VAE 前必缩到画布,864 即模型所见上限,1280 的像素全浪费)")
    p.add_argument("--no-edges", action="store_true", help="关闭黑色描边(Freestyle)")
    p.add_argument("--edge-width", type=float, default=2.0)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--push", type=float, default=0.70,
                   help="W/S/A/D 移动距离 = 主体距离 z × 该系数(2026-09-23 定稿 0.70,"
                        "0.35 实测运动幅度偏弱)")
    p.add_argument("--bare", action="store_true",
                   help="极简场:地面换无限大 + 1m 网格(5m 主线),无其他参照物"
                        "(极简测试场样式)")
    p.add_argument("--no-corner-refs", action="store_true",
                   help="不加左前/右前两个静止红色参考长方体(默认加,反衬主体"
                        "运动/运镜;位置避开主体全程扫掠区,首帧完整入画)")
    p.add_argument("--action", default=None,
                   choices=ACTIONS + ["all16"],
                   help="WBench third_person 动作(16 种,语义见 ACTIONS 注释);"
                        "all16=全量各出一条 walk_<动作>.mp4('+'→'_')")
    p.add_argument("--no-preview", action="store_true",
                   help="跳过 preview.png(并行渲染同目录时防写冲突)")
    return p.parse_args()


def pick_subject(record, subject):
    """class==subject 的实例里取分最高的 1 个。"""
    cands = [o for o in record["objects"]
             if o["class"] == subject and len(o["box3d_cam"]) == 10]
    if not cands:
        raise SystemExit(f"det.json 里没有 class={subject!r} 的实例"
                         f"(该图 texts={record.get('texts')});换 --subject 或检查 --texts")
    return max(cands, key=lambda o: o["score"])


# ---------------------------------------------------------------- 主体锚定
def anchor_dims(record, o, h_real, raw_3d=False):
    """主体距离与尺寸(针孔锚定)→ (z, w, l, h)(相机系米)。

    2026-09-23 修订:放弃 WildDet3D 四元数(实测存在 ~90° 系统性偏移,长边
    转向相机后近端角点透视放大,首帧蓝框比 det 框偏大且底边出画)。改为
    按 det 2D 框投影拟合:
      w = (x2-x1)·z/f   —— 投影像素宽 = det 像素宽(构造性)
      h = h_real        —— 投影像素高 = det 像素高(z 由针孔反推,构造性)
      l = min(bw,bl)·k  —— 保留检测的深度量级(orbit 动作看侧面),中心放
                           z + l/2 使近面贴锚定深度(place),近面矩形=det 框。
    """
    f = record["focal"]
    x1, y1, x2, y2 = o["box2d_xyxy"]
    b3 = o["box3d_cam"]
    bw, bl, bh = b3[3], b3[4], b3[5]  # 槽位 [W, L, H(竖直)]
    cz_wd = max(b3[2], 0.1)
    if raw_3d:
        return b3[2], bw, bl, bh
    ph = max(y2 - y1, 1.0)
    z = float(np.clip(f * h_real / ph, 0.5, 300.0))  # 针孔:像素高 ↔ 真实高
    k = z / cz_wd
    w = (x2 - x1) * z / f                     # 投影拟合宽(det 像素宽 ↧ 真实米)
    l = float(min(min(bw, bl) * k, w))        # 深度取检测短边(不超投影宽)
    return z, w, l, h_real


def feet_row_hcam(record, o, z):
    """针孔底边行反推相机高度:hcam = (v_bottom-cy)·z/f,夹紧到合理区间。"""
    cy, f, H = record["cy"], record["focal"], record["H"]
    v_bottom = min(o["box2d_xyxy"][3], H - 1)
    hcam = (v_bottom - cy) * z / f if v_bottom > cy else float("nan")
    if not np.isfinite(hcam) or hcam <= 0.05:  # 主体在地平线上方=退化框
        print("[clay-subject] 警告:底边行反推 hcam 无效(主体验证退化),回退 1.6m")
        hcam = 1.6
    lo, hi = 0.1, (3.2 if o["class"] == "person" else 30.0)  # 下限 0.1:低机位(贴地拍猫/狗)合法,夹紧会破坏首帧对齐
    return float(np.clip(hcam, lo, hi)), (lo, hi)


# ---------------------------------------------------------------- 放置/渲染
def center_x_cv(record, o, z):
    """2D 框中心 → 3D 中心 x(相机系)。"""
    f, cx = record["focal"], record["cx"]
    x1, _, x2, _ = o["box2d_xyxy"]
    return ((x1 + x2) / 2 - cx) / f * z


def place(x_cv, z, w, l, h, hcam, o=None):
    """→ (center_b, dims_b, quat_b):底面落地、轴对齐(W 朝相机)。

    2026-09-23 修订:弃用 WildDet3D yaw(检测四元数有 ~90° 系统性偏移,曾致
    长边朝向相机、近端透视放大,首帧蓝框偏大且底边出画)。中心放 z+l/2,
    使盒子**近面**恰在锚定深度 z → 近面矩形=det 2D 框(与 anchor_dims 投影
    拟合配套)。o 参数保留兼容(合成柱/检测物不再区分旋转)。"""
    y_cv = hcam - h / 2                   # 底面落地、竖直居中(相机系 y 向下)
    center_b = (x_cv, -y_cv, -(z + l / 2))  # CV → Blender:近面贴 z
    dims_b = (w, h, l)                    # Blender x=屏幕横向(CV x), y=竖直, z=深度(CV z)
    return center_b, dims_b, None


# ---------------------------------------------------------------- 参考长方体
def corner_refs(record, center_b, dims_b, z, dist, hcam):
    """左前/右前两个静止红色参考长方体 → [(center_b, dims_b, quat_b, label)]。

    静止不动,靠视差反衬主体平移与相机公转。位置两条硬约束:
      1. 全程不碰撞:横向净空 lat 覆盖 A/D 满行程(dist),W/S 纵移即使扫过
         参考体深度带也横向错开;left/right/up/down 主体静止亦然;
      2. 首帧完整入画:长焦(水平画幅窄)或低机位近距(画幅底边照不到
         那么近的地面)时逐级加深前向间隙 fg——更深则画幅更宽而横向净空
         不变,单调收敛。横向偏移 > 半盒宽,不碰 det 框矩形,首帧对齐不受影响。
    """
    f, W, H, cy = record["focal"], record["W"], record["H"], record["cy"]
    tan_h = (W / 2) / f                        # 首帧水平半画幅(无量纲,×深度=半宽)
    w, h, l = dims_b
    h_r = float(np.clip(0.65 * h, 0.4, 1.8))   # 只用于足迹尺寸;实际高度无限(见下)
    s = max(0.25, 0.35 * h_r)                  # 方截面随主体档位
    H_INF = 60.0                               # 「无限高」:任何机位/±45° 俯仰都出画
    lat = w / 2 + dist + s / 2 + 0.4           # 横向净空(A/D 满行程 + 安全边)
    d_vis = hcam * f / max((H - 1) - cy, 1.0)  # 画幅底边照到的最近地面深度
    # 屏占比定深:让红柱落在半宽的 0.70 处(太近会贴画面左右边缘),深度加深
    # 不影响 A/D 净空(扫掠是横向的)。zc = lat / (0.70·tan_h)
    zc = max(lat / (0.70 * tan_h), z + l + 0.5 + s / 2, 1.15 * d_vis + s / 2)
    out = []
    for side in (-1.0, 1.0):
        c = (center_b[0] + side * lat, -hcam + H_INF / 2, -zc)
        out.append((c, (s, H_INF, s), None, "red"))
    print(f"[clay-subject] corner-refs: lat={lat:.2f}m zc={zc:.2f}m "
          f"footprint={s:.2f}m 高=无限({H_INF:.0f}m) 屏占比0.70 左右对称")
    return out


# ------------------------------------------------------------- WBench 动作(16 种)
# 语义对齐 WBench src/models/text/prompt_builder.py THIRD_PERSON_ACTIONS
# (third 表缺 S+A,按 first 表补齐):
#   W/S/A/D     主体世界平移(方向锚定世界轴),相机刚体跟随(相对位姿锁定);
#   left/right  主体静止,相机绕主体中轴上距相机最近点(同高)水平公转,
#               left 向相机像方左侧(-X)→ 看到主体左侧面,right 反之;
#   up/down     同中心竖直弧 + 俯仰补偿:up 相机升起+俯视,down 下降+仰视
#               (down 受地面钳制 φ_max=asin((hcam-margin)/r));
#   组合键      分量同时生效(对角平移 1.41×;W+left=主体前进+相机同时公转)。
ACTIONS = ["W", "S", "A", "D", "left", "right", "up", "down",
           "W+A", "W+D", "S+A", "S+D", "W+left", "W+right", "W+up", "W+down"]
YAW_ARC = 45.0        # left/right 水平公转角(°)
PITCH_ARC = 45.0      # up/down 仰角变化(°)
GROUND_MARGIN = 0.25  # down 相机离地最小间隙(米)
MOVE_DIR = {"W": (0.0, 0.0, -1.0), "S": (0.0, 0.0, 1.0),
            "A": (-1.0, 0.0, 0.0), "D": (1.0, 0.0, 0.0)}


def animate_action(cam, subj, center_b, action, dist, hcam):
    """WBench third_person 单动作:逐帧解析算主体/相机位姿(无 rig,防滚转)。

    相机初始位姿=原点朝 -Z(与原图构图对齐);主体中轴过 center_b 竖直。
    O(f) = 中轴上与相机同高的点,随主体平移;相机位姿 = O + R_y(θ)·V(φ),
    V(φ) 为初始偏移在含轴竖直面内的升起旋转;姿态 euler=(-φ, θ, 0)。
    """
    sc = bpy.context.scene
    s, e = sc.frame_start, sc.frame_end
    keys = action.split("+")
    disp = tuple(sum(MOVE_DIR[k][i] for k in keys if k in MOVE_DIR)
                 for i in range(3))
    # 2026-09-23 修正:left/right 曾反号(+θ 实际绕出逆时针,与 WBench/prompt 的
    # "right=clockwise / left=counterclockwise" 相反)。现 left=+θ(逆时针)、right=-θ。
    theta = (-YAW_ARC if "right" in keys else
             YAW_ARC if "left" in keys else 0.0)
    r = math.hypot(center_b[0], center_b[2])      # 相机到中轴水平距离(水平半轴)
    phi_amp = 0.0
    rv = r                                        # 竖直半轴(默认圆弧)
    if "up" in keys:
        phi_amp = PITCH_ARC
    elif "down" in keys:
        phi_amp = -PITCH_ARC
        # 地面净空不足时压扁成椭圆:竖直半轴收缩,水平扫动保持,仍转满 45°
        rv = min(r, max(hcam - GROUND_MARGIN, 0.05) / max(math.sin(math.radians(PITCH_ARC)), 1e-6))
    v0x, v0z = -center_b[0], -center_b[2]         # C0−O 的 xz 偏移
    for f in range(s, e + 1):
        t = b2c.ease((f - s) / max(e - s, 1))
        dx, dy, dz = (disp[i] * dist * t for i in range(3))
        th = math.radians(theta * t)
        ph = math.radians(phi_amp * t)
        subj.location = (center_b[0] + dx, center_b[1] + dy, center_b[2] + dz)
        subj.keyframe_insert("location", frame=f)
        vx, vz = v0x * math.cos(ph), v0z * math.cos(ph)   # 椭圆弧(水平半轴 r,竖直半轴 rv)
        vy = rv * math.sin(ph)
        cx = vx * math.cos(th) + vz * math.sin(th)        # 绕竖轴转 θ
        cz = -vx * math.sin(th) + vz * math.cos(th)
        pitch = -math.atan2(rv * math.sin(ph), r * math.cos(ph))  # 俯仰补偿盯住中轴点
        cam.location = (center_b[0] + dx + cx, vy, center_b[2] + dz + cz)
        cam.rotation_euler = (pitch, th, 0.0)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)


def build_scene(record, args, center_b, dims_b, quat_b, hcam, refs):
    """清场重建:材质/灯光/地面/描边/蓝框/参照物/相机。返回 (subject_obj, cam)。"""
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for blk in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for b in list(blk):  # 清孤儿数据防 .001 堆积
            if b.users == 0:
                blk.remove(b)
    W, H = record["W"], record["H"]
    scale = args.width / W
    res_w = int(round(W * scale / 2)) * 2
    res_h = int(round(H * scale / 2)) * 2
    b2c.setup_render(res_w, res_h, args.samples, frames=args.frames, fps=args.fps,
                     denoise=args.denoise, engine=args.engine)
    mat_subj = (b2c.toon_material("ToonSubject", BLUE)
                if args.engine == "eevee"
                else b2c.clay_material("ClaySubject", BLUE))
    mat_border = b2c.border_material("ClayBorder", engine=args.engine)
    mat_edge = b2c.flat_material("ClayEdge", (0.0, 0.0, 0.0))  # 纯0自发光=Freestyle级黑
    mat_ref = b2c.clay_material("ClayRef", (0.86, 0.85, 0.82))
    mat_red = b2c.toon_material("RefRed", (0.78, 0.08, 0.10))  # 参考长方体:Toon 红与主体同风格
    mat_ground = b2c.clay_material("ClayGround", (0.80, 0.80, 0.80), rough=0.8)
    b2c.add_sun()
    if args.bare:
        # 极简场:无限地面 + 1m 网格(5m 主线),无参照物(bare_stage 同款;
        # 原蓝色起点标记已删——红色参考长方体接管锚定/参照职责)
        # eevee 无 Freestyle:网格加深加粗补偿运动视差对比度
        if args.engine == "eevee":
            gl, gm = 0.0, 0.0
            t_line, t_major = 0.045, 0.09
        else:
            gl, gm = 0.45, 0.30
            t_line, t_major = 0.03, 0.06
        # eevee 网格用自发光纯黑(AgX 会把深色材质提亮,达不到 Freestyle 的黑度)
        mat_line = (b2c.flat_material("BareGrid", (gl, gl, gl))
                    if args.engine == "eevee"
                    else b2c.clay_material("BareGrid", (gl, gl, gl), rough=0.8))
        mat_major = (b2c.flat_material("BareGridMajor", (gm, gm, gm))
                     if args.engine == "eevee"
                     else b2c.clay_material("BareGridMajor", (gm, gm, gm), rough=0.8))
        b2c.add_box("BareGround", (0, -hcam - 0.05, 0), (4000, 0.1, 4000), None, mat_ground)
        for i in range(-20, 21):
            major = (i % 5 == 0)
            mat = mat_major if major else mat_line
            t = t_major if major else t_line
            b2c.add_box(f"GridX{i:+03d}", (i, -hcam + 0.005, 0), (t, 0.01, 40), None, mat)
            b2c.add_box(f"GridZ{i:+03d}", (0, -hcam + 0.005, i), (40, 0.01, t), None, mat)
    else:
        b2c.add_ground_flat(hcam, mat_ground)
    if not args.no_edges and (not args.bare or args.freestyle):
        b2c.enable_edges(args.edge_width)
    subj = b2c.add_box("Subject", center_b, dims_b, quat_b, mat_subj)
    if args.bare and not args.no_edges and not args.freestyle:
        # 实时描边二选一(Freestyle ~1.2s/帧固定开销,弃用):
        #   --edge-tubes:12 棱细圆柱+角球贴盒子棱线(所有 Mesh Edge 成黑线);
        #   默认反转壳:放大 δ 黑壳,EEVEE 翻法线+backface culling,Cycles Backfacing 混透明。
        f_px = record["focal"] * res_w / W
        delta = 2.5 * (-center_b[2]) / f_px
        if args.edge_tubes:
            border = b2c.add_edge_tubes("SubjectEdges", center_b, dims_b, quat_b,
                                        mat_edge, radius=delta)
        else:
            border = b2c.add_box("SubjectBorder", center_b,
                                 tuple(d + 2 * delta for d in dims_b), quat_b,
                                 mat_border, flip=(args.engine == "eevee"))
        # 挂到主体上随其平移(W/A/S/D);matrix_parent_inverse 保持世界位姿不变
        # (坑:须先 view_layer.update 再取逆,否则 stale 单位阵导致双重变换)
        bpy.context.view_layer.update()
        border.parent = subj
        border.matrix_parent_inverse = subj.matrix_world.inverted()
    f_px = record["focal"] * res_w / W
    for i, (c, d, q, name) in enumerate(refs):
        b2c.add_box(f"Ref{i:02d}", c, d, q, mat_red if name == "red" else mat_ref)
        if name == "red" and not args.no_edges:
            # 红柱黑描边(与蓝盒同款棱线圆柱;半径按其深度定标,与蓝盒观感一致)
            b2c.add_edge_tubes(f"RefEdge{i:02d}", c, d, q, mat_edge,
                               radius=2.5 * (-c[2]) / f_px)
    return subj, b2c.make_cam("Cam", record["focal"], (W, H))


def main():
    args = parse_args()
    det_dir, out_dir = Path(args.det_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads((det_dir / "det.json").read_text())
    W, H = record["W"], record["H"]

    o = pick_subject(record, args.subject)
    h_real = args.subject_height or SUBJECT_HEIGHTS.get(args.subject, 1.7)
    if args.subject_height is None and args.subject not in SUBJECT_HEIGHTS:
        print(f"[clay-subject] 主题 '{args.subject}' 无内置高度,按 1.7m 锚定(建议 --subject-height)")

    z, w, l, h = anchor_dims(record, o, h_real, raw_3d=args.raw_3d)
    x_cv = center_x_cv(record, o, z)
    hcam, (lo, hi) = feet_row_hcam(record, o, z)

    center_b, dims_b, quat_b = place(x_cv, z, w, l, h, hcam, o)
    print(f"[clay-subject] {det_dir.name}: subject={args.subject} score={o['score']:.2f} "
          f"box={tuple(round(v, 2) for v in dims_b)}m z={-center_b[2]:.2f}m "
          f"hcam={hcam:.2f}m (feet-row 针孔){' [raw-3d]' if args.raw_3d else ''}")

    dist = args.push * z
    if args.action:
        # WBench 动作分支:场景只建一次,逐动作解析驱动(帧目录按动作名隔离,可并行)
        acts = ACTIONS if args.action == "all16" else [args.action]
        print(f"[clay-subject] WBench 动作 {acts} dist={dist:.2f}m")
        refs = ([] if args.no_corner_refs
                else corner_refs(record, center_b, dims_b, z, dist, hcam))
        subj, cam = build_scene(record, args, center_b, dims_b, quat_b, hcam, refs)
        sc = bpy.context.scene
        if not args.no_preview:
            sc.frame_set(1)
            sc.camera = cam
            sc.render.filepath = str(out_dir / "preview.png")
            bpy.ops.render.render(write_still=True)
            print(f"[clay-subject] preview → {out_dir / 'preview.png'}")
        for act in acts:
            animate_action(cam, subj, center_b, act, dist, hcam)
            mp4 = b2c.render_cam(cam, out_dir, f"walk_{act.replace('+', '_')}")
            print(f"[clay-subject] walk_{act} → {mp4}")
        print(f"[clay-subject] DONE\n{out_dir / 'preview.png'}")


if __name__ == "__main__":
    main()
