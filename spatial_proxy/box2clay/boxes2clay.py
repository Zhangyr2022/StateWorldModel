"""box2clay 渲染函数库:场景搭建/材质/描边/相机/FFMPEG 直出(被 clay_subject、
fp_previz 按路径装载)。

CV 相机系(x 右, y 下, z 前) → Blender 系(x 右, y 上, z 后)的变换约定与
相机内参对齐(focal=max(H,W) 默认内参 → sensor_fit AUTO + lens 换算)见各函数。
"""
import math
from pathlib import Path

import numpy as np
import bpy

# ---- CV 相机系(x 右, y 下, z 前) → Blender 系(x 右, y 上, z 后):M = diag(1,-1,-1)
M = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------- 场景搭建
def setup_render(res_w, res_h, samples, frames, fps, denoise=False, engine="eevee"):
    sc = bpy.context.scene
    if engine == "eevee":
        # EEVEE Next 实时渲染(bpy4.2 headless 走 EGL);描边用反转壳+backface culling,
        # 不跑 Freestyle 线条计算(实测 ~1.2s/帧固定开销)
        try:
            sc.render.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            sc.render.engine = "BLENDER_EEVEE"
        sc.eevee.taa_render_samples = samples
    else:
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type != "CPU"
        sc.cycles.device = "GPU"
        sc.cycles.samples = samples
        # 降噪实测 0.1s→0.4s/帧(占大头);previz 参考视频高采样下无噪点,默认关
        sc.cycles.use_denoising = denoise
        if denoise:
            try:
                sc.cycles.denoiser = "OPTIX"  # 默认 OIDN 走 CPU,OPTIX 走 GPU
            except Exception:
                pass
        sc.render.use_persistent_data = True  # 动画逐帧复用 BVH/网格,只动相机时省同步
    sc.render.resolution_x = res_w
    sc.render.resolution_y = res_h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.compression = 0  # 写盘实测占比小,0 再省 ~7%
    sc.render.fps = fps
    sc.frame_start, sc.frame_end = 1, frames
    # 白模观感:浅灰天光 + 轻雾
    world = bpy.data.worlds["World"] if "World" in bpy.data.worlds else bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.82, 0.84, 0.87, 1.0)
    bg.inputs[1].default_value = 1.0
    try:
        sc.view_settings.look = "AgX - Base Contrast"
    except TypeError:
        pass


def clay_material(name, rgb, rough=0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def toon_material(name, rgb, size=1.2, smooth=0.1):
    """Toon 材质:按面法线离散分带(盒子每面一档明暗),公转时朝向可读。"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    out = nt.nodes["Material Output"]
    toon = nt.nodes.new("ShaderNodeBsdfToon")
    toon.inputs["Color"].default_value = (*rgb, 1.0)
    toon.inputs["Size"].default_value = size
    toon.inputs["Smooth"].default_value = smooth
    nt.links.new(toon.outputs[0], out.inputs["Surface"])
    return mat


def flat_material(name, rgb=(0.02, 0.02, 0.02)):
    """自发光恒色材质:不受光照/AgX 影响,EEVEE 下恒为该色(网格黑线用)。"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    out = nt.nodes["Material Output"]
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def add_edge_tubes(name, center_b, dims_b, quat_b, mat, radius):
    """盒子 12 条棱 → 细六棱柱 + 8 角球,合并为单个 mesh 对象(实时黑描边)。

    替代 Freestyle(~1.2s/帧固定开销):纯几何,EEVEE 零额外成本。
    radius=棱线外露半径(米);中心与 quat 与盒子一致,可 parent 随动。
    """
    import bmesh
    from mathutils import Matrix, Vector
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    hx, hy, hz = (d / 2 for d in dims_b)
    half = (hx, hy, hz)
    for axis in range(3):
        # 圆柱默认沿 Z;把 Z 轴转到目标棱方向:X 棱→绕 Y 转 90°,Y 棱→绕 X 转 90°
        if axis == 0:
            o = Matrix.Rotation(math.pi / 2, 4, "Y")
        elif axis == 1:
            o = Matrix.Rotation(math.pi / 2, 4, "X")
        else:
            o = Matrix.Identity(4)
        depth = 2 * half[axis] + 2 * radius          # 略超出到角球内,防缝隙
        for s1 in (-half[(axis + 1) % 3], half[(axis + 1) % 3]):
            for s2 in (-half[(axis + 2) % 3], half[(axis + 2) % 3]):
                off = [0.0, 0.0, 0.0]
                off[(axis + 1) % 3] = s1
                off[(axis + 2) % 3] = s2
                bmesh.ops.create_cone(bm, cap_ends=True, segments=6,
                                      radius1=radius, radius2=radius, depth=depth,
                                      matrix=Matrix.Translation(Vector(off)) @ o)
    for sx in (-hx, hx):
        for sy in (-hy, hy):
            for sz in (-hz, hz):
                bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=radius,
                                          matrix=Matrix.Translation(Vector((sx, sy, sz))))
    bm.to_mesh(mesh)
    bm.free()
    if quat_b is not None:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = quat_b
    obj.location = center_b
    obj.data.materials.append(mat)
    return obj


def enable_edges(width_px=2.0):
    """Freestyle 黑描边:恒定屏幕像素宽,远近方块都能看清(与 edge-tubes 实时
    描边二选一;Freestyle 有 ~1.2s/帧固定开销)。要求场景已建 Freestyle 线集;
    名为 Ground 的地面 slab 不描边。"""
    sc = bpy.context.scene
    sc.render.use_freestyle = True
    vl = bpy.context.view_layer
    vl.use_freestyle = True
    ls = vl.freestyle_settings.linesets.active
    ls.linestyle.color = (0.0, 0.0, 0.0)
    ls.linestyle.thickness = width_px
    ls.select_silhouette = True    # 轮廓
    ls.select_crease = True        # 折角(方块棱边)
    ls.select_border = True        # 边界
    for attr in ("select_ridge", "select_valley",
                 "select_suggestive_contour", "select_material_boundary"):
        if hasattr(ls, attr):      # 老版本才有 ridge/valley
            setattr(ls, attr, False)
    ground = bpy.data.objects.get("Ground")
    if ground is not None and hasattr(ground, "visible_freestyle"):
        ground.visible_freestyle = False  # 地面 slab 不描边


def border_material(name, rgb=(0.02, 0.02, 0.02), engine="eevee"):
    """反转壳描边材质:正面透明、背面黑。配放大 δ 的外壳,任意视角只在
    内层物体的轮廓外露出 δ 黑边(Freestyle 有 ~1.2s/帧固定开销,故用此替代)。"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    if engine == "eevee":
        mat.use_backface_culling = True  # 近面剔除,只剩背面=经典反转壳
        return mat
    out = nt.nodes["Material Output"]
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    trans = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(trans.outputs[0], mix.inputs[1])   # Fac=0(正面)→透明
    nt.links.new(bsdf.outputs[0], mix.inputs[2])    # Fac=1(背面)→黑
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def add_sun():
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.5
    sun.data.angle = math.radians(15)
    sun.rotation_euler = (math.radians(55), 0, math.radians(35))
    bpy.context.scene.collection.objects.link(sun)


def add_box(name, center_b, dims_b, quat_b, mat, flip=False):
    """Blender 系长方体:center_b 中心,dims_b 全长 (dx,dy,dz),quat_b 四元数(w,x,y,z)。
    flip=True 翻转法线(反转壳描边用:EEVEE backface culling 才能剔近面留远面)。"""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    if flip:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    obj.scale = [d / 2 for d in dims_b]
    if quat_b is not None:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = quat_b
    obj.location = center_b
    obj.data.materials.append(mat)
    return obj


def add_ground_flat(hcam, mat):
    """无限平面地面(顶面在 y=-hcam;配合 add_box 的中心/全长语义)。"""
    add_box("Ground", (0, -hcam - 0.05, 30), (600, 0.1, 400), None, mat)


# ---------------------------------------------------------------- 相机
def make_cam(name, focal_px, orig_wh):
    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "AUTO"
    # AUTO fit:sensor_width 对应渲染长边;等比缩放下等效于
    # lens = focal_px(原图长边像素焦距) × 36mm / 原图长边
    cam_data.lens = focal_px * 36.0 / max(*orig_wh)
    cam_data.clip_start, cam_data.clip_end = 0.05, 2000.0
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    return cam


def render_cam(cam, frames_dir, tag):
    """渲染 frame_start..frame_end → <frames_dir>/<tag>.mp4,返回 mp4 路径。

    FFMPEG 直出:render(animation=True) 单次调用出全段 H264(CRF HIGH),无
    中间文件。实测逐帧 render(write_still)+PNG 编码写盘 ~230ms/帧,直出全程
    ~120ms/帧——大头是逐帧调用固定开销与 PNG 落盘,不是像素吞吐;且帧目录在
    两代渲染进程并发时会被 ffmpeg 按断档静默截断(直出无帧目录,坑消失)。
    (旧逐帧 PNG + 外部 ffmpeg 路线已随公开版整理移除)
    """
    sc = bpy.context.scene
    sc.camera = cam
    mp4 = Path(frames_dir) / f"{tag}.mp4"
    sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"
    sc.render.ffmpeg.codec = "H264"
    sc.render.ffmpeg.constant_rate_factor = "HIGH"
    sc.render.ffmpeg.audio_codec = "NONE"
    sc.render.filepath = str(mp4)  # 必须带 .mp4 后缀:不带后缀 Blender 会补成 <tag>0001-0120.mp4
    bpy.ops.render.render(animation=True)
    return mp4


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)
