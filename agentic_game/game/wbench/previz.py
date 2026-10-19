"""WBench 的控制视频生产方(Box2Clay 白模 previz 管线的引擎侧封装)。

实现 engine 的 ReferenceProducer 协议:`__call__(script, visual_state) -> Path`。
engine 不知道 spatial_proxy/检测的存在;本模块负责把 MotionReference 翻译成
box2clay 五阶段流水线的前两段(spatial_proxy/README.md Pipeline):

    ① 检测(仅 third_person;WildDet3D,wilddet3d env)
    ② 控制视频(bpy311 env:clay_subject.py tp / fp_previz.py fp)

末帧链接约定(2026-09-24 定稿):所有回合链式推进首帧,因此控制视频
**逐轮按当前帧生产**——
- tp:对当前锚定帧(上一轮末帧,首轮为 case 初始图)重新检测,蓝盒锚定
  该帧主体位置(首帧对齐硬约束),再渲染该动作的白模控制视频;
- fp:previz 与画面无关(统一白模、固定视高,相机即主体),同一动作的
  控制视频可跨轮复用。

存储策略(批量不落盘):产物一律写入 per-case 临时目录 work_root
(<seg_dir>/previz),由驱动在 case 成功后删除;--debug-previz 保留供排查。
canonical 只读缓存仍会命中复用:
- 初始图锚定的 tp 检测/控制视频(results/subject、subject_v2,含归档);
- fp 控制视频(fp_v1/batch_*,含归档)。

关键约定(spatial_proxy/README.md Key conventions,勿破坏):
- key 约定 <det_dir>_walk_<动作>('+'→'_'),文件/manifest/prompt 三处一致;
- tp 动作 16 种对齐 WBench prompt_builder;fp 生成器支持 all16;
- 产物 120 帧 864 宽;新生成的控制视频默认过 check_jumps(--skip-checks 关闭)。

GPU:spatial_proxy/wilddet3d 需要一张卡;默认 CUDA_VISIBLE_DEVICES=0(EEVEE/
WildDet3D 显存占用小,可与常驻 H3 共存);可经 previz_gpu 覆盖。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

from engine.config import WORKSPACE_ROOT, conda_bin
from engine.renderer.base import RenderError, RenderFailedError
from engine.visual.script import VisualScript

BOX2CLAY_ROOT = WORKSPACE_ROOT / "spatial_proxy" / "box2clay"
BPY_PY = Path(conda_bin("bpy311", "python"))
WILDDET_PY = Path(conda_bin("wilddet3d", "python"))
# check_jumps 的文档化解释器(numpy;最终流程记录 §2.③)
CHECK_PY = Path(conda_bin("h3-a800", "python"))

# canonical 只读缓存根(现行 + 归档;命中即用,不改动)
_TP_VIDEO_ROOTS = (
    BOX2CLAY_ROOT / "results" / "subject_v2",
    BOX2CLAY_ROOT / "_archive_20260923" / "results" / "subject_v2",
    BOX2CLAY_ROOT / "results" / "batch_v6",
    BOX2CLAY_ROOT / "results" / "batch_v7",
)
_DET_ROOTS = (
    BOX2CLAY_ROOT / "results" / "subject",
    BOX2CLAY_ROOT / "_archive_20260923" / "results" / "subject",
)
_FP_VIDEO_ROOTS = (
    BOX2CLAY_ROOT / "results" / "fp_v1",
    BOX2CLAY_ROOT / "_archive_20260923" / "results" / "fp_v1",
    BOX2CLAY_ROOT / "results" / "batch_v6",
    BOX2CLAY_ROOT / "results" / "batch_v7",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 主体类别/高度推断与 box2clay 清单工具零漂移(直接加载同一文件)
_wbench_manifest = _load_module("wb_box2clay_manifest", BOX2CLAY_ROOT / "wbench_manifest.py")


class Box2ClayPreviz:
    """per-case 构造一次;produce() 幂等(同锚定帧+动作重复调用零成本)。"""

    def __init__(
        self,
        case: dict,
        *,
        work_root: Path,
        initial_image: Path | None = None,
        previz_gpu: str = "0",
        skip_checks: bool = False,
    ) -> None:
        self.case = case
        self.case_id = str(case.get("id", ""))
        settings = case.get("settings", {})
        self.perspective0 = settings.get("perspective", "first_person")
        subject = settings.get("subject") or {}
        self.subject_type = subject.get("type") or "other"
        # 类别/高度/检测词与 box2clay 清单同一推断逻辑(零漂移):
        # coco = 类型默认;world_term = desc 关键词具体类(兜底 coco);
        # det 目录命名与 clay_subject --subject 都用 world_term 的首个词
        # (如 animal+horse → case_18_horse,--subject horse)。
        coco, _ = _wbench_manifest.TYPE_DEFAULT.get(self.subject_type, ("person", 1.7))
        self.world_term = _wbench_manifest.guess_world_term(subject.get("desc"), coco)
        self.subject_class = self.world_term.split(",")[0]
        self.subject_height = _wbench_manifest.guess_height(
            self.subject_type, subject.get("desc"))
        self.work_root = Path(work_root)      # per-case 临时目录(驱动负责清理)
        self.initial_image = initial_image    # case 初始图(canonical 缓存判定)
        self.previz_gpu = previz_gpu
        self.skip_checks = skip_checks

    # -- ReferenceProducer 协议 ------------------------------------------------

    def __call__(self, script: VisualScript, visual_state) -> Path:
        ref = script.reference
        if ref is None:  # pragma: no cover - loop 只在 reference 非 None 时调用
            raise RenderError("Box2ClayPreviz called without MotionReference")
        return self.produce(ref.perspective, ref.action_token,
                            anchor=visual_state.current_frame)

    # -- 生产入口 ---------------------------------------------------------------

    def produce(self, perspective: str, action_token: str,
                anchor: Path | None = None) -> Path:
        """确保 (锚定帧, perspective, action) 的控制视频存在,返回其路径。

        anchor = 本回合首帧(末帧链接:首轮为 case 初始图,之后为上一轮
        末帧);tp 必须提供,fp 不需要(与画面无关)。
        """
        stem = action_token.replace("+", "_")
        if perspective == "first_person":
            return self._ensure_fp(action_token, stem)
        return self._ensure_tp(action_token, stem, anchor)

    # -- first_person:统一白模,与锚定帧无关,可跨轮复用 ------------------------

    def _ensure_fp(self, action_token: str, stem: str) -> Path:
        hit = self._find_cached(_FP_VIDEO_ROOTS, self._fp_dir_names(), stem)
        if hit is not None:
            return hit
        out_dir = self.work_root / "fp" / f"case_{self.case_id}"
        video = out_dir / f"walk_{stem}.mp4"
        if not video.exists():  # 同 case 重复动作天然复用临时产物
            self._run(
                [str(BPY_PY), str(BOX2CLAY_ROOT / "fp_previz.py"), str(out_dir),
                 "--action", action_token, "--move", "3.2",
                 "--no-preview", "--width", "864"],
                what=f"fp previz {action_token}",
            )
            self._check_jumps(video)
        return video

    # -- third_person:按当前锚定帧检测 → 蓝盒控制视频 ----------------------------

    def _ensure_tp(self, action_token: str, stem: str, anchor: Path | None) -> Path:
        if anchor is None:
            raise RenderFailedError(
                "third_person control video requires an anchor frame "
                "(visual_state.current_frame)")
        anchor = Path(anchor).resolve()
        at_initial = (
            self.initial_image is not None
            and anchor == Path(self.initial_image).resolve()
        )
        det_dir = self._ensure_det(anchor, at_initial)
        if at_initial:
            # 初始图锚定 = canonical 命名,可复用 box2clay 既有控制视频
            hit = self._find_cached(_TP_VIDEO_ROOTS, [det_dir.name], stem)
            if hit is not None:
                return hit
        out_dir = self.work_root / "subject" / det_dir.name
        video = out_dir / f"walk_{stem}.mp4"
        if not video.exists():  # 重试轮复用(同锚定帧+动作)
            self._run(
                [str(BPY_PY), str(BOX2CLAY_ROOT / "clay_subject.py"),
                 str(det_dir), str(out_dir),
                 "--subject", self.subject_class,
                 "--subject-height", str(self.subject_height),
                 "--samples", "16", "--bare", "--edge-tubes",
                 "--action", action_token, "--no-preview",
                 "--width", "864", "--push", "0.70"],
                what=f"clay previz {action_token}@{anchor.stem}",
            )
            self._check_jumps(video)
        return video

    def _ensure_det(self, anchor: Path, at_initial: bool) -> Path:
        """det 目录:初始图先试 canonical(box2clay 批量脚本的历史命名
        case_<id>_<class>);未命中或生成帧 → 现检到 work_root/det,
        目录名 = detect_boxes 原生命名 <图片stem>(同帧重试天然复用)。"""
        if at_initial:
            for root in _DET_ROOTS:
                cand = root / f"case_{self.case_id}_{self.subject_class}"
                if (cand / "det.json").is_file():
                    return cand
        det_root = self.work_root / "det"
        cand = det_root / anchor.stem
        if (cand / "det.json").is_file():
            self._normalize_det_class(cand)
            return cand
        # 检测 texts:开放词表(具体类别词优先,兜底 COCO 类)
        texts = self.world_term or self.subject_class
        self._run(
            [str(WILDDET_PY), str(BOX2CLAY_ROOT / "detect_boxes.py"),
             str(anchor), "--out", str(det_root), "--texts", texts],
            what=f"detect {anchor.stem}",
            offline_hf=True,
        )
        if not (cand / "det.json").is_file():
            raise RenderFailedError(
                f"detect_boxes produced no det.json for {anchor.name}")
        self._normalize_det_class(cand)
        return cand

    def _normalize_det_class(self, det_dir: Path) -> None:
        """开放词表误标兜底(就地改写 work_root 下的 det.json)。

        WildDet3D 在 WBench 图上常把主体标成同 texts 词表内的另一个词
        (woman→person、cat→dog、samurai→person;box2clay nav8 已知现象),
        而 clay_subject 按 class 精确取实例。没有 class==subject 时,把
        词表内其他类里得分最高的实例改标成 subject——同一主体,只是
        检测器的措辞不同。仅作用于临时 work_root,canonical 只读缓存
        (其目录名即 det.json class,历史上已一致)不受影响。
        """
        det_json = det_dir / "det.json"
        try:
            data = json.loads(det_json.read_text(encoding="utf-8"))
            objects = data["objects"]
        except Exception:
            return
        if not isinstance(objects, list) or not objects:
            return
        if any(o.get("class") == self.subject_class for o in objects):
            return
        vocab = [w.strip() for w in (self.world_term or "").split(",")
                 if w.strip()]
        fallback = [o for o in objects
                    if o.get("class") in set(vocab) | {self.subject_class}
                    and o.get("class") != self.subject_class]
        if not fallback:
            return
        best = max(fallback, key=lambda o: float(o.get("score") or 0.0))
        best["class"] = self.subject_class
        det_json.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    def _fp_dir_names(self) -> list[str]:
        return [f"case_{self.case_id}", f"fp_case_{self.case_id}"]

    @staticmethod
    def _find_cached(roots: tuple[Path, ...], dir_names: list[str], stem: str) -> Path | None:
        for root in roots:
            for name in dir_names:
                cand = root / name / f"walk_{stem}.mp4"
                if cand.is_file() and cand.stat().st_size > 0:
                    return cand
        return None

    # -- 子进程与校验 ------------------------------------------------------------

    def _run(self, cmd: list[str], *, what: str, offline_hf: bool = False) -> None:
        import os

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.previz_gpu
        if offline_hf:
            env["HF_HUB_OFFLINE"] = "1"
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=str(BOX2CLAY_ROOT), timeout=1800)  # noqa: S603 固定脚本
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            raise RenderFailedError(f"{what} failed (rc={proc.returncode}): {tail}")

    def _check_jumps(self, video: Path) -> None:
        """新生成的控制视频过跳变校验(最终流程记录 §3.7;可 skip)。

        输出格式 `jumps=[帧号]` 或 `jumps=无`;up 类后段亚像素位移是
        已知误报(目测排除),放行但只告警。
        """
        if self.skip_checks:
            return
        proc = subprocess.run(
            [str(CHECK_PY), str(BOX2CLAY_ROOT / "check_jumps.py"), str(video)],
            capture_output=True, text=True, timeout=300)  # noqa: S603
        out = proc.stdout.strip()
        if proc.returncode == 0 and "jumps=无" in out:
            return
        if proc.returncode == 0 and "up" in video.stem:
            print(f"[previz] check_jumps flagged {video.name} (up 类已知误报,放行): "
                  f"{out[-200:]}", flush=True)
            return
        raise RenderFailedError(
            f"check_jumps rejected {video.name}: {(out or proc.stderr)[-400:]}")
