#!/usr/bin/env python3
"""WBench 实验 C 驱动:逐 case 跑 GameLoop,产出 case_<id>_combined.mp4。

用法(agentic_game 根目录,系统 python3):
    python3 game/wbench/run_wbench.py \
        --wbench-root "$WORKSPACE_ROOT/WBench" \
        --out WBench/work_dirs/h3-agentic \
        [--limit N] [--only case_1,case_7] [--passes 2]
        [--ref {off,control,first_frame}]

ref 模式(控制视频接入,2026-09-24 融合定稿;三模式统一末帧链接):
  - off(默认):实验基线。常驻 i2v 引擎 + 标准 NEWSKILL prompt。
  - control:navigation 回合在标准 prompt 上叠加 reference overlay
    (<Picture 1>/<Video 1>/跟随句/反残留),逐轮按当前首帧重新检测+
    生成白模控制视频(ref2va 引擎:refs = [首帧, 控制视频]);非
    navigation 回合纯首帧 ref,prompt 与基线一致。
  - first_frame:消融。与 control 同一后端/同一末帧链接/同一标准
    prompt,只是永不带控制视频——只隔离 <Video 1> 一个变量。
  - control_official(实验 D,2026-09-24):白模控制视频 + 裸官方
    prompt 族(锚定句+<Video 1> 场景句+官方动作句,见
    game/wbench/official_prompt.py)。生产/后端/链接与 control 全同,
    只隔离"框架 prompt 工程"一个变量(对照 control 的 NEWSKILL 族)。
  控制视频为临时产物(segments/<case>/previz/),case 成功后删除;
    --debug-previz 保留供排查。ref2va 与 t2v 服务互斥占卡,
    驱动会在启动前自动停掉占卡的另一个。

约定:
  - 每轮 seed = base_seed + index(框架原生策略,42,43,...;与 A/B 的固定
    42 不同,属可接受的解码参数差异,已记录)
  - render_failure_policy=ABORT:某轮渲染失败则整 case 重试(轮缺失会
    导致 per-turn 评测错位,宁可重跑整 case)
  - 断点续跑:已存在的 case_<id>_combined.mp4 直接跳过
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

AG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AG_ROOT))

from engine.loop import GameLoop                       # noqa: E402
from engine.config import WORKSPACE_ROOT               # noqa: E402
from engine.renderer.remote import H3RemoteRenderer    # noqa: E402
from engine.visual.compiler import PromptCompiler     # noqa: E402
from engine.visual.state import VisualState           # noqa: E402

from game.wbench.case_runtime import (                 # noqa: E402
    ScriptedAgent, WBenchRuntime)
from game.wbench.director import WBenchVisualDirector              # noqa: E402
from game.wbench.official_prompt import OfficialRef2VACompiler      # noqa: E402
from game.wbench.previz import BPY_PY, BOX2CLAY_ROOT, Box2ClayPreviz  # noqa: E402

from engine.renderer.service import default_socket_path  # noqa: E402

SOCKET_T2V = default_socket_path(8, "480p")
# ref2va 常驻服务统一监听 box2clay 的 socket 路径:engine worker 讲双协议
# (engine 键名 + box2clay 键名),一份引擎同时服务 agentic runner 与
# box2clay CLI(ref2va_serve.py generate / h3serve_batch.py),互不切换。
SOCKET_REF2VA = BOX2CLAY_ROOT / "results" / "h3_ref2va.sock"
# 探活 ping 的超时:必须 > 渲染队列最长等待(多分片满载 ~2 分钟),
# 否则排队的 ping 超时会被误判成服务死亡(见 ensure_service pong 注释)
_PING_TIMEOUT_S = 180.0
# 2026-09-24 之前的 engine ref2va 默认 socket(已被 box2clay 路径取代);
# 仍在跑的旧实例要停掉,避免双实例占卡
LEGACY_SOCKET_REF2VA = Path("/tmp/agentic-game-h3-8gpu-480p-ref2va.sock")
AG_GAME_DIR = AG_ROOT
# engine worker pong 自带的服务标记(区分 box2clay 原生 ref2va_serve)
_ENGINE_TAG = "agentic-engine-h3"


def _socket_for(ref_mode: str) -> Path:
    return SOCKET_REF2VA if ref_mode != "off" else SOCKET_T2V


def ensure_service(ref_mode: str = "off", poll_s: int = 60, attempts: int = 40) -> bool:
    """探活;未运行则(锁内)start 并轮询至 socket 可用(装载 ~6 分钟)。

    **只起步,不停人**(2026-09-24 事故链定稿):旧版里的 t2v 互斥 stop、
    legacy socket 清理、box2clay 原生实例接管(ref2va_serve.py stop 会向
    同一 socket 发 {"cmd":"shutdown"},engine 整组应声退出变僵尸)多个
    "先停别人"分支曾多次把健康实例送走。现在本函数绝不停任何实例;
    t2v/ref2va 的互斥由编排层/人工保证。多 runner 并发 start 用文件锁
    串行化 + 拿锁后重新探活,至多一个实例在装载。
    """
    task = "ref2va" if ref_mode != "off" else "t2v"
    socket = _socket_for(ref_mode)

    def pong(sock: Path) -> dict | None:
        # 探活超时必须覆盖渲染排队:4 分片下服务可连续 ~2 分钟满载,
        # ping 在 backlog 里等 10s 就超时的旧值会把"忙"误判成"死",
        # 触发重复 spawn(2026-09-24 07:0x 三实例叠加事故)
        try:
            r = H3RemoteRenderer(output_dir=Path("/tmp"), socket_path=sock)
            return r.ping(timeout_s=_PING_TIMEOUT_S)
        except Exception:
            return None

    def usable(p: dict | None) -> bool:
        return p is not None and (task != "ref2va" or
                                  p.get("service") == _ENGINE_TAG)

    current = pong(socket)
    if usable(current):
        return True
    if current is not None:
        # socket 上有应答但不是 engine 双协议实例(原生 box2clay):
        # 不停它、也不对它讲协议,直接失败交人工处理——绝不动别人的实例
        print(f"[ensure] socket {socket} answered without engine tag "
              f"({current}); refusing to touch it", flush=True)
        return False

    import tempfile

    lock_path = Path(tempfile.gettempdir()) / "agentic-game-service-start.lock"
    with open(lock_path, "w") as lock_fh:  # 分片并发 start 串行化
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        # 拿到锁后重新探活:别的进程可能已把服务拉起
        if usable(pong(socket)):
            return True
        _spawn_ref2va_direct(socket)
        for _ in range(attempts):
            time.sleep(poll_s)
            if usable(pong(socket)):
                return True
            # 装载进程已消失且 socket 不存在:启动已失败,
            # 继续等满 attempts 只是空耗(2026-09-24 实测卡死 18 分钟)
            workers = subprocess.run(
                ["pgrep", "-f", "engine/renderer/h3_worker.py"],
                capture_output=True)
            if workers.returncode != 0:
                return False
    return False


def _spawn_ref2va_direct(socket: Path) -> None:
    """绕过 service.py 直接 torchrun 拉起 engine ref2va 服务。

    service.py start 的 wait_until_ready 在 torchrun 静默死亡时会空转
    满 startup_timeout(2026-09-24 两次实测各挂 20+ 分钟,无任何输出);
    这里自己 Popen(分离会话、日志进 socket 旁路文件)+ 自轮询,装载
    进程一死即可早退。
    """
    from engine.config import load_config  # noqa: F401  (与 service.py 同源)
    from engine.renderer.minimax_h3 import (
        DEFAULT_H3_REPO, DEFAULT_TORCHRUN, h3_environment)
    from engine.renderer.service import REF2VA_ADAPTER

    log_path = Path(str(socket) + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab")
    cmd = [
        str(DEFAULT_TORCHRUN), "--standalone", "--nproc_per_node=8",
        str(AG_GAME_DIR / "engine" / "renderer" / "h3_worker.py"),
        "--h3-repo", str(DEFAULT_H3_REPO),
        "--model", str(DEFAULT_H3_REPO / "checkpoints" / "MiniMax-H3"),
        "--adapter", str(DEFAULT_H3_REPO / REF2VA_ADAPTER),
        "--attention-backend", "dense",
        "--task", "ref2va", "--res", "480p",
        "--session-tag", f"agentic-game-h3-serve-{socket.name}",
        "--serve", "--socket", str(socket),
    ]
    subprocess.Popen(  # noqa: S603 - 固定命令行
        cmd, cwd=str(DEFAULT_H3_REPO), env=h3_environment(),
        stdout=log_fh, stderr=log_fh, start_new_session=True)
    print(f"[ensure] torchrun spawned for {socket} (log {log_path})",
          flush=True)


def ffmpeg_concat(segments: list[Path], out: Path, seg_dir: Path) -> bool:
    """同源段落:concat demuxer + stream copy;失败退回重编码。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = seg_dir / f"{out.stem}_list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in segments), encoding="utf-8")
    base = ["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if subprocess.run(base + ["-c", "copy", str(out)], capture_output=True).returncode == 0 \
            and out.exists() and out.stat().st_size > 0:
        return True
    subprocess.run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-c:a", "copy", str(out)], capture_output=True)
    return out.exists() and out.stat().st_size > 0


def run_case(case: dict, wbench_root: Path, out_root: Path, base_seed: int,
             ref_mode: str = "off", previz_gpu: str = "0",
             skip_previz_checks: bool = False,
             debug_previz: bool = False) -> tuple[bool, str]:
    case_id = str(case["id"])
    seg_dir = out_root / "segments" / f"case_{case_id}"
    combined = out_root / "videos" / f"case_{case_id}_combined.mp4"
    if combined.exists() and combined.stat().st_size > 0:
        return True, "skip"

    img = case.get("settings", {}).get("initial_image", "")
    if img:
        img_p = Path(img)
        if img_p.is_absolute():
            cands = [img_p]
        else:  # 官方约定:相对路径相对 <wbench_root>/data/;兼容根目录
            cands = [wbench_root / "data" / img_p, wbench_root / img_p]
        img_p = next((c for c in cands if c.exists()), None)
        if img_p is None:
            return False, f"missing initial image {img}"
    else:
        img_p = None

    # 控制视频临时目录:per-case,成功后删除(--debug-previz 保留)
    previz_dir = seg_dir / "previz"
    reference_producer = None
    if ref_mode in ("control", "control_official"):
        reference_producer = Box2ClayPreviz(
            case, work_root=previz_dir,
            initial_image=img_p.resolve() if img_p else None,
            previz_gpu=previz_gpu, skip_checks=skip_previz_checks)

    renderer = H3RemoteRenderer(output_dir=seg_dir, socket_path=_socket_for(ref_mode))
    loop = GameLoop(
        runtime=WBenchRuntime(case),
        agent=ScriptedAgent(),
        visual_director=WBenchVisualDirector(ref_mode=ref_mode),
        renderer=renderer,
        output_dir=seg_dir,
        base_seed=base_seed,
        max_turns=len(case.get("interactions", [])) + 2,
        render_retries=2,
        render_failure_policy="abort",
        # 三模式统一末帧链接:首轮首帧 = case 初始图,之后 = 上一轮末帧;
        # tp 控制视频生产方按当前帧逐轮重新检测/生成。
        visual_state=VisualState(current_frame=img_p.resolve() if img_p else None),
        # control_official(实验 D):裸官方 prompt 族编译器;其余 NEWSKILL
        compiler=OfficialRef2VACompiler() if ref_mode == "control_official"
                 else PromptCompiler(),
        reference_producer=reference_producer,
    )
    summary = loop.run()
    segments = [Path(r.video_path) for r in summary.records if r.video_path]
    if len(segments) != len(case.get("interactions", [])):
        return False, f"segments {len(segments)}/{len(case['interactions'])}"
    if not ffmpeg_concat(segments, combined, seg_dir):
        return False, "concat failed"
    if not debug_previz:
        shutil.rmtree(previz_dir, ignore_errors=True)
    return True, "ok"


def _service_quick_ok(ref_mode: str) -> bool:
    """轻量探活:服务掉线时 runner 应等待恢复,而不是把 case 列表逐个
    快速烧穿(每个 case 3 次失败渲染 ~15-20s,70 个就是半小时白费)。"""
    try:
        r = H3RemoteRenderer(output_dir=Path("/tmp"),
                             socket_path=_socket_for(ref_mode))
        return r.ping(timeout_s=60) is not None
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbench-root", default=str(WORKSPACE_ROOT / "WBench"))
    ap.add_argument("--ref", default="off",
                    choices=("off", "control", "first_frame", "control_official"),
                    help="off=基线(i2v+NEWSKILL);control=navigation 回合白模"
                         "控制视频(ref2va);first_frame=消融(纯首图 ref);"
                         "control_official=实验 D(白模+裸官方 prompt 族)")
    ap.add_argument("--out", default="",
                    help="默认按模式分目录:h3-agentic / h3-agentic-ref / "
                         "h3-agentic-first-frame / h3-ref2va-control")
    ap.add_argument("--cases-dir", default="data/cases")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--previz-gpu", default="0",
                    help="白模/检测子进程的 CUDA_VISIBLE_DEVICES(默认 0)")
    ap.add_argument("--skip-previz-checks", action="store_true",
                    help="跳过新生成控制视频的 check_jumps 校验")
    ap.add_argument("--debug-previz", action="store_true",
                    help="保留控制视频临时产物(默认 case 成功后删除)")
    args = ap.parse_args()

    wbench_root = Path(args.wbench_root).resolve()
    default_out = {
        "off": "h3-agentic", "control": "h3-agentic-ref",
        "first_frame": "h3-agentic-first-frame",
        "control_official": "h3-ref2va-control",
    }[args.ref]
    out_root = Path(args.out).resolve() if args.out else \
        (wbench_root / "work_dirs" / default_out).resolve()
    (out_root / "videos").mkdir(parents=True, exist_ok=True)
    print(f"[runner] ref_mode={args.ref} out={out_root}", flush=True)

    case_files = sorted((wbench_root / args.cases_dir).glob("case_*.json"))
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        want = {w if w.startswith("case_") else f"case_{w}" for w in want}
        case_files = [f for f in case_files if f.stem in want]
    if args.limit:
        case_files = case_files[:args.limit]

    ok, fail = 0, 0
    t0 = time.time()
    for pass_i in range(1, args.passes + 1):
        if not ensure_service(args.ref):
            print("[runner] H3 unavailable for retry pass", flush=True)
            return 1
        print(f"[runner] === pass {pass_i} start {time.strftime('%T')} ===", flush=True)
        redo = []
        for i, cf in enumerate(case_files, 1):
            case = json.loads(cf.read_text(encoding="utf-8"))
            cid = case.get("id", cf.stem)
            combined = out_root / "videos" / f"case_{cid}_combined.mp4"
            if not (combined.exists() and combined.stat().st_size > 0) \
                    and not _service_quick_ok(args.ref):
                print(f"[runner] service down before case_{cid}; "
                      "waiting for recovery", flush=True)
                if not ensure_service(args.ref):
                    print("[runner] service did not recover", flush=True)
                    return 1
            t = time.time()
            try:
                good, msg = run_case(case, wbench_root, out_root, args.base_seed,
                                     ref_mode=args.ref, previz_gpu=args.previz_gpu,
                                     skip_previz_checks=args.skip_previz_checks,
                                     debug_previz=args.debug_previz)
            except Exception as exc:  # 单 case 失败不拖垮批次
                good, msg = False, f"{type(exc).__name__}: {exc}"
            if good:
                if msg != "skip":
                    ok += 1
                    print(f"[runner] [{i}/{len(case_files)}] case_{cid} OK "
                          f"({time.time()-t:.0f}s) {msg}", flush=True)
            else:
                fail += 1
                redo.append(cid)
                print(f"[runner] [{i}/{len(case_files)}] case_{cid} FAIL: {msg}", flush=True)
        done = sum(1 for cf in case_files
                   if (out_root / "videos" / f"{cf.stem}_combined.mp4").exists())
        print(f"[runner] pass {pass_i} done: total_done={done}/{len(case_files)}", flush=True)
        if done >= len(case_files) or not redo:
            break
        case_files = [cf for cf in case_files
                      if cf.stem.removeprefix("case_") in set(redo)]
        time.sleep(20)

    total = len(list((out_root / "videos").glob("case_*_combined.mp4")))
    print(f"[runner] FINISHED ok={ok} fail_last_pass={fail} total_on_disk={total} "
          f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
