"""WBench 的 VisualDirector:把当前 interaction 翻译成 VisualScript。

信息来源只有 case 合法字段 + 官方 prompt_builder 词典(直接 import,
保证与 Wan/Kling/Seedance 的动作措辞零漂移)。框架在此之上增加:
  - 稳定性约束(stability):身份/场景不变、动作唯一、事件隔离
  - 动作轮锁镜(subject_action / event_edit)
  - 首轮场景锚定(first_frame_anchor = environment_prompt)

ref 模式(2026-09-24 融合定稿):control/first_frame 与 off 共用同一套标准
映射与同一 prompt 族(NEWSKILL),script 字段完全一致;唯一差异是
ref_mode="control" 且当前回合为 navigation 时,在标准 script 上声明
MotionReference(控制视频)——prompt 由 PromptCompiler 叠加 reference
overlay,渲染走 ref2va(逐轮按当前首帧重新检测/生成白模)。first_frame
消融不声明 → prompt 与基线逐字节一致,只是渲染后端为 ref2va 纯首帧 ref。

control_official(2026-09-24 追加,实验 D):裸官方 prompt 族 + 控制视频。
声明 MotionReference 与 control 完全相同(逐轮重新检测/白模),但把官方
build_turn_prompt 等价句写进 script.meta("prompt_family"/"official_prompt"),
由 OfficialRef2VACompiler 排版——隔离"框架 prompt 工程"这一个变量
(对照 control 的 NEWSKILL 族;非 navigation 回合退化为纯首帧锚定句,
与裸 ref2va 消融 h3-ref2va 的 prompt 逐字一致)。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from engine.config import WORKSPACE_ROOT
from engine.visual.script import MotionReference, VisualScript

# 官方动作词典:从 WBench 仓库按文件路径直接加载 prompt_builder.py
# (零漂移;不走 src.models 包链,避免牵起 cv2 等重依赖)。
# 懒加载:import 本模块不要求 WBench 仓存在(它是外部第三方依赖,
# 见包 __init__ 文档),首次真正取词典时才加载并给出可操作的报错。
_WBENCH_ROOT = WORKSPACE_ROOT / "WBench"
_OFFICIAL_CACHE: dict[str, dict] | None = None


def _official_actions() -> tuple[dict, dict]:
    """返回 (FIRST_PERSON_ACTIONS, THIRD_PERSON_ACTIONS),首用时从 WBench 仓加载。"""
    global _OFFICIAL_CACHE
    if _OFFICIAL_CACHE is None:
        _pb = _WBENCH_ROOT / "src" / "models" / "text" / "prompt_builder.py"
        if not _pb.is_file():
            raise RuntimeError(
                f"WBench 官方仓未找到:期望 {_pb}。WBench 是第三方基准"
                "(Fudan & Meituan LongCat,arXiv:2605.25874),不在本仓发布;"
                "请从官方来源获取并放置到 $WORKSPACE_ROOT/WBench,或用 "
                "--wbench-root 指定(见本包 __init__ 文档)。")
        _spec = importlib.util.spec_from_file_location("wb_prompt_builder", _pb)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _OFFICIAL_CACHE = {"fp": _mod.FIRST_PERSON_ACTIONS,
                           "tp": _mod.THIRD_PERSON_ACTIONS}
    return _OFFICIAL_CACHE["fp"], _OFFICIAL_CACHE["tp"]


def first_person_actions() -> dict:
    """官方第一人称动作词典(懒加载)。"""
    return _official_actions()[0]


def official_dictionary_available() -> bool:
    """WBench 官方动作词典是否就位(外部仓存在且可加载)。"""
    try:
        _official_actions()
        return True
    except Exception:
        return False


def third_person_actions() -> dict:
    """官方第三人称动作词典(懒加载)。"""
    return _official_actions()[1]

# box2clay 白模控制视频的跟随句对(17 条成片验证措辞;逐字加载
# wbench_prompt.py 保证零漂移):(previz 镜头句, 主体跟随句)。
_b2c_prompt = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "wb_box2clay_prompt",
    WORKSPACE_ROOT / "spatial_proxy" / "box2clay" / "wbench_prompt.py"))
_b2c_prompt.__loader__.exec_module(_b2c_prompt)
TP_MOTION = _b2c_prompt.MOTION

#视角切换的官方句式(与 prompt_builder.build_turn_prompt 一致)
_SWITCH_SENTENCE = {
    "fp_to_tp": ("The view", "as seen through the camera",
                 "transitions from first-person to third-person, "
                 "revealing the subject from behind"),
    "tp_to_fp": ("The view", "as seen through the camera",
                 "transitions from third-person to first-person, "
                 "the camera enters the subject's eyes"),
}


def _split_entry(entry: str) -> tuple[str, str]:
    """官方词典句 → (主体动作短语, 镜头句)。

    tp 条目形如 "The subject moves forward. The camera follows, ..." 或
    "The camera orbits clockwise around the subject. The subject stays still."
    """
    action_part, camera_part = "", ""
    for sent in (s.strip() for s in entry.split(".")):
        if not sent:
            continue
        if sent.startswith("The subject "):
            action_part = sent[len("The subject "):]
        elif sent.startswith("The camera "):
            camera_part = sent
    if not action_part:          # up/down 等纯镜头条目
        action_part = "stays in place"
    return action_part, camera_part


# 预解析缓存:action_token → (主体动作短语, 镜头句);首用时构建。
_TP: dict | None = None
_FP: dict | None = None


def _preparsed(perspective: str) -> dict:
    """返回该视角的 action_token → 短语 预解析表(懒构建)。"""
    global _TP, _FP
    if _TP is None or _FP is None:
        _TP = {k: _split_entry(v) for k, v in third_person_actions().items()}
        # fp 条目:"The camera pushes forward." → 动作短语 "pushes forward"
        _FP = {k: v.rstrip(".").removeprefix("The camera ").strip()
               for k, v in first_person_actions().items()}
    return _TP if perspective == "third_person" else _FP


def _official_turn_prompt(perspective: str, itype: str, action_txt: str) -> str:
    """官方 build_turn_prompt(is_first_turn=False 路径)的等价实现。

    与 src/models/text/prompt_builder.py 逐字一致:任何轮次不带场景文本
    (场景由首帧图像承载),navigation 查官方词典(词表外回退
    "Camera moves: X"),event/subject 透传 action 原文,视角切换用官方
    固定句。供实验 D 的裸官方 prompt 族使用。
    """
    if itype == "navigation":
        action_map = (third_person_actions() if perspective == "third_person"
                      else first_person_actions())
        return action_map.get(action_txt, f"Camera moves: {action_txt}")
    if itype in ("event_edit", "subject_action"):
        return action_txt
    if itype == "perspective_switch":
        if action_txt == "fp_to_tp":
            return ("The viewpoint transitions from first-person to "
                    "third-person, revealing the subject from behind.")
        if action_txt == "tp_to_fp":
            return ("The viewpoint transitions from third-person to "
                    "first-person, the camera enters the subject's eyes.")
        return action_txt
    return action_txt


def _first_sentence(text: str) -> str:
    """取 character_prompt 的第一句作外观短语(去尾点,防组装出断句)。"""
    head = text.strip().split(". ")[0].strip().rstrip(".")
    return head


def _no_brackets(text: str) -> str:
    return text.replace("[", "(").replace("]", ")")


class WBenchVisualDirector:
    """callable(result, visual_state) -> VisualScript

    ref_mode:
      "off"          —— 实验基线(标准映射,零改动)
      "control"      —— 标准 mapping + navigation 回合声明控制视频(ref2va)
      "first_frame"  —— 消融:与 control 同构,但不声明控制视频(prompt 与
                        基线逐字节一致,后端 ref2va 纯首图 ref)
      "control_official" —— 实验 D:声明同 control,但 prompt 族为裸官方句
                        (meta 供 OfficialRef2VACompiler 排版)
    """

    def __init__(self, ref_mode: str = "off"):
        if ref_mode not in ("off", "control", "first_frame", "control_official"):
            raise ValueError(f"unknown ref_mode: {ref_mode!r}")
        self.ref_mode = ref_mode

    def __call__(self, result, visual_state) -> VisualScript:
        state = result.state_before
        itype = result.action.description
        action_txt = result.action.label

        script = self._standard_script(state, itype, action_txt, visual_state)
        script.meta["ref_mode"] = self.ref_mode
        if self.ref_mode == "control_official":
            # 实验 D:官方 build_turn_prompt 等价句(不传 is_first_turn 的
            # 官方 text 协议——任何轮次不带场景文本,与裸 h3-ref2va 同源)
            script.meta["prompt_family"] = "official"
            script.meta["official_prompt"] = _official_turn_prompt(
                state.perspective_now, itype, action_txt)
        if self.ref_mode == "off":
            return script

        # control/ control_official:navigation 回合声明控制视频(词表外
        # 动作不声明→自动降级)。除 reference 声明外,各模式 script 字段
        # 完全一致;生产失败降级回到同一 prompt(official 族降级即纯首帧
        # 锚定句,与 h3-ref2va 消融逐字一致)。
        if self.ref_mode in ("control", "control_official") and itype == "navigation":
            if state.perspective_now == "third_person":
                previz_motion, follow = TP_MOTION.get(action_txt, ("", ""))
                if previz_motion and follow:
                    script.reference = MotionReference(
                        perspective="third_person",
                        action_token=action_txt,
                        previz_motion=previz_motion,
                        subject_motion=follow,
                        official_action=third_person_actions().get(action_txt, ""),
                    )
            else:
                sentence = first_person_actions().get(action_txt, "")
                if sentence:
                    script.reference = MotionReference(
                        perspective="first_person",
                        action_token=action_txt,
                        previz_motion=sentence,
                    )
        return script

    # -- 标准映射(off 基线;control/first_frame 在其上叠加) ----------------------

    def _standard_script(self, state, itype, action_txt, visual_state) -> VisualScript:
        is_first = visual_state.turn_index == 0
        appearance = _first_sentence(state.character_prompt) or "moving naturally"

        if itype == "navigation":
            if state.perspective_now == "third_person":
                action, camera = _preparsed("third_person").get(action_txt, ("moves forward", ""))
                actor = "The subject"
            else:
                action, camera = _preparsed("first_person").get(action_txt, "pushes forward"), ""
                actor = "The camera"
                appearance = "mounted at eye height"
            stability = ("the subject keeps its appearance unchanged",
                         "the environment layout stays consistent during the move")
            end = "the camera completes the intended move"

        elif itype == "subject_action":
            actor, action = "The subject", _no_brackets(action_txt)
            camera = "static locked-off camera, steady framing"
            stability = ("the subject performs only the described action",
                         "the environment stays unchanged during the action")
            end = "the subject completes the described action"

        elif itype == "event_edit":
            actor, action = "The scene", _no_brackets(action_txt)
            appearance = "as established in the opening frame"
            camera = "static camera, locked-off framing"
            stability = ("only the described event changes",
                         "everything else in the scene stays identical")
            end = "the described event has played out in the scene"

        elif itype == "perspective_switch":
            # 已知 token(fp_to_tp / tp_to_fp)用官方固定句;未知 token
            # 透传 action 原文(官方 build_turn_prompt else 分支同款,
            # 保住 case 自带的跟拍目标等指令)
            actor, appearance = "The view", "as seen through the camera"
            action = _SWITCH_SENTENCE.get(action_txt, ("", "", action_txt))[2]
            camera = ""
            stability = ("the subject and environment remain identical "
                         "across the viewpoint change",)
            end = "the viewpoint settles in the new perspective"

        else:  # 未知类型兜底:忠实透传 case 原文
            actor, appearance = "The scene", "as established in the opening frame"
            action, camera = _no_brackets(action_txt), ""
            stability = ("the scene stays consistent",)
            end = "the described change has been applied"

        return VisualScript(
            first_frame_anchor=_no_brackets(state.environment_prompt) if is_first else "",
            camera=camera,
            actor=actor,
            actor_appearance=appearance,
            action=_no_brackets(action),
            stability=stability,
            end_state=end,
            use_first_frame=True,
            duration_s=5,
            meta={"itype": itype, "action": action_txt,
                  "turn": visual_state.turn_index},
        )
