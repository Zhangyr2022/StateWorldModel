"""WBench × ref2va prompt 模版。

现行默认 **flat 基线句式**(逐动作方向句 + 红盒三处文字,h3_flat_v3/redbox 实测)。
`structured`(六段式,wbench_template.md 为唯一事实源,动作无关)为对照支线;
`fp` 第一人称(fp_previz.py 统一白模控制视频,无蓝盒,相机即主体)。

WBench case 字段取舍(对齐上游 WBench/src/models/text/prompt_builder.py):
  进 prompt = environment_prompt / perspective_prompt / settings.subject.desc /
  action 键位(仅用于选择控制视频);settings.* 元信息、scene_adherence、
  subject_adherence、metric_list、character_prompt(带姿势尾句会与单动作冲突)
  一律不进。

--style flat 为现行默认(17 条成片验证,逐动作措辞);structured 六段式对照;
fp 第一人称。

用法:
    python wbench_prompt.py --case case_105.json --action W        # 打印单条(结构化)
    python wbench_prompt.py --case 105 --action W --style flat     # 旧扁平式
    python wbench_prompt.py --all --out prompts_all.json           # 全部 74 个
                                                                   # third_person 导航 case
"""
import argparse
import json
import os
from pathlib import Path

# WBench 数据目录:WBENCH_DATA 可覆盖;缺省按 workspace 并排布局推导
WBENCH = Path(os.environ.get(
    "WBENCH_DATA", Path(__file__).resolve().parents[2] / "WBench" / "data"))
TEMPLATE_MD = Path(__file__).resolve().parent / "wbench_template.md"
SLOTS = ("desc", "env", "camera")

# previz 侧运动句 / 主体跟随句(16 动作,与 clay_subject.py --action 及
# WBench prompt_builder.THIRD_PERSON_ACTIONS 对齐;措辞 = nav8 定稿;
# flat(现行默认)逐动作使用——结构化文本动作无关,运动由 <Video 1> 承载)
MOTION = {
    "W":   ("the camera tracks alongside the box as it moves forward",
            "moves forward exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "S":   ("the camera tracks alongside the box as it moves backward",
            "moves backward exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "A":   ("the camera tracks alongside the box as it moves to the left",
            "moves to the left exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "D":   ("the camera tracks alongside the box as it moves to the right",
            "moves to the right exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "left": ("the camera orbits counterclockwise around the static box, keeping it centered",
             "stays still in its position"),
    "right": ("the camera orbits clockwise around the static box, keeping it centered",
              "stays still in its position"),
    "up":  ("the camera cranes upward around the static box, tilting down to keep it centered",
            "stays still in its position"),
    "down": ("the camera cranes downward around the static box, tilting up to keep it centered",
             "stays still in its position"),
    "W+A": ("the box moves diagonally forward-left while the camera follows along the same diagonal",
            "moves diagonally forward-left exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "W+D": ("the box moves diagonally forward-right while the camera follows along the same diagonal",
            "moves diagonally forward-right exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "S+A": ("the box moves diagonally backward-left while the camera follows along the same diagonal",
            "moves diagonally backward-left exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "S+D": ("the box moves diagonally backward-right while the camera follows along the same diagonal",
            "moves diagonally backward-right exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "W+left": ("the box moves forward while the camera arcs counterclockwise around it",
               "moves forward exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "W+right": ("the box moves forward while the camera arcs clockwise around it",
                "moves forward exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "W+up": ("the box moves forward while the camera cranes upward around it",
             "moves forward exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
    "W+down": ("the box moves forward while the camera cranes downward around it",
               "moves forward exactly as the blue box does, matching its position, direction, size, timing and trajectory"),
}

FLAT_TEMPLATE = (
    "<Picture 1> is the exact first frame of the target video: begin the video on "
    "<Picture 1>, reproducing it faithfully, and keep its style and lighting for the "
    "whole shot. <Subject 1> is {desc} in <Picture 1>. <Video 1> is the motion and "
    "camera reference: a minimal previz containing only a blue box standing in for "
    "<Subject 1> on an infinite light-gray ground with a 1-meter grid and two static "
    "red reference boxes near the subject — no other objects "
    "— {previz_motion}. <Subject 1> {subject_motion}, while "
    "keeping the exact appearance and screen size from <Picture 1>. Camera: {camera} "
    "Replace the blue box, the red reference boxes and the empty grid floor with the "
    "real scene from <Picture 1>: {env} Everything else stays consistent with "
    "<Picture 1>. Do not preserve the blue box's placeholder appearance, never show "
    "the two red reference boxes (they are previz-only aids), and let no previz "
    "geometry remain: the output is the final polished cinematic shot in the "
    "exact style of <Picture 1>."
)


def load_template() -> str:
    """模版全文 = wbench_template.md「模版全文」节代码块(唯一事实源)。"""
    block = TEMPLATE_MD.read_text() \
        .split("## 模版全文(六段式,段间空行)")[1].split("```")[1].strip("\n")
    missing = [s for s in SLOTS if "{" + s + "}" not in block]
    assert not missing, f"wbench_template.md 模版缺槽位: {missing}"
    return block


def _check(case, action, desc):
    if action not in MOTION:
        raise KeyError(f"action {action!r} 不在 16 动作表内: {sorted(MOTION)}")
    if case.get("settings", {}).get("perspective") != "third_person":
        raise ValueError(f"case_{case.get('id')} 是 "
                         f"{case.get('settings', {}).get('perspective')},previz 管线仅支持 third_person")
    subj = case.get("settings", {}).get("subject") or {}
    return desc or subj.get("desc") or case.get("character_prompt", "")


def build_structured(case: dict, action: str, desc: str = None) -> str:
    """结构化 prompt(wbench_template.md 定稿,动作无关)。camera 剥尾部句号
    (模版在 {camera} 后自带句号,避免 "space.." 双句号)。"""
    desc = _check(case, action, desc)
    return load_template().format(
        desc=desc,
        env=case.get("environment_prompt", ""),
        camera=case.get("perspective_prompt", "").rstrip("."))


def build_flat(case: dict, action: str, desc: str = None) -> str:
    """nav8 实测扁平模版(legacy,17 条成片验证,逐动作措辞)。"""
    desc = _check(case, action, desc)
    previz, subject_m = MOTION[action]
    return FLAT_TEMPLATE.format(
        desc=desc, previz_motion=previz, subject_motion=subject_m,
        camera=case.get("perspective_prompt", ""),
        env=case.get("environment_prompt", ""))


# 第一人称(fp_previz.py 控制视频:无蓝盒统一白模——网格地面 + 前方两个静止
# 红色参考盒,固定视高 1.6m;相机即主体)。运动句 = WBench FIRST_PERSON_ACTIONS 原文。
FIRST_PERSON_MOTION = {
    "W": "The camera pushes forward.",
    "S": "The camera pulls back.",
    "A": "The camera moves to the left.",
    "D": "The camera moves to the right.",
    "left": "The camera pans to the left.",
    "right": "The camera pans to the right.",
    "up": "The camera tilts up.",
    "down": "The camera tilts down.",
    "W+A": "The camera moves diagonally forward-left.",
    "W+D": "The camera moves diagonally forward-right.",
    "W+left": "The camera pushes forward while panning to the left.",
    "W+right": "The camera pushes forward while panning to the right.",
    "W+up": "The camera pushes forward while tilting up.",
    "W+down": "The camera pushes forward while tilting down.",
}

FP_TEMPLATE = (
    "<Picture 1> is the exact first frame of the target video: begin the video on "
    "<Picture 1>, reproducing it faithfully, and keep its style and lighting for the "
    "whole shot. <Video 1> is the motion and camera reference: a minimal first-person "
    "previz with only an infinite light-gray ground with a 1-meter grid and two static "
    "red reference boxes ahead — no other objects — {fp_motion} The camera follows the "
    "movement, framing and timing of <Video 1> exactly, in continuous first-person "
    "view. Camera: {camera} Replace the previz grid ground and the two red reference "
    "boxes with the real scene from <Picture 1>: {env} Everything else stays "
    "consistent with <Picture 1>. No previz geometry may remain: the output is the "
    "final polished cinematic first-person shot in the exact style of <Picture 1>."
)


def build_first_person(case: dict, action: str) -> str:
    """第一人称 prompt:无 <Subject 1>(相机即主体),运动句 + 环境 + 反残留。"""
    if action not in FIRST_PERSON_MOTION:
        raise KeyError(f"action {action!r} 不在第一人称动作表内: {sorted(FIRST_PERSON_MOTION)}")
    if case.get("settings", {}).get("perspective") != "first_person":
        raise ValueError(f"case_{case.get('id')} 不是 first_person")
    return FP_TEMPLATE.format(
        fp_motion=FIRST_PERSON_MOTION[action],
        camera=case.get("perspective_prompt", "").rstrip("."),
        env=case.get("environment_prompt", ""))


def build_prompt(case: dict, action: str, desc: str = None, style: str = "flat") -> str:
    if style == "fp":
        return build_first_person(case, action)
    builder = build_structured if style == "structured" else build_flat
    return builder(case, action, desc)


def iter_third_person_nav(cases_dir=WBENCH / "cases"):
    """全部 third_person 导航 case(nav_cate ∈ 六类)。"""
    nav = {"progressive", "roundtrip", "repeat", "loop", "l_shape", "zigzag"}
    for f in sorted(cases_dir.glob("case_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        c = json.loads(f.read_text())
        if c.get("nav_cate") in nav and c.get("settings", {}).get("perspective") == "third_person":
            yield c


def self_check() -> None:
    """md 模版槽位齐全,且三个槽位可无冲突 format(无多余花括号)。"""
    block = load_template()
    block.format(desc="D", env="E", camera="C")
    print("self-check ok: wbench_template.md 模版槽位 {desc}/{env}/{camera} 齐全且可 format")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="case json 路径(或纯 id 如 105)")
    ap.add_argument("--action", default="W")
    ap.add_argument("--style", choices=("structured", "flat", "fp"), default="flat",
                    help="flat=基线句式+红盒(默认);structured=六段式对照;fp=第一人称")
    ap.add_argument("--all", action="store_true", help="全部 third_person 导航 case 批量导出")
    ap.add_argument("--out", default=None, help="--all 时的输出 json(默认 stdout)")
    ap.add_argument("--self-check", action="store_true",
                    help="校验 md 模版槽位后退出")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    if args.all:
        out = {}
        for c in iter_third_person_nav():
            for act in dict.fromkeys(i["action"] for i in c["interactions"]
                                     if i["type"] == "navigation"):
                out[f"case_{c['id']}_{act.replace('+', '_')}"] = build_prompt(c, act, style=args.style)
        if args.out:
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
            print(f"{len(out)} prompts ({args.style}) → {args.out}")
        else:
            print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    path = Path(args.case)
    if not path.exists():
        path = WBENCH / "cases" / (f"case_{int(args.case)}.json"
                                   if args.case.isdigit() else args.case)
    print(build_prompt(json.loads(path.read_text()), args.action, style=args.style))


if __name__ == "__main__":
    main()
