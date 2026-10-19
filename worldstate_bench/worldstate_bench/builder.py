"""构建核心：spec → twin YAML + render CSV + Case Admission Test 硬校验。

新 schema（WorldState-Bench counterfactual hidden-state pair，对齐 design 的数据字段(见 README「Dataset facts」)）：
  spec：cid/genre/family/style/scene/player/target/objects/n_persons/
        hidden{var,a,b}/action/vars/init/effect/else_effect/branch_a/b/
        judge{action_q,outcome_q,outcome_opts,initial_qs}/prompt_ok(豁免表)

方法契约（勿随意变更，下游数据 schema 依赖）：
  - YAML 字段：id/counterpart/genre/interaction_family/initial_scene/
    hidden_state/rule/shared_action/instruction/gold_A/gold_B/
    terminal_visual_states（双终态：本 twin occurs + 对侧 not_occurs，
    含 instead_remains 反证短语——强调对侧没有发生）/judge/constraints
    + 机器字段 initial_state/gold_end_state/delta/visual_vocab/prompt_ok（审查用）
  - render.csv 列：case,family,gate,prompt,negative（gate=personN，N=画面人数）
  - gold 全部由 DSL 模拟机械生成（人不写答案）
  - A/B twin 共享同一首帧 prompt：hidden 值与终态短语均不入图
  - instruction 显式负向断言：本分支终态发生 + 对侧终态不发生（改为
    保持初态的正向描述），judge.outcome_question 带 not_expected

硬校验（任一失败整 case 丢弃并打印原因）：
  V1 文法：precondition 恰为 <hidden.var> == '<A>'；effect 均单条 path=literal
  V2 两分支 (path,value) 互异
  V3 分支目标在 vars 且短语覆盖全部可达值（含 init）
  V4 hidden 真翻转分支（A 走 effect、B 走 else）
  V5 双 twin delta 均非空（两侧都要明显变化）
  V6 初值中立（被分支改变的变量 init ∉ 分支值集合）
  V7 outcome judge 每 twin 恰一选项命中且 A/B 期望不同；action judge 存在
  V8 action 中性（LEAK_WORDS/hidden 值不得出现，prompt_ok 豁免）、单句 ≤80 字符
  V9 hidden 值不入生图 prompt（prompt_ok 豁免——撞词的可见对象名词不泄露
      分支，A/B 同图）；初态短语逐字入 prompt；第三人称玩家入 prompt
  V10 initial_questions 非空且为封闭式（非 Wh- 开头，可 yes 作答）
"""
from __future__ import annotations

import copy
import csv
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worldstate_bench.dsl import (StrictGrammarError, eval_pre, get_path, parse_effect,  # noqa: E402
                     parse_pre, set_path, simulate)

# 动作里的结局泄露词（动作必须中性：说玩家做什么，不说世界变成什么）。
# v4（50-case trigger 版）：动作全是身体 trigger（举杖/挥手/鼓掌/鞠躬…），
# 姿态词放行；只拦"世界变化"谓词与名词。颜色/方位/up/down 不拦。
LEAK_WORDS = re.compile(
    r"\b(?:open|opens|opened|close|closes|closed|shut|shuts|light|lights|lit|"
    r"glow|glows|glowing|ignite|ignites|extinguish|burn|burns|flame|flames|"
    r"fire|fires|freeze|freezes|frozen|shatter|shatters|crack|cracks|"
    r"break|breaks|broken|split|splits|bloom|blooms|sprout|sprouts|"
    r"spray|sprays|jet|jets|unfurl|unfurls|unfold|unfolds|sink|sinks)\b", re.I)

WH_OPENERS = re.compile(r"^\s*(what|why|how|which|where|when|who)\b", re.I)

# 视频 instruction 模板（机械 Prompt 拼装规范(见模板注释)）：
# 保首帧 → 已结算分支（只注入本 twin 唯一正确终态）→ 共享动作 → 定格 ~1s
VIDEO_DURATION = "3-second"


def image_prompt(scene: str, player: str, init_phrases: list[str],
                 style: str) -> str:
    """共享首帧 prompt：场景 + 第三人称玩家 + 全部初态短语 + 风格。

    初态短语逐字取自 vars（单一来源）；A/B twin 用同一张图，hidden 值与
    终态短语均不入图。v4 的 STYLE 自带机位（medium-wide fixed gameplay
    camera），此处不再重复机位描述。未点燃初态一律用正向熄灭描述
    （pale grey cold ash / dark matte glass / dead bulb），FLUX 无负向通道。
    """
    return (f"{scene}, {player}, {', '.join(init_phrases)}, "
            f"all main objects large and fully visible with no occlusion, "
            f"{style}, 16:9 landscape")


def validate_prompt(prompt: str, *, init_phrases: list[str],
                    hidden_tokens: list[str], prompt_ok=(),
                    terminal_phrases=()) -> list[str]:
    """prompt 机械校验（V9 子集，build 与 review 修订路径共用）。

    返回错误列表，空 = 合法。terminal_phrases 仅 review 修订路径启用
    （build 期不检查——首帧 prompt 本来就不含终态短语，V1-V10 行为不变）。
    """
    errs: list[str] = []
    if "third-person" not in prompt:
        errs.append("prompt must state third-person camera")
    ok_words = set(prompt_ok)
    for t in hidden_tokens:
        if t in ok_words:
            continue
        if re.search(rf"\b{re.escape(t)}s?\b", prompt, re.I):
            errs.append(f"hidden value `{t}` appears in prompt")
    for ph in init_phrases:
        if ph not in prompt:
            errs.append(f"init phrase missing verbatim: `{ph[:60]}…`")
    for ph in terminal_phrases or ():
        if ph.lower() in prompt.lower():
            errs.append(f"terminal-state phrase must not appear: `{ph[:60]}…`")
    return errs


def _single_literal(stmts: list, src: str) -> tuple[str, object]:
    """要求 effect 恰为单条 `path = literal`，返回 (path, value)。"""
    if len(stmts) != 1:
        raise StrictGrammarError(f"effect must be single stmt: `{src}`")
    e = stmts[0]
    if not hasattr(e, "value") or e.value is None or hasattr(e, "num") \
            or hasattr(e, "func"):
        raise StrictGrammarError(f"effect must be `path = literal`: `{src}`")
    return e.path, e.value


def _expected_label(jq: dict, post: dict) -> str | None:
    for label, when in jq["options"]:
        if all(get_path(post, p) == v for p, v in when.items()):
            return label
    return None


def build_case(s: dict) -> tuple[list[dict], dict | None, list[str]]:
    """返回 (yaml_docs[2], csv_row, failures)。failures 非空则前两者为空。"""
    bad: list[str] = []
    cid, fam = s["cid"], s["family"]
    genre = s["genre"]
    hid = s["hidden"]
    var, a, b = hid["var"], hid["a"], hid["b"]
    toks = (a, b)
    eff, xeff = s["effect"], s["else_effect"]
    pre = f"{var} == '{a}'"

    # ---- V1 文法 ----------------------------------------------------------
    try:
        pc = parse_pre(pre)
        ep, ev = _single_literal(parse_effect(eff), eff)
        xp, xv = _single_literal(parse_effect(xeff), xeff)
    except StrictGrammarError as e:
        return [], None, [f"[{cid}] grammar: {e}"]
    if (pc.path, pc.op, pc.value) != (var, "==", a):
        bad.append(f"[{cid}] precondition must be `{pre}`")

    # ---- V2 两分支互异 ----------------------------------------------------
    if (ep, ev) == (xp, xv):
        bad.append(f"[{cid}] branches identical: `{eff}` vs `{xeff}`")

    # ---- V3 词表覆盖 ------------------------------------------------------
    for v in (ep, xp):
        if v not in s["vars"]:
            bad.append(f"[{cid}] vars missing branch target `{v}`")
    if bad:
        return [], None, bad
    for v, mapping in s["vars"].items():
        for val in {get_path(s["init"], v), ev if ep == v else None,
                    xv if xp == v else None}:
            if val is not None and val not in mapping:
                bad.append(f"[{cid}] vars[{v}] missing phrase for {val!r}")

    # ---- twin 状态：仅 hidden 值不同 --------------------------------------
    st_a = copy.deepcopy(s["init"])
    set_path(st_a, var, a)
    st_b = copy.deepcopy(st_a)
    set_path(st_b, var, b)

    # ---- V4 分支翻转 + 机械模拟 -------------------------------------------
    if not (eval_pre(pre, st_a) and not eval_pre(pre, st_b)):
        bad.append(f"[{cid}] hidden value does not flip branch")
        return [], None, bad
    post_a, post_b = simulate(st_a, eff), simulate(st_b, xeff)

    def delta_of(pre_st: dict, post_st: dict) -> dict:
        return {v: get_path(post_st, v) for v in s["vars"]
                if get_path(post_st, v) != get_path(pre_st, v)}

    delta_a, delta_b = delta_of(st_a, post_a), delta_of(st_b, post_b)

    # ---- V5 双方都要明显变化 ----------------------------------------------
    if not delta_a:
        bad.append(f"[{cid}] twin A produced no state change")
    if not delta_b:
        bad.append(f"[{cid}] twin B produced no state change")

    # ---- V6 初值中立 ------------------------------------------------------
    for v in {ep, xp}:
        branch_vals = {val for p, val in ((ep, ev), (xp, xv)) if p == v}
        if get_path(st_a, v) in branch_vals:
            bad.append(f"[{cid}] init of `{v}` equals a branch outcome "
                       f"(initial state must be neutral)")

    # ---- V7 judge 机械回填 -------------------------------------------------
    jq = s["judge"]
    if not jq.get("action_q"):
        bad.append(f"[{cid}] judge.action_q missing")
    oq_opts = [(lab, dict(when)) for lab, when in jq.get("outcome_opts", [])]
    if len(oq_opts) < 2:
        bad.append(f"[{cid}] judge.outcome_opts needs >=2 options")
    else:
        for twin, post in (("A", post_a), ("B", post_b)):
            n_hit = sum(1 for _, when in oq_opts
                        if all(get_path(post, p) == v for p, v in when.items()))
            if n_hit != 1:
                bad.append(f"[{cid}] outcome judge matched {n_hit} options on "
                           f"twin {twin}: `{jq['outcome_q']}`")
        exp_a = _expected_label({"options": oq_opts}, post_a)
        exp_b = _expected_label({"options": oq_opts}, post_b)
        if exp_a == exp_b:
            bad.append(f"[{cid}] outcome judge not discriminative: "
                       f"twins share expected answer {exp_a!r}")
    if not jq.get("initial_qs"):
        bad.append(f"[{cid}] judge.initial_qs missing")

    # ---- V8 动作中性 -------------------------------------------------------
    act = s["action"]
    ok_words = set(s.get("prompt_ok", []))
    if m := LEAK_WORDS.search(act):
        bad.append(f"[{cid}] action leaks outcome word `{m.group(0)}`: {act}")
    for t in toks:
        if t not in ok_words and re.search(rf"\b{re.escape(t)}s?\b", act, re.I):
            bad.append(f"[{cid}] action mentions hidden value `{t}`: {act}")
    if len(act) > 120 or act.strip().rstrip(".").count("."):
        bad.append(f"[{cid}] action must be one short sentence: {act}")

    # ---- V9 共享首帧 prompt：初态短语入图、hidden 值不入图 ------------------
    init_phrases = [s["vars"][v][get_path(st_a, v)] for v in s["vars"]]
    scene, player = s["scene"], s["player"]
    prompt = image_prompt(scene, player, init_phrases, s["style"])
    bad += [f"[{cid}] {e}" for e in
            validate_prompt(prompt, init_phrases=init_phrases,
                            hidden_tokens=list(toks),
                            prompt_ok=s.get("prompt_ok", []))]
    if re.search(rf"\b{re.escape(b)}s?\b", s["branch_a"], re.I) or \
            re.search(rf"\b{re.escape(a)}s?\b", s["branch_b"], re.I):
        bad.append(f"[{cid}] branch text mentions the other hidden value")

    # ---- V10 initial_questions 封闭式 --------------------------------------
    for q in jq.get("initial_qs", []):
        if WH_OPENERS.match(q):
            bad.append(f"[{cid}] initial question not closed-form: {q}")

    if bad:
        return [], None, bad

    rule_text = (f"If the {var} is {a}, {s['branch_a']}; "
                 f"if it is {b}, {s['branch_b']}.")

    def doc(twin: str) -> dict:
        tok = a if twin == "A" else b
        st, post = (st_a, post_a) if twin == "A" else (st_b, post_b)
        other_post = post_b if twin == "A" else post_a
        branch = s["branch_a"] if twin == "A" else s["branch_b"]
        own_p, own_v = (ep, ev) if twin == "A" else (xp, xv)
        oth_p, oth_v = (xp, xv) if twin == "A" else (ep, ev)
        exp = _expected_label({"options": oq_opts}, post)
        exp_other = _expected_label({"options": oq_opts}, other_post)
        occ_phrase = s["vars"][own_p][own_v]
        not_phrase = s["vars"][oth_p][oth_v]
        # 同变量双分支（如 orb 红/蓝火焰）：不发生侧的实际呈现就是发生侧终态；
        # 异变量分支：对侧变量保持初态（off 短语）
        remains_phrase = occ_phrase if oth_p == own_p \
            else s["vars"][oth_p][get_path(st, oth_p)]
        instruction = (
            f"Preserve the first frame. {var} = {tok}, so the resolved rule is: "
            f"{branch}. Action: {act} Show the action, then exactly this "
            f"consequence: {branch} The counter-outcome must NOT happen at "
            f"any point: {not_phrase}. Instead, {remains_phrase} throughout "
            f"the clip. Hold the final state for about the last second. One "
            f"continuous {VIDEO_DURATION} third-person clip, fixed camera, "
            f"no HUD, no subtitles, no readable text.")
        return {
            "id": f"{cid}-{twin}",
            "counterpart": f"{cid}-{'B' if twin == 'A' else 'A'}",
            "genre": genre,
            "interaction_family": fam,
            "initial_scene": {
                "description": f"{scene} {player}",
                "third_person_player": True,
                "player_pose_and_item": player,
                "action_target": s.get("target", ""),
                "other_key_objects": list(s["objects"]),
                "shared_visual_state": list(init_phrases),
                "camera": "locked third-person, fixed framing",
                "style": s["style"],
                "keyframe": f"frames/{cid}.png",
            },
            "hidden_state": {
                "variable": var,
                "A": a,
                "B": b,
                "visible_in_initial_frame": False,
            },
            "rule": {
                "text": rule_text,
                "precondition": pre,
                "effect": eff,
                "else_effect": xeff,
                "mapping": {a: eff, b: xeff},
            },
            "shared_action": act,
            "instruction": instruction,
            # ---- 机器字段（审计/校验用）----
            "initial_state": st,
            "gold_end_state": {v: get_path(post, v) for v in s["vars"]},
            "delta": delta_of(st, post),
            "visual_vocab": copy.deepcopy(s["vars"]),
            "gold_A": s["branch_a"],
            "gold_B": s["branch_b"],
            # ---- 双 terminal visual state：本 twin 的发生 + 对侧的不发生 ----
            "terminal_visual_states": {
                "occurs": {
                    "variable": own_p,
                    "value": own_v,
                    "phrase": occ_phrase,
                    "judge_label": exp,
                    "in_final_frame": True,
                },
                "not_occurs": {
                    "variable": oth_p,
                    "value": oth_v,
                    "phrase": not_phrase,
                    "judge_label": exp_other,
                    "in_final_frame": False,
                    "instead_remains": remains_phrase,
                },
            },
            "prompt_ok": list(s.get("prompt_ok", [])),
            "judge": {
                "initial_questions": [
                    {"question": f"Is it true that {q}?",
                     "statement": q,
                     "answers": ["yes", "no"],
                     "expected": "yes"}
                    for q in jq["initial_qs"]],
                "action_question": {
                    "question": jq["action_q"],
                    "answers": ["yes", "no"],
                    "expected": "yes",
                },
                "outcome_question": {
                    "question": jq["outcome_q"],
                    "answers": [lab for lab, _ in oq_opts] + ["unclear"],
                    "expected": exp,
                    "not_expected": exp_other,
                },
            },
            "constraints": {
                "identical_initial_frame": True,
                "hidden_state_not_visually_leaked": True,
                "same_action_across_twins": True,
                "both_outcomes_change": True,
                "one_action_per_chunk": True,
                "duration_seconds": "2-5",
            },
        }

    n_persons = int(s.get("n_persons", 1))
    gate = f"person{n_persons}" + ("_soft" if s.get("soft_gate") else "")
    row = dict(case=cid, family=fam, gate=gate,
               prompt=prompt, negative="__STD__")
    return [doc("A"), doc("B")], row, bad


# ---- spec 加载 -------------------------------------------------------------
def load_specs(specs_dir: Path,
               names: list[str] | None = None) -> tuple[list[tuple[str, list]], list[str]]:
    """按模块名加载 SPECS（specs_*.py）。返回 ([(module_label, specs)], missing)。"""
    import importlib.util

    out, missing = [], []
    labels = names or sorted(p.stem for p in specs_dir.glob("specs_*.py"))
    for label in labels:
        mod = label if label.startswith("specs_") else f"specs_{label}"
        f = specs_dir / f"{mod}.py"
        if f.exists():
            spec = importlib.util.spec_from_file_location(mod, f)
            ns = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ns)
            out.append((mod, list(getattr(ns, "SPECS"))))
        else:
            missing.append(mod)
    return out, missing


# ---- 写出 ------------------------------------------------------------------
def write_yaml(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                                   width=100), encoding="utf-8")


def write_render_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "family", "gate",
                                          "prompt", "negative"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
