"""数据自洽性检查(headless 等价,零 GPU)。

新数据的"机制层"已由数据生成方预先结算并写进 instruction/gold/judge,
没有可重放的 DSL;本模块用纯文本/结构交叉验证替代旧 headless 重放。

可靠关系(100/100 数据上核实):
- outcome expected == occurs.judge_label;not_occurs.judge_label 是同问题的
  另一选项,A/B 两侧 label 镜像对称;
- instruction 的 resolved-rule 句与 gold_branch_self 逐字相等;
- must-NOT / Instead 两句与 not_occurs 的 phrase/instead_remains 逐字一致;
- gold_end_state[occurs.variable] == occurs.value,反物体落在补状态,
  delta 是 gold_end_state 的子集且含 occurs 变量;
- rule.mapping[hidden_value] 把 occurs 变量赋成 occurs.value;
- resolved 句 / gold_branch_self 只含本侧判别词(两分支短语的差集 token),
  gold_branch_other 只含对侧判别词——分支描述混串在此被拦下。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.casebench.dataset import Case, TwinSpec
from evaluation.casebench.framework import get_path, resolve_rule

_RESOLVED_RULE_RE = re.compile(r", so the resolved rule is: (.+?)\. ", re.DOTALL)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# 从不判别分支的虚词(判别词差集中剔除)。
_STOPWORDS = {"into", "onto", "with", "while", "and", "of", "to", "on", "in",
              "at", "by", "for", "from", "over", "across", "behind", "the",
              "a", "an", "is", "are", "it", "its", "only", "one"}


def _norm(value: object) -> str:
    return str(value).strip().lower().replace("'", "").replace('"', "")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _side_text(instruction: str, side_gold: str) -> tuple[str, str]:
    """(resolved-rule 句, gold_branch_self)。"""
    matched = _RESOLVED_RULE_RE.search(instruction)
    resolved = " ".join(matched.group(1).split()) if matched else ""
    return resolved, " ".join(side_gold.split())


def check_twin(twin: TwinSpec) -> list[str]:
    """单侧数据自洽检查;返回问题列表(空 = 通过)。"""
    problems: list[str] = []
    terminal = twin.raw.get("terminal_visual_states") or {}
    occurs = terminal.get("occurs") or {}
    not_occurs = terminal.get("not_occurs") or {}
    outcome = twin.outcome_question
    expected = str(outcome["expected"])

    def bad(msg: str) -> None:
        problems.append(msg)

    # 1. outcome 问题的 expected / answers 与 occurs/not_occurs 的 label 互指
    if expected != str(occurs.get("judge_label")):
        bad(f"outcome expected {expected!r} != occurs.judge_label "
            f"{occurs.get('judge_label')!r}")
    counter_label = str(not_occurs.get("judge_label", ""))
    if counter_label == expected:
        bad("not_occurs.judge_label equals expected; branches must differ")
    if counter_label and counter_label not in [str(a) for a in outcome["answers"]]:
        bad(f"not_occurs.judge_label {counter_label!r} not in outcome answers")
    if "unclear" not in [_norm(a) for a in outcome["answers"]]:
        bad("outcome answers missing 'unclear' option")

    # 2. resolved-rule 句存在,且与 gold_branch_self 逐字相等
    resolved, side_gold = _side_text(twin.instruction, twin.gold_branch_self)
    if not resolved:
        bad("instruction missing '<var> = <value>, so the resolved rule is:' sentence")
    elif resolved != side_gold:
        bad(f"resolved-rule sentence != gold_branch_self: {resolved!r} vs {side_gold!r}")

    # 3. must-NOT / Instead 两句与 not_occurs 声明逐字一致
    if not_occurs.get("phrase"):
        must_not = (f"The counter-outcome must NOT happen at any point: "
                    f"{not_occurs['phrase']}.")
        if must_not not in twin.instruction:
            bad(f"instruction missing verbatim must-NOT sentence: {must_not!r}")
    instead = not_occurs.get("instead_remains")
    if instead and f"Instead, {instead} throughout the clip." not in twin.instruction:
        bad(f"instruction missing verbatim Instead sentence for {instead!r}")

    # 4. gold 终态表与 occurs/not_occurs 的变量取值互指(delta 是其子集)
    gold_end = {str(k): v for k, v in twin.gold_end_state.items()}
    var = str(occurs.get("variable", ""))
    if not var:
        bad("occurs.variable missing")
    elif var not in gold_end:
        bad(f"gold_end_state missing occurs variable {var!r}")
    elif _norm(gold_end[var]) != _norm(occurs.get("value")):
        bad(f"gold_end_state[{var!r}]={gold_end[var]!r} != occurs.value "
            f"{occurs.get('value')!r}")
    counter_var = str(not_occurs.get("variable", ""))
    if counter_var and counter_var != var:
        if counter_var not in gold_end:
            bad(f"gold_end_state missing counter variable {counter_var!r}")
        elif _norm(gold_end[counter_var]) == _norm(not_occurs.get("value")):
            bad(f"gold_end_state[{counter_var!r}] equals not_occurs.value; "
                "counter must stay in the complementary state")
    if var and var not in twin.delta:
        bad(f"delta missing occurs variable {var!r}")
    elif var and _norm(twin.delta[var]) != _norm(occurs.get("value")):
        bad(f"delta[{var!r}]={twin.delta[var]!r} != occurs.value "
            f"{occurs.get('value')!r}")
    if set(twin.delta) - set(gold_end):
        bad(f"delta has keys outside gold_end_state: "
            f"{sorted(set(twin.delta) - set(gold_end))}")

    # 5. rule.mapping[hidden_value] 必须把 occurs 变量赋成 occurs.value
    mapping = (twin.raw.get("rule") or {}).get("mapping") or {}
    if mapping:
        entry = mapping.get(twin.hidden_value)
        if entry is None:
            bad(f"rule.mapping missing hidden value {twin.hidden_value!r}")
        elif f"{_norm(var)}={_norm(occurs.get('value'))}" != _norm(entry).replace(" ", ""):
            bad(f"rule.mapping[{twin.hidden_value!r}]={entry!r} does not assign "
                f"{var!r}={occurs.get('value')!r}")

    # 7. 规则结算重放(迷你引擎,零 GPU 的 headless 等价):终态必须与
    #    gold_end_state 逐键一致——机制层从此有构造性保证。
    try:
        replayed, _executed = resolve_rule(twin)
    except Exception as exc:  # noqa: BLE001 - 解析失败本身就是数据问题
        bad(f"rule replay failed: {exc}")
    else:
        for path, value in twin.gold_end_state.items():
            actual = get_path(replayed, str(path))
            if actual is None:
                bad(f"rule replay state missing {path!r} (gold {value!r})")
            elif _norm(actual) != _norm(value):
                bad(f"rule replay {path}={actual!r} != gold {value!r}")

    # 6. 分支描述不得混串:本侧文本不得包含反结局 judge_label(完整串),
    #    且必须含本侧判别词;对侧文本对称。判别词 = 两分支短语的差集 token
    #    (去虚词);不检查差集 token 的"泄漏"方向,因为自侧描述合法地
    #    会以补状态提及反物体(如 "while the right one stays dark")。
    if not_occurs.get("phrase") and resolved:
        occ_tokens = _tokens(str(occurs.get("phrase", "")))
        not_tokens = _tokens(str(not_occurs["phrase"]))
        self_unique = occ_tokens - not_tokens - _STOPWORDS
        counter_unique = not_tokens - occ_tokens - _STOPWORDS
        if counter_label and counter_label.lower() in resolved.lower():
            bad("resolved-rule/gold_self contains counter judge_label "
                f"{counter_label!r}")
        if self_unique and not (self_unique & _tokens(resolved)):
            bad(f"resolved-rule/gold_self lacks any self-unique token "
                f"{sorted(self_unique)}")
        other_text = " ".join(twin.gold_branch_other.split())
        if expected.lower() in other_text.lower():
            bad(f"gold_branch_other contains self judge_label {expected!r}")
        if counter_unique and not (counter_unique & _tokens(other_text)):
            bad(f"gold_branch_other lacks any counter-unique token "
                f"{sorted(counter_unique)}")
    return problems


def check_pair(first: TwinSpec, second: TwinSpec) -> list[str]:
    """成对检查:label 与变量必须 A↔B 镜像对称。"""
    problems: list[str] = []
    ta = (first.raw.get("terminal_visual_states") or {})
    tb = (second.raw.get("terminal_visual_states") or {})
    if ta.get("not_occurs", {}).get("judge_label") != tb.get("occurs", {}).get("judge_label"):
        problems.append(f"{first.clip_id}: not_occurs label != counterpart occurs label")
    if tb.get("not_occurs", {}).get("judge_label") != ta.get("occurs", {}).get("judge_label"):
        problems.append(f"{second.clip_id}: not_occurs label != counterpart occurs label")
    if ta.get("occurs", {}).get("variable") != tb.get("not_occurs", {}).get("variable"):
        problems.append(f"{first.clip_id}: occurs.variable != counterpart "
                        "not_occurs.variable")
    return problems


def run_headless(cases: list[Case]) -> tuple[int, dict[str, list[str]]]:
    """全量检查;返回 (通过侧数, {clip_id: problems})。"""
    checked: dict[str, list[str]] = {}
    passed = 0
    for case in cases:
        first, second = case.twins
        pair_problems = check_pair(first, second)
        for twin in case.twins:
            problems = check_twin(twin)
            if twin is first:
                problems = problems + [p for p in pair_problems
                                       if p.startswith(first.clip_id)]
            else:
                problems = problems + [p for p in pair_problems
                                       if p.startswith(second.clip_id)]
            checked[twin.clip_id] = problems
            if not problems:
                passed += 1
    return passed, checked


def write_headless(run_dir: Path, passed: int, checked: dict[str, list[str]]) -> Path:
    path = Path(run_dir) / "headless.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "passed": passed,
        "total": len(checked),
        "problems": {k: v for k, v in checked.items() if v},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


__all__ = ["check_pair", "check_twin", "run_headless", "write_headless"]
