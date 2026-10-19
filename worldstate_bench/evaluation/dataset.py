"""案例装载:WorldState-Bench 数据集 data/cases/<CASE>/ → 类型化孪生案例。

数据集随仓库分发在 ``data/cases/``(package 步骤的产物快照);重新生成
后由 ``run.py package`` 自动刷新。``--cases-root`` 可指向任意同构目录。

目录契约(数据自足,无外部 DSL):每个案例目录 =
- ``input.png``                两孪生共享的首帧(已渲染好,评测不再产关键帧);
- ``case_A.json``/``case_B.json``  两侧完整规格(instruction/gold/judge 问题);
- ``card_A.txt``/``card_B.txt``    人类可读卡片(评测不消费,仅 baseline
                                   prompt 构造时取「Inputs to the video model」段)。

孪生语义:同首帧 + 同动作 + 同一条 if/else 规则,只有 hidden_state 取值
相反——视频模型必须按各自 side 的隐藏状态解析出对应分支结局。judge 问题
全部由数据预写,评测代码只负责按帧位提问并核对 expected。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CASES_ROOT = Path(__file__).resolve().parents[1] / "data" / "cases"

SIDES = ("A", "B")

# instruction 里点名反结局主动态的 must-NOT 句(不同物体案例进 prompt 前摘除)。
_COUNTER_SENTENCE_RE = re.compile(
    r"\s*The counter-outcome must NOT happen at any point: .+?(?= Instead,)")


@dataclass(frozen=True)
class TwinSpec:
    """案例的一侧(A 或 B):一段待评测 clip 的完整规格。"""

    case_id: str                    # "B01"
    side: str                       # "A" | "B"
    clip_id: str                    # "B01-A"(= 数据里的 id)
    root: Path                      # 案例目录
    input_image: Path               # 共享首帧
    genre: str
    family: str
    instruction: str                # 数据给视频模型的完整指令(原样使用,不改写)
    rule_text: str
    shared_action: str
    hidden_variable: str
    hidden_value: str               # 本侧取值(如 RED_FLAME)
    hidden_other: str               # 对侧取值
    hidden_visible: bool            # 隐藏状态是否允许在首帧可见(恒 False)
    gold_end_state: dict            # 本侧终态(如 {"orb.flame": "red"})
    delta: dict
    gold_branch_self: str           # 本侧结局描述(gold_A/gold_B 中的本侧句)
    gold_branch_other: str          # 对侧结局描述
    initial_questions: tuple[dict, ...]
    action_question: dict
    outcome_question: dict
    counter_question: dict          # 反分支问题(由 terminal_visual_states 机械派生)
    counter_same_object: bool       # A/B 变化的是否同一物体(同物体反结局只能点名否定)
    constraints: dict
    baseline_text: str              # card txt「Inputs to the video model」整段原文
    raw: dict                       # 原始 JSON(headless 自洽检查用)

    @property
    def counterpart(self) -> str:
        return f"{self.case_id}-{'B' if self.side == 'A' else 'A'}"

    @property
    def gold_end_phrases(self) -> list[str]:
        """gold 终态 × visual_vocab → 结局表现短语(raw_state baseline 用)。"""
        vocab = self.raw.get("visual_vocab") or {}
        phrases: list[str] = []
        for path, value in self.gold_end_state.items():
            options = vocab.get(path) or {}
            key = value if isinstance(value, str) else ("true" if value else "false")
            phrase = options.get(key) or options.get(_norm_key(key))
            if phrase:
                phrases.append(str(phrase))
        return phrases

    @property
    def render_instruction(self) -> str:
        """进 prompt 的 instruction:一律摘除点名反结局主动态的 must-NOT 句。

        生成模型不消化否定,must-NOT 句把反结局的画面点名进 prompt 反而
        易把它渲出来(实测 B08 成对镜像翻转、双结局齐亮)。摘除后反分支
        由紧随的 "Instead, <反物体保持初始状态> throughout the clip."
        表达(remain-unchanged 口径);同物体案例(如球红/蓝光)不再生成
        这一句,由 "exactly this consequence" 自身的正描述承担分支。
        """
        return _COUNTER_SENTENCE_RE.sub("", self.instruction, count=1)


def build_raw_state_prompt(twin: TwinSpec) -> str:
    """raw_state baseline 的完整 prompt:raw 原文 + 追加 gold 终态表现短语。

    与 baseline/prompts_common.py 的同名函数逐字一致(那边从 case 目录
    构造,这边从已装载的 TwinSpec 构造;评测报告复现实际 prompt 用)。
    """
    phrases = "; ".join(twin.gold_end_phrases)
    if not phrases:
        raise ValueError(f"{twin.clip_id}: no gold end phrases from visual_vocab")
    return (f"{twin.baseline_text}\n\nThe clip must end as: {phrases}. "
            "Hold this final state for about the last second.")


@dataclass(frozen=True)
class Case:
    """一对孪生(A/B)构成一个案例。"""

    case_id: str
    genre: str
    family: str
    twins: tuple[TwinSpec, ...]     # 恒 (A, B) 两侧


def load_cases(root: Path | None = None) -> list[Case]:
    """装载全部案例;结构不合格直接报错(评测数据必须完整可用)。"""
    root = Path(root) if root is not None else DEFAULT_CASES_ROOT
    if not root.is_dir():
        raise SystemExit(f"cases root not found: {root}")
    cases = [_load_case(directory) for directory in sorted(root.iterdir())
             if directory.is_dir() and not directory.name.startswith(".")]
    if not cases:
        raise SystemExit(f"no case directories under {root}")
    return cases


def iter_twins(cases: list[Case]) -> list[TwinSpec]:
    """按案例序、A 前/B 后展平(评测的稳定执行序)。"""
    return [twin for case in cases for twin in case.twins]


def filter_cases(cases: list[Case], case_ids: set[str]) -> list[Case]:
    unknown = case_ids - {case.case_id for case in cases}
    if unknown:
        raise SystemExit(f"unknown case ids: {sorted(unknown)}")
    return [case for case in cases if case.case_id in case_ids]


# -- 装载与校验 ---------------------------------------------------------------


def _load_case(case_dir: Path) -> Case:
    twins = tuple(_load_twin(case_dir, side) for side in SIDES)
    _validate_pair(case_dir, twins)
    return Case(case_id=twins[0].case_id, genre=twins[0].genre,
                family=twins[0].family, twins=twins)


def _load_twin(case_dir: Path, side: str) -> TwinSpec:
    raw = json.loads((case_dir / f"case_{side}.json").read_text(encoding="utf-8"))
    case_id = case_dir.name
    clip_id = str(raw.get("id", ""))
    if clip_id != f"{case_id}-{side}":
        raise SystemExit(f"{case_dir / f'case_{side}.json'}: id {clip_id!r} does not "
                         f"match directory/case id {case_id}-{side}")
    judge = raw.get("judge") or {}
    initial = tuple(judge.get("initial_questions") or [])
    action = judge.get("action_question") or {}
    outcome = judge.get("outcome_question") or {}
    if not initial or not action or not outcome:
        raise SystemExit(f"{clip_id}: judge questions incomplete "
                         "(need initial_questions / action_question / outcome_question)")
    for question in (*initial, action, outcome):
        _validate_question(clip_id, question)
    hidden = raw.get("hidden_state") or {}
    input_image = (case_dir / str(raw.get("input_image", "input.png"))).resolve()
    if not input_image.is_file():
        raise SystemExit(f"{clip_id}: input image missing: {input_image}")
    gold_branches = (str(raw.get("gold_A", "")).strip(), str(raw.get("gold_B", "")).strip())
    if not all(gold_branches):
        raise SystemExit(f"{clip_id}: gold_A/gold_B descriptions missing")
    index = SIDES.index(side)
    terminal = raw.get("terminal_visual_states") or {}
    occurs_var = str((terminal.get("occurs") or {}).get("variable", ""))
    not_occurs_var = str((terminal.get("not_occurs") or {}).get("variable", ""))
    counter_same_object = bool(occurs_var) and occurs_var == not_occurs_var
    return TwinSpec(
        case_id=case_id,
        side=side,
        clip_id=clip_id,
        root=case_dir,
        input_image=input_image,
        genre=str(raw.get("genre", "")),
        family=str(raw.get("interaction_family", "")),
        instruction=str(raw.get("instruction", "")).strip(),
        rule_text=str((raw.get("rule") or {}).get("text", "")).strip(),
        shared_action=str(raw.get("shared_action", "")).strip(),
        hidden_variable=str(hidden.get("variable", "")),
        hidden_value=str(hidden.get(side, "")),
        hidden_other=str(hidden.get(SIDES[1 - index], "")),
        hidden_visible=bool(hidden.get("visible_in_initial_frame", False)),
        gold_end_state=dict(raw.get("gold_end_state") or {}),
        delta=dict(raw.get("delta") or {}),
        gold_branch_self=gold_branches[index],
        gold_branch_other=gold_branches[1 - index],
        initial_questions=initial,
        action_question=action,
        outcome_question=outcome,
        counter_question=_counter_question(clip_id, raw),
        counter_same_object=counter_same_object,
        constraints=dict(raw.get("constraints") or {}),
        baseline_text=_baseline_text(clip_id, case_dir, side),
        raw=raw,
    )


def _norm_key(key) -> str:
    return str(key).strip().lower()


def _counter_question(clip_id: str, raw: dict) -> dict:
    """从 terminal_visual_states.not_occurs 机械派生反分支问题。

    数据口径:一个 clip 只允许出现 A/B 其中一个结果。反分支的短语由
    数据给出,代码只排版成可判定问题,不发明任何内容。
    """
    counter = (raw.get("terminal_visual_states") or {}).get("not_occurs") or {}
    phrase = str(counter.get("phrase", "")).strip()
    if not phrase:
        raise SystemExit(f"{clip_id}: terminal_visual_states.not_occurs missing "
                         "(严格结局口径需要反分支短语)")
    return {
        "question": f"At any point in the clip, does {phrase} happen?",
        "answers": ["yes", "no", "unclear"],
        "expected": "no",
    }


def _baseline_text(clip_id: str, case_dir: Path, side: str) -> str:
    """card txt 的「Inputs to the video model」整段原文(baseline 的 prompt)。"""
    card = case_dir / f"card_{side}.txt"
    lines: list[str] = []
    inside = False
    for line in (card.read_text(encoding="utf-8") if card.is_file() else "").splitlines():
        if line.startswith("──"):
            inside = "Inputs to the video model" in line
            continue
        if inside:
            lines.append(line.rstrip())
    block = "\n".join(lines).strip()
    if not block:
        raise SystemExit(f"{clip_id}: card txt missing 'Inputs to the video model' section")
    return block


def _validate_pair(case_dir: Path, twins: tuple[TwinSpec, ...]) -> None:
    """成对完整性:互指 counterpart、隐藏状态取值相反、规则一致。"""
    first, second = twins
    if first.counterpart != second.clip_id or second.counterpart != first.clip_id:
        raise SystemExit(f"{case_dir.name}: A/B ids do not reference each other")
    if first.hidden_value == second.hidden_value:
        raise SystemExit(f"{case_dir.name}: both twins share hidden value "
                         f"{first.hidden_value!r}")
    if first.rule_text != second.rule_text or first.instruction == second.instruction:
        raise SystemExit(f"{case_dir.name}: rule must be shared and instructions "
                         "must differ between twins")
    if first.input_image != second.input_image:
        raise SystemExit(f"{case_dir.name}: twins must share one input image")


def _validate_question(clip_id: str, question: dict) -> None:
    text = str(question.get("question", "")).strip()
    answers = [str(answer) for answer in question.get("answers") or []]
    expected = str(question.get("expected", ""))
    if not text or not answers:
        raise SystemExit(f"{clip_id}: question missing text/answers: {question!r}")
    if expected not in answers:
        raise SystemExit(f"{clip_id}: expected {expected!r} not in answers {answers} "
                         f"for question {text!r}")
