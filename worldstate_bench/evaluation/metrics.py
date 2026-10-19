"""指标:verdict + 案例 gold → 单 clip 指标行与聚合。

clip 级核心指标(全部是"答案逐字等于 expected"):
- ENDFOLLOW     结局问题答对本侧分支(隐藏状态解析正确);
- CNTABSENT    反分支问题答 no(counter-outcome 全片未出现);
- STRICT       ENDFOLLOW ∧ CNTABSENT——数据口径"只允许出现
  A/B 其中一个结果"的直接判据(头条指标)。
initial(前提)与 action(动作)问题的回答不进指标、不参与任何派生。
案例级指标:
- PairFollow / PairStrict   A/B 两侧对应核心指标都成立;
- PairCounter               A/B 两侧都无反分支。

派生诊断指标(aggregate,全部由既有判定零 GPU 重算):
- DirectionError / LeakageOnly / DoubleError
  严格失败的三分:方向走错 / 结局对但纯泄反分支 / 双错;
- TwinContrast   案例内 A/B 结局回答被区分的比例(防"两侧同答案");
- CopyPaste      A/B 判定逐字段全同的比例(完全无视隐藏状态);
- AnyUnanswered  outcome/counter 答案缺失(解析失败或越界)率;
- BranchGap      A/B 两侧 STRICT 之差(分支偏好)。
outcome 答案为 'unclear' 单独计数(UnclearOutcome),不算对也不另立口径。
"""

from __future__ import annotations

from evaluation.dataset import Case, TwinSpec


def judge_row(twin: TwinSpec, verdict: dict, *, judge_name: str,
              n_images: int, video_path: str) -> dict:
    """单 clip 判定结果 → metrics 行(verdict 答案已对齐选项原文)。"""
    action_answer = verdict.get("action")
    outcome_answer = verdict.get("outcome")
    counter_answer = verdict.get("counter")
    outcome_ok = outcome_answer == str(twin.outcome_question["expected"])
    counter_ok = counter_answer == str(twin.counter_question["expected"])
    return {
        "case_id": twin.case_id,
        "clip_id": twin.clip_id,
        "side": twin.side,
        "genre": twin.genre,
        "family": twin.family,
        "hidden_variable": twin.hidden_variable,
        "hidden_value": twin.hidden_value,
        "outcome_ok": outcome_ok,
        "counter_ok": counter_ok,
        "strict_ok": outcome_ok and counter_ok,
        "direction_error": not outcome_ok,
        "leakage_only": outcome_ok and not counter_ok,
        "double_error": (not outcome_ok) and (not counter_ok),
        "any_unanswered": outcome_answer is None or counter_answer is None,
        "outcome_unclear": outcome_answer is not None
        and outcome_answer.lower() == "unclear",
        "outcome_answer": outcome_answer,
        "outcome_expected": str(twin.outcome_question["expected"]),
        "counter_answer": counter_answer,
        "counter_question": twin.counter_question["question"],
        "action_answer": action_answer,
        "evidence": str(verdict.get("evidence", "")),
        "judge": judge_name,
        "n_images": n_images,
        "video": video_path,
    }


def aggregate(rows: list[dict], cases: list[Case]) -> dict:
    """metrics 行 → 汇总(核心口径 / 失败分解 / 孪生对比 / 条件与稳定性)。"""
    by_clip = {row["clip_id"]: row for row in rows}
    total = len(rows)
    summary: dict = {
        "clips": total,
        "cases": len(cases),
        # -- 核心口径 --
        "ENDFOLLOW": _rate(rows, "outcome_ok"),
        "CNTABSENT": _rate(rows, "counter_ok"),
        "STRICT": _rate(rows, "strict_ok"),
        "UNCLEAR": _rate(rows, "outcome_unclear"),
    }
    pair_rows = []
    contrast_flags: list[bool] = []
    copy_flags: list[bool] = []
    for case in cases:
        first = by_clip.get(f"{case.case_id}-A")
        second = by_clip.get(f"{case.case_id}-B")
        if not (first and second):
            continue
        pair_rows.append({
            "case_id": case.case_id,
            "family": case.family,
            "pair_follow": all(row["outcome_ok"] for row in (first, second)),
            "pair_strict": all(row["strict_ok"] for row in (first, second)),
            "pair_counter": all(row["counter_ok"] for row in (first, second)),
        })
        # 孪生对比:两侧结局回答是否被区分(防"两侧同答案"的偷懒模式)
        a_outcome, b_outcome = first["outcome_answer"], second["outcome_answer"]
        if a_outcome is not None and b_outcome is not None:
            contrast_flags.append(a_outcome != b_outcome)
        copy_flags.append((a_outcome == b_outcome
                           and first["counter_answer"] == second["counter_answer"]))
    summary["pairs_judged"] = len(pair_rows)
    summary["PAIRFOLLOW"] = _rate(pair_rows, "pair_follow")
    summary["PAIRSTRICT"] = _rate(pair_rows, "pair_strict")
    summary["PairCounter"] = _rate(pair_rows, "pair_counter")
    # -- 孪生判别与退化为 --
    summary["TWINCONTRAST"] = (round(sum(contrast_flags) / len(contrast_flags), 4)
                               if contrast_flags else None)
    summary["CopyPaste"] = (round(sum(copy_flags) / len(copy_flags), 4)
                            if copy_flags else None)
    # -- 失败分解(相对严格口径) --
    summary["DirectionError"] = _rate(rows, "direction_error")
    summary["LeakageOnly"] = _rate(rows, "leakage_only")
    summary["DoubleError"] = _rate(rows, "double_error")
    # -- 答案完整性 --
    summary["AnyUnanswered"] = (round(sum(1 for row in rows if row["any_unanswered"])
                                      / total, 4) if total else None)
    side_strict = {side: _rate([row for row in rows if row["side"] == side], "strict_ok")
                   for side in ("A", "B")}
    if None not in side_strict.values():
        summary["BranchGap"] = round(abs(side_strict["A"] - side_strict["B"]), 4)
    summary["by_side"] = {}
    for side in ("A", "B"):
        side_rows = [row for row in rows if row["side"] == side]
        summary["by_side"][side] = {
            "clips": len(side_rows),
            "ENDFOLLOW": _rate(side_rows, "outcome_ok"),
            "CNTABSENT": _rate(side_rows, "counter_ok"),
            "STRICT": _rate(side_rows, "strict_ok"),
        }
    summary["by_family"] = {}
    for family in sorted({case.family for case in cases}):
        family_rows = [row for row in rows if row["family"] == family]
        family_pairs = [pair for pair in pair_rows if pair["family"] == family]
        summary["by_family"][family] = {
            "clips": len(family_rows),
            "ENDFOLLOW": _rate(family_rows, "outcome_ok"),
            "CNTABSENT": _rate(family_rows, "counter_ok"),
            "STRICT": _rate(family_rows, "strict_ok"),
            "PAIRSTRICT": _rate(family_pairs, "pair_strict"),
        }
    return summary


def failures(rows: list[dict]) -> list[dict]:
    """严格口径没过的 clip(结局答错或反分支出现;报告失败清单,按案例序)。"""
    return [row for row in rows if not row["strict_ok"]]


def _rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row.get(key)) / len(rows), 4)


__all__ = ["aggregate", "failures", "judge_row"]
