"""报告:verdicts → metrics.json + report.md + FAILURES.md(失败 clip 全档)。"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.casebench.dataset import Case, TwinSpec
from evaluation.casebench.metrics import aggregate, failures, judge_row
from evaluation.casebench.render import build_prompt

VERDICTS_DIRNAME = "verdicts"


def verdict_path(run_dir: Path, clip_id: str) -> Path:
    return Path(run_dir) / VERDICTS_DIRNAME / f"{clip_id}.json"


def write_verdict(run_dir: Path, twin, verdict: dict, *, judge_name: str,
                  n_images: int, images: list[Path], video_path: Path) -> Path:
    """单 clip 判定产物(断点续跑的存在性依据)。"""
    path = verdict_path(run_dir, twin.clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "clip_id": twin.clip_id,
        "judge": judge_name,
        "n_images": n_images,
        "images": [str(image) for image in images],
        "video": str(video_path),
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_rows(run_dir: Path, cases: list[Case]) -> list[dict]:
    """从 verdicts 重算 metrics 行(report 阶段独立于 judge 阶段可跑)。"""
    rows = []
    for case in cases:
        for twin in case.twins:
            path = verdict_path(run_dir, twin.clip_id)
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            rows.append(judge_row(
                twin, record["verdict"], judge_name=str(record.get("judge", "")),
                n_images=int(record.get("n_images", 0)),
                video_path=str(record.get("video", "")),
            ))
    return rows


def write_report(run_dir: Path, cases: list[Case], meta: dict) -> tuple[Path, Path]:
    """聚合 + 落盘;返回 (metrics.json, report.md)。同时产出 FAILURES.md。"""
    run_dir = Path(run_dir)
    rows = load_rows(run_dir, cases)
    summary = aggregate(rows, cases)
    summary["meta"] = meta
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(
        {"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    report_path = run_dir / "report.md"
    report_path.write_text(render_markdown(summary, rows), encoding="utf-8")
    failures_path = run_dir / "FAILURES.md"
    failures_path.write_text(
        render_failures_md(run_dir, cases, rows, prompt_mode=str(meta.get("prompt_mode", ""))),
        encoding="utf-8")
    return metrics_path, report_path


def _prompt_of(twin: TwinSpec, prompt_mode: str, run_dir: Path) -> str:
    """复现该 run 实际发给 H3 的完整 prompt(优先读渲染时存档)。"""
    stored = Path(run_dir) / "prompts" / f"{twin.clip_id}.txt"
    if stored.is_file():
        return stored.read_text(encoding="utf-8")
    if prompt_mode == "raw":
        return twin.baseline_text
    if prompt_mode == "raw_state":
        from evaluation.casebench.render import build_raw_state_prompt

        return build_raw_state_prompt(twin)
    if prompt_mode == "remain":
        return build_prompt(twin.render_instruction)
    return build_prompt(twin.instruction)


def render_failures_md(run_dir: Path, cases: list[Case], rows: list[dict],
                       *, prompt_mode: str) -> str:
    """严格口径失败 clip 的全档案:判定 + gold + 实际 prompt + 产物路径。"""
    by_clip = {row["clip_id"]: row for row in rows}
    lines = ["# 失败 clip 档案(严格口径:结局答对 且 反分支未出现)", "",
             f"prompt_mode: `{prompt_mode or '(未记录,按 envelope 复现)'}`", ""]
    missed = failures(rows)
    if not missed:
        lines.append("(无失败)")
        return "\n".join(lines) + "\n"
    lines.append(f"共 {len(missed)} 个失败 clip。")
    for case in cases:
        for twin in case.twins:
            row = by_clip.get(twin.clip_id)
            if row is None or row["strict_ok"]:
                continue
            outcome_mark = "✓" if row["outcome_ok"] else "✗"
            counter_mark = "✓" if row["counter_ok"] else "✗"
            kinds = []
            if not row["outcome_ok"]:
                kinds.append("结局走错/不明" if row["outcome_answer"] != "unclear"
                             else "结局不明")
            if not row["counter_ok"]:
                kinds.append("反分支出现")
            lines += [
                "", f"## {twin.clip_id} — {'; '.join(kinds) or '失败'}",
                "",
                f"- genre/family: {twin.genre} / {twin.family}",
                f"- hidden: `{twin.hidden_variable} = {twin.hidden_value}`"
                f"(对侧 `{twin.hidden_other}`)",
                f"- 结局判定: {row['outcome_answer']!r}(期望 "
                f"{row['outcome_expected']!r}){outcome_mark}",
                f"- counter 判定: {row['counter_answer']!r}(期望 no){counter_mark}",
                f"- 判定证据: {row['evidence']}",
                f"- gold 终态: `{json.dumps(twin.gold_end_state, ensure_ascii=False)}`",
                f"- 视频: `{row['video'] or '(缺失)'}`",
                f"- verdict: `{verdict_path(run_dir, twin.clip_id)}`",
                "", "### 实际发送给 H3 的 prompt", "", "```text",
                _prompt_of(twin, prompt_mode, run_dir), "```",
            ]
    return "\n".join(lines) + "\n"


def render_markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# WorldState-Bench harness 评测报告",
        "",
        f"- clips: {summary['clips']} / cases: {summary['cases']}"
        f"(成对判定 {summary['pairs_judged']} 对)",
    ]
    for key, label in (("EndingFollow", "结局跟随 EndingFollow"),
                       ("CounterAbsent", "反分支未出现 CounterAbsent"),
                       ("StrictFollow", "严格结局 StrictFollow(头条:只出现本侧结果)"),
                       ("PairStrict", "成对严格 PairStrict"),
                       ("UnclearOutcome", "结局不明 UnclearOutcome")):
        value = summary.get(key)
        lines.append(f"- {label}: {_pct(value)}")
    lines += [
        "",
        "### 派生诊断",
        "",
        f"- 失败三分:方向走错 DirectionError {_pct(summary.get('DirectionError'))} · "
        f"纯泄漏 LeakageOnly {_pct(summary.get('LeakageOnly'))} · "
        f"双错 DoubleError {_pct(summary.get('DoubleError'))}",
        f"- 孪生判别 TwinContrast(A/B 结局回答被区分): "
        f"{_pct(summary.get('TwinContrast'))} · "
        f"退化复制 CopyPaste: {_pct(summary.get('CopyPaste'))}",
        f"- 分支偏好 BranchGap(A/B Strict 之差): "
        f"{summary.get('BranchGap', '-')}",
        f"- 答案缺失 AnyUnanswered: {_pct(summary.get('AnyUnanswered'))} · "
        f"PairCounter: {_pct(summary.get('PairCounter'))} · "
        f"PairFollow: {_pct(summary.get('PairFollow'))}",
    ]
    lines += ["", "## 按 side / family", "",
              "| 切面 | clips | StrictFollow | EndingFollow | CounterAbsent | PairStrict |",
              "|---|---|---|---|---|---|"]
    for side, stats in summary.get("by_side", {}).items():
        lines.append(f"| side {side} | {stats['clips']} | {_pct(stats['StrictFollow'])} | "
                     f"{_pct(stats['EndingFollow'])} | {_pct(stats['CounterAbsent'])} | - |")
    for family, stats in summary.get("by_family", {}).items():
        lines.append(f"| {family} | {stats['clips']} | {_pct(stats['StrictFollow'])} | "
                     f"{_pct(stats['EndingFollow'])} | {_pct(stats['CounterAbsent'])} | "
                     f"{_pct(stats['PairStrict'])} |")
    missed = failures(rows)
    lines += ["", f"## 严格口径失败的 clip({len(missed)})", ""]
    if missed:
        lines += ["| clip | 结局(期望/判定) | counter 出现? | 证据 |",
                  "|---|---|---|---|"]
        lines += [f"| {row['clip_id']} | {row['outcome_expected']} / "
                  f"{row['outcome_answer']} | {row['counter_answer']} | "
                  f"{row['evidence'][:80]} |" for row in missed]
    else:
        lines.append("(无)")
    lines += ["", "## 逐 clip", "",
              "| clip | End | Counter | strict | outcome |",
              "|---|---|---|---|---|"]
    lines += [f"| {row['clip_id']} | "
              f"{_mark(row['outcome_ok'])} | "
              f"{_mark(row['counter_ok'])} | {_mark(row['strict_ok'])} | "
              f"{row['outcome_answer']} |" for row in rows]
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


__all__ = ["load_rows", "render_markdown", "verdict_path", "write_report", "write_verdict"]
