"""报告:verdicts → metrics.json + report.md + FAILURES.md(失败 clip 全档)。"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.dataset import (Case, TwinSpec,
                                                 build_raw_state_prompt)
from evaluation.metrics import aggregate, failures, judge_row

VERDICTS_DIRNAME = "verdicts"


def verdict_path(run_dir: Path, clip_id: str) -> Path:
    return Path(run_dir) / VERDICTS_DIRNAME / f"{clip_id}.json"


def video_path_for(run_dir: Path, twin) -> Path:
    """run 布局里该 clip 的视频路径(videos/<CLIP>.mp4,可为软链)。"""
    return Path(run_dir) / "videos" / f"{twin.clip_id}.mp4"


def final_frame_for(run_dir: Path, twin) -> Path:
    """run 布局里该 clip 的末帧路径(frames/<CLIP>.png)。"""
    return Path(run_dir) / "frames" / f"{twin.clip_id}.png"


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
        render_failures_md(run_dir, cases, rows,
                           prompt_mode=str(meta.get("prompt_mode", ""))),
        encoding="utf-8")
    return metrics_path, report_path


def _prompt_of(twin: TwinSpec, prompt_mode: str, run_dir: Path) -> str:
    """复现该 run 实际发给视频模型的完整 prompt(优先读推理时存档)。"""
    stored = Path(run_dir) / "prompts" / f"{twin.clip_id}.txt"
    if stored.is_file():
        return stored.read_text(encoding="utf-8")
    if prompt_mode == "raw":
        return twin.baseline_text
    if prompt_mode == "raw_state":
        return build_raw_state_prompt(twin)
    return twin.instruction


def render_failures_md(run_dir: Path, cases: list[Case], rows: list[dict],
                       *, prompt_mode: str) -> str:
    """Full dossier of STRICT failures: verdict + gold + actual prompt + artifact paths."""
    by_clip = {row["clip_id"]: row for row in rows}
    mode_note = prompt_mode or "(not recorded; reconstructed from stored archives / raw text)"
    lines = ["# Failed clips under the STRICT criterion (own ending shown AND "
             "counter-outcome absent)", "",
             f"prompt_mode: `{mode_note}`", ""]
    missed = failures(rows)
    if not missed:
        lines.append("(no failures)")
        return "\n".join(lines) + "\n"
    lines.append(f"{len(missed)} failed clip(s).")
    for case in cases:
        for twin in case.twins:
            row = by_clip.get(twin.clip_id)
            if row is None or row["strict_ok"]:
                continue
            outcome_mark = "✓" if row["outcome_ok"] else "✗"
            counter_mark = "✓" if row["counter_ok"] else "✗"
            kinds = []
            if not row["outcome_ok"]:
                kinds.append("wrong/unclear ending" if row["outcome_answer"] != "unclear"
                             else "unclear ending")
            if not row["counter_ok"]:
                kinds.append("counter-outcome appeared")
            lines += [
                "", f"## {twin.clip_id} — {'; '.join(kinds) or 'failed'}",
                "",
                f"- genre/family: {twin.genre} / {twin.family}",
                f"- hidden: `{twin.hidden_variable} = {twin.hidden_value}`"
                f" (counterpart `{twin.hidden_other}`)",
                f"- ending verdict: {row['outcome_answer']!r} (expected "
                f"{row['outcome_expected']!r}){outcome_mark}",
                f"- counter verdict: {row['counter_answer']!r} (expected no){counter_mark}",
                f"- judge evidence: {row['evidence']}",
                f"- gold end state: `{json.dumps(twin.gold_end_state, ensure_ascii=False)}`",
                f"- video: `{row['video'] or '(missing)'}`",
                f"- verdict: `{verdict_path(run_dir, twin.clip_id)}`",
                "", "### Prompt actually sent to the video model", "", "```text",
                _prompt_of(twin, prompt_mode, run_dir), "```",
            ]
    return "\n".join(lines) + "\n"


def render_markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# WorldState-Bench evaluation report",
        "",
        f"- clips: {summary['clips']} / cases: {summary['cases']}"
        f" (pairs judged: {summary['pairs_judged']})",
    ]
    for key, label in (("ENDFOLLOW", "ENDFOLLOW (prescribed ending shown)"),
                       ("CNTABSENT", "CNTABSENT (forbidden outcome absent)"),
                       ("STRICT", "STRICT (headline: only the own outcome appears)"),
                       ("PAIRSTRICT", "PAIRSTRICT (both twins STRICT-correct)"),
                       ("UNCLEAR", "UNCLEAR (unjudgeable output)")):
        value = summary.get(key)
        lines.append(f"- {label}: {_pct(value)}")
    lines += [
        "",
        "### Derived diagnostics",
        "",
        f"- Failure triad: direction error {_pct(summary.get('DirectionError'))} · "
        f"leakage-only {_pct(summary.get('LeakageOnly'))} · "
        f"double error {_pct(summary.get('DoubleError'))}",
        f"- TWINCONTRAST (judged endings differ across the pair): "
        f"{_pct(summary.get('TWINCONTRAST'))} · "
        f"copy-paste (identical answers): {_pct(summary.get('CopyPaste'))}",
        f"- Branch gap (|STRICT_A − STRICT_B|): "
        f"{summary.get('BranchGap', '-')}",
        f"- AnyUnanswered (missing verdict): {_pct(summary.get('AnyUnanswered'))} · "
        f"PairCounter: {_pct(summary.get('PairCounter'))} · "
        f"PAIRFOLLOW: {_pct(summary.get('PAIRFOLLOW'))}",
    ]
    lines += ["", "## By side / family", "",
              "| facet | clips | STRICT | ENDFOLLOW | CNTABSENT | PAIRSTRICT |",
              "|---|---|---|---|---|---|"]
    for side, stats in summary.get("by_side", {}).items():
        lines.append(f"| side {side} | {stats['clips']} | {_pct(stats['STRICT'])} | "
                     f"{_pct(stats['ENDFOLLOW'])} | {_pct(stats['CNTABSENT'])} | - |")
    for family, stats in summary.get("by_family", {}).items():
        lines.append(f"| {family} | {stats['clips']} | {_pct(stats['STRICT'])} | "
                     f"{_pct(stats['ENDFOLLOW'])} | {_pct(stats['CNTABSENT'])} | "
                     f"{_pct(stats['PAIRSTRICT'])} |")
    missed = failures(rows)
    lines += ["", f"## Clips failing STRICT ({len(missed)})", ""]
    if missed:
        lines += ["| clip | ending (expected / judged) | counter appeared? | evidence |",
                  "|---|---|---|---|"]
        lines += [f"| {row['clip_id']} | {row['outcome_expected']} / "
                  f"{row['outcome_answer']} | {row['counter_answer']} | "
                  f"{row['evidence'][:80]} |" for row in missed]
    else:
        lines.append("(none)")
    lines += ["", "## Per clip", "",
              "| clip | END | CNT | strict | outcome |",
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


__all__ = ["final_frame_for", "load_rows", "render_markdown", "verdict_path",
           "video_path_for", "write_report", "write_verdict"]
