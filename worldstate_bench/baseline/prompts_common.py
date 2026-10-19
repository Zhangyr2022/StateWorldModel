"""共用: 从数据集 data/cases 构造 raw / raw_state prompt (与论文两个 probe 的措辞逐字对齐) + manifest。

用法: 数据集随仓库分发在 data/cases/(重新生成后 run.py package 会自动刷新), 再
  python3 baseline/prompts_common.py longcat hy15 wan22   # 生成 baseline/manifests/*.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # worldstate_bench 项目根
CASES = ROOT / "data" / "cases"


def extract_inputs_section(case_dir: Path, side: str) -> str:
    card = case_dir / f"card_{side}.txt"
    lines, inside = [], False
    for line in (card.read_text(encoding="utf-8") if card.is_file() else "").splitlines():
        if line.startswith("──"):
            inside = "Inputs to the video model" in line
            continue
        if inside:
            lines.append(line.rstrip())
    return "\n".join(lines).strip()


def gold_end_phrases(case_json: dict) -> list:
    vocab = case_json.get("visual_vocab") or {}
    phrases = []
    for path, value in (case_json.get("gold_end_state") or {}).items():
        options = vocab.get(path) or {}
        key = value if isinstance(value, str) else ("true" if value else "false")
        phrase = options.get(key) or options.get(str(key).strip().lower())
        if phrase:
            phrases.append(str(phrase))
    return phrases


def build_raw_state_prompt(case_dir: Path, side: str) -> str:
    raw_text = extract_inputs_section(case_dir, side)
    case_json = json.loads((case_dir / f"case_{side}.json").read_text(encoding="utf-8"))
    phrases = "; ".join(gold_end_phrases(case_json))
    if not phrases:
        raise ValueError(f"{case_dir.name}-{side}: no gold end phrases from visual_vocab")
    return (f"{raw_text}\n\n"
            f"The clip must end as: {phrases}. "
            f"Hold this final state for about the last second.")


def iter_clips():
    """按 case 顺序 yield (case, side, case_dir); raw 与 raw_state 同 clip 相邻在 runner 内处理。"""
    for case_dir in sorted(p for p in CASES.iterdir() if p.is_dir()):
        for side in ("A", "B"):
            yield case_dir.name, side, case_dir


def build_manifest(model: str) -> Path:
    """manifest: 每行 {case, side, mode, prompt, image, out}; raw 在前 raw_state 紧随。"""
    outdir_root = ROOT / "runs" / f"{model}"
    man = ROOT / "baseline" / "manifests" / f"{model}.jsonl"
    man.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(man, "w", encoding="utf-8") as f:
        for case, side, case_dir in iter_clips():
            raw = extract_inputs_section(case_dir, side)
            for mode, prompt in (("raw", raw), ("raw_state", build_raw_state_prompt(case_dir, side))):
                out = outdir_root / mode / f"{case}-{side}.mp4"
                pfile = outdir_root / "prompts" / f"{case}-{side}.{mode}.txt"
                pfile.parent.mkdir(parents=True, exist_ok=True)
                pfile.write_text(prompt, encoding="utf-8")
                f.write(json.dumps({"case": case, "side": side, "mode": mode,
                                    "prompt": prompt,
                                    "image": str(case_dir / "input.png"),
                                    "out": str(out)}) + "\n")
                n += 1
    print(f"[manifest] {model}: {n} entries -> {man}")
    return man


def load_manifest(model: str) -> list:
    man = ROOT / "baseline" / "manifests" / f"{model}.jsonl"
    if not man.is_file():
        raise SystemExit(
            f"manifest 不存在: {man}\n"
            f"(数据集缺失: 确认 data/cases/ 存在, 或先运行 python3 generation/run.py package, "
            f"再运行 python3 baseline/prompts_common.py {model})")
    return [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]


def done(entry: dict) -> bool:
    out = Path(entry["out"])
    return out.is_file() and out.stat().st_size > 50_000


if __name__ == "__main__":
    for m in sys.argv[1:]:
        build_manifest(m)
