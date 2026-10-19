#!/usr/bin/env python3
"""Build the HuggingFace dataset-viewer table (data/cases.parquet).

One row per twin (100 rows) with an embedded 640px first-frame preview per
row; the full-resolution frame stays a file, referenced by `image_path`.
Upload the result to the dataset repo root after regenerating the benchmark:

    hf upload Louis0411/StateWorldModel data/cases.parquet data/cases.parquet --repo-type dataset
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "data" / "cases"
OUT = ROOT / "data" / "cases.parquet"


def main() -> int:
    import datasets as hfds

    rows = []
    for case in sorted(CASES.iterdir()):
        if not case.is_dir():
            continue
        for side in ("A", "B"):
            c = json.loads((case / f"case_{side}.json").read_text(encoding="utf-8"))
            hs = c.get("hidden_state", {})
            rule = c.get("rule", {})
            tvs = c.get("terminal_visual_states", {})
            j = c.get("judge", {}).get("outcome_question", {})
            img = Image.open(case / "input.png").convert("RGB")
            img.thumbnail((640, 360))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            rows.append({
                "case_id": c.get("id", case.name),
                "preview": {"bytes": buf.getvalue(), "path": None},
                "counterpart": c.get("counterpart", ""),
                "image_path": f"data/cases/{case.name}/input.png",
                "theme": c.get("genre", ""),
                "interaction_family": c.get("interaction_family", ""),
                "initial_scene": c.get("initial_scene", ""),
                "hidden_variable": hs.get("variable", ""),
                "hidden_value": hs.get(side, ""),
                "hidden_value_other": hs.get("B" if side == "A" else "A", ""),
                "rule_text": rule.get("text", ""),
                "shared_action": str(c.get("shared_action", ""))[:400],
                "instruction": c.get("instruction", ""),
                "ending_occurs": tvs.get("occurs", {}).get("phrase", ""),
                "ending_not_occurs": tvs.get("not_occurs", {}).get("phrase", ""),
                "outcome_question": j.get("question", ""),
                "outcome_expected": str(j.get("expected", "")),
                "counter_question": c.get("judge", {}).get("counter_question", {}).get("question", ""),
                "gold_end_state": json.dumps(c.get("gold_end_state", {}), ensure_ascii=False),
            })

    ds = hfds.Dataset.from_list(rows)
    # 列顺序 = 展示顺序:preview 放第二列,方便 viewer 直接看到首帧
    feats = {k: v for k, v in ds.features.items() if k != "preview"}
    cast = hfds.Features(**{"case_id": feats["case_id"],
                            "preview": hfds.Image(),
                            **{k: v for k, v in feats.items() if k != "case_id"}})
    ds = ds.cast(cast)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(OUT)
    print(f"{OUT}: {len(ds)} rows, {OUT.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
