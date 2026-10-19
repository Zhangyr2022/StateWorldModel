"""evaluation 单元测试:装载 / 规则重放 / 判定解析 / 指标 / 报告(fake 全链路)。

运行:项目根目录下  python3 -m unittest worldstate_bench/tests/test_evaluation.py
(或 unittest discover -s worldstate_bench/tests)。判定后端只测 heuristic
与纯函数;LocalJudge 的模型加载路径由 review.py 的生产使用背书。
"""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.dataset import (build_raw_state_prompt,
                                                 load_cases)
from evaluation.headless import run_headless
from evaluation.judge import (HeuristicJudge, build_user_prompt,
                                               parse_verdict)
from evaluation.metrics import aggregate, judge_row
from evaluation.report import write_report, write_verdict
from evaluation.rules import resolve_rule

# 1×1 灰色 PNG(无 PIL 依赖)
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _case_spec(case_id: str, side: str, counter_same: bool = True) -> dict:
    """构造一份自洽的案例规格;counter_same=False 时 A 变球、B 变火盆。"""
    other = "B" if side == "A" else "A"
    value, other_value = ("RED", "BLUE") if side == "A" else ("BLUE", "RED")
    phrase = {"RED": "the orb erupts with large bright red flames",
              "BLUE": "the orb erupts with large bright blue flames"}
    brazier_phrase = "the brazier burns with tall flames"
    brazier_label = "the brazier burns"
    if counter_same:
        occurs = {"variable": "orb.flame", "value": value.lower(),
                  "phrase": phrase[value], "judge_label": f"{value.lower()} flames"}
        counter = {"variable": "orb.flame", "value": other_value.lower(),
                   "phrase": phrase[other_value],
                   "judge_label": f"{other_value.lower()} flames",
                   "instead_remains": phrase[value]}
        self_phrase = phrase[value]
        gold_end = {"orb.flame": value.lower()}
        delta = {"orb.flame": value.lower()}
        outcome_answers = ["red flames", "blue flames", "unclear"]
        expected = f"{value.lower()} flames"
        initial_state = {"orb": {"flame": "none"}, "element_mode": value}
        rule = {"text": "If element_mode is RED, the orb burns red; if BLUE, blue.",
                "precondition": "element_mode == 'RED'",
                "effect": "orb.flame = 'red'", "else_effect": "orb.flame = 'blue'"}
        visual_vocab = {"orb.flame": {
            "none": "the large white orb is completely dark",
            "red": phrase["RED"], "blue": phrase["BLUE"]}}
    elif side == "A":
        occurs = {"variable": "orb.flame", "value": "red", "phrase": phrase["RED"],
                  "judge_label": "red flames"}
        counter = {"variable": "brazier.lit", "value": True,
                   "phrase": brazier_phrase, "judge_label": brazier_label,
                   "instead_remains": "the brazier holds only cold grey ash"}
        self_phrase, expected = phrase["RED"], "red flames"
        gold_end = {"orb.flame": "red", "brazier.lit": False}
        delta = {"orb.flame": "red"}
        outcome_answers = ["red flames", brazier_label, "unclear"]
        initial_state = {"orb": {"flame": "none"}, "brazier": {"lit": False},
                         "element_mode": value}
        rule = {"text": "If element_mode is RED the orb burns; else the brazier.",
                "precondition": "element_mode == 'RED'",
                "effect": "orb.flame = 'red'", "else_effect": "brazier.lit = true"}
        visual_vocab = {"orb.flame": {
            "none": "the large white orb is completely dark", "red": phrase["RED"]},
            "brazier.lit": {"true": brazier_phrase,
                            "false": "the brazier holds only cold grey ash"}}
    else:
        occurs = {"variable": "brazier.lit", "value": True,
                  "phrase": brazier_phrase, "judge_label": brazier_label}
        counter = {"variable": "orb.flame", "value": "red", "phrase": phrase["RED"],
                   "judge_label": "red flames",
                   "instead_remains": "the orb keeps its matte white glow"}
        self_phrase, expected = brazier_phrase, brazier_label
        gold_end = {"orb.flame": "none", "brazier.lit": True}
        delta = {"brazier.lit": True}
        outcome_answers = ["red flames", brazier_label, "unclear"]
        initial_state = {"orb": {"flame": "none"}, "brazier": {"lit": False},
                         "element_mode": value}
        rule = {"text": "If element_mode is RED the orb burns; else the brazier.",
                "precondition": "element_mode == 'RED'",
                "effect": "orb.flame = 'red'", "else_effect": "brazier.lit = true"}
        visual_vocab = {"orb.flame": {
            "none": "the large white orb is completely dark", "red": phrase["RED"]},
            "brazier.lit": {"true": brazier_phrase,
                            "false": "the brazier holds only cold grey ash"}}
    other_phrase = counter["phrase"]
    instruction = (
        f"Preserve the first frame. element_mode = {value}, so the resolved rule "
        f"is: {self_phrase}. Action: The wizard raises the wand. "
        f"Show the action, then exactly this consequence: {self_phrase} "
        f"The counter-outcome must NOT happen at any point: {other_phrase}. "
        f"Instead, {counter['instead_remains']} throughout the clip."
    )
    return {
        "id": f"{case_id}-{side}",
        "counterpart": f"{case_id}-{other}",
        "genre": "magic_academy",
        "interaction_family": "trigger_gesture",
        "hidden_state": {"variable": "element_mode", side: value,
                         other: other_value, "visible_in_initial_frame": False},
        "rule": rule,
        "initial_state": initial_state,
        "visual_vocab": visual_vocab,
        "shared_action": "The wizard raises the wand.",
        "instruction": instruction,
        "gold_end_state": gold_end,
        "delta": delta,
        "gold_A": phrase["RED"],
        "gold_B": phrase["BLUE"] if counter_same else brazier_phrase,
        "terminal_visual_states": {"occurs": occurs, "not_occurs": counter},
        "judge": {
            "initial_questions": [
                {"question": "Is a wizard present?", "answers": ["yes", "no"],
                 "expected": "yes"},
            ],
            "action_question": {"question": "Did the wizard raise the wand?",
                                "answers": ["yes", "no"], "expected": "yes"},
            "outcome_question": {"question": "What happens at the end?",
                                 "answers": outcome_answers, "expected": expected},
        },
        "input_image": "input.png",
    }


_CARD_SECTION = """── Inputs to the video model ──
hidden state: element_mode = RED (runtime only, not visible in the frame)
rule: If element_mode is RED, the orb burns red; if BLUE, blue.
shared action: The wizard raises the wand.
instruction: element_mode = RED. Show the orb burning red.
chunk: one action-consequence transition, 2-5 s
"""


def _write_case(root: Path, case_id: str = "B01", counter_same: bool = True) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "input.png").write_bytes(_PNG_BYTES)
    for side in ("A", "B"):
        (case_dir / f"case_{side}.json").write_text(
            json.dumps(_case_spec(case_id, side, counter_same=counter_same)),
            encoding="utf-8")
        (case_dir / f"card_{side}.txt").write_text(_CARD_SECTION, encoding="utf-8")


class DatasetTest(unittest.TestCase):
    def test_load_and_pair_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            cases = load_cases(root)
            self.assertEqual([case.case_id for case in cases], ["B01"])
            first, second = cases[0].twins
            self.assertEqual((first.side, second.side), ("A", "B"))
            self.assertEqual(first.counterpart, second.clip_id)
            self.assertEqual(first.hidden_value, "RED")
            self.assertEqual(second.hidden_value, "BLUE")
            self.assertNotEqual(first.instruction, second.instruction)

    def test_rejects_shared_hidden_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            spec_a = _case_spec("B02", "A")
            spec_b = _case_spec("B02", "B")
            spec_a["hidden_state"]["B"] = "RED"   # 两侧同值 → 非法孪生
            spec_b["hidden_state"]["B"] = "RED"
            (root / "B02").mkdir()
            (root / "B02" / "input.png").write_bytes(_PNG_BYTES)
            (root / "B02" / "case_A.json").write_text(json.dumps(spec_a), encoding="utf-8")
            (root / "B02" / "case_B.json").write_text(json.dumps(spec_b), encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_cases(root)

    def test_counter_question_derived_from_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            twin_a, twin_b = load_cases(root)[0].twins
            self.assertEqual(twin_a.counter_question["expected"], "no")
            self.assertIn("blue flames", twin_a.counter_question["question"])
            self.assertIn("red flames", twin_b.counter_question["question"])

    def test_baseline_text_is_card_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            twin = load_cases(root)[0].twins[0]
            self.assertIn("hidden state: element_mode = RED", twin.baseline_text)
            self.assertIn("chunk: one action-consequence transition, 2-5 s",
                          twin.baseline_text)
            self.assertNotIn("Gold", twin.baseline_text)

    def test_raw_state_prompt_appends_gold_end_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            twin = load_cases(root)[0].twins[0]
            self.assertEqual(twin.gold_end_phrases,
                             ["the orb erupts with large bright red flames"])
            prompt = build_raw_state_prompt(twin)
            self.assertIn(twin.baseline_text, prompt)
            self.assertIn("The clip must end as: the orb erupts with large "
                          "bright red flames.", prompt)


class RulesTest(unittest.TestCase):
    """规则重放必须与生成侧 DSL 一致(A 走 effect,B 走 else)。"""

    def test_resolve_rule_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            twin_a, twin_b = load_cases(root)[0].twins
            end_a, exec_a = resolve_rule(twin_a)
            end_b, exec_b = resolve_rule(twin_b)
            self.assertTrue(exec_a)
            self.assertEqual(end_a["orb"]["flame"], "red")
            self.assertFalse(exec_b)
            self.assertEqual(end_b["orb"]["flame"], "blue")

    def test_resolve_rule_does_not_mutate_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            twin = load_cases(root)[0].twins[0]
            before = json.dumps(twin.raw["initial_state"], sort_keys=True)
            resolve_rule(twin)
            self.assertEqual(json.dumps(twin.raw["initial_state"], sort_keys=True),
                             before)


class JudgeParseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        _write_case(Path(tmp.name))
        twins = load_cases(Path(tmp.name))[0].twins
        cls.twin = twins[0]
        cls.other_twin = twins[1]

    def test_parse_and_canonicalise(self) -> None:
        raw = '{"initial": ["Yes"], "action": "yes", "outcome": "Red Flames", ' \
              '"counter": "NO", "evidence": "orb is red."}'
        verdict = parse_verdict(f"noise {raw} noise", self.twin)
        self.assertEqual(verdict["initial"], ["yes"])
        self.assertEqual(verdict["outcome"], "red flames")
        self.assertEqual(verdict["action"], "yes")
        self.assertEqual(verdict["counter"], "no")

    def test_out_of_vocab_answer_becomes_none(self) -> None:
        verdict = parse_verdict('{"initial": ["maybe"], "action": "yes", '
                                '"outcome": "green flames", "counter": "never"}',
                                self.twin)
        self.assertIsNone(verdict["initial"][0])
        self.assertIsNone(verdict["outcome"])
        self.assertIsNone(verdict["counter"])

    def test_missing_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_verdict("the orb is red", self.twin)

    def test_user_prompt_carries_questions_and_shape(self) -> None:
        prompt = build_user_prompt(self.twin, 16)
        self.assertIn("Is a wizard present?", prompt)
        self.assertIn("What happens at the end?", prompt)
        self.assertIn("Counter-outcome question", prompt)
        self.assertIn("does the orb erupts with large bright blue flames happen?",
                      prompt)
        self.assertIn('"counter": "<answer>"', prompt)
        self.assertIn("exactly 1 answers", prompt)
        self.assertIn("Picture 1", prompt)

    def test_heuristic_follows_expectations(self) -> None:
        verdict = HeuristicJudge().judge(self.twin, [Path("f.png")])
        self.assertEqual(verdict["outcome"], self.twin.outcome_question["expected"])
        self.assertEqual(verdict["counter"], "no")
        with self.assertRaises(ValueError):
            HeuristicJudge().judge(self.twin, [])


class MetricsTest(unittest.TestCase):
    def test_row_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            cases = load_cases(root)
            rows = []
            for index, twin in enumerate(cases[0].twins):
                verdict = HeuristicJudge().judge(twin, [Path("f.png")])
                rows.append(judge_row(twin, verdict, judge_name="heuristic",
                                      n_images=1, video_path="v.mp4"))
                if index == 1:  # 侧 B 故意判成 unclear(counter 同步失守)
                    rows[-1]["outcome_ok"] = False
                    rows[-1]["outcome_answer"] = "unclear"
                    rows[-1]["outcome_unclear"] = True
                    rows[-1]["counter_ok"] = False
                    rows[-1]["counter_answer"] = "yes"
                    rows[-1]["strict_ok"] = False
            summary = aggregate(rows, cases)
            self.assertEqual(summary["clips"], 2)
            self.assertEqual(summary["ENDFOLLOW"], 0.5)
            self.assertEqual(summary["CNTABSENT"], 0.5)
            self.assertEqual(summary["STRICT"], 0.5)
            self.assertEqual(summary["PAIRFOLLOW"], 0.0)
            self.assertEqual(summary["PAIRSTRICT"], 0.0)
            self.assertEqual(summary["UNCLEAR"], 0.5)
            self.assertEqual(summary["by_side"]["A"]["STRICT"], 1.0)
            self.assertEqual(summary["by_side"]["B"]["STRICT"], 0.0)
            self.assertEqual(summary["by_family"]["trigger_gesture"]["PAIRSTRICT"], 0.0)


class HeadlessTest(unittest.TestCase):
    def test_real_fixture_is_self_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            _write_case(root, "B02", counter_same=False)
            passed, checked = run_headless(load_cases(root))
            self.assertEqual(passed, 4, checked)

    def test_flags_inconsistent_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            spec = json.loads((root / "B01" / "case_B.json").read_text())
            spec["judge"]["outcome_question"]["expected"] = "red flames"  # 应为 blue
            (root / "B01" / "case_B.json").write_text(json.dumps(spec))
            passed, checked = run_headless(load_cases(root))
            self.assertEqual(passed, 1)
            self.assertTrue(checked["B01-B"])


class ReportTest(unittest.TestCase):
    def test_write_report_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            cases = load_cases(root)
            run_dir = root / "run"
            for twin in cases[0].twins:
                verdict = HeuristicJudge().judge(twin, [Path("f.png")])
                write_verdict(run_dir, twin, verdict, judge_name="heuristic",
                              n_images=1, images=[Path("f.png")],
                              video_path=Path("v.mp4"))
            metrics_path, md_path = write_report(run_dir, cases, meta={
                "run_id": "test", "judge": "heuristic", "prompt_mode": "raw"})
            self.assertEqual(metrics_path.name, "metrics.json")
            self.assertEqual(md_path.name, "report.md")
            self.assertTrue((run_dir / "FAILURES.md").is_file())
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["clips"], 2)
            self.assertEqual(data["summary"]["STRICT"], 1.0)


if __name__ == "__main__":
    unittest.main()
