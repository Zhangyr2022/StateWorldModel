"""casebench 单元测试:装载/信封/判定解析/指标(fake 全链路干跑)。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.casebench.dataset import Case, TwinSpec, load_cases
from evaluation.casebench.framework import resolve_rule, run_framework
from evaluation.casebench.headless import run_headless
from evaluation.casebench.judge import HeuristicJudge, build_user_prompt, parse_verdict
from evaluation.casebench.metrics import aggregate, judge_row
from evaluation.casebench.render import (apply_ref2va_anchor, build_prompt,
                                         build_raw_state_prompt, render_twin)


def _png_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (8, 8), (200, 200, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


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
        self_phrase, gold_end, delta = phrase[value], {"orb.flame": value.lower()}, \
            {"orb.flame": value.lower()}
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
    (case_dir / "input.png").write_bytes(_png_bytes())
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
            (root / "B02" / "input.png").write_bytes(_png_bytes())
            (root / "B02" / "case_A.json").write_text(json.dumps(spec_a), encoding="utf-8")
            (root / "B02" / "case_B.json").write_text(json.dumps(spec_b), encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_cases(root)


class RenderPromptTest(unittest.TestCase):
    def test_instruction_wrapped_verbatim(self) -> None:
        prompt = build_prompt("element_mode = RED. Show the orb burning red.")
        lines = prompt.splitlines()
        self.assertTrue(lines[0].startswith("For the target video"))
        self.assertIn("integrated_multimodal_description: [Shot 1] "
                      "element_mode = RED. Show the orb burning red.", prompt)
        self.assertIn("overall_soundscape: N/A", prompt)
        self.assertIn("non_diegetic_music: N/A", prompt)

    def test_blank_instruction_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt("   ")

    def test_render_instruction_strips_must_not_for_all_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)  # 同物体(B01)与不同物体(B02)都摘除 must-NOT 句
            _write_case(root, "B02", counter_same=False)
            for case in load_cases(root):
                for twin in case.twins:
                    self.assertNotIn("must NOT happen at any point",
                                     twin.render_instruction)
                    self.assertIn("Instead,", twin.render_instruction)
                    self.assertIn("must NOT happen at any point", twin.instruction)

    def test_render_instruction_drops_must_not_for_different_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, "B02", counter_same=False)  # 反物体是 brazier/orb
            for twin in load_cases(root)[0].twins:
                self.assertFalse(twin.counter_same_object)
                prefix = "At any point in the clip, does "
                counter_phrase = twin.counter_question["question"][len(prefix):-len(" happen?")]
                self.assertNotIn("must NOT happen at any point",
                                 twin.render_instruction)
                self.assertNotIn(counter_phrase, twin.render_instruction)
                self.assertIn(counter_phrase, twin.instruction)
                self.assertIn("Instead,", twin.render_instruction)

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

    def test_render_twin_rejects_unknown_prompt_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            twin = load_cases(root)[0].twins[0]
            with self.assertRaises(ValueError):
                render_twin(None, twin, root / "run", prompt_mode="bogus")


class Ref2vaPromptTest(unittest.TestCase):
    """ref2va 首帧模式(--ref2va,不开白模):对齐行语义切换。"""

    def test_envelope_default_is_i2v(self) -> None:
        prompt = build_prompt("element_mode = RED. Show the orb burning red.")
        self.assertTrue(prompt.startswith("For the target video"))

    def test_anchor_swaps_alignment_line_keeps_body(self) -> None:
        prompt = apply_ref2va_anchor(
            build_prompt("element_mode = RED. Show the orb burning red."))
        lines = prompt.splitlines()
        self.assertTrue(lines[0].startswith("<Picture 1> is the exact first frame"))
        self.assertIn("begin the video on <Picture 1>", lines[0])
        self.assertNotIn("For the target video", prompt)
        # 描述主体与声场行逐字节不动
        self.assertIn("integrated_multimodal_description: [Shot 1] "
                      "element_mode = RED. Show the orb burning red.", prompt)
        self.assertIn("overall_soundscape: N/A", prompt)
        self.assertIn("non_diegetic_music: N/A", prompt)

    def test_anchor_prepends_to_raw_without_alignment_line(self) -> None:
        prompt = apply_ref2va_anchor("Inputs to the video model: raw card text.")
        self.assertTrue(prompt.startswith("<Picture 1> is the exact first frame"))
        self.assertIn("\n\nInputs to the video model: raw card text.", prompt)

    def test_render_twin_ref2va_prompt_and_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            twin = load_cases(root)[0].twins[0]
            run_dir = root / "run"
            from engine.renderer.fake import FakeRenderer

            render_twin(FakeRenderer(run_dir), twin, run_dir,
                        prompt_mode="envelope", ref2va=True)
            prompt = (run_dir / "prompts" / f"{twin.clip_id}.txt").read_text()
            self.assertTrue(
                prompt.startswith("<Picture 1> is the exact first frame"))
            self.assertNotIn("For the target video", prompt)

    def test_framework_ref2va_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            run_dir = root / "run"
            from engine.renderer.fake import FakeRenderer

            renderer = FakeRenderer(run_dir)
            for twin in load_cases(root)[0].twins:
                video, fresh = run_framework(renderer, twin, run_dir, seed=0,
                                             ref2va=True)
                self.assertTrue(fresh and video.is_file())
                prompt = (run_dir / "prompts" / f"{twin.clip_id}.txt").read_text()
                self.assertTrue(
                    prompt.startswith("<Picture 1> is the exact first frame"))
                self.assertNotIn("For the target video", prompt)
                # NEWSKILL 描述主体(场景锚/动作/结局)不受后端切换影响
                self.assertIn("The scene is exactly as shown in <Picture 1>",
                              prompt)
                self.assertIn("the orb erupts with large bright "
                              f"{twin.hidden_value.lower()} flames", prompt)


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

    def test_counter_question_derived_from_data(self) -> None:
        self.assertEqual(self.twin.counter_question["expected"], "no")
        self.assertIn("blue flames", self.twin.counter_question["question"])
        self.assertIn("red flames", self.other_twin.counter_question["question"])

    def test_baseline_text_is_card_section(self) -> None:
        self.assertIn("hidden state: element_mode = RED", self.twin.baseline_text)
        self.assertIn("instruction: element_mode = RED.", self.twin.baseline_text)
        self.assertIn("chunk: one action-consequence transition, 2-5 s",
                      self.twin.baseline_text)
        self.assertNotIn("Gold", self.twin.baseline_text)
        self.assertNotIn("[zh]", self.twin.baseline_text)


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
            self.assertEqual(summary["EndingFollow"], 0.5)
            self.assertEqual(summary["CounterAbsent"], 0.5)
            self.assertEqual(summary["StrictFollow"], 0.5)
            self.assertEqual(summary["PairFollow"], 0.0)
            self.assertEqual(summary["PairStrict"], 0.0)
            self.assertEqual(summary["UnclearOutcome"], 0.5)
            self.assertNotIn("StartKeep", summary)
            self.assertNotIn("ActionShown", summary)
            self.assertEqual(summary["by_side"]["A"]["StrictFollow"], 1.0)
            self.assertEqual(summary["by_side"]["B"]["StrictFollow"], 0.0)
            self.assertEqual(summary["by_family"]["trigger_gesture"]["PairStrict"], 0.0)


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


class FrameworkTest(unittest.TestCase):
    def test_resolve_rule_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)  # 同物体:A=RED 走 effect,B=BLUE 走 else
            twin_a, twin_b = load_cases(root)[0].twins
            end_a, exec_a = resolve_rule(twin_a)
            end_b, exec_b = resolve_rule(twin_b)
            self.assertTrue(exec_a)
            self.assertEqual(end_a["orb"]["flame"], "red")
            self.assertFalse(exec_b)
            self.assertEqual(end_b["orb"]["flame"], "blue")

    def test_framework_end_to_end_with_fake_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root)
            run_dir = root / "run"
            from engine.renderer.fake import FakeRenderer

            renderer = FakeRenderer(run_dir)
            twins = load_cases(root)[0].twins
            for twin in twins:
                video, fresh = run_framework(renderer, twin, run_dir, seed=0)
                self.assertTrue(fresh and video.is_file())
                prompt = (run_dir / "prompts" / f"{twin.clip_id}.txt").read_text()
                self.assertIn("For the target video", prompt)
                self.assertIn("the orb erupts with large bright "
                              f"{twin.hidden_value.lower()} flames", prompt)
            # A/B 各自归位,互不覆盖
            for twin in twins:
                self.assertTrue((run_dir / "videos" / f"{twin.clip_id}.mp4")
                                .is_file())


if __name__ == "__main__":
    unittest.main()
