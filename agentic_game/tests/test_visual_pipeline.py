"""Visual Pipeline 测试:VisualScript → PromptCompiler → H3 prompt。

用内联 VisualScript 直接测引擎编译器(engine/visual/compiler.py),
与具体游戏解耦(战斗演出示例取自 duel_oracle_machine_epic 的短语风格)。
"""

import unittest

from engine.runtime import Action, Rng
from engine.visual.compiler import PromptCompiler
from engine.visual.script import VisualScript
from game.registry import GAMES

_PARTS = GAMES["duel_oracle_machine_epic"].build()


def _oracle_action(slot: str) -> Action:
    spec = _PARTS.runtime.skin.action(slot)
    return Action(id=slot, label=spec.label, description=spec.description)


def _base_script() -> VisualScript:
    """一份字段齐全的最小演出(机位锁定 + 稳定约束齐备)。"""
    return VisualScript(
        style="3D CG, cinematic, dark fantasy",
        actor="the black-robed oracle",
        action="raises the staff and looses a humming bolt of judgement-light",
        end_state="lowers the staff, feet planted on the cracked stone",
        effects=("glyph light flickering across the floor",),
        camera="The camera stays locked-off and static, and she never leaves her place",
        soundscape="a low machine-hum bleeds through the sanctum",
        music="Low tense strings at a moderate tempo",
        first_frame_anchor="her pose matches the previous frame exactly",
    )


class TestPromptCompiler(unittest.TestCase):
    def setUp(self):
        self.compiler = PromptCompiler()

    def prompt_for(self, script: VisualScript | None = None,
                   has_first_frame: bool = True) -> str:
        return self.compiler.compile(script or _base_script(),
                                     has_first_frame=has_first_frame)

    def test_i2va_structure_follows_guide(self):
        """H3 官方 I2VA 信封结构:对齐指令→空行→三核心字段。"""
        prompt = self.prompt_for()
        lines = prompt.splitlines()
        self.assertEqual(
            lines[0],
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.",
        )
        self.assertEqual(lines[1], "")
        self.assertTrue(lines[2].startswith("integrated_multimodal_description: [Shot 1]"))
        self.assertTrue(any(line.startswith("overall_soundscape:") for line in lines))
        self.assertTrue(any(line.startswith("non_diegetic_music:") for line in lines))

    def test_single_shot_only(self):
        prompt = self.prompt_for()
        self.assertIn("[Shot 1]", prompt)
        self.assertNotIn("[Shot 2]", prompt)
        self.assertNotIn("00:", prompt)

    def test_prompt_contains_key_information(self):
        prompt = self.prompt_for()
        self.assertIn("black-robed oracle", prompt)        # 角色标识 → 角色一致性
        self.assertIn("judgement-light", prompt)           # 主体动作
        self.assertIn("The clip ends as", prompt)          # 结尾状态显式描述
        self.assertIn("3D CG", prompt)                     # 风格
        self.assertIn("<Picture 1>", prompt)               # 首帧对齐指令(I2VA)
        self.assertIn("matches the previous frame", prompt)  # 首帧锚进入 prompt
        self.assertIn("locked-off", prompt)                # 镜头锁定进入 prompt
        self.assertIn("never leaves her place", prompt)    # 走位约束进入 prompt

    def test_t2va_drops_alignment_line(self):
        prompt = self.prompt_for(has_first_frame=False)
        self.assertNotIn("<Picture 1>", prompt)
        self.assertTrue(prompt.startswith("integrated_multimodal_description: [Shot 1]"))

    def test_prompt_is_deterministic(self):
        self.assertEqual(self.prompt_for(), self.prompt_for())

    def test_missing_required_fields_rejected(self):
        base = dict(style="s", first_frame_anchor="a", camera="c")
        with self.assertRaises(ValueError):
            self.compiler.compile(VisualScript(actor="mage", action="", end_state="ends", **base),
                                  has_first_frame=True)

    def test_machine_director_produces_compilable_script(self):
        """端到端:状态机导演产出的 VisualScript 必须能被编译器编译。"""
        runtime = _PARTS.runtime
        state = runtime.initial_state()
        outcome = runtime.resolve_turn(state, _oracle_action("arcane"), Rng(1))
        from engine.runtime import TurnResult

        result = TurnResult(
            turn_id="turn-001", index=0,
            action=_oracle_action("arcane"),
            state_before=state, state_after=outcome.state_after, outcome=outcome,
        )
        script = _PARTS.visual_director(result, _PARTS.visual_state)
        prompt = self.compiler.compile(script, has_first_frame=True)
        self.assertIn("[Shot 1]", prompt)
        self.assertTrue(prompt.strip())


if __name__ == "__main__":
    unittest.main()
