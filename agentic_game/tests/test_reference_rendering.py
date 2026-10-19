"""Reference-Guided Realization(控制视频/ref2va)测试。

覆盖(2026-09-24 融合定稿):reference overlay 叠加在标准 NEWSKILL 输出上、
无视频时 prompt 与基线逐字节一致(消融切换)、RenderRequest.reference_video
校验、GameLoop 的 reference_producer 挂点(末帧链接下按当前帧生产、
失败降级/中止)。全部用假渲染器与假控制视频文件,不触真实模型(§25)。
"""

import tempfile
import unittest
from pathlib import Path

from engine.agents import ScriptedAgent
from engine.loop import GameLoop
from engine.renderer.base import (
    InvalidRenderRequestError,
    RenderFailedError,
    RenderRequest,
    RenderResult,
    Renderer,
)
from engine.visual.compiler import PromptCompiler
from engine.visual.script import MotionReference, VisualScript
from engine.visual.state import VisualState
from game.wbench.case_runtime import (
    ScriptedAgent as WBScriptedAgent,
    WBenchRuntime,
    interaction_perspectives,
)
from game.wbench.director import WBenchVisualDirector, official_dictionary_available

# reference 渲染测试以 WBench case 为 fixture;官方动作词典来自外部第三方仓
# (见 game/wbench/__init__.py),仓不在时跳过这两组,而不是报错。
_WBENCH_PRESENT = official_dictionary_available()
_SKIP_WBENCH = unittest.skipUnless(
    _WBENCH_PRESENT,
    "requires the external WBench repository (see game/wbench/__init__.py)")


def _nav_script(perspective: str = "third_person",
                with_reference: bool = True) -> VisualScript:
    """标准映射产出的 navigation script 形态(字段名与 director 输出一致)。"""
    reference = None
    if perspective == "third_person":
        if with_reference:
            reference = MotionReference(
                perspective="third_person", action_token="W",
                previz_motion="the camera tracks alongside the box as it moves forward",
                subject_motion=("moves forward exactly as the blue box does, matching "
                                "its position, direction, size, timing and trajectory"),
                official_action=("The subject moves forward. The camera follows, "
                                 "maintaining distance and framing."))
        return VisualScript(
            first_frame_anchor="A dirt road across an open plain.",
            camera="The camera follows, maintaining distance and framing.",
            actor="The subject",
            actor_appearance="a chestnut brown horse with a dark mane",
            action="moves forward",
            stability=("the subject keeps its appearance unchanged",
                       "the environment layout stays consistent during the move"),
            end_state="the camera completes the intended move",
            reference=reference,
            meta={"itype": "navigation"},
        )
    if with_reference:
        reference = MotionReference(
            perspective="first_person", action_token="W",
            previz_motion="The camera pushes forward.")
    return VisualScript(
        camera="",
        actor="The camera",
        actor_appearance="mounted at eye height",
        action="pushes forward",
        stability=("the subject keeps its appearance unchanged",),
        end_state="the camera completes the intended move",
        reference=reference,
        meta={"itype": "navigation"},
    )


class CompileOverlayTest(unittest.TestCase):
    def test_control_prompt_is_newskill_plus_overlay(self):
        script = _nav_script()
        prompt = PromptCompiler().compile(
            script, has_first_frame=True, has_reference_video=True)
        # overlay:首帧锚定 + <Video 1> 运动参考 + 跟随句 + 反残留
        self.assertTrue(prompt.startswith(
            "<Picture 1> is the exact first frame of the target video"))
        self.assertIn("<Video 1> is the motion and camera reference", prompt)
        self.assertIn("a minimal previz containing only a blue box", prompt)
        self.assertIn("the camera tracks alongside the box as it moves forward", prompt)
        self.assertIn("<Subject 1> is a chestnut brown horse with a dark mane", prompt)
        self.assertIn("exactly as the blue box does", prompt)
        self.assertIn("never show the two red reference boxes", prompt)
        # 官方动作句全文插在外观句之后(主体句内)
        self.assertIn("in <Picture 1>. The subject moves forward. "
                      "The camera follows, maintaining distance and framing. "
                      "<Subject 1> moves forward exactly as the blue box does",
                      prompt)
        # 官方句已含镜头句 → 不再单独渲染 camera 段(全文只出现一次)
        self.assertEqual(
            prompt.count("The camera follows, maintaining distance and framing."), 1)
        # NEWSKILL 骨架不变:单 [Shot 1] 描述 + 声画字段 + 稳定性/收尾
        self.assertIn("integrated_multimodal_description: [Shot 1]", prompt)
        self.assertIn("A dirt road across an open plain.", prompt)
        self.assertIn("Throughout the clip, the subject keeps its appearance", prompt)
        self.assertIn("The clip ends as the camera completes the intended move.", prompt)
        self.assertIn("overall_soundscape: N/A", prompt)
        self.assertIn("non_diegetic_music: N/A", prompt)
        self.assertNotIn("..", prompt)  # 句号拼接无重复

    def test_fp_overlay_has_no_subject(self):
        script = _nav_script(perspective="first_person")
        prompt = PromptCompiler().compile(
            script, has_first_frame=True, has_reference_video=True)
        self.assertNotIn("<Subject 1>", prompt)
        self.assertIn("a minimal first-person previz", prompt)
        self.assertIn("The camera pushes forward.", prompt)
        # 主体句:官方动作句 + 跟随语义拼成一句
        self.assertIn("The camera pushes forward, follows the movement, "
                      "framing and timing of <Video 1> exactly, "
                      "in continuous first-person view.", prompt)
        self.assertIn("never show the two red reference boxes", prompt)
        self.assertIn("let no previz geometry remain", prompt)

    def test_no_video_is_byte_identical_to_baseline(self):
        """消融切换:has_reference_video=False 时,prompt 与不声明 reference
        的 script(基线)逐字节一致——script.reference 残留不改变输出。"""
        with_ref = _nav_script(with_reference=False)
        with_ref.reference = MotionReference(  # 声明存在但未生产成(降级)
            perspective="third_person", action_token="W",
            previz_motion="the camera tracks alongside the box as it moves forward",
            subject_motion="moves forward exactly as the blue box does")
        without_ref = _nav_script(with_reference=False)
        compiler = PromptCompiler()
        a = compiler.compile(with_ref, has_first_frame=True, has_reference_video=False)
        b = compiler.compile(without_ref, has_first_frame=True, has_reference_video=False)
        self.assertEqual(a, b)
        self.assertNotIn("<Video 1>", a)
        self.assertIn("integrated_multimodal_description: [Shot 1]", a)
        self.assertIn("For the target video, at 0.00 seconds", a)  # 默认 I2VA 对齐行

    def test_video_requires_motion_reference(self):
        with self.assertRaises(ValueError):
            PromptCompiler().compile(
                _nav_script(with_reference=False),
                has_first_frame=True, has_reference_video=True)

    def test_tp_overlay_requires_desc_and_follow(self):
        script = _nav_script()
        script.reference = MotionReference(
            perspective="third_person", action_token="W",
            previz_motion="the camera tracks alongside the box")
        with self.assertRaises(ValueError):
            PromptCompiler().compile(
                script, has_first_frame=True, has_reference_video=True)


class RenderRequestReferenceTest(unittest.TestCase):
    def test_reference_video_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "walk_W.mp4"
            video.write_bytes(b"x")
            RenderRequest(turn_id="t", prompt="p", seed=1,
                          reference_video=video).validate()  # 存在即合法
            with self.assertRaises(InvalidRenderRequestError):
                RenderRequest(turn_id="t", prompt="p", seed=1,
                              reference_video=Path(tmp) / "missing.mp4").validate()


class _StubRenderer(Renderer):
    """记录请求;每轮返回递增的末帧(验证末帧链接)。"""

    def __init__(self, frame0: Path) -> None:
        self.frame0 = frame0
        self.finals: list[Path] = []
        self.requests: list[RenderRequest] = []

    def render(self, request: RenderRequest) -> RenderResult:
        self.requests.append(request)
        final = request.first_frame if request.first_frame else self.frame0
        # 简化:末帧路径每轮唯一(文件名带轮次),驱动链接语义
        final = final.with_name(f"final_{len(self.requests)}.png")
        final.write_bytes(b"x")
        self.finals.append(final)
        return RenderResult(turn_id=request.turn_id, video_path=self.frame0,
                            final_frame=final, seed=request.seed)


@_SKIP_WBENCH
class ReferenceProducerHookTest(unittest.TestCase):
    """reference_producer 挂点:prompt 切换、记录、末帧链接、失败降级。"""

    def _run(self, producer, policy: str = "skip"):
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "initial.png"
            frame.write_bytes(b"x")
            video = Path(tmp) / "walk_W.mp4"
            video.write_bytes(b"x")
            case = {"id": "t", "interactions": [
                {"turn": 1, "type": "navigation", "action": "W"},
                {"turn": 2, "type": "navigation", "action": "W"},
            ]}
            renderer = _StubRenderer(frame)
            loop = GameLoop(
                runtime=WBenchRuntime(case),
                agent=WBScriptedAgent(),
                visual_director=WBenchVisualDirector(ref_mode="control"),
                renderer=renderer,
                output_dir=Path(tmp),
                max_turns=4,
                render_failure_policy=policy,
                visual_state=VisualState(current_frame=frame),
                reference_producer=producer(video),
            )
            summary = loop.run()
            return summary, renderer, frame, video

    def test_reference_flows_and_chain_anchors_advance(self):
        """请求携带控制视频;producer 收到的锚定帧逐轮推进(末帧链接)。"""
        anchors: list[Path | None] = []

        def producer(video):
            def make(script, visual_state):
                anchors.append(visual_state.current_frame)
                return video
            return make

        summary, renderer, frame, video = self._run(producer)
        self.assertEqual(summary.turns_played, 2)
        self.assertEqual([r.reference_video for r in renderer.requests],
                         [video, video])
        self.assertEqual([r.first_frame for r in renderer.requests],
                         [frame, renderer.finals[0]])  # 链接:第二轮锚定第一轮末帧
        self.assertEqual(anchors[0], frame)
        self.assertEqual(anchors[1], renderer.finals[0])
        for record in summary.records:
            self.assertEqual(record.reference_video, str(video))
            self.assertIn("<Video 1>", record.prompt)

    def test_producer_failure_degrades_prompt_under_skip_policy(self):
        def producer(video):
            def make(script, visual_state):
                raise RenderFailedError("blender exploded")
            return make

        summary, renderer, _, _ = self._run(producer, policy="skip")
        self.assertEqual(summary.turns_played, 2)  # 游戏状态不受影响(§16)
        for request in renderer.requests:
            self.assertIsNone(request.reference_video)
        for record in summary.records:
            self.assertNotIn("<Video 1>", record.prompt)
            self.assertNotIn("previz", record.prompt)
            # 降级 prompt = 标准 NEWSKILL(与基线同族)
            self.assertIn("integrated_multimodal_description: [Shot 1]", record.prompt)

    def test_producer_failure_aborts_game(self):
        def producer(video):
            def make(script, visual_state):
                raise RenderFailedError("blender exploded")
            return make

        with self.assertRaises(RenderFailedError):
            self._run(producer, policy="abort")


@_SKIP_WBENCH
class WBenchRefDirectorTest(unittest.TestCase):
    def _case(self, perspective: str) -> dict:
        return {
            "id": "18",
            "environment_prompt": "A dirt road.",
            "character_prompt": "A horse trots.",
            "perspective_prompt": "Third-person view.",
            "settings": {"perspective": perspective,
                         "subject": {"type": "animal", "desc": "a horse"}},
            "interactions": [{"turn": 1, "type": "navigation", "action": "W"}],
        }

    def _script(self, ref_mode: str, itype: str = "navigation",
                action: str = "W", perspective: str = "third_person"):
        from engine.runtime import Action, TurnOutcome, TurnResult

        director = WBenchVisualDirector(ref_mode=ref_mode)
        state = WBenchRuntime(self._case(perspective)).initial_state()
        result = TurnResult(
            turn_id="turn-001", index=0,
            action=Action(id="0", label=action, description=itype),
            state_before=state, state_after=state,
            outcome=TurnOutcome(events=(), state_after=state),
        )
        return director(result, VisualState(current_frame=Path("/tmp/x.png")))

    def test_control_mode_navigation_declares_reference(self):
        script = self._script("control")
        self.assertIsNotNone(script.reference)
        self.assertEqual(script.reference.action_token, "W")
        self.assertTrue(script.reference.subject_motion.startswith(
            "moves forward exactly as the blue box does"))
        # 融合:标准字段保持官方词典措辞(非 box2clay 句)
        self.assertEqual(script.action, "moves forward")
        self.assertEqual(script.camera,
                         "The camera follows, maintaining distance and framing")

    def test_first_frame_ablation_never_declares_reference(self):
        script = self._script("first_frame")
        self.assertIsNone(script.reference)

    def test_first_frame_script_equals_off_script(self):
        """消融与基线的 script 完全一致(同 prompt 同链接,只差后端);
        meta.ref_mode 仅作运行诊断,engine 不解释。"""
        def strip_meta(script):
            out = script.to_dict()
            out.pop("meta")
            return out
        self.assertEqual(strip_meta(self._script("first_frame")),
                         strip_meta(self._script("off")))

    def test_non_navigation_turn_has_no_reference(self):
        script = self._script("control", itype="subject_action",
                              action="waves its tail")
        self.assertIsNone(script.reference)

    def test_fp_navigation_declares_fp_reference(self):
        script = self._script("control", perspective="first_person")
        self.assertEqual(script.reference.perspective, "first_person")
        self.assertEqual(script.reference.previz_motion,
                         "The camera pushes forward.")
        self.assertEqual(script.reference.subject_motion, "")

    def test_interaction_perspectives_tracks_switch(self):
        case = {
            "settings": {"perspective": "first_person"},
            "interactions": [
                {"type": "navigation", "action": "W"},
                {"type": "perspective_switch", "action": "fp_to_tp"},
                {"type": "navigation", "action": "W"},
            ],
        }
        self.assertEqual(interaction_perspectives(case),
                         ["first_person", "third_person", "third_person"])

    def test_known_switch_token_keeps_official_sentence(self):
        script = self._script("control", itype="perspective_switch",
                              action="fp_to_tp")
        self.assertEqual(
            script.action,
            "transitions from first-person to third-person, "
            "revealing the subject from behind")

    def test_unknown_switch_token_passes_action_through(self):
        label = "tp_to_tp: switch to follow the helicopter flying over the ridge"
        script = self._script("control", itype="perspective_switch", action=label)
        # 官方 else 分支同款:未知 token 原文透传,不丢跟拍指令
        self.assertEqual(script.action, label)
        self.assertEqual(script.actor, "The view")


if __name__ == "__main__":
    unittest.main()
