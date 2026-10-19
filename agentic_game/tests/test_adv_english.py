"""adv 英文局(language: "en")测试:皮肤加载、英文契约、守卫方向、演出语言。

英文皮肤 = zh 皮肤的同构翻译(game/skins/*_en.json):对白、剧情事件、
LLM prompt 模板、事件措辞全部换英文;结算规则与中文版逐字同构。
LLM 用进程内 FakeLLM,不依赖常驻服务。
"""

import json
import tempfile
import unittest
from pathlib import Path

from engine.runtime import Action, Rng
from game.adv_theme.json_skin import load_json_skin
from game.adv_theme.runtime import AdvRuntime
from game.registry import GAMES

SKINS = Path(__file__).resolve().parents[1] / "game" / "skins"
EN_SKIN = SKINS / "before_rain_mio_en.json"


def _en_json(reply="...Fine. You win this one.", delta=1, mood="flustered") -> str:
    return json.dumps({"reply": reply, "delta_reason": "cared about me",
                       "relationship_delta": delta, "mood": mood},
                      ensure_ascii=False)


class FakeLLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, seed=0, max_tokens=256, timeout_s=None):
        self.calls.append({"messages": messages})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_runtime(responses, skin_path=EN_SKIN):
    skin = load_json_skin(skin_path)
    return AdvRuntime(skin, FakeLLM(responses)), skin


class TestEnglishSkin(unittest.TestCase):
    def test_language_field_and_english_flag(self):
        skin = load_json_skin(EN_SKIN)
        self.assertEqual(skin.language, "en")
        self.assertTrue(skin.is_english)
        self.assertEqual(skin.game_id, "before_rain_mio_en")
        self.assertEqual(skin.girl_name, "Mio Amamiya")

    def test_zh_skin_defaults_to_chinese(self):
        skin = load_json_skin(SKINS / "before_rain_mio.json")
        self.assertEqual(skin.language, "zh")
        self.assertFalse(skin.is_english)

    def test_registered_in_registry(self):
        entry = GAMES["before_rain_mio_en"]
        self.assertEqual(entry.category, "adv")
        self.assertTrue(entry.needs_llm)

    def test_all_five_en_skins_load(self):
        for gid in ("before_rain_mio_en", "last_train_hina_en",
                    "starlight_sister_yui_en", "after_last_broadcast_ren_en",
                    "tokyo_eri_sakura_en"):
            skin = load_json_skin(SKINS / f"{gid}.json")
            self.assertEqual(skin.language, "en")
            self.assertEqual(len(skin.nodes), 5)
            # 运行时会加 "Never " 前缀,约束短语本身不得已带 never
            for c in skin.girl_constraints:
                self.assertFalse(c.strip().lower().startswith("never"), c)


class TestEnglishRuntime(unittest.TestCase):
    def test_system_prompt_is_english(self):
        rt, _ = make_runtime([_en_json()])
        state = rt.initial_state()
        node = rt.skin.node_for_turn(1)
        prompt = rt._build_system_prompt(state, node)
        self.assertNotRegex(prompt, r"[一-鿿]")
        self.assertIn("Mio Amamiya", prompt)
        self.assertIn("spoken English", prompt)
        self.assertIn("Never analyse yourself", prompt)

    def test_chinese_system_prompt_unchanged(self):
        rt, _ = make_runtime([_en_json()], SKINS / "before_rain_mio.json")
        state = rt.initial_state()
        prompt = rt._build_system_prompt(state, rt.skin.node_for_turn(1))
        self.assertIn("【角色】", prompt)
        self.assertIn("回应必须使用中文(简体)", prompt)

    def test_resolve_produces_english_events(self):
        rt, _ = make_runtime([_en_json(delta=2)])
        state = rt.initial_state()
        outcome = rt.resolve_turn(state, Action(id="talk", label="Nice to see you."),
                                  Rng(1))
        kinds = {e.type: e.message for e in outcome.events}
        self.assertEqual(kinds["player_line"], "Nice to see you.")
        self.assertEqual(kinds["girl_reply"], "...Fine. You win this one.")
        self.assertNotRegex(kinds["relationship_change"], r"[一-鿿]")
        state_after = outcome.state_after
        self.assertEqual(state_after.last_reply, "...Fine. You win this one.")
        describe = rt.describe_state(state_after)
        self.assertTrue(describe.startswith("Turn 1 [Chapter 1"))
        self.assertIn("Mio Amamiya", describe)
        self.assertNotRegex(describe, r"[一-鿿]")

    def test_dialogue_labelled_english_in_visual_script(self):
        from engine.loop import GameLoop
        from engine.renderer.fake import FakeRenderer
        from game.adv_theme import build_adv_game

        class Scripted:
            def __init__(self):
                self.i = 0

            def choose(self, state, actions):
                self.i += 1
                return Action(id="talk", label=f"line {self.i}")

        skin = load_json_skin(EN_SKIN)
        parts = build_adv_game(skin, FakeLLM([_en_json()] * 30))
        with tempfile.TemporaryDirectory() as td:
            renderer = FakeRenderer(Path(td))
            loop = GameLoop(runtime=parts.runtime, agent=Scripted(),
                            visual_director=parts.visual_director, renderer=renderer,
                            visual_state=parts.visual_state,
                            output_dir=Path(td), base_seed=1, max_turns=2)
            loop.run()
            prompts = [r["prompt"] for r in loop_run_records(loop)]
        self.assertTrue(prompts)
        for p in prompts:
            self.assertIn("<d>[English]", p)
            self.assertNotIn("<d>[Chinese]", p)

    def test_chinese_reply_rejected_in_english_mode(self):
        # 英文局守卫方向翻转:模型违规输出中文 → 中性英文兜底,不进历史原文
        rt, _ = make_runtime(["{\"reply\": \"嗯,知道了。\", \"relationship_delta\": 0, \"mood\": \"calm\"}"])
        state = rt.initial_state()
        outcome = rt.resolve_turn(state, Action(id="talk", label="Hello there."),
                                  Rng(2))
        self.assertNotRegex(outcome.state_after.last_reply, r"[一-鿿]")

    def test_chinese_mode_guard_unchanged(self):
        # 中文局守卫:模型违规输出无中文文本 → 中文兜底(回归)
        rt, _ = make_runtime([_en_json()], SKINS / "before_rain_mio.json")
        state = rt.initial_state()
        outcome = rt.resolve_turn(state, Action(id="talk", label="你好呀。"), Rng(3))
        self.assertIn("轻轻点了点头", outcome.state_after.last_reply)

    def test_history_narration_prefix_english(self):
        from game.adv_theme.runtime import build_history_messages
        skin = load_json_skin(EN_SKIN)
        history = [("hi", "hello"), ("hi again", "hello again")]
        msgs = build_history_messages(skin, history, whose="girl")
        self.assertIn("(story · Chapter 1", msgs[0]["content"])


def loop_run_records(loop):
    """GameLoop 的 prompt 落盘在会话日志里;逐行读回。"""
    log = (loop.output_dir / "turn_log.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in log.splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
