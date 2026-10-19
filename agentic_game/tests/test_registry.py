"""游戏注册表测试:登记完整性、装配契约与 LLM 依赖声明。"""

import unittest

from engine.parts import GameParts
from game.registry import GAMES, GAME_IDS, GameEntry, get


class RegistryTest(unittest.TestCase):
    def test_all_games_registered(self):
        self.assertIn("duel_oracle_machine_epic", GAMES)
        self.assertIn("openworld", GAMES)
        # 全部主题皮肤都在册(romance 系列已随公开版移除)
        for game_id in ("duel_oracle_machine_epic", "duel_eclipse_devourer_epic",
                        "before_rain_mio", "last_train_hina",
                        "starlight_sister_yui", "after_last_broadcast_ren",
                        "tokyo_eri_sakura"):
            self.assertIn(game_id, GAMES)
        self.assertNotIn("romance", GAMES)  # romance 基础示例随类归档
        self.assertEqual(tuple(GAMES), GAME_IDS)  # 下拉框顺序 = 字典顺序

    def test_entries_are_well_formed(self):
        for game_id, entry in GAMES.items():
            with self.subTest(game=game_id):
                self.assertIsInstance(entry, GameEntry)
                self.assertEqual(entry.game_id, game_id)
                self.assertIn(entry.category, ("romance", "machine", "adv", "openworld"))
                self.assertTrue(entry.title)

    def test_duel_games_build_without_llm(self):
        for game_id, entry in GAMES.items():
            if entry.needs_llm:
                continue
            with self.subTest(game=game_id):
                parts = entry.build(None)
                self.assertIsInstance(parts, GameParts)
                self.assertIsNotNone(parts.runtime)
                self.assertTrue(callable(parts.visual_director))
                self.assertIsNotNone(parts.ending_director)
                self.assertTrue(parts.frame)  # 都声明了首帧图
                self.assertTrue(parts.ending_captions)

    def test_llm_games_reject_none_llm(self):
        """LLM 参与结算的游戏在组装时就应显式拒绝(绝不静默带着 None 跑)。"""
        for game_id, entry in GAMES.items():
            if not entry.needs_llm:
                continue
            with self.subTest(game=game_id):
                with self.assertRaises(ValueError):
                    entry.build(None)

    def test_get_rejects_unknown_game(self):
        with self.assertRaises(KeyError):
            get("no-such-game")

    def test_openworld_scene_option(self):
        entry = get("openworld")
        names = [option.name for option in entry.options]
        self.assertIn("scene", names)
        scene = next(o for o in entry.options if o.name == "scene")
        self.assertIn(scene.default, scene.choices)
        parts = _build_with_stub_llm(entry, scene="western")
        self.assertEqual(parts.runtime.world_id, "western")


def _build_with_stub_llm(entry, **options):
    """LLM 类游戏的组装只要一个带 chat 的对象即可完成(规则层鸭子类型)。"""

    class _Stub:
        pass

    return entry.build(_Stub(), **options)


if __name__ == "__main__":
    unittest.main()
