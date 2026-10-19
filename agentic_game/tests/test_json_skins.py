"""JSON 皮肤测试:加载/校验/注册表挂载。

皮肤的唯一形态就是 game/skins/*.json(原 definition.py 已全部迁移为 JSON)。
本套件保证:目录里每个文件都能加载、都注册进注册表、machine/adv 皮肤都能
build,以及校验器能拦住设计师常见的错误。
"""

import json
import tempfile
import unittest
from pathlib import Path

from game.registry import GAMES

SKINS_DIR = Path(__file__).resolve().parents[1] / "game" / "skins"

# 在册 = duel(machine) 系列 + adv 系列(5 款中文 + 5 款英文 *_en)。
# 历史 romance 系列与经典简化皮肤(museum_heist 等)已随公开版整理移除,
# 不在扫描范围。
_EXPECTED_SKINS = {
    "duel_oracle_machine_epic",
    "blood_tide_leviathan_machine",
    "duel_abyssal_leviathan_machine_v2",
    "duel_nine_heavens_artificer", "duel_shogun_mask_palace_epic", "duel_last_human_colossus", "duel_worldspine_hunter", "duel_moon_dragon_swordsman", "duel_eclipse_devourer_epic",
    "before_rain_mio", "last_train_hina", "starlight_sister_yui",
    "after_last_broadcast_ren", "tokyo_eri_sakura",
    "before_rain_mio_en", "last_train_hina_en", "starlight_sister_yui_en",
    "after_last_broadcast_ren_en", "tokyo_eri_sakura_en",
}


def _load(path: Path):
    category = json.loads(path.read_text(encoding="utf-8")).get("category")
    if category == "romance":
        raise AssertionError(
            "romance 皮肤不应再出现在 game/skins/(历史系列已随公开版移除)")
    if category == "adv":
        from game.adv_theme.json_skin import load_json_skin
        return load_json_skin(path)
    from game.machine_theme.json_skin import load_json_skin
    return load_json_skin(path)


class TestSkinDirectory(unittest.TestCase):
    def test_all_skins_load(self):
        """game/skins/ 里每个 JSON 都必须能通过校验加载。"""
        paths = sorted(SKINS_DIR.glob("*.json"))
        self.assertTrue(paths)
        ids = set()
        for path in paths:
            skin = _load(path)  # 不合法会抛 SkinValidationError
            ids.add(skin.game_id)
        self.assertEqual(ids, _EXPECTED_SKINS)

    def test_all_skins_registered(self):
        """每个皮肤都以自己的 game_id 出现在注册表里。"""
        for game_id in _EXPECTED_SKINS:
            self.assertIn(game_id, GAMES, game_id)

    def test_machine_skins_build_offline(self):
        """machine 类皮肤(状态机)全部能离线组装并开局。"""
        for path in sorted(SKINS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("category") != "machine":
                continue
            with self.subTest(skin=path.name):
                parts = GAMES[data["game_id"]].build()
                state = parts.runtime.initial_state()
                self.assertTrue(state.current)
                self.assertTrue(callable(parts.visual_director))

    def test_romance_skins_reject_none_llm(self):
        """romance 类皮肤组装时必须显式拒绝 None LLM(绝不静默带 None 跑)。"""
        for path in sorted(SKINS_DIR.glob("romance_*.json")):
            with self.subTest(skin=path.name):
                game_id = json.loads(path.read_text(encoding="utf-8"))["game_id"]
                with self.assertRaises(ValueError):
                    GAMES[game_id].build(None)

    def test_adv_skins_reject_none_llm(self):
        """adv 皮肤必须显式拒绝 None LLM(绝不静默带着 None 跑)。"""
        for path in sorted(SKINS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("category") != "adv":
                continue
            game_id = data["game_id"]
            with self.subTest(skin=path.name):
                with self.assertRaises(ValueError):
                    GAMES[game_id].build(None)

    def test_broken_skin_in_skins_dir_does_not_kill_registry(self):
        """game/skins/ 里放一个坏文件(非法 JSON/未知 category):被跳过并告警,
        其他游戏不受影响。"""
        from game import registry

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "unknown_category.json"
            # 已归档的 duel 路线 category 同样落在"未知 category"守卫里
            bad.write_text('{"category": "duel"}', encoding="utf-8")
            original = registry._JSON_SKIN_DIR
            try:
                registry._JSON_SKIN_DIR = Path(tmp)
                self.assertEqual(registry._json_skin_entries(), {})
            finally:
                registry._JSON_SKIN_DIR = original
            self.assertIn("duel_oracle_machine_epic", GAMES)


if __name__ == "__main__":
    unittest.main()
