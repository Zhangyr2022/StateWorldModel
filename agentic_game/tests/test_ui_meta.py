"""大厅展示元数据测试(engine/ui/meta.py):封面 / 简介 / 筛选,全部离线。

meta.py 是界面的纯数据层:不允许 import gradio,坏皮肤静默降级,
封面一律来自皮肤 frame 字段(不允许按 game_id 猜文件名)。
"""

import tempfile
import unittest
from pathlib import Path

from engine.ui import meta
from game.registry import GAMES


class GameMetaTest(unittest.TestCase):
    def setUp(self):
        self.metas = meta.load_game_meta()

    def test_covers_full_registry(self):
        self.assertEqual(set(self.metas), set(GAMES))

    def test_filter_counts(self):
        # openworld 按 scene_choices 展开成 6 张场景卡(其余游戏一张卡)
        for label, expected in (("全部", 25), ("剧情 ADV", 10),
                                ("对决 DUEL", 9), ("开放世界", 6)):
            images, ids = meta.gallery_items(self.metas, label)
            self.assertEqual(len(ids), expected, label)
            self.assertEqual(len(images), expected, label)  # 平行列表
            # 仓库自带全部封面:条目都真实存在
            for path, _ in images:
                self.assertTrue(Path(path).exists(), path)

    def test_gallery_ids_carry_scene(self):
        _, ids = meta.gallery_items(self.metas, "全部")
        for entry in ids:
            self.assertIn("game_id", entry)
            # 非 openworld 卡无场景;openworld 卡必须带 scene
            if entry["game_id"] == "openworld":
                self.assertTrue(entry["scene"], entry)
            else:
                self.assertIsNone(entry["scene"], entry)

    def test_openworld_scene_cards(self):
        images, ids = meta.gallery_items(self.metas, "开放世界")
        scenes = [e["scene"] for e in ids]
        # registry 暴露的 6 个场景一张不少,封面互不相同且存在
        self.assertEqual(sorted(scenes),
                         sorted(["western", "anime", "chinese", "japan",
                                 "roman", "gta"]))
        covers = [Path(p) for p, _ in images]
        self.assertEqual(len({c.name for c in covers}), 6)
        # 文件名去尾数字匹配:gta 卡的封面是 gta6.png
        gta = next(Path(p) for (p, _), e in zip(images, ids) if e["scene"] == "gta")
        self.assertEqual(gta.name, "gta6.png")

    def test_cover_comes_from_frame_field_not_filename(self):
        # 血潮皮肤的游戏 id 与封面文件名不同(blood_tide_* → duel_leviathan.png):
        # 封面必须来自 frame 字段,按 game_id 猜文件名会错。
        cover = self.metas["blood_tide_leviathan_machine"].cover
        self.assertEqual(Path(cover).name, "duel_leviathan.png")
        self.assertTrue(Path(cover).exists())
        # 剧情皮肤封面 = 立绘首帧
        self.assertEqual(Path(self.metas["before_rain_mio"].cover).name, "mio1.png")

    def test_language_and_short_title(self):
        self.assertEqual(self.metas["before_rain_mio_en"].language, "EN")
        self.assertEqual(self.metas["before_rain_mio"].language, "中文")
        self.assertEqual(self.metas["before_rain_mio"].short_title, "雨停之前(雨宫澪)")
        self.assertTrue(self.metas["before_rain_mio"].title.startswith("before_rain_mio · "))

    def test_blurb_is_single_sentence(self):
        for game_id, m in self.metas.items():
            self.assertNotIn("\n", m.blurb, game_id)
            self.assertLessEqual(len(m.blurb), 80, game_id)
        # duel 简介是 "玩家 VS Boss"
        self.assertIn("VS", self.metas["duel_moon_dragon_swordsman"].blurb)
        # EN 皮肤的人设是英文 → 简介非空且非中文人设
        self.assertTrue(self.metas["before_rain_mio_en"].blurb)

    def test_adv_max_turns_from_skin(self):
        self.assertEqual(self.metas["before_rain_mio"].max_turns, 25)
        self.assertIsNone(self.metas["duel_oracle_machine_epic"].max_turns)  # duel 无声明

    def test_broken_skin_dir_degrades_silently(self):
        """坏皮肤静默降级;好皮肤不受影响;索引按文件内 game_id 而非文件名。"""
        import json as _json

        def _entry(game_id, title, category):
            from game.registry import GameEntry
            return GameEntry(game_id, title, category, False, build=lambda *a, **k: None)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # 文件名故意与 game_id 不同(registry 允许,meta 必须按内部 id 索引)
            good = {"game_id": "my_game", "category": "machine", "title": "测试皮肤",
                    "player_name": "甲", "boss": {"name": "乙"},
                    "frame": "assets/duel/duel_leviathan.png"}
            (tmp / "zzz_unrelated_name.json").write_text(
                _json.dumps(good, ensure_ascii=False), encoding="utf-8")
            (tmp / "bad.json").write_text("{broken", encoding="utf-8")
            games = {"my_game": _entry("my_game", "my_game · 测试", "machine"),
                     "no_skin": _entry("no_skin", "no_skin · 无皮肤", "machine")}
            metas = meta.load_game_meta(skin_dir=tmp, games=games)

            self.assertEqual(Path(metas["my_game"].cover).name, "duel_leviathan.png")
            self.assertEqual(metas["my_game"].blurb, "甲 VS 乙")
            # 坏 JSON / 无皮肤:全字段降级,不抛错
            for game_id in ("no_skin",):
                self.assertIsNone(metas[game_id].cover, game_id)
                self.assertEqual(metas[game_id].blurb, "", game_id)

    def test_unknown_filter_degrades_to_all(self):
        """未知筛选值(不会从下拉框产生):优雅退化为不过滤。"""
        images, ids = meta.gallery_items(self.metas, "不存在的筛选")
        all_images, all_ids = meta.gallery_items(self.metas, "全部")
        self.assertEqual((images, ids), (all_images, all_ids))


class LanguageTest(unittest.TestCase):
    """界面语言:大厅列表 / 标题 / 简介按语言切换。"""

    def setUp(self):
        self.metas = meta.load_game_meta()

    def test_lobby_lists_per_language(self):
        for lang in ("en", "zh"):
            _, ids = meta.gallery_items(self.metas, "全部", lang)
            self.assertEqual(len(ids), 20, lang)  # 6 场景卡 + 5 ADV + 9 DUEL
            adv = [e["game_id"] for e in ids if self.metas[e["game_id"]].category == "adv"]
            self.assertEqual(len(adv), 5, lang)
            # ADV 只列与界面语言一致的那版皮肤
            self.assertTrue(all(g.endswith("_en") == (lang == "en") for g in adv), adv)
            # 开放世界(仅英文)与 DUEL 两种语言都列
            cats = {self.metas[e["game_id"]].category for e in ids}
            self.assertIn("openworld", cats)
            self.assertIn("machine", cats)

    def test_english_titles_have_no_chinese(self):
        _, ids = meta.gallery_items(self.metas, "全部", "en")
        for entry in ids:
            m = self.metas[entry["game_id"]]
            for text in (meta.display_title(m, "en"), meta.display_blurb(m, "en")):
                self.assertTrue(text, m.game_id)
                self.assertFalse(any("一" <= ch <= "鿿" for ch in text),
                                 f"{m.game_id}: {text}")
        duel = self.metas["duel_abyssal_leviathan_machine_v2"]
        self.assertEqual(meta.display_title(duel, "en"), "Keira vs. Tarasos")
        self.assertEqual(meta.display_title(duel, "zh"), "深渊猎手 VS 星渊巨鲸")

    def test_counterpart_on_language_switch(self):
        self.assertEqual(meta.counterpart_id("before_rain_mio", self.metas, "en"),
                         "before_rain_mio_en")
        self.assertEqual(meta.counterpart_id("before_rain_mio_en", self.metas, "zh"),
                         "before_rain_mio")
        self.assertEqual(meta.counterpart_id("openworld", self.metas, "zh"), "openworld")
        self.assertEqual(meta.counterpart_id("duel_worldspine_hunter", self.metas, "en"),
                         "duel_worldspine_hunter")

    def test_openworld_cards_named_after_scenes(self):
        """开放世界卡片/标题用场景名(GTA / Japan ...),不再是统一的游戏名。"""
        for lang, want in (("en", ["Western", "Anime", "Chinese", "Japan", "Roman", "GTA"]),
                           ("zh", ["西部", "动漫", "古风", "日本", "罗马", "GTA"])):
            images, _ = meta.gallery_items(self.metas, "开放世界", lang)
            self.assertEqual([cap for _, cap in images], want, lang)
        ow = self.metas["openworld"]
        self.assertEqual(meta.display_title(ow, "en", "japan"), "Japan")
        self.assertEqual(meta.display_title(ow, "en"), "Western")  # 缺省 = 首个场景
        self.assertEqual(meta.scene_title("new_world", "en"), "New World")  # 未收录兜底

    def test_duel_titles_are_a_vs_b(self):
        """中文 DUEL 标题只显示 "A VS B",去掉括注与副标题。"""
        self.assertEqual(
            meta.display_title(self.metas["duel_abyssal_leviathan_machine_v2"], "zh"),
            "深渊猎手 VS 星渊巨鲸")
        self.assertEqual(meta.display_title(self.metas["duel_oracle_machine_epic"], "zh"),
                         "失格神谕 VS 六翼审判者")
        for m in self.metas.values():
            if m.category == "machine":
                title = meta.display_title(m, "zh")
                self.assertRegex(title, r"^\S+ VS \S+$", title)

    def test_openworld_is_english_only(self):
        self.assertEqual(self.metas["openworld"].language, "EN")

    def test_scene_cover_per_scene(self):
        """点哪张开放世界卡,详情大图就是哪个场景(不再固定 western)。"""
        self.assertEqual(meta.scene_cover("anime").name, "anime.png")
        self.assertEqual(meta.scene_cover("gta").name, "gta6.png")
        self.assertIsNone(meta.scene_cover("no-such-scene"))


class UiTextTest(unittest.TestCase):
    def test_every_key_has_both_languages(self):
        from engine.ui import i18n

        for key, pair in i18n._TEXT.items():
            self.assertEqual(len(pair), 2, key)
            self.assertTrue(all(pair), key)
        self.assertEqual(i18n.t("en", "turns_n", n=5), "5 turns")
        self.assertEqual(i18n.t("zh", "turns_n", n=5), "5 回合")
        self.assertEqual(set(i18n.CATEGORY_HINT["en"]), set(i18n.CATEGORY_HINT["zh"]))
        self.assertEqual(set(i18n.FREE_TEXT_LABEL["en"]), set(i18n.FREE_TEXT_LABEL["zh"]))


class NoGradioImportTest(unittest.TestCase):
    """界面数据/表现层是纯 Python:不许依赖 gradio(离线可测的前提)。"""

    def test_meta_and_theme_do_not_import_gradio(self):
        from engine.ui import theme  # noqa: F401 - 能 import 本身就是一半验证

        from engine.ui import i18n

        for module in (meta, theme, i18n):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("import gradio", source, module.__name__)


if __name__ == "__main__":
    unittest.main()
