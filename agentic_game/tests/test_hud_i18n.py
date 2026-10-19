"""HUD i18n 覆盖测试:全部 duel 皮肤的名称/动作/描述经 tr() 后不得残留中文。

rebuild_replay_hud 用皮肤词条重建 HUD,再经 montage 的 tr() 上屏;任何
未收录的中文词条都会以中文混入英文回放。本测试把 9 款皮肤的全部 HUD
可见文本枚举一遍,保证 AGENTIC_GAME_HUD_LANG=en(默认)时全英文。
"""

import json
import re
import unittest
from pathlib import Path

from engine.hud_i18n import LANG, current_lang, hud_lang, tr
from game.adv_theme.json_skin import load_json_skin

CJK = re.compile(r"[一-鿿]")
SKINS_DIR = Path(__file__).resolve().parents[1] / "game" / "skins"


def _machine_skins():
    from game.machine_theme.json_skin import load_json_skin
    for path in sorted(SKINS_DIR.glob("*.json")):
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("category") == "machine":
            yield load_json_skin(path)


class TestHudI18nEnglish(unittest.TestCase):
    def test_default_lang_is_english(self):
        self.assertEqual(LANG, "en")

    def test_hud_lang_override(self):
        """界面按玩家所选语言出 HUD:上下文覆盖 + 显式 lang 参数。"""
        self.assertEqual(tr("敌方行动", "zh"), "敌方行动")
        with hud_lang("zh"):
            self.assertEqual(current_lang(), "zh")
            self.assertEqual(tr("敌方行动"), "敌方行动")
            self.assertEqual(tr("敌方行动", "en"), "Enemy Action")
            with hud_lang(None):  # None = 不覆盖,沿用外层
                self.assertEqual(current_lang(), "zh")
        self.assertEqual(current_lang(), LANG)
        self.assertEqual(tr("敌方行动"), "Enemy Action")

    def test_openworld_and_adv_vocabulary(self):
        """英文界面不得残留中文:开放世界固定措辞 / 主线阶段 / adv 说话动作。"""
        from game.adv_theme.actions import ACTION_TALK
        from game.openworld.actions import ACTION_ACT, ACTION_AUTO, ACTION_CUSTOM
        from game.openworld.quests import QUESTS
        from game.openworld.rules import FALLBACK_ACTIONS

        texts = [a.label for a in (ACTION_ACT, ACTION_AUTO, ACTION_CUSTOM, ACTION_TALK)]
        texts += [a.description for a in (ACTION_ACT, ACTION_AUTO, ACTION_CUSTOM, ACTION_TALK)]
        texts += list(FALLBACK_ACTIONS)
        texts += ["本幕自动探索 · 下一回合由你决定", "VLM 提议的行动", "冒险刚刚开始。"]
        for quest in QUESTS.values():
            texts += [quest.title, *quest.stages]
        for text in texts:
            self.assertNotRegex(tr(text, "en"), CJK, text)

    def test_duel_ending_captions(self):
        """DUEL 终局字幕(皮肤可自定义)经 tr 后全英文;取组装后的 GameParts。"""
        from game.registry import GAMES

        checked = 0
        for game_id, entry in GAMES.items():
            if entry.category != "machine":
                continue
            for caption in entry.build(None).ending_captions.values():
                checked += 1
                self.assertNotRegex(tr(caption, "en"), CJK, f"{game_id}:{caption}")
        self.assertGreater(checked, 0)

    def test_fullwidth_punctuation_normalised(self):
        self.assertEqual(tr("体力 30\uff08力竭剩余 1 幕\uff09", "en"),
                         "Stamina 30 (exhausted for 1 more act(s))")

    def test_machine_state_labels(self):
        """对局面板"当前局势"里的状态名(英文界面经 tr 显示)。"""
        for skin in _machine_skins():
            for state_id in skin.states:
                label = skin.state(state_id).label
                self.assertNotRegex(tr(label, "en"), CJK, f"{skin.game_id}:{label}")

    def test_fixed_labels(self):
        for zh, want in (("敌方行动", "Enemy Action"),
                         ("Boss 按兵不动", "Boss holds position"),
                         ("◆ 战斗胜利 ◆", "◆ Victory ◆"),
                         ("◆ 战斗失败 ◆", "◆ Defeat ◆"),
                         ("◆ 故事落幕 ◆", "◆ The Story Ends ◆")):
            self.assertEqual(tr(zh), want)

    def test_templates(self):
        cases = {
            "第3回合 · 神谕·第一句": "Turn 3 · Oracle: First Sentence",
            "Boss 招式:云中凝视": "Boss Move: Gaze from the Clouds",
            "Boss 招式:月蚀龙息(Boss盾,蓄力中)":
                "Boss Move: Eclipse Dragon Breath(Boss Shield,Charging)",
            "关系 52/100": "Relationship 52/100",
            "关系:礼貌的同学 · 心情:慌乱":
                "Relationship: a polite classmate · Mood: Flustered",
            "Story: x": "Story: x",  # 英文原文透传
            "剧情:节点一 · 相识": "Story: 节点一 · 相识",  # 节点名属皮肤内容
            "恋爱意图@第21回合": "Love intent @ turn 21",
            "结局:special": "Ending: special",
            "态度 +2 · 雨宫澪看起来慌乱": "Attitude +2 · 雨宫澪 looks Flustered",
            "态度持平 · 她看起来平静": "Attitude unchanged · 她 looks Calm",
            "她的心情:calm(关系 +2)": "Her mood: calm(relationship +2)",
        }
        for zh, want in cases.items():
            self.assertEqual(tr(zh), want, zh)

    def test_machine_skin_strings_all_translate(self):
        """全部 duel(状态机)皮肤的 HUD 可见文本 tr() 后无中文。"""
        checked = 0
        for skin in _machine_skins():
            strings = [skin.player_name]
            if getattr(skin, "mp_name", ""):
                strings.append(skin.mp_name)
            for a in skin.actions or ():
                strings += [a.label, a.description or ""]
            boss = getattr(skin, "boss", None)
            if boss is not None:
                strings.append(boss.name)
                for a in boss.actions or ():
                    strings.append(a.label)
            for s in strings:
                if not s:
                    continue
                checked += 1
                self.assertFalse(CJK.search(tr(s)), f"{skin.game_id}: {s!r}")
        self.assertGreater(checked, 200)  # 9 款皮肤 × ~18 条

    def test_machine_hud_lines_all_translate(self):
        """hud_machine 拼出的整行(含 Boss 招式/标记)tr() 后无中文。"""
        for skin in _machine_skins():
            boss = skin.boss
            for a in boss.actions or ():
                line = f"Boss 招式:{a.label}"
                self.assertFalse(CJK.search(tr(line)),
                                 f"{skin.game_id}: {line!r}")
            marked = f"Boss 招式:{boss.actions[0].label}(Boss盾,狂暴,蓄力中)"
            self.assertFalse(CJK.search(tr(marked)),
                             f"{skin.game_id}: {marked!r}")

    def test_adv_hud_labels_translate(self):
        """rebuild_replay_hud 的 hud_adv 固定措辞(混英文皮肤内容)可整行翻译。"""
        skin = load_json_skin(SKINS_DIR / "before_rain_mio.json")
        label = skin.node_for_turn(1).label  # 节点一 · 相识(中文皮肤 → 中文节点名)
        lines = [
            f"第1回合 · {label}",
            f"关系 92/100",
            f"关系:心意接近摊牌 · 心情:温软",
            f"剧情:{label}",
            "恋爱意图@第21回合",
            "结局:special",
            f"态度 +2 · {skin.girl_name}看起来温软",
            f"态度持平 · {skin.girl_name}看起来平静",
        ]
        for line in lines:
            tr(line)  # 不抛错即可;逐行精确断言在下面
        # 固定部分必须翻出(皮肤内容——节点名/她名——允许保留原文)
        self.assertTrue(tr(lines[0]).startswith("Turn 1 · "))
        self.assertEqual(tr(lines[1]), "Relationship 92/100")
        self.assertTrue(tr(lines[2]).startswith(
            "Relationship: close to confession · Mood: Warm"))
        self.assertTrue(tr(lines[3]).startswith("Story: "))
        self.assertEqual(tr(lines[4]), "Love intent @ turn 21")
        self.assertEqual(tr(lines[5]), "Ending: special")
        self.assertTrue(tr(lines[6]).startswith("Attitude +2 · "))
        self.assertTrue(tr(lines[6]).endswith(" looks Warm"))
        self.assertTrue(tr(lines[7]).startswith("Attitude unchanged · "))


if __name__ == "__main__":
    unittest.main()
