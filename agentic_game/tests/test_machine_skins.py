"""machine(状态机)主题测试:JSON 加载、转移求值、终局、离线可玩。"""

import json
import unittest
from pathlib import Path

from engine.runtime import Rng
from game.machine_theme import build_machine_game
from game.machine_theme.json_skin import (
    SkinValidationError,
    load_json_skin,
    skin_from_dict,
)
from game.machine_theme.runtime import MachineRuntime
from game.registry import GAMES

SKINS_DIR = Path(__file__).resolve().parents[1] / "game" / "skins"
# museum_heist 不在游戏清单(探索类机制样例),作为机制测试载具放在 fixtures/
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
MUSEUM_HEIST = FIXTURES_DIR / "museum_heist.json"


def _stone_machine() -> MachineRuntime:
    return MachineRuntime(load_json_skin(MUSEUM_HEIST))


def _action(runtime, slot: str):
    return next(a for a in runtime.available_actions(
        runtime.initial_state()) if a.id == slot)


class TestMachineSkin(unittest.TestCase):
    def test_example_loads_and_registers(self):
        self.assertIn("duel_oracle_machine_epic", GAMES)
        entry = GAMES["duel_oracle_machine_epic"]
        self.assertEqual(entry.category, "machine")
        self.assertFalse(entry.needs_llm)

    def test_all_machine_skins_valid(self):
        for path in sorted(SKINS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("category") == "machine":
                _load = load_json_skin(path)  # 不合法会抛
                self.assertTrue(_load.states)

    def test_transition_applies_effects_and_events(self):
        runtime = _stone_machine()
        state = runtime.initial_state()
        outcome = runtime.resolve_turn(state, _action(runtime, "investigate"), Rng(1))
        self.assertIn("clue_found", [e.type for e in outcome.events])
        # 概率 0.7,固定 seed 只会命中或停留;变量只被 effects 修改
        self.assertIn(outcome.state_after.vars["clues"], (0, 1))

    def test_condition_gates_transition(self):
        runtime = _stone_machine()
        state = runtime.initial_state()
        state.vars["clues"] = 3  # investigate 的转移条件是 clues < 3
        outcome = runtime.resolve_turn(state, _action(runtime, "investigate"), Rng(2))
        self.assertTrue(any(e.detail.get("stayed") for e in outcome.events))
        self.assertEqual(outcome.state_after.vars["clues"], 3)

    def test_state_actions_restricted(self):
        runtime = _stone_machine()
        state = runtime.initial_state()
        state.current = "vault"
        ids = {a.id for a in runtime.available_actions(state)}
        self.assertEqual(ids, {"grab", "lie_low"})

    def test_win_condition_and_ending(self):
        runtime = _stone_machine()
        state = runtime.initial_state()
        state.vars.update(clues=3, in_vault=True, has_loot=True)
        self.assertEqual(runtime.status_of(state), "won")
        parts = build_machine_game(load_json_skin(MUSEUM_HEIST))
        ending = parts.ending_director("won", parts.visual_state)
        self.assertIsNotNone(ending)
        self.assertIn("satchel", ending.action)

    def test_deterministic_replay(self):
        def play(seed: int):
            runtime = _stone_machine()
            state = runtime.initial_state()
            rng = Rng(seed)
            trace = []
            for _ in range(30):
                actions = runtime.available_actions(state)
                if not actions:
                    break
                pick = actions[0]
                outcome = runtime.resolve_turn(state, pick, rng)
                state = outcome.state_after
                trace.append((pick.id, state.current,
                              tuple(sorted(state.vars.items()))))
            return tuple(trace)

        # 同 seed ⇒ 完全相同轨迹(确定性;不同 seed 不保证不同,只保证可复现)
        self.assertEqual(play(9), play(9))
        self.assertEqual(play(10), play(10))

    def test_boss_actions_have_description(self):
        for name in ("duel_oracle_machine_epic.json", "duel_oracle_machine_epic.json"):
            skin = load_json_skin(SKINS_DIR / name)
            for spec in skin.boss.actions:
                self.assertTrue(spec.description, f"{name}:{spec.id} 缺 description")

    def test_boss_menu_and_forced_action(self):
        skin = load_json_skin(SKINS_DIR / "duel_oracle_machine_epic.json")
        runtime = MachineRuntime(skin)
        state = runtime.initial_state()
        menu = runtime.boss_action_menu(state)
        self.assertEqual([a.id for a in menu],
                         ["windup", "release", "fist", "shield_raise", "mend",
                          "enrage", "idle"])
        self.assertTrue(all(a.description for a in menu))
        # 强制 release:前置(windup)未满足也照常接管,一次性、用后即清
        runtime.forced_boss_action = "release"
        outcome = runtime.resolve_turn(state, _action(runtime, "arcane"), Rng(1))
        self.assertTrue(any(e.type == "state_change"
                            and e.detail.get("action") == "release"
                            for e in outcome.events))
        self.assertIsNone(runtime.forced_boss_action)

    def test_validation_unknown_state_rejected(self):
        data = json.loads((MUSEUM_HEIST).read_text(encoding="utf-8"))
        data["transitions"][0]["to"] = "nowhere"
        with self.assertRaises(SkinValidationError) as ctx:
            skin_from_dict(data)
        self.assertIn("未知状态", str(ctx.exception))

    def test_validation_unknown_condition_op_rejected(self):
        data = json.loads((MUSEUM_HEIST).read_text(encoding="utf-8"))
        data["endings"]["won"]["condition"][0]["op"] = "~="
        with self.assertRaises(SkinValidationError) as ctx:
            skin_from_dict(data)
        self.assertIn("op", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
