"""Determinism 测试:相同 (initial state, action 序列, seed) ⇒ 完全相同的结果。

载具:duel_oracle_machine_epic(状态机战斗;引擎级测试与具体游戏解耦)。
"""

import base64
import unittest
from pathlib import Path

from engine.agents import ScriptedAgent
from engine.loop import GameLoop
from engine.renderer.fake import FakeRenderer
from engine.runtime import STATUS_WON, Rng
from engine.runtime import Action
from game.registry import GAMES

_PARTS = GAMES["duel_oracle_machine_epic"].build()
_SKIN = _PARTS.runtime.skin


def _action(slot: str) -> Action:
    spec = _SKIN.action(slot)
    return Action(id=slot, label=spec.label, description=spec.description)


def MachineRuntime():
    return GAMES["duel_oracle_machine_epic"].build().runtime


def initial_visual_state():
    return GAMES["duel_oracle_machine_epic"].build().visual_state


def state_digest(state) -> dict:
    return {"current": state.current, "turn": state.turn,
            "ending": state.ending, "vars": dict(state.vars)}


# 1x1 的合法 PNG(测试里不能用坏图:ffmpeg 会耗满超时才放弃)
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def game_digest(seed: int, script: tuple[str, ...], tmpdir: Path) -> dict:
    """跑一整局,返回与路径无关的结果摘要。"""
    initial_frame = tmpdir / f"initial-{seed}.png"
    initial_frame.write_bytes(TINY_PNG)
    visual_state = initial_visual_state()
    visual_state.current_frame = initial_frame
    renderer = FakeRenderer(output_dir=tmpdir / f"session-{seed}")
    loop = GameLoop(
        runtime=MachineRuntime(),
        agent=ScriptedAgent(script, fallback=True),
        visual_director=_PARTS.visual_director,
        renderer=renderer,
        visual_state=visual_state,
        output_dir=tmpdir / f"session-{seed}",
        base_seed=seed,
        max_turns=12,
    )
    summary = loop.run()
    return {
        "statuses": [record.status for record in summary.records],
        "actions": [record.action_id for record in summary.records],
        "final_status": summary.final_status,
        "prompts": [record.prompt for record in summary.records],
    }


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_action_sequence_same_result(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = ("fireball", "ice_shield", "heal", "arcane", "fireball", "frostbolt")
            first = game_digest(42, script, tmpdir)
            second = game_digest(42, script, tmpdir)
        self.assertEqual(first, second)

    def test_resolve_is_pure_given_same_seed(self):
        runtime = MachineRuntime()
        state = runtime.initial_state()
        first = runtime.resolve_turn(state, _action('arcane'), Rng(7))
        second = runtime.resolve_turn(state, _action('arcane'), Rng(7))
        self.assertEqual([e.type for e in first.events],
                         [e.type for e in second.events])
        self.assertEqual(state_digest(first.state_after),
                         state_digest(second.state_after))

    def test_rng_reproducible_per_seed(self):
        self.assertEqual(Rng(1).randint(0, 100), Rng(1).randint(0, 100))
        self.assertEqual(Rng(1).random(), Rng(1).random())

    def test_different_seed_changes_outcome_stream(self):
        """不同 seed 至少在前 3 回合产生不同的事件流(几乎必然)。"""
        digests = []
        for seed in (1, 2, 3, 4):
            runtime = MachineRuntime()
            state = runtime.initial_state()
            rng = Rng(seed)
            turns = []
            for _ in range(3):
                outcome = runtime.resolve_turn(state, _action('arcane'), rng)
                state = outcome.state_after
                turns.extend((e.type, e.message) for e in outcome.events)
            digests.append(turns)
        self.assertEqual(len({repr(digest) for digest in digests}), 4)

    def test_full_game_replay_is_reproducible(self):
        from engine.runtime import STATUS_ONGOING

        def play(runtime, state, rng):
            turns = 0
            while runtime.status_of(state) == STATUS_ONGOING and turns < 40:
                action = _action('fireball') if state.vars.get('player_mp', 0) >= 5 \
                    else _action('arcane')
                outcome = runtime.resolve_turn(state, action, rng)
                state = outcome.state_after
                turns += 1
            return state, turns

        runtime = MachineRuntime()
        state, turns = play(runtime, runtime.initial_state(), Rng(3))
        self.assertNotEqual(runtime.status_of(state), STATUS_ONGOING)

        # 重放:相同 seed 从头再来,逐回合状态一致(无论胜负)
        replay_runtime = MachineRuntime()
        replay_state, replay_turns = play(replay_runtime,
                                          replay_runtime.initial_state(), Rng(3))
        self.assertEqual(turns, replay_turns)
        self.assertEqual(state_digest(state), state_digest(replay_state))


if __name__ == "__main__":
    unittest.main()
