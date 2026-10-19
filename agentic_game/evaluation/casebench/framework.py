"""完整游戏框架路径:GameLoop → 规则结算 → VisualScript → PromptCompiler。

每个孪生侧被包装成一个单回合"游戏",走引擎原装链路(与 duel/romance
同一条代码路径,评测层只提供游戏定义):

- CaseRuntime(GameRuntime 协议):initial_state = 数据 initial_state;
  shared_action 作为唯一动作;resolve_turn 用迷你规则引擎对数据的
  precondition/effect/else_effect 做确定性结算(if 命中走 effect,否则
  else_effect)——结果先于渲染确定(Rule-Grounded);
- director_for(VisualDirector):结算前后状态 × visual_vocab 全表重申
  生成 VisualScript(场景锚点 = 起始全表;end_state = 终态全表,反物体
  以 remain 状态自然入场),不使用数据 instruction 的任何文本;
- GameLoop + PromptCompiler:编译 H3 信封 prompt 并渲染,末帧回填。

prompt 逐句可审计:每句都能追溯到 scene / action / vocab×状态表。
"""

from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path

from engine.agents import ScriptedAgent
from engine.events import GameEvent
from engine.loop import RENDER_FAILURE_ABORT, GameLoop
from engine.runtime import STATUS_ONGOING, STATUS_WON, Action, GameRuntime, TurnOutcome
from engine.visual.compiler import PromptCompiler
from engine.visual.script import VisualScript
from engine.visual.state import VisualState

from evaluation.casebench.dataset import TwinSpec
from evaluation.casebench.render import (apply_ref2va_anchor, final_frame_for,
                                         video_path_for)

# 状态内私有标记:规则是否已结算(结算后 GameLoop 即收局)。
_DONE = "__resolved__"

_PRECONDITION_RE = re.compile(r"^\s*([\w.]+)\s*==\s*(.+?)\s*$")
_ASSIGN_RE = re.compile(r"^\s*([\w.]+)\s*=\s*(.+?)\s*$")

_CAMERA = ("locked-off static wide shot with no pan, no zoom and no cut; "
           "every figure keeps the same position as in the frame")
_APPEARANCE = ("matching the person shown in <Picture 1>, unchanged in outfit "
               "and build")
_STABILITY = ("no on-screen text, subtitles, numbers or HUD elements appear",
              "the protagonist and the scene keep the same appearance and "
              "position as <Picture 1>")


def _norm(value: object) -> str:
    return str(value).strip().lower().replace("'", "").replace('"', "")


def _parse_literal(text: str):
    """规则右值字面量:带引号 → 字符串;true/false → bool;整数 → int。"""
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    if t.lower() == "true":
        return True
    if t.lower() == "false":
        return False
    try:
        return int(t)
    except ValueError:
        return t


def get_path(state: dict, path: str):
    """按 a.b.c 点路径取嵌套 dict 值;缺失返回 None。"""
    node: object = state
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_path(state: dict, path: str, value) -> None:
    """按 a.b.c 点路径写嵌套 dict 值(中间层缺失则建空 dict)。"""
    parts = path.split(".")
    node = state
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def resolve_rule(twin: TwinSpec, state: dict | None = None) -> tuple[dict, bool]:
    """确定性结算:precondition 命中 → effect,否则 else_effect。

    返回 (终态, 是否命中 if 分支);不修改传入状态。
    """
    rule = twin.raw.get("rule") or {}
    working = copy.deepcopy(state if state is not None
                            else twin.raw.get("initial_state") or {})
    executed = False
    matched = _PRECONDITION_RE.match(str(rule.get("precondition", "")))
    if matched:
        left = get_path(working, matched.group(1))
        executed = _norm(left) == _norm(_parse_literal(matched.group(2)))
    assignment = str(rule.get("effect") if executed
                     else rule.get("else_effect") or "")
    assigned = _ASSIGN_RE.match(assignment)
    if not assigned:
        raise ValueError(f"{twin.clip_id}: unparsable rule assignment {assignment!r}")
    set_path(working, assigned.group(1), _parse_literal(assigned.group(2)))
    return working, executed


def state_phrases(twin: TwinSpec, state: dict) -> list[str]:
    """visual_vocab × 状态表 → 短语列表(全表重申,来源可审计)。"""
    vocab = twin.raw.get("visual_vocab") or {}
    phrases: list[str] = []
    for path, options in vocab.items():
        value = get_path(state, str(path))
        key = value if isinstance(value, str) else ("true" if value else "false")
        phrase = options.get(key) or options.get(_norm(key))
        if phrase:
            phrases.append(str(phrase))
    return phrases


def _clean(text: str) -> str:
    """压空白并去掉方括号(PromptCompiler 禁止描述自带 shot 标记)。"""
    return " ".join(text.replace("[", " ").replace("]", " ").split())


def build_visual_script(twin: TwinSpec, state_before: dict,
                        state_after: dict) -> VisualScript:
    scene = twin.raw.get("initial_scene") or {}
    before = state_phrases(twin, state_before)
    after = state_phrases(twin, state_after)
    return VisualScript(
        style=_clean(f"{scene.get('style', '')}, cinematic game screenshot"),
        first_frame_anchor=(
            f"the scene is exactly as shown in <Picture 1>: "
            f"{_clean(str(scene.get('description', '')))}"
            + ("; " + "; ".join(before) if before else "")
            + "; every character stays in the same position and outfit as in "
              "the frame"
        ),
        camera=_CAMERA,
        actor="the player character",
        actor_appearance=_APPEARANCE,
        action=_clean(twin.shared_action),
        stability=_STABILITY,
        end_state="; ".join(after),
        meta={"clip_id": twin.clip_id, "case_id": twin.case_id, "side": twin.side,
              "family": twin.family, "prompt_mode": "framework"},
        use_first_frame=True,
        duration_s=5,
    )


def director_for(twin: TwinSpec):
    """VisualDirector 闭包:TurnResult → VisualScript(GameLoop 构造形态)。"""

    def direct(result, _visual_state: VisualState) -> VisualScript:
        return build_visual_script(twin, result.state_before, result.state_after)

    return direct


class Ref2vaPromptCompiler:
    """ref2va 首帧模式的 compiler 包装:标准 NEWSKILL 编译后,只把开头的
    I2VA 对齐行换成 ref2va 首帧锚定句,其余逐字节不动。

    为什么包装而不改 PromptCompiler:游戏/引擎链路(loop.py)只认 i2v 语义
    (image 参数即帧 0);ref2va 是评测层的渲染后端选择,引擎 prompt 组织
    不应被评测的消融需求牵动。
    """

    def __init__(self) -> None:
        self._inner = PromptCompiler()

    def compile(self, script: VisualScript, *, has_first_frame: bool,
                has_reference_video: bool = False) -> str:
        return apply_ref2va_anchor(self._inner.compile(
            script, has_first_frame=has_first_frame,
            has_reference_video=has_reference_video))


class CaseRuntime(GameRuntime):
    """把一份数据规格变成单回合游戏:shared_action → 规则结算。"""

    def __init__(self, twin: TwinSpec) -> None:
        self._twin = twin

    def initial_state(self) -> dict:
        state = copy.deepcopy(self._twin.raw.get("initial_state") or {})
        state[_DONE] = False
        return state

    def available_actions(self, state: dict) -> list[Action]:
        return [Action(id="shared_action", label=self._twin.shared_action,
                       description=self._twin.rule_text)]

    def resolve_turn(self, state: dict, action: Action, rng) -> TurnOutcome:
        end, executed = resolve_rule(self._twin, state)
        end = copy.deepcopy(end)
        end[_DONE] = True
        event = GameEvent(
            type="rule_execute" if executed else "rule_reject",
            subject=action.id,
            message=self._twin.rule_text,
            detail={"branch": "effect" if executed else "else_effect"},
        )
        return TurnOutcome(events=(event,), state_after=end)

    def status_of(self, state: dict) -> str:
        return STATUS_WON if state.get(_DONE) else STATUS_ONGOING

    def describe_state(self, state: dict) -> str:
        return "; ".join(state_phrases(self._twin, state))


def run_framework(renderer, twin: TwinSpec, run_dir: Path, *,
                  seed: int = 0, force: bool = False,
                  ref2va: bool = False) -> tuple[Path, bool]:
    """GameLoop 全链路渲染单回合;产物归位到评测布局并落盘 prompt。

    GameLoop 的原生产物在 <run_dir>/game/<clip>/ 下(turn-001.mp4 /
    turn_log.jsonl,保留作审计),归位副本写 videos/<clip>.mp4 与
    frames/<clip>.png 供判定阶段复用。ref2va=True 时换 Ref2vaPromptCompiler
    (对齐行 → ref2va 首帧锚定句),渲染请求走 ref2va 引擎。
    """
    video_path = video_path_for(run_dir, twin)
    if video_path.is_file() and video_path.stat().st_size > 0 and not force:
        return video_path, False
    scene = twin.raw.get("initial_scene") or {}
    game_dir = Path(run_dir) / "game" / twin.clip_id
    loop = GameLoop(
        runtime=CaseRuntime(twin),
        agent=ScriptedAgent(["shared_action"]),
        visual_director=director_for(twin),
        renderer=renderer,
        compiler=Ref2vaPromptCompiler() if ref2va else None,
        output_dir=game_dir,
        base_seed=seed,
        max_turns=1,
        render_failure_policy=RENDER_FAILURE_ABORT,
        visual_state=VisualState(style=str(scene.get("style", "")),
                                 current_frame=twin.input_image),
    )
    summary = loop.run()
    if len(summary.records) != 1:
        raise RuntimeError(f"{twin.clip_id}: expected 1 turn, got "
                           f"{len(summary.records)}")
    record = summary.records[0]
    if record.render_error or not record.video_path:
        raise RuntimeError(f"{twin.clip_id}: framework render failed: "
                           f"{record.render_error}")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record.video_path, video_path)
    if record.final_frame:
        frame_path = final_frame_for(run_dir, twin)
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record.final_frame, frame_path)
    prompts_dir = Path(run_dir) / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / f"{twin.clip_id}.txt").write_text(record.prompt,
                                                     encoding="utf-8")
    return video_path, True


__all__ = ["CaseRuntime", "Ref2vaPromptCompiler", "build_visual_script",
           "director_for", "get_path", "resolve_rule", "run_framework",
           "set_path", "state_phrases"]
