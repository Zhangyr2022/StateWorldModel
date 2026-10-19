"""开放世界 Demo 的视觉表现:TurnResult → VisualScript + 终局动画。

画面文本的核心来自导演 VLM:consequence(英文画面描述)就是下一段视频
的动作主体;shot2 非空时编译器会生成合法的多 shot 切镜。锚点始终锚定
"同一旅人、同一环境、无缝衔接",保证跨回合连续性。
"""

from __future__ import annotations

from pathlib import Path

from engine.runtime import STATUS_WON, TurnResult
from engine.visual.script import Shot, VisualScript
from engine.visual.state import VisualState
from game.openworld.state import MODE_AUTO, ExplorationState

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "openworld"

# 开场场景表(首帧图 + 中文名)
SCENES: dict[str, dict] = {
    "western": {"label": "西部边陲小镇", "image": "western.png"},
    "anime": {"label": "现代都市街头", "image": "anime.png"},
    "chinese": {"label": "古风城镇", "image": "chinese.png"},
    "japan": {"label": "东瀛街巷", "image": "japan.png"},
    "roman": {"label": "罗马古城", "image": "roman.png"},
    "gta": {"label": "罪恶都市", "image": "gta6.png"},
}

STYLE_HINTS = (
    "cinematic third-person open-world adventure game, epic scale, "
    "detailed environment, filmic color grade"
)

CAMERA = (
    "Third-person camera behind the young explorer's shoulder, mostly steady "
    "with subtle handheld follow, keeping the explorer in frame as the "
    "environment unfolds"
)

EXPLORER_APPEARANCE = (
    "the same young explorer from the frame, unchanged in appearance and outfit"
)


def scene_image_path(scene_id: str) -> Path | None:
    entry = SCENES.get(scene_id)
    if entry is None:
        return None
    path = ASSETS_DIR / entry["image"]
    return path if path.exists() else None


def initial_openworld_visual_state(scene_id: str = "western") -> VisualState:
    entry = SCENES.get(scene_id, SCENES["western"])
    return VisualState(
        scene=f"an open-world adventure starting in the {entry['label']} scene",
        style=STYLE_HINTS,
        appearances={"旅人": EXPLORER_APPEARANCE},
        meta={"scene": scene_id},
    )


def build_openworld_visual_script(turn_result: TurnResult, visual_state: VisualState) -> VisualScript:
    """把一幕探索翻译成约 5 秒的演出;画面文本来自导演 VLM 的 consequence。

    自动探索回合(见 game/openworld/rules.py)的 end_state 明确
    落在一个"尚未被处理的机会"上,为下一回合的交互制造决策点。
    """
    state: ExplorationState = turn_result.state_after
    consequence_event = next(
        (e for e in turn_result.events if e.type == "consequence"), None
    )
    detail = consequence_event.detail if consequence_event is not None else {}
    consequence = str(detail.get("consequence") or state.last_consequence or
                      "The young explorer continues onward through the landscape.")
    shot2_text = str(detail.get("shot2") or "")
    cut_at_s = float(detail.get("cut_at_s") or 2.5)

    shots: tuple[Shot, ...] = ()
    if shot2_text:
        shots = (Shot(cut_at_s=cut_at_s, text=shot2_text),)

    end_state = (
        "the explorer pauses before a clear opportunity they have noticed "
        "but not yet acted on — a doorway, a stranger, a fork in the road "
        "or an object worth a decision, left unresolved for the player"
        if state.last_mode == MODE_AUTO else
        "the explorer comes to rest at a new vantage point, the journey "
        "obviously continuing beyond the frame"
    )

    return VisualScript(
        style=visual_state.style,
        first_frame_anchor=(
            "the young explorer shown in <Picture 1> continues seamlessly from "
            "the frame, same appearance, outfit and environment, the world "
            "around them alive and moving"
        ),
        camera=CAMERA,
        actor="the young explorer",
        actor_appearance=EXPLORER_APPEARANCE,
        action=consequence,
        effects=("living environment", "atmospheric depth"),
        end_state=end_state,
        soundscape=(
            "Wind, footsteps and distant life of the settlement or wilderness "
            "around the explorer"
        ),
        music=(
            "A sweeping adventure orchestral pad, moderate tempo, growing "
            "slightly with discovery"
        ),
        shots=shots,
        meta={"scene": state.scene_id},
        duration_s=5,
    )


def ending_openworld_script(status: str, visual_state: VisualState) -> VisualScript | None:
    """终局动画:剧情到达高潮时由导演的 climax 文本收束;其余不生成。"""
    if status != STATUS_WON:
        return None
    state: ExplorationState | None = visual_state.meta.get("exploration_state")
    climax = (state.climax_text if state else "") or (
        "The explorer's journey reaches its breathtaking climax."
    )
    return VisualScript(
        style=visual_state.style,
        first_frame_anchor=(
            "the young explorer shown in <Picture 1> at the heart of the "
            "current scene, unchanged in appearance, the whole environment "
            "gathered around this moment"
        ),
        camera=(
            "The camera holds a locked-off hero shot with a slow tilt up, no "
            "cut, letting the climax breathe"
        ),
        actor="the young explorer",
        actor_appearance=EXPLORER_APPEARANCE,
        action=(
            "stands at the center of the final moment of the story as the "
            "world reacts around them, every thread of the journey visibly "
            "coming together"
        ),
        effects=("golden hour light", "sweeping atmosphere", "grand scale"),
        end_state=(
            "the story closes on the explorer's silhouette against the horizon, "
            "the adventure complete"
        ),
        soundscape="The world falls quiet, one last breath of wind, then stillness",
        music=(
            "The main adventure theme returns in full orchestra for one final "
            "statement, ending on a warm resolved chord"
        ),
        shots=(),
        meta={"scene": visual_state.meta.get("scene", "western")},
        duration_s=5,
    )
