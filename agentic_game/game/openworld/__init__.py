"""第三人称开放世界探索 Demo:VLM 担任评估者与导演,双回合模式探索。

循环(hook 架构,设计见本包各模块 docstring):
1. VLM 观看当前画面(首图或上一段视频的头/中/尾 3 帧),判断是否存在
   有意义的行动机会:
   - 有 → INTERACTIVE 回合:提出 3 个行动选项,玩家点选 / CLI 序号 /
     自定义输入;
   - 无(且上一回合不是 AUTO)→ AUTO 回合:不弹出选择,AI 按自动探索
     规则推进一幕;硬性规则:AUTO 之后的下一回合必须回到 INTERACTIVE。
2. VLM 作为导演输出:剧情旁白(中文)+ 下一段视频的画面描述(英文,
   可含第二个镜头)→ VisualScript → 渲染;
3. post_render 钩子取新视频的 3 帧,回到 1。

主线节奏只数交互幕(AUTO 不替玩家推进主线);多次探索(≥12 交互幕)后
导演可宣布剧情完结(plot_complete)→ 好结局 + 结束动画。本游戏允许多
shot:进入新场景时导演可以直接硬切([Shot 2])。
"""

from engine.parts import GameParts
from game.openworld.actions import ACTION_AUTO, ACTION_CUSTOM
from game.openworld.agents import OpenworldAutoAgent, OpenworldCliAgent
from game.openworld.rules import OpenworldRuntime
from game.openworld.state import ExplorationState
from game.openworld.visual import (
    ASSETS_DIR,
    SCENES,
    build_openworld_visual_script,
    ending_openworld_script,
    initial_openworld_visual_state,
    scene_image_path,
)

# 回放终局字幕(Noto Sans CJK 中有字形)
ENDING_CAPTIONS = {"won": "◆ 剧终 ◆"}


# 回放 HUD 旁白显示上限(显示宽度单位:ASCII 1 / CJK 2)。左下面板描述区
# 宽 490px、15px 字号下约容 68 个英文字符或 32 个汉字;取 64 保证连同
# "..." 单行放下。规则层存档的 story 本身在 80 字处硬截断(常断在词中),
# 这里统一按词边界截短并以 "..." 收尾。
HUD_STORY_MAX = 64


def clip_hud_text(text: str, limit: int = HUD_STORY_MAX) -> str:
    """把旁白截到 limit 个显示单位以内;截断时退到词边界并加 "..."。"""
    text = " ".join((text or "").split())

    def width(s: str) -> int:
        return sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)

    if width(text) <= limit:
        return text
    budget = limit - 3
    cut, used = 0, 0
    for i, ch in enumerate(text):
        used += 2 if ord(ch) > 0x2E7F else 1
        if used > budget:
            break
        cut = i + 1
    head = text[:cut]
    # 英文退到最后一个空格,避免断词;中文(无空格)直接按字截
    if cut < len(text) and text[cut] != " " and " " in head:
        head = head[:head.rindex(" ")]
    return head.rstrip(" ,.;:!?,。;:!?、—-\"'“”") + "..."


def _openworld_hud(state, events=()) -> dict:
    """探险回放 HUD:右上当前地点/体力/主线进度;底部行动与最新旁白。"""
    from game.openworld.quests import quest_for
    from game.openworld.state import MODE_AUTO

    quest = quest_for(state.scene_id)
    stage = min(state.quest_stage, quest.total - 1)
    location = clip_hud_text(state.scene_name, 26)
    stamina = getattr(state, "stamina", 100)
    rows = [("text", f"地点:{location or quest.title}"),
            ("bar", stamina / 100.0, "stamina", f"体力 {stamina}/100"),
            ("pips", min(state.quest_stage, quest.total), quest.total,
             f"主线《{quest.title}》 {stage + 1}/{quest.total}")]
    if getattr(state, "exhausted_turns", 0) > 0:
        rows.insert(1, ("text", "状态:力竭——只能缓慢走动"))
    auto = getattr(state, "last_mode", "") == MODE_AUTO
    if auto:
        rows.insert(1, ("text", "本幕自动探索 · 下一回合由你决定"))
    return {
        "theme": "openworld",
        "tag": f"第{state.turn}幕 · {state.scene_name}" + (" · 自动探索" if auto else ""),
        "right_name": "旅行者",
        "right_rows": rows,
        "action_desc": clip_hud_text(state.story[-1]) if state.story else "冒险刚刚开始",
    }


def build_game(llm=None, *, scene: str = "western") -> GameParts:
    """组装全部游戏部件(llm:engine.llm.LLMClient,常驻服务)。"""
    if llm is None:
        raise ValueError("openworld 需要常驻 LLM 服务(先启动 engine.llm.service)")
    entry = SCENES[scene]
    runtime = OpenworldRuntime(
        llm,
        scene_image=str(ASSETS_DIR / entry["image"]),
        scene_name=entry["label"],
        world_id=scene,
    )
    return GameParts(
        runtime=runtime,
        visual_director=build_openworld_visual_script,
        visual_state=initial_openworld_visual_state(scene),
        ending_director=ending_openworld_script,
        post_render=lambda record: runtime.on_video_rendered(record.index, record.video_path),
        frame=f"assets/openworld/{entry['image']}",
        describe_state=runtime.describe_state,
        free_text_action=ACTION_CUSTOM,
        # 声明式自动行动:AUTO 回合是唯一可选行动时,UI 不弹选择直接播放
        auto_action=ACTION_AUTO,
        scene_labels={sid: item["label"] for sid, item in SCENES.items()},
        ending_captions=dict(ENDING_CAPTIONS),
        auto_agent_factory=lambda _llm=None: OpenworldAutoAgent(),
        hud_of=_openworld_hud,
        cli_agent_factory=lambda _llm=None: OpenworldCliAgent(),
    )


__all__ = [
    "OpenworldRuntime",
    "ExplorationState",
    "OpenworldCliAgent",
    "OpenworldAutoAgent",
    "ACTION_AUTO",
    "build_game",
    "build_openworld_visual_script",
    "ending_openworld_script",
    "initial_openworld_visual_state",
    "scene_image_path",
]
