"""游戏注册表:所有可选游戏的唯一清单。

run_demo 与 run_ui(Gradio)都不再硬编码游戏列表——它们面向本注册表
工作:枚举下拉选项、按 needs_llm 决定是否连接 LLM 服务、把 UI 选项
(options)透传给 build。

上线一个游戏的三种方式:
1. JSON 皮肤(设计师首选):放一个 .json 进 game/skins/,自动扫描上线;
2. 模板皮肤(Python):game/<id>/ 提供 SKIN + build_game,在下面登记一行;
3. 全新机制:game/<id>/ 实现完整 GameRuntime 五文件,在下面登记一行。

category 用于 UI 分组:machine(本地结算,战斗等)/ adv(
固定剧情骨架 + 自由输入 + LLM 女主回应)/ openworld(VLM 导演探索)。
(romance 类——LLM 参与结算的对话——不随公开版分发,扫描到即跳过。)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class GameOption:
    """一个游戏级 UI 选项(透传给 build 的关键字参数)。"""

    name: str        # build 的关键字参数名
    label: str       # UI 展示名
    choices: tuple   # 可选值
    default: str = ""


@dataclass(frozen=True)
class GameEntry:
    """一个可选游戏的登记项。"""

    game_id: str
    title: str                # UI 展示名
    category: str             # duel / romance / openworld
    needs_llm: bool           # True = 组装前需准备常驻 LLMClient
    build: Callable           # (llm=None, **options) -> GameParts
    options: tuple = field(default_factory=tuple)  # (GameOption, ...)


def _romance_entry(game_id: str, title: str) -> GameEntry:
    """romance 类皮肤游戏:LLM 参与结算。"""
    return GameEntry(game_id, title, "romance", True,
                     build=import_module(f"game.{game_id}").build_game)


# (game_id, 展示名);顺序即 UI 下拉框顺序。
# 战斗玩法全部由 game/skins/*.json(状态机)提供。
# romance 类(手写基础示例 + romance_theme 模板 + 6 个皮肤)不随公开版分发,
# 此元组恒为空,仅为保留 ROMANCE 分类的登记形态。
_ROMANCE_GAMES: tuple = ()


def _build_registry() -> dict[str, GameEntry]:
    registry: dict[str, GameEntry] = {}
    for game_id, title in _ROMANCE_GAMES:
        registry[game_id] = _romance_entry(game_id, title)
    registry["openworld"] = GameEntry(
        game_id="openworld",
        title="openworld · 开放世界探索(VLM 导演)",
        category="openworld",
        needs_llm=True,
        build=import_module("game.openworld").build_game,
        options=(
            GameOption(
                name="scene",
                label="开场场景",
                # 取值必须与 game.openworld.visual.SCENES 的键一致(gta6 是
                # 图片文件名,场景 id 是 gta——此前写成 gta6 会在开局时 KeyError)
                choices=("western", "anime", "chinese", "japan", "roman", "gta"),
                default="western",
            ),
        ),
    )
    for game_id, entry in _json_skin_entries().items():
        if game_id in registry:
            print(f"[registry] 警告: JSON 皮肤 {game_id} 与既有游戏重名,已跳过"
                  f"({entry.title})", file=sys.stderr)
            continue
        registry[game_id] = entry
    return registry


# 设计师交付的 JSON 皮肤目录:放一个 .json 进来,重启后即出现在游戏清单。
# 只扫描本层(不递归;下线一款皮肤 = 把 JSON 移出本层);
# 单个文件不合法只跳过该文件,不影响其他游戏。
_JSON_SKIN_DIR = Path(__file__).resolve().parent / "skins"


def _json_skin_entries() -> dict[str, GameEntry]:
    from game.machine_theme import build_machine_game
    from game.machine_theme.json_skin import load_json_skin as load_machine_skin
    from game.skin_validation import SkinValidationError

    entries: dict[str, GameEntry] = {}
    if not _JSON_SKIN_DIR.is_dir():
        return entries
    for path in sorted(_JSON_SKIN_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            category = str(data.get("category", ""))
            if category == "adv":
                from game.adv_theme import build_adv_game
                from game.adv_theme.json_skin import load_json_skin as load_adv_skin
                skin = load_adv_skin(path)
                title = str(data.get("title", skin.game_id))
                entries[skin.game_id] = GameEntry(
                    skin.game_id, f"{skin.game_id} · {title}", "adv", True,
                    build=lambda llm=None, s=skin: build_adv_game(s, llm))
            elif category == "romance":
                # romance 类不随公开版分发(模板与皮肤均已移除);皮肤目录里
                # 不应有 romance 皮肤,真出现时给出可操作的告警而非崩溃。
                print(f"[registry] 跳过 {path.name}: romance 类不随公开版分发",
                      file=sys.stderr)
                continue
            elif category == "machine":
                skin = load_machine_skin(path)
                title = str(data.get("title", skin.game_id))
                entries[skin.game_id] = GameEntry(
                    skin.game_id, f"{skin.game_id} · {title}", "machine", False,
                    build=lambda llm=None, s=skin: build_machine_game(s))
            else:
                print(f"[registry] 跳过 {path.name}: category 必须是 "
                      f"machine/adv,当前 {category!r}", file=sys.stderr)
        except (SkinValidationError, json.JSONDecodeError) as exc:
            print(f"[registry] 跳过无效皮肤 {path.name}: {exc}", file=sys.stderr)
    return entries


GAMES: dict[str, GameEntry] = _build_registry()
GAME_IDS: tuple[str, ...] = tuple(GAMES)  # UI 下拉框顺序:duel → romance → openworld


def get(game_id: str) -> GameEntry:
    """按 id 取游戏登记项;未知 id 报错并列出可选项。"""
    try:
        return GAMES[game_id]
    except KeyError:
        raise KeyError(
            f"unknown game: {game_id!r} (available: {', '.join(GAME_IDS)})"
        ) from None


__all__ = ["GAMES", "GAME_IDS", "GameEntry", "GameOption", "get"]
