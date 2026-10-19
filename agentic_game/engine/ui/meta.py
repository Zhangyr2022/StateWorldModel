"""GameMeta:大厅封面墙与详情面板的展示元数据(纯数据,不依赖 gradio)。

registry 只声明 game_id / title / category / needs_llm / build;界面还需要
封面图、一句话简介、语言、皮肤声明的回合数等"卡片文案"。本模块只读扫描
game/skins/*.json(registry 本身也扫描该目录)把这些信息取出来——不修改
任何游戏层对象,缺什么就静默降级什么(封面缺失 → 占位图;简介缺失 → 空串)。

封面必须来自皮肤 JSON 的 frame 字段经 resolve_frame 解析,不允许按 game_id
猜文件名(blood_tide_leviathan_machine 的封面是 assets/duel/duel_leviathan.png,
文件名对不上);openworld 不是皮肤,用固定的场景画当封面,也不 import
game.openworld(scene→文件名的映射属于游戏层知识)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from engine.parts import REPO_ROOT, resolve_frame

SKIN_DIR = REPO_ROOT / "game" / "skins"

# openworld 不是皮肤:用默认场景(western)的画面当封面。
OPENWORLD_COVER = REPO_ROOT / "assets" / "openworld" / "western.png"
OPENWORLD_BLURB = ("扮演一名旅人,由 VLM 导演实时作画:没有固定剧本,"
                   "想到什么就去做什么,画面平淡时自动探索。")
OPENWORLD_TITLE_EN = "Open World Exploration (VLM Director)"
OPENWORLD_BLURB_EN = ("Play a traveler while a VLM director paints the world live: "
                      "no fixed script, go wherever you like; quiet moments "
                      "auto-explore.")

CATEGORY_LABELS = {"adv": "剧情 ADV", "machine": "对决 DUEL",
                   "openworld": "开放世界", "romance": "恋爱"}
FILTER_CHOICES = ("全部", "剧情 ADV", "对决 DUEL", "开放世界")
FILTER_TO_CATEGORY = {"全部": None, "剧情 ADV": "adv",
                      "对决 DUEL": "machine", "开放世界": "openworld"}
# 界面语言 → 筛选/类别的展示名(取值仍用上面的中文键,保持稳定)
FILTER_LABELS_EN = {"全部": "All", "剧情 ADV": "Story ADV",
                    "对决 DUEL": "Duel", "开放世界": "Open World"}
CATEGORY_LABELS_EN = {"adv": "Story ADV", "machine": "DUEL",
                      "openworld": "Open World", "romance": "Romance"}

# 界面语言:"zh" | "en"
LANGS = ("en", "zh")

# 开放世界场景卡/详情/对局标题的展示名(纯表现层;未收录的场景按 id 首字母大写)
SCENE_TITLES = {
    "western": ("西部", "Western"),
    "anime": ("动漫", "Anime"),
    "chinese": ("古风", "Chinese"),
    "japan": ("日本", "Japan"),
    "roman": ("罗马", "Roman"),
    "gta": ("GTA", "GTA"),
}

# 简介最长字数(皮肤 persona 往往是整段人设,卡片上只放首句)。
_BLURB_MAX = 72


@dataclass(frozen=True)
class GameMeta:
    """一个游戏在大厅展示所需的全部文案(全部可降级)。"""

    game_id: str
    title: str                      # registry 展示名("id · 名")
    short_title: str                # 去掉 "id · " 前缀的展示名
    category: str
    cover: Path | None              # 封面图绝对路径;None = 无图
    blurb: str                      # 一句话简介;"" = 无
    language: str                   # "中文" | "EN"
    max_turns: int | None           # 皮肤声明的回合上限;None = 跟随开局设置
    needs_llm: bool
    scene_choices: tuple = ()       # 场景类游戏(registry options)的可选场景
    title_en: str = ""              # 英文界面的标题;"" = 沿用 short_title
    blurb_en: str = ""              # 英文界面的简介;"" = 沿用 blurb


def load_game_meta(skin_dir: Path | None = None, games=None) -> dict[str, GameMeta]:
    """扫描皮肤目录 + 注册表,产出 game_id → GameMeta(注册表全量覆盖)。

    skin_dir / games 可注入以便离线测试;单个皮肤 JSON 坏掉只影响自己
    (该游戏封面/简介降级,不抛错)——registry 在启动时已对坏皮肤告警过。
    """
    if games is None:
        from game.registry import GAMES
        games = GAMES
    skin_dir = Path(skin_dir) if skin_dir is not None else SKIN_DIR
    # 与 registry 同一索引方式:按文件内的 game_id(而非文件名——
    # duel_blood_tide_leviathan_machine.json 里的 id 就不带 duel_ 前缀)。
    skins = _scan_skins(skin_dir)

    metas: dict[str, GameMeta] = {}
    for game_id, entry in games.items():
        data = skins.get(game_id) or {}
        scene_choices = next(
            (tuple(option.choices) for option in entry.options
             if option.name == "scene"), ())
        metas[game_id] = GameMeta(
            game_id=game_id,
            title=entry.title,
            short_title=short_title(entry.title),
            category=entry.category,
            cover=_cover_of(game_id, data),
            blurb=blurb_from_skin(data, entry.category),
            language=("EN" if game_id.endswith("_en") or entry.category == "openworld"
                      else "中文"),
            max_turns=_int_or_none(data.get("max_turns")),
            needs_llm=entry.needs_llm,
            scene_choices=scene_choices,
            title_en=_title_en(game_id, entry, data),
            blurb_en=_blurb_en(entry.category, data),
        )
    return metas


def scene_title(scene: str, lang: str = "zh") -> str:
    """开放世界场景的展示名(如 gta → "GTA")。"""
    pair = SCENE_TITLES.get(scene)
    if pair is None:
        return scene.replace("_", " ").title()
    return pair[1] if lang == "en" else pair[0]


def display_title(meta: GameMeta, lang: str = "zh", scene: str | None = None) -> str:
    """按界面语言取标题(英文缺失时回落中文短标题)。

    场景类游戏(开放世界)以场景名作标题:scene 缺省取第一个场景;
    DUEL 只显示 "A VS B"(去掉皮肤标题里的括注与 " · " 副标题)。
    """
    if meta.scene_choices:
        return scene_title(str(scene or meta.scene_choices[0]), lang)
    if lang == "en":
        return meta.title_en or meta.short_title
    if meta.category == "machine":
        return duel_title(meta.short_title)
    return meta.short_title


def duel_title(title: str) -> str:
    """"深渊猎手 VS 星渊巨鲸(状态机:深海远程决斗)" → "深渊猎手 VS 星渊巨鲸"。"""
    title = re.sub(r"\s*[（(][^（）()]*[）)]", "", title)
    return title.split(" · ", 1)[0].strip()


def display_blurb(meta: GameMeta, lang: str = "zh") -> str:
    """按界面语言取简介(英文缺失时回落原简介)。"""
    return (meta.blurb_en or meta.blurb) if lang == "en" else meta.blurb


def category_label(category: str, lang: str = "zh") -> str:
    table = CATEGORY_LABELS_EN if lang == "en" else CATEGORY_LABELS
    return table.get(category, category)


def filter_label(key: str, lang: str = "zh") -> str:
    return FILTER_LABELS_EN.get(key, key) if lang == "en" else key


def visible_in(meta: GameMeta, lang: str | None) -> bool:
    """该游戏在某界面语言的大厅里是否出现(lang=None 不过滤)。

    剧情 ADV 有中/英两版皮肤,只列与界面语言一致的那版;对决 DUEL 皮肤
    是中文,英文界面经 hud_i18n 词表出英文标题/HUD,两种语言都列;开放
    世界只支持英文,两种语言都列(界面标"仅英文")。
    """
    if lang is None or meta.category != "adv":
        return True
    return (meta.language == "EN") == (lang == "en")


def counterpart_id(game_id: str, metas: dict[str, GameMeta], lang: str) -> str:
    """切换界面语言时,把已选游戏换成该语言的同款(X ↔ X_en);无则原样。"""
    meta = metas.get(game_id)
    if meta is None or visible_in(meta, lang):
        return game_id
    other = game_id[:-3] if game_id.endswith("_en") else f"{game_id}_en"
    return other if other in metas and visible_in(metas[other], lang) else ""


def short_title(title: str) -> str:
    """registry 的展示名形如 "<id> · <名>",取 " · " 之后的段。"""
    return title.split(" · ", 1)[-1] if " · " in title else title


def blurb_from_skin(data: dict | None, category: str) -> str:
    """皮肤数据 → 一句话简介;缺字段/坏数据返回 ""。"""
    if not isinstance(data, dict):
        return ""
    if category == "adv":
        text = _first_sentence(str(data.get("girl_persona") or "")) \
            or _first_sentence(str(data.get("player_scenario") or ""))
        return text
    if category == "machine":
        player = str(data.get("player_name") or "")
        boss = data.get("boss")
        boss_name = str(boss.get("name") or "") if isinstance(boss, dict) else ""
        if player and boss_name:
            return f"{player} VS {boss_name}"
        return player or boss_name
    if category == "openworld":
        return OPENWORLD_BLURB
    return ""


def gallery_items(metas: dict[str, GameMeta], filter_label: str = "全部",
                  lang: str | None = None) -> tuple[list[tuple[str, str]], list[dict]]:
    """筛选后的封面墙条目:([(封面路径, 短标题)...], [条目...]) 平行列表。

    条目 = {"game_id": ..., "scene": ...}(scene 仅场景类游戏有值,其余 None)。
    场景类游戏(openworld)按 scene_choices 展开成每场景一张卡——用户在
    assets/openworld 放了几张场景画,大厅就有几张卡。封面缺失的游戏跳过
    (界面另有按名称选择的下拉兜底);顺序跟随注册表。
    lang 非空时按界面语言过滤(见 visible_in)并用该语言的标题作卡片文字。
    """
    images: list[tuple[str, str]] = []
    ids: list[dict] = []
    for game_id, meta in metas.items():
        if not _filter_ok(meta, filter_label) or not visible_in(meta, lang):
            continue
        title = display_title(meta, lang or "zh")
        if meta.scene_choices:
            for scene in meta.scene_choices:
                cover = scene_cover(str(scene))
                if cover is None:
                    continue
                images.append((str(cover), scene_title(str(scene), lang or "zh")))
                ids.append({"game_id": game_id, "scene": str(scene)})
            continue
        if meta.cover is None or not Path(meta.cover).exists():
            continue
        images.append((str(meta.cover), title))
        ids.append({"game_id": game_id, "scene": None})
    return images, ids


def scene_cover(choice: str) -> Path | None:
    """场景画封面:assets/openworld/<choice>.png;无则匹配去尾数字同名
    (gta → gta6.png)。该目录是场景素材的唯一来源,直接扫描。"""
    directory = OPENWORLD_COVER.parent
    exact = directory / f"{choice}.png"
    if exact.exists():
        return exact
    for path in sorted(directory.glob("*.png")):
        if path.stem.rstrip("0123456789") == choice:
            return path
    return None


# -- 内部 ----------------------------------------------------------------------

def _title_en(game_id: str, entry, data: dict | None) -> str:
    """英文标题:openworld 固定;英文 ADV 皮肤去掉 " [EN]" 尾标;
    DUEL 用词表翻译双方名后取逗号前的称呼(如 "Keira vs. Tarasos")。"""
    if entry.category == "openworld":
        return OPENWORLD_TITLE_EN
    title = short_title(entry.title)
    if game_id.endswith("_en"):
        return title.removesuffix(" [EN]").strip()
    if entry.category == "machine":
        player, boss = _duel_names_en(data)
        if player and boss:
            return f"{player.split(',')[0]} vs. {boss.split(',')[0]}"
    return ""


def _blurb_en(category: str, data: dict | None) -> str:
    if category == "openworld":
        return OPENWORLD_BLURB_EN
    if category == "machine":
        player, boss = _duel_names_en(data)
        if player and boss:
            return f"{player} VS {boss}"
    return ""


def _duel_names_en(data: dict | None) -> tuple[str, str]:
    """DUEL 双方名经 hud_i18n 词表译成英文;仍含中文(词表未收录)视为无。"""
    from engine.hud_i18n import tr

    if not isinstance(data, dict):
        return "", ""
    boss = data.get("boss")
    names = (str(data.get("player_name") or ""),
             str(boss.get("name") or "") if isinstance(boss, dict) else "")
    out = tuple(tr(name, "en") for name in names)
    if any(not name or any("一" <= ch <= "鿿" for ch in name) for name in out):
        return "", ""
    return out


def _scan_skins(skin_dir: Path) -> dict[str, dict]:
    """目录内全部皮肤 JSON → {文件内 game_id: 数据};坏文件跳过。"""
    skins: dict[str, dict] = {}
    if not skin_dir.is_dir():
        return skins
    for path in sorted(skin_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("game_id"):
            skins[str(data["game_id"])] = data
    return skins


def _cover_of(game_id: str, data: dict | None) -> Path | None:
    """皮肤 frame 字段 → 封面绝对路径;openworld 用固定场景画。"""
    if game_id == "openworld":
        return OPENWORLD_COVER if OPENWORLD_COVER.exists() else None
    if isinstance(data, dict) and data.get("frame"):
        frame = resolve_frame(data["frame"])
        if frame is not None and frame.exists():
            return frame
    return None


def _first_sentence(text: str) -> str:
    """取第一句话(。/!.? 处截断),压平换行,超长截断加省略号。"""
    text = " ".join(text.split())
    if not text:
        return ""
    for sep in ("。", ".", "!", "?", "!", "?"):
        idx = text.find(sep)
        if idx != -1:
            text = text[:idx + 1]
            break
    if len(text) > _BLURB_MAX:
        text = text[:_BLURB_MAX - 1] + "…"
    return text


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filter_ok(meta: GameMeta, filter_label: str) -> bool:
    category = FILTER_TO_CATEGORY.get(filter_label)
    return category is None or meta.category == category
