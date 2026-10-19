"""HUD 文案中英对照(回放视频叠层用)。

默认英文;设环境变量 AGENTIC_GAME_HUD_LANG=zh 后,montage 绘制的所有
HUD 文本原样保留中文。未收录的词原样透传,因此新增皮肤不影响。
调用方可用 `with hud_lang("zh"|"en"):` 临时覆盖本线程(上下文)的语言
(界面按玩家所选语言出 HUD),或给 tr() 直接传 lang。

覆盖五类:
1. 固定标签(敌方行动/心动变化/冒险见闻/心情 等);
2. 模板句(第N回合/第N幕、好感±N、主线《X》等,正则规则);
3. 游戏专属词条(9 款 duel 皮肤 + duel_oracle 技能名与双方名、恋爱心情、
   adv 剧情措辞、终局字幕);
4. 兜底:剩余含中文字符的字符串原样保留。

皮肤专属词条按皮肤分组维护;校验见 tests/test_hud_i18n.py(全部 duel
皮肤的动作名/描述/双方名经 tr() 后不得残留中文)。
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar

# 默认英文;如需中文 HUD,设 AGENTIC_GAME_HUD_LANG=zh
LANG = os.environ.get("AGENTIC_GAME_HUD_LANG", "en").strip().lower()

# 上下文级覆盖(None = 跟随 LANG);见 hud_lang()
_LANG_OVERRIDE: ContextVar[str | None] = ContextVar("hud_lang", default=None)


def current_lang() -> str:
    """当前生效的 HUD 语言("en" | "zh")。"""
    return _LANG_OVERRIDE.get() or LANG


@contextmanager
def hud_lang(lang: str | None):
    """在 with 块内把 HUD 语言切到 lang(None 表示不覆盖,沿用外层)。"""
    if not lang:
        yield
        return
    token = _LANG_OVERRIDE.set(lang.strip().lower())
    try:
        yield
    finally:
        _LANG_OVERRIDE.reset(token)

_EXACT = {
    # -- 固定标签 --------------------------------------------------------
    "敌方行动": "Enemy Action",
    "心动变化": "Affection",
    "冒险见闻": "Field Notes",
    "观察战场,按兵不动": "Holding position, watching",
    "冒险刚刚开始": "The adventure has just begun",
    # -- machine(状态机)HUD 固定措辞 ------------------------------------
    "Boss 按兵不动": "Boss holds position",
    "Boss盾": "Boss Shield",
    "狂暴": "Enraged",
    "蓄力中": "Charging",
    # machine 状态名(describe_state / 对局面板"当前局势")
    "对峙": "Standoff",
    "深渊对峙": "Abyssal Standoff",
    "踏潮对峙": "Tide-Treading Standoff",
    "狙击对峙": "Sniper Standoff",
    "月下对峙": "Moonlit Standoff",
    "远猎对峙": "Long-Range Hunt Standoff",
    # -- adv(剧情)HUD 固定措辞 ------------------------------------------
    "心意接近摊牌": "close to confession",
    "开始习惯你": "getting used to you",
    "礼貌的同学": "a polite classmate",
    "疏远客气": "distant and polite",
    "似乎更近了一些": "seems a little closer",
    "似乎疏远了一些": "seems a little more distant",
    "维持原样": "unchanged",
    # adv 心情词(hud_adv 的 _ADV_MOOD_LABELS 中转后在此落英文)
    "平静": "Calm",
    "温软": "Warm",
    "戒备": "Guarded",
    "疲惫": "Tired",
    "低落": "Sad",
    "慌乱": "Flustered",
    # -- duel_oracle 双方名 ----------------------------------------------
    "失格神谕者·伊芙": "Eve, the Fallen Oracle",
    "六翼审判者·塞拉菲姆": "Seraphim, the Six-Winged Arbiter",
    # -- duel_moon_dragon_swordsman 双方名 -------------------------------
    "月下孤剑·凌霄": "Lingxiao, the Lone Sword Beneath the Moon",
    "苍鳞天龙·朔渊": "Shuoyuan, the Azure-Scaled Celestial Dragon",
    # -- duel_blood_tide_leviathan_machine 双方名 ------------------------
    "赤潮断界者·烬璃": "Jinli, the Red-Tide Sunderer",
    "渊海吞星兽·摩伽": "Moga, the Star-Devouring Abyssal Beast",
    # -- duel_abyssal_leviathan_machine_v2 双方名 ------------------------
    "深渊猎手·凯拉": "Keira, the Abyssal Hunter",
    "星渊巨鲸·塔拉索斯": "Tarasos, the Star-Abyss Leviathan",
    # -- duel_eclipse_devourer_epic 双方名 -------------------------------
    "星蚀秘仪师·阿斯特拉": "Astra, the Eclipse Occultist",
    "万眼噬界者·阿波菲斯": "Apophis, the Myriad-Eyed World Devourer",
    # -- duel_last_human_colossus 双方名 ---------------------------------
    "末日狙击手·灰隼": "Gray Falcon, the Doomsday Marksman",
    "赤环天启机神·欧米伽": "Omega, the Crimson-Ring Apocalypse God-Machine",
    # -- duel_nine_heavens_artificer 双方名 ------------------------------
    "浪客机关师·墨弦": "Moxian, the Wandering Artificer",
    "九天镇狱神像": "The Nine-Heavens Prison-Guardian Idol",
    # -- duel_shogun_mask_palace_epic 双方名 -----------------------------
    "绯樱狐巫姬·凛": "Rin, the Crimson-Sakura Fox Shrine Maiden",
    "百面魔将军·胧正": "Kagemasa, the Hundred-Mask Oni General",
    # -- duel_worldspine_hunter 双方名 -----------------------------------
    "天脊猎神者·卡恩": "Kahn, the Worldspine Huntsman",
    "沉眠山龙·奥罗冈": "Orogon, the Slumbering Mountain Dragon",
    # -- 资源名(mp 上限标签)---------------------------------------------
    "圣言": "Oracle Word",
    "剑气": "Sword Qi",
    "潮息": "Tide Breath",
    "渊息": "Abyss Breath",
    "星蚀": "Star Eclipse",
    "源能": "Source Energy",
    "机关": "Machina",
    "灵符": "Ofuda",
    "猎灵": "Hunt Spirit",
    # -- duel_oracle 技能名 ----------------------------------------------
    "神谕·第一句": "Oracle: First Sentence",
    "禁典·红日坠落": "Forbidden Scripture: Falling Red Sun",
    "悖论·零度圣痕": "Paradox: Zero-Degree Stigma",
    "不可侵犯之环": "Inviolable Ring",
    "终末之门": "Gate of the End",
    "神说:尚未终结": "God Said: Not Yet Ended",
    # -- duel_moon_dragon_swordsman 技能名 --------------------------------
    "月痕·横断天河": "Moon Scar: River-Cleaving Arc",
    "枫烬·万叶穿云": "Maple Embers: Ten Thousand Leaves Through Clouds",
    "天隙·一星贯月": "Sky Rift: One Star Pierces the Moon",
    "镜湖·无波界": "Mirror Lake: Waveless Boundary",
    "霜轮·寒月结界": "Frost Wheel: Cold-Moon Ward",
    "归月·清辉返照": "Returning Moon: Clear Radiance Reflected",
    # -- duel_blood_tide_leviathan_machine 技能名 -------------------------
    "断潮·横海": "Tide Sunder: Sea-Cleaving Cut",
    "血月·坠刃": "Blood Moon: Falling Blades",
    "白瀑·逆鳞": "White Cascade: Uprooted Scale",
    "铁潮·镇海": "Iron Tide: Sea-Calming Bulwark",
    "白沫·息渊": "White Foam: Breath of the Deep",
    "返潮·赤生": "Returning Tide: Crimson Renewal",
    # -- duel_abyssal_leviathan_machine_v2 技能名 -------------------------
    "磁轨·穿鳞点射": "Magrail: Scale-Piercing Burst",
    "脊翼·猎鲨齐射": "Spine Wings: Shark-Hunter Volley",
    "裂频·静默针": "Frequency Rift: Silent Needles",
    "相位偏折场": "Phase Deflection Field",
    "静压锚盾": "Static-Pressure Anchor Shield",
    "纳米·战损回织": "Nanoweave: Battle-Damage Reweave",
    # -- duel_eclipse_devourer_epic 技能名 --------------------------------
    "星轨·十二宫穿刺": "Star Orbit: Zodiac Pierce",
    "日冕·焚星坠仪": "Corona: Burning Star Fall",
    "逆星·绝对静默": "Retrograde Star: Absolute Silence",
    "天球·黄金经纬": "Celestial Sphere: Golden Meridians",
    "黑月·无光界面": "Black Moon: Lightless Interface",
    "命宫·群星归位": "Life Palace: Stars Return to Their Posts",
    # -- duel_last_human_colossus 技能名 ----------------------------------
    "终线·贯星钉": "Final Line: Star-Piercing Nail",
    "反物质·日蚀弹": "Antimatter: Eclipse Round",
    "磁轨·断环三连": "Magrail: Ring-Breaking Triplet",
    "折光·棱镜幕": "Refracted Light: Prism Curtain",
    "惯性·黑锚场": "Inertia: Black Anchor Field",
    "纳米·战损回卷": "Nanorepair: Battle-Damage Rewind",
    # -- duel_nine_heavens_artificer 技能名 -------------------------------
    "机簧·千机连星": "Clockspring: Thousand Linked Stars",
    "火器·赤霄龙铳": "Firearm: Crimson-Dragon Blunderbuss",
    "雷契·九霄破阵钉": "Thunder Pact: Ninth-Heaven Formation Breaker",
    "机关·万轮天幕": "Machina: Myriad-Wheel Canopy",
    "秘械·避雷天门": "Arcane Engine: Lightning-Rod Heaven Gate",
    "匠心·回天续命匣": "Craftsman's Heart: Life-Restoring Kit",
    # -- duel_shogun_mask_palace_epic 技能名 ------------------------------
    "狐火·千灯巡夜": "Foxfire: Thousand-Lantern Night Patrol",
    "绯樱·八重天葬": "Crimson Sakura: Eightfold Sky Burial",
    "白面·破军封魔矢": "White Mask: Star-Breaker Demon-Sealing Arrow",
    "千本鸟居·不动结界": "Thousand Torii: Immovable Barrier",
    "镜月·白狐神域": "Mirror Moon: White Fox Sanctuary",
    "神乐·樱魂还生": "Kagura: Sakura-Soul Revival",
    # -- duel_worldspine_hunter 技能名 ------------------------------------
    "脊骨·贯岳箭": "Spine Bone: Peak-Piercing Arrow",
    "猎日·金穹坠矢": "Sun Hunt: Golden Vault Falling Arrow",
    "裂谷·回声穿心": "Rift Valley: Echo Heart-Piercer",
    "巨骸·肋骨天垒": "Colossal Bones: Rib Rampart",
    "瀑幕·千层水镜": "Waterfall Veil: Thousand-Layer Water Mirror",
    "祖猎·兽魂归息": "Ancestral Hunt: Beast-Soul Return",
    # -- Boss 行为名(duel_oracle)----------------------------------------
    "最终审判·蓄力": "Final Judgment: Charging",
    "最终审判": "Final Judgment",
    "天罚之矛": "Spear of Heaven's Punishment",
    "三重天律": "Threefold Law",
    "圣骸重构": "Sacred Reconstruction",
    "启示录形态": "Apocalypse State",
    "按兵不动": "holding position",
    # -- Boss 行为名(duel_moon_dragon_swordsman)--------------------------
    "月蚀龙息·蓄势": "Eclipse Dragon Breath: Gathering",
    "月蚀龙息": "Eclipse Dragon Breath",
    "天尾·雷潮": "Celestial Tail: Thunder Tide",
    "龙角·碎星雨": "Dragon Horns: Star-Shattering Rain",
    "苍鳞·云甲": "Azure Scales: Cloud Armor",
    "龙脉·月髓回生": "Dragon Vein: Lunar Marrow Renewal",
    "逆鳞·天威尽显": "Reversed Scale: Heavenly Majesty Unveiled",
    "云中凝视": "Gaze from the Clouds",
    # -- Boss 行为名(duel_blood_tide_leviathan_machine)-------------------
    "吞潮大啸·蓄潮": "Tide-Swallowing Roar: Gathering Tide",
    "吞潮大啸": "Tide-Swallowing Roar",
    "渊尾·沉城": "Abyssal Tail: Sunken City",
    "万骸覆甲": "Myriad Bones Armor",
    "噬城回生": "City-Devouring Renewal",
    "赤潮暴走": "Red-Tide Frenzy",
    "潜潮窥伺": "Submerged Tide Stalking",
    # -- Boss 行为名(duel_abyssal_leviathan_machine_v2)-------------------
    "深渊鸣潮·蓄压": "Abyssal Resonance Tide: Pressurizing",
    "深渊鸣潮": "Abyssal Resonance Tide",
    "眼核·裂海射线": "Eye-Core: Sea-Splitting Ray",
    "触须·电弧围猎": "Tendrils: Arc Hunt",
    "共振鳞幕": "Resonant Scale Curtain",
    "深海共生修复": "Deep-Sea Symbiotic Repair",
    "星渊暴潮形态": "Star-Abyss Storm Tide Form",
    "深海凝视": "Deep-Sea Gaze",
    # -- Boss 行为名(duel_eclipse_devourer_epic)--------------------------
    "终极星噬·蓄鸣": "Ultimate Star Devourer: Gathering Hum",
    "终极星噬·噬界吐息": "Ultimate Star Devourer: World-Devouring Breath",
    "星骸雨·天坠": "Star-Remnant Rain: Heavenly Fall",
    "万眼·视差裁决": "Myriad Eyes: Parallax Verdict",
    "万眼·虚空眼幕": "Myriad Eyes: Void Eye Veil",
    "蜕星·噬光再生": "Star Molting: Light-Devouring Regeneration",
    "群星饥渴": "Hunger of the Stars",
    "吞星凝视": "Star-Devouring Gaze",
    # -- Boss 行为名(duel_last_human_colossus)----------------------------
    "赤环灭世·蓄力": "Crimson Ring Apocalypse: Charging",
    "赤环灭世": "Crimson Ring Apocalypse",
    "掌心·裂城脉冲": "Palm: City-Splitting Pulse",
    "背环·天轨矛雨": "Back Ring: Sky-Track Spear Rain",
    "相位·千层装甲": "Phase: Thousand-Layer Armor",
    "机群·钢骸回铸": "Drone Swarm: Steel-Frame Reforge",
    "欧米伽·末日超载": "Omega: Doomsday Overload",
    "静默演算": "Silent Computation",
    # -- Boss 行为名(duel_nine_heavens_artificer)-------------------------
    "天狱雷诏·聚敕": "Heavenly-Prison Thunder Edict: Summoning",
    "天狱雷诏": "Heavenly-Prison Thunder Edict",
    "镇岳神掌": "Mountain-Suppressing Divine Palm",
    "玄岩法壁": "Dark-Rock Spell Wall",
    "万壑归骸": "Ten Thousand Ravines Return to Bone",
    "神魔共生": "God-Demon Symbiosis",
    "九天巡狱": "Nine-Heavens Prison Patrol",
    # -- Boss 行为名(duel_shogun_mask_palace_epic)------------------------
    "百鬼月蚀·蓄势": "Hundred-Demons Eclipse: Gathering",
    "百鬼月蚀": "Hundred-Demons Eclipse",
    "鬼枪·雷门崩": "Oni Lance: Thunder-Gate Breaker",
    "能面·哭岚千刃": "Noh Mask: Weeping Storm of a Thousand Blades",
    "百面铁幕": "Hundred-Mask Iron Curtain",
    "鬼铠·百匠返魂": "Oni Armor: Hundred Artisans' Rebirth",
    "修罗百面": "Asura Hundred Masks",
    "魔座凝视": "Gaze from the Demon Throne",
    # -- Boss 行为名(duel_worldspine_hunter)------------------------------
    "熔瞳天火·蓄势": "Molten-Eye Skyfire: Gathering",
    "熔瞳天火": "Molten-Eye Skyfire",
    "天脊·万岳崩落": "Worldspine: Ten Thousand Peaks Collapse",
    "古骸·肋枪雨": "Ancient Bones: Rib-Spear Rain",
    "玄岩甲": "Dark-Rock Carapace",
    "山脉回生": "Mountain Range Renewal",
    "天脊震怒": "Worldspine Wrath",
    "山眠凝视": "Mountain-Sleep Gaze",
    # -- duel Boss 行为说明(旧版 oracle 会话兼容)------------------------
    "审判蓄力——下回合将释放最终审判": "Charging — Final Judgment comes next turn",
    "最终审判——吐息命中": "Final Judgment — breath hit",
    "最终审判——被完全格挡": "Final Judgment — fully blocked",
    "最终审判——穿透了防御": "Final Judgment — pierced the guard",
    "天罚之矛——重击命中": "Spear of Heaven — strike hit",
    "天罚之矛——被完全格挡": "Spear of Heaven — fully blocked",
    "展开三重天律——火系攻击将被挡下": "Threefold Law raised — fire attacks will be blocked",
    "圣骸重构——修复了伤势": "Sacred Reconstruction — wounds mended",
    "启示录形态——进入狂暴,攻击提升": "Apocalypse State — enraged, attack up",
    # -- 终局字幕 ----------------------------------------------------------
    "◆ 战斗胜利 ◆": "◆ Victory ◆",
    "◆ 战斗失败 ◆": "◆ Defeat ◆",
    "◆ 告白成功 ◆": "◆ Confession Accepted ◆",
    "◆ 她离开了 ◆": "◆ She Walked Away ◆",
    "◆ 剧终 ◆": "◆ The End ◆",
    "◆ 故事落幕 ◆": "◆ The Story Ends ◆",
    # -- 角色名 ------------------------------------------------------------
    "小雪": "Xiaoxue",
    "小湄": "Xiaomei",
    "书宁": "Shuning",
    "千夏": "Chinatsu",
    "凛": "Rin",
    # -- 场景名 ------------------------------------------------------------
    "温馨卧室": "Cozy Bedroom",
    "街角咖啡厅": "Corner Café",
    "旧图书馆": "Old Library",
    "霓虹月台": "Neon Platform",
    "西部边陲小镇": "Frontier Town",
    "现代都市街头": "Modern City Street",
    "古风城镇": "Ancient Chinese Town",
    "东瀛街巷": "Japanese Street",
    "罗马古城": "Ancient Rome",
    "罪恶都市": "Vice City",
    "罪恶都市首夜": "First Night in Vice City",
    "状态:力竭——只能缓慢走动": "Status: Exhausted — slow walk only",
    "体力透支:本幕只能缓慢走动": "Stamina depleted — slow walk this turn",
    "通缉犯黑荆": "Blackthorn the Outlaw",
    "消失的街头偶像": "The Vanished Street Idol",
    "失传的古卷": "The Lost Scroll",
    "狐狸的谢礼": "The Fox's Gift",
    "角斗场的黎明": "Dawn of the Gladiator",
    "旅行者": "Traveler",
    # -- openworld 面板/HUD 措辞(英文界面不得残留中文)--------------------
    "▶ 自动探索": "▶ Auto-explore",
    "自由行动": "Free action",
    "行动": "Action",
    "自动探索": "Auto-explore",
    "本幕自动探索 · 下一回合由你决定": "Auto-explored this act · you choose next turn",
    "上幕自动探索": "auto-explored last act",
    "VLM 提议的行动": "Suggested by the VLM director",
    "玩家自定义行动(label=内容)": "Type your own action",
    "按 VLM 提议行动(label=提议内容)": "Act on the VLM director's suggestion",
    "画面暂无有意义的交互机会,本回合由 AI 推进场景":
        "Nothing worth choosing here; the AI moves the scene forward",
    "冒险刚刚开始。": "The adventure has just begun.",
    "画面平淡,旅途自动向前推进。": "A quiet stretch; the journey moves on by itself.",
    "沿着大路继续向前探索": "Keep exploring along the main road",
    "仔细观察四周的环境": "Look carefully around",
    "走进附近显眼的建筑一探究竟": "Enter the nearby landmark building",
    # openworld 主线阶段(game/openworld/quests.py 的 stages,逐字一致)
    "在酒馆里从醉汉口中套听到黑荆藏身峡谷的线索":
        "In the saloon, coax a clue about Blackthorn's canyon hideout from a drunk",
    "沿着峡谷马蹄印追踪,发现黑荆的营地篝火":
        "Follow the hoofprints through the canyon to Blackthorn's campfire",
    "与黑荆在峡谷尽头正面对峙,将他绳之以法":
        "Face Blackthorn at the canyon's end and bring him to justice",
    "在公告栏的演出海报前找到她留下的记号":
        "Find the mark she left on the concert poster at the notice board",
    "循着记号进入涩谷地下的秘密 Livehouse":
        "Follow the marks to a secret underground livehouse in Shibuya",
    "登上 Livehouse 的舞台,陪她完成复出演唱会":
        "Take the livehouse stage and see her comeback concert through",
    "拜访镇口老掌柜,得到半张藏宝图残页":
        "Visit the old shopkeeper at the town gate and get half a treasure map",
    "按残页指引在山中古庙解开石门机关":
        "Follow the map to a mountain temple and open its stone-door mechanism",
    "在古庙深处请出古卷,带回交予老掌柜":
        "Recover the scroll deep in the temple and bring it to the shopkeeper",
    "在山脚神社向稻荷神祈愿,钱币泛起微光":
        "Pray to Inari at the foothill shrine until the coin begins to glow",
    "循着夜路上的狐火上山,穿过千本鸟居":
        "Follow the foxfire up the night path through the thousand torii gates",
    "在山顶神域见到白狐真身,接受它的谢礼":
        "Meet the white fox's true form at the summit and accept its gift",
    "在武器匠铺取得父亲留下的短剑":
        "Claim your father's short sword from the weaponsmith",
    "在竞技场练兵场完成入场操演,赢得观众欢呼":
        "Finish the entry drill in the training yard and win the crowd",
    "在大竞技场的正午决斗中胜出,获得自由":
        "Win the noon duel in the great arena and earn your freedom",
    "在滨海大道拦下一辆亮着尾灯的超跑,证明你玩得起这座城市的游戏":
        "Flag down a supercar on the ocean drive and prove you can play this city's game",
    "甩开警车的追截,把车停进霓虹巷的地下车库":
        "Shake off the police and park in the neon alley's underground garage",
    "在顶层夜店的天台见到线人,接过属于你的第一单活":
        "Meet the informant on the rooftop club and take your first job",
    # -- adv 行动(英文皮肤共用的"说话"自由输入动作)------------------------
    "说话": "Speak",
    "以男主角的身份对她说一句话(自由输入)": "Say one line to her as the protagonist",
    # -- duel 终局字幕(皮肤 endings 自定义)--------------------------------
    "◆ 赤潮已断 ◆": "◆ The Red Tide Is Broken ◆",
    "◆ 沉入渊海 ◆": "◆ Sunk into the Abyss ◆",
    "◆ 调伏成功 ◆": "◆ Subdued ◆",
    "◆ 调伏失败 ◆": "◆ Subjugation Failed ◆",
    "◆ 狩猎胜利 ◆": "◆ The Hunt Is Won ◆",
    "◆ 狩猎失败 ◆": "◆ The Hunt Is Lost ◆",
    # -- machine 调试菜单(Boss 行动前置说明)------------------------------
    "(当前前置不满足,强制执行可能不产生结算)":
        "(requirement not met; forcing it may have no effect)",
    "陌生疏远": "Distant",
    "初识": "Acquainted",
    "朋友": "Friends",
    "暧昧": "Flirting",
    "恋人": "Lovers",
}

# 技能/面板描述(整句);键必须与皮肤 JSON 的 description 逐字一致
_EXACT.update({
    # -- duel_oracle ------------------------------------------------------
    "无消耗的稳定输出,不受任何克制":
        "Costless steady output; unaffected by anything",
    "高伤害火系攻击;会被三重天律挡下":
        "High-damage fire attack; blocked by the Threefold Law",
    "冰系攻击;击碎三重天律":
        "Ice attack; shatters the Threefold Law",
    "完全格挡重击;被最终审判烧穿半伤":
        "Fully blocks the heavy strike; burned through by Final Judgment for half damage",
    "完全格挡最终审判;被重击砸穿半伤":
        "Fully blocks Final Judgment; smashed through by the heavy strike for half damage",
    # -- 共享 -------------------------------------------------------------
    "恢复 18 点生命": "Restores 18 HP",
    # -- duel_moon_dragon_swordsman ---------------------------------------
    "无消耗稳定剑气；不受龙鳞云障克制。":
        "Costless steady sword qi; unaffected by the Scale Cloud Barrier.",
    "高伤害群射剑术；会被龙鳞云障完全挡下。":
        "High-damage volley swordplay; fully blocked by the Scale Cloud Barrier.",
    "破盾穿刺剑气；击碎龙鳞云障。":
        "Shield-breaking piercing sword qi; shatters the Scale Cloud Barrier.",
    "完全格挡龙尾雷潮；会被月蚀龙息烧穿并承受半伤。":
        "Fully blocks the Tail Thunder Tide; burned through by the Eclipse Dragon Breath for half damage.",
    "完全格挡月蚀龙息；会被龙尾雷潮砸穿并承受半伤。":
        "Fully blocks the Eclipse Dragon Breath; smashed through by the Tail Thunder Tide for half damage.",
    "消耗剑意恢复 18 点生命。":
        "Spends Sword Intent to restore 18 HP.",
    # -- duel_blood_tide_leviathan_machine --------------------------------
    "0 消耗稳定攻击；不受甲壳克制，并回复 3 点潮息":
        "Costless steady strike; unaffected by carapace, restores 3 Tide Breath",
    "5 潮息高伤害斩击；会被深渊骨甲完全挡下":
        "5 Tide Breath high-damage slashes; fully blocked by the Abyssal Bone Armor",
    "4 潮息穿甲攻击；击碎 Boss 的深渊骨甲":
        "4 Tide Breath armor-piercing strike; shatters the boss's Abyssal Bone Armor",
    "3 潮息防御；完全格挡尾槌，吞潮大啸只能减半":
        "3 Tide Breath defense; fully blocks the Tail Hammer, the Tide-Swallowing Roar is halved",
    "3 潮息防御；完全格挡吞潮大啸，尾槌只能减半":
        "3 Tide Breath defense; fully blocks the Tide-Swallowing Roar, the Tail Hammer is halved",
    "5 潮息恢复 20 点生命，最多恢复至 96":
        "5 Tide Breath to restore 20 HP, up to at most 96",
    # -- duel_abyssal_leviathan_machine_v2 --------------------------------
    "无消耗的稳定电磁输出，不受任何护盾克制":
        "Costless steady electromagnetic output; unaffected by any shield",
    "高伤害导弹群；会被共振鳞幕挡下":
        "High-damage missile swarm; blocked by the Resonant Scale Curtain",
    "穿透共振鳞幕的破盾攻击":
        "Shield-breaking attack that pierces the Resonant Scale Curtain",
    "完全格挡眼核重击；被深渊鸣潮烧穿半伤":
        "Fully blocks the Eye-Core Smash; burned through by the Abyssal Resonance Tide for half damage",
    "完全格挡深渊鸣潮；被眼核重击穿透半伤":
        "Fully blocks the Abyssal Resonance Tide; pierced by the Eye-Core Smash for half damage",
    # -- duel_eclipse_devourer_epic ---------------------------------------
    "无消耗稳定输出;以黄道星矛贯穿虚空,不受任何克制":
        "Costless steady output; the ecliptic star lance pierces the void, unaffected by anything",
    "高伤害日冕攻击;会被虚空眼幕完全吞没":
        "High-damage corona strike; fully engulfed by the Void Eye Veil",
    "寒寂攻击;可击碎虚空眼幕并冻结其引力结构":
        "Frost-silence strike; shatters the Void Eye Veil and freezes its gravity structure",
    "完全格挡星骸轰击;会被噬界吐息烧穿并承受半伤":
        "Fully blocks the Star-Remnant Bombardment; burned through by the World-Devouring Breath for half damage",
    "完全格挡噬界吐息;会被星骸轰击砸穿并承受半伤":
        "Fully blocks the World-Devouring Breath; smashed through by the Star-Remnant Bombardment for half damage",
    "恢复 18 点生命;以本命星图回溯伤势":
        "Restores 18 HP; rewinds wounds along the natal star chart",
    # -- duel_last_human_colossus -----------------------------------------
    "无消耗稳定射击;不受护盾克制":
        "Costless steady shot; unaffected by shields",
    "高伤害爆裂弹;会被相位装甲完全挡下":
        "High-damage explosive round; fully blocked by the Phase Armor",
    "穿甲三连射;击碎相位装甲":
        "Three-round armor-piercing burst; shatters the Phase Armor",
    "完全格挡脉冲炮;被重力轰击穿透半伤":
        "Fully blocks the Pulse Cannon; pierced by the Gravity Bombardment for half damage",
    "完全格挡重力轰击;被脉冲炮烧穿半伤":
        "Fully blocks the Gravity Bombardment; burned through by the Pulse Cannon for half damage",
    # -- duel_nine_heavens_artificer --------------------------------------
    "无消耗稳定输出;连续机关矢不受克制":
        "Costless steady output; linked machina bolts unaffected by counters",
    "高伤害爆燃攻击;会被玄岩法壁挡下":
        "High-damage deflagration shot; blocked by the Dark-Rock Spell Wall",
    "雷霆穿甲攻击;专门击碎玄岩法壁":
        "Thunder armor-piercing strike; purpose-built to shatter the Dark-Rock Spell Wall",
    "完全格挡镇岳神掌;被天狱雷诏烧穿半伤":
        "Fully blocks the Mountain-Suppressing Divine Palm; burned through by the Heavenly-Prison Thunder Edict for half damage",
    "完全格挡天狱雷诏;被镇岳神掌砸穿半伤":
        "Fully blocks the Heavenly-Prison Thunder Edict; smashed through by the Mountain-Suppressing Divine Palm for half damage",
    "消耗机关储能;恢复 18 点生命":
        "Spends Machina charge; restores 18 HP",
    # -- duel_shogun_mask_palace_epic -------------------------------------
    "无消耗稳定输出;白狐火追踪命中,不受护盾克制":
        "Costless steady output; white foxfire homes in on the target, unaffected by shields",
    "高伤害樱雷仪式;会被百面铁幕挡下":
        "High-damage sakura-thunder ritual; blocked by the Hundred-Mask Iron Curtain",
    "穿透式封魔攻击;击碎百面铁幕":
        "Piercing demon-sealing attack; shatters the Hundred-Mask Iron Curtain",
    "完全格挡鬼枪轰击;被百鬼月蚀烧穿半伤":
        "Fully blocks the Oni Lance bombardment; burned through by the Hundred-Demons Eclipse for half damage",
    "完全格挡百鬼月蚀;被鬼枪轰击砸穿半伤":
        "Fully blocks the Hundred-Demons Eclipse; smashed through by the Oni Lance for half damage",
    # -- duel_worldspine_hunter -------------------------------------------
    "无消耗的稳定穿刺射击；命中后回复 3 点猎息。":
        "Costless steady piercing shot; restores 3 Hunt Spirit on hit.",
    "高伤害猎息箭；会被玄岩甲完全挡下。":
        "High-damage Hunt Spirit arrow; fully blocked by the Dark-Rock Carapace.",
    "破盾重箭；专门击碎玄岩甲，伤害略低于猎日箭。":
        "Shield-breaking heavy arrow; purpose-built to shatter the Dark-Rock Carapace, slightly weaker than the Sun Hunt arrow.",
    "完全格挡山崩重击；会被熔瞳天火烧穿并承受半伤。":
        "Fully blocks the Landslide Smash; burned through by the Molten-Eye Skyfire for half damage.",
    "完全格挡熔瞳天火；会被山崩重击砸穿并承受半伤。":
        "Fully blocks the Molten-Eye Skyfire; smashed through by the Landslide Smash for half damage.",
    "消耗猎息恢复 18 点生命，并保持原位继续狩猎。":
        "Spends Hunt Spirit to restore 18 HP and hold position to keep hunting.",
    # -- 旧版 oracle 技能描述(历史会话兼容)--------------------------------
    "无消耗的稳定奥术攻击,不受敌方护盾克制":
        "Costless stable arcane attack; never blocked by the enemy shield",
    "高伤害火系攻击,会被敌方护盾完全挡下":
        "High-damage fire attack; fully blocked by the enemy shield",
    "中等伤害冰系攻击,可击碎敌方已展开的护盾":
        "Mid-damage ice attack; shatters an active enemy shield",
    "防御技能,可完全格挡敌方物理重击":
        "Defense; fully blocks the enemy physical strike",
    "防御技能,可完全格挡敌方吐息类最终审判":
        "Defense; fully blocks the enemy breath (Final Judgment)",
    "消耗法力恢复生命值,适合在低血量时稳定续航":
        "Spends MP to restore HP; sustain at low health",
})

# 复合句内的子串替换(标签、技能名等)
_SUBSTR = [
    ("体力", "Stamina"),
    ("好感", "Affection"),
    ("敌方行动", "Enemy Action"),
    ("心动变化", "Affection"),
    ("冒险见闻", "Field Notes"),
]

# 心情词(嵌在模板里,单独映射;romance 的中文心情 + adv 的英文心情 id 落地词)
_MOOD = {"开心": "happy", "害羞": "shy", "俏皮": "playful",
         "感动": "touched", "生气": "upset", "失落": "blue"}

_RULES = [
    (re.compile(r"力竭剩余 (\d+) 幕"), r"exhausted for \1 more act(s)"),
    (re.compile(r"前置:"), "Requires: "),
    # 复合句式在前(内嵌 第N回合/心情 等子句式,必须先于通用规则)
    (re.compile(r"恋爱意图@第(\d+)回合"), r"Love intent @ turn \1"),
    (re.compile(r"她的心情:(.+)"), lambda m: "Her mood: " + _mood(m.group(1))),
    (re.compile(r"好感 ([+-]?\d+) · 她看起来(.+)"),
     lambda m: f"{m.group(1)} · She looks {_mood(m.group(2))}"),
    (re.compile(r"好感持平 · 她看起来(.+)"),
     lambda m: f"unchanged · She looks {_mood(m.group(1))}"),
    (re.compile(r"态度 ([+-]?\d+) · (.+?)看起来(.+)"),
     lambda m: f"Attitude {m.group(1)} · {m.group(2)} looks {_mood(m.group(3))}"),
    (re.compile(r"态度持平 · (.+?)看起来(.+)"),
     lambda m: f"Attitude unchanged · {m.group(1)} looks {_mood(m.group(2))}"),
    (re.compile(r"Boss 招式:(.+)"), r"Boss Move: \1"),
    # 通用句式在后
    (re.compile(r"第(\d+)回合"), r"Turn \1"),
    (re.compile(r"第(\d+)幕"), r"Act \1"),
    (re.compile(r"主线《(.+?)》"), r'Quest: "\1"'),
    (re.compile(r"地点:(.+)"), r"Location: \1"),
    (re.compile(r"关系:(.+)"), r"Relationship: \1"),
    (re.compile(r"关系 (\d+)\s*/\s*(\d+)"), r"Relationship \1/\2"),
    (re.compile(r"剧情:(.+)"), r"Story: \1"),
    (re.compile(r"结局:(.+)"), r"Ending: \1"),
    (re.compile(r"心情:(.+)"), lambda m: "Mood: " + _mood(m.group(1))),
    (re.compile(r"\(关系 ([+-]?\d+)\)"), r"(relationship \1)"),
]


def _mood(word: str) -> str:
    return _MOOD.get(word, word)


def tr(text: str, lang: str | None = None) -> str:
    """按 lang(缺省 = current_lang())翻译 HUD 文本;zh 原样返回,未收录词条透传。"""
    if not text or (lang or current_lang()) != "en":
        return text
    if text in _EXACT:
        return _EXACT[text]
    out = text
    for pattern, repl in _RULES:
        out = pattern.sub(repl, out) if callable(repl) else pattern.sub(repl, out)
    substrings = sorted([*_EXACT.items(), *_SUBSTR],
                        key=lambda kv: len(kv[0]), reverse=True)
    for zh, en in substrings:  # 长词条优先,避免短词抢先吃掉长词
        if len(zh) >= 2 and zh in out:
            out = out.replace(zh, en)
    return _ascii_punct(out)


# 英文输出里残留的全角标点 → ASCII(如 "(1/3):" → " (1/3): ")
_PUNCT = {"\uff08": " (", "\uff09": ") ", "\uff1a": ": ", "\uff0c": ", ",
          "\u3002": ". ", "\uff1b": "; ", "\uff01": "! ", "\uff1f": "? ",
          "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"',
          "\u2014\u2014": " \u2014 "}


def _ascii_punct(text: str) -> str:
    if not any(wide in text for wide in _PUNCT):
        return text
    for wide, ascii_ in _PUNCT.items():
        text = text.replace(wide, ascii_)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" ([,.;:!?)])", r"\1", text)
    text = re.sub(r"\( ", "(", text)
    return text.strip()


__all__ = ["tr", "LANG", "current_lang", "hud_lang"]
