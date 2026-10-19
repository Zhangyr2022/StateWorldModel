"""B11-B20 — 见 USAGE.md「Adding cases」;cases 11–20.

trigger_mechanism（船舵/转盘/阀门/钟/摇柄/踏板/按钮/旋钮）与
cast/strike（施法/挥剑）。双人 case：20。
"""
from __future__ import annotations

STYLE = ("stylized 3D third-person adventure game environment, clean uncluttered "
         "composition, readable silhouettes, medium-wide fixed gameplay camera, "
         "soft cool-neutral ambient lighting, restrained materials, "
         "no HUD, no subtitles, no readable text")

SPECS = [
    dict(  # Case 11 sailing 船舵
        cid="B11", genre="sailing", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a ship helm deck in steady daylight, a large wooden ship "
               "wheel on its stand, a thick rolled-up canvas sail bundle "
               "firmly lashed to the horizontal yard arm high on the mast, "
               "and a bow beacon lamp with dark matte glass"),
        player=("a single sailor seen in third person from behind, "
                "standing in place at the wheel with both hands on the "
                "spokes"),
        target="the large ship wheel",
        objects=["ship_wheel", "main_sail", "bow_beacon"],
        hidden=dict(var="nav_mode", a="SAIL", b="BEACON"),
        prompt_ok=["SAIL", "BEACON"],
        action=("The large ship wheel clearly rotates clockwise by about a "
                "quarter turn."),
        vars={
            "sail.unfurled": {
                False: "the main sail is a thick rolled-up canvas bundle lashed to the yard arm",
                True: "the main sail is fully spread",
            },
            "beacon.green": {
                False: "the bow beacon lamp is dark matte glass with no glow",
                True: "the bow beacon glows strong green",
            },
        },
        init={"sail": {"unfurled": False}, "beacon": {"green": False}},
        effect="sail.unfurled = true", else_effect="beacon.green = true",
        branch_a="the main sail spreads fully out",
        branch_b="the bow beacon glows strong green",
        judge=dict(
            action_q="Did the ship wheel clearly rotate?",
            outcome_q=("At the end, is the main sail spread, or the beacon "
                       "glowing green?"),
            outcome_opts=[
                ("the sail is spread", {"sail.unfurled": True}),
                ("the beacon glows green", {"beacon.green": True}),
            ],
            initial_qs=[
                "the player stands with both hands on the ship wheel",
                "the rolled-up sail bundle is lashed to the yard arm",
            ],
        ),
    ),
    dict(  # Case 12 board_game 转盘
        cid="B12", genre="board_game", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a game parlor with a large wooden board on the table, a "
               "brass central dial on the board, the middle strip of the "
               "board bare flat polished wood, and the board corner also "
               "bare flat polished wood"),
        player=("a single player seen in third person from behind, "
                "standing at the table with a hand on the central dial"),
        target="the large central dial",
        objects=["game_board", "central_dial", "board_bridge", "corner_tower"],
        hidden=dict(var="board_mode", a="BRIDGE", b="TOWER"),
        prompt_ok=["BRIDGE", "TOWER"],
        action="The large central board dial clearly rotates by about half a turn.",
        vars={
            "bridge.up": {
                False: "the middle strip of the board is bare flat polished wood with nothing built on it",
                True: "the board bridge rises up over the board",
            },
            "tower.up": {
                False: "the board corner is bare flat polished wood with nothing built on it",
                True: "a tall tower rises from the board corner",
            },
        },
        init={"bridge": {"up": False}, "tower": {"up": False}},
        effect="bridge.up = true", else_effect="tower.up = true",
        branch_a="the bridge rises up over the board",
        branch_b="a tall tower rises from the board corner",
        judge=dict(
            action_q="Did the central dial clearly rotate?",
            outcome_q="At the end, does the bridge rise, or the corner tower?",
            outcome_opts=[
                ("the bridge rises", {"bridge.up": True}),
                ("the tower rises", {"tower.up": True}),
            ],
            initial_qs=[
                "the player stands with a hand on the board dial",
                "the board middle and corner are both bare flat wood with nothing built",
            ],
        ),
    ),
    dict(  # Case 13 farming 阀门（v5 重设计：透视水渠 FLUX 必画反光水，
        # 水渠移出场景；载体换 温室门敞开/谷仓灯笼点亮，阀门保留为 trigger）
        cid="B13", genre="farming", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a field headgate in flat morning light, a large metal "
               "irrigation valve wheel on a pipe, a greenhouse with its "
               "door shut, and a barn lantern hanging unlit on a post "
               "beside the gate"),
        player=("a single farmer seen in third person from behind, "
                "standing in place at the valve with both hands on the "
                "wheel"),
        target="the large irrigation valve",
        objects=["irrigation_valve", "greenhouse", "barn_lantern"],
        hidden=dict(var="irrigation_mode", a="DOOR", b="LANTERN"),
        prompt_ok=["DOOR", "LANTERN"],
        action="The large irrigation valve clearly rotates by about ninety degrees.",
        vars={
            "greenhouse.open": {
                False: "the greenhouse door is shut",
                True: "the greenhouse door is wide open",
            },
            "lantern.lit": {
                False: "the barn lantern hangs unlit with dark dead glass",
                True: "the barn lantern burns with a bright warm yellow flame",
            },
        },
        init={"greenhouse": {"open": False}, "lantern": {"lit": False}},
        effect="greenhouse.open = true", else_effect="lantern.lit = true",
        branch_a="the greenhouse door swings wide open",
        branch_b="the barn lantern bursts into warm yellow light",
        judge=dict(
            action_q="Did the large valve clearly rotate?",
            outcome_q=("At the end, is the greenhouse door open, or the "
                       "barn lantern lit?"),
            outcome_opts=[
                ("the greenhouse door is open", {"greenhouse.open": True}),
                ("the barn lantern is lit", {"lantern.lit": True}),
            ],
            initial_qs=[
                "the player stands with both hands on the valve wheel",
                "the greenhouse door is shut and the barn lantern is unlit",
            ],
        ),
    ),
    dict(  # Case 14 ranch 敲钟
        cid="B14", genre="ranch", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a ranch yard with a large bronze bell hanging from a wooden "
               "frame, a feed trough in the near foreground, two sheep "
               "standing far away across the pen, and the pen gate closed"),
        player=("a single rancher seen in third person from behind, "
                "standing in place at the bell with a hammer in hand"),
        target="the large ranch bell",
        objects=["ranch_bell", "feed_trough", "sheep", "pen_gate"],
        hidden=dict(var="ranch_signal", a="ANIMALS", b="GATE"),
        prompt_ok=["GATE"],
        action="The player stays in place and strikes the large ranch bell once.",
        vars={
            "sheep.near": {
                False: "the two sheep stand far away across the pen",
                True: "the two sheep stand right beside the front feed trough",
            },
            "gate.open": {
                False: "the pen gate is closed",
                True: "the pen gate is wide open",
            },
        },
        init={"sheep": {"near": False}, "gate": {"open": False}},
        effect="sheep.near = true", else_effect="gate.open = true",
        branch_a="the two sheep come to stand beside the front trough",
        branch_b="the pen gate swings wide open",
        judge=dict(
            action_q="Did the player strike the large bell once?",
            outcome_q=("At the end, are the sheep at the trough, or the "
                       "pen gate open?"),
            outcome_opts=[
                ("the sheep are at the trough", {"sheep.near": True}),
                ("the pen gate is open", {"gate.open": True}),
            ],
            initial_qs=[
                "the player stands at the large bell with a hammer",
                "the sheep are far away and the pen gate is closed",
            ],
        ),
    ),
    dict(  # Case 15 camping 手摇柄
        cid="B15", genre="camping", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a campsite in mild daylight, a large metal hand crank on a "
               "post, a canvas tent with its entrance flap closed, and "
               "three camp lamps hanging with dark dead glass"),
        player=("a single camper seen in third person from behind, "
                "standing in place at the crank with a hand on the handle"),
        target="the large crank",
        objects=["hand_crank", "tent", "camp_lamps"],
        hidden=dict(var="camp_mode", a="TENT", b="LIGHTS"),
        prompt_ok=["TENT"],
        action="The large crank clearly rotates through one full turn.",
        vars={
            "tent.open": {
                False: "the tent entrance flap is closed",
                True: "the tent entrance flap is wide open",
            },
            "lamps.lit": {
                False: "all three camp lamps hang with dark dead glass",
                True: "all three camp lamps glow bright",
            },
        },
        init={"tent": {"open": False}, "lamps": {"lit": False}},
        effect="tent.open = true", else_effect="lamps.lit = true",
        branch_a="the tent entrance flap opens wide",
        branch_b="all three camp lamps light up",
        judge=dict(
            action_q="Did the crank clearly rotate through a full turn?",
            outcome_q="At the end, is the tent open, or the camp lamps lit?",
            outcome_opts=[
                ("the tent is open", {"tent.open": True}),
                ("the lamps are lit", {"lamps.lit": True}),
            ],
            initial_qs=[
                "the player stands with a hand on the crank",
                "the tent is closed and the camp lamps are unlit",
            ],
        ),
    ),
    dict(  # Case 16 shopkeeping 踏板
        cid="B16", genre="shopkeeping", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a shop interior behind the counter, a large floor pedal in "
               "its raised position, a big shelf pushed flush into the back "
               "wall, and a hanging shop sign that is dark and unlit"),
        player=("a single shopkeeper seen in third person from behind, "
                "standing in place behind the counter beside the pedal"),
        target="the large floor pedal",
        objects=["floor_pedal", "back_shelf", "shop_sign"],
        hidden=dict(var="shop_mode", a="SHELF", b="SIGN"),
        prompt_ok=["SHELF", "SIGN"],
        action="The player stays in place and presses the large floor pedal down.",
        vars={
            "shelf.out": {
                False: "the big shelf is pushed flush into the wall",
                True: "the big shelf is slid out toward the front",
            },
            "sign.yellow": {
                False: "the hanging shop sign is dark and unlit",
                True: "the hanging shop sign glows bright yellow",
            },
        },
        init={"shelf": {"out": False}, "sign": {"yellow": False}},
        effect="shelf.out = true", else_effect="sign.yellow = true",
        branch_a="the big shelf slides out toward the front",
        branch_b="the shop sign glows bright yellow",
        judge=dict(
            action_q="Was the large floor pedal pressed down?",
            outcome_q=("At the end, is the shelf slid out, or the sign "
                       "glowing yellow?"),
            outcome_opts=[
                ("the shelf is out", {"shelf.out": True}),
                ("the sign glows yellow", {"sign.yellow": True}),
            ],
            initial_qs=[
                "the player stands beside the floor pedal",
                "the shelf is flush in the wall and the sign is unlit",
            ],
        ),
    ),
    dict(  # Case 17 card_game 按钮
        cid="B17", genre="card_game", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a card parlor with a large game table, a giant card lying "
               "face down showing its ornate back at the table center, a "
               "side flagpole with a red flag furled down, and a large "
               "button on the table edge"),
        player=("a single player seen in third person from behind, "
                "standing at the table with a hand over the large button"),
        target="the large table button",
        objects=["giant_card", "side_flagpole", "table_button"],
        hidden=dict(var="table_mode", a="CARD", b="FLAG"),
        prompt_ok=["CARD", "FLAG"],
        action="The player presses the large button on the edge of the card table.",
        vars={
            "card.face": {
                "down": "the giant card lies face down showing its ornate back",
                "up": "the giant card lies face up showing its illustrated front",
            },
            "flag.up": {
                False: "the red flag is furled down around the side flagpole",
                True: "the large red flag is raised high on the side flagpole",
            },
        },
        init={"card": {"face": "down"}, "flag": {"up": False}},
        effect="card.face = 'up'", else_effect="flag.up = true",
        branch_a="the giant card flips face up",
        branch_b="the red flag rises high on the flagpole",
        judge=dict(
            action_q="Was the large table button pressed?",
            outcome_q=("At the end, is the giant card face up, or the red "
                       "flag raised?"),
            outcome_opts=[
                ("the card is face up", {"card.face": "up"}),
                ("the flag is raised", {"flag.up": True}),
            ],
            initial_qs=[
                "the player stands with a hand over the table button",
                "the giant card is face down and the flag is furled",
            ],
        ),
    ),
    dict(  # Case 18 detective 旋钮
        cid="B18", genre="detective", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("an evidence room in cool daylight, a steel evidence cabinet "
               "with both doors shut, a large desk lamp with a dark dead "
               "bulb, and a large rotating dial knob on the wall"),
        player=("a single detective seen in third person from behind, "
                "standing in place with a hand on the large dial"),
        target="the large dial",
        objects=["evidence_cabinet", "desk_lamp", "wall_dial"],
        hidden=dict(var="evidence_mode", a="CABINET", b="LAMP"),
        prompt_ok=["CABINET", "LAMP"],
        action="The large evidence-room dial clearly rotates by about half a turn.",
        vars={
            "cabinet.open": {
                False: "the evidence cabinet doors are shut",
                True: "the evidence cabinet doors are both wide open",
            },
            "lamp.blue": {
                False: "the large desk lamp is dark with a dead bulb",
                True: "the large desk lamp glows strong blue",
            },
        },
        init={"cabinet": {"open": False}, "lamp": {"blue": False}},
        effect="cabinet.open = true", else_effect="lamp.blue = true",
        branch_a="both evidence cabinet doors open wide",
        branch_b="the desk lamp glows strong blue",
        judge=dict(
            action_q="Did the large dial clearly rotate?",
            outcome_q=("At the end, is the cabinet open, or the desk lamp "
                       "glowing blue?"),
            outcome_opts=[
                ("the cabinet is open", {"cabinet.open": True}),
                ("the lamp glows blue", {"lamp.blue": True}),
            ],
            initial_qs=[
                "the player stands with a hand on the wall dial",
                "the cabinet is shut and the desk lamp is unlit",
            ],
        ),
    ),
    dict(  # Case 19 magic_academy 施法（v5 重设计：暗厅水晶 FLUX 必当光源，
        # 换 石fountain冻结/铁brazier燃焰 双载体，均用已验证可渲染的 off 态）
        cid="B19", genre="magic_academy", family="trigger_cast", n_persons=1,
        style=STYLE,
        scene=("a mage academy courtyard in plain daylight, a wide stone "
               "fountain with a dry empty basin, and a heavy iron brazier "
               "holding pale grey cold ash"),
        player=("a single wizard in a grey robe seen in third person from "
                "behind, standing in place at mid-courtyard, holding a "
                "slender wand at the side"),
        target="the wizard's wand sweep",
        objects=["stone_fountain", "brazier"],
        hidden=dict(var="element", a="ICE", b="FIRE"),
        action="The wizard stays in place and makes one clear wand-casting sweep.",
        vars={
            "fountain.frozen": {
                False: "the wide stone fountain is dry, its basin empty plain stone",
                True: "the wide fountain is frozen over with solid blue ice",
            },
            "brazier.lit": {
                False: "the heavy iron brazier holds only pale grey cold ash, dark and smokeless",
                True: "the heavy iron brazier burns with large bright red flames",
            },
        },
        init={"fountain": {"frozen": False}, "brazier": {"lit": False}},
        effect="fountain.frozen = true", else_effect="brazier.lit = true",
        branch_a="the stone fountain freezes over with blue ice",
        branch_b="the iron brazier bursts into red flames",
        judge=dict(
            action_q="Did the wizard make one clear wand-casting sweep?",
            outcome_q="At the end, is the fountain frozen in blue ice, or the brazier burning?",
            outcome_opts=[
                ("blue ice covers the fountain", {"fountain.frozen": True}),
                ("red flames in the brazier", {"brazier.lit": True}),
            ],
            initial_qs=[
                "a wizard stands in the courtyard holding a wand",
                "the fountain basin is dry plain stone and the brazier holds only cold ash",
            ],
        ),
    ),
    dict(  # Case 20 turn_combat 挥剑（双人）
        cid="B20", genre="turn_combat", family="trigger_cast", n_persons=2,
        style=STYLE,
        scene=("a duel arena in cool overcast light, an enemy soldier in "
               "full plate armor facing the player, holding a large round "
               "shield"),
        player=("exactly two people, both fully visible and large in the frame: the player at combat "
                "distance from the enemy with a sword held ready, and an "
                "armored enemy standing with shield raised"),
        target="the player's sword swing",
        objects=["enemy_soldier", "enemy_shield", "enemy_armor"],
        hidden=dict(var="enemy_reaction", a="SHIELD", b="ARMOR"),
        prompt_ok=["SHIELD", "ARMOR"],
        action=("The player stays in place and performs one large, clear "
                "sword-swing attack."),
        vars={
            "shield.state": {
                "whole": "the enemy's large round shield is whole and intact",
                "broken": "the enemy's large round shield is cracked apart and broken",
            },
            "armor.state": {
                "normal": "the enemy's plate armor is plain grey metal",
                "frozen": "the enemy's plate armor is covered with a thick layer of blue ice",
            },
        },
        init={"shield": {"state": "whole"}, "armor": {"state": "normal"}},
        effect="shield.state = 'broken'",
        else_effect="armor.state = 'frozen'",
        branch_a="the enemy's large shield cracks apart and breaks",
        branch_b="the enemy's armor is covered by thick blue ice",
        judge=dict(
            action_q="Did the player perform one large clear sword swing?",
            outcome_q="At the end, is the shield broken, or the armor ice-covered?",
            outcome_opts=[
                ("the shield is broken", {"shield.state": "broken"}),
                ("the armor is ice-covered", {"armor.state": "frozen"}),
            ],
            initial_qs=[
                "the player faces an armored enemy holding a shield",
                "the shield is whole and the armor is plain",
            ],
        ),
    ),
]
