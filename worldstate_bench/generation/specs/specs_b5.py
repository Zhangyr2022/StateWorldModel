"""B41-B50 — 见 USAGE.md「Adding cases」;cases 41–50.

gesture / cast 收尾。双人 case：41、43。B50 为画面左右（viewer's
left/right）路由 case：两个相同火盆，hidden 决定哪个燃绿火。
"""
from __future__ import annotations

STYLE = ("stylized 3D third-person adventure game environment, clean uncluttered "
         "composition, readable silhouettes, medium-wide fixed gameplay camera, "
         "soft cool-neutral ambient lighting, restrained materials, "
         "no HUD, no subtitles, no readable text")

SPECS = [
    dict(  # Case 41 dating 挥手（双人）
        cid="B41", genre="dating", family="trigger_gesture", n_persons=2,
        soft_gate=True,
        style=STYLE,
        scene=("a garden plaza in soft afternoon light, an empty stone "
               "bench beside a flowerbed, and a wide stone fountain "
               "standing dry and still"),
        player=("exactly two people, both fully visible and large in the frame: the player standing "
                "upright at the plaza center, and one NPC standing upright "
                "beside the empty bench"),
        target="the player's wave",
        objects=["npc", "bench", "fountain"],
        hidden=dict(var="date_mode", a="NPC", b="FOUNTAIN"),
        prompt_ok=["NPC", "FOUNTAIN"],
        action="The player stays in place and gives one clear wave.",
        vars={
            "npc.seated": {
                False: "the NPC stands upright beside the empty bench",
                True: "the NPC sits down on the bench",
            },
            "fountain.running": {
                False: "the wide stone fountain is dry and still",
                True: "the fountain jets a tall plume of water",
            },
        },
        init={"npc": {"seated": False}, "fountain": {"running": False}},
        effect="npc.seated = true", else_effect="fountain.running = true",
        branch_a="the NPC sits down on the bench",
        branch_b="the fountain jets a tall plume of water",
        judge=dict(
            action_q="Did the player give one clear wave?",
            outcome_q="At the end, does the NPC sit down, or the fountain jet water?",
            outcome_opts=[
                ("the NPC sits down", {"npc.seated": True}),
                ("the fountain jets water", {"fountain.running": True}),
            ],
            initial_qs=[
                "the player and an NPC stand in the garden",
                "the NPC is standing and the fountain is still",
            ],
        ),
    ),
    dict(  # Case 42 card_game 举手
        cid="B42", genre="card_game", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a card parlor with a large game table, a giant card lying "
               "face down showing its ornate back at the center, and two "
               "iron braziers flanking the table holding only dark cold "
               "ash"),
        player=("a single player seen in third person from behind, "
                "standing in place behind the table, arms at the sides"),
        target="the player's raised hand",
        objects=["giant_card", "braziers"],
        hidden=dict(var="card_effect", a="CARD", b="FIRE"),
        prompt_ok=["CARD"],
        action="The player stays in place and raises one hand high above the head.",
        vars={
            "card.face": {
                "down": "the giant card lies face down showing its ornate back",
                "up": "the giant card lies face up showing its illustrated front",
            },
            "braziers.blue": {
                False: "both braziers hold only dark cold ash",
                True: "both braziers burn with large blue flames",
            },
        },
        init={"card": {"face": "down"}, "braziers": {"blue": False}},
        effect="card.face = 'up'", else_effect="braziers.blue = true",
        branch_a="the giant card flips face up",
        branch_b="both braziers burst into blue flames",
        judge=dict(
            action_q="Did the player raise one hand high above the head?",
            outcome_q=("At the end, is the giant card face up, or both "
                       "braziers burning blue?"),
            outcome_opts=[
                ("the card is face up", {"card.face": "up"}),
                ("the braziers burn blue", {"braziers.blue": True}),
            ],
            initial_qs=[
                "the player stands behind the card table",
                "the giant card is face down and the braziers are unlit",
            ],
        ),
    ),
    dict(  # Case 43 turn_combat 挥剑（双人）
        cid="B43", genre="turn_combat", family="trigger_cast", n_persons=2,
        style=STYLE,
        scene=("a duel arena in cool overcast light, an enemy soldier "
               "facing the player at combat distance, and a large stone "
               "magic pillar standing dark behind the enemy"),
        player=("exactly two people, both fully visible and large in the frame: the player at combat "
                "distance holding a sword ready, and an enemy standing "
                "upright on both feet"),
        target="the player's sword swing",
        objects=["enemy", "magic_pillar"],
        hidden=dict(var="combat_effect", a="ENEMY", b="PILLAR"),
        prompt_ok=["ENEMY", "PILLAR"],
        action=("The player stays in place and performs one clear forward "
                "sword swing without requiring blade contact."),
        vars={
            "enemy.kneeling": {
                False: "the enemy stands upright on both feet",
                True: "the enemy kneels down on one knee and stays kneeling",
            },
            "pillar.red": {
                False: "the large magic pillar is dark stone with no glow",
                True: "the large magic pillar burns with red flames",
            },
        },
        init={"enemy": {"kneeling": False}, "pillar": {"red": False}},
        effect="enemy.kneeling = true", else_effect="pillar.red = true",
        branch_a="the enemy kneels down and stays kneeling",
        branch_b="the magic pillar behind burns with red flames",
        judge=dict(
            action_q="Did the player perform one clear forward sword swing?",
            outcome_q=("At the end, does the enemy kneel, or the pillar "
                       "burn red?"),
            outcome_opts=[
                ("the enemy kneels", {"enemy.kneeling": True}),
                ("the pillar burns red", {"pillar.red": True}),
            ],
            initial_qs=[
                "the player faces a standing enemy",
                "the magic pillar behind the enemy is dark",
            ],
        ),
    ),
    dict(  # Case 44 survival 举臂（v5 重设计：收音机 whip 天线先验无解，
        # 收音机移出场景；载体换 火坑木柴点燃/三盏营灯点亮，营灯暗态已验证）
        cid="B44", genre="survival", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a survival camp in grey dawn light, a fire pit with a neat "
               "stack of pale grey logs, and three camp lamps hanging with "
               "dark dead glass"),
        player=("a single survivor seen in third person from behind, "
                "standing in place at the camp center"),
        target="the player's raised arm",
        objects=["fire_pit", "camp_lamps"],
        hidden=dict(var="power_mode", a="BLAZE", b="LIGHTS"),
        action="The player stays in place and extends one arm straight upward.",
        vars={
            "fire.lit": {
                False: "the stack of pale grey logs sits unlit in the fire pit",
                True: "the log stack burns with large bright orange flames",
            },
            "lamps.on": {
                False: "all three camp lamps hang with dark dead glass",
                True: "all three camp lamps glow bright white",
            },
        },
        init={"fire": {"lit": False}, "lamps": {"on": False}},
        effect="fire.lit = true", else_effect="lamps.on = true",
        branch_a="the fire pit logs burst into orange flames",
        branch_b="all three camp lamps glow white",
        judge=dict(
            action_q="Did the player extend one arm straight upward?",
            outcome_q="At the end, does the campfire burn, or the camp lamps glow?",
            outcome_opts=[
                ("the campfire burns", {"fire.lit": True}),
                ("the lamps glow", {"lamps.on": True}),
            ],
            initial_qs=[
                "the player stands at the camp center",
                "the fire pit logs are unlit and the camp lamps are dark",
            ],
        ),
    ),
    dict(  # Case 45 shopkeeping 张臂
        cid="B45", genre="shopkeeping", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a boutique interior, a large display window covered by a "
               "curtain drawn fully closed, and a big top sign panel "
               "that is dark and unlit"),
        player=("a single shopkeeper seen in third person from behind, "
                "standing in place at the shop center, arms at the sides"),
        target="the player's arm spread",
        objects=["window_curtain", "top_sign"],
        hidden=dict(var="opening_mode", a="CURTAIN", b="SIGN"),
        prompt_ok=["CURTAIN", "SIGN"],
        action="The player stays in place and clearly spreads both arms sideways.",
        vars={
            "curtain.open": {
                False: "the window curtain is drawn fully closed",
                True: "the window curtain is pulled fully open to the sides",
            },
            "sign.red": {
                False: "the big top sign panel is dark and unlit",
                True: "the big top sign panel glows bright red",
            },
        },
        init={"curtain": {"open": False}, "sign": {"red": False}},
        effect="curtain.open = true", else_effect="sign.red = true",
        branch_a="the window curtain pulls fully open",
        branch_b="the big top sign glows bright red",
        judge=dict(
            action_q="Did the player clearly spread both arms sideways?",
            outcome_q=("At the end, is the window curtain open, or the "
                       "top sign glowing red?"),
            outcome_opts=[
                ("the curtain is open", {"curtain.open": True}),
                ("the sign glows red", {"sign.red": True}),
            ],
            initial_qs=[
                "the player stands at the shop center",
                "the window curtain is closed and the sign is unlit",
            ],
        ),
    ),
    dict(  # Case 46 farming 举手
        cid="B46", genre="farming", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a farm field in mild morning light, a central potted "
               "plant bearing only green leaves, and a row of irrigation "
               "sprinkler heads standing dry and still"),
        player=("a single farmer seen in third person from behind, "
                "standing in place at the field edge"),
        target="the player's two-hand raise",
        objects=["central_plant", "sprinklers"],
        hidden=dict(var="growth_mode", a="FLOWER", b="WATER"),
        action=("The player stays in place and raises both hands in one "
                "clear casting-like gesture."),
        vars={
            "plant.flower": {
                False: "the central plant bears only green leaves",
                True: "the central plant bears one large yellow flower",
            },
            "sprinklers.on": {
                False: "the sprinkler heads stand dry and still",
                True: "all the sprinkler heads spray wide water jets",
            },
        },
        init={"plant": {"flower": False}, "sprinklers": {"on": False}},
        effect="plant.flower = true", else_effect="sprinklers.on = true",
        branch_a="the central plant opens one large yellow flower",
        branch_b="all the sprinklers spray wide water jets",
        judge=dict(
            action_q="Did the player raise both hands?",
            outcome_q=("At the end, does the plant flower yellow, or the "
                       "sprinklers spray?"),
            outcome_opts=[
                ("a yellow flower", {"plant.flower": True}),
                ("the sprinklers spray", {"sprinklers.on": True}),
            ],
            initial_qs=[
                "the player stands at the field edge",
                "the plant has only leaves and the sprinklers are still",
            ],
        ),
    ),
    dict(  # Case 47 ranch 举臂
        cid="B47", genre="ranch", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a ranch entrance in flat daylight, a large wooden "
               "windmill with its blades standing still, and a big barn "
               "door shut"),
        player=("a single rancher seen in third person from behind, "
                "standing in place at the entrance"),
        target="the player's raised arm",
        objects=["windmill", "barn_door"],
        hidden=dict(var="ranch_mode", a="WINDMILL", b="BARN"),
        prompt_ok=["WINDMILL", "BARN"],
        action="The player stays in place and raises one arm high.",
        vars={
            "windmill.spinning": {
                False: "the large windmill blades stand completely still",
                True: "the large windmill blades spin fast",
            },
            "barn.open": {
                False: "the big barn door is shut",
                True: "the big barn door is wide open",
            },
        },
        init={"windmill": {"spinning": False}, "barn": {"open": False}},
        effect="windmill.spinning = true", else_effect="barn.open = true",
        branch_a="the windmill blades start spinning fast",
        branch_b="the big barn door swings wide open",
        judge=dict(
            action_q="Did the player raise one arm high?",
            outcome_q=("At the end, does the windmill spin, or the barn "
                       "door open?"),
            outcome_opts=[
                ("the windmill spins", {"windmill.spinning": True}),
                ("the barn door is open", {"barn.open": True}),
            ],
            initial_qs=[
                "the player stands at the ranch entrance",
                "the windmill is still and the barn door is shut",
            ],
        ),
    ),
    dict(  # Case 48 sailing 指令手势
        cid="B48", genre="sailing", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a ship deck in steady daylight, the main sail furled and "
               "tied to its yard, and a large wooden water wheel on the "
               "ship's side standing still in the water"),
        player=("a single sailor seen in third person from behind, "
                "standing in place at mid-deck"),
        target="the player's two-arm command",
        objects=["main_sail", "side_water_wheel"],
        hidden=dict(var="ship_mode", a="SAIL", b="WHEEL"),
        prompt_ok=["SAIL", "WHEEL"],
        action=("The player stays in place and makes one clear forward "
                "command gesture with both arms."),
        vars={
            "sail.unfurled": {
                False: "the main sail is furled and tied to the yard",
                True: "the main sail is fully spread",
            },
            "wheel.spinning": {
                False: "the large side water wheel stands still",
                True: "the large side water wheel spins fast",
            },
        },
        init={"sail": {"unfurled": False}, "wheel": {"spinning": False}},
        effect="sail.unfurled = true", else_effect="wheel.spinning = true",
        branch_a="the main sail spreads fully out",
        branch_b="the side water wheel spins fast",
        judge=dict(
            action_q="Did the player make one clear two-arm command gesture?",
            outcome_q=("At the end, is the sail spread, or the water "
                       "wheel spinning?"),
            outcome_opts=[
                ("the sail is spread", {"sail.unfurled": True}),
                ("the wheel spins", {"wheel.spinning": True}),
            ],
            initial_qs=[
                "the player stands at mid-deck",
                "the sail is furled and the water wheel is still",
            ],
        ),
    ),
    dict(  # Case 49 dungeon 举杖
        cid="B49", genre="dungeon", family="trigger_cast", n_persons=1,
        style=STYLE,
        scene=("a grand hall in cold stone light, a large stone door shut "
               "in the left wall, and a winged stone statue on the right "
               "with its wings folded closed against its body"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place at the hall center, holding a wooden "
                "staff upright"),
        target="the player's vertical staff raise",
        objects=["left_stone_door", "winged_statue"],
        hidden=dict(var="hall_mode", a="DOOR", b="STATUE"),
        prompt_ok=["DOOR", "STATUE"],
        action="The player stays in place and raises the staff vertically.",
        vars={
            "door.up": {
                False: "the left stone door is fully lowered shut",
                True: "the left stone door is raised fully open",
            },
            "wings.open": {
                False: "the statue's wings are folded closed against its body",
                True: "the statue's wings spread fully open",
            },
        },
        init={"door": {"up": False}, "wings": {"open": False}},
        effect="door.up = true", else_effect="wings.open = true",
        branch_a="the left stone door rises fully open",
        branch_b="the statue's wings spread fully open",
        judge=dict(
            action_q="Did the player raise the staff vertically?",
            outcome_q=("At the end, does the left door rise open, or the "
                       "statue's wings spread?"),
            outcome_opts=[
                ("the left door is open", {"door.up": True}),
                ("the wings are spread", {"wings.open": True}),
            ],
            initial_qs=[
                "the player stands at the hall center with a staff",
                "the left door is shut and the statue's wings are folded",
            ],
        ),
    ),
    dict(  # Case 50 magic_academy 施法路由（画面左右）
        cid="B50", genre="magic_academy", family="trigger_cast", n_persons=1,
        style=STYLE,
        scene=("a mage training ground under a pale sky, two identical "
               "large stone braziers standing far apart, one on the "
               "viewer's left and one on the viewer's right of the "
               "wizard, both holding only dark cold ash"),
        player=("a single wizard in a grey robe seen in third person from "
                "behind, standing in place at the center between the two "
               "braziers, holding a slender wand at the side"),
        target="the wizard's wand sweep",
        objects=["viewer_left_brazier", "viewer_right_brazier"],
        hidden=dict(var="target_route", a="VIEWER_LEFT", b="VIEWER_RIGHT"),
        action=("The wizard stays in place, raises the wand high, and "
                "makes one clear forward casting sweep."),
        vars={
            "left.lit": {
                False: "the viewer's left brazier holds only dark cold ash",
                True: "the viewer's left brazier burns with large green flames",
            },
            "right.lit": {
                False: "the viewer's right brazier holds only dark cold ash",
                True: "the viewer's right brazier burns with large green flames",
            },
        },
        init={"left": {"lit": False}, "right": {"lit": False}},
        effect="left.lit = true", else_effect="right.lit = true",
        branch_a=("the brazier on the viewer's left burns with large green "
                  "flames while the right one stays dark"),
        branch_b=("the brazier on the viewer's right burns with large "
                  "green flames while the left one stays dark"),
        judge=dict(
            action_q="Did the wizard raise the wand high and sweep it forward?",
            outcome_q=("At the end, which brazier burns with green flames "
                       "— the one on the viewer's left or the viewer's "
                       "right?"),
            outcome_opts=[
                ("viewer's left brazier", {"left.lit": True}),
                ("viewer's right brazier", {"right.lit": True}),
            ],
            initial_qs=[
                "the wizard stands between two identical braziers",
                "both braziers are completely unlit",
            ],
        ),
    ),
]
