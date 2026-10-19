"""B31-B40 — 见 USAGE.md「Adding cases」;cases 31–40.

zone trigger（踏板/压力板/魔法圆/区域标记）与 gesture。B31 稻草人
软门控（soft_gate：渲染端跳过人数计数，审计端保留 VLM 人数题）。
双人 case：40。
"""
from __future__ import annotations

STYLE = ("stylized 3D third-person adventure game environment, clean uncluttered "
         "composition, readable silhouettes, medium-wide fixed gameplay camera, "
         "soft cool-neutral ambient lighting, restrained materials, "
         "no HUD, no subtitles, no readable text")

SPECS = [
    dict(  # Case 31 farming 踏板（软门控：稻草人防误检）
        cid="B31", genre="farming", family="trigger_zone", n_persons=1,
        soft_gate=True, style=STYLE,
        scene=("a farm field edge in flat morning light, a large stone "
               "floor pedal, a dry empty irrigation channel, and a big "
               "scarecrow built of a bare wooden cross frame with a "
               "crossbar, sagging rag clothes, sack head, and no real "
               "face, hands, feet, or skin, its cloth arms hanging down"),
        player=("a single farmer seen in third person from behind, "
                "standing in place beside the floor pedal"),
        target="the large floor pedal",
        objects=["floor_pedal", "channel", "scarecrow"],
        hidden=dict(var="field_mode", a="WATER", b="SCARECROW"),
        prompt_ok=["SCARECROW"],
        action="The player stays in place and presses the large floor pedal down.",
        vars={
            "channel.flowing": {
                False: "the irrigation channel is dry and empty",
                True: "fast flowing water fills the field channel",
            },
            "scarecrow.arms": {
                False: "the scarecrow's cloth arms hang down slack",
                True: "the scarecrow's arms stick out level and horizontal",
            },
        },
        init={"channel": {"flowing": False}, "scarecrow": {"arms": False}},
        effect="channel.flowing = true", else_effect="scarecrow.arms = true",
        branch_a="fast water fills the field channel",
        branch_b="the scarecrow's arms stick out level and horizontal",
        judge=dict(
            action_q="Was the large floor pedal pressed down?",
            outcome_q=("At the end, does the channel flow with water, or "
                       "do the scarecrow's arms stick out horizontal?"),
            outcome_opts=[
                ("water fills the channel", {"channel.flowing": True}),
                ("the scarecrow's arms are out", {"scarecrow.arms": True}),
            ],
            initial_qs=[
                "exactly one real person stands beside the floor pedal",
                "the scarecrow is a rag figure on a wooden cross with arms hanging down",
            ],
        ),
    ),
    dict(  # Case 32 escape_room 压力板
        cid="B32", genre="escape_room", family="trigger_zone", n_persons=1,
        style=STYLE,
        scene=("a puzzle corridor in cold dim light, a large circular "
               "pressure plate on the floor, a stone door shut at the "
               "corridor end, and a big square wall panel showing its "
               "plain flat front"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place at the edge of the pressure plate"),
        target="the large pressure plate",
        objects=["pressure_plate", "stone_door", "wall_panel"],
        hidden=dict(var="floor_mode", a="DOOR", b="PANEL"),
        prompt_ok=["DOOR", "PANEL"],
        action=("The player takes one short step forward onto the center "
                "of the large pressure plate."),
        vars={
            "door.up": {
                False: "the stone door is fully lowered shut across its doorway",
                True: "the stone door is raised fully open, the passage clear",
            },
            "panel.turned": {
                False: "the wall panel shows its plain flat front",
                True: "the wall panel is rotated half a turn onto its side",
            },
        },
        init={"door": {"up": False}, "panel": {"turned": False}},
        effect="door.up = true", else_effect="panel.turned = true",
        branch_a="the stone door rises fully open",
        branch_b="the wall panel rotates half a turn",
        judge=dict(
            action_q="Did the player step onto the center of the pressure plate?",
            outcome_q="At the end, is the stone door open, or the wall panel rotated?",
            outcome_opts=[
                ("the stone door is open", {"door.up": True}),
                ("the panel is rotated", {"panel.turned": True}),
            ],
            initial_qs=[
                "the player stands at the edge of the circular pressure plate",
                "the stone door is shut and the panel shows its flat front",
            ],
        ),
    ),
    dict(  # Case 33 magic_academy 魔法圆
        cid="B33", genre="magic_academy", family="trigger_zone", n_persons=1,
        style=STYLE,
        scene=("an arcane study in cold blue-white light, a magic circle of "
               "faint carved lines on the floor, a large dull crystal "
               "resting low on a pedestal at the circle center, and "
               "bookshelves of closed volumes around the walls"),
        player=("a single apprentice seen in third person from behind, "
                "standing in place at the circle's edge"),
        target="the magic circle",
        objects=["magic_circle", "crystal", "bookshelves"],
        hidden=dict(var="circle_mode", a="CRYSTAL", b="BOOKS"),
        prompt_ok=["CRYSTAL", "BOOKS"],
        action="The player takes one short step forward into the magic circle.",
        vars={
            "crystal.up": {
                False: "the large crystal rests low on its pedestal, dull matte glass",
                True: "the large crystal floats high in the air glowing blue",
            },
            "books.open": {
                False: "the books on the shelves are all closed",
                True: "the books on the shelves are all open with spread pages",
            },
        },
        init={"crystal": {"up": False}, "books": {"open": False}},
        effect="crystal.up = true", else_effect="books.open = true",
        branch_a="the large crystal floats up glowing blue",
        branch_b="all the shelf books flip open",
        judge=dict(
            action_q="Did the player step into the magic circle?",
            outcome_q=("At the end, does the crystal float glowing, or are "
                       "the books open?"),
            outcome_opts=[
                ("the crystal floats glowing", {"crystal.up": True}),
                ("the books are open", {"books.open": True}),
            ],
            initial_qs=[
                "the player stands at the magic circle's edge",
                "the crystal rests low and the shelf books are closed",
            ],
        ),
    ),
    dict(  # Case 34 camping 区域触发
        cid="B34", genre="camping", family="trigger_zone", n_persons=1,
        style=STYLE,
        scene=("a campsite in fading dusk light, a circular stone marker "
               "on the ground, a stone campfire ring with pale grey cold "
               "ash, and a canvas tent with its entrance flap closed"),
        player=("a single camper seen in third person from behind, "
                "standing in place at the edge of the circular marker"),
        target="the circular campsite marker",
        objects=["circular_marker", "campfire_ring", "tent"],
        hidden=dict(var="camp_trigger", a="FIRE", b="TENT"),
        prompt_ok=["TENT"],
        action=("The player takes one short step forward into the circular "
                "campsite marker."),
        vars={
            "fire.lit": {
                False: "the campfire ring holds only pale grey cold ash and charred black logs",
                True: "the campfire burns with tall bright flames",
            },
            "tent.open": {
                False: "the tent entrance flap is closed",
                True: "the tent entrance flap is wide open",
            },
        },
        init={"fire": {"lit": False}, "tent": {"open": False}},
        effect="fire.lit = true", else_effect="tent.open = true",
        branch_a="the campfire bursts into tall flames",
        branch_b="the tent entrance flap opens wide",
        judge=dict(
            action_q="Did the player step into the circular marker?",
            outcome_q="At the end, is the campfire burning, or the tent open?",
            outcome_opts=[
                ("the campfire burns", {"fire.lit": True}),
                ("the tent is open", {"tent.open": True}),
            ],
            initial_qs=[
                "the player stands at the circular camp marker",
                "the campfire ring holds only cold ash",
            ],
        ),
    ),
    dict(  # Case 35 museum 区域触发
        cid="B35", genre="museum", family="trigger_zone", n_persons=1,
        style=STYLE,
        scene=("a gallery hall in cool even light, a large brass floor "
               "marker before the center statue which faces straight "
               "forward, and a heavy curtain drawn fully closed across "
               "the back wall"),
        player=("a single visitor seen in third person from behind, "
                "standing in place just before the floor marker"),
        target="the gallery floor marker",
        objects=["floor_marker", "center_statue", "curtain"],
        hidden=dict(var="gallery_mode", a="STATUE", b="CURTAIN"),
        prompt_ok=["STATUE", "CURTAIN"],
        action=("The player takes one short step forward onto the large "
                "gallery floor marker."),
        vars={
            "statue.turned": {
                False: "the statue faces straight forward",
                True: "the statue is rotated a quarter turn to the side",
            },
            "curtain.open": {
                False: "the curtain is drawn fully closed across the wall",
                True: "the curtain is pulled fully open to the sides",
            },
        },
        init={"statue": {"turned": False}, "curtain": {"open": False}},
        effect="statue.turned = true", else_effect="curtain.open = true",
        branch_a="the statue rotates a quarter turn",
        branch_b="the back curtain pulls fully open",
        judge=dict(
            action_q="Did the player step onto the floor marker?",
            outcome_q=("At the end, is the statue rotated, or the curtain "
                       "open?"),
            outcome_opts=[
                ("the statue is rotated", {"statue.turned": True}),
                ("the curtain is open", {"curtain.open": True}),
            ],
            initial_qs=[
                "the player stands before the brass floor marker",
                "the statue faces forward and the curtain is closed",
            ],
        ),
    ),
    dict(  # Case 36 open_world 区域触发
        cid="B36", genre="open_world", family="trigger_zone", n_persons=1,
        style=STYLE,
        scene=("a ruin entrance under a pale sky, a low stone line marking "
               "the threshold, two stone pillars sunk low into the ground "
               "inside, and a wide stone fountain standing dry and still"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place one step before the entrance line"),
        target="the ruin entrance line",
        objects=["entrance_line", "stone_pillars", "fountain"],
        hidden=dict(var="zone_mode", a="PILLARS", b="FOUNTAIN"),
        prompt_ok=["PILLARS", "FOUNTAIN"],
        action=("The player takes one short step forward across the ruin "
                "entrance line."),
        vars={
            "pillars.up": {
                False: "both stone pillars are sunk low into the ground",
                True: "both stone pillars stand tall, risen from the ground",
            },
            "fountain.running": {
                False: "the stone fountain is dry and still",
                True: "the fountain jets a tall plume of water",
            },
        },
        init={"pillars": {"up": False}, "fountain": {"running": False}},
        effect="pillars.up = true", else_effect="fountain.running = true",
        branch_a="both stone pillars rise tall from the ground",
        branch_b="the fountain jets a tall plume of water",
        judge=dict(
            action_q="Did the player step across the entrance line?",
            outcome_q=("At the end, are the pillars risen tall, or the "
                       "fountain jetting water?"),
            outcome_opts=[
                ("the pillars are risen", {"pillars.up": True}),
                ("the fountain jets water", {"fountain.running": True}),
            ],
            initial_qs=[
                "the player stands one step before the entrance line",
                "the pillars are sunk low and the fountain is dry",
            ],
        ),
    ),
    dict(  # Case 37 cooking 手势
        cid="B37", genre="cooking", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a rustic kitchen control station in mild daylight, a "
               "large pot of pale broth on the stove, and a big stone "
               "bake oven whose top is cold dark metal"),
        player=("a single chef in a white apron seen in third person "
                "from behind, standing in place at the control station"),
        target="the chef's start gesture",
        objects=["soup_pot", "bake_oven"],
        hidden=dict(var="recipe_mode", a="SOUP", b="OVEN"),
        prompt_ok=["SOUP", "OVEN"],
        action="The chef stays in place and raises one hand in a clear start gesture.",
        vars={
            "soup.green": {
                False: "the broth in the large pot is plain pale",
                True: "the broth in the large pot is vivid green",
            },
            "oven.flame": {
                False: "the oven top is cold dark metal with no fire",
                True: "bright orange flames rise from the oven top",
            },
        },
        init={"soup": {"green": False}, "oven": {"flame": False}},
        effect="soup.green = true", else_effect="oven.flame = true",
        branch_a="the broth in the pot turns vivid green",
        branch_b="orange flames rise from the oven top",
        judge=dict(
            action_q="Did the chef raise one hand in a clear start gesture?",
            outcome_q=("At the end, is the broth green, or orange flames "
                       "on the oven top?"),
            outcome_opts=[
                ("the broth is green", {"soup.green": True}),
                ("flames on the oven", {"oven.flame": True}),
            ],
            initial_qs=[
                "a chef stands at the kitchen control station",
                "the broth is pale and the oven top is cold",
            ],
        ),
    ),
    dict(  # Case 38 fishing 远程收线
        cid="B38", genre="fishing", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a lakeshore in soft morning light, a fishing rod fixed on "
               "a rod rack with its line out on the water, a large red "
               "float resting flat on the surface, a reel on the rod "
               "standing still, and a shore lamp with dark dead glass"),
        player=("a single angler seen in third person from behind, "
                "standing in place on the shore beside the rod rack"),
        target="the reel-in command",
        objects=["rod_rack", "reel", "float", "shore_lamp"],
        hidden=dict(var="fishing_mode", a="FLOAT", b="LIGHT"),
        prompt_ok=["FLOAT", "LIGHT"],
        action=("The player raises the right arm in a clear reel-in "
                "command while the fixed reel visibly rotates."),
        vars={
            "float.up": {
                False: "the large red float rests flat on the water surface",
                True: "the large red float is pulled up high out of the water",
            },
            "lamp.blue": {
                False: "the shore lamp is dark with a dead bulb",
                True: "the shore lamp glows bright blue",
            },
        },
        init={"float": {"up": False}, "lamp": {"blue": False}},
        effect="float.up = true", else_effect="lamp.blue = true",
        branch_a="the large float is pulled up high out of the water",
        branch_b="the shore lamp glows bright blue",
        judge=dict(
            action_q="Did the player raise the arm while the reel visibly rotated?",
            outcome_q=("At the end, is the float pulled up out of the "
                       "water, or the shore lamp blue?"),
            outcome_opts=[
                ("the float is up", {"float.up": True}),
                ("the shore lamp glows blue", {"lamp.blue": True}),
            ],
            initial_qs=[
                "the player stands on the shore beside the rod rack",
                "the float rests on the water and the shore lamp is unlit",
            ],
        ),
    ),
    dict(  # Case 39 detective 指令手势
        cid="B39", genre="detective", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("an investigation room in cool daylight, a wide evidence "
               "wall of plain corkboard with pinned notes, and a large "
               "steel evidence cabinet with both doors shut"),
        player=("a single detective seen in third person from behind, "
                "standing in place facing the evidence wall"),
        target="the player's command gesture",
        objects=["evidence_wall", "evidence_cabinet"],
        hidden=dict(var="investigation_mode", a="WALL", b="CABINET"),
        prompt_ok=["WALL", "CABINET"],
        action=("The player stays in place and raises one arm in a clear "
                "command gesture."),
        vars={
            "wall.mark": {
                False: "the evidence wall is plain corkboard with pinned notes",
                True: "large blue glowing outlines mark items on the evidence wall",
            },
            "cabinet.open": {
                False: "the large evidence cabinet doors are shut",
                True: "the large evidence cabinet doors are wide open",
            },
        },
        init={"wall": {"mark": False}, "cabinet": {"open": False}},
        effect="wall.mark = true", else_effect="cabinet.open = true",
        branch_a="blue glowing outlines appear on the evidence wall",
        branch_b="the evidence cabinet doors open wide",
        judge=dict(
            action_q="Did the player raise one arm in a clear command gesture?",
            outcome_q=("At the end, do blue outlines mark the wall, or is "
                       "the cabinet open?"),
            outcome_opts=[
                ("blue outlines on the wall", {"wall.mark": True}),
                ("the cabinet is open", {"cabinet.open": True}),
            ],
            initial_qs=[
                "the player stands facing the evidence wall",
                "the wall is plain and the cabinet is shut",
            ],
        ),
    ),
    dict(  # Case 40 visual_novel 鞠躬（双人）
        cid="B40", genre="visual_novel", family="trigger_gesture", n_persons=2,
        style=STYLE,
        scene=("a small theater stage in cool light, an empty wooden "
               "chair at the side of the stage, and rows of stage "
               "lanterns hanging dark overhead"),
        player=("exactly two people, both fully visible and large in the frame: the player standing "
                "upright at center stage, and one NPC standing upright "
                "beside the empty chair"),
        target="the player's bow",
        objects=["npc", "chair", "stage_lanterns"],
        hidden=dict(var="scene_mode", a="NPC", b="LIGHTS"),
        prompt_ok=["NPC", "LIGHTS"],
        action="The player stays in place and performs one clear bow.",
        vars={
            "npc.seated": {
                False: "the NPC stands upright beside the empty chair",
                True: "the NPC sits down on the chair",
            },
            "lights.purple": {
                False: "the stage lanterns hang dark and unlit",
                True: "the stage lanterns all glow purple",
            },
        },
        init={"npc": {"seated": False}, "lights": {"purple": False}},
        effect="npc.seated = true", else_effect="lights.purple = true",
        branch_a="the NPC sits down on the chair",
        branch_b="all the stage lanterns glow purple",
        judge=dict(
            action_q="Did the player perform one clear bow?",
            outcome_q="At the end, does the NPC sit down, or the lanterns glow purple?",
            outcome_opts=[
                ("the NPC sits down", {"npc.seated": True}),
                ("the lanterns glow purple", {"lights.purple": True}),
            ],
            initial_qs=[
                "the player and an NPC both stand on the stage",
                "the stage lanterns are unlit",
            ],
        ),
    ),
]
