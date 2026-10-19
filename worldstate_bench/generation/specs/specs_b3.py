"""B21-B30 — 见 USAGE.md「Adding cases」;cases 21–30.

cast/strike（举杖劈雕像/治疗施法/斧劈/扫描/拉弓/吹哨/敲钟）与
mechanism（服务铃）。双人 case：22、26。
"""
from __future__ import annotations

STYLE = ("stylized 3D third-person adventure game environment, clean uncluttered "
         "composition, readable silhouettes, medium-wide fixed gameplay camera, "
         "soft cool-neutral ambient lighting, restrained materials, "
         "no HUD, no subtitles, no readable text")

SPECS = [
    dict(  # Case 21 dungeon 举杖
        cid="B21", genre="dungeon", family="trigger_cast", n_persons=1,
        style=STYLE,
        scene=("a crypt hall in cold stone light, a large carved skull "
               "statue standing at the center, and a big stone door shut "
               "in the wall behind the statue"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place facing the skull statue, holding a "
                "wooden staff upright"),
        target="the player's staff sweep",
        objects=["skull_statue", "stone_door"],
        hidden=dict(var="curse_mode", a="STATUE", b="GATE"),
        prompt_ok=["STATUE"],
        action=("The player raises the staff and makes one clear downward "
                "casting sweep."),
        vars={
            "statue.state": {
                "whole": "the large skull statue is whole and intact",
                "cracked": "the skull statue is cracked apart into two pieces",
            },
            "door.open": {
                False: "the big stone door is shut in the wall",
                True: "the big stone door is wide open",
            },
        },
        init={"statue": {"state": "whole"}, "door": {"open": False}},
        effect="statue.state = 'cracked'", else_effect="door.open = true",
        branch_a="the skull statue cracks apart into two pieces",
        branch_b="the big stone door behind it opens wide",
        judge=dict(
            action_q="Did the player raise the staff and sweep it down once?",
            outcome_q="At the end, is the statue cracked apart, or the stone door open?",
            outcome_opts=[
                ("the statue is cracked", {"statue.state": "cracked"}),
                ("the door is open", {"door.open": True}),
            ],
            initial_qs=[
                "the player stands facing a skull statue",
                "the statue is whole and the stone door is shut",
            ],
        ),
    ),
    dict(  # Case 22 healer 施法（双人）
        cid="B22", genre="healer", family="trigger_cast", n_persons=2,
        style=STYLE,
        scene=("a bright infirmary room, a patient standing by a bed, and "
               "a potted plant beside the window bearing only a closed "
               "bud"),
        player=("exactly two people, both fully visible and large in the frame: a healer in white robes "
                "standing at arm's length holding a healing staff, and a "
                "patient standing calmly facing the healer"),
        target="the healer's staff pulse",
        objects=["patient", "potted_plant"],
        hidden=dict(var="heal_route", a="PATIENT", b="PLANT"),
        prompt_ok=["PATIENT", "PLANT"],
        action=("The healer raises the healing staff and releases one "
                "clearly visible pulse of energy."),
        vars={
            "patient.aura": {
                False: "the patient stands with no glow around them",
                True: "the patient is surrounded by a large bright green glowing aura",
            },
            "plant.bloomed": {
                False: "the potted plant bears only a closed tight bud",
                True: "the potted plant bears one large open white flower",
            },
        },
        init={"patient": {"aura": False}, "plant": {"bloomed": False}},
        effect="patient.aura = true", else_effect="plant.bloomed = true",
        branch_a="a large green glowing aura surrounds the patient",
        branch_b="the plant opens one large white flower",
        judge=dict(
            action_q="Did the healer release one visible pulse of light from the staff?",
            outcome_q=("At the end, does the patient have a green aura, or "
                       "a white flower on the plant?"),
            outcome_opts=[
                ("a green aura on the patient", {"patient.aura": True}),
                ("a white flower on the plant", {"plant.bloomed": True}),
            ],
            initial_qs=[
                "a healer with a staff stands near a patient",
                "the potted plant has only a closed bud",
            ],
        ),
    ),
    dict(  # Case 23 survival 斧劈
        cid="B23", genre="survival", family="trigger_cast", n_persons=1,
        style=STYLE,
        scene=("a forest clearing in grey afternoon light, a large whole "
               "wood stump at the center, and a stone campfire ring with "
               "pale grey cold ash behind it"),
        player=("a single woodsman seen in third person from behind, "
                "standing in place beside the stump with an axe held "
                "ready"),
        target="the player's axe swing",
        objects=["wood_stump", "campfire_ring"],
        hidden=dict(var="survival_mode", a="LOG", b="FIRE"),
        prompt_ok=["LOG"],
        action="The player stays in place and performs one clear downward axe swing.",
        vars={
            "stump.state": {
                "whole": "the large wood stump is whole, a single solid round log",
                "split": "the large wood stump is split into two halves",
            },
            "fire.lit": {
                False: "the campfire ring holds only pale grey cold ash and charred black logs",
                True: "the campfire burns with tall bright flames",
            },
        },
        init={"stump": {"state": "whole"}, "fire": {"lit": False}},
        effect="stump.state = 'split'", else_effect="fire.lit = true",
        branch_a="the stump splits into two halves",
        branch_b="the campfire behind bursts into tall flames",
        judge=dict(
            action_q="Did the player perform one clear downward axe swing?",
            outcome_q="At the end, is the stump split, or the campfire burning?",
            outcome_opts=[
                ("the stump is split", {"stump.state": "split"}),
                ("the campfire burns", {"fire.lit": True}),
            ],
            initial_qs=[
                "the player stands beside a whole wood stump with an axe",
                "the campfire ring holds only cold ash",
            ],
        ),
    ),
    dict(  # Case 24 museum 扫描
        cid="B24", genre="museum", family="trigger_cast", n_persons=1,
        style=STYLE,
        scene=("a conservation hall in cool even light, a large stone "
               "statue on a plinth, and a transparent glass display case "
               "sitting low over its base beside the statue"),
        player=("a single conservator seen in third person from behind, "
                "standing in place facing the statue, holding a handheld "
                "scanner at the side"),
        target="the scanner beam",
        objects=["stone_statue", "glass_case"],
        hidden=dict(var="scan_mode", a="PATTERN", b="CASE"),
        prompt_ok=["CASE"],
        action=("The player raises the scanner toward the statue and emits "
                "one clearly visible scanning beam."),
        vars={
            "statue.pattern": {
                False: "the statue surface is plain carved stone",
                True: "the statue surface is covered with large blue glowing tracery",
            },
            "case.up": {
                False: "the glass display case sits low over its base",
                True: "the glass display case is raised high and open above its base",
            },
        },
        init={"statue": {"pattern": False}, "case": {"up": False}},
        effect="statue.pattern = true", else_effect="case.up = true",
        branch_a="large blue tracery covers the statue",
        branch_b="the glass case rises high and open",
        judge=dict(
            action_q="Did a scanning beam sweep toward the statue?",
            outcome_q=("At the end, does the statue show blue tracery, or "
                       "is the case raised open?"),
            outcome_opts=[
                ("blue tracery on the statue", {"statue.pattern": True}),
                ("the case is raised open", {"case.up": True}),
            ],
            initial_qs=[
                "the player faces the statue holding a scanner",
                "the statue is plain and the case sits low",
            ],
        ),
    ),
    dict(  # Case 25 open_world 拉弓
        cid="B25", genre="open_world", family="trigger_cast", n_persons=1,
        style=STYLE,
        scene=("an ancient field under a pale sky, a tall stone monolith "
               "at the center, and two fire towers flanking it holding "
               "only dark cold ash"),
        player=("a single ranger seen in third person from behind, "
                "standing in place facing the monolith, holding an "
                "unstrung bow at the side"),
        target="the player's bow release",
        objects=["monolith", "fire_towers"],
        hidden=dict(var="relic_mode", a="MONOLITH", b="TORCHES"),
        prompt_ok=["MONOLITH"],
        action=("The player stays in place, clearly draws the bow, and "
                "releases once."),
        vars={
            "monolith.lit": {
                False: "the tall stone monolith is plain grey rock",
                True: "the whole monolith glows strong purple",
            },
            "towers.lit": {
                False: "both fire towers hold only dark cold ash",
                True: "both fire towers burn with large bright flames",
            },
        },
        init={"monolith": {"lit": False}, "towers": {"lit": False}},
        effect="monolith.lit = true", else_effect="towers.lit = true",
        branch_a="the whole monolith glows strong purple",
        branch_b="both fire towers burst into large flames",
        judge=dict(
            action_q="Did the player clearly draw the bow and release once?",
            outcome_q=("At the end, does the monolith glow purple, or do "
                       "both towers burn?"),
            outcome_opts=[
                ("the monolith glows purple", {"monolith.lit": True}),
                ("both towers burn", {"towers.lit": True}),
            ],
            initial_qs=[
                "the player stands facing the tall monolith",
                "both fire towers hold only cold ash",
            ],
        ),
    ),
    dict(  # Case 26 stealth 吹哨（双人）
        cid="B26", genre="stealth", family="trigger_gesture", n_persons=2,
        soft_gate=True,
        style=STYLE,
        scene=("a guarded courtyard in cold blue-grey light, a low cover "
               "crate beside the player, a guard standing at mid-distance "
               "across the yard full-body and clearly visible head to toe, "
               "and a large alarm lamp high on the wall with a "
               "dark dead lens"),
        player=("exactly two people, both fully visible and large in the frame: the player crouched in "
                "place beside the cover, and a guard standing at "
                "mid-distance across the yard, full-body and clearly "
                "visible head to toe"),
        target="the player's whistle gesture",
        objects=["cover_crate", "guard", "alarm_lamp"],
        hidden=dict(var="security_mode", a="GUARD", b="ALARM"),
        prompt_ok=["GUARD", "ALARM"],
        action=("The player stays behind the cover and performs one clear "
                "whistling gesture."),
        vars={
            "guard.near": {
                False: "the guard stands at mid-distance across the yard, full-body visible",
                True: "the guard stands in the near foreground beside the cover",
            },
            "alarm.red": {
                False: "the large alarm lamp is dark with a dead lens",
                True: "the large alarm lamp flashes bright red",
            },
        },
        init={"guard": {"near": False}, "alarm": {"red": False}},
        effect="guard.near = true", else_effect="alarm.red = true",
        branch_a="the guard walks up to the near foreground",
        branch_b="the alarm lamp flashes bright red",
        judge=dict(
            action_q="Did the player perform one clear whistling gesture?",
            outcome_q=("At the end, is the guard in the foreground, or the "
                       "alarm lamp flashing red?"),
            outcome_opts=[
                ("the guard is in the foreground", {"guard.near": True}),
                ("the alarm flashes red", {"alarm.red": True}),
            ],
            initial_qs=[
                "the player crouches beside a cover with a guard visible at mid-distance",
                "the alarm lamp is unlit",
            ],
        ),
    ),
    dict(  # Case 27 ranch 吹哨
        cid="B27", genre="ranch", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a ranch pasture in flat daylight, the player at the center, "
               "two sheep grazing far away, and a big barn door shut"),
        player=("a single rancher seen in third person from behind, "
                "standing in place at the pasture center with a hand at "
                "the mouth"),
        target="the player's whistle gesture",
        objects=["sheep", "barn_door"],
        hidden=dict(var="whistle_mode", a="HERD", b="BARN"),
        prompt_ok=["BARN"],
        action="The player stays in place and performs one clear whistling gesture.",
        vars={
            "sheep.near": {
                False: "the two sheep graze far away across the pasture",
                True: "the two sheep stand close beside the player",
            },
            "barn.open": {
                False: "the big barn door is shut",
                True: "the big barn door is wide open",
            },
        },
        init={"sheep": {"near": False}, "barn": {"open": False}},
        effect="sheep.near = true", else_effect="barn.open = true",
        branch_a="the two sheep come to stand close beside the player",
        branch_b="the big barn door swings wide open",
        judge=dict(
            action_q="Did the player perform one clear whistling gesture?",
            outcome_q=("At the end, are the sheep close to the player, or "
                       "the barn door open?"),
            outcome_opts=[
                ("the sheep are close", {"sheep.near": True}),
                ("the barn door is open", {"barn.open": True}),
            ],
            initial_qs=[
                "the player stands at the pasture center",
                "the sheep are far away and the barn door is shut",
            ],
        ),
    ),
    dict(  # Case 28 sailing 敲钟
        cid="B28", genre="sailing", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a ship deck in steady daylight, a large bronze bell on a "
               "frame at the bow, the main sail furled and tied to its "
               "yard, and a stern signal lamp with dark dead glass"),
        player=("a single sailor seen in third person from behind, "
                "standing in place at the bell with a striker in hand"),
        target="the large deck bell",
        objects=["deck_bell", "main_sail", "stern_lamp"],
        hidden=dict(var="ship_signal", a="SAIL", b="LIGHT"),
        prompt_ok=["SAIL"],
        action="The player strikes the large deck bell once.",
        vars={
            "sail.unfurled": {
                False: "the main sail is furled and tied to the yard",
                True: "the main sail is fully spread",
            },
            "lamp.red": {
                False: "the stern signal lamp is dark with a dead bulb",
                True: "the stern signal lamp glows bright red",
            },
        },
        init={"sail": {"unfurled": False}, "lamp": {"red": False}},
        effect="sail.unfurled = true", else_effect="lamp.red = true",
        branch_a="the main sail spreads fully out",
        branch_b="the stern signal lamp glows bright red",
        judge=dict(
            action_q="Did the player strike the large bell once?",
            outcome_q=("At the end, is the main sail spread, or the stern "
                       "lamp glowing red?"),
            outcome_opts=[
                ("the sail is spread", {"sail.unfurled": True}),
                ("the stern lamp glows red", {"lamp.red": True}),
            ],
            initial_qs=[
                "the player stands at the deck bell with a striker",
                "the sail is furled and the stern lamp is unlit",
            ],
        ),
    ),
    dict(  # Case 29 cooking 服务铃
        cid="B29", genre="cooking", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a rustic kitchen in mild daylight, a large service bell on "
               "the prep counter, a big oven with its door shut, and a "
               "large pot of pale broth on the stove"),
        player=("a single cook seen in third person from behind, "
                "standing in place at the counter with a hand over the "
                "bell"),
        target="the large service bell",
        objects=["service_bell", "oven", "soup_pot"],
        hidden=dict(var="kitchen_mode", a="OVEN", b="POT"),
        prompt_ok=["OVEN", "POT"],
        action="The player presses the large kitchen service bell once.",
        vars={
            "oven.open": {
                False: "the big oven door is shut",
                True: "the big oven door is wide open",
            },
            "soup.red": {
                False: "the broth in the large pot is plain pale",
                True: "the broth in the large pot is vivid fresh red",
            },
        },
        init={"oven": {"open": False}, "soup": {"red": False}},
        effect="oven.open = true", else_effect="soup.red = true",
        branch_a="the big oven door swings wide open",
        branch_b="the broth in the pot turns vivid red",
        judge=dict(
            action_q="Was the large service bell pressed once?",
            outcome_q="At the end, is the oven open, or the broth red?",
            outcome_opts=[
                ("the oven is open", {"oven.open": True}),
                ("the broth is red", {"soup.red": True}),
            ],
            initial_qs=[
                "the player stands with a hand over the service bell",
                "the oven is shut and the broth is pale",
            ],
        ),
    ),
    dict(  # Case 30 shopkeeping 鼓掌
        cid="B30", genre="shopkeeping", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a shop interior, the player behind the counter, a large "
               "shelf pushed flush into the back wall, and a top sign "
               "panel that is dark and unlit"),
        player=("a single shopkeeper seen in third person from behind, "
                "standing in place behind the counter, arms at the sides"),
        target="the player's clap",
        objects=["back_shelf", "top_sign"],
        hidden=dict(var="store_scene", a="SHELF", b="SIGN"),
        prompt_ok=["SHELF", "SIGN"],
        action="The player stays in place and performs one clear clap.",
        vars={
            "shelf.out": {
                False: "the large shelf is pushed flush into the wall",
                True: "the large shelf is slid out toward the front",
            },
            "sign.green": {
                False: "the top sign panel is dark and unlit",
                True: "the top sign panel glows bright green",
            },
        },
        init={"shelf": {"out": False}, "sign": {"green": False}},
        effect="shelf.out = true", else_effect="sign.green = true",
        branch_a="the large shelf slides out toward the front",
        branch_b="the top sign glows bright green",
        judge=dict(
            action_q="Did the player perform one clear clap?",
            outcome_q=("At the end, is the shelf slid out, or the sign "
                       "glowing green?"),
            outcome_opts=[
                ("the shelf is out", {"shelf.out": True}),
                ("the sign glows green", {"sign.green": True}),
            ],
            initial_qs=[
                "the player stands behind the shop counter",
                "the shelf is flush and the top sign is unlit",
            ],
        ),
    ),
]
