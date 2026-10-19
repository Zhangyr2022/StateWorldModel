"""B01-B10 — 见 USAGE.md「Adding cases」;cases 01–10.

v4 trigger 版：动作是易观察的 trigger event（手势/拉杆），非物理搬运；
outcome 是大视觉变化。双人 case：04。未点燃/关闭初态用正向熄灭描述
（FLUX 无负向通道）。
全局风格句见本文件顶部常量。
"""
from __future__ import annotations

STYLE = ("stylized 3D third-person adventure game environment, clean uncluttered "
         "composition, readable silhouettes, medium-wide fixed gameplay camera, "
         "soft cool-neutral ambient lighting, restrained materials, "
         "no HUD, no subtitles, no readable text")

SPECS = [
    dict(  # Case 01 magic_academy 举杖（单变量三值）
        cid="B01", genre="magic_academy", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a mage practice hall with tall arched windows and a plain "
               "stone floor, a large white magic orb resting on a stone "
               "pedestal across the room"),
        player=("a single wizard in a blue robe seen in third person from "
                "behind, standing in place at mid-room, holding a wooden "
                "wand down at the side"),
        target="the wizard's wand raise",
        objects=["magic_orb", "pedestal"],
        hidden=dict(var="element_mode", a="RED_FLAME", b="BLUE_FLAME"),
        action=("The wizard stays in place and raises the wand clearly "
                "above the head."),
        vars={
            "orb.flame": {
                "none": "the large white orb is plain matte white, completely dark with no glow",
                "red": "the orb erupts with large bright red flames",
                "blue": "the orb erupts with large bright blue flames",
            },
        },
        init={"orb": {"flame": "none"}},
        effect="orb.flame = 'red'", else_effect="orb.flame = 'blue'",
        branch_a="the magic orb erupts with large red flames",
        branch_b="the magic orb erupts with large blue flames",
        judge=dict(
            action_q="Did the wizard raise the wand clearly above the head?",
            outcome_q="At the end, does the orb burn with red flames or blue flames?",
            outcome_opts=[
                ("red flames", {"orb.flame": "red"}),
                ("blue flames", {"orb.flame": "blue"}),
            ],
            initial_qs=[
                "a wizard stands in the room holding a wand",
                "the large white orb is completely unlit",
            ],
        ),
    ),
    dict(  # Case 02 dungeon 手势
        cid="B02", genre="dungeon", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a cold stone dungeon hall, a carved statue standing against "
               "the far wall, a heavy stone door set into the right wall, "
               "and a large iron brazier on the floor"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place at mid-room facing the distant statue, "
                "arms at the sides"),
        target="the player's right-arm raise",
        objects=["statue", "stone_door", "brazier"],
        hidden=dict(var="shrine_mode", a="GATE", b="BRAZIER"),
        prompt_ok=["BRAZIER"],
        action="The player stays in place and clearly raises the right arm.",
        vars={
            "door.up": {
                False: "the stone door is fully lowered shut across its doorway",
                True: "the stone door is raised fully open, the passage clear",
            },
            "brazier.lit": {
                False: "the large brazier holds only pale grey cold ash, dark and smokeless",
                True: "the large brazier burns with tall bright flames",
            },
        },
        init={"door": {"up": False}, "brazier": {"lit": False}},
        effect="door.up = true", else_effect="brazier.lit = true",
        branch_a="the stone door rises fully open",
        branch_b="the large brazier bursts into tall flames",
        judge=dict(
            action_q="Did the player clearly raise the right arm?",
            outcome_q="At the end, is the stone door open, or the brazier burning?",
            outcome_opts=[
                ("the stone door is open", {"door.up": True}),
                ("the brazier burns", {"brazier.lit": True}),
            ],
            initial_qs=[
                "the player stands facing the distant statue",
                "the stone door is shut and the brazier holds only cold ash",
            ],
        ),
    ),
    dict(  # Case 03 open_world 手势
        cid="B03", genre="open_world", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("an ancient ruin plaza under a pale sky, a tall stone tower "
               "at the back and a wide stone fountain at the center"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place at the plaza edge, arms at the sides"),
        target="the player's two-hand raise",
        objects=["stone_tower", "fountain"],
        hidden=dict(var="relic_mode", a="TOWER", b="FOUNTAIN"),
        prompt_ok=["TOWER", "FOUNTAIN"],
        action="The player stays in place and raises both hands above the head.",
        vars={
            "tower.beam": {
                False: "the stone tower stands plain against the sky",
                True: "a strong golden beam of light shoots from the top of the tower",
            },
            "fountain.running": {
                False: "the stone fountain is dry and still",
                True: "the stone fountain jets a tall plume of water",
            },
        },
        init={"tower": {"beam": False}, "fountain": {"running": False}},
        effect="tower.beam = true", else_effect="fountain.running = true",
        branch_a="a strong golden beam shoots from the tower top",
        branch_b="the fountain jets a tall plume of water",
        judge=dict(
            action_q="Did the player raise both hands above the head?",
            outcome_q=("At the end, does the tower show a golden beam, or "
                       "does the fountain jet water?"),
            outcome_opts=[
                ("a golden beam on the tower", {"tower.beam": True}),
                ("the fountain jets water", {"fountain.running": True}),
            ],
            initial_qs=[
                "the player stands at the ruin plaza",
                "the fountain is dry and the tower is plain",
            ],
        ),
    ),
    dict(  # Case 04 healer 手势（双人）
        cid="B04", genre="healer", family="trigger_gesture", n_persons=2,
        soft_gate=True,
        style=STYLE,
        scene=("a calm infirmary in cool north light, a bed in the near "
               "foreground, and a large healing crystal resting on a low "
               "stand beside the bed"),
        player=("exactly two people, both fully visible and large in the frame: a "
                "healer in white robes standing at the bedside with arms at "
                "the sides, and the patient standing upright beside the "
                "foreground bed, both figures full-body"),
        target="the healer's arm spread",
        objects=["patient_bed", "healing_crystal"],
        hidden=dict(var="healing_mode", a="PATIENT", b="CRYSTAL"),
        prompt_ok=["PATIENT", "CRYSTAL"],
        action="The healer stays in place and clearly spreads both arms.",
        vars={
            "patient.aura": {
                False: "the patient stands beside the bed with no glow around them",
                True: "the patient is surrounded by a large bright green glowing aura",
            },
            "crystal.up": {
                False: "the large healing crystal rests low on its stand, dull matte glass",
                True: "the large healing crystal rises up glowing bright white",
            },
        },
        init={"patient": {"aura": False}, "crystal": {"up": False}},
        effect="patient.aura = true", else_effect="crystal.up = true",
        branch_a="a large green glowing aura surrounds the patient",
        branch_b="the healing crystal rises up glowing bright white",
        judge=dict(
            action_q="Did the healer clearly spread both arms?",
            outcome_q=("At the end, does the patient have a green aura, or "
                       "does the crystal rise glowing white?"),
            outcome_opts=[
                ("a green aura on the patient", {"patient.aura": True}),
                ("the crystal rises glowing", {"crystal.up": True}),
            ],
            initial_qs=[
                "a healer and a standing patient are both beside the bed",
                "the healing crystal is dull and resting low",
            ],
        ),
    ),
    dict(  # Case 05 visual_novel 挥手（双人）
        cid="B05", genre="visual_novel", family="trigger_gesture", n_persons=2,
        style=STYLE,
        scene=("a small theater stage with a heavy red curtain drawn fully "
               "closed across the back wall and a row of stage lanterns "
               "hanging dark overhead"),
        player=("exactly two people, both fully visible and large in the frame: "
                "the player standing at center stage with arms at the sides, "
                "and one NPC standing clearly beside the player, the camera "
                "at gameplay distance on the stage boards"),
        target="the player's wave",
        objects=["red_curtain", "stage_lanterns"],
        hidden=dict(var="scene_flag", a="CURTAIN", b="LIGHTS"),
        prompt_ok=["CURTAIN", "LIGHTS"],
        action="The player stays in place and gives one clear wave.",
        vars={
            "curtain.open": {
                False: "the red curtain is drawn fully closed across the stage",
                True: "the red curtain is pulled fully open to the sides",
            },
            "lights.lit": {
                False: "the stage lanterns hang dark and unlit",
                True: "the stage lanterns all shine bright",
            },
        },
        init={"curtain": {"open": False}, "lights": {"lit": False}},
        effect="curtain.open = true", else_effect="lights.lit = true",
        branch_a="the red curtain pulls fully open",
        branch_b="all the stage lanterns light up",
        judge=dict(
            action_q="Did the player give one clear wave?",
            outcome_q="At the end, is the curtain open, or the stage lanterns lit?",
            outcome_opts=[
                ("the curtain is open", {"curtain.open": True}),
                ("the lanterns are lit", {"lights.lit": True}),
            ],
            initial_qs=[
                "the player and an NPC stand on the stage",
                "the red curtain is closed and the lanterns are unlit",
            ],
        ),
    ),
    dict(  # Case 06 dating 鼓掌（双人）
        cid="B06", genre="dating", family="trigger_gesture", n_persons=2,
        style=STYLE,
        scene=("a garden terrace with a white pergola frame whose canopy is "
               "folded up against the frame, and strings of heart-shaped "
               "lanterns hanging dark between the posts"),
        player=("exactly two people, both fully visible: the player standing on "
                "the terrace with arms at the sides, and one NPC standing "
                "clearly opposite the player"),
        target="the player's clap",
        objects=["pergola_canopy", "heart_lanterns"],
        hidden=dict(var="date_scene", a="CANOPY", b="LIGHTS"),
        prompt_ok=["CANOPY"],
        action="The player stays in place and performs one clear clap.",
        vars={
            "canopy.out": {
                False: "the pergola canopy is folded and retracted against its frame",
                True: "the pergola canopy is fully spread out overhead",
            },
            "hearts.lit": {
                False: "the heart-shaped lanterns hang dark and unlit",
                True: "the heart-shaped lanterns all glow bright pink",
            },
        },
        init={"canopy": {"out": False}, "hearts": {"lit": False}},
        effect="canopy.out = true", else_effect="hearts.lit = true",
        branch_a="the pergola canopy spreads fully out overhead",
        branch_b="the heart lanterns glow bright pink",
        judge=dict(
            action_q="Did the player perform one clear clap?",
            outcome_q=("At the end, is the canopy spread out, or the heart "
                       "lanterns glowing pink?"),
            outcome_opts=[
                ("the canopy is spread", {"canopy.out": True}),
                ("the heart lanterns glow", {"hearts.lit": True}),
            ],
            initial_qs=[
                "the player and an NPC stand on the garden terrace",
                "the heart lanterns are unlit",
            ],
        ),
    ),
    dict(  # Case 07 museum 指向
        cid="B07", genre="museum", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a gallery hall with a large stone statue on a plinth at the "
               "center, and a heavy curtain drawn fully closed across the "
               "back wall"),
        player=("a single visitor seen in third person from behind, "
                "standing in place facing the statue, arms at the sides"),
        target="the player's pointing arm",
        objects=["large_statue", "gallery_curtain"],
        hidden=dict(var="exhibit_mode", a="STATUE", b="CURTAIN"),
        prompt_ok=["STATUE", "CURTAIN"],
        action="The player raises one arm and clearly points toward the statue.",
        vars={
            "statue.pattern": {
                False: "the statue surface is plain carved stone",
                True: "the statue surface is covered with bright blue glowing tracery",
            },
            "curtain.open": {
                False: "the curtain is drawn fully closed across the wall",
                True: "the curtain is pulled fully open to the sides",
            },
        },
        init={"statue": {"pattern": False}, "curtain": {"open": False}},
        effect="statue.pattern = true", else_effect="curtain.open = true",
        branch_a="bright blue tracery covers the statue",
        branch_b="the back curtain pulls fully open",
        judge=dict(
            action_q="Did the player raise one arm and point at the statue?",
            outcome_q=("At the end, does the statue show blue tracery, or "
                       "is the curtain open?"),
            outcome_opts=[
                ("blue tracery on the statue", {"statue.pattern": True}),
                ("the curtain is open", {"curtain.open": True}),
            ],
            initial_qs=[
                "the player stands facing the large statue",
                "the statue is plain and the curtain is closed",
            ],
        ),
    ),
    dict(  # Case 08 survival 手势
        cid="B08", genre="survival", family="trigger_gesture", n_persons=1,
        style=STYLE,
        scene=("a survival camp in flat grey daylight, a stone campfire ring "
               "and a tall pole with a red signal flag furled down around "
               "the pole"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place at the camp center, arms at the sides"),
        target="the player's V-arm gesture",
        objects=["campfire_ring", "signal_pole"],
        hidden=dict(var="signal_mode", a="FIRE", b="FLAG"),
        prompt_ok=["FLAG"],
        action="The player stays in place and raises both arms into a clear V shape.",
        vars={
            "fire.lit": {
                False: "the campfire ring holds only pale grey cold ash and charred black logs",
                True: "the campfire burns with tall bright flames",
            },
            "flag.up": {
                False: "the red signal flag is furled down around its pole",
                True: "the large red signal flag is raised high on its pole",
            },
        },
        init={"fire": {"lit": False}, "flag": {"up": False}},
        effect="fire.lit = true", else_effect="flag.up = true",
        branch_a="the campfire bursts into tall flames",
        branch_b="the red signal flag rises high on its pole",
        judge=dict(
            action_q="Did the player raise both arms into a clear V shape?",
            outcome_q="At the end, is the campfire burning, or the red flag raised?",
            outcome_opts=[
                ("the campfire burns", {"fire.lit": True}),
                ("the flag is raised", {"flag.up": True}),
            ],
            initial_qs=[
                "the player stands at the camp center",
                "the campfire ring holds only cold ash",
            ],
        ),
    ),
    dict(  # Case 09 escape_room 拉杆
        cid="B09", genre="escape_room", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a puzzle chamber with a large iron wall lever set at chest "
               "height in its upper position, a stone door in the far wall, "
               "and a large wooden drawer set flush into the side wall"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place beside the wall lever with a hand at it"),
        target="the large wall lever",
        objects=["wall_lever", "stone_door", "wall_drawer"],
        hidden=dict(var="mechanism_mode", a="DOOR", b="DRAWER"),
        prompt_ok=["DOOR", "DRAWER"],
        action=("The large wall lever moves clearly from the upper position "
                "to the lower position."),
        vars={
            "door.up": {
                False: "the stone door is fully lowered shut across its doorway",
                True: "the stone door is raised fully open, the passage clear",
            },
            "drawer.out": {
                False: "the large wall drawer is pushed flush shut into the wall",
                True: "the large wall drawer is popped clearly out of the wall",
            },
        },
        init={"door": {"up": False}, "drawer": {"out": False}},
        effect="door.up = true", else_effect="drawer.out = true",
        branch_a="the stone door rises fully open",
        branch_b="the large wall drawer pops clearly out",
        judge=dict(
            action_q="Was the large wall lever pulled down to the lower position?",
            outcome_q="At the end, is the stone door open, or the drawer popped out?",
            outcome_opts=[
                ("the stone door is open", {"door.up": True}),
                ("the drawer is popped out", {"drawer.out": True}),
            ],
            initial_qs=[
                "the player stands beside the wall lever",
                "the stone door and the wall drawer are both shut",
            ],
        ),
    ),
    dict(  # Case 10 dungeon 拉杆
        cid="B10", genre="dungeon", family="trigger_mechanism", n_persons=1,
        style=STYLE,
        scene=("a chasm chamber in cold stone light, a heavy stone lever on "
               "a pedestal, a wide dark chasm gap in the floor spanned by "
               "nothing, a stone drawbridge standing fully vertical and "
               "flush against the far cliff wall above the gap, and two "
               "large cold iron braziers filled with dark dead coals and no embers flanking the gap"),
        player=("a single adventurer seen in third person from behind, "
                "standing in place beside the stone lever with a hand on it"),
        target="the stone lever",
        objects=["stone_lever", "drawbridge", "braziers"],
        hidden=dict(var="lever_mode", a="BRIDGE", b="FIRE"),
        prompt_ok=["BRIDGE"],
        action="The stone lever is clearly pushed down.",
        vars={
            "bridge.down": {
                False: "the stone drawbridge stands fully vertical, flush against the far wall above the open gap",
                True: "the stone drawbridge lies flat across the gap",
            },
            "braziers.lit": {
                False: "both large braziers hold only pale grey cold ash with dark dead coals, no embers and no smoke",
                True: "both large braziers burn with tall bright flames",
            },
        },
        init={"bridge": {"down": False}, "braziers": {"lit": False}},
        effect="bridge.down = true", else_effect="braziers.lit = true",
        branch_a="the stone drawbridge drops flat across the gap",
        branch_b="both braziers burst into tall flames",
        judge=dict(
            action_q="Was the stone lever clearly pushed down?",
            outcome_q=("At the end, is the drawbridge flat across the gap, "
                       "or both braziers burning?"),
            outcome_opts=[
                ("the drawbridge is down", {"bridge.down": True}),
                ("the braziers burn", {"braziers.lit": True}),
            ],
            initial_qs=[
                "the player stands beside the stone lever",
                "the drawbridge stands vertical against the wall and both braziers are dead cold",
            ],
        ),
    ),
]
