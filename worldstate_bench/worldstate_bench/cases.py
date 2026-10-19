"""打包核心：cases/<CASE>/ = 共享首帧 + A/B twin 的 json + card（幂等）。

card 按 README/USAGE 的数据字段说明输出：genre / interaction_family /
hidden_state / rule / shared_action / gold_A / gold_B / 三类 judge /
constraints（identical frame、hidden 不泄露、同 action、双 outcome 变化、
单 action、2-5 秒）。
"""
from __future__ import annotations

from pathlib import Path

import yaml

from worldstate_bench.config import Config

FAMILY_ZH = {
    "give_use_character": "递交/作用于角色",
    "use_item_object": "道具作用于物体",
    "place_add": "放置/加入",
    "collect_pull": "拿取/收拉",
    "enter_move": "进入/移动",
    "strike_cast": "打击/施放",
    "operate_manipulate": "操纵/机关",
}

GENRE_ZH = {
    "dating": "恋爱", "card_game": "卡牌", "escape_room": "密室", "cooking": "烹饪",
    "farming": "农耕", "fishing": "钓鱼", "stealth": "潜行", "detective": "侦探",
    "board_game": "桌游", "ranch": "牧场", "sailing": "航海", "camping": "露营",
    "museum": "博物馆", "open_world": "开放世界", "turn_combat": "回合战斗",
    "healer": "治疗", "visual_novel": "视觉小说", "shopkeeping": "经营",
    "magic_academy": "魔法学院", "dungeon": "地牢", "survival": "生存",
}


def _fmt(v) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


def card(d: dict) -> str:
    gold = "; ".join(f"{k}={_fmt(v)}" for k, v in d["gold_end_state"].items())
    delta = "; ".join(f"{k}={_fmt(v)}" for k, v in d["delta"].items())
    hs = d["hidden_state"]
    fam_zh = FAMILY_ZH.get(d["interaction_family"], d["interaction_family"])
    gen_zh = GENRE_ZH.get(d["genre"], d["genre"])
    jd = d["judge"]
    initq = "\n".join(f"  Q: {q['question']}  answers={q['answers']} "
                      f"expected={q['expected']}" for q in jd["initial_questions"])
    tvs = d["terminal_visual_states"]
    tvs_block = "\n".join([
        f"  occurs     (final frame: YES): {tvs['occurs']['phrase']}"
        f"  [{tvs['occurs']['variable']}={_fmt(tvs['occurs']['value'])}]",
        f"  not_occurs (final frame: NO) : {tvs['not_occurs']['phrase']}"
        f"  [{tvs['not_occurs']['variable']}={_fmt(tvs['not_occurs']['value'])}]",
        f"  instead remains: {tvs['not_occurs']['instead_remains']}",
    ])
    vocab = "\n".join(f"  - {p}" for m in d["visual_vocab"].values()
                      for p in m.values())
    c = d["constraints"]
    return "\n".join([
        f"[{d['id']}]  genre: {d['genre']}  family: {d['interaction_family']}  "
        f"counterpart: {d['counterpart']}",
        "input image: input.png (shared by both twins)",
        "",
        "── Inputs to the video model ──",
        f"hidden state: {hs['variable']} = "
        f"{hs['A'] if d['id'].endswith('-A') else hs['B']} "
        f"(runtime only, not visible in the frame)",
        f"rule: {d['rule']['text']}",
        f"shared action: {d['shared_action']}",
        f"instruction: {d['instruction']}",
        f"chunk: one action-consequence transition, {c['duration_seconds']} s",
        "",
        "── Gold ──",
        f"gold_A: {d['gold_A']}",
        f"gold_B: {d['gold_B']}",
        f"gold_end_state (this twin): {gold}",
        f"delta (this twin): {delta}",
        "",
        "── Terminal visual states (both, explicit) ──",
        tvs_block,
        f"outcome not_expected: {jd['outcome_question']['not_expected']}",
        "",
        "── Judges ──",
        f"initial questions:\n{initq}",
        f"action question:\n  Q: {jd['action_question']['question']}  "
        f"answers={jd['action_question']['answers']} "
        f"expected={jd['action_question']['expected']}",
        f"outcome question:\n  Q: {jd['outcome_question']['question']}  "
        f"answers={jd['outcome_question']['answers']} "
        f"expected={jd['outcome_question']['expected']}",
        "",
        "── Visual vocab (for auditing the shared first frame) ──",
        vocab,
        "",
        "── Constraints ──",
        f"identical_initial_frame={c['identical_initial_frame']}  "
        f"hidden_state_not_visually_leaked={c['hidden_state_not_visually_leaked']}  "
        f"same_action_across_twins={c['same_action_across_twins']}",
        f"both_outcomes_change={c['both_outcomes_change']}  "
        f"one_action_per_chunk={c['one_action_per_chunk']}  "
        f"duration_seconds={c['duration_seconds']}",
        "",
        f"[zh] {d['id']}（{gen_zh}/{fam_zh}）："
        f"hidden {hs['variable']}={hs['A'] if d['id'].endswith('-A') else hs['B']} → "
        f"gold [{'A 支线' if d['id'].endswith('-A') else 'B 支线'}] "
        f"{d['gold_A'] if d['id'].endswith('-A') else d['gold_B']}；"
        f"终态: 发生「{tvs['occurs']['phrase']}」且不发生"
        f"「{tvs['not_occurs']['phrase']}」；"
        f"outcome 判定期望: {jd['outcome_question']['expected']}",
    ]) + "\n"


def main(cfg: Config) -> int:
    import json
    import shutil

    cases_dir = cfg.out_cases
    if cases_dir.exists():
        shutil.rmtree(cases_dir)
    n, missing = 0, []
    for f in sorted(cfg.out_yaml.glob("*-A.yaml")):
        cid = f.name[:-len("-A.yaml")]
        out = cases_dir / cid
        out.mkdir(parents=True, exist_ok=True)
        src = cfg.out_frames / f"{cid}.png"
        if src.exists():
            shutil.copy2(src, out / "input.png")
        else:
            missing.append(str(src))
        for twin in ("A", "B"):
            d = yaml.safe_load(open(cfg.out_yaml / f"{cid}-{twin}.yaml",
                                    encoding="utf-8"))
            case = dict(d)
            case["input_image"] = "input.png"
            (out / f"case_{twin}.json").write_text(
                json.dumps(case, indent=1, ensure_ascii=False),
                encoding="utf-8")
            (out / f"card_{twin}.txt").write_text(card(d), encoding="utf-8")
        n += 1
    print(f"cases packed: {n} folders (x2 twins = {2*n} samples)")
    if missing:
        print(f"MISSING FRAMES ({len(missing)}):")
        for m in missing:
            print(" ", m)
    # 刷新随仓库分发的数据集快照(data/cases;评测与 baseline 的默认读取位置)
    data_dir = cfg.data_cases
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    shutil.copytree(cases_dir, data_dir)
    print(f"dataset snapshot refreshed: {data_dir}")
    return 0
