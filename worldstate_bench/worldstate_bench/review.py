"""审查核心（v5 六维审查 + prompt 修订，替代旧 audit 的多选题双序审计）。

每 case 一次结构化 VLM 调用（JSON 输出），按六个维度判首帧：
  1 sanity        生成是否崩坏（伪影/畸形/乱码的基本检查）
  2 entities      实验所需角色（恰 N 人，gate=personN）与关键对象在场
  3 initial_predicates  初始视觉 predicate 逐条呈现在首帧
  4 action_ready  首帧能稳定产生并观察后续 shared action（构图/遮挡/空间）
  5 premature_terminal 两个 terminal visual state 都不提前出现
  6 terminal_room 两个 terminal visual state 之后有足够展示空间

通过判定全部在 Python 里机械计算（不信模型自报的布尔）。失败（且非
parse_error/missing_frame）时同进程做第二次 VLM 调用修订渲染 prompt，
修订结果过 worldstate_bench.builder.validate_prompt 机械校验（初态短语逐字保留、
hidden 值/终态短语不入图、style 保留），违规重试一次，仍违规则保留
旧 prompt 转 manual。

方法论：失败改 prompt 措辞、同 seed 重渲——由 run.py cycle 编排，
本模块只负责"审 + 开方"。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import yaml

from worldstate_bench.config import Config, load_config

_NWORDS = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
           "five": 5}


# ---------------------------------------------------------------- 解析工具
def parse_json_block(text: str) -> dict | None:
    """宽松 JSON 提取：剥 </think> 段与 ``` 围栏，取首 { 到尾 }。"""
    tail = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    tail = re.sub(r"```[a-zA-Z]*\s*|\s*```", "", tail).strip()
    start, end = tail.find("{"), tail.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(tail[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "ok"):
            return True
        if s in ("false", "no", "broken"):
            return False
    return None


def _int(v) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s.isdigit():
            return int(s)
        if s in _NWORDS:
            return _NWORDS[s]
    return None


def person_target(gate: str) -> tuple[int, bool]:
    """gate=personN(_soft) → (期望人数, 是否软门控)。"""
    m = re.match(r"person(\d+)", str(gate))
    n = int(m.group(1)) if m else 1
    return n, str(gate).endswith("_soft")


def isc_phrase(doc: dict, key: str) -> str:
    """initial_predicates 的编号键（"1"/"2"/…）→ 对应初态短语。"""
    sv = doc["initial_scene"]["shared_visual_state"]
    i = int(key) - 1 if key.isdigit() else -1
    return sv[i] if 0 <= i < len(sv) else key


def load_gate_map(render_csv: Path) -> dict[str, str]:
    return {r["case"]: r["gate"] for r in
            csv.DictReader(open(render_csv, encoding="utf-8"))}


# ---------------------------------------------------------------- prompt 构造
def review_question(doc: dict, n_persons: int, prompt: str = "") -> str:
    """六维审查题面（调用 #1）。全部事实取自 YAML，单一来源。

    关键对象分两类：名字出现在当前渲染 prompt 里的 = 首帧应在场；
    其余（如 B12 的 board_bridge/corner_tower）是后续帧才出现的结局
    载体，缺席是正确状态，不得计为 missing。
    """
    isc = doc["initial_scene"]
    tvs = doc["terminal_visual_states"]
    preds = "\n".join(f"  {i + 1}. {p}" for i, p
                      in enumerate(isc["shared_visual_state"]))
    pl = prompt.lower()
    req = [o for o in isc["other_key_objects"]
           if o.replace("_", " ").lower() in pl]
    later = [o for o in isc["other_key_objects"] if o not in req]
    objs = ", ".join(req) or "none besides the player"
    later_line = (f"\n- Later-frame objects (appear only AFTER the action; their "
                  f"absence now is CORRECT — never list them as missing): "
                  f"{', '.join(later)}." if later else "")
    return f"""You are a strict QA reviewer for generated game keyframe images. This image is the shared FIRST frame of a two-outcome video benchmark case. Judge it against the case facts. Reply with ONLY one JSON object — no markdown fences, no text before or after.

CASE FACTS
- Required people: exactly {n_persons} human character(s). Player: {isc['player_pose_and_item']}
  Statues, mannequins, scarecrows, portraits or reflections do not count as people.
- Required key objects: {objs}.{later_line}
- Initial visual predicates — each must be clearly TRUE in this image:
{preds}
- Later action that must be performable and clearly observable in this framing:
  "{doc['shared_action']}"
- Terminal outcome A (belongs to a LATER frame; must NOT be visible yet):
  "{tvs['occurs']['phrase']}"
- Terminal outcome B (the counterpart outcome; must NOT be visible yet):
  "{tvs['not_occurs']['phrase']}"
- Room check: after the action, the carrier of outcome A and the carrier of outcome B must EACH be able to show its change fully inside the frame with clear space around it (a door can swing open, flames can rise, water can fill, arms can extend, etc.).

ANSWER SCHEMA — all fields required, booleans as true/false:
{{"sanity": {{"ok": true, "note": "..."}},
 "entities": {{"persons_seen": 0, "missing_objects": [], "note": "..."}},
 "initial_predicates": {{"1": true, "2": true}},
 "action_ready": {{"ok": true, "note": "..."}},
 "premature_terminal": {{"A_visible": false, "B_visible": false, "note": "..."}},
 "terminal_room": {{"A": true, "B": true, "note": "..."}}}}
sanity.ok=false only for corruption: gross artifacts, extra limbs or faces, incoherent geometry, or not a scene at all. action_ready covers framing: the player and the action target are visible and unoccluded, with room to perform and observe the action. premature_terminal is true only if an outcome state is ALREADY present in this first frame. Use every note field to cite what you actually see."""


def revise_question(doc: dict, problems: list[str], notes: dict,
                    current_prompt: str, banned_words: list[str]) -> str:
    """prompt 修订题面（调用 #2）。带失败维度与证据 note，输出新 prompt。"""
    isc = doc["initial_scene"]
    tvs = doc["terminal_visual_states"]
    failed = "\n".join(f"- {p}: {notes.get(p, '')}".rstrip() for p in problems)
    preds = "\n".join(f"  - {p}" for p in isc["shared_visual_state"])
    banned = ", ".join(banned_words) or "none"
    return f"""You are fixing a text-to-image prompt for FLUX.2 (it has NO negative prompt: everything must be stated positively). The prompt below produced the attached image, which failed QA. Rewrite the prompt so the same problems cannot recur.

FAILED CHECKS
{failed}

CURRENT PROMPT
"{current_prompt}"

HARD CONSTRAINTS
1. Keep every initial-state phrase below VERBATIM (copy character-for-character):
{preds}
2. Never mention or describe either terminal outcome — forbidden, including paraphrase:
  - {tvs['occurs']['phrase']}
  - {tvs['not_occurs']['phrase']}
  Describe off states positively: cold, dark, matte, still, slack, plain.
3. Do not use these words: {banned}.
4. Keep the style sentence verbatim: "{isc['style']}"
5. Keep "third-person" and "16:9 landscape"; keep all key objects large and fully visible with no occlusion.
6. Fix the failed checks by rewording scene/composition only: explicit object placement, spatial separation between the required objects, camera distance, headroom and clearance around each outcome carrier, stronger positive off-state wording, or making the second required character explicit.
7. One single line, English, under 1200 characters.

Reply with ONLY this JSON: {{"prompt": "<revised prompt>"}}"""


# ---------------------------------------------------------------- 修订校验
def validate_revised_prompt(prompt: str, doc: dict) -> list[str]:
    """修订 prompt 的机械验收：V9 子集 + style/16:9/单行长度。"""
    from worldstate_bench.builder import validate_prompt

    isc = doc["initial_scene"]
    tvs = doc["terminal_visual_states"]
    hs = doc["hidden_state"]
    errs = validate_prompt(
        prompt,
        init_phrases=isc["shared_visual_state"],
        hidden_tokens=[hs["A"], hs["B"]],
        prompt_ok=doc.get("prompt_ok", []),
        terminal_phrases=[tvs["occurs"]["phrase"],
                          tvs["not_occurs"]["phrase"]])
    if isc["style"] not in prompt:
        errs.append("style sentence missing verbatim")
    if "16:9" not in prompt:
        errs.append("16:9 landscape missing")
    if "\n" in prompt:
        errs.append("must be one single line")
    if len(prompt) > 1200:
        errs.append(f"over 1200 characters ({len(prompt)})")
    if prompt.strip() == "":
        errs.append("empty prompt")
    return errs


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None, cfg: Config | None = None) -> int:
    ap = argparse.ArgumentParser(prog="worldstate_bench.review")
    ap.add_argument("--model", default=None, help="判定器路径（默认取 config）")
    ap.add_argument("--yaml-dir", type=Path, default=None)
    ap.add_argument("--render-csv", type=Path, default=None)
    ap.add_argument("--frames-root", type=Path, default=None)
    ap.add_argument("--only", default=None,
                    help="只审这些 case，逗号分隔精确 cid")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--round", type=int, default=0,
                    help="迭代轮次（cycle 显式传；断点续跑按轮去重）")
    ap.add_argument("--redo", action="store_true",
                    help="无视本轮已有记录强制重审")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    default=None)
    ap.add_argument("--thinking", dest="thinking", action="store_true",
                    default=None)
    ap.add_argument("--judge", default=None,
                    help="判定器标签（仅用于输出文件命名）")
    args = ap.parse_args(argv)

    cfg = cfg or load_config()  # 子进程模式：读项目根 config.yaml（默认值仅作缺键回退）
    rc = cfg.review
    model = args.model or rc.get("model")
    if not model:
        raise SystemExit("config.yaml 缺少 review.model,且未传 --model")
    judge = args.judge or rc.get("judge", "qwen38")
    thinking = args.thinking if args.thinking is not None \
        else bool(rc.get("thinking", False))
    yaml_dir = args.yaml_dir or cfg.out_yaml
    render_csv = args.render_csv or (cfg.out_work / "render.csv")
    frames_root = args.frames_root or cfg.out_frames
    out = cfg.out_work / f"review_{judge}_shard{args.shard}.jsonl"
    fixes_out = cfg.out_work / f"review_{judge}_fixes_shard{args.shard}.jsonl"
    think_tokens = int(rc.get("thinking_max_new_tokens", 1024))
    nothink_tokens = int(rc.get("nothinking_max_new_tokens", 768))
    person_check = bool(rc.get("person_check", True))
    json_retry = int(rc.get("json_retry", 1))

    import os

    # ---- 本轮待审集合先算：全已审则早退（不载模型、不写失败文件）--------
    done: set[str] = set()
    if not args.redo:
        for f in cfg.out_work.glob(f"review_{judge}_shard*.jsonl"):
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("round") == args.round:
                    done.add(r.get("case"))

    gates = load_gate_map(render_csv)
    prompts = {r["case"]: r["prompt"] for r in
               csv.DictReader(open(render_csv, encoding="utf-8"))}
    docs = sorted(yaml_dir.glob("*-A.yaml"))
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        docs = [d for d in docs if d.name[:-len("-A.yaml")] in wanted]
    if args.limit:
        docs = docs[: args.limit]
    docs = docs[args.shard::args.nshards]
    todo_docs = [d for d in docs
                 if args.redo or d.name[:-len("-A.yaml")] not in done]
    if not todo_docs:
        print(f"[review:{judge}] round={args.round} nothing to do "
              f"(all {len(docs)} case(s) of this shard already reviewed); "
              f"failures file left untouched", flush=True)
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(model)
    model_obj = AutoModelForImageTextToText.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model_obj.eval()
    print(f"[review:{judge}] model ready on gpus={args.gpus}", flush=True)

    def ask(img, text: str) -> str:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": text}]}]
        kw = {} if thinking else {"enable_thinking": False}
        try:
            t = proc.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, **kw)
        except TypeError:  # 模板不支持该参数的判定器
            t = proc.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        inputs = proc(text=t, images=[img], return_tensors="pt").to("cuda:0")
        max_new = think_tokens if thinking else nothink_tokens
        with torch.no_grad():
            o = model_obj.generate(**inputs, max_new_tokens=max_new,
                                   do_sample=False)
        return proc.decode(o[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=False)

    def ask_json(img, text: str) -> tuple[dict | None, str]:
        """带 JSON 重试的调用：失败追加提醒再问，最多 json_retry 次。"""
        raw = ""
        for i in range(json_retry + 1):
            raw = ask(img, text if i == 0 else
                      text + "\n\nYour previous reply was not valid JSON. "
                             "Reply again with ONLY the JSON object.")
            obj = parse_json_block(raw)
            if obj is not None:
                return obj, raw
        return None, raw

    outf = open(out, "a", encoding="utf-8")
    fixf = open(fixes_out, "a", encoding="utf-8")
    n_pass = n_fail = n_skip = 0
    fails: list[str] = []
    for n, dpath in enumerate(todo_docs, 1):
        doc = yaml.safe_load(open(dpath, encoding="utf-8"))
        cid = doc["id"].rsplit("-", 1)[0]
        f = frames_root / f"{cid}.png"
        if not f.exists():
            n_fail += 1
            fails.append(cid)
            outf.write(json.dumps(dict(
                case=cid, round=args.round, gate=gates.get(cid, ""),
                passed=False, problems=["missing_frame"],
                checks={}, notes={})) + "\n")
            outf.flush()
            continue
        img = Image.open(f).convert("RGB").resize((960, 540))
        gate = gates.get(cid, "person1")
        n_persons, soft = person_target(gate)
        cur_prompt = prompts.get(cid, "")

        # ---- 调用 #1：六维审查 -------------------------------------------
        q = review_question(doc, n_persons, cur_prompt)
        obj, raw = ask_json(img, q)
        if obj is None:
            n_fail += 1
            fails.append(cid)
            outf.write(json.dumps(dict(
                case=cid, round=args.round, gate=gate, passed=False,
                problems=["parse_error"], checks={}, notes={},
                error=raw[-300:])) + "\n")
            outf.flush()
            print(f"[review:{judge}] {n}/{len(todo_docs)} {cid} PARSE_ERROR",
                  flush=True)
            continue  # 解析失败不触发修订（case 进下一轮/manual）

        # ---- 通过判定全部在 Python 里机械计算 ----------------------------
        san = obj.get("sanity") or {}
        ent = obj.get("entities") or {}
        pre = obj.get("premature_terminal") or {}
        room = obj.get("terminal_room") or {}
        preds_raw = obj.get("initial_predicates") or {}
        preds = {k: _bool(v) for k, v in preds_raw.items()} if \
            isinstance(preds_raw, dict) else {}
        persons_seen = _int(ent.get("persons_seen"))
        missing = ent.get("missing_objects") or []
        if isinstance(missing, str):
            missing = [missing]

        c_sanity = _bool(san.get("ok")) is True
        c_objects = not missing
        c_persons = (persons_seen == n_persons) if (
            person_check and not soft and persons_seen is not None) else True
        c_entities = c_objects and c_persons
        c_preds = bool(preds) and all(v is True for v in preds.values())
        c_action = _bool((obj.get("action_ready") or {}).get("ok")) is True
        c_pre_a = _bool(pre.get("A_visible")) is False
        c_pre_b = _bool(pre.get("B_visible")) is False
        c_room_a = _bool(room.get("A")) is True
        c_room_b = _bool(room.get("B")) is True

        checks = {"sanity": c_sanity, "entities": c_entities,
                  "objects": c_objects, "persons": c_persons,
                  "persons_seen": persons_seen,
                  "initial_predicates": preds, "action_ready": c_action,
                  "premature_terminal": {"A": c_pre_a, "B": c_pre_b},
                  "terminal_room": {"A": c_room_a, "B": c_room_b}}
        problems = []
        if not c_sanity:
            problems.append("sanity")
        if not c_objects:
            problems.append("entities.objects")
        if not c_persons:
            problems.append("entities.persons")
        if not c_preds:
            problems.append("initial_predicates")
        if not c_action:
            problems.append("action_ready")
        if not c_pre_a:
            problems.append("premature_terminal.A")
        if not c_pre_b:
            problems.append("premature_terminal.B")
        if not c_room_a:
            problems.append("terminal_room.A")
        if not c_room_b:
            problems.append("terminal_room.B")
        if soft and person_check and persons_seen is not None \
                and persons_seen != n_persons:
            problems.append("warn.persons_soft")  # 软门控：仅警告不计失败
        notes = {"sanity": san.get("note", ""), "entities": ent.get("note", ""),
                 "action_ready": (obj.get("action_ready") or {}).get("note", ""),
                 "premature_terminal": pre.get("note", ""),
                 "terminal_room": room.get("note", ""),
                 "missing_objects": missing}
        # 修订用证据：problem 名 → 可操作的失败描述
        revise_notes: dict[str, str] = {}
        for p in problems:
            if p == "sanity":
                revise_notes[p] = f"image is corrupted; {notes['sanity']}"
            elif p == "entities.objects":
                revise_notes[p] = f"missing objects {missing}; " \
                                  f"{notes['entities']}"
            elif p == "entities.persons":
                revise_notes[p] = (f"persons seen = {persons_seen}, "
                                   f"want exactly {n_persons}; "
                                   f"{notes['entities']}")
            elif p == "initial_predicates":
                bad = [f"{k}: {isc_phrase(doc, k)}" for k, v in preds.items()
                       if v is not True]
                revise_notes[p] = "; ".join(bad) or "some predicate not visible"
            elif p == "action_ready":
                revise_notes[p] = notes["action_ready"]
            elif p.startswith("premature_terminal."):
                revise_notes[p] = (f"terminal state {p.rsplit('.', 1)[1]} "
                                   f"already visible; "
                                   f"{notes['premature_terminal']}")
            elif p.startswith("terminal_room."):
                revise_notes[p] = (f"no room to show terminal state "
                                   f"{p.rsplit('.', 1)[1]} later; "
                                   f"{notes['terminal_room']}")
            elif p == "warn.persons_soft":
                revise_notes[p] = (f"(soft gate warning) persons seen = "
                                   f"{persons_seen}, want {n_persons}")
        passed = not [p for p in problems if not p.startswith("warn.")]
        n_pass += passed
        n_fail += (not passed)
        if not passed:
            fails.append(cid)

        # ---- 调用 #2：修订 prompt（同进程，模型已加载）--------------------
        fix_rec = None
        if not passed:
            ok_words = set(doc.get("prompt_ok", []))
            banned = [t for t in (doc["hidden_state"]["A"],
                                  doc["hidden_state"]["B"])
                      if t not in ok_words]
            rq = revise_question(doc, problems, revise_notes, cur_prompt,
                                 banned)
            robj, _ = ask_json(img, rq)
            new_prompt, validated, errs = None, False, []
            if robj is not None and isinstance(robj.get("prompt"), str):
                new_prompt = robj["prompt"].strip()
                errs = validate_revised_prompt(new_prompt, doc)
                if not errs:
                    validated = True
                else:  # 带错误重试一次
                    rq2 = rq + ("\n\nYour revised prompt violated: "
                                + "; ".join(errs)
                                + " Rewrite obeying ALL constraints.")
                    robj2, _ = ask_json(img, rq2)
                    if robj2 is not None and isinstance(robj2.get("prompt"),
                                                        str):
                        new_prompt = robj2["prompt"].strip()
                        errs = validate_revised_prompt(new_prompt, doc)
                        validated = not errs
            if new_prompt and not validated:
                new_prompt = None  # 两试不过：保留旧 prompt，转 manual
            fix_rec = dict(case=cid, round=args.round, reasons=problems,
                           notes=notes, old_prompt=cur_prompt,
                           new_prompt=new_prompt, validated=validated,
                           errors=errs)
            fixf.write(json.dumps(fix_rec) + "\n")
            fixf.flush()

        outf.write(json.dumps(dict(
            case=cid, round=args.round, gate=gate, passed=passed,
            problems=problems, checks=checks, notes=notes)) + "\n")
        outf.flush()
        if n % 10 == 0 or not passed:
            print(f"[review:{judge}] {n}/{len(todo_docs)} {cid} "
                  f"{'PASS' if passed else 'FAIL:' + ','.join(problems)}",
                  flush=True)

    outf.close()
    fixf.close()
    n_skip = len(docs) - len(todo_docs)
    fail_path = cfg.out_work / f"review_{judge}_failures_shard{args.shard}.txt"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.write_text("\n".join(sorted(fails)) + ("\n" if fails else ""))
    print(f"[review:{judge}] done. pass={n_pass} fail={n_fail} "
          f"skipped(done)={n_skip} failures->{fail_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
