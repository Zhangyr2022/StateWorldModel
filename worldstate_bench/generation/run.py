#!/usr/bin/env python3
"""WorldState-Bench 数据生成统一入口（Part 1 · benchmark 生成；v5：YAML → 渲染 → VLM
六维审查 → 改 prompt 迭代）。在仓库根目录运行：

  python3 generation/run.py build [b1 b2 ...]     # spec → out/yaml + out/work/render.csv
  python3 generation/run.py render [--only CASE] [--seeds 42,43] [--limit N]
                                            # 无 --only 时按 config 并行分片
  python3 generation/run.py review  [--only A,B] [--limit N] [--round R] [--gpus 0]
                                            # VLM 六维审查；失败 case 现场开
                                            # 出修订 prompt（fixes jsonl）
  python3 generation/run.py cycle  [--limit N] [--only A,B] [--dry-run]
                                            # 渲→审→改prompt同seed重渲→复审
                                            # 自动迭代到全过或轮次耗尽
  python3 generation/run.py package               # → out/cases/
  python3 generation/run.py selftest              # 用 specs 首模块首 case 冒烟自测

build/package 在当前进程跑（仅依赖 pyyaml）；render/review 自动用
config.envs 里对应 conda 环境的 python 以子进程跑（-m worldstate_bench.render /
worldstate_bench.review）。配置唯一来源：仓库根 config.yaml（specs 在
generation/specs，产物在 out/，均相对仓库根）。

不变式：out/work/prompt_fixes.json 是 VLM 修订的持久 journal，render.csv
只是物化视图——build 重写 csv 后由 _apply_prompt_fixes 恢复覆盖。
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # generation/
PROJECT = ROOT.parent                             # 仓库根(worldstate_bench 包、config.yaml 所在)
sys.path.insert(0, str(PROJECT))

from worldstate_bench.config import load_config  # noqa: E402


def _spawn(cfg, mod, extra, log):
    """用对应 conda 环境的 python 以子进程跑 worldstate_bench 模块，日志落 log。"""
    py = {"worldstate_bench.render": cfg.render_python,
          "worldstate_bench.review": cfg.review_python}.get(mod, cfg.review_python)
    if not Path(py).exists():
        # 解释器缺失时 execve 只会抛晦涩的 OSError,这里给出可操作的报错
        raise SystemExit(
            f"config.yaml 的解释器不存在: {py}\n"
            f"(换机部署后请修改 config.yaml 的 envs.render_python / envs.review_python)")
    cmd = [py, "-m", mod, *extra]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT)
    env.setdefault("HF_HUB_OFFLINE", "1")
    log.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log, "a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=PROJECT, stdout=lf, stderr=lf, env=env)


# ---------------------------------------------------------------- build
def cmd_build(cfg, names) -> int:
    from worldstate_bench.builder import build_case, load_specs, write_render_csv, write_yaml

    mods, missing = load_specs(cfg.specs_dir, names or None)
    if missing:
        print(f"specs not found: {missing}")
        return 1
    rows, all_bad, n_ok = [], [], 0
    for mod, specs in mods:
        for s in specs:
            docs, row, bad = build_case(s)
            all_bad += [f"[{mod}] {b}" for b in bad]
            if docs:
                n_ok += 1
                for d in docs:
                    write_yaml(cfg.out_yaml / f"{d['id']}.yaml", d)
                rows.append(row)
    # 注意：每次调用都会重写 render.csv —— 全量渲染前必须单次带全模块
    write_render_csv(cfg.out_work / "render.csv", rows)
    n_docs = len(list(cfg.out_yaml.glob("*.yaml")))
    print(f"cases ok={n_ok} failed={len(all_bad)} rows={len(rows)} "
          f"yaml={n_docs}")
    for b in all_bad:
        print(" FAIL", b)
    return 1 if all_bad else 0


# ---------------------------------------------------------------- render
def cmd_render(cfg, only=None, seeds=None, limit=None) -> int:
    csv_path = cfg.out_work / "render.csv"
    if not csv_path.exists():
        print(f"render.csv 不存在，先跑 build：{csv_path}")
        return 1
    n_fix = _apply_prompt_fixes(cfg)  # journal（prompt_fixes.json）→ csv
    if n_fix:
        print(f"render: applied {n_fix} VLM prompt fixes onto render.csv")
    if limit:  # demo 验证：截取前 N 行生成子 csv，再按分片渲染
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        demo = cfg.out_work / f"render_demo{limit}.csv"
        demo.write_text("\n".join(lines[: limit + 1]) + "\n", encoding="utf-8")
        csv_path, n_rows = demo, min(limit, len(lines) - 1)
        print(f"render demo: first {n_rows} cases -> {csv_path}")
    if only and "," in only:  # 批量补渲：子 csv + 全卡分片并行
        wanted = {c.strip() for c in only.split(",") if c.strip()}
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        rows = [l for l in lines[1:] if l.split(",", 1)[0] in wanted]
        sub = cfg.out_work / "render_reseed.csv"
        sub.write_text("\n".join([lines[0]] + rows) + "\n", encoding="utf-8")
        csv_path = sub
        print(f"render batch-reseed: {len(rows)} cases -> {csv_path}")
        only = None
    pairs = cfg.gpu_pairs
    procs = []
    if only:  # 单 case 补渲：单进程走第一对卡
        extra = ["--csv", str(csv_path), "--frames-root", str(cfg.out_frames),
                 "--gpus", pairs[0], "--only", only]
        if seeds:
            extra += ["--seeds", seeds]
        procs.append(_spawn(cfg, "worldstate_bench.render", extra,
                            cfg.out_work / "render.log"))
    else:     # 分片并行（全量/demo/批量重渲）：行交错切分 + 断点续渲幂等
        for i, pair in enumerate(pairs):
            extra = ["--csv", str(csv_path),
                     "--frames-root", str(cfg.out_frames), "--gpus", pair,
                     "--shard", str(i), "--nshards", str(len(pairs))]
            if seeds:
                extra += ["--seeds", seeds]
            procs.append(_spawn(cfg, "worldstate_bench.render", extra,
                                cfg.out_work / f"render_shard{i}.log"))
    rc = 0
    for p in procs:
        rc |= p.wait()
    print(f"render done rc={rc} ({len(procs)} procs)")
    return rc


# ---------------------------------------------------------------- review
def _max_review_round(cfg) -> int:
    judge = cfg.review.get("judge", "qwen38")
    best = -1
    for f in cfg.out_work.glob(f"review_{judge}_shard*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            best = max(best, int(r.get("round", 0)))
    return best


def cmd_review(cfg, only=None, limit=None, round_=None, gpus=None) -> int:
    """VLM 六维审查；standalone 默认开新一轮（max round + 1）。"""
    csv_path = cfg.out_work / "render.csv"
    if not csv_path.exists():
        print(f"render.csv 不存在，先跑 build：{csv_path}")
        return 1
    judge = cfg.review.get("judge", "qwen38")
    for f in cfg.out_work.glob(f"review_{judge}_failures_shard*.txt"):
        f.unlink()
    if round_ is None:
        round_ = _max_review_round(cfg) + 1
    if gpus is None:
        gpus = cfg.review_gpus
    elif isinstance(gpus, str):  # CLI 传 "0" / "0,1" → 卡号列表
        gpus = [int(g) for g in gpus.split(",") if g.strip()]
    procs = []
    for i, g in enumerate(gpus):
        extra = ["--yaml-dir", str(cfg.out_yaml),
                 "--render-csv", str(csv_path),
                 "--frames-root", str(cfg.out_frames), "--gpus", str(g),
                 "--shard", str(i), "--nshards", str(len(gpus)),
                 "--round", str(round_)]
        if only:
            extra += ["--only", only]
        if limit:
            extra += ["--limit", str(limit)]
        procs.append(_spawn(cfg, "worldstate_bench.review", extra,
                            cfg.out_work / f"review_{judge}_shard{i}.log"))
    rc = 0
    for p in procs:
        rc |= p.wait()
    if rc:
        print(f"review rc={rc} (有分片异常退出，failures_all 保持不变)")
        return rc
    # 从本轮 jsonl 记录聚合失败（空转分片不写文件，不能靠分片 txt 聚合）
    fails = sorted({r.get("case")
                    for f in cfg.out_work.glob(f"review_{judge}_shard*.jsonl")
                    for r in (json.loads(l) for l in
                              open(f, encoding="utf-8"))
                    if r.get("round") == round_ and not r.get("passed")})
    fail_all = cfg.out_work / f"review_{judge}_failures_all.txt"
    fail_all.write_text("\n".join(fails) + ("\n" if fails else ""))
    print(f"review done rc={rc} round={round_}; failures={len(fails)} "
          f"-> {fail_all}")
    return rc


# ------------------------------------------------------- prompt 修订 overlay
def _merge_fixes(cfg) -> dict:
    """review_*_fixes_shard*.jsonl → out/work/prompt_fixes.json。

    最新 round 胜出；validated=false（两试不过机械校验）→ manual:true,
    prompt:null——case 转 manual，不再自动迭代。
    """
    judge = cfg.review.get("judge", "qwen38")
    merged: dict[str, dict] = {}
    files = sorted(cfg.out_work.glob(f"review_{judge}_fixes_shard*.jsonl"))
    for f in files:
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = r.get("case")
            if not c:
                continue
            if c not in merged or r.get("round", 0) >= merged[c].get("round",
                                                                     -1):
                merged[c] = r
    out = {}
    for c, r in merged.items():
        if r.get("validated") and r.get("new_prompt"):
            out[c] = {"round": r["round"], "prompt": r["new_prompt"],
                      "reasons": r.get("reasons", []),
                      "manual": False}
        else:
            out[c] = {"round": r.get("round", 0), "prompt": None,
                      "reasons": r.get("reasons", []),
                      "manual": True}
    path = cfg.out_work / "prompt_fixes.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    return out


def _apply_prompt_fixes(cfg) -> int:
    """prompt_fixes.json 覆盖 render.csv 的 prompt 列（幂等，返回替换行数）。

    journal 是真源：build 重写 render.csv 后由此恢复 VLM 修订。
    """
    path = cfg.out_work / "prompt_fixes.json"
    if not path.exists():
        return 0
    fixes = json.loads(path.read_text(encoding="utf-8"))
    live = {c: f for c, f in fixes.items() if f.get("prompt")}
    csv_path = cfg.out_work / "render.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    n = 0
    for r in rows:
        f = live.get(r["case"])
        if f and r["prompt"] != f["prompt"]:
            r["prompt"] = f["prompt"]
            n += 1
    if n:
        from worldstate_bench.builder import write_render_csv
        write_render_csv(csv_path, rows)
    return n


def _last_seeds(cfg, cases: list[str]) -> dict[str, int | None]:
    """扫 out/frames/_render_log.jsonl，每 case 取最后一条记录的 seed。"""
    log = cfg.out_frames / "_render_log.jsonl"
    seed: dict[str, int | None] = {c: None for c in cases}
    if not log.exists():
        return seed
    for line in open(log, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = r.get("case")
        if c in seed and r.get("seed") is not None:
            seed[c] = int(r["seed"])
    return seed


def _latest_passed(cfg, cases) -> dict[str, bool]:
    """每 case 取最新 round 审查记录的 passed（无记录视为 False）。

    cycle 的失败判定用这个而不是 failures_all 文件——manual case 不进
    复审清单，按轮聚合的文件会把它漏掉，导致假 ALL_PASS。
    """
    judge = cfg.review.get("judge", "qwen38")
    best: dict[str, tuple[int, bool]] = {}
    for f in cfg.out_work.glob(f"review_{judge}_shard*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = r.get("case")
            if c in cases and (c not in best
                               or int(r.get("round", 0)) >= best[c][0]):
                best[c] = (int(r.get("round", 0)), bool(r.get("passed")))
    return {c: best.get(c, (0, False))[1] for c in cases}


def _latest_problems(cfg, cases: list[str]) -> dict[str, list[str]]:
    """每 case 取最新 round 审查记录里的 problems（写 manual 清单用）。"""
    judge = cfg.review.get("judge", "qwen38")
    best: dict[str, tuple[int, list]] = {}
    for f in cfg.out_work.glob(f"review_{judge}_shard*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = r.get("case")
            if c in cases and (c not in best
                               or r.get("round", 0) >= best[c][0]):
                best[c] = (r.get("round", 0), r.get("problems", []))
    return {c: p for c, (_, p) in best.items()}


# ---------------------------------------------------------------- cycle
def cmd_cycle(cfg, only=None, limit=None, dry_run=False) -> int:
    """渲→审→改prompt同seed重渲→复审 自动迭代。

    修复哲学：失败改 prompt 措辞、沿用失败帧原 seed（config.cycle.keep_seed）
    ——每次迭代是措辞的单变量实验；只有无历史 seed 的 case 才用默认阶梯。
    """
    status = cfg.out_work / "cycle_status.txt"
    maxr = int(cfg.cycle.get("max_rounds", 4))
    keep_seed = bool(cfg.cycle.get("keep_seed", True))

    def log(msg):
        status.parent.mkdir(parents=True, exist_ok=True)
        with open(status, "a", encoding="utf-8") as f:
            f.write(f"{msg} {time.strftime('%F %T')}\n")

    _apply_prompt_fixes(cfg)
    all_cases = [r["case"] for r in
                 csv.DictReader(open(cfg.out_work / "render.csv",
                                     encoding="utf-8"))]
    universe = set(all_cases[:limit]) if limit else set(all_cases)
    if only:
        universe &= {c.strip() for c in only.split(",") if c.strip()}

    log(f"=== cycle start universe={len(universe)} max_rounds={maxr} ===")
    if dry_run and not _latest_problems(cfg, sorted(universe)):
        n_missing = sum(1 for c in sorted(universe)
                        if not (cfg.out_frames / f"{c}.png").exists())
        print(f"[cycle dry-run] round 0 plan: render {n_missing} missing "
              f"frame(s); review {len(universe)} case(s) "
              f"({len(cfg.review_gpus)}-GPU shards)")
        print("[cycle dry-run] no review records yet — run cycle without "
              "--dry-run to execute round 0")
        return 0
    if not dry_run:  # dry-run 且已有记录：直接进循环打印计划，不执行 round 0
        cmd_render(cfg, only=only, limit=limit)
        cmd_review(cfg, round_=0, limit=limit, only=only)

    for rnd in range(1, maxr):
        passed_map = _latest_passed(cfg, sorted(universe))
        failed = [c for c in sorted(universe) if not passed_map[c]]
        if not failed:
            log(f"=== ALL_PASS at round={rnd - 1} ===")
            return 0
        fixes = _merge_fixes(cfg)
        with_fix = [c for c in failed if fixes.get(c, {}).get("prompt")]
        manual = [c for c in failed if c not in with_fix]  # 含 validated=false
        log(f"round={rnd} failed={len(failed)} refix={len(with_fix)} "
            f"manual={len(manual)} {failed}")
        if dry_run:
            seeds = _last_seeds(cfg, with_fix)
            print(f"[cycle dry-run] round={rnd} re-render {with_fix} "
                  f"seeds={seeds} manual={manual}")
            for c in with_fix:
                print(f"[cycle dry-run] {c}: {fixes[c]['reasons']}")
                print(f"  old: {fixes[c].get('prompt', '')[:120]}…")
            return 0
        _apply_prompt_fixes(cfg)  # render.csv 换上新 prompt
        # 按 seed 分组批量补渲（--seeds 每次调用只能统一一个值）
        seeds = _last_seeds(cfg, with_fix) if keep_seed else {}
        groups: dict[str | None, list[str]] = {}
        for c in with_fix:
            groups.setdefault(seeds.get(c), []).append(c)
        for seed, group in groups.items():
            for c in group:  # 续渲跳过要求先删旧帧
                f = cfg.out_frames / f"{c}.png"
                if f.exists():
                    f.unlink()
            cmd_render(cfg, only=",".join(group),
                       seeds=str(seed) if seed is not None else None)
        recheck = [c for c in failed if c not in manual]
        if recheck:
            cmd_review(cfg, round_=rnd, only=",".join(recheck))

    passed_map = _latest_passed(cfg, sorted(universe))
    failed = [c for c in sorted(universe) if not passed_map[c]]
    if not failed:
        log(f"=== ALL_PASS at round={maxr - 1} ===")
        return 0
    problems = _latest_problems(cfg, failed)
    manual_path = cfg.out_work / "cycle_manual.txt"
    manual_path.write_text("\n".join(
        f"{c}: {', '.join(problems.get(c, []))}" for c in sorted(failed))
        + "\n", encoding="utf-8")
    log(f"=== ROUNDS EXHAUSTED: {len(failed)} case(s) need manual review "
        f"-> {manual_path} ===")
    print(f"cycle: {len(failed)} case(s) -> {manual_path}")
    return 1


# ---------------------------------------------------------------- package
def cmd_package(cfg) -> int:
    from worldstate_bench.cases import main as pack
    return pack(cfg)


# ---------------------------------------------------------------- selftest
def cmd_selftest(cfg) -> int:
    """用 specs/ 第一个模块的第一个 case 冒烟自测构建链（不碰真实产物）。"""
    import shutil
    from worldstate_bench.config import Config
    from worldstate_bench.builder import load_specs

    mods = sorted(p.stem for p in cfg.specs_dir.glob("specs_*.py"))
    if not mods:
        print("selftest FAIL: specs/ 下没有 specs_*.py")
        return 1
    out, missing = load_specs(cfg.specs_dir, [mods[0]])
    if missing or not out:
        print(f"selftest FAIL: load {mods[0]} missing={missing}")
        return 1
    first, specs = out[0]
    one = specs[0]["cid"]
    tmp = cfg.root / "out" / "_selftest"
    shutil.rmtree(tmp, ignore_errors=True)
    tcfg = Config(cfg.raw, cfg.root, overrides={
        "out_yaml": str(tmp / "yaml"), "out_work": str(tmp / "work")})
    from worldstate_bench.builder import build_case, write_render_csv, write_yaml
    docs, row, bad = build_case(specs[0])
    for d in docs or []:
        write_yaml(tcfg.out_yaml / f"{d['id']}.yaml", d)
    write_render_csv(tcfg.out_work / "render.csv", [row] if row else [])
    n_yaml = len(list((tmp / "yaml").glob(f"{one}*.yaml")))
    n_csv = sum(1 for _ in open(tmp / "work" / "render.csv")) - 1
    schema_ok = bool(docs) and all(
        d["terminal_visual_states"]["not_occurs"]["instead_remains"]
        and d["judge"]["outcome_question"]["not_expected"]
        != d["judge"]["outcome_question"]["expected"]
        and d["terminal_visual_states"]["occurs"]["in_final_frame"] is True
        and d["terminal_visual_states"]["not_occurs"]["in_final_frame"] is False
        for d in docs)
    ok = not bad and n_yaml == 2 and n_csv == 1 and schema_ok
    print(f"selftest: module={first} case={one} failures={bad} "
          f"yaml={n_yaml}/2 csv_rows={n_csv}/1 "
          f"terminal_visual_states={'ok' if schema_ok else 'MISSING'}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("selftest PASS (临时目录已清理)" if ok else "selftest FAIL")
    return 0 if ok else 1


def _after(names, flag):
    return names[names.index(flag) + 1] if flag in names else None


def main() -> int:
    cfg = load_config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if cmd == "build":
        return cmd_build(cfg, rest)
    if cmd == "render":
        lim = _after(rest, "--limit")
        return cmd_render(cfg, only=_after(rest, "--only"),
                          seeds=_after(rest, "--seeds"),
                          limit=int(lim) if lim else None)
    if cmd == "review":
        lim = _after(rest, "--limit")
        rnd = _after(rest, "--round")
        return cmd_review(cfg, only=_after(rest, "--only"),
                          limit=int(lim) if lim else None,
                          round_=int(rnd) if rnd else None,
                          gpus=_after(rest, "--gpus"))
    if cmd == "cycle":
        lim = _after(rest, "--limit")
        return cmd_cycle(cfg, only=_after(rest, "--only"),
                         limit=int(lim) if lim else None,
                         dry_run="--dry-run" in rest)
    if cmd == "package":
        return cmd_package(cfg)
    if cmd == "selftest":
        return cmd_selftest(cfg)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
