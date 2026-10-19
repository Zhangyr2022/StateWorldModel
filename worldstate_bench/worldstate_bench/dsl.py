"""benchmark_all DSL — 严格封闭文法的规则转移解析/求值/模拟。

WorldState-Bench 专用窄文法（刻意比通用规则 DSL 窄：无 and/or/not、无括号、无算术嵌套）：

    precondition := PATH OP LITERAL            （OP ∈ == != >= > <= <）
                 | has(PATH, IDENT)
    effect       := STMT (';' STMT)*
    STMT         := PATH '=' LITERAL
                 | PATH '=' PATH ('+'|'-') INT
                 | take(PATH, IDENT) | give(PATH, IDENT)

禁止：and / or / not / 括号 / 算术嵌套 / 其他内建。越界写法直接抛
StrictGrammarError —— 数据集构造期就拦住，不会留到评测端。

LITERAL := true | false | INT | 'word'（单引号字符串，用于字符串状态值）
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


class StrictGrammarError(ValueError):
    pass


# ---------------------------------------------------------------- tokenizer
_TOKEN_RE = re.compile(
    r"""(?P<WS>\s+)
      | (?P<NUM>\d+)
      | (?P<STRING>'[^']*')
      | (?P<OP>==|!=|>=|<=|>|<|\+|-|\(|\)|,|=|;)
      | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
    """, re.VERBOSE)


@dataclass
class Tok:
    kind: str  # NUM STRING OP IDENT KW
    text: str
    pos: int


def tokenize(src: str) -> list[Tok]:
    out, i = [], 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise StrictGrammarError(f"unexpected character at {i}: `{src}`")
        i = m.end()
        if m.lastgroup == "WS":
            continue
        out.append(Tok(m.lastgroup, m.group(), m.start()))
    return out


# ---------------------------------------------------------------- AST
@dataclass
class Cmp:            # path OP literal  |  has(path, item)
    path: str
    op: str           # == != >= > <= <  or 'has'
    value: Any        # literal 或 item 名


@dataclass
class SetLit:         # path = literal
    path: str
    value: Any


@dataclass
class SetArith:       # path = path ± int
    path: str
    src: str
    op: str
    num: int


@dataclass
class GiveTake:
    func: str         # give | take
    actor: str
    item: str


Effect = SetLit | SetArith | GiveTake


# ---------------------------------------------------------------- parser
def parse_pre(src: str) -> Cmp:
    toks = tokenize(src)
    if not toks:
        raise StrictGrammarError(f"empty precondition: `{src}`")
    # has(actor_path, item_name) —— 6 token: has ( path , item )
    if len(toks) == 6 and toks[0].text == "has" and toks[1].text == "(" \
            and toks[2].kind == "IDENT" and toks[3].text == "," \
            and toks[4].kind == "IDENT" and "." not in toks[4].text \
            and toks[5].text == ")":
        return Cmp(toks[2].text, "has", toks[4].text)
    # path OP literal
    if len(toks) == 3 and toks[0].kind == "IDENT" and toks[1].kind == "OP" \
            and toks[1].text in ("==", "!=", ">=", ">", "<=", "<"):
        return Cmp(toks[0].text, toks[1].text, _literal(toks[2], src))
    raise StrictGrammarError(
        f"precondition must be `path OP value` or `has(actor, item)`: `{src}`")


def _literal(t: Tok, src: str) -> Any:
    if t.kind == "NUM":
        return int(t.text)
    if t.kind == "STRING":
        return t.text[1:-1]
    if t.kind == "IDENT" and t.text in ("true", "false"):
        return t.text == "true"
    raise StrictGrammarError(f"bad literal `{t.text}` in `{src}` "
                             f"(use true/false/int/'word')")


def parse_effect(src: str) -> list[Effect]:
    out: list[Effect] = []
    for raw in src.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        out.append(_parse_stmt(stmt))
    if not out:
        raise StrictGrammarError(f"empty effect: `{src}`")
    return out


def _parse_stmt(src: str) -> Effect:
    toks = tokenize(src)
    # give/take
    if len(toks) == 6 and toks[0].text in ("give", "take") and toks[1].text == "(" \
            and toks[2].kind == "IDENT" and toks[3].text == "," \
            and toks[4].kind == "IDENT" and "." not in toks[4].text \
            and toks[5].text == ")":
        return GiveTake(toks[0].text, toks[2].text, toks[4].text)
    # path = ...
    if len(toks) >= 3 and toks[0].kind == "IDENT" and toks[1].text == "=":
        path = toks[0].text
        rest = toks[2:]
        if len(rest) == 1:
            return SetLit(path, _literal(rest[0], src))
        if len(rest) == 3 and rest[0].kind == "IDENT" \
                and rest[1].text in ("+", "-") and rest[2].kind == "NUM":
            return SetArith(path, rest[0].text, rest[1].text, int(rest[2].text))
    raise StrictGrammarError(
        f"effect stmt must be `path = value`, `path = path ± int`, "
        f"`give(a,i)` or `take(a,i)`: `{src}`")


# ---------------------------------------------------------------- state helpers
def get_path(state: dict, path: str) -> Any:
    cur: Any = state
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def set_path(state: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = state
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


# ---------------------------------------------------------------- eval / simulate
def eval_pre(pre: str, state: dict) -> bool:
    c = parse_pre(pre)
    if c.op == "has":
        bag = get_path(state, c.path)
        inv = bag.get("inventory", []) if isinstance(bag, dict) else []
        return c.value in inv
    v = get_path(state, c.path)
    op, lit = c.op, c.value
    if op == "==":
        return v == lit
    if op == "!=":
        return v != lit
    try:
        vf = float(v)
    except (TypeError, ValueError):
        return False
    return {"<": vf < lit, "<=": vf <= lit, ">": vf > lit, ">=": vf >= lit}[op]


def simulate(state: dict, effect: str) -> dict:
    """在深拷贝上执行 effect，返回 post-state（不改变输入）。"""
    st = copy.deepcopy(state)
    for e in parse_effect(effect):
        if isinstance(e, SetLit):
            set_path(st, e.path, e.value)
        elif isinstance(e, SetArith):
            cur = get_path(st, e.src)
            if not isinstance(cur, (int, float)) or isinstance(cur, bool):
                raise StrictGrammarError(
                    f"arith source {e.src}={cur!r} is not numeric")
            set_path(st, e.path, cur + e.num if e.op == "+" else cur - e.num)
        else:  # GiveTake —— give/take 操作 <actor>.inventory
            inv_path = f"{e.actor}.inventory"
            inv = get_path(st, inv_path)
            if inv is None:
                set_path(st, inv_path, [])
                inv = []
            if e.func == "give":
                if e.item not in inv:
                    inv = inv + [e.item]
            else:
                inv = [i for i in inv if i != e.item]
            set_path(st, inv_path, inv)
    return st


def check_grammar(pre: str, effect: str) -> None:
    """只做文法检查（builder 的静态 lint）。"""
    parse_pre(pre)
    parse_effect(effect)
