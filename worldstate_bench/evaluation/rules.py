"""规则结算重放(评测侧):用生成侧同一套闭式 DSL 做 if/else 确定性重放。

原 agentic_game 版本自带迷你解析器(数据"自足,无外部 DSL"的约束下);
本仓库就是数据生成方,直接复用 ``worldstate_bench.dsl``(与 build 期
V1–V10 校验同一实现),headless 自洽检查由此获得构造性保证:重放终态
必须与 gold_end_state 逐键一致。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from worldstate_bench.dsl import eval_pre, get_path, parse_pre, simulate, set_path

if TYPE_CHECKING:
    from evaluation.dataset import TwinSpec


@dataclass
class RuleError(ValueError):
    """规则缺失/不可解析(数据问题,headless 记入问题清单)。"""


def resolve_rule(twin: "TwinSpec", state: dict | None = None) -> tuple[dict, bool]:
    """确定性结算:precondition 命中 → effect,否则 else_effect。

    返回 (终态, 是否命中 if 分支);不修改传入状态。precondition 测试
    的是 top-level 隐藏变量(build 期写入 initial_state),effect 走
    dsl 单赋值语句(V1 保证)。
    """
    rule = twin.raw.get("rule") or {}
    working = copy.deepcopy(state if state is not None
                            else twin.raw.get("initial_state") or {})
    precondition = str(rule.get("precondition", "")).strip()
    if not precondition:
        raise RuleError(f"{twin.clip_id}: rule.precondition missing")
    try:
        executed = eval_pre(precondition, working)
    except Exception as exc:  # noqa: BLE001 - 解析失败即数据问题
        raise RuleError(f"{twin.clip_id}: bad precondition {precondition!r}: {exc}") from exc
    effect = str(rule.get("effect") if executed
                 else rule.get("else_effect") or "").strip()
    if not effect:
        raise RuleError(f"{twin.clip_id}: rule effect/else_effect missing "
                        f"(executed={executed})")
    try:
        return simulate(working, effect), executed
    except Exception as exc:  # noqa: BLE001
        raise RuleError(f"{twin.clip_id}: bad effect {effect!r}: {exc}") from exc


__all__ = ["RuleError", "get_path", "resolve_rule", "set_path"]
