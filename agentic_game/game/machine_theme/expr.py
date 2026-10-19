"""machine 主题的表达式求值器:让 JSON 声明算术与条件,而不是穷举分支。

白名单语法(用 ast 解析后逐节点校验,绝不执行任意代码):
    数字 / 变量名(读状态 vars) / 一元负号
    + - * / //            算术(变量表中不存在的变量按 0 处理)
    == != > >= < <=       比较(返回布尔)
    and or not            逻辑
    roll(base, variance)  base + 随机整数[0, variance](由传入的 Rng 驱动)
    ifelse(cond, a, b)    求值 cond 后取 a 或 b(非零/真 即真)
    A if cond else B      Python 三元式(与 ifelse 等价)

compile_expr(expr) 在加载 JSON 时做语法与白名单校验;
evaluate(expr, vars, rng) 在运行时求值。
"""

from __future__ import annotations

import ast

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)
_ALLOWED_CMPOPS = (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE)


class ExprError(ValueError):
    """表达式不合法;message 面向设计师。"""


def compile_expr(expr: str) -> ast.Expression:
    """语法与白名单校验,返回 AST(求值时复用)。"""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"表达式语法错误: {expr!r}({exc.msg})") from exc
    _check(tree.body)
    return tree


def _check(node) -> None:
    if isinstance(node, ast.Expression):
        _check(node.body)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool, str)):
            raise ExprError(f"只允许数字/字符串/布尔常量,不接受 {node.value!r}")
    elif isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise ExprError(f"变量名不能以下划线开头: {node.id}")
    elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd, ast.Not)):
        _check(node.operand)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        _check(node.left)
        _check(node.right)
    elif isinstance(node, ast.BoolOp):
        _check_all(node.values)
    elif isinstance(node, ast.Compare) and isinstance(node.ops[0], _ALLOWED_CMPOPS) \
            and len(node.ops) == 1:
        _check(node.left)
        _check(node.comparators[0])
    elif isinstance(node, ast.IfExp):
        _check(node.test)
        _check(node.body)
        _check(node.orelse)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ("roll", "ifelse"):
            raise ExprError("只允许调用 roll(base, variance) 与 ifelse(cond, a, b)")
        for arg in node.args:
            _check(arg)
    else:
        raise ExprError(f"表达式不允许 {type(node).__name__} 节点")


def _check_all(nodes) -> None:
    for node in nodes:
        _check(node)


def evaluate(expr: str | ast.Expression, vars: dict, rng) -> object:
    """求值。变量表中不存在的变量按 0 处理;roll 由传入的 rng 驱动。"""
    tree = compile_expr(expr) if isinstance(expr, str) else expr

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return vars.get(node.id, 0)
        if isinstance(node, ast.UnaryOp):
            value = ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.Not):
                return not value
            return value
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right if right else 0
            return left / right if right else 0
        if isinstance(node, ast.BoolOp):
            values = [bool(ev(v)) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left, right = ev(node.left), ev(node.comparators[0])
            op = node.ops[0]
            return {ast.Eq: left == right, ast.NotEq: left != right,
                    ast.Gt: left > right, ast.GtE: left >= right,
                    ast.Lt: left < right, ast.LtE: left <= right}[type(op)]
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [ev(a) for a in node.args]
            if name == "roll":
                base, variance = args
                return base + (rng.randint(0, variance) if variance else 0)
            if name == "ifelse":
                cond, if_true, if_false = args
                return if_true if cond else if_false
        raise ExprError(f"表达式不允许 {type(node).__name__} 节点")

    value = ev(tree.body)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value


__all__ = ["ExprError", "compile_expr", "evaluate"]
