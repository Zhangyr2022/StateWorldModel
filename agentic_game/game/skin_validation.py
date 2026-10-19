"""JSON 皮肤共用校验错误类型(machine/romance 加载器共用,便于上层统一捕获)。"""


class SkinValidationError(ValueError):
    """皮肤 JSON 不合法;message 面向设计师(指出字段与原因)。"""
