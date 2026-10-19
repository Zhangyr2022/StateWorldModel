"""Visual Pipeline:Turn Result → Visual Script → Prompt(Phase 3)。

Semantic Bridge(论文 §7):游戏层负责把 TurnResult 翻译成 VisualScript
(描述"应该表现什么"),基础层负责 VisualScript → Prompt 的组织方式
(描述"如何告诉模型",按 engine/visual/NEWSKILL.md 规范编译)。
"""

from engine.visual.compiler import PromptCompiler
from engine.visual.script import VisualScript
from engine.visual.standard import (
    Dialogue,
    count_sentences,
    render_stability,
    validate_standard,
)
from engine.visual.state import VisualState

__all__ = [
    "PromptCompiler",
    "VisualScript",
    "VisualState",
    "Dialogue",
    "count_sentences",
    "render_stability",
    "validate_standard",
]
