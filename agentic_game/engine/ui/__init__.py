"""engine.ui:交互式(真人玩家)界面层。

session.py 是与框架无关的逐步驱动器(复用 GameLoop 的回合机制);
app.py 把它包成 Gradio 界面。游戏只提供 GameParts(engine/parts.py),
界面零改动覆盖所有游戏。
"""

from engine.ui.session import GameFinishedError, InteractiveGame

__all__ = ["GameFinishedError", "InteractiveGame"]
