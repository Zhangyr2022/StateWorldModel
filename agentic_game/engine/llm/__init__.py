"""LLM 基础层:与常驻大模型服务对话的唯一出口。

与 engine/renderer 同构:游戏代码只依赖 LLMClient 抽象,不关心传输与
模型部署;worker 常驻、装载一次。主要供恋爱对话类游戏在结算阶段
生成角色回应(以及 LLM 驱动的玩家代理)。
"""

from engine.llm.client import LLMClient, LLMError, LLMFailedError, LLMTimeoutError

__all__ = ["LLMClient", "LLMError", "LLMFailedError", "LLMTimeoutError"]
