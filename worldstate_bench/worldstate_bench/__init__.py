"""worldstate_bench —— WorldState-Bench 数据生成核心包。

模块：
  config   配置加载（config.yaml 唯一来源）
  dsl      闭式文法 DSL（解析/求值/模拟）
  builder  spec → twin YAML + render CSV + V1-V10 硬校验（含 validate_prompt）
  render   FLUX.2 首帧渲染 + 结构门控（personN）+ 断点续渲
  review   VLM 六维审查 + 渲染 prompt 修订（替代旧 audit）
  cases    打包 cases/（幂等）
入口见 ../run.py（build/render/review/cycle/package/selftest）。
"""
__version__ = "3.0.0"
