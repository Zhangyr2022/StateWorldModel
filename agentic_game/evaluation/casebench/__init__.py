"""WorldState-Bench harness 评测 → 引擎渲染 → VLM 判定 → 指标。

数据自足:每个案例目录自带共享首帧(input.png)与两侧完整规格
(case_{A,B}.json,含预写 judge 问题),不需要 DSL/关键帧渲染/分镜生成。
评测代码只做三件事:按 H3 I2VA 信封原样包装 instruction 送渲染、
按帧位把预写问题喂给 VLM、核对答案与 expected 出指标。
"""
