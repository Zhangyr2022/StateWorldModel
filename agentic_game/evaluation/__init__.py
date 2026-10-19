"""评测包:WorldState-Bench harness 评测 → 引擎渲染 → VLM 判定 → 指标。

本包产出论文 Table 1 的 STATEFULHARNESS 行:对 WorldState-Bench 数据集
(默认 ../worldstate_bench/data/cases)按 harness 条件渲染与判定。
每个案例目录自带共享首帧(input.png)与两侧完整规格
(case_{A,B}.json,instruction/gold/judge 问题都由数据预写)。
产物写至 evaluation/output/casebench/runs/<run_id>/;
engine 与 game 代码零改动。入口:python3 -m evaluation.run_cases --help。
"""

from evaluation.casebench.dataset import Case, TwinSpec, load_cases

__all__ = ["Case", "TwinSpec", "load_cases"]
