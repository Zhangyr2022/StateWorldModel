"""WBench 评测适配:把一个 WBench case 变成一局"脚本化游戏"。

交互序列 = 确定的玩家行动脚本(无 LLM 导演、无数值结算),框架负责
状态维护(perspective 翻转)、VisualScript 编译(带防错约束)与渲染循环。
仅供 WBench 评测驱动(run_wbench.py),不进 game/registry(UI 无需枚举)。

外部依赖:WBench 是第三方基准(Fudan & Meituan LongCat,arXiv:2605.25874,
MIT),其官方仓库与数据集**不在本仓发布**。复现论文表 2/表 8 请从官方来源
(https://meituan-longcat.github.io/WBench/ ,
hf datasets meituan-longcat/WBench)获取,并把仓库放到
`$WORKSPACE_ROOT/WBench`(或用 --wbench-root / WBENCH_DATA 指向)。
"""
