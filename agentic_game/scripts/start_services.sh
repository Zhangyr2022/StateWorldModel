#!/usr/bin/env bash
# 1+4 默认部署:1 卡 Qwen(LLM 对话)+ 4 卡 H3(渲染),其余卡留给其他任务。
#
# 用法:
#     scripts/start_services.sh start     # 两个常驻服务(Qwen ~2 分钟,H3 ~6 分钟)
#     scripts/start_services.sh status
#     scripts/start_services.sh stop
#     scripts/start_services.sh restart
#
# 卡位与模型路径可用环境变量覆盖:
#     QWEN_GPU=0 H3_GPUS=1,2,3,4 QWEN_MODEL=/path/to/Qwen scripts/start_services.sh start
#
# H3 4 卡显存贴顶(~78-80G/卡),脚本固定带 PYTORCH_ALLOC_CONF=expandable_segments:True
# 缓解偶发 OOM 拖死服务——见 docs/duel/duel_pipeline.md §8。
# 客户端(run_ui / run_demo)无需 --socket:config.toml 已固定 socket_path。
set -euo pipefail
cd "$(dirname "$0")/.."

export CONDA_ENVS_ROOT="${CONDA_ENVS_ROOT:-$HOME/miniconda3/envs}"
QWEN_GPU="${QWEN_GPU:-0}"
H3_GPUS="${H3_GPUS:-1,2,3,4}"
# Qwen 权重本地目录(必填;HF 上游 id 见仓库根 README 的 Checkpoints 表):
QWEN_MODEL="${QWEN_MODEL:-}"
QWEN_MEM="${QWEN_MEM:-70GiB}"   # 单卡 80G:27B 权重 ~54G + KV,低于此会 offload 到磁盘
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

run_llm() { CUDA_VISIBLE_DEVICES="$QWEN_GPU" python3 -m engine.llm.service "$@"; }
run_h3()  { CUDA_VISIBLE_DEVICES="$H3_GPUS"  python3 -m engine.renderer.service "$@"; }

cmd="${1:-status}"
if { [ "$cmd" = start ] || [ "$cmd" = restart ]; } && [ -z "$QWEN_MODEL" ]; then
  echo "error: QWEN_MODEL is not set (local directory of the Qwen weights;" \
       "upstream id: see the repo-root README, Checkpoints table); e.g." \
       "QWEN_MODEL=/path/to/Qwen $0 start" >&2
  exit 2
fi
case "$cmd" in
  start)
    run_llm start --model "$QWEN_MODEL" --per-gpu-mem "$QWEN_MEM"
    run_h3 start
    ;;
  stop)     run_h3 stop; run_llm stop ;;
  restart)  run_h3 stop || true; run_llm stop || true
            run_llm start --model "$QWEN_MODEL" --per-gpu-mem "$QWEN_MEM"
            run_h3 start ;;
  status)   run_llm status; run_h3 status ;;
  *)        echo "用法: $0 {start|stop|status|restart}" >&2; exit 2 ;;
esac
