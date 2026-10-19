#!/bin/bash
# 单卡 worker: 依次跑分配给该卡的任务链(断点续跑,已完成条目自动跳过)
# usage: source baseline/env.sh && bash baseline/worker.sh <gpu> <model> [<model> ...]
#   model ∈ {longcat, hy15, wan22}
set -u
GPU=$1; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
mkdir -p "$LOGS_ROOT"

run_model() {
  local m=$1
  if [ -f "$BASELINE_ROOT/done_${m}" ]; then echo "[worker$GPU] $m already done"; return; fi
  echo "[worker$GPU] start $m at $(date +%F_%T)"
  case $m in
    longcat)
      cd "$CODEBASE_ROOT/LongCat-Video"
      export PATH="$CONDA_ROOT/envs/longcat/bin:$PATH"
      CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 "$BASELINE_ROOT/longcat_batch.py" \
        >> "$LOGS_ROOT/longcat.log" 2>&1 ;;
    hy15)
      cd "$CODEBASE_ROOT/HunyuanVideo-1.5"
      export PATH="$CONDA_ROOT/envs/hunyuan15/bin:$PATH"
      HY15_MODEL="$(ls -d "$MODELS_ROOT"/hub/models--tencent--HunyuanVideo-1.5/snapshots/*/ 2>/dev/null | head -1)"
      CUDA_VISIBLE_DEVICES=$GPU python "$BASELINE_ROOT/hy15_batch.py" \
        --resolution 480p --aspect_ratio 16:9 --video_length 121 --num_inference_steps 50 \
        --rewrite false --sr false --offloading true \
        --model_path "$HY15_MODEL" \
        >> "$LOGS_ROOT/hy15.log" 2>&1 ;;
    wan22)
      cd "$CODEBASE_ROOT/wan2.2"
      export PATH="$CONDA_ROOT/envs/wan22/bin:$PATH"
      CUDA_VISIBLE_DEVICES=$GPU python "$BASELINE_ROOT/wan22_batch.py" \
        >> "$LOGS_ROOT/wan22.log" 2>&1 ;;
    *)
      echo "[worker$GPU] UNKNOWN MODEL $m"; return 1 ;;
  esac
  local rc=$?
  local n=$(ls "$RUNS_ROOT"/$m/raw/*.mp4 "$RUNS_ROOT"/$m/raw_state/*.mp4 2>/dev/null | wc -l)
  echo "[worker$GPU] $m exit=$rc videos=$n at $(date +%F_%T)"
  if [ $rc -eq 0 ] && [ "$n" -ge 200 ]; then touch "$BASELINE_ROOT/done_${m}"; else
    echo "[worker$GPU] $m INCOMPLETE (will retry once)"
    sleep 10
  fi
}

for m in "$@"; do
  # 等该卡上同模型的冒烟进程退出
  while pgrep -f "baseline/${m}_batch.py" > /dev/null; do sleep 60; done
  for attempt in 1 2; do
    run_model "$m"
    [ -f "$BASELINE_ROOT/done_${m}" ] && break
  done
done
echo "[worker$GPU] chain finished at $(date +%F_%T)"
