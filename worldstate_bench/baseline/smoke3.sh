#!/bin/bash
# 三文本模型冒烟: GPU0=longcat GPU1=hy15 GPU2=wan22, 各跑 B01-A.raw 一条(真实配置)
# usage: source baseline/env.sh && bash baseline/smoke3.sh
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
mkdir -p "$LOGS_ROOT"

cd "$CODEBASE_ROOT/LongCat-Video"
export CUDA_VISIBLE_DEVICES=0 PATH="$CONDA_ROOT/envs/longcat/bin:$PATH"
nohup torchrun --nproc_per_node=1 "$BASELINE_ROOT/longcat_batch.py" --only B01-A.raw \
  > "$LOGS_ROOT/longcat_smoke.log" 2>&1 &

cd "$CODEBASE_ROOT/HunyuanVideo-1.5"
export CUDA_VISIBLE_DEVICES=1 PATH="$CONDA_ROOT/envs/hunyuan15/bin:$PATH"
HY15_MODEL="$(ls -d "$MODELS_ROOT"/hub/models--tencent--HunyuanVideo-1.5/snapshots/*/ | head -1)"
nohup python "$BASELINE_ROOT/hy15_batch.py" \
  --resolution 480p --aspect_ratio 16:9 --video_length 121 --num_inference_steps 50 \
  --rewrite false --sr false --offloading true \
  --model_path "$HY15_MODEL" \
  --only B01-A.raw \
  > "$LOGS_ROOT/hy15_smoke.log" 2>&1 &

cd "$CODEBASE_ROOT/wan2.2"
export CUDA_VISIBLE_DEVICES=2 PATH="$CONDA_ROOT/envs/wan22/bin:$PATH"
nohup python "$BASELINE_ROOT/wan22_batch.py" --only B01-A.raw \
  > "$LOGS_ROOT/wan22_smoke.log" 2>&1 &
echo "smokes launched"
