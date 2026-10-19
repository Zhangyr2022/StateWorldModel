#!/bin/bash
# 各模型生成进度: python脚本见 baseline/prompts_common.py; 本脚本统计已有视频
# usage: source baseline/env.sh && bash baseline/progress.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
for d in "$RUNS_ROOT"/*/; do
  m=$(basename "$d")
  r=$(ls "$d"/raw/*.mp4 2>/dev/null | wc -l); s=$(ls "$d"/raw_state/*.mp4 2>/dev/null | wc -l)
  printf "%-12s raw=%3d/100  raw_state=%3d/100  total=%3d/200\n" "$m" "$r" "$s" $((r+s))
done
