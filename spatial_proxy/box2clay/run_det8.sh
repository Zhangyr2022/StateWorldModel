#!/usr/bin/env bash
# WBench 导航子集第一步:8 case × 8 GPU 并行 WildDet3D 检测(2026-09-23)
# 只做 ①检测;②深度/③控制视频暂不跑(控制视频代码待修)。
# case 表:GPU | case_id | nav_cate | 动作 | --texts(开放词表) | 目录后缀
#   0  124  progressive  W+D+left+right      person      person
#   1  120  progressive  W+A+W+D             dog         dog
#   2  14   l_shape      W+W+A+A             man,person  man
#   3  26   roundtrip    A+A+D+D             person      person
#   4  17   repeat       W+W+W+W             jeep,car    jeep
#   5  68   repeat       A+A+A+A             cat,dog     cat
#   6  46   roundtrip    left+left+right+right car       car
#   7  24   zigzag       right+W+right+W     person      person
set -uo pipefail
cd "$(dirname "$0")"

# WBench 数据与解释器:环境变量可覆盖,缺省按 workspace 并排布局/部署约定
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${WBENCH_DATA:-$HERE/../../WBench/data}"
PY="${WILDDET_PYTHON:-${CONDA_ENVS_ROOT:-$HOME/miniconda3/envs}/wilddet3d/bin/python}"
LOGDIR=results/subject/logs
mkdir -p "$LOGDIR"

run_one() {
    local gpu=$1 cid=$2 texts=$3 suffix=$4
    local img=$BASE/images/case_${cid}.jpg
    local out=results/subject/case_${cid}_${suffix}
    if [ -f "$out/det.json" ]; then
        echo "[gpu$gpu] case_$cid 已有 det.json,跳过"
        return 0
    fi
    echo "[gpu$gpu] case_$cid texts=$texts → $out"
    CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 "$PY" detect_boxes.py \
        "$img" --out results/subject --texts "$texts" \
        > "$LOGDIR/det_case_${cid}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "results/subject/case_${cid}/det.json" ]; then
        mv "results/subject/case_${cid}" "$out"
        echo "[gpu$gpu] case_$cid DONE → $out"
    else
        echo "[gpu$gpu] case_$cid FAILED rc=$rc (log: $LOGDIR/det_case_${cid}.log)"
    fi
}

run_one 0 124 person        person &
run_one 1 120 dog           dog    &
run_one 2 14  man,person    person &
run_one 3 26  person        person &
run_one 4 17  jeep,car      jeep   &
run_one 5 68  cat,dog       dog    &
run_one 6 46  car           car    &
run_one 7 24  person        person &
wait
echo "ALL DONE"
