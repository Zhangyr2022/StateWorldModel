#!/bin/bash
# 目标机重建 conda 环境（flux2 渲染 / kairos 审查）。
# 用法：bash ops/envs/rebuild_envs.sh <conda安装前缀>   # 如 ~/miniforge3
# 说明：env 实际落盘位置由目标机 ~/.condarc 的 envs_dirs 决定（-n 只给名字）；
#       建好后把 <env实际路径>/bin/python 填入 config.yaml 的 envs.*。
# 依据：ops/envs/*.pip-freeze.txt（2026-09-19 源机导出，已清 conda-forge 脏行）
# 幂等：已存在的同名 env 先删再建。
set -euo pipefail

CONDA_PREFIX=${1:?用法: rebuild_envs.sh <conda安装前缀, 如 ~/miniforge3>}
HERE=$(cd "$(dirname "$0")" && pwd)

source "$CONDA_PREFIX/etc/profile.d/conda.sh"

# ---- flux2：Python 3.11，torch 2.14.0（PyPI 版，自带 CUDA）----
conda env remove -n flux2 -y 2>/dev/null || true
conda create -y -n flux2 python=3.11
conda activate flux2
pip install -r "$HERE/flux2.pip-freeze.txt"
conda deactivate

# ---- kairos：Python 3.10，torch 2.6.0+cu126（必须走 PyTorch 官方 index）----
conda env remove -n kairos -y 2>/dev/null || true
conda create -y -n kairos python=3.10
conda activate kairos
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
# flash_attn 用源机带来的本地 wheel（PyPI 只有源码包，编译代价大）
pip install "$HERE/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl"
grep -vE '^(torch|torchvision|torchaudio)==' "$HERE/kairos.pip-freeze.txt" \
    | pip install -r /dev/stdin
conda deactivate

echo "完成。env 实际路径见 conda env list；把两个 python 绝对路径填入 config.yaml 的 envs.render_python / envs.review_python"
