#!/bin/bash
# 目标机重建 conda 环境（flux2 渲染 / kairos 审查）。
# 用法：bash ops/envs/rebuild_envs.sh <conda安装前缀> [--force]   # 如 ~/miniforge3
#   环境名可用 FLUX2_ENV / KAIROS_ENV 覆盖（默认 flux2 / kairos）。
# 说明：env 实际落盘位置由目标机 ~/.condarc 的 envs_dirs 决定（-n 只给名字）；
#       建好后把 <env实际路径>/bin/python 填入 config.yaml 的 envs.*。
# 依据：ops/envs/*.pip-freeze.txt（2026-09-19 源机导出，已清 conda-forge 脏行）
# 安全：同名 env 已存在时默认报错退出、不做任何改动；只有显式 --force 才删掉重建。
set -euo pipefail

CONDA_PREFIX=${1:?用法: rebuild_envs.sh <conda安装前缀, 如 ~/miniforge3> [--force]}
FORCE=0
[ "${2:-}" = "--force" ] && FORCE=1
FLUX2_ENV=${FLUX2_ENV:-flux2}
KAIROS_ENV=${KAIROS_ENV:-kairos}
HERE=$(cd "$(dirname "$0")" && pwd)

source "$CONDA_PREFIX/etc/profile.d/conda.sh"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }
for env in "$FLUX2_ENV" "$KAIROS_ENV"; do
  if env_exists "$env"; then
    if [ "$FORCE" = 1 ]; then
      conda env remove -n "$env" -y
    else
      echo "conda 环境 '$env' 已存在，未做任何改动。换个名字（FLUX2_ENV=… KAIROS_ENV=…）或加 --force 删除重建。" >&2
      exit 1
    fi
  fi
done

# ---- flux2：Python 3.11，torch 2.14.0（PyPI 版，自带 CUDA）----
conda create -y -n "$FLUX2_ENV" python=3.11
conda activate "$FLUX2_ENV"
pip install -r "$HERE/flux2.pip-freeze.txt"
conda deactivate

# ---- kairos：Python 3.10，torch 2.6.0+cu126（必须走 PyTorch 官方 index）----
conda create -y -n "$KAIROS_ENV" python=3.10
conda activate "$KAIROS_ENV"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
# flash_attn：优先用本地 wheels/ 下的 wheel；没有就取上游发布的预编译 wheel，再不行才源码编译
FA_LOCAL="$HERE/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
if [ -f "$FA_LOCAL" ]; then
  pip install "$FA_LOCAL"
else
  pip install "$FA_URL" || pip install flash-attn==2.7.4.post1 --no-build-isolation
fi
grep -vE '^(torch|torchvision|torchaudio|flash[-_]attn)==' "$HERE/kairos.pip-freeze.txt" \
    | pip install -r /dev/stdin
conda deactivate

echo "完成。env 实际路径见 conda env list；把两个 python 绝对路径填入 config.yaml 的 envs.render_python / envs.review_python"
