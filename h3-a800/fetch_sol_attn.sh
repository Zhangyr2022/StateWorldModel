#!/usr/bin/env bash
# Fetch the Sol-Attn kernel package used by `--attention-backend sol` (not redistributed here).
#
#   bash fetch_sol_attn.sh            # → h3_runtime/third_party/sol_attn/
#
# Source: NVlabs/Sana, branch sol-engine, pinned commit below, path
# models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn (Apache-2.0; nested FlashAttention
# BSD-3 and cuDNN Frontend Apache-2.0 notices ship with it). One local patch is applied:
# patches/sol_attn-pin-num-stages.patch pins the Triton autotune num_stages to 1 (DESIGN.md §7.1).
# The dense and vsa backends do not need this package.
#
# Overrides: SANA_REPO (git URL or local clone), SANA_COMMIT, DEST. Alternatively point
# H3_SOL_ATTN_ROOT at a directory that contains a sol_attn/ package.
set -euo pipefail
cd "$(dirname "$0")"
SANA_REPO="${SANA_REPO:-https://github.com/NVlabs/Sana.git}"
SANA_COMMIT="${SANA_COMMIT:-8e0db4fa562d727ea28b8d63015c196db7d97cae}"
SRC_PATH="models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn"
DEST="${DEST:-h3_runtime/third_party/sol_attn}"

[ -e "$DEST" ] && { echo "$DEST already exists; remove it to fetch again" >&2; exit 1; }
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git -C "$tmp" init -q
git -C "$tmp" remote add origin "$SANA_REPO"
git -C "$tmp" sparse-checkout set --no-cone "/$SRC_PATH/"
for attempt in 1 2 3; do
  if git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$SANA_COMMIT"; then break; fi
  [ "$attempt" = 3 ] && { echo "fetch from $SANA_REPO failed" >&2; exit 1; }
  sleep 5
done
git -C "$tmp" checkout -q FETCH_HEAD

mkdir -p "$(dirname "$DEST")"
cp -a "$tmp/$SRC_PATH" "$DEST"
patch -s -p1 -d "$(dirname "$DEST")" < patches/sol_attn-pin-num-stages.patch
echo "sol_attn @ ${SANA_COMMIT:0:12} → $DEST ($(find "$DEST" -type f | wc -l) files, num_stages pinned)"
