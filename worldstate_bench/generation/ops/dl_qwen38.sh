#!/bin/bash
# 从 hf-mirror 并行下载 Qwen3.8-27B（18 分片 + 配置，可断点续传）
# usage: bash ops/dl_qwen38.sh <目标目录>    # 如 /data/xxx/models/Qwen3.8-27B
set -e
D=${1:?用法: dl_qwen38.sh <目标目录>}
BASE=https://hf-mirror.com/Qwen/Qwen3.8-27B/resolve/main
mkdir -p "$D" && cd "$D"
for f in .gitattributes LICENSE README.md chat_template.jinja config.json crc32.txt generation_config.json merges.txt model.safetensors.index.json preprocessor_config.json tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do
  wget -c -q --tries=5 "$BASE/$f" -O "$f" || true
done
seq -w 1 18 | sed 's/^/model-000/;s/$/-of-00018.safetensors/' | \
  xargs -P 4 -I{} sh -c 'wget -c --tries=5 -q "'"$BASE"'/{}" -O "{}" && echo "done {}"'
echo "ALL_FILES_DONE"
