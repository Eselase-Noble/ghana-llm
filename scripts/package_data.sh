#!/bin/zsh
# Package the corpus (clean text + tokenized shards + tokenizer) for upload to
# YOUR Google Drive. The corpus is never committed to git — this is how it gets
# to Colab. Checkpoints are NOT included (they live/persist on Drive separately).
#
# Usage:   bash scripts/package_data.sh
# Then:    upload ghana-llm-data.tar.gz to Drive MyDrive/ghana-llm/ and unpack:
#            (in Drive or Colab)  tar xzf ghana-llm-data.tar.gz
#          so you end up with MyDrive/ghana-llm/corpus/...
set -e
cd "$(dirname "$0")/.."
OUT=ghana-llm-data.tar.gz
tar czf "$OUT" \
    corpus/clean \
    corpus/tokenized
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Upload it to Google Drive: MyDrive/ghana-llm/  then unpack it there."
