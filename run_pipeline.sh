#!/bin/zsh
# End-to-end local pipeline. Run after the venv is set up (see SETUP in README).
set -e
cd "$(dirname "$0")"
PY=${PY:-./venv/bin/python}

echo "== [1/4] build clean corpus from Bible parallel data =="
$PY src/data/build_corpus.py

echo "== [2/4] train tokenizer + tokenize corpus =="
$PY src/data/prepare.py --dataset gh_all --glob 'bible_*.txt' --vocab-size 16000

echo "== [3/4] train model =="
$PY src/train/train.py --config configs/small.json --dataset gh_all

echo "== [4/4] sample =="
$PY src/generate/sample.py --dataset gh_all --prompt "Mawu" --tokens 120

echo "done."
