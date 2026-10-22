# Ghana-LLM

A from-scratch Ghanaian / Black-history language model — our own tokenizer,
transformer, training loop, and weights. PyTorch is used only as the tensor +
autograd engine. Nothing pretrained.

**Goal:** a model whose fundamental knowledge centers on Ghana and Black history
(expanding to other African countries over time). Built as a working proof to
justify real training hardware.

## Repo layout
```
src/tokenizer/   our byte-level BPE (handles ɛ ɔ ŋ ɖ ƒ ʋ losslessly)
src/model/       our decoder-only transformer (configurable size)
src/data/        corpus building + tokenization to uint16 shards
src/train/       training loop (CUDA / MPS / CPU, mixed precision, resumable)
src/generate/    sampling / inference
src/scrape/      Ghanaian data collectors (Wikipedia, extensible)
configs/         model + training configs (small=CPU, gpu/base_gpu=Colab)
corpus/          data — NEVER committed (copyrighted/sensitive). Lives locally + on your Drive.
notebooks/       train_colab.ipynb — pull code + train on Colab GPU
scripts/         package_data.sh — bundle corpus for upload to your Drive
```

> **Data boundary:** code goes to GitHub; the **corpus and checkpoints never do**
> (copyrighted Bible text + sensitive project data). `.gitignore` enforces this.
> Colab gets the data from *your* Google Drive, not from the repo.

## Train on Colab (GPU) — primary path
1. Code is on GitHub (private repo).
2. One-time: `bash scripts/package_data.sh` → upload `ghana-llm-data.tar.gz` to
   your Drive `MyDrive/ghana-llm/` and unpack (so `MyDrive/ghana-llm/corpus/...` exists).
3. Open `notebooks/train_colab.ipynb` in Colab, runtime = **T4 GPU**.
4. Run cells: clone code → mount Drive (data + checkpoints) → train → sample.
Checkpoints persist on Drive across Colab session resets; the train cell is
**resumable** — just re-run it to continue.

Configs: `configs/gpu.json` (current data, ~17M params, fast) →
`configs/base_gpu.json` (~85M params, use once the corpus is large).

## Run locally (Intel-mac / CPU) — dev path
```
/usr/local/opt/python@3.11/bin/python3.11 -m venv venv
./venv/bin/pip install -r requirements-mac.txt      # torch 2.2.2 (last Intel-mac build)
./run_pipeline.sh                                    # build corpus -> tokenize -> train -> sample
```

## Data pipeline
```
src/data/build_corpus.py    # Bible parallel jsonl + scraped wiki jsonl -> corpus/clean/*.txt
src/data/prepare.py         # train/load BPE + tokenize -> corpus/tokenized/<dataset>/{train,val}.bin
src/scrape/wiki.py          # one wiki: --wiki en|tw|ee --mode all|category --category ... --depth N
src/scrape/ghana_en.py      # expanded English crawl over many Ghana categories
```

See `ROADMAP.md` for the phased plan and `DATA.md` for the data-gathering strategy.

## Status
v1 (CPU, 17M params, Bible-only) trained to val-perplexity ~37; generates coherent
text in English, Ewe, Twi, Fante, Ga. v2 adds Ghanaian Wikipedia. Next: scale the
corpus substantially, then train a larger model on Colab GPU.
