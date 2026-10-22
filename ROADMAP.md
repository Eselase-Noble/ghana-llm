# Ghana-LLM — a sovereign, from-scratch Ghanaian language model

**Principles**
- 100% local. No data, corpus, code, or weights ever leave this machine or go online.
  The *only* online activity is: (a) one-time tooling install (Python/PyTorch), and
  (b) the deliberate data scraper that *pulls Ghanaian data in*.
- Built from first principles: our own tokenizer, our own transformer, our own training
  loop, our own weights. Nothing pretrained. PyTorch is used only as the CPU
  tensor/autograd engine.
- No rush. Correctness and understanding over speed.

**Hardware envelope (fixed — no cloud)**
- Intel i9-9880H, 8 cores, CPU-only training (no usable GPU).
- 16 GB RAM, ~395 GB disk.
- Env: Python 3.11 + torch 2.2.2 (last Intel-macOS build), CPU.
- Realistic model size: ~5–60M params (GPT-2-small class at most).

---

## Phases

### Phase 0 — Environment  ✅ in progress
- Python 3.11 venv, torch 2.2.2 (CPU), numpy, regex. All local thereafter.

### Phase 1 — Data foundation
- Seed corpus: the parallel Bible corpus already built (EN/Ewe/Twi/Fante/Ga, ~31k verses each).
- Scraper (`src/scrape/`): politely fetch Ghanaian-interest text — history, culture,
  news, language resources, Wikipedia (Ghana + Ghanaian languages), gov/edu sources.
  Store raw → clean → dedup. Everything to `corpus/`.
- Cleaning pipeline: normalize unicode (Ghanaian diacritics: ɛ ɔ ŋ ɖ ƒ ʋ etc.), strip
  boilerplate, dedup, language-tag, shard.

### Phase 2 — Tokenizer (ours)
- Byte-level BPE implemented from scratch in Python, trained on our corpus.
- Must handle Ghanaian orthography (special glyphs) losslessly → byte-level fallback.
- Target vocab ~16k–32k. Save `vocab.json` + `merges.txt` (our format).

### Phase 3 — Model (ours)
- Decoder-only transformer (`src/model/gpt.py`), written by us as `nn.Module`:
  token+positional embeddings, pre-norm, multi-head causal self-attention, MLP,
  weight tying, RoPE or learned pos (TBD). Configurable depth/width via `configs/`.

### Phase 4 — Training pipeline
- Data loader over tokenized shards, cross-entropy LM loss, AdamW, cosine LR + warmup,
  grad clipping, grad accumulation (small batches on CPU), checkpointing, resumable,
  train/val split, loss logging. All CPU, multi-threaded.

### Phase 5 — Evaluation & generation
- Held-out perplexity, sampling (`src/generate/`), qualitative checks per language.

### Phase 6 — Iterate
- Grow corpus, retrain tokenizer if needed, scale model within hardware limits,
  curriculum (start on Bible parallel data, add scraped Ghanaian text).

---

## Layout
```
ghana-llm/
  src/tokenizer/   our BPE
  src/model/       transformer
  src/data/        loading, cleaning, sharding
  src/train/       training loop
  src/scrape/      Ghanaian data collectors
  src/generate/    sampling / inference
  corpus/{raw,clean,tokenized}/
  checkpoints/     model weights (ours)
  configs/         model + training configs
  logs/
  tests/
```
