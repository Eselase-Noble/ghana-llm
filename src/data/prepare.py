"""
Train (or load) our BPE tokenizer on the clean corpus, then tokenize the
corpus into compact uint16 binary shards for fast memmapped training.

Outputs under corpus/tokenized/<dataset>/:
  train.bin, val.bin   (uint16 token ids, concatenated, EOT-separated)
  meta.json            (vocab_size, counts)
And the tokenizer under: corpus/tokenized/<dataset>/tokenizer/
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tokenizer.bpe import BPETokenizer

CLEAN = ROOT / "corpus" / "clean"
TOKD = ROOT / "corpus" / "tokenized"


def read_docs(files):
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gh_all", help="name for this tokenized set")
    ap.add_argument("--glob", default="bible_*.txt", help="which clean files to include")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--retrain-tokenizer", action="store_true")
    args = ap.parse_args()

    files = sorted(CLEAN.glob(args.glob))
    if not files:
        print(f"No clean files matching {args.glob} in {CLEAN}. Run build_corpus.py first.")
        sys.exit(1)
    print("Source files:", [f.name for f in files])

    outdir = TOKD / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    tok_dir = outdir / "tokenizer"

    # --- tokenizer ---
    if tok_dir.exists() and not args.retrain_tokenizer:
        print("Loading existing tokenizer...")
        tok = BPETokenizer.load(tok_dir)
    else:
        print(f"Training BPE tokenizer (target vocab {args.vocab_size})...")
        tok = BPETokenizer().train(read_docs(files), vocab_size=args.vocab_size)
        tok.save(tok_dir)
        print(f"Saved tokenizer -> {tok_dir}")

    # --- tokenize corpus ---
    # Shuffle DOCUMENTS (seeded) before concatenating so the val split is a
    # representative mix of all languages/sources — not just the first file.
    import random
    eot = tok.special["<|endoftext|>"]
    all_docs = list(read_docs(files))
    random.Random(1337).shuffle(all_docs)
    docs = len(all_docs)

    # split at the DOCUMENT level, then tokenize each partition
    n_val_docs = max(1, int(docs * args.val_frac))
    val_docs = all_docs[:n_val_docs]
    train_docs = all_docs[n_val_docs:]

    def encode_docs(doc_list):
        ids = []
        for d in doc_list:
            ids.extend(tok.encode(d)); ids.append(eot)
        return np.array(ids, dtype=np.uint16)

    train = encode_docs(train_docs)
    val = encode_docs(val_docs)
    assert tok.vocab_size <= 65536, "uint16 can't hold this vocab; use uint32"
    train.tofile(outdir / "train.bin")
    val.tofile(outdir / "val.bin")
    total = int(len(train) + len(val))

    meta = {
        "vocab_size": tok.vocab_size,
        "n_tokens_total": total,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_docs": docs,
        "val_split": "document-level, seed=1337, stratified by shuffle",
        "files": [f.name for f in files],
    }
    json.dump(meta, open(outdir / "meta.json", "w"), indent=2)
    print(f"Tokens: {total:,}  (train {len(train):,} / val {len(val):,})")
    print(f"Vocab: {tok.vocab_size}  Docs: {docs}")
    print(f"Wrote {outdir}/train.bin, val.bin, meta.json")


if __name__ == "__main__":
    main()
