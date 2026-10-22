"""
Byte-level Byte-Pair Encoding tokenizer — written from scratch.

Design notes:
- Byte-level base vocab (0..255) so ANY input is representable losslessly.
  This matters for Ghanaian orthography (ɛ ɔ ŋ ɖ ƒ ʋ ɣ Ɛ Ɔ Ŋ ...): we never
  hit an unknown character, we just fall back to its UTF-8 bytes.
- GPT-2 style regex pre-tokenization keeps merges from crossing word/space
  boundaries in silly ways.
- Training = greedy most-frequent-adjacent-pair merging until target vocab.

No external ML libraries. Only `regex` (for Unicode-aware pre-tokenization) + stdlib.
"""
from __future__ import annotations
import json, regex, heapq
from collections import Counter, defaultdict
from pathlib import Path

# GPT-2's pre-tokenization pattern (Unicode-aware; needs the `regex` module).
SPLIT_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|unk|>"]


def get_stats(ids_list, counts=None):
    """Count adjacent symbol pairs across all pre-token sequences (weighted by freq)."""
    counts = counts if counts is not None else Counter()
    for ids, freq in ids_list:
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += freq
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (int,int) -> int   (merge rank order)
        self.vocab = {}           # int -> bytes
        self.special = {}         # str -> int

    # ---------- training (incremental — only touches words a merge affects) ----------
    def train(self, text_iter, vocab_size=16000, verbose=True):
        # 1) pre-tokenize into unique word chunks with frequencies
        word_freq = Counter()
        for text in text_iter:
            for chunk in SPLIT_PATTERN.findall(text):
                word_freq[chunk] += 1
        words = [list(chunk.encode("utf-8")) for chunk in word_freq]
        freqs = [word_freq[chunk] for chunk in word_freq]

        # 2) base vocab = 256 bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256
        n_merges = max(0, vocab_size - 256 - len(SPECIAL_TOKENS))

        # 3) initial pair counts + inverted index (pair -> set of word indices) + max-heap
        pair_counts = Counter()
        where = defaultdict(set)
        for wi, w in enumerate(words):
            f = freqs[wi]
            for pr in zip(w, w[1:]):
                pair_counts[pr] += f
                where[pr].add(wi)
        heap = [(-c, pr) for pr, c in pair_counts.items()]
        heapq.heapify(heap)

        def best():
            # lazy heap: pop until the top entry matches the live count
            while heap:
                negc, pr = heapq.heappop(heap)
                if pair_counts.get(pr, 0) == -negc and -negc > 0:
                    return pr, -negc
            return None, 0

        done = 0
        for _ in range(n_merges):
            pair, count = best()
            if pair is None or count < 2:
                break
            A, B = pair
            Z = next_id
            self.merges[pair] = Z
            self.vocab[Z] = self.vocab[A] + self.vocab[B]

            for wi in list(where[pair]):
                w = words[wi]; f = freqs[wi]
                # remove this word's old pair contributions
                for pr in zip(w, w[1:]):
                    pair_counts[pr] -= f
                    where[pr].discard(wi)
                    if pair_counts[pr] > 0:
                        heapq.heappush(heap, (-pair_counts[pr], pr))
                    else:
                        pair_counts.pop(pr, None)
                # merge A,B -> Z inside the word
                neww, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == A and w[i + 1] == B:
                        neww.append(Z); i += 2
                    else:
                        neww.append(w[i]); i += 1
                words[wi] = neww
                # add new pair contributions
                for pr in zip(neww, neww[1:]):
                    pair_counts[pr] += f
                    where[pr].add(wi)
                    heapq.heappush(heap, (-pair_counts[pr], pr))
            where.pop(pair, None)
            next_id += 1
            done += 1
            if verbose and (done % 1000 == 0 or done == n_merges):
                print(f"  merge {done}/{n_merges}  last={pair} count={count}  vocab={next_id}")

        # 4) special tokens get the final ids
        self.special = {tok: next_id + i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, tid in self.special.items():
            self.vocab[tid] = tok.encode("utf-8")
        if verbose:
            print(f"Done. {done} merges. vocab size = {len(self.vocab)} (incl. {len(self.special)} special)")
        return self

    # ---------- encode / decode ----------
    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats = {p: self.merges[p] for p in zip(ids, ids[1:]) if p in self.merges}
            if not stats:
                break
            pair = min(stats, key=stats.get)   # lowest rank = earliest merge
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text, add_eot=False):
        ids = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        if add_eot:
            ids.append(self.special["<|endoftext|>"])
        return ids

    def decode(self, ids):
        parts = [self.vocab.get(i, b"") for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ---------- persistence (our own format) ----------
    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        # merges in rank order
        with open(d / "merges.txt", "w", encoding="utf-8") as f:
            f.write("# ghana-llm BPE merges (rank order)\n")
            for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
                f.write(f"{a} {b} {idx}\n")
        meta = {
            "special": self.special,
            "vocab_size": self.vocab_size,
            # store vocab bytes as latin-1 so it's reversible text
            "vocab": {str(i): v.decode("latin-1") for i, v in self.vocab.items()},
        }
        with open(d / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        t = cls()
        meta = json.load(open(d / "vocab.json", encoding="utf-8"))
        t.special = meta["special"]
        t.vocab = {int(i): v.encode("latin-1") for i, v in meta["vocab"].items()}
        t.merges = {}
        for line in open(d / "merges.txt", encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b, idx = line.split()
            t.merges[(int(a), int(b))] = int(idx)
        return t
