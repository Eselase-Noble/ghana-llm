"""Smoke tests: BPE round-trips losslessly, incl. Ghanaian diacritics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenizer.bpe import BPETokenizer

SAMPLES = [
    "In the beginning God created the heaven and the earth.",
    "Le gɔmedzedzea me la, Mawu wɔ dziƒo kple anyigba.",      # Ewe
    "Mfitiaseɛ no Onyankopɔn bɔɔ ɔsoro ne asase.",             # Twi
    "Shishijee mli lɛ Nyɔŋmɔ bɔ ŋwɛi kɛ shikpɔŋ.",             # Ga
    "Ahyɛse no Nyankopɔn bɔɔ sor nye asaase.",                 # Fante
]


def test_roundtrip_untrained():
    # byte-level base vocab alone must already round-trip anything
    t = BPETokenizer()
    t.vocab = {i: bytes([i]) for i in range(256)}
    t.special = {"<|pad|>": 256, "<|endoftext|>": 257, "<|unk|>": 258}
    for i, tok in enumerate(["<|pad|>", "<|endoftext|>", "<|unk|>"]):
        t.vocab[256 + i] = tok.encode("utf-8")
    for s in SAMPLES:
        assert t.decode(t.encode(s)) == s, f"roundtrip failed (untrained): {s!r}"


def test_train_and_roundtrip():
    corpus = SAMPLES * 50
    t = BPETokenizer().train(iter(corpus), vocab_size=400, verbose=False)
    for s in SAMPLES:
        assert t.decode(t.encode(s)) == s, f"roundtrip failed (trained): {s!r}"
    # merges should have compressed something
    assert t.vocab_size > 256
    # encoding should be shorter than raw bytes for in-domain text
    raw = len(SAMPLES[0].encode("utf-8"))
    enc = len(t.encode(SAMPLES[0]))
    assert enc <= raw


if __name__ == "__main__":
    test_roundtrip_untrained()
    test_train_and_roundtrip()
    print("tokenizer tests passed")
