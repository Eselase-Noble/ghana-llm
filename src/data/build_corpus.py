"""
Build clean plaintext training shards from source corpora.

For now the seed source is the parallel Bible corpus (EN/Ewe/Twi/Fante/Ga).
Later the scraper's output (corpus/clean/*.txt) is picked up here too.

Output: corpus/clean/<name>.txt  (one document / passage per line block,
verses joined per chapter so the model sees running text, not isolated verses).
"""
from __future__ import annotations
import json, unicodedata
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
BIBLE_JSONL = ROOT / "corpus" / "raw" / "bibles" / "bible_parallel.jsonl"
SCRAPED = ROOT / "corpus" / "raw" / "scraped"
BOOKS_JSONL = ROOT / "corpus" / "raw" / "books" / "gutenberg.jsonl"
OUT_DIR = ROOT / "corpus" / "clean"

# Light curation for books: drop colonial-adventure FICTION so a Black-history
# model isn't trained on imperialist framing / fiction-as-fact. We keep slave
# narratives, abolitionist works, histories, and factual accounts.
FICTION_MARKERS = ("fiction",)  # Gutenberg subject tag
AUTHOR_BLOCKLIST = ("henty",)   # G.A. Henty — colonial adventure novels

LANGS = ["en", "ewe", "twi", "fante", "ga"]
MIN_PARA = 20  # drop fragments shorter than this many chars


def normalize(text: str) -> str:
    # NFC keeps Ghanaian diacritics composed and consistent.
    return unicodedata.normalize("NFC", text).strip()


def build_from_bible():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BIBLE_JSONL.exists():
        print(f"!! {BIBLE_JSONL} not found; skipping bible build")
        return
    # group verses by (lang, book, chapter) to form running-text chapters
    chapters = defaultdict(list)  # (lang, book, chap) -> [(verse, text)]
    with open(BIBLE_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for lang in LANGS:
                t = r.get(lang)
                if t:
                    chapters[(lang, r["book"], r["chapter"])].append((r["verse"], normalize(t)))

    per_lang_counts = defaultdict(int)
    writers = {lang: open(OUT_DIR / f"bible_{lang}.txt", "w", encoding="utf-8") for lang in LANGS}
    # deterministic order
    for (lang, book, chap) in sorted(chapters.keys()):
        verses = sorted(chapters[(lang, book, chap)], key=lambda x: x[0])
        passage = " ".join(t for _, t in verses)
        writers[lang].write(passage + "\n")
        per_lang_counts[lang] += 1
    for w in writers.values():
        w.close()

    print("Wrote clean chapter-passages:")
    for lang in LANGS:
        p = OUT_DIR / f"bible_{lang}.txt"
        size = p.stat().st_size
        print(f"  {lang:6s} {per_lang_counts[lang]:5d} passages  {size/1024:.0f} KB  -> {p.name}")


def clean_paras(text):
    """Split an article extract into clean paragraph lines."""
    out = []
    for ln in normalize(text).split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("==") and ln.endswith("=="):
            continue  # section heading
        if len(ln) < MIN_PARA:
            continue
        out.append(ln)
    return out


def build_from_scraped():
    """Rebuild clean wiki_<wiki>.txt deterministically from raw scraped jsonl."""
    if not SCRAPED.exists():
        print("(no scraped data yet; skipping)")
        return
    files = sorted(SCRAPED.glob("*.jsonl"))
    if not files:
        print("(no scraped jsonl yet; skipping)")
        return
    print("Rebuilding clean text from scraped articles:")
    for jf in files:
        wiki = jf.stem
        n_art = n_para = 0
        with open(OUT_DIR / f"wiki_{wiki}.txt", "w", encoding="utf-8") as out:
            seen = set()
            for line in open(jf, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("pageid") in seen:
                    continue
                seen.add(r.get("pageid"))
                paras = clean_paras(r.get("text", ""))
                if not paras:
                    continue
                n_art += 1
                for p in paras:
                    out.write(p + "\n")
                    n_para += 1
        size = (OUT_DIR / f"wiki_{wiki}.txt").stat().st_size
        print(f"  wiki_{wiki:3s} {n_art:5d} articles  {n_para:6d} paras  {size/1024:.0f} KB")


def build_from_books():
    """Clean Gutenberg books -> books_gutenberg.txt, with LIGHT curation
    (drop fiction + colonial-adventure authors)."""
    if not BOOKS_JSONL.exists():
        print("(no books jsonl yet; skipping)")
        return
    kept = dropped = n_para = 0
    dropped_titles = []
    with open(OUT_DIR / "books_gutenberg.txt", "w", encoding="utf-8") as out:
        for line in open(BOOKS_JSONL, encoding="utf-8"):
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            subjects = " ".join(b.get("subjects", [])).lower()
            authors = " ".join(b.get("authors", [])).lower()
            is_fiction = any(m in subjects for m in FICTION_MARKERS)
            is_blocked = any(a in authors for a in AUTHOR_BLOCKLIST)
            if is_fiction or is_blocked:
                dropped += 1
                dropped_titles.append(b.get("title", "?")[:55])
                continue
            kept += 1
            for para in clean_paras(b.get("text", "")):
                out.write(para + "\n")
                n_para += 1
    size = (OUT_DIR / "books_gutenberg.txt").stat().st_size
    print(f"Books (light curation): kept {kept}, dropped {dropped} "
          f"(fiction/colonial) -> {n_para} paras, {size/1024/1024:.1f} MB")
    for t in dropped_titles:
        print(f"    dropped: {t}")


if __name__ == "__main__":
    build_from_bible()
    print()
    build_from_scraped()
    print()
    build_from_books()
