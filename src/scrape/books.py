"""
Public-domain BOOKS ingestion — Project Gutenberg via the Gutendex API.

Why books: dense, long-form, factual prose — far richer per document than
Wikipedia stubs. We focus on Ghana / Black / West-African history and, importantly,
first-person slave narratives & abolitionist works (authentic Black-history voices,
e.g. Equiano, Douglass), not just colonial-era accounts.

- Gutendex (https://gutendex.com) = free JSON search over the Gutenberg catalog.
- Download each book's plain-text, strip the Gutenberg header/footer boilerplate,
  clean into paragraphs.
- stdlib only; polite (UA + delay); resumable (dedup by Gutenberg id).

Output:
  corpus/raw/books/gutenberg.jsonl   {id,title,authors,subjects,text}
  corpus/clean/books_gutenberg.txt   cleaned paragraphs (one per line)

NOTE on bias: historical texts include dated/colonial framing. We lean the queries
toward Black-authored narratives & abolitionist works, but curation matters — review
before heavy weighting in training.
"""
from __future__ import annotations
import sys, json, time, re, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "corpus" / "raw" / "books"
CLEAN = ROOT / "corpus" / "clean"

UA = ("GhanaLLM-corpus-builder/0.1 (non-commercial language-model research; "
      "contact: https://github.com/Eselase-Noble/ghana-llm)")
DELAY = 1.0

# Curated search queries — Ghana + Black/West-African history + diaspora voices.
QUERIES = [
    "Ashanti", "Gold Coast Africa", "Ghana", "Dahomey", "Yoruba",
    "West Africa", "slave trade", "slave narrative", "slavery abolition",
    "Africa history", "Negro history", "Benin Africa", "Fetish West Africa",
    "African folklore", "Equiano", "Frederick Douglass", "Booker Washington",
    "Liberia", "Sierra Leone", "Pan-African",
]

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def http_get(url, as_json=False, retries=4):
    backoff = 2.0
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            return json.loads(data) if as_json else data.decode("utf-8", "replace")
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))


def gutendex_search(query):
    """Yield book records for a query (follows pagination)."""
    url = "https://gutendex.com/books?" + urllib.parse.urlencode({"search": query, "languages": "en"})
    while url:
        d = http_get(url, as_json=True)
        for b in d.get("results", []):
            yield b
        url = d.get("next")
        time.sleep(DELAY)


def plain_text_url(formats):
    # prefer utf-8 plain text; skip zips
    for mime, u in formats.items():
        if mime.startswith("text/plain") and "charset=utf-8" in mime and not u.endswith(".zip"):
            return u
    for mime, u in formats.items():
        if mime.startswith("text/plain") and not u.endswith(".zip"):
            return u
    return None


def strip_boilerplate(text):
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[:m.start()]
    return text


def clean_paras(text):
    text = strip_boilerplate(text)
    # join hard-wrapped lines into paragraphs (blank line = paragraph break)
    paras, buf = [], []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            if buf:
                paras.append(" ".join(buf)); buf = []
        else:
            buf.append(s)
    if buf:
        paras.append(" ".join(buf))
    return [p for p in paras if len(p) >= 40]


def run(limit_per_query=0):
    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / "gutenberg.jsonl"
    clean_path = CLEAN / "books_gutenberg.txt"

    seen = set()
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            try: seen.add(json.loads(line)["id"])
            except: pass
    print(f"already have {len(seen)} books", flush=True)

    n_new = 0
    with open(raw_path, "a", encoding="utf-8") as rawf, open(clean_path, "a", encoding="utf-8") as cf:
        for q in QUERIES:
            print(f"\n### query: {q!r} ###", flush=True)
            got = 0
            for b in gutendex_search(q):
                if limit_per_query and got >= limit_per_query:
                    break
                bid = b.get("id")
                if bid in seen:
                    continue
                url = plain_text_url(b.get("formats", {}))
                if not url:
                    continue
                try:
                    raw = http_get(url)
                except Exception as e:
                    print(f"  skip {bid}: {e}", flush=True); continue
                paras = clean_paras(raw)
                if len(paras) < 20:   # skip tiny/failed
                    continue
                seen.add(bid); got += 1; n_new += 1
                rec = {"id": bid, "title": b.get("title"),
                       "authors": [a.get("name") for a in b.get("authors", [])],
                       "subjects": b.get("subjects", []),
                       "text": "\n".join(paras)}
                rawf.write(json.dumps(rec, ensure_ascii=False) + "\n"); rawf.flush()
                for p in paras:
                    cf.write(p + "\n")
                cf.flush()
                print(f"  +{rec['title'][:60]!r} ({len(paras)} paras)", flush=True)
                time.sleep(DELAY)
    print(f"\nDONE. {n_new} new books -> {raw_path.name}, {clean_path.name}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-query", type=int, default=0, help="cap books per query (0=all)")
    args = ap.parse_args()
    run(args.limit_per_query)
