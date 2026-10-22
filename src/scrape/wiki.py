"""
Ghanaian data collector — Wikipedia (MediaWiki API).

Pulls plaintext articles from:
  - English Wikipedia: Ghana-related categories (history, culture, people, ...).
  - Twi (tw) and Ewe (ee) Wikipedias: all articles (they are small).

Design:
  - stdlib only (urllib) so it runs in any Python here.
  - Polite: descriptive User-Agent, rate-limited, batched extract queries.
  - Resumable: records fetched pageids per wiki.
  - Output:
      corpus/raw/scraped/<wiki>.jsonl   (one article: {pageid,title,text,wiki})
      corpus/clean/wiki_<wiki>.txt       (plaintext, one paragraph per line)

This is the ONE deliberate online step that pulls data IN. Collected data stays local.
"""
from __future__ import annotations
import json, time, urllib.parse, urllib.request, argparse, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "corpus" / "raw" / "scraped"
CLEAN = ROOT / "corpus" / "clean"

UA = ("GhanaLLM-corpus-builder/0.1 (non-commercial language-model research; "
      "operated via Python urllib; contact: https://github.com/Eselase-Noble/ghana-llm)")
DELAY = 0.5  # seconds between API calls — polite (paired w/ good UA + maxlag + backoff)


def api(lang, params, retries=6):
    # maxlag: back off automatically if Wikimedia servers are lagged (etiquette).
    params = {**params, "format": "json", "maxlag": "5"}
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    backoff = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.load(r)
            # maxlag error comes back 200 with an error body sometimes
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                raise urllib.error.HTTPError(url, 503, "maxlag", None, None)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                wait = backoff * (attempt + 1)
                # honour Retry-After if present
                ra = getattr(e, "headers", None) and e.headers.get("Retry-After")
                if ra and ra.isdigit():
                    wait = max(wait, int(ra))
                print(f"  [{lang}] {e.code}; backing off {wait:.0f}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1)); continue
            raise


def category_members(lang, category, seen_cats=None, depth=1):
    """Yield page titles under a category (optionally descending subcategories)."""
    seen_cats = seen_cats if seen_cats is not None else set()
    if category in seen_cats or depth < 0:
        return
    seen_cats.add(category)
    cont = {}
    while True:
        d = api(lang, {"action": "query", "list": "categorymembers",
                       "cmtitle": category, "cmlimit": "500",
                       "cmtype": "page|subcat", **cont})
        for m in d.get("query", {}).get("categorymembers", []):
            if m["ns"] == 14 and depth > 0:   # subcategory
                yield from category_members(lang, m["title"], seen_cats, depth - 1)
            elif m["ns"] == 0:                # article
                yield m["title"]
        time.sleep(DELAY)
        if "continue" in d:
            cont = d["continue"]
        else:
            break


def all_pages(lang):
    """Yield every article title in a (small) wiki."""
    cont = {}
    while True:
        d = api(lang, {"action": "query", "list": "allpages",
                       "aplimit": "500", "apnamespace": "0", **cont})
        for p in d.get("query", {}).get("allpages", []):
            yield p["title"]
        time.sleep(DELAY)
        if "continue" in d:
            cont = d["continue"]
        else:
            break


def fetch_extracts(lang, titles):
    """Batch fetch plaintext extracts for a list of titles."""
    for i in range(0, len(titles), 20):
        batch = titles[i:i+20]
        d = api(lang, {"action": "query", "prop": "extracts", "explaintext": "1",
                       "exlimit": "20", "titles": "|".join(batch)})
        for p in d.get("query", {}).get("pages", {}).values():
            if "extract" in p and p["extract"].strip():
                yield {"pageid": p.get("pageid"), "title": p.get("title"),
                       "text": p["extract"], "wiki": lang}
        time.sleep(DELAY)


def clean_text(t):
    t = unicodedata.normalize("NFC", t)
    lines = []
    for ln in t.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("==") and ln.endswith("=="):
            continue  # drop section headings / blanks
        if len(ln) < 20:
            continue  # drop stubs/fragments
        lines.append(ln)
    return lines


def run(wiki, mode, category, depth, limit):
    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{wiki}.jsonl"
    clean_path = CLEAN / f"wiki_{wiki}.txt"

    seen = set()
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            try: seen.add(json.loads(line)["pageid"])
            except: pass
    print(f"[{wiki}] already have {len(seen)} articles")

    # gather titles
    if mode == "all":
        titles = list(all_pages(wiki))
    else:
        titles = list(dict.fromkeys(category_members(wiki, category, depth=depth)))
    if limit:
        titles = titles[:limit]
    print(f"[{wiki}] {len(titles)} candidate titles")

    n_new = 0
    with open(raw_path, "a", encoding="utf-8") as raw_f, \
         open(clean_path, "a", encoding="utf-8") as clean_f:
        for art in fetch_extracts(wiki, titles):
            if art["pageid"] in seen:
                continue
            seen.add(art["pageid"])
            raw_f.write(json.dumps(art, ensure_ascii=False) + "\n")
            for para in clean_text(art["text"]):
                clean_f.write(para + "\n")
            raw_f.flush(); clean_f.flush()
            n_new += 1
            if n_new % 25 == 0:
                print(f"[{wiki}] +{n_new} new (last: {art['title']})")
    print(f"[{wiki}] done. {n_new} new articles -> {raw_path.name}, {clean_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True, help="wiki lang code: en, tw, ee, ...")
    ap.add_argument("--mode", choices=["all", "category"], default="category")
    ap.add_argument("--category", default="Category:Ghana")
    ap.add_argument("--depth", type=int, default=1, help="subcategory recursion depth")
    ap.add_argument("--limit", type=int, default=0, help="cap titles (0 = no cap)")
    args = ap.parse_args()
    run(args.wiki, args.mode, args.category, args.depth, args.limit or None)
