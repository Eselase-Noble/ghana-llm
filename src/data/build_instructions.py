"""
Build the instruction-tuning (SFT) dataset: (prompt, response) pairs.

Sources (all local, already gathered):
  1. Translation  — from the 5-language parallel Bible (verse-aligned). Many
     directions (en<->ewe/twi/fante/ga).
  2. History Q&A  — from scraped English Wikipedia articles (title -> intro).
  3. Seed         — hand-written Ghana/Black-history Q&A + chat (tone/identity),
     upweighted so the model adopts the assistant style.

Output: corpus/instructions/sft.jsonl  (one {"prompt","response"} per line)
This is DATA (gitignored) — it contains Bible-derived pairs; stays off git,
travels to Colab in the Drive bundle.
"""
from __future__ import annotations
import json, random, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIBLE = ROOT / "corpus" / "raw" / "bibles" / "bible_parallel.jsonl"
WIKI_EN = ROOT / "corpus" / "raw" / "scraped" / "en.jsonl"
SEED = ROOT / "src" / "data" / "seed_instructions.json"
OUT_DIR = ROOT / "corpus" / "instructions"

LANG_NAME = {"en": "English", "ewe": "Ewe", "twi": "Twi", "fante": "Fante", "ga": "Ga"}
RNG = random.Random(1337)

N_TRANSLATION_VERSES = 7000   # verses sampled for translation tasks
DIRS_PER_VERSE = 2            # translation directions per sampled verse
SEED_REPEAT = 6              # upweight the hand-written style examples
MAX_ANSWER_CHARS = 450


def norm(s):
    return unicodedata.normalize("NFC", s).strip()


def translation_examples():
    if not BIBLE.exists():
        print("(no parallel Bible; skipping translation)"); return []
    rows = []
    for line in open(BIBLE, encoding="utf-8"):
        r = json.loads(line)
        present = {k: norm(r[k]) for k in LANG_NAME if r.get(k)}
        if len(present) >= 2:
            rows.append(present)
    RNG.shuffle(rows)
    rows = rows[:N_TRANSLATION_VERSES]
    out = []
    for present in rows:
        langs = list(present.keys())
        for _ in range(DIRS_PER_VERSE):
            src, tgt = RNG.sample(langs, 2)
            out.append({
                "prompt": f"Translate to {LANG_NAME[tgt]}:\n{present[src]}",
                "response": present[tgt],
            })
    return out


def _truncate(text):
    text = norm(text).replace("\n", " ")
    if len(text) <= MAX_ANSWER_CHARS:
        return text
    cut = text[:MAX_ANSWER_CHARS]
    dot = cut.rfind(". ")
    return (cut[:dot + 1] if dot > 100 else cut).strip()


def history_qa():
    if not WIKI_EN.exists():
        print("(no wiki articles; skipping Q&A)"); return []
    out = []
    for line in open(WIKI_EN, encoding="utf-8"):
        a = json.loads(line)
        title = (a.get("title") or "").strip()
        text = a.get("text") or ""
        if not title or ":" in title or "(" in title and ")" not in title:
            continue
        # first real paragraph
        para = next((p for p in text.split("\n") if len(p.strip()) > 60), "")
        if not para:
            continue
        ans = _truncate(para)
        if len(ans) < 60:
            continue
        q = RNG.choice([f"Tell me about {title}.",
                        f"What is {title}?",
                        f"Who or what was {title}?"])
        out.append({"prompt": q, "response": ans})
    return out


def seed_examples():
    if not SEED.exists():
        return []
    data = json.load(open(SEED, encoding="utf-8"))
    return [dict(e) for e in data for _ in range(SEED_REPEAT)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = translation_examples()
    qa = history_qa()
    sd = seed_examples()
    allex = tr + qa + sd
    RNG.shuffle(allex)
    out_path = OUT_DIR / "sft.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for e in allex:
            if e["prompt"].strip() and e["response"].strip():
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"translation: {len(tr)} | history Q&A: {len(qa)} | seed: {len(sd)}")
    print(f"total SFT examples: {len(allex)} -> {out_path}")


if __name__ == "__main__":
    main()
