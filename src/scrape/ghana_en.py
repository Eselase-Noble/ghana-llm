"""
Expanded English crawl: many Ghana-related categories, crawled sequentially
into the same en.jsonl / wiki_en.txt (single writer => safe dedup & append).

Dedup is automatic: wiki.run() reloads the seen-pageid set from the growing
file at the start of each category, so overlap across categories is skipped.
Missing categories just return empty (harmless).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiki

# Seed categories spanning history, peoples, culture, places, language, polity.
CATEGORIES = [
    "Category:Ghana",
    "Category:History of Ghana",
    "Category:Ashanti Empire",
    "Category:Ghana Empire",
    "Category:Akan",
    "Category:Akan people",
    "Category:Culture of Ghana",
    "Category:Ghanaian culture",
    "Category:Ghanaian people",
    "Category:Populated places in Ghana",
    "Category:Regions of Ghana",
    "Category:Languages of Ghana",
    "Category:Ewe people",
    "Category:Ga-Adangbe",
    "Category:Dagbon",
    "Category:Economy of Ghana",
    "Category:Politics of Ghana",
    "Category:Ghanaian cuisine",
    "Category:Religion in Ghana",
    "Category:Education in Ghana",
]

DEPTH = 2

if __name__ == "__main__":
    for i, cat in enumerate(CATEGORIES, 1):
        print(f"\n########## [{i}/{len(CATEGORIES)}] {cat} (depth {DEPTH}) ##########", flush=True)
        try:
            wiki.run("en", "category", cat, DEPTH, None)
        except Exception as e:
            print(f"!! {cat} failed: {e}", flush=True)
    print("\nALL CATEGORIES DONE", flush=True)
