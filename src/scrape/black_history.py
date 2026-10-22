"""
Focused crawl: Ghana + Black history (the project's fundamental-knowledge core).
Other African countries come LATER — this stays centered on Ghana and the
Black/African-diaspora story that radiates from it.

Single-writer, sequential across categories -> safe dedup/append into en.jsonl.
Reuses wiki.run(). Missing categories return empty (harmless).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiki

CATEGORIES = [
    # --- Ghana core ---
    "Category:Ghana",
    "Category:History of Ghana",
    "Category:Precolonial history of Ghana",
    "Category:Gold Coast (British colony)",
    "Category:Ashanti Empire",
    "Category:Ashanti",
    "Category:Akan",
    "Category:Akan people",
    "Category:Fante people",
    "Category:Ga-Adangbe",
    "Category:Ewe people",
    "Category:Kingdoms of Ghana",
    "Category:Regions of Ghana",
    "Category:Populated places in Ghana",
    "Category:Ghanaian people",
    "Category:Ghanaian culture",
    "Category:Culture of Ghana",
    "Category:Kwame Nkrumah",
    "Category:Politics of Ghana",
    "Category:Ghanaian cuisine",
    "Category:Languages of Ghana",
    # --- Black history / diaspora radiating from Ghana & West Africa ---
    "Category:Atlantic slave trade",
    "Category:Slavery in Africa",
    "Category:African diaspora",
    "Category:Pan-Africanism",
    "Category:History of West Africa",
    "Category:West African kingdoms",
    "Category:African empires",
    "Category:Trans-Saharan trade",
    "Category:Abolitionism",
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
