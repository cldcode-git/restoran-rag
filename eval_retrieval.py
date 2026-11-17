"""Geri getirme (retrieval) kalitesini ölçer: her soru için beklenen metin,
ilk k chunk'tan birinde geçiyor mu? Dense / BM25 / hibrit karşılaştırması.

    .venv/bin/python eval_retrieval.py
"""
import numpy as np

from embedder import embed_query
from rag import MenuIndex, detect_restaurants, strip_restaurant_names, tr_fold

# (soru, beklenen restoran id, chunk metninde geçmesi gereken ifade)
CASES = [
    ("Big Chefs'te hangi salatalar var ve kalorileri ne?", "bigchefs", "BAHÇE SALATA"),
    ("Big Chefs San Sebastian cheesecake kaç kalori?", "bigchefs", "SAN SEBASTIAN"),
    ("Big Chefs'te Türk kahvesi var mı?", "bigchefs", "TÜRK KAHVESİ"),
    ("Big Chefs kahvaltı fiyatları", "bigchefs", "DOLU DOLU KAHVALTI"),
    ("Big Chefs'te hangi biralar var?", "bigchefs", "Tuborg"),
    ("Big Chefs vegan yemek", "bigchefs", "VEGAN SCHNITZEL"),
    ("Big Chefs glütensiz tatlı", "bigchefs", "GLÜTENSİZ BROWNIE"),
    ("Midpoint burger çeşitleri neler?", "midpoint", "Colorado Burger"),
    ("Midpoint kahvaltı saatleri", "midpoint", "09:00 - 12:00"),
    ("Midpoint'te Anne Kahvaltısı kaç kişilik?", "midpoint", "2 Kişilik"),
    ("Midpoint çocuk menüsü", "midpoint", "ÇOCUKLAR İÇİN"),
    ("Midpoint'te hangi viskiler var?", "midpoint", "Johnnie Walker"),
    ("Midpoint Omega 3 salata içeriği", "midpoint", "Omega 3"),
    ("Midpoint alerjen uyarıları", "midpoint", "Alerjen"),
    ("Happy Moon's burger menüsü", "happymoons", "SIZZLER BURGER"),
    ("Happy Moon's alerjen açıklamaları", "happymoons", "ALERJEN"),
    ("Happy Moon's Happy Wheel içinde ne var?", "happymoons", "HAPPY WHEEL"),
    ("Happy Moon's makarna çeşitleri", "happymoons", "PENNE"),
    ("Happy Moon's kırmızı şarap listesi", "happymoons", "KIRMIZI ŞARAP"),
    ("Happy Moon's bubble tea", "happymoons", "BUBBLE TEA"),
    ("Hangi restoranda mojito var?", None, "MOJITO"),
    ("İçinde somon olan yemekler", None, "somon"),
    ("Tavuk dürüm hangi restoranlarda var?", None, "TAVUKLU DÜRÜM"),
    ("cocuk menusu olan restoran", None, "ÇOCUKLAR İÇİN"),
]


def ranked(index, scores, allowed):
    masked = np.where(allowed, scores, -np.inf)
    return [int(i) for i in np.argsort(-masked) if allowed[i]]


def main(k=6):
    index = MenuIndex()
    hits = {"dense": 0, "bm25": 0, "hybrid": 0}
    for q, rid, expect in CASES:
        rest = detect_restaurants(q)
        query = strip_restaurant_names(q) if rest else q
        allowed = np.array([(rest is None or d.restaurant_id in rest) and d.page != 0 for d in index.docs])
        dense = ranked(index, index.matrix @ embed_query(query), allowed)[:k]
        bm = ranked(index, index.bm25(query), allowed)[:k]
        hyb = [index.docs.index(d) for d in index.search(query, k=k, restaurants=rest)]
        row = []
        for name, ids in (("dense", dense), ("bm25", bm), ("hybrid", hyb)):
            ok = any(tr_fold(expect) in tr_fold(index.docs[i].text) for i in ids)
            hits[name] += ok
            row.append("✓" if ok else "✗")
        print(f"{' '.join(row)}  {q}")
    n = len(CASES)
    print(f"\nhit@{k}: " + ", ".join(f"{m} {h}/{n}" for m, h in hits.items()))


if __name__ == "__main__":
    main()
