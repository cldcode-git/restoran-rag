"""Menü PDF'lerini chunk'lar, embedding'leri üretir ve restoran başına bir JSON'a yazar.

Kullanım:
    .venv/bin/python build_index.py              # 3 restoranın hepsi
    .venv/bin/python build_index.py midpoint     # sadece biri
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone

import pymupdf

from chunker import Chunk, chunk_pdf
from config import CHUNK_MAX_CHARS, EMBED_MODEL, INDEX_DIR, RESTAURANTS
from embedder import embed_passages


def overview_chunk(rid: str, cfg: dict, chunks: list[Chunk]) -> Chunk:
    """Restoran hakkında genel soruları ("hangi kategoriler var?") yanıtlamak için özet chunk."""
    sections: list[str] = []
    for c in chunks:
        top = c.section.split(" > ")[0]
        if c.lang == "tr" and top and top not in sections and top != "Sayfa başlıkları / notlar":
            sections.append(top)
    priced = sum(1 for c in chunks if re.search(r"₺\s*\d|\d\s*₺", c.body()))
    has_kcal = any("kcal" in c.body() for c in chunks)
    langs = sorted({c.lang for c in chunks})
    lines = [
        f"{cfg['name']} restoranının menüsü ({pymupdf.open(cfg['pdf']).page_count} sayfa).",
        "Menü bölümleri: " + ", ".join(sections) + ".",
        ("Menüde fiyatlar Türk lirası (₺) olarak yer alıyor."
         if priced >= 10 else "Bu menüde fiyat bilgisi (neredeyse hiç) yer almıyor."),
        ("Ürünlerin kalori (kcal) değerleri menüde belirtilmiş."
         if has_kcal else "Menüde kalori bilgisi yok."),
    ]
    if "en" in langs:
        lines.append("Menünün İngilizce çevirisi de mevcut.")
    return Chunk(rid, cfg["name"], 0, "tr", "Restoran genel bilgisi", lines)


def build(rid: str) -> None:
    cfg = RESTAURANTS[rid]
    chunks = chunk_pdf(rid, cfg, CHUNK_MAX_CHARS)
    chunks.insert(0, overview_chunk(rid, cfg, chunks))
    texts = [c.text() for c in chunks]
    print(f"[{rid}] {len(chunks)} chunk için embedding üretiliyor...")
    vecs = embed_passages(texts)

    data = {
        "restaurant_id": rid,
        "restaurant": cfg["name"],
        "source_pdf": cfg["pdf"].name,
        "embedding_model": EMBED_MODEL,
        "embedding_dim": int(vecs.shape[1]),
        "normalized": True,
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "id": f"{rid}-{i:04d}",
                "page": c.page,
                "lang": c.lang,
                "section": c.section,
                "text": t,
                "embedding": [round(float(x), 6) for x in v],
            }
            for i, (c, t, v) in enumerate(zip(chunks, texts, vecs))
        ],
    }
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out = INDEX_DIR / f"{rid}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"[{rid}] yazıldı: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(RESTAURANTS)
    for rid in targets:
        if rid not in RESTAURANTS:
            sys.exit(f"Bilinmeyen restoran: {rid}. Seçenekler: {', '.join(RESTAURANTS)}")
        build(rid)
