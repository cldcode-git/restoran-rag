"""Chunk'ları incelemek için data/debug/<id>_chunks.txt dosyalarına yazar (embedding üretmez).

    .venv/bin/python inspect_chunks.py
"""
import statistics

from chunker import chunk_pdf
from config import CHUNK_MAX_CHARS, RESTAURANTS, ROOT

DEBUG_DIR = ROOT / "data" / "debug"

for rid, cfg in RESTAURANTS.items():
    cs = chunk_pdf(rid, cfg, CHUNK_MAX_CHARS)
    L = [len(c.body()) for c in cs]
    print(f"{rid}: {len(cs)} chunk, uzunluk min/medyan/max {min(L)}/{statistics.median(L)}/{max(L)}, "
          f"bölümsüz: {sum(1 for c in cs if not c.section)}")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_DIR / f"{rid}_chunks.txt", "w") as f:
        for i, c in enumerate(cs):
            f.write(f"\n----- #{i} s.{c.page} {c.lang}\n{c.text()}\n")
