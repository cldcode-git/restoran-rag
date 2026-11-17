"""Proje ayarları: restoranlar, dosya yolları ve model isimleri."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "index"

# Her restoran: kaynak PDF, görünen ad ve (varsa) İngilizce sayfalar.
RESTAURANTS = {
    "bigchefs": {
        "name": "Big Chefs",
        "pdf": Path.home() / "Downloads" / "r1.pdf",
        "en_pages": {2},  # 2. sayfa menünün İngilizce çevirisi
        "aliases": ["big chefs", "bigchefs", "big chef"],
    },
    "midpoint": {
        "name": "Midpoint",
        "pdf": Path.home() / "Downloads" / "r2.pdf",
        "en_pages": set(),
        "aliases": ["midpoint", "mid point"],
    },
    "happymoons": {
        "name": "Happy Moon's",
        "pdf": Path.home() / "Downloads" / "r3.pdf",
        "en_pages": set(),
        "aliases": ["happy moon's", "happy moons", "happy moon", "happymoons"],
    },
}

# Yerel, çok dilli (Türkçe destekli) embedding modeli. API anahtarı gerektirmez.
EMBED_MODEL = "intfloat/multilingual-e5-base"

# Cevap üretimi için Claude modeli.
CLAUDE_MODEL = "claude-opus-5-5"

# Chunk boyutu (karakter).
CHUNK_MAX_CHARS = 800
