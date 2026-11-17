"""Yerel çok dilli embedding modeli (multilingual-e5). İlk çalıştırmada model indirilir."""
from __future__ import annotations

import os
from functools import lru_cache

# Model yükleme sırasındaki ilerleme çubuklarını/uyarıları sustur.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import numpy as np

from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    # E5 modelleri dokümanlar için "passage: " ön ekini bekler.
    return _model().encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                           batch_size=16, show_progress_bar=True)


def embed_query(text: str) -> np.ndarray:
    # ... ve sorgular için "query: " ön ekini.
    return _model().encode([f"query: {text}"], normalize_embeddings=True)[0]
