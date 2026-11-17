"""Restoran menüleri üzerinde RAG: hibrit arama (embedding + BM25) + Claude ile cevap.

Kullanım:
    .venv/bin/python rag.py "Big Chefs'te hangi salatalar var?"
    .venv/bin/python rag.py                      # etkileşimli mod
    .venv/bin/python rag.py --no-llm "vegan"     # sadece bulunan chunk'ları göster
    .venv/bin/python rag.py -r midpoint "burger" # tek restoranla sınırla
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

from config import CLAUDE_MODEL, INDEX_DIR, RESTAURANTS

TOP_K = 10
CLAUDE_EFFORT = "medium"  # basit sorular için "low" daha hızlı; karşılaştırmalar için "high"

KNOWN_NAMES = ", ".join(cfg["name"] for cfg in RESTAURANTS.values())

SYSTEM_PROMPT = f"""Sen yalnızca şu {len(RESTAURANTS)} restoranın menüleri hakkında soruları yanıtlayan bir asistansın: {KNOWN_NAMES}.

Kurallar:
- Soru bu listede olmayan bir restoran, kafe ya da zincir hakkındaysa (ör. "Starbucks'ta latte var mı?"), o restoran hakkında bilgin olmadığını açıkça söyle ve bağlamdaki menü parçalarını o restorana aitmiş gibi kullanma. İstersen yalnızca bilgin olan restoranları ({KNOWN_NAMES}) belirt. Soru hem bilinen hem bilinmeyen restoranları içeriyorsa bilinenler için cevap ver, bilinmeyen için bilgin olmadığını söyle.
- Yalnızca <menu_baglami> içindeki menü parçalarına dayan. Parçalarda olmayan bir bilgiyi (fiyat, içerik, kalori, saat vb.) uydurma; yoksa açıkça "menüde bu bilgi yok" de.
- Menü metni PDF'ten otomatik çıkarıldı: satır sırası ve büyük/küçük harf bozuk olabilir. Bir yemeğin adı genellikle büyük harfli/ayrı satırdadır, altındaki satırlar açıklaması ve "₺ 570" gibi satırlar fiyatıdır. Bir satırda iki fiyat varsa ("₺ 675 ₺ 2.385") bunlar genellikle kadeh/sürahi ya da şişe/kadeh fiyatlarıdır; yakındaki "(Kadeh ..., sürahi ...)" gibi notlara göre yorumla, emin değilsen bunu söyle.
- Her bilginin hangi restorana ait olduğunu belirt. Birden fazla restoran ilgiliyse restoran restoran grupla.
- Diyet/alerjen sorularında (vegan, glutensiz, fıstık vb.) içerik listesinden çıkarım yapabilirsin, ama bunun menü açıklamasına dayalı bir tahmin olduğunu ve restorana teyit ettirilmesi gerektiğini söyle.
- Sorunun dilinde cevap ver. Kısa ve net ol; listelerde madde işareti kullan."""


# --------------------------------------------------------------------------- #
# Metin normalizasyonu (Türkçe)
# --------------------------------------------------------------------------- #
_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def tr_fold(text: str) -> str:
    """Türkçe-duyarlı küçük harf + aksan katlama ("İÇECEK" -> "icecek", "çocuk" == "cocuk")."""
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return text.translate(_FOLD)


# Soru kalıplarında geçen, içerikle ilgisiz kelimeler (katlanmış biçimde).
STOPWORDS = set("""
var yok mi mu ne neler nedir hangi hangisi hangileri ve ile ya veya da de ta te bir bu su o
icin kac nasil olan nerede menu menusu menude menuden restoran restoranda restoranin
restoranlarda cesit cesitleri cesitler secenek secenekleri bana soyle listele
the a an is are what which do does have has there in of and or with any
""".split())


def tokenize(text: str) -> list[str]:
    """Katla, stopword at ve F5 kök bulma uygula (Türkçe ekleri için: "salatalar" -> "salat")."""
    out = []
    for t in re.findall(r"[a-z0-9]+", tr_fold(text)):
        if len(t) < 2 or t in STOPWORDS:
            continue
        out.append(t if t.isdigit() else t[:5])
    return out


# --------------------------------------------------------------------------- #
# İndeks
# --------------------------------------------------------------------------- #
@dataclass
class Doc:
    id: str
    restaurant_id: str
    restaurant: str
    page: int
    section: str
    text: str


class MenuIndex:
    def __init__(self):
        self.docs: list[Doc] = []
        vecs = []
        files = sorted(INDEX_DIR.glob("*.json"))
        if not files:
            sys.exit(f"{INDEX_DIR} içinde indeks yok. Önce: .venv/bin/python build_index.py")
        models = set()
        for f in files:
            data = json.loads(f.read_text())
            models.add(data["embedding_model"])
            for c in data["chunks"]:
                self.docs.append(Doc(c["id"], data["restaurant_id"], data["restaurant"],
                                     c["page"], c["section"], c["text"]))
                vecs.append(c["embedding"])
        if len(models) > 1:
            sys.exit(f"İndeksler farklı modellerle üretilmiş: {models}. build_index.py'yi yeniden çalıştırın.")
        self.matrix = np.asarray(vecs, dtype=np.float32)
        self._build_bm25()

    def _build_bm25(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(d.text)) for d in self.docs]
        self.doc_len = np.array([sum(c.values()) for c in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean())
        df = Counter(t for c in self.doc_tokens for t in c)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.docs), dtype=np.float32)
        q = set(tokenize(query))
        for i, tf in enumerate(self.doc_tokens):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
            scores[i] = s
        return scores

    def overview(self, rid: str) -> Doc | None:
        return next((d for d in self.docs if d.restaurant_id == rid and d.page == 0), None)

    def search(self, query: str, k: int = TOP_K, restaurants: list[str] | None = None) -> list[Doc]:
        from embedder import embed_query

        allowed = np.array([restaurants is None or d.restaurant_id in restaurants for d in self.docs])
        allowed &= np.array([d.page != 0 for d in self.docs])  # genel bilgi chunk'ları ayrıca eklenir
        dense = self.matrix @ embed_query(query)
        sparse = self.bm25(query)

        # Reciprocal Rank Fusion: iki sıralamayı ölçek farkından bağımsız birleştir.
        fused = np.zeros(len(self.docs))
        for scores in (dense, sparse):
            masked = np.where(allowed, scores, -np.inf)
            order = np.argsort(-masked)
            for rank, idx in enumerate(order):
                if not allowed[idx] or (scores is sparse and sparse[idx] <= 0):
                    continue
                fused[idx] += 1.0 / (60 + rank)
        top = [int(i) for i in np.argsort(-fused)[:k] if fused[i] > 0]
        return [self.docs[i] for i in top]


def detect_restaurants(question: str) -> list[str] | None:
    q = tr_fold(question)
    found = [rid for rid, cfg in RESTAURANTS.items()
             if any(tr_fold(a) in q for a in cfg["aliases"])]
    return found or None


def strip_restaurant_names(question: str) -> str:
    """Restoran filtresi uygulanınca adı sorgudan çıkar: her chunk'ın başlığında
    geçtiği için hem BM25'i hem embedding benzerliğini gereksiz yere domine ediyor."""
    q = question
    for cfg in RESTAURANTS.values():
        for alias in sorted(cfg["aliases"], key=len, reverse=True):
            q = re.sub(re.escape(alias) + r"[\w'’]*", " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip() or question


# --------------------------------------------------------------------------- #
# Cevap üretimi (Claude)
# --------------------------------------------------------------------------- #
def build_context(docs: list[Doc]) -> str:
    parts = []
    for d in docs:
        attrs = f'id="{d.id}" restoran="{d.restaurant}" sayfa="{d.page}"'
        parts.append(f"<parca {attrs}>\n{d.text}\n</parca>")
    return "<menu_baglami>\n" + "\n".join(parts) + "\n</menu_baglami>"


def user_message(question: str, docs: list[Doc]) -> str:
    return f"{build_context(docs)}\n\nSoru: {question}"


def answer(question: str, docs: list[Doc]) -> None:
    import anthropic

    try:
        client = anthropic.Anthropic()
        with client.beta.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            output_config={"effort": CLAUDE_EFFORT},
            # Güvenlik sınıflandırıcısı nadiren yanlışlıkla reddederse, istek
            # sunucu tarafında uygun bir yedek modelle otomatik tekrar denenir.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user_message(question, docs)}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()
        print()
        if final.stop_reason == "refusal":
            print("\n[Model bu isteği yanıtlamadı.]")
        elif final.stop_reason == "max_tokens":
            print("\n[Cevap uzunluk sınırında kesildi.]")
    except anthropic.AuthenticationError:
        sys.exit("Claude API kimlik doğrulaması başarısız: ANTHROPIC_API_KEY geçersiz.")
    except anthropic.RateLimitError:
        sys.exit("Claude API hız sınırına takıldı; biraz sonra tekrar deneyin.")
    except anthropic.APIConnectionError:
        sys.exit("Claude API'ye bağlanılamadı; internet bağlantısını kontrol edin.")
    except anthropic.APIStatusError as e:
        sys.exit(f"Claude API hatası ({e.status_code}): {e.message}")
    except (anthropic.AnthropicError, TypeError) as e:
        # TypeError: SDK hiç kimlik bilgisi bulamadığında ("Could not resolve authentication method").
        if isinstance(e, TypeError) and "authentication" not in str(e):
            raise
        sys.exit("\nClaude API kimlik bilgisi bulunamadı. ANTHROPIC_API_KEY ortam değişkenini "
                 "ayarlayın ya da --no-llm ile sadece arama sonuçlarını görün.")


def run(index: MenuIndex, question: str, k: int, only: str | None, no_llm: bool, show: bool,
        prompt_only: bool = False) -> None:
    restaurants = [only] if only else detect_restaurants(question)
    query = strip_restaurant_names(question) if restaurants else question
    docs = index.search(query, k=k, restaurants=restaurants)
    overviews = [index.overview(rid) for rid in (restaurants or RESTAURANTS)]
    context = [o for o in overviews if o] + docs

    if prompt_only:
        # Claude'a gönderilecek girdinin birebir aynısı (API anahtarı olmadan test için).
        print(f"=== SİSTEM İSTEMİ ===\n{SYSTEM_PROMPT}\n\n=== KULLANICI MESAJI ===\n"
              f"{user_message(question, context)}")
        return
    if show or no_llm:
        scope = ", ".join(RESTAURANTS[r]["name"] for r in restaurants) if restaurants else "tümü"
        print(f"\n[Arama kapsamı: {scope}] Bulunan parçalar:")
        for d in docs:
            print(f"  - {d.id:16} {d.restaurant:13} s.{d.page:<3} {d.section[:60]}")
        print()
    if no_llm:
        for d in docs:
            print(f"----- {d.id}\n{d.text}\n")
        return
    answer(question, context)


def main():
    ap = argparse.ArgumentParser(description="Restoran menüleri RAG")
    ap.add_argument("question", nargs="*", help="Soru (boş bırakılırsa etkileşimli mod)")
    ap.add_argument("-k", type=int, default=TOP_K, help="Getirilecek chunk sayısı")
    ap.add_argument("-r", "--restaurant", choices=list(RESTAURANTS), help="Tek restoranla sınırla")
    ap.add_argument("--no-llm", action="store_true", help="Claude çağırmadan sadece arama sonuçlarını göster")
    ap.add_argument("--show", action="store_true", help="Cevapla birlikte bulunan chunk'ları da listele")
    ap.add_argument("--prompt", action="store_true",
                    help="Claude'u çağırmadan, ona gidecek tam girdiyi (sistem + bağlam + soru) yazdır")
    args = ap.parse_args()

    index = MenuIndex()
    if args.question:
        run(index, " ".join(args.question), args.k, args.restaurant, args.no_llm, args.show, args.prompt)
        return
    print("Restoran menü asistanı — çıkmak için boş satır veya Ctrl+C.")
    while True:
        try:
            q = input("\nSoru> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        run(index, q, args.k, args.restaurant, args.no_llm, args.show, args.prompt)


if __name__ == "__main__":
    main()
