# Restoran Menüsü RAG

Üç restoranın menü PDF'leri üzerinden yemek ve restoran sorularını yanıtlayan bir RAG sistemi.

| ID | Restoran | Kaynak | Not |
|---|---|---|---|
| `bigchefs` | Big Chefs | `~/Downloads/r1.pdf` (2 sayfa: TR + EN) | ₺ fiyat ve kcal var |
| `midpoint` | Midpoint | `~/Downloads/r2.pdf` (2 sayfa) | fiyatsız menü |
| `happymoons` | Happy Moon's | `~/Downloads/r3.pdf` (28 sayfa) | fiyat yok, alerjen sayfası var |

## Mimari

```
PDF ──► chunker.py ──► build_index.py ──► data/index/<restoran>.json ──► rag.py ──► Claude
        (düzen analizi)   (multilingual-e5)    (chunk + embedding)        (hibrit arama)
```

- **Chunk'lama (`chunker.py`)**: Menüler çok sütunlu olduğu için düz metin çıkarma sırayı bozuyor. Satırlar font/konum bilgisiyle alınır, **recursive XY-cut** ile okuma sırası çıkarılır, font büyüklüğüne göre kategori başlıkları ve yemek adları ayrılır. Her chunk bir bölümden (ör. `ALKOLLÜ İÇECEKLER > Kırmızı Şaraplar`) ardışık yemekler içerir (en fazla ~800 karakter) ve başında restoran adı ile bölüm yazar. Big Chefs'teki ayrı span'lara çizilmiş `₺` sembolü ve fiyat rakamları konuma göre birleştirilir (kadeh/sürahi gibi çift fiyatlar dahil).
- **Embedding**: Yerel, Türkçe destekli `intfloat/multilingual-e5-base` (768 boyut, normalize). API anahtarı gerektirmez; ilk çalıştırmada ~1 GB model indirilir.
- **JSON indeks**: Her restoran için `data/index/<id>.json` — chunk metni, sayfa, dil, bölüm ve embedding birlikte. Her dosyanın ilk chunk'ı restoran genel bilgisidir (bölümler, fiyat/kalori bilgisi var mı).
- **Arama (`rag.py`)**: Embedding benzerliği + BM25 (Türkçe için aksan katlama, stopword ve F5 kök bulma) **Reciprocal Rank Fusion** ile birleşir. Soruda restoran adı geçerse arama o restoranla sınırlanır.
- **Cevap**: `claude-opus-5-5`, yalnızca getirilen menü parçalarına dayanarak, akış (streaming) halinde cevap verir. Güvenlik sınıflandırıcısının nadir yanlış reddi için sunucu tarafı `fallbacks: "default"` açık.

## Kurulum

```bash
cd ~/Github/restaurant-rag
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

İndeksi (yeniden) üretmek — PDF'ler değişirse ya da embedding modeli değiştirilirse:

```bash
.venv/bin/python build_index.py
```

## Kullanım

Cevap üretimi için Claude API anahtarı gerekir:

```bash
export ANTHROPIC_API_KEY=...
```

```bash
.venv/bin/python rag.py "Big Chefs'te hangi salatalar var, kalorileri ne?"
```

```bash
.venv/bin/python rag.py
```

Seçenekler: `-r midpoint` (tek restoran), `-k 15` (daha fazla chunk), `--show` (bulunan chunk'ları listele), `--no-llm` (Claude'u çağırmadan sadece arama sonuçlarını göster; API anahtarı gerekmez), `--prompt` (Claude'a gidecek tam girdiyi — sistem istemi + bulunan parçalar + soru — yazdır; anahtar olmadan test için).

Listede olmayan bir restoran sorulursa (ör. "Starbucks'ta latte var mı?") model o restoran hakkında bilgisi olmadığını söyler; bilinen restoranların menü parçalarını ona aitmiş gibi kullanmaz.

## Değerlendirme ve hata ayıklama

```bash
.venv/bin/python eval_retrieval.py
```

24 soruluk küçük sette geri getirme isabeti (hit@6): dense 23/24, BM25 23/24, **hibrit 24/24**.

```bash
.venv/bin/python inspect_chunks.py
```

Chunk'ları `data/debug/` altına okunabilir metin olarak yazar.

## Bilinen sınırlamalar

- Menülerdeki **alerjen ikonları** (gluten, süt vb. simgeler) görsel olduğu için metne çıkmıyor; alerjen soruları içerik listesinden çıkarımla yanıtlanır.
- Harf aralıklı başlıklarda kelime boşlukları PDF'te olmadığından bazı bölüm adları bitişik ("ARTİZANPİZZA").
- Happy Moon's'un görsel ağırlıklı birkaç sayfasında (ör. pizza çarkı) bölüm başlığı yok ya da yemek adı ile açıklaması ayrı chunk'lara düşebiliyor.
- Restoran adları/takma adları `config.py` içinde; yeni restoran eklemek için oraya girdi ekleyip `build_index.py` çalıştırmak yeterli.
