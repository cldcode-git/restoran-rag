"""Menü PDF'lerini düzen (layout) farkındalıklı chunk'lara böler.

Menü PDF'leri çok sütunlu; metin blokları karışık sırada geliyor ve yemek adı
ile açıklaması ayrı bloklarda olabiliyor. Bu yüzden metni olduğu gibi bölmek
yerine sayfa düzenini analiz ediyoruz:

  1. Satırları konum + font bilgisiyle al; ayrı span'lardaki "₺" ile fiyatı birleştir.
  2. Her satıra rol ver: başlık (büyük punto), yemek adı (kalın font),
     gövde (açıklama/fiyat/hacim) ya da ayraç (____).
  3. Recursive XY-cut ile sayfayı en geniş boşluklardan özyinelemeli bölüp
     doğal okuma sırasını çıkar (bölgeden bölgeye değişen sütun düzenlerini çözer).
  4. Okuma sırasındaki satırları öğelere (yemek adı + açıklama + fiyat) ayır,
     başlık yığınıyla her öğeye "Kategori > Alt kategori" bölümü ata.
  5. Aynı bölümdeki öğeleri boyut sınırına kadar chunk'larda topla.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import pymupdf

SEP_RE = re.compile(r"^_{4,}$")
PRICE_RE = re.compile(r"^(₺\s*)?[\d.,]+(\s*₺)?$")
VOLUME_RE = re.compile(r"^\(?[\d.,]+\s*(cl|ml|gr|g|lt|l|kg)\b.*$", re.I)
SPACED_WORD = re.compile(r"^(?:\S ){2,}\S$")  # "K A H V A L T I"
EMPHASIS = ("Bold", "Medium", "Semibold", "SemiBold", "Black", "Heavy")


def clean_text(text: str) -> str:
    text = text.replace("¨", "₺").replace(" ", " ")
    text = re.sub(r"_{4,}", " ", text)
    parts = [p.replace(" ", "") if SPACED_WORD.match(p) else p
             for p in re.split(r"\s{2,}", text.strip())]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _letters(text: str) -> str:
    return re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşüÂâ]", "", text)


def _is_minor(text: str) -> bool:
    """Tek başına anlam taşımayan kısa fiyat/hacim satırı mı? ("₺ 255", "20 cl")"""
    return len(text) <= 16 and bool(PRICE_RE.match(text) or VOLUME_RE.match(text))


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str
    kind: str = "body"  # header | name | body | sep
    spaced: bool = False  # harf aralıklı yazılmış başlık parçası ("B U R G E R")

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


# --------------------------------------------------------------------------- #
# 1-2. Satır çıkarma ve sınıflandırma
# --------------------------------------------------------------------------- #
def _is_lira(span: dict) -> bool:
    return span["text"].strip() in ("¨", "₺")  # ¨: menü fontunda ₺ glifi


def _segments(ln: dict) -> list[list[dict]]:
    """Bir PDF satırını span gruplarına böl: büyük yatay boşluklarda ve her ₺ sembolünde.

    ₺ sembolleri ayrı "fiyat yuvası" olur; rakamlar çoğu zaman ayrı span/satır olarak
    yuvanın üstüne çizildiği için sonradan konuma göre eşleştirilir.
    """
    segs: list[list[dict]] = []
    for s in ln["spans"]:
        if not s["text"].strip():
            if segs and not _is_lira(segs[-1][-1]):  # "Y"+" "+"U R" -> "Y U R"
                segs[-1].append(s)
            continue
        if _is_lira(s):
            segs.append([s])
        elif segs and not _is_lira(segs[-1][0]) and s["bbox"][0] - segs[-1][-1]["bbox"][2] <= 15:
            segs[-1].append(s)
        else:
            segs.append([s])
    return segs


def extract_lines(page: pymupdf.Page) -> list[Line]:
    lines: list[Line] = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            for seg in _segments(ln):
                raw = "".join(s["text"] for s in seg)
                x0, x1 = seg[0]["bbox"][0], seg[-1]["bbox"][2]
                y0 = min(s["bbox"][1] for s in seg)
                y1 = max(s["bbox"][3] for s in seg)
                main = max(seg, key=lambda s: len(s["text"].strip()))
                if SEP_RE.match(raw.strip()):
                    lines.append(Line("", x0, y0, x1, y1, main["size"], main["font"], "sep"))
                    continue
                # "K A H V A L T I      Ç O R B A": tek satırda yan yana iki başlık.
                pieces = [(m.start(), m.group()) for m in re.finditer(r"\S+(?: \S+)*", raw)]
                if not (len(pieces) > 1 and all(SPACED_WORD.match(p) or len(p) == 1 for _, p in pieces)):
                    pieces = [(0, raw)]
                for idx, piece in pieces:
                    text = clean_text(piece)
                    if text:
                        px0 = x0 + (x1 - x0) * idx / max(len(raw), 1)
                        lines.append(Line(text, px0, y0, x1, y1, main["size"], main["font"],
                                          spaced=bool(SPACED_WORD.match(piece.strip()))))
    return _merge_spaced(_merge_prices(lines))


def _merge_spaced(lines: list[Line]) -> list[Line]:
    """Harf aralıklı başlığın ayrı span'lara düşmüş parçalarını birleştir ("YUMU"+"RTALI")."""
    out: list[Line] = []
    for l in sorted(lines, key=lambda l: (round(l.yc), l.x0)):
        prev = out[-1] if out else None
        if (prev is not None and prev.spaced and l.spaced and abs(prev.yc - l.yc) < 2
                and 0 <= l.x0 - prev.x1 < 40):
            prev.text += l.text
            prev.x1 = l.x1
        else:
            out.append(l)
    return out


def _merge_prices(lines: list[Line]) -> list[Line]:
    """Fiyat rakamlarını en yakın ₺ yuvasına yerleştir ("₺" + "570" -> "₺ 570").

    Bir satırda birden çok yuva olabilir (ör. kadeh / sürahi fiyatı); her rakam,
    aynı satır bandında solundaki en yakın ₺ ile eşleşir.
    """
    slots = [l for l in lines if l.text == "₺"]
    filled: set[int] = set()
    drop: set[int] = set()
    for j, n in enumerate(lines):
        if not re.fullmatch(r"[\d.,]+", n.text):
            continue
        best, best_d = None, None
        for k, l in enumerate(slots):
            d = n.x0 - l.x0
            if k in filled or abs(n.yc - l.yc) >= 0.6 * l.h or not -6 <= d <= 45:
                continue
            if best_d is None or d < best_d:
                best, best_d = k, d
        if best is not None:
            slots[best].text = f"₺ {n.text}"
            slots[best].x1 = max(slots[best].x1, n.x1)
            filled.add(best)
            drop.add(j)
    # Fiyatı silinmiş (boş bırakılmış) ₺ yuvalarını at.
    return [l for k, l in enumerate(lines) if k not in drop and l.text != "₺"]


def doc_sizes(all_lines: list[Line]) -> tuple[float, float]:
    """Belge düzeyinde gövde ve yemek-adı punto boyutları."""
    content = [l for l in all_lines if l.kind != "sep"]
    body = statistics.median(l.size for l in content)
    emph = [round(l.size, 1) for l in content
            if any(e in l.font for e in EMPHASIS) and len(_letters(l.text)) >= 3
            and l.size <= body * 1.5]  # dev başlıkları yemek adı sayma
    name = statistics.mode(emph) if emph else body
    return body, max(name, body)


def classify(lines: list[Line], body_size: float, name_size: float) -> None:
    for l in lines:
        if l.kind == "sep":
            continue
        letters = _letters(l.text)
        if _is_minor(l.text) or len(letters) < 2:
            l.kind = "body"
        elif l.size >= name_size * 1.35 and len(letters) >= 3:
            l.kind = "header"
        elif any(e in l.font for e in EMPHASIS) or l.size > body_size * 1.1:
            l.kind = "name"
        else:
            l.kind = "body"


# --------------------------------------------------------------------------- #
# 3. Recursive XY-cut ile okuma sırası
# --------------------------------------------------------------------------- #
def _gaps(intervals: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Aralıkların izdüşümündeki boşlukları (başlangıç, bitiş) olarak döndür."""
    iv = sorted(intervals)
    gaps, reach = [], iv[0][1]
    for a, b in iv[1:]:
        if a - reach >= min_gap:
            gaps.append((reach, a))
        reach = max(reach, b)
    return gaps


def _substantive(lines: list[Line]) -> bool:
    return any(l.kind in ("name", "header") or (l.kind == "body" and not _is_minor(l.text))
               for l in lines)


def xy_cut(lines: list[Line]) -> list[Line]:
    content = [l for l in lines]
    if len(content) <= 1:
        return content
    # Önce sütun (dikey boşluk) bölmesi: iki taraf da anlamlı içerik taşımalı;
    # yalnızca "₺ 255" / "20 cl" içeren bir fiyat sütununu ayırma.
    x_gaps = sorted(_gaps([(l.x0, l.x1) for l in content], 5),
                    key=lambda g: g[1] - g[0], reverse=True)
    for g0, g1 in x_gaps:
        cut = (g0 + g1) / 2
        left = [l for l in content if l.x1 <= cut]
        right = [l for l in content if l.x0 >= cut]
        if _substantive(left) and _substantive(right):
            return xy_cut(left) + xy_cut(right)
    # Sonra satır bandı (yatay boşluk) bölmesi: en geniş boşluktan.
    y_gaps = _gaps([(l.y0, l.y1) for l in content], 0.5)
    if y_gaps:
        g0, g1 = max(y_gaps, key=lambda g: g[1] - g[0])
        cut = (g0 + g1) / 2
        top = [l for l in content if l.y1 <= cut]
        bottom = [l for l in content if l.y0 >= cut]
        return xy_cut(top) + xy_cut(bottom)
    # Bölünemeyen yaprak: aynı satırdakileri soldan sağa sırala.
    return sorted(content, key=lambda l: (round(l.yc / 3), l.x0))


# --------------------------------------------------------------------------- #
# 4-5. Öğelere ayırma, bölüm atama ve chunk'lama
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    section: str
    rows: list[str] = field(default_factory=list)


def _rows(ordered: list[Line]) -> list[list[Line]]:
    """Okuma sırasındaki satırları görsel satırlara (aynı y bandı) grupla."""
    rows: list[list[Line]] = []
    for l in ordered:
        last = rows[-1][-1] if rows else None
        if (last is not None and l.kind != "sep" and last.kind != "sep"
                and abs(l.yc - last.yc) < 0.5 * min(l.h, last.h)
                and ((l.x0 >= last.x1 - 2 and l.x0 - last.x1 < 260) or (l.x1 <= last.x0 + 2 and last.x0 - l.x1 < 260))):
            rows[-1].append(l)
        else:
            rows.append([l])
    return rows


@dataclass
class Header:
    lines: list[Line]
    parent: "Header | None" = None

    @property
    def size(self): return max(l.size for l in self.lines)
    @property
    def y0(self): return min(l.y0 for l in self.lines)
    @property
    def x0(self): return min(l.x0 for l in self.lines)
    @property
    def x1(self): return max(l.x1 for l in self.lines)
    @property
    def text(self): return re.sub(r"\s*&\s*", " & ", " ".join(l.text for l in self.lines))

    def path(self) -> str:
        chain, h = [], self
        while h is not None:
            chain.append(h.text)
            h = h.parent
        return " > ".join(reversed(chain))


def _find_parent(h: Header, seen: list[Header]) -> Header | None:
    """Üst başlık: daha büyük puntolu, yukarıda ve yatayda örtüşen en yakın başlık."""
    best, best_d = None, None
    for p in seen:
        if p.size <= h.size + 0.5 or p.y0 >= h.y0:
            continue
        if min(p.x1, h.x1) - max(p.x0, h.x0) <= -20:
            continue
        d = h.y0 - p.y0
        if best_d is None or d < best_d:
            best, best_d = p, d
    return best


def page_items(ordered: list[Line]) -> list[Item]:
    items: list[Item] = []
    seen: list[Header] = []
    current: Header | None = None
    cur: Item | None = None
    prev_kind = None

    for row in _rows(ordered):
        kinds = {l.kind for l in row}
        text = " ".join(l.text for l in sorted(row, key=lambda l: l.x0) if l.text)
        if kinds == {"sep"}:
            cur, prev_kind = None, "sep"
            continue
        if kinds == {"header"}:
            size = max(l.size for l in row)
            if prev_kind == "header" and current is not None and abs(current.size - size) <= 0.5:
                current.lines.extend(row)  # aynı punto: çok satırlı başlık
            else:
                current = Header(list(row))
                current.parent = _find_parent(current, seen)
                seen.append(current)
            cur, prev_kind = None, "header"
            continue
        kind = "name" if "name" in kinds else "body"
        if cur is None or (kind == "name" and prev_kind == "body"):
            cur = Item(current.path() if current else "")
            items.append(cur)
        cur.rows.append(re.sub(r"₺\s*₺", "₺", text))
        prev_kind = kind
    if not items and seen:  # sadece başlık içeren sayfa
        items.append(Item("Sayfa başlıkları / notlar", [h.text for h in seen]))
    return items


@dataclass
class Chunk:
    restaurant_id: str
    restaurant: str
    page: int
    lang: str
    section: str
    lines: list[str] = field(default_factory=list)

    def body(self) -> str:
        return "\n".join(self.lines)

    def text(self) -> str:
        head = f"Restoran: {self.restaurant}"
        if self.section:
            head += f"\nBölüm: {self.section}"
        return f"{head}\n{self.body()}"


def chunk_pdf(restaurant_id: str, cfg: dict, max_chars: int) -> list[Chunk]:
    doc = pymupdf.open(cfg["pdf"])
    pages = [(page.number + 1, extract_lines(page)) for page in doc]
    body_size, name_size = doc_sizes([l for _, ls in pages for l in ls])
    chunks: list[Chunk] = []
    for pno, lines in pages:
        if not lines:
            continue
        lang = "en" if pno in cfg["en_pages"] else "tr"
        classify(lines, body_size, name_size)
        current: Chunk | None = None
        for item in page_items(xy_cut(lines)):
            item_len = len("\n".join(item.rows))
            if (current is None or current.section != item.section
                    or len(current.body()) + item_len > max_chars):
                current = Chunk(restaurant_id, cfg["name"], pno, lang, item.section)
                chunks.append(current)
            for row in item.rows:
                if current.lines and len(current.body()) + len(row) > max_chars:
                    # Tek öğe sınırı aşıyorsa: son satırı bağlam için taşıyarak böl.
                    current = Chunk(restaurant_id, cfg["name"], pno, lang, item.section,
                                    current.lines[-1:])
                    chunks.append(current)
                current.lines.append(row)
    # Sadece numara/etiket kalıntısı olan chunk'ları at (ör. pizza çarkı numaraları).
    return [c for c in chunks if len(_letters(c.body())) >= 6]
