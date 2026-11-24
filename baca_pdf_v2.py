import math
from pathlib import Path

import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:
    import fitz


PDF_MASUK = Path(r"C:/Users/fakhr/Downloads/TesKotakPDF.pdf")
MODEL = Path(r"C:\Pendahuluan TA\Tugas 3 - Classify Square and Text\hasil_training\train_utama\weights\best.pt")
PAKAI_YOLO = True      # False = hanya mengikuti garis (tanpa YOLO)
CONF = 0.25            # batas keyakinan YOLO (naikkan kalau terlalu banyak salah deteksi)
IMGSZ_YOLO = 1024      # ukuran gambar untuk YOLO (halaman PDF besar, jadi dibuat lebih besar dari 640)
DPI = 200              # ketajaman halaman saat dibaca (naikkan kalau tulisan kecil sulit terbaca)
MAKS_SISI_PX = 4500    # halaman yang terlalu besar otomatis diturunkan DPI-nya
HALAMAN = None         # None = semua halaman; atau daftar nomor halaman, mis. [1, 2]
SUMBER_TEKS = "auto"   # "auto" = pakai teks asli dalam PDF bila ada, kalau tidak pakai OCR
                       # "pdf" = hanya teks asli PDF, "ocr" = selalu OCR (mis. PDF hasil scan)
CARI_TAMBAHAN = True   # cari kotak yang terlewat YOLO (mis. garis putus-putus) dari garisnya saja

# Pengukuran accuracy + confusion matrix
# Contoh:  ANOTASI = {1: {"kotak": 9, "bukan": 9, "salah": []}, 2: {"kotak": 15, "bukan": 13, "salah": [8, 16]}}
ANOTASI = {}

WARNA_KOTAK = (0.0, 0.0, 0.5)
PALET_TEKS = [
    (1.0, 0.5, 0.0), (0.85, 0.0, 0.85), (0.0, 0.35, 1.0), (0.9, 0.1, 0.1),
    (0.5, 0.2, 0.8), (0.0, 0.6, 0.6), (0.6, 0.35, 0.1),
]

PALET_KOTAK = [(0.90, 0.10, 0.29), (0.24, 0.71, 0.29), (0.26, 0.39, 0.85), (0.96, 0.51, 0.19),
               (0.57, 0.12, 0.71), (0.00, 0.50, 0.50), (0.94, 0.20, 0.90), (0.60, 0.39, 0.14),
               (0.50, 0.00, 0.00), (0.50, 0.50, 0.00), (0.00, 0.00, 0.46), (0.83, 0.63, 0.00),
               (0.00, 0.59, 0.65), (0.49, 0.70, 0.26), (0.93, 0.25, 0.48), (0.42, 0.35, 0.80)]


def warna_ke(n):
    """Warna untuk kotak ke-n. Setelah palet habis, dibuat warna baru yang tetap berbeda-beda."""
    if n < len(PALET_KOTAK):
        return PALET_KOTAK[n]
    import colorsys
    return colorsys.hsv_to_rgb((n * 0.61803398875) % 1.0, 0.75, 0.72)

# OCR (Dipakai kalau PDF tidak punya teks asli) ---
OCR_BAHASA = ["id", "en"]
OCR_YAKIN_MIN = 0.2

# Mengikuti alur garis
SISI_KERJA = 2000      # halaman diperkecil ke ukuran ini dulu untuk melacak garis
UKURAN_JENDELA = 0.08  # jendela grey_closing/opening = 8% sisi gambar
BUANG_BINTIK_PX = 4    # potongan goresan lebih kecil dari ini (piksel) dianggap noise
JARAK_GABUNG = (0.003, 0.006, 0.012, 0.022, 0.035)   # dicoba dari kecil ke besar: kecil = kotak berdekatan tidak menyatu,
                                                     # besar = celah garis putus-putus yang lebar ikut tersambung
BUKA_FRAC = 0.02       # garis tipis yang menempel di kotak (panah, penghubung) dipotong
MIN_DUKUNGAN = 0.30    # tiap sisi kotak minimal sekian bagian harus benar-benar ada garisnya
MIN_BAGIAN_TEPI = 0.25 # minimal sekian bagian goresan di dalam kotak harus berada di tepinya (paragraf ditolak)
POLARITAS = "auto"     # "auto" = tinta gelap di kertas terang (atau sebaliknya); bisa dipaksa "gelap" / "terang"
CEK_LATAR = True       # kotak harus berlatar seperti kertas
TOLERANSI_WARNA = 0.05
TERANG_MIN = 0.4
TERANG_MAX = 1.8
SEBARAN_LATAR = 0.25
BATAS_SUDUT_BULAT = 0.12   # sudut kotak boleh sedikit bulat

# 1. Ambil goresan
def otsu(arr_uint8):
    batas, _ = cv2.threshold(arr_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return batas

def perkecil(img):
    h, w = img.shape[:2]
    s = min(1.0, SISI_KERJA / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img, s

def ambil_goresan(kecil):
    """Mask goresan (True = tinta). Tahan bayangan: grey_closing/opening seperti model lama."""
    h, w = kecil.shape[:2]
    sisi = min(h, w)
    halus = cv2.GaussianBlur(kecil, (0, 0), 0.7)

    # 1. Sinyal goresan
    ukuran = max(9, int(sisi * UKURAN_JENDELA)) | 1
    kern = np.ones((ukuran, ukuran), np.uint8)
    gelap = np.zeros((h, w), np.float32)
    terang = np.zeros((h, w), np.float32)
    for c in range(3):
        k = halus[:, :, c]
        kf = k.astype(np.float32)
        gelap += (cv2.morphologyEx(k, cv2.MORPH_CLOSE, kern).astype(np.float32) - kf) ** 2
        terang += (kf - cv2.morphologyEx(k, cv2.MORPH_OPEN, kern).astype(np.float32)) ** 2
    gelap, terang = np.sqrt(gelap), np.sqrt(terang)
    if POLARITAS == "auto":
        tinta_gelap = float(np.median(cv2.cvtColor(kecil, cv2.COLOR_BGR2GRAY))) >= 110
    else:
        tinta_gelap = POLARITAS == "gelap"
    sinyal = gelap if tinta_gelap else terang

    mask = np.zeros((h, w), bool)
    if sinyal.max() >= 15:
        puncak = max(float(np.percentile(sinyal, 99.9)), 0.5 * float(sinyal.max()))
        arr = np.clip(sinyal / puncak * 255, 0, 255).astype(np.uint8)
        batas = max(otsu(arr) * 0.5, 30)           # sama dengan model lama
        mask = arr > batas

    # 2. Tepi
    gray = cv2.GaussianBlur(cv2.cvtColor(kecil, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    tepi = cv2.Canny(gray, 30, 90) > 0

    ink = (mask | tepi).astype(np.uint8)

    # 3. Buang bintik (noise)
    n, lab, stat, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    kecil_ = np.where(stat[:, cv2.CC_STAT_AREA] < BUANG_BINTIK_PX)[0]
    kecil_ = kecil_[kecil_ != 0]
    if len(kecil_):
        ink[np.isin(lab, kecil_)] = 0
    return ink


#2. Gabung goresan berdekatan, cari bentuk kotak
def elips(d):
    d = max(3, int(d) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

def kontur_pada_jarak(ink, r, sisi):
    """Goresan yang jaraknya <= r dianggap satu (dilate), lubangnya diisi, lalu dikembalikan ke ukuran asli
    (erode) dan garis tipis penghubung dipotong (open). Sama idenya dengan distance_transform_edt <= r."""
    gabung = cv2.dilate(ink, elips(2 * r + 1))
    cnts, _ = cv2.findContours(gabung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(ink)
    cv2.drawContours(solid, cnts, -1, 255, -1)
    solid = cv2.erode(solid, elips(2 * r + 1))
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, elips(max(5, BUKA_FRAC * sisi)))
    cnts2, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts2

def sudut_siku(poly, toleransi=15):
    p = poly.reshape(4, 2).astype(np.float32)
    for i in range(4):
        a, b, c = p[i - 1], p[i], p[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if abs(np.degrees(np.arccos(np.clip(cosang, -1, 1))) - 90) > toleransi:
            return False
    return True

def poligon_kotak(cnt, min_isi=0.88):
    """4 titik sudut kalau kontur berbentuk kotak (boleh miring), kalau bukan None."""
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(hull)
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    area_rect = cv2.contourArea(box)
    if area <= 0 or area_rect <= 0 or area / area_rect < min_isi:
        return None
    approx = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
    if len(approx) == 4:
        if not sudut_siku(approx):
            return None
        if sudut_siku(approx, 6):          # hampir persegi panjang sempurna: pakai persegi panjang yang rapi
            return box.astype(np.float32)
        return approx.reshape(4, 2).astype(np.float32)
    return box.astype(np.float32)

def periksa_tepi(poly, ink, dist_ink, tol):
    """Ikuti garisnya: tiap sisi poligon harus benar-benar ditempeli goresan, dan sebagian besar goresan
    di dalam kotak harus berada di tepi, bukan tersebar."""
    H, W = ink.shape
    for i in range(4):
        p, q = poly[i], poly[(i + 1) % 4]
        n = max(3, int(np.linalg.norm(q - p) / 2))
        t = np.linspace(0, 1, n)
        xs = np.clip((p[0] + (q[0] - p[0]) * t).round().astype(int), 0, W - 1)
        ys = np.clip((p[1] + (q[1] - p[1]) * t).round().astype(int), 0, H - 1)
        if (dist_ink[ys, xs] <= tol).mean() < MIN_DUKUNGAN:
            return False

    sisi_pendek = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4))
    batas_sudut = max(3 * tol, BATAS_SUDUT_BULAT * sisi_pendek)
    ada = 0
    for px_, py_ in poly:
        cx_, cy_ = int(np.clip(round(px_), 0, W - 1)), int(np.clip(round(py_), 0, H - 1))
        if dist_ink[cy_, cx_] <= batas_sudut:
            ada += 1
    if ada < 3:
        return False

    x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
    pad = int(tol) + 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)
    loc = (poly - [x0, y0]).astype(np.int32)
    sub = ink[y0:y1, x0:x1] > 0
    dalam = np.zeros(sub.shape, np.uint8)
    cv2.fillPoly(dalam, [loc], 255)
    dalam = cv2.dilate(dalam, elips(2 * tol + 1))
    pita = np.zeros(sub.shape, np.uint8)
    cv2.polylines(pita, [loc], True, 255, int(2 * tol + 1))
    total = np.count_nonzero(sub & (dalam > 0))
    if total == 0:
        return False
    return np.count_nonzero(sub & (pita > 0)) / total >= MIN_BAGIAN_TEPI


def latar_mirip_kertas(g, poly):
    """Warna di sekeliling kotak harus seperti kertas dan seragam."""
    kecil, ink, kertas, sisi = g.kecil, g.ink, g.kertas, g.sisi
    H, W = ink.shape
    ring = max(8, int(sisi * 0.025))
    x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
    p = ring + 4
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + bw + p), min(H, y + bh + p)
    loc = (poly - [x0, y0]).astype(np.int32)
    luar = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(luar, [loc], 255)
    jarak = cv2.distanceTransform(255 - luar, cv2.DIST_L2, 3)
    tinta = cv2.dilate(ink[y0:y1, x0:x1], elips(5)) > 0
    cincin = (jarak > 3) & (jarak <= ring) & ~tinta
    if cincin.sum() < 30:
        return True
    piksel = kecil[y0:y1, x0:x1][cincin].astype(np.float32)
    warna = np.median(piksel, axis=0)
    kroma_o = warna / max(float(warna.sum()), 1.0)
    kroma_k = kertas / max(float(kertas.sum()), 1.0)
    if float(np.abs(kroma_o - kroma_k).max()) > TOLERANSI_WARNA:
        return False
    rasio = (float(warna.mean()) + 1) / (float(kertas.mean()) + 1)
    if not (TERANG_MIN <= rasio <= TERANG_MAX):
        return False
    rata = g.rata[y0:y1, x0:x1][cincin]
    p10, p90 = np.percentile(rata, 10), np.percentile(rata, 90)
    return float((p90 - p10) / (rata.mean() + 1)) <= SEBARAN_LATAR


def rect_dari_poligon(poly):
    x, y, bw, bh = cv2.boundingRect(poly.astype(np.float32))
    return (x, y, x + bw, y + bh)


def iou_rect(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tepi_area_berwarna(kecil):
    sisi = min(kecil.shape[:2])
    halus = cv2.GaussianBlur(kecil, (0, 0), 1.5).astype(np.int16)
    kertas = np.median(kecil.reshape(-1, 3), axis=0).astype(np.int16)
    beda = np.abs(halus - kertas).max(axis=2)
    area = (beda > 20).astype(np.uint8)
    area = cv2.morphologyEx(area, cv2.MORPH_OPEN, elips(max(9, 0.03 * sisi)))
    tepi = area - cv2.erode(area, np.ones((3, 3), np.uint8))
    return cv2.dilate(tepi, np.ones((3, 3), np.uint8))


class Goresan:
    """Hasil ambil_goresan untuk satu gambar"""
    def __init__(self, img, teks_xyxy=()):
        self.kecil, self.skala = perkecil(img)
        self.sisi = min(self.kecil.shape[:2])
        self.ink = ambil_goresan(self.kecil) | tepi_area_berwarna(self.kecil)
        self.tol = max(3, int(0.004 * self.sisi))
        self.dist = cv2.distanceTransform(1 - self.ink, cv2.DIST_L2, 3)
        # versi goresan tanpa area tulisan: dipakai untuk mencari kotak baru, supaya paragraf teks tidak dikira kotak
        self.ink2 = self.ink.copy()
        for x1, y1, x2, y2 in teks_xyxy:
            m = 0.12 * min(x2 - x1, y2 - y1)
            a, b = int(max(0, (y1 + m) * self.skala)), int(max(0, (x1 + m) * self.skala))
            self.ink2[a:int((y2 - m) * self.skala) + 1, b:int((x2 - m) * self.skala) + 1] = 0
        self.dist2 = cv2.distanceTransform(1 - self.ink2, cv2.DIST_L2, 3)
        self.kertas = np.median(self.kecil.reshape(-1, 3), axis=0).astype(np.float32)
        gray = cv2.cvtColor(self.kecil, cv2.COLOR_BGR2GRAY)
        k_bg = max(15, (self.sisi // 25) | 1)
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, elips(k_bg))
        self.rata = cv2.divide(gray, bg, scale=255).astype(np.float32)   # gambar tanpa bayangan

    def radius(self):
        return [max(2, int(f * self.sisi)) for f in JARAK_GABUNG]


def cari_kotak_global(g):
    """Cari semua kotak hanya dengan mengikuti garisnya"""
    H, W = g.ink.shape
    hasil = []
    for r in g.radius():
        for c in kontur_pada_jarak(g.ink2 * 255, r, g.sisi):
            luas = cv2.contourArea(c)
            if luas < 0.002 * H * W or luas > 0.8 * H * W:
                continue
            poly = poligon_kotak(c)
            if poly is None or not periksa_tepi(poly, g.ink2, g.dist2, g.tol):
                continue
            if CEK_LATAR and not latar_mirip_kertas(g, poly):
                continue
            rc = rect_dari_poligon(poly)
            if all(iou_rect(rc, rect_dari_poligon(p)) < 0.3 for p in hasil):
                hasil.append(poly)
    return hasil


def perhalus_dalam_kotak(g, xyxy_kecil, pad_frac=0.08):
    """Cari garis kotak yang pas di sekitar kotak dari YOLO"""
    H, W = g.ink.shape
    x1, y1, x2, y2 = xyxy_kecil
    pad = int(pad_frac * max(x2 - x1, y2 - y1))
    cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    cx2, cy2 = min(W, int(x2) + pad), min(H, int(y2) + pad)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    ink = g.ink[cy1:cy2, cx1:cx2]
    dist = g.dist[cy1:cy2, cx1:cx2]
    ref = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
    for r in g.radius():
        terbaik, skor = None, 0.4
        for c in kontur_pada_jarak(ink * 255, r, g.sisi):
            poly = poligon_kotak(c)
            if poly is None or not periksa_tepi(poly, ink, dist, g.tol):
                continue
            iou = iou_rect(rect_dari_poligon(poly), ref)
            if iou > skor:
                terbaik, skor = poly, iou
        if terbaik is not None:
            return terbaik + [cx1, cy1]
    return None


# 3. Gabungkan YOLO + garis
def tumpang_kecil(a, b):
    """Seberapa besar irisan dua persegi panjang terhadap yang lebih kecil"""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    kecil = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / kecil if kecil > 0 else 0.0


def proses(img, box_kotak, box_teks, cari_tambahan=CARI_TAMBAHAN, teks_kecuali=None):
    """box_kotak dan box_teks: daftar (x1, y1, x2, y2) dari YOLO (koordinat gambar asli).
    teks_kecuali: area tulisan yang diabaikan saat mencari kotak baru (bawaan: sama dengan box_teks).
    Hasil: daftar kotak [{'poly', 'sumber', 'teks'}] (koordinat gambar asli).
    Aturan: garis adalah penentu. Kotak YOLO hanya dipakai kalau garisnya benar-benar berbentuk kotak."""
    g = Goresan(img, box_teks if teks_kecuali is None else teks_kecuali)
    s = g.skala
    kandidat = [p / s for p in cari_kotak_global(g)]      # semua kotak yang bentuk garisnya benar
    kandidat.sort(key=lambda p: -cv2.contourArea(p.astype(np.float32)))   # yang besar dulu
    kotak = []

    def luas(r):
        return (r[2] - r[0]) * (r[3] - r[1])

    def potongan(r_kecil, r_besar):
        """True kalau r_kecil hanya potongan dari r_besar: ada di dalamnya dan tipis (sepotong garis) atau sangat kecil
        (huruf tebal). Kotak yang sungguh berada di dalam kotak lain (tidak tipis, tidak sangat kecil) tetap dipertahankan."""
        if tumpang_kecil(r_kecil, r_besar) < 0.6 or luas(r_kecil) > luas(r_besar):
            return False
        w, h = r_kecil[2] - r_kecil[0], r_kecil[3] - r_kecil[1]
        return min(w, h) / max(w, h, 1e-6) < 0.35 or luas(r_kecil) < 0.08 * luas(r_besar)

    def tambah(poly, sumber):
        r = rect_dari_poligon(poly)
        for k in kotak:
            rk = rect_dari_poligon(k["poly"])
            if iou_rect(r, rk) >= 0.3 or potongan(r, rk):
                return
        kotak[:] = [k for k in kotak if not potongan(rect_dari_poligon(k["poly"]), r)]
        kotak.append({"poly": poly, "sumber": sumber})

    # 1) kotak dari YOLO: dirapikan mengikuti garisnya.
    for xyxy in box_kotak:
        poly = perhalus_dalam_kotak(g, [v * s for v in xyxy])
        if poly is not None:
            poly = poly / s
        else:
            cocok = [(iou_rect(rect_dari_poligon(c), tuple(xyxy)), i) for i, c in enumerate(kandidat)]
            cocok = [c for c in cocok if c[0] >= 0.3]
            if not cocok:
                continue
            poly = kandidat[max(cocok)[1]]
        tambah(poly, "YOLO+garis")

    # 2) kotak yang terlewat YOLO (mis. garis putus-putus), dicari dari garisnya saja
    if cari_tambahan:
        for poly in kandidat:
            tambah(poly, "garis")

    # 3) tentukan teks mana yang berada di dalam tiap kotak
    for k in kotak:
        k["teks"] = []
        kontur = k["poly"].reshape(-1, 1, 2).astype(np.float32)
        for t in box_teks:
            pusat = ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
            if cv2.pointPolygonTest(kontur, pusat, False) >= 0:
                k["teks"].append(t)
    return kotak


# 3B. Pembaca OCR
_pembaca = None


def pembaca_ocr():
    """EasyOCR dimuat sekali saja (pertama kali jalan akan mengunduh model bahasa, sekali saja)."""
    global _pembaca
    if _pembaca is None:
        import easyocr
        _pembaca = easyocr.Reader(OCR_BAHASA, gpu=False, verbose=False)
    return _pembaca


# 5. Render halaman PDF
def render_halaman(page):
    dpi = DPI
    w_pt, h_pt = page.rect.width, page.rect.height
    if max(w_pt, h_pt) * dpi / 72 > MAKS_SISI_PX:
        dpi = MAKS_SISI_PX * 72 / max(w_pt, h_pt)
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), dpi / 72.0


# 6. Tulisan asli dari PDF
def teks_dari_pdf(page, skala):
    """Baris tulisan asli PDF: [{'teks', 'xyxy' (piksel), 'sudut' (derajat), 'tinggi' (piksel)}]."""
    rot = page.rotation_matrix
    rot0 = fitz.Matrix(rot.a, rot.b, rot.c, rot.d, 0, 0)
    hasil = []
    for blok in page.get_text("dict")["blocks"]:
        if blok.get("type") != 0:
            continue
        for baris in blok["lines"]:
            teks = "".join(sp["text"] for sp in baris["spans"]).strip()
            if not teks:
                continue
            r = fitz.Rect(baris["bbox"]) * rot
            v = fitz.Point(*baris["dir"]) * rot0
            tinggi = max(sp["size"] for sp in baris["spans"]) * skala
            hasil.append({"teks": teks, "xyxy": (r.x0 * skala, r.y0 * skala, r.x1 * skala, r.y1 * skala),
                          "sudut": math.degrees(math.atan2(v.y, v.x)), "tinggi": tinggi})
    return hasil

def jarak_kotak(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)

def kelompokkan_baris(baris):
    """Baris yang berdekatan dan sejajar digabung jadi satu tulisan (satu kotak tulisan)."""
    n = len(baris)
    induk = list(range(n))

    def cari(i):
        while induk[i] != i:
            induk[i] = induk[induk[i]]
            i = induk[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            beda = abs((baris[i]["sudut"] - baris[j]["sudut"] + 180) % 360 - 180)
            if beda < 10 and jarak_kotak(baris[i]["xyxy"], baris[j]["xyxy"]) <= 0.9 * max(baris[i]["tinggi"], baris[j]["tinggi"]):
                induk[cari(i)] = cari(j)
    grup = {}
    for i in range(n):
        grup.setdefault(cari(i), []).append(baris[i])
    hasil = []
    for g in grup.values():
        g.sort(key=lambda b: (round(b["xyxy"][1] / max(b["tinggi"], 1)), b["xyxy"][0]))
        xyxy = (min(b["xyxy"][0] for b in g), min(b["xyxy"][1] for b in g),
                max(b["xyxy"][2] for b in g), max(b["xyxy"][3] for b in g))
        hasil.append({"teks": " ".join(b["teks"] for b in g), "region": xyxy, "sudut": g[0]["sudut"], "sumber_teks": "pdf"})
    return hasil

def kelompokkan_box(box_teks):
    """Kotak teks YOLO yang berdekatan (mis. dua baris tulisan) digabung jadi satu tulisan."""
    n = len(box_teks)
    induk = list(range(n))

    def cari(i):
        while induk[i] != i:
            induk[i] = induk[induk[i]]
            i = induk[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = box_teks[i], box_teks[j]
            tinggi = max(min(a[2] - a[0], a[3] - a[1]), min(b[2] - b[0], b[3] - b[1]))
            if jarak_kotak(a, b) <= 0.9 * tinggi:
                induk[cari(i)] = cari(j)
    grup = {}
    for i in range(n):
        grup.setdefault(cari(i), []).append(box_teks[i])
    return [{"teks": None, "sudut": None, "sumber_teks": "ocr",
             "region": (min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g))}
            for g in grup.values()]


# 7. Kotak tulisan yang mengikuti kemiringan tulisan
def sudut_kotak(poly):
    """Arah sisi terpanjang kotak (derajat, -90..90): dipakai kalau tulisannya terlalu pendek untuk ditebak miringnya."""
    p = poly.astype(np.float32)
    sisi = [(p[(i + 1) % 4] - p[i]) for i in range(4)]
    v = max(sisi, key=lambda s: np.hypot(*s))
    return normal_sudut(math.degrees(math.atan2(v[1], v[0])))

def normal_sudut(a):
    a = (a + 90) % 180 - 90
    return 0.0 if abs(a) < 0.8 else a

def batas_putar(pts, theta):
    """Batas persegi panjang yang diputar sebesar theta, mengabaikan 0,5% titik terluar (bintik noise)."""
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    u = pts[:, 0] * c + pts[:, 1] * s
    v = -pts[:, 0] * s + pts[:, 1] * c
    u0, u1 = np.percentile(u, [0.5, 99.5])
    v0, v1 = np.percentile(v, [0.5, 99.5])
    return u0, u1, v0, v1, (u1 - u0) * (v1 - v0)

def pas_kotak_teks(img, region, poly_kotak, sudut_prior):
    """Cari goresan tulisan di dalam 'region' (dan di dalam kotak), lalu bungkus dengan persegi panjang
    yang miring mengikuti tulisan. Hasil: {'poly' (4 titik), 'sudut', 'w', 'h'} atau None kalau tak ada tulisan."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = region
    pad = int(0.25 * min(x2 - x1, y2 - y1)) + 3
    cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    cx2, cy2 = min(W, int(x2) + pad), min(H, int(y2) + pad)
    if cx2 - cx1 < 6 or cy2 - cy1 < 6:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    ink = ambil_goresan(crop)

    # buang garis kotak: semua goresan yang tersambung ke garis tepi kotak bukan bagian tulisan
    (_, (kw, kh), _) = cv2.minAreaRect(poly_kotak.astype(np.float32))
    pita = np.zeros(ink.shape, np.uint8)
    cv2.polylines(pita, [(poly_kotak - [cx1, cy1]).astype(np.int32)], True, 1, max(7, int(0.06 * min(kw, kh))))
    n, lab, _, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    kena = np.unique(lab[(pita > 0) & (ink > 0)])
    ink[np.isin(lab, kena[kena != 0])] = 0

    # hanya bagian dalam kotak
    mask = np.zeros(ink.shape, np.uint8)
    cv2.fillPoly(mask, [(poly_kotak - [cx1, cy1]).astype(np.int32)], 1)
    mask = cv2.erode(mask, elips(max(5, 0.07 * min(kw, kh))))
    # dan hanya di sekitar region
    batas = np.zeros(ink.shape, np.uint8)
    m = int(0.12 * min(x2 - x1, y2 - y1)) + 2
    batas[max(0, int(y1) - cy1 - m):int(y2) - cy1 + m, max(0, int(x1) - cx1 - m):int(x2) - cx1 + m] = 1
    ink = ink * mask * batas

    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 4:
            ink[lab == i] = 0
    ys, xs = np.nonzero(ink)
    if len(xs) < 12:
        return None
    pts = np.column_stack([xs + cx1, ys + cy1]).astype(np.float32)

    if sudut_prior is not None:
        theta = normal_sudut(sudut_prior)
        u0, u1, v0, v1, _ = batas_putar(pts, theta)
    else:
        tk = sudut_kotak(poly_kotak)
        theta = normal_sudut(tk)
        tegak = normal_sudut(tk + 90)
        if batas_putar(pts, tegak)[4] < 0.85 * batas_putar(pts, theta)[4]:
            theta = tegak
        luas0 = batas_putar(pts, theta)[4]
        (_, (rw, rh), _) = rect = cv2.minAreaRect(pts)
        if max(rw, rh) / max(min(rw, rh), 1e-6) >= 1.6:
            bp_ = cv2.boxPoints(rect)
            vv = max([bp_[(i + 1) % 4] - bp_[i] for i in range(4)], key=lambda s_: np.hypot(*s_))
            bebas = normal_sudut(math.degrees(math.atan2(vv[1], vv[0])))
            if batas_putar(pts, bebas)[4] < 0.8 * luas0:
                theta = bebas
        u0, u1, v0, v1, _ = batas_putar(pts, theta)

    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    p = max(2.5, 0.14 * (v1 - v0))
    u0, u1, v0, v1 = u0 - p, u1 + p, v0 - p, v1 + p
    sudut = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    poly = np.column_stack([sudut[:, 0] * c - sudut[:, 1] * s, sudut[:, 0] * s + sudut[:, 1] * c]).astype(np.float32)
    return {"poly": poly, "sudut": theta, "w": float(u1 - u0), "h": float(v1 - v0)}


# 8. OCR (hanya dipakai kalau PDF-nya hasil scan/tidak punya teks asli)
def ocr_item(img, kt):
    """Luruskan kotak tulisan yang miring, lalu baca dengan EasyOCR."""
    W, H = max(8, int(round(kt["w"]))), max(8, int(round(kt["h"])))
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(kt["poly"].astype(np.float32), dst)
    kertas = tuple(int(x) for x in np.median(img.reshape(-1, 3), axis=0))
    crop = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=kertas)
    if min(crop.shape[:2]) < 48:
        f = 48 / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=kertas)

    putaran = [0, 90, 180, 270] if abs(kt["sudut"]) >= 60 else [0]
    terbaik, skor_terbaik = [], -1.0
    for rot in putaran:
        gambar = crop if rot == 0 else cv2.rotate(crop, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot])
        potongan = []
        for kotak4, teks, yakin in pembaca_ocr().readtext(gambar, detail=1, paragraph=False):
            if yakin >= OCR_YAKIN_MIN and teks.strip():
                ys, xs = [q[1] for q in kotak4], [q[0] for q in kotak4]
                potongan.append((sum(ys) / 4, min(xs), max(ys) - min(ys), teks.strip(), yakin))
        skor = sum(q[4] * len(q[3]) for q in potongan)
        if skor > skor_terbaik:
            terbaik, skor_terbaik = potongan, skor
    potongan = [q[:4] for q in terbaik]
    if not potongan:
        return ""
    tinggi = float(np.median([q[2] for q in potongan]))
    baris = []
    for cy, x, h, teks in sorted(potongan, key=lambda q: q[0]):
        if baris and abs(cy - baris[-1][0]) < 0.6 * tinggi:
            baris[-1][1].append((x, teks))
        else:
            baris.append([cy, [(x, teks)]])
    return " ".join(" ".join(t for _, t in sorted(b[1])) for b in baris)


# 9. Proses satu halaman
def urutkan_baca(kotak, tinggi_halaman):
    """Nomor kotak mengikuti urutan baca: baris dari atas ke bawah, dalam satu baris dari kiri ke kanan."""
    if not kotak:
        return kotak
    pusat = [(k["poly"][:, 0].mean(), k["poly"][:, 1].mean()) for k in kotak]
    tinggi = np.median([np.ptp(k["poly"][:, 1]) for k in kotak])
    urut = sorted(range(len(kotak)), key=lambda i: pusat[i][1])
    baris, terakhir = [], None
    for i in urut:
        if terakhir is not None and pusat[i][1] - terakhir < 0.5 * tinggi:
            baris[-1].append(i)
        else:
            baris.append([i])
        terakhir = pusat[i][1]
    return [kotak[i] for b in baris for i in sorted(b, key=lambda j: pusat[j][0])]


def proses_halaman(page, model, id_kotak, id_teks):
    img, skala = render_halaman(page)

    box_kotak, box_teks = [], []
    if model is not None:
        hasil = model.predict(img, conf=CONF, imgsz=IMGSZ_YOLO, classes=[id_kotak, id_teks], verbose=False)[0]
        for b in hasil.boxes:
            (box_kotak if int(b.cls) == id_kotak else box_teks).append(b.xyxy[0].tolist())

    baris_pdf = teks_dari_pdf(page, skala) if SUMBER_TEKS in ("auto", "pdf") else []
    pakai_pdf = len(baris_pdf) > 0
    kotak = proses(img, box_kotak, [] if pakai_pdf else box_teks,
                      teks_kecuali=[b["xyxy"] for b in baris_pdf] if pakai_pdf else None)
    kotak = urutkan_baca(kotak, img.shape[0])

    for n_k, k in enumerate(kotak):
        k["warna"] = warna_ke(n_k)
        poly = k["poly"]
        poly_cv = poly.reshape(-1, 1, 2).astype(np.float32)
        if pakai_pdf:
            di_dalam = [b for b in baris_pdf
                        if cv2.pointPolygonTest(poly_cv, ((b["xyxy"][0] + b["xyxy"][2]) / 2, (b["xyxy"][1] + b["xyxy"][3]) / 2), False) >= 0]
            item = kelompokkan_baris(di_dalam) if di_dalam else []
            if not item and SUMBER_TEKS == "auto":
                # halaman campuran (sebagian teks asli, sebagian gambar hasil scan): kotak yang tidak punya teks asli dibaca dengan OCR
                x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
                item = [{"teks": None, "region": (x, y, x + bw, y + bh), "sudut": None, "sumber_teks": "ocr"}]
        else:
            item = kelompokkan_box(k["teks"])
            if not item:   # YOLO tidak menemukan kotak tulisan: cari tulisannya langsung di dalam kotak
                x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
                item = [{"teks": None, "region": (x, y, x + bw, y + bh), "sudut": None, "sumber_teks": "ocr"}]

        k["item"] = []
        for it in item:
            kt = pas_kotak_teks(img, it["region"], poly, it["sudut"])
            if kt is None:
                continue
            if it["teks"] is None:
                it["teks"] = ocr_item(img, kt)
            if not it["teks"]:
                continue
            it["kotak"] = kt
            it["warna"] = k["warna"]
            k["item"].append(it)
    return kotak, skala


# 10. Tulis hasil ke PDF
def ke_pdf(page, titik_px, skala):
    """Titik piksel gambar render -> koordinat halaman PDF (mengikuti rotasi halaman)."""
    return [fitz.Point(x / skala, y / skala) * page.derotation_matrix for x, y in titik_px]


def latin(teks):
    return teks.encode("latin-1", "replace").decode("latin-1")


def gambar_pdf(page, kotak, skala, nomor_halaman):
    sisi = max(page.rect.width, page.rect.height)
    fs = max(7.0, 0.011 * sisi)             # ukuran huruf label
    lw = max(0.8, 0.0016 * sisi)            # tebal garis
    rot = page.rotation
    shape = page.new_shape()

    for i, k in enumerate(kotak, start=1):
        wk = k.get("warna") or warna_ke(i - 1)  # satu warna untuk kotak, nomor, kotak tulisan, dan isi tulisannya
        for it in k["item"]:                # kotak tulisan (miring), garis lebih tipis
            pts = ke_pdf(page, it["kotak"]["poly"], skala)
            shape.draw_polyline(pts + [pts[0]])
            shape.finish(color=wk, width=lw, closePath=True)
        pts = ke_pdf(page, k["poly"], skala)    # kotak: garis lebih tebal
        shape.draw_polyline(pts + [pts[0]])
        shape.finish(color=wk, width=lw * 1.4, closePath=True)
    shape.commit()

    def teks_di(titik_tampil, teks, warna, tebal=False, halo=True):
        """Tulis teks di posisi (dalam koordinat tampilan, yaitu setelah halaman diputar)."""
        nama = "hebo" if tebal else "helv"
        lebar = fitz.get_text_length(teks, fontname=nama, fontsize=fs)
        x, y = titik_tampil
        if halo:
            a = fitz.Point(x - 1, y - fs) * page.derotation_matrix
            b = fitz.Point(x + lebar + 1, y + 0.3 * fs) * page.derotation_matrix
            r = fitz.Rect(a, b).normalize()
            page.draw_rect(r, color=None, fill=(1, 1, 1), fill_opacity=0.75)
        page.insert_text(fitz.Point(x, y) * page.derotation_matrix, teks, fontsize=fs, fontname=nama, color=warna, rotate=rot)
        return lebar

    # Penempatan label
    def titik_garis(poly_px):
        """Titik-titik sepanjang sisi poligon (koordinat tampilan, satuan pt), untuk cek tabrakan."""
        p = np.asarray(poly_px, np.float32) / skala
        out = []
        for j in range(len(p)):
            a_, b_ = p[j], p[(j + 1) % len(p)]
            n = max(2, int(np.linalg.norm(b_ - a_) / 1.5))
            out.append(a_ + (b_ - a_) * np.linspace(0, 1, n)[:, None])
        return np.vstack(out)

    garis_kotak = [titik_garis(k["poly"]) for k in kotak]
    garis_teks = [np.vstack([titik_garis(it["kotak"]["poly"]) for it in k["item"]]) if k["item"] else np.zeros((0, 2))
                  for k in kotak]
    terpasang = []
    W_hal, H_hal = page.rect.width, page.rect.height

    def kandidat_menempel(k, W, g, t_label):
        """Persegi panjang label yang menempel (jarak g pt) di luar tiap sisi poligon kotak, urut dari yang paling dekat sudut kiri-atas."""
        P = np.asarray(k["poly"], np.float32) / skala
        c = P.mean(axis=0)
        jangkar = P[np.argmin(P[:, 0] + P[:, 1])]
        lw_, lh_ = W + 2, t_label
        hasil = []
        for a_, b_ in zip(P, np.roll(P, -1, axis=0)):
            panj = float(np.linalg.norm(b_ - a_))
            if panj < 1e-3:
                continue
            arah = (b_ - a_) / panj
            n = np.array([arah[1], -arah[0]], np.float32)
            if np.dot(n, (a_ + b_) / 2 - c) < 0:
                n = -n
            hn = abs(n[0]) * lw_ / 2 + abs(n[1]) * lh_ / 2
            for sk in np.linspace(0, 1, max(2, int(panj / 3))):
                p_ = a_ + (b_ - a_) * sk
                cen = p_ + n * (g + hn)
                biaya = float(np.linalg.norm(p_ - jangkar)) + (0.0 if n[1] < -0.3 else 60.0)
                hasil.append((biaya, (float(cen[0] - lw_ / 2), float(cen[1] - lh_ / 2), float(cen[0] + lw_ / 2), float(cen[1] + lh_ / 2))))
        hasil.sort(key=lambda t_: t_[0])
        return hasil

    def lebar_label(k, i):
        w = fitz.get_text_length(str(i), fontname="hebo", fontsize=fs)
        for it in k["item"]:
            isi = it["teks"] if len(it["teks"]) <= 40 else it["teks"][:37] + "..."
            w += 0.5 * fs + fitz.get_text_length(latin(isi), fontname="helv", fontsize=fs)
        return w

    def bebas(r, idx, di_dalam):
        x0, y0, x1, y1 = r
        if x0 < 1 or y0 < 1 or x1 > W_hal - 1 or y1 > H_hal - 1:
            return False
        for (a0, b0, a1, b1) in terpasang:
            if x0 < a1 + 1 and x1 > a0 - 1 and y0 < b1 + 1 and y1 > b0 - 1:
                return False
        pad = 1.5
        for j in range(len(kotak)):
            for pts_ in ((garis_kotak[j], garis_teks[j]) if (j != idx or di_dalam) else (garis_teks[j],)):
                if len(pts_) and np.any((pts_[:, 0] > x0 - pad) & (pts_[:, 0] < x1 + pad) & (pts_[:, 1] > y0 - pad) & (pts_[:, 1] < y1 + pad)):
                    return False
        return True

    for i, k in enumerate(kotak, start=1):
        xs = [p[0] for p in k["poly"]]
        ys = [p[1] for p in k["poly"]]
        kx0, kx1, ky0, ky1 = min(xs) / skala, max(xs) / skala, min(ys) / skala, max(ys) / skala
        W = lebar_label(k, i)
        t, bawah = fs * 1.3, 0.3 * fs
        pilih = None
        # 1) Label menempel di sisi kotak yang sebenarnya
        for g in (2.2, 4.0, 7.0, 12.0):
            for _, r in kandidat_menempel(k, W, g, t_label=fs + bawah):
                if bebas(r, i - 1, True):
                    pilih = (r[0] + 1, r[3] - bawah, r)
                    break
            if pilih is not None:
                break
        # 2) Cadangan: di atas/bawah kotak pembatas atau di dalam kotak
        kandidat = [
            (kx0, ky0 - bawah - 1.5, False), ((kx0 + kx1 - W) / 2, ky0 - bawah - 1.5, False), (kx1 - W, ky0 - bawah - 1.5, False),
            (kx0, ky1 + t + 1.5, False), (kx1 - W, ky1 + t + 1.5, False),
            (kx0 + 4, ky0 + t + 4, True), (kx0 + 4, ky1 - bawah - 4, True),
        ]
        for (x, y, dlm) in (kandidat if pilih is None else []):
            x = min(max(x, 2), W_hal - W - 2)
            r = (x - 1, y - fs, x + W + 1, y + bawah)
            if bebas(r, i - 1, dlm):
                pilih = (x, y, r)
                break
        if pilih is None:                  # semua sempit: pakai di atas-kiri saja
            x, y = min(max(kx0, 2), W_hal - W - 2), max(ky0 - bawah - 1.5, fs + 2)
            pilih = (x, y, (x - 1, y - fs, x + W + 1, y + bawah))
        x, y, r = pilih
        terpasang.append(r)
        wk = k.get("warna") or warna_ke(i - 1)
        lebar = teks_di((x, y), str(i), wk, tebal=True)
        x += lebar + 0.5 * fs
        for it in k["item"]:                  # isi tulisan di sebelah nomor, warnanya sama dengan nomornya
            isi = it["teks"] if len(it["teks"]) <= 40 else it["teks"][:37] + "..."
            x += teks_di((x, y), latin(isi), wk) + 0.5 * fs

    judul = f"Halaman {nomor_halaman}: {len(kotak)} kotak"
    teks_di((4, fs + 2), judul, (0.15, 0.15, 0.15), tebal=True)


# 11. Jalankan
_model_cache = {}


def muat_model():
    """Muat YOLO sekali saja. Hasil: (model, id_kotak, id_teks)."""
    if "m" not in _model_cache:
        from ultralytics import YOLO
        model = YOLO(str(MODEL))
        id_kotak = [i for i, n in model.names.items() if n == "kotak"][0]
        id_teks = [i for i, n in model.names.items() if n == "text"][0]
        _model_cache["m"] = (model, id_kotak, id_teks)
    return _model_cache["m"]


def analisis_pdf(pdf_masuk, halaman=None, pakai_yolo=None, sumber_teks=None, dpi=None, laporan=print):
    """Tahap berat (YOLO + garis + OCR) tanpa menggambar. Hasil: [{'halaman', 'skala', 'kotak': [...]}], bisa disunting lalu digambar lagi."""
    global SUMBER_TEKS, DPI
    lama = (SUMBER_TEKS, DPI)
    if sumber_teks is not None:
        SUMBER_TEKS = sumber_teks
    if dpi is not None:
        DPI = dpi
    try:
        doc = fitz.open(str(pdf_masuk))
        model = id_kotak = id_teks = None
        if PAKAI_YOLO if pakai_yolo is None else pakai_yolo:
            model, id_kotak, id_teks = muat_model()
        daftar = halaman if halaman else list(range(1, len(doc) + 1))
        data = []
        for nomor in daftar:
            laporan(f"Memproses halaman {nomor} dari {len(daftar)} ...")
            kotak, skala = proses_halaman(doc[nomor - 1], model, id_kotak, id_teks)
            data.append({"halaman": nomor, "skala": skala, "kotak": kotak})
        return data
    finally:
        SUMBER_TEKS, DPI = lama


def tulis_pdf(pdf_masuk, data, pdf_keluar):
    """Gambar semua kotak + tulisan di data ke salinan PDF asli. Hasil: ringkasan per halaman."""
    doc = fitz.open(str(pdf_masuk))
    ringkasan = []
    for h in data:
        kotak = h["kotak"]
        gambar_pdf(doc[h["halaman"] - 1], kotak, h["skala"], h["halaman"])
        ringkasan.append({"halaman": h["halaman"],
                          "kotak": [{"nomor": i, "sumber": k["sumber"], "tulisan": [it["teks"] for it in k["item"]]}
                                    for i, k in enumerate(kotak, start=1)]})
    doc.save(str(pdf_keluar), deflate=True)
    return ringkasan


def proses_pdf(pdf_masuk, pdf_keluar, halaman=None, pakai_yolo=None, sumber_teks=None, dpi=None, laporan=print):
    """Proses satu file PDF dan simpan PDF hasilnya. Hasil: [{'halaman', 'kotak': [{'nomor', 'sumber', 'tulisan': [...]}]}]"""
    data = analisis_pdf(pdf_masuk, halaman, pakai_yolo, sumber_teks, dpi, laporan)
    return tulis_pdf(pdf_masuk, data, pdf_keluar)


# 12. Edit hasil di editor web
def _objek_dari_poly(poly_px, skala):
    """Poligon kotak (piksel) -> {cx, cy, w, h, a} dalam satuan pt halaman; a = derajat, positif searah jarum jam."""
    P = np.asarray(poly_px, np.float32) / skala
    b = cv2.boxPoints(cv2.minAreaRect(P))
    e1, e2 = b[1] - b[0], b[2] - b[1]
    a = math.degrees(math.atan2(e1[1], e1[0]))
    w, h = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    while a > 45:
        a -= 90
        w, h = h, w
    while a <= -45:
        a += 90
        w, h = h, w
    c = b.mean(axis=0)
    return {"cx": float(c[0]), "cy": float(c[1]), "w": w, "h": h, "a": float(a)}

def _poly_dari_objek(o, skala):
    t = math.radians(o["a"])
    c, s = math.cos(t), math.sin(t)
    pts = []
    for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = i * o["w"] / 2, j * o["h"] / 2
        pts.append([o["cx"] + lx * c - ly * s, o["cy"] + lx * s + ly * c])
    return np.array(pts, np.float32) * skala

def _hex(warna):
    return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in warna)

def _rgb(hexwarna):
    h = hexwarna.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

def data_ke_editor(data):
    """Data hasil analisis -> struktur ringan (JSON) untuk editor di browser."""
    halaman = []
    for h in data:
        sk = h["skala"]
        kotak = []
        for i, k in enumerate(h["kotak"]):
            wk = _hex(k.get("warna") or warna_ke(i))
            kotak.append({"id": i, "obj": _objek_dari_poly(k["poly"], sk), "warna": wk,
                          "item": [{"id": j, "obj": _objek_dari_poly(it["kotak"]["poly"], sk), "teks": it["teks"], "warna": wk}
                                   for j, it in enumerate(k["item"])]})
        halaman.append({"halaman": h["halaman"], "kotak": kotak})
    return halaman

def editor_ke_data(data, edit):
    """Data asli + hasil suntingan dari editor -> data baru siap digambar oleh tulis_pdf.
    Objek yang tidak diubah geometrinya dipertahankan persis seperti hasil deteksi."""
    import copy
    asli = {h["halaman"]: h for h in data}
    hasil = []
    for e in edit:
        h0 = asli.get(e["halaman"])
        if h0 is None:
            continue
        sk = h0["skala"]
        kotak_baru = []
        for ek in e["kotak"]:
            if ek.get("id") is not None:
                k = copy.deepcopy(h0["kotak"][ek["id"]])
            else:
                k = {"poly": None, "item": [], "teks": [], "sumber": "manual"}
            if ek.get("d") or k["poly"] is None:
                k["poly"] = _poly_dari_objek(ek["obj"], sk)
            item_baru = []
            for ei in ek.get("item", []):
                isi = str(ei.get("teks", "")).strip()
                if not isi:
                    continue
                if ei.get("id") is not None:
                    it = copy.deepcopy(h0["kotak"][ek["id"]]["item"][ei["id"]])
                else:
                    it = {"kotak": {"poly": None}, "sudut": None, "sumber_teks": "manual",
                          "warna": PALET_TEKS[(len(item_baru) + len(kotak_baru)) % len(PALET_TEKS)]}
                it["teks"] = isi
                if ei.get("d") or it["kotak"]["poly"] is None:
                    it["kotak"]["poly"] = _poly_dari_objek(ei["obj"], sk)
                item_baru.append(it)
            if ek.get("warna"):
                k["warna"] = _rgb(ek["warna"])
            elif not k.get("warna"):
                k["warna"] = warna_ke(len(kotak_baru))
            for it in item_baru:
                it["warna"] = k["warna"]            # isi tulisan selalu sewarna dengan kotaknya
            k["item"] = item_baru
            kotak_baru.append(k)
        hasil.append({"halaman": e["halaman"], "skala": sk, "kotak": kotak_baru})
    return hasil


# Accuracy + confusion matrix (khusus deteksi kotak)
def hitung_evaluasi(ringkasan, anotasi):
    """ringkasan = hasil proses_pdf. anotasi = {halaman: {"kotak": n, "bukan": n, "salah": [nomor, ...]}}.
    Hasil: (daftar baris per halaman, baris total). Tiap baris: halaman, terdeteksi, TP, FP, FN, TN, dan ukuran-ukurannya.
      TP = kotak asli yang terdeteksi      FP = bukan kotak tapi terdeteksi sebagai kotak
      FN = kotak asli yang terlewat        TN = bukan kotak dan benar tidak terdeteksi"""
    baris = []
    for h in ringkasan:
        a = anotasi.get(h["halaman"])
        if a is None:
            continue
        nomor_ada = {k["nomor"] for k in h["kotak"]}
        salah = {n for n in a.get("salah", []) if n in nomor_ada}
        fp = len(salah)
        tp = len(h["kotak"]) - fp
        fn = max(0, a["kotak"] - tp)
        tn = max(0, a["bukan"] - fp)
        baris.append(_ukuran({"halaman": h["halaman"], "terdeteksi": len(h["kotak"]), "tp": tp, "fp": fp, "fn": fn, "tn": tn}))
    total = {"halaman": "TOTAL", "terdeteksi": sum(b["terdeteksi"] for b in baris)}
    for kunci in ("tp", "fp", "fn", "tn"):
        total[kunci] = sum(b[kunci] for b in baris)
    return baris, _ukuran(total)


def _ukuran(b):
    tp, fp, fn, tn = b["tp"], b["fp"], b["fn"], b["tn"]
    n = tp + fp + fn + tn
    b["accuracy"] = (tp + tn) / n if n else 0.0
    b["precision"] = tp / (tp + fp) if tp + fp else 0.0
    b["recall"] = tp / (tp + fn) if tp + fn else 0.0
    pr = b["precision"] + b["recall"]
    b["f1"] = 2 * b["precision"] * b["recall"] / pr if pr else 0.0
    return b


def _lukis_matriks(fig, total, judul):
    m = np.array([[total["tp"], total["fn"]], [total["fp"], total["tn"]]])
    ax = fig.subplots()
    ax.imshow(m, cmap="Blues", vmin=0, vmax=max(1, m.max()))
    nama = [["TP (kotak, terdeteksi)", "FN (kotak, terlewat)"], ["FP (bukan kotak,\ndikira kotak)", "TN (bukan kotak,\nbenar diabaikan)"]]
    for i in range(2):
        for j in range(2):
            warna = "white" if m[i, j] > m.max() * 0.55 else "black"
            ax.text(j, i, f"{m[i, j]}\n{nama[i][j]}", ha="center", va="center", color=warna, fontsize=10)
    ax.set_xticks([0, 1], ["Kotak", "Bukan kotak"])
    ax.set_yticks([0, 1], ["Kotak", "Bukan kotak"])
    ax.set_xlabel("Tebakan program")
    ax.set_ylabel("Kenyataan")
    ax.set_title(f"{judul}\naccuracy {total['accuracy']:.1%}  precision {total['precision']:.1%}  "
                 f"recall {total['recall']:.1%}  F1 {total['f1']:.1%}", fontsize=9)
    fig.tight_layout()


def tampilkan_confusion_matrix(total, judul="Confusion matrix deteksi kotak"):
    """Tampilkan confusion matrix langsung di Spyder (panel Plots)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.4, 4.8), dpi=110)
    _lukis_matriks(fig, total, judul)
    plt.show()


def main():
    keluar = PDF_MASUK.with_name(PDF_MASUK.stem + "_hasil.pdf")
    ringkas = proses_pdf(PDF_MASUK, keluar, halaman=HALAMAN)
    for h in ringkas:
        print(f"Halaman {h['halaman']}: {len(h['kotak'])} kotak")
        for k in h["kotak"]:
            print(f"  Kotak {k['nomor']}: " + (", ".join(repr(t) for t in k["tulisan"]) if k["tulisan"] else "(tidak ada tulisan)"))
    print("PDF hasil disimpan di:", keluar)
    if ANOTASI:
        baris, total = hitung_evaluasi(ringkas, ANOTASI)
        print("\n=== Evaluasi deteksi kotak ===")
        for b in baris + [total]:
            print(f"Halaman {b['halaman']}: terdeteksi {b['terdeteksi']}  TP={b['tp']} FP={b['fp']} FN={b['fn']} TN={b['tn']}  "
                  f"accuracy={b['accuracy']:.1%} precision={b['precision']:.1%} recall={b['recall']:.1%} F1={b['f1']:.1%}")
        tampilkan_confusion_matrix(total)


if __name__ == "__main__":
    main()