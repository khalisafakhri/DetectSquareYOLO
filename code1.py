# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:56:42 2026

@author: fakhr
"""

from pathlib import Path
from collections import Counter
import yaml

DATASET = Path(r"C:\Pendahuluan TA\Tugas 3 - Classify Square and Text\DatasetDetectSquare-yolo11")

with open(DATASET / "data.yaml") as f:
    cfg = yaml.safe_load(f)

names = cfg["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
print("Kelas:", names)

for split in ["train", "valid", "test"]:
    lbl_dir = DATASET / split / "labels"
    img_dir = DATASET / split / "images"
    if not lbl_dir.exists():
        continue

    counter = Counter()
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                counter[names[int(line.split()[0])]] += 1

    n_img = len(list(img_dir.glob("*")))
    print(f"\n{split}: {n_img} gambar")
    for kelas, jumlah in counter.most_common():
        print(f"  {kelas}: {jumlah}")
        
#%%
import random
from pathlib import Path
import cv2
import yaml

DATASET = Path(r"C:\Pendahuluan TA\Tugas 3 - Classify Square and Text\DatasetDetectSquare-yolo11")
OUT = DATASET.parent / "cek_visual"
OUT.mkdir(exist_ok=True)

CEK = ["rectangle", "square", "process"]  # kelas yang mau dilihat
JUMLAH = 6                                # gambar per kelas

with open(DATASET / "data.yaml") as f:
    cfg = yaml.safe_load(f)
names = cfg["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]


def cari_gambar(stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = DATASET / "train" / "images" / (stem + ext)
        if p.exists():
            return p
    return None


random.seed(0)
label_files = list((DATASET / "train" / "labels").glob("*.txt"))

for kelas in CEK:
    cid = names.index(kelas)

    # cari file label yang memuat kelas ini
    kandidat = []
    for lf in label_files:
        ids = {int(l.split()[0]) for l in lf.read_text().splitlines() if l.strip()}
        if cid in ids:
            kandidat.append(lf)

    for lf in random.sample(kandidat, min(JUMLAH, len(kandidat))):
        img_path = cari_gambar(lf.stem)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        for line in lf.read_text().splitlines():
            p = line.split()
            if not p or int(p[0]) != cid:
                continue
            xc, yc, bw, bh = map(float, p[1:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(OUT / f"{kelas}_{lf.stem}.jpg"), img)

print("Selesai. Buka folder:", OUT)

#%%
import shutil
from collections import Counter
from pathlib import Path
import yaml

SRC = Path(r"C:\Pendahuluan TA\Tugas 3 - Classify Square and Text\DatasetDetectSquare-yolo11")
DST = SRC.parent / "DatasetKotak-yolo11"

GABUNG = ["rectangle", "square", "process"]  # kelas yang digabung
NAMA_GABUNGAN = "kotak"                      # nama kelas hasil gabungan

with open(SRC / "data.yaml") as f:
    cfg = yaml.safe_load(f)
names = cfg["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]

# daftar kelas baru: "kotak" jadi id 0, sisanya mengikuti urutan lama
lainnya = [n for n in names if n not in GABUNG]
new_names = [NAMA_GABUNGAN] + lainnya

# id lama -> id baru
old2new = {}
for i, n in enumerate(names):
    old2new[i] = 0 if n in GABUNG else new_names.index(n)


def iou(a, b):
    """a dan b berformat (x_tengah, y_tengah, lebar, tinggi), ternormalisasi."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


splits_ada = []
for split in ["train", "valid", "test"]:
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"
    if not img_dir.exists():
        continue
    splits_ada.append(split)
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = Counter()
    duplikat = 0
    for img in img_dir.iterdir():
        shutil.copy2(img, DST / split / "images" / img.name)

        lbl = lbl_dir / (img.stem + ".txt")
        baris_baru = []
        kotak_boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if not p:
                    continue
                baru = old2new[int(p[0])]

                # buang kotak ganda (IoU sangat tinggi) setelah penggabungan
                if baru == 0 and len(p) == 5:
                    box = tuple(float(x) for x in p[1:5])
                    if any(iou(box, k) > 0.95 for k in kotak_boxes):
                        duplikat += 1
                        continue
                    kotak_boxes.append(box)

                baris_baru.append(" ".join([str(baru)] + p[1:]))
                counts[new_names[baru]] += 1

        (DST / split / "labels" / (img.stem + ".txt")).write_text("\n".join(baris_baru))

    print(f"\n{split}: {len(list(img_dir.iterdir()))} gambar, kotak ganda dibuang: {duplikat}")
    for kelas, jumlah in counts.most_common():
        print(f"  {kelas}: {jumlah}")

new_cfg = {
    "path": str(DST.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "nc": len(new_names),
    "names": new_names,
}
if "test" in splits_ada:
    new_cfg["test"] = "test/images"
with open(DST / "data.yaml", "w") as f:
    yaml.safe_dump(new_cfg, f)

print("\nSelesai. Dataset baru ada di:", DST.resolve())


#%%
from pathlib import Path
from ultralytics import YOLO

DATASET = Path(r"C:\Pendahuluan TA\Tugas 3 - Classify Square and Text\DatasetKotak-yolo11")
DATA_YAML = DATASET / "data.yaml"

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=3,         # uji coba singkat
        fraction=0.1,     # pakai 10% data latih saja untuk uji cepat
        imgsz=640,
        batch=8,
        workers=0,        # lebih aman di Windows + Spyder
        project="hasil_training",
        name="uji_coba",
    )
    
#%%
from pathlib import Path
from ultralytics import YOLO
from grafik_training import pasang_akurasi_latih

DATASET = Path(r"C:\Pendahuluan TA\Tugas 3 - Classify Square and Text\DatasetKotak-yolo11")
DATA_YAML = DATASET / "data.yaml"
HASIL = Path(r"C:\Pendahuluan TA\Tugas 3 - Classify Square and Text\hasil_training")

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    # catat akurasi data latih tiap epoch (fraction=0.5 SAMA dengan fraction di model.train di bawah)
    pasang_akurasi_latih(model, DATA_YAML, fraction=0.5)

    model.train(
        data=str(DATA_YAML),
        epochs=15,
        fraction=0.5,        # pakai 50% data latih agar muat di CPU
        imgsz=640,
        batch=8,
        workers=0,           # lebih aman di Windows + Spyder
        degrees=10.0,        # rotasi acak sampai 10 derajat (kotak miring)
        perspective=0.0005,  # sedikit distorsi perspektif (foto dari sudut)
        hsv_v=0.5,           # variasi kecerahan (gelap/bayangan ringan)
        close_mosaic=3,      # matikan mosaic di 3 epoch terakhir
        project=str(HASIL),
        name="train_utama",
    )