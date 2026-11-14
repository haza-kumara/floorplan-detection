# SmartPlan – Sistem Deteksi Denah Rumah dan Estimasi Kebutuhan Material Bangunan Berbasis Computer Vision

SmartPlan adalah aplikasi berbasis **Computer Vision klasik (OpenCV)** yang dirancang untuk membaca dan menganalisis gambar denah rumah dua dimensi (2D), mendeteksi garis dinding dan area ruangan, menghitung dimensi serta luas setiap ruangan dalam satuan meter persegi ($\text{m}^2$), serta memperkirakan kebutuhan material bangunan (**Keramik Lantai** & **Cat Dinding**) beserta **Estimasi Biaya**.

---

## Struktur Direktori Projek

```text
floorplan-detection/
├── app.py                      # Aplikasi web interaktif berbasis Streamlit (Tahap 11 & 12)
├── main.py                     # Script utama CLI untuk menjalankan seluruh pipeline & evaluasi
├── test.py                     # Script pengujian unit & integrasi sistem (Tahap 13)
├── download_dataset.py         # Script pengunduh dataset Kaggle (asutoshprad/floor-plan-dataset)
├── data/
│   └── material_prices.csv     # Database tabel harga satuan material (dikelola dengan Pandas)
├── dataset/
│   ├── raw/                    # Dataset asli Kaggle (121 gambar denah di X/ & anotasi di Y/)
│   └── train/                  # Sampel gambar denah untuk pengujian (denah1.png, denah2.png, denah3.png)
├── results/                    # Output visualisasi tahapan OpenCV, deteksi, dan laporan CSV
└── src/
    ├── __init__.py
    ├── preprocessing.py        # Tahap 6: Grayscale, Gaussian Blur, Thresholding, Morphological Ops
    ├── detection.py            # Tahap 7: Hough Line Transform, Flood Fill, & Contour Detection
    ├── measurement.py          # Tahap 8: Konversi skala piksel->meter, dimensi & luas ruangan
    ├── material_calculator.py  # Tahap 9: Perhitungan kebutuhan keramik lantai, cat dinding, & material dinding
    ├── cost_estimator.py       # Tahap 10: Perhitungan estimasi biaya berbasis Pandas DataFrame
    └── evaluation.py           # Tahap 4 & 13: Pemeriksaan dataset & evaluasi akurasi pengukuran
```

---

## Tahapan Alur Kerja Sistem

1. **Pengolahan Gambar (`src/preprocessing.py`)**:
   - Membaca gambar menggunakan `cv2.imread()`
   - Konversi ke citra keabuan (*Grayscale*) menggunakan `cv2.cvtColor()`
   - Mengurangi *noise* menggunakan `cv2.GaussianBlur()`
   - Memisahkan garis denah dari latar belakang dengan `cv2.threshold()` (`THRESH_BINARY_INV`)
   - Memperbaiki garis dinding dan menghilangkan gangguan visual (teks, angka ukuran, arsiran teras) menggunakan *Morphological Operations* (`erosion`, `dilation`, `opening`, `closing`).
2. **Deteksi Dinding dan Ruangan (`src/detection.py`)**:
   - **Hough Line Transform (`cv2.HoughLinesP`)** pada *skeleton* dinding untuk mendeteksi segmen garis dinding horizontal dan vertikal.
   - **Flood Fill (`cv2.floodFill`)** dari area eksterior untuk memisahkan latar luar dan menemukan area ruangan tertutup.
   - **Contour Detection (`cv2.findContours`)** untuk mengidentifikasi batas dan luas piksel setiap ruangan.
3. **Perhitungan Dimensi dan Luas Bangunan (`src/measurement.py`)**:
   - Konversi ukuran piksel ke meter berdasarkan skala atau satu ukuran referensi (contoh: panjang dinding `200 piksel` = `4 meter` $\rightarrow$ skala `50 piksel/meter`).
   - Menghitung panjang, lebar, dan luas setiap ruangan ($\text{m}^2$), total luas lantai bangunan ($\text{m}^2$), serta panjang total dinding ($\text{m}$).
4. **Perhitungan Kebutuhan Material (`src/material_calculator.py`)**:
   - **Keramik Lantai**: Berdasarkan luas lantai bangunan, pilihan ukuran keramik ($30\times30$, $40\times40$, $50\times50$, $60\times60$, $80\times80\text{ cm}$), dan faktor cadangan (*waste factor*, default `10%`).
   - **Cat Dinding**: Berdasarkan panjang dinding $\times$ tinggi dinding (input pengguna), pengurangan pintu/jendela, daya sebar cat ($\text{m}^2/\text{liter}$), serta jumlah lapisan pengecatan (*coats*).
   - **Material Konstruksi Lanjutan (Opsional)**: Estimasi kebutuhan batu bata, semen, dan pasir.
5. **Estimasi Biaya (`src/cost_estimator.py`)**:
   - Mengalikan jumlah kebutuhan material dengan harga satuan dari `data/material_prices.csv` menggunakan **Pandas** dan menyajikan tabel rincian biaya beserta total estimasi biaya.

---

## Cara Menjalankan Program

### 1. Menjalankan Pengujian Sistem (`test.py`)
```bash
python test.py
```

### 2. Menjalankan Pipeline Lengkap via Terminal (`main.py`)
```bash
python main.py
```
Atau dengan argumen kustom (termasuk estimasi material konstruksi lanjutan):
```bash
python main.py --image dataset/train/denah1.png --ref-pixels 269.7 --ref-meters 3.35 --wall-height 3.0 --tile-size 40 --waste 10 --structural
```

### 3. Menjalankan Antarmuka Aplikasi Web Interaktif (`Streamlit`)
```bash
streamlit run app.py
```
