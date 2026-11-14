# SmartPlan – Versi Jupyter Notebook

Folder ini berisi implementasi lengkap projek **SmartPlan** dalam bentuk **Jupyter Notebook** yang berdiri sendiri (*self-contained*), terpisah dari aplikasi Streamlit (`app.py`) dan CLI (`main.py`) pada direktori utama.

## Isi Folder
- `SmartPlan_Notebook.ipynb` — Notebook utama berisi seluruh tahapan (Tahap 4–13):
  1. Pemeriksaan Dataset Kaggle (`dataset/raw/`) & Sampel Uji (`dataset/train/`)
  2. Pengolahan Gambar (*Grayscale, Gaussian Blur, Thresholding, Morphological Operations*)
  3. Deteksi Garis Dinding (*Hough Line Transform*) & Area Ruangan (*Flood Fill + Contours*)
  4. Perhitungan Dimensi & Luas Ruangan ($\text{m}^2$)
  5. Kalkulator Kebutuhan Material (**Keramik Lantai**, **Cat Dinding**, & material konstruksi lanjutan)
  6. Estimasi Biaya Material Berbasis **Pandas**
  7. Evaluasi Akurasi & Benchmark Lintas 10 Denah
- `outputs/` — Folder keluaran visualisasi PNG dan tabel CSV khusus dari notebook.

## Cara Menjalankan
Buka langsung `SmartPlan_Notebook.ipynb` di VS Code (dengan ekstensi Jupyter), atau jalankan melalui terminal:
```bash
jupyter notebook notebooks/SmartPlan_Notebook.ipynb
```
Atau dengan JupyterLab:
```bash
jupyter lab notebooks/SmartPlan_Notebook.ipynb
```

