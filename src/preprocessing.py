"""
Tahap 6: Pengolahan Gambar (Image Processing / Preprocessing)
Menggunakan OpenCV untuk membaca gambar, konversi grayscale, Gaussian Blur,
Thresholding, serta operasi morfologi (erosion, dilation, opening, closing)
guna memperjelas garis dinding dan menghilangkan gangguan visual (teks, garis ukur, arsiran).
"""

import os
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_image(image_source: Union[str, np.ndarray]) -> np.ndarray:
    """Membaca gambar dari path file menggunakan cv2.imread() atau array numpy BGR."""
    if isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_source}")
        image = cv2.imread(image_source)
        if image is None:
            raise ValueError(f"Gagal membaca gambar dari: {image_source}")
        return image
    elif isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            return cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
        elif image_source.shape[2] == 4:
            return cv2.cvtColor(image_source, cv2.COLOR_RGBA2BGR)
        return image_source.copy()
    else:
        raise TypeError("image_source harus berupa path string atau numpy.ndarray")


def preprocess_floorplan(
    image_source: Union[str, np.ndarray],
    blur_ksize: int = 3,
    thresh_val: int = 200,
    close_ksize: int = 9,
    open_ksize: int = 7,
    door_gap_px: int = 115,
    ext_gap_px: int = 230,
    auto_scale_kernels: bool = True,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Menjalankan seluruh tahapan preprocessing gambar denah rumah (Bagian 6):
    1. Membaca gambar (cv2.imread)
    2. Konversi ke Grayscale (cv2.cvtColor)
    3. Gaussian Blur (cv2.GaussianBlur)
    4. Thresholding (cv2.threshold)
    5. Operasi Morfologi (Erosion, Dilation, Opening, Closing) untuk memisahkan
       dinding utama dari teks, simbol furnitur, arsiran teras, dan garis dimensi.
    """
    original = load_image(image_source)
    h_img, w_img = original.shape[:2]
    scale = (max(h_img, w_img) / 1950.0) if auto_scale_kernels else 1.0

    # 1. Konversi ke Grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur (pastikan ukuran kernel ganjil >= 3)
    bk = max(3, int(blur_ksize))
    if bk % 2 == 0:
        bk += 1
    blur = cv2.GaussianBlur(gray, (bk, bk), 0)

    # 3. Thresholding Binary Inverse (garis denah menjadi putih=255, latar belakang hitam=0)
    _, threshold = cv2.threshold(blur, int(thresh_val), 255, cv2.THRESH_BINARY_INV)

    # 4. Operasi Morfologi Dasar:
    # 4a. Light opening (erosion -> dilation) untuk memutus garis tipis 1px dari huruf/angka
    k_light = max(1, min(2, int(round(2 * scale))))
    kernel_light = cv2.getStructuringElement(cv2.MORPH_RECT, (k_light, k_light))
    eroded_light = cv2.erode(threshold, kernel_light, iterations=1)
    opened_light = cv2.dilate(eroded_light, kernel_light, iterations=1)

    # Filter komponen terhubung kecil (menghapus huruf, angka ukuran, simbol kecil)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(opened_light, connectivity=8)
    no_text = np.zeros_like(threshold)
    min_span = max(15, int(round(45 * scale)))
    min_area = max(80, int(round(600 * scale * scale)))
    for i in range(1, num_l):
        w_c = stats[i, cv2.CC_STAT_WIDTH]
        h_c = stats[i, cv2.CC_STAT_HEIGHT]
        area_c = stats[i, cv2.CC_STAT_AREA]
        if max(w_c, h_c) > min_span or area_c > min_area:
            no_text[labels == i] = 255

    # Kembalikan ketebalan garis dinding dengan dilation
    no_text = cv2.dilate(no_text, kernel_light, iterations=1)

    # 4b. Morphological Closing untuk menyatukan arsiran dinding ganda (////)
    kc = max(3, min(11, int(round(close_ksize * scale))))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kc, kc))
    morph_closed = cv2.morphologyEx(no_text, cv2.MORPH_CLOSE, kernel_close)

    # 4c. Morphological Opening untuk menghapus garis tipis (garis dimensi, arsiran teras, busur pintu)
    # Dibatasi maks 7px agar dinding solid hitam (~8-10px pada denah resolusi 3508px) tetap utuh
    ko = max(3, min(7, int(round(open_ksize * scale))))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko))
    morph_opened = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, kernel_open)

    # Fallback apabila gambar denah memiliki garis dinding tunggal yang tipis
    if np.count_nonzero(morph_opened) < 0.002 * (h_img * w_img):
        kernel_thin_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_opened = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, kernel_thin_open)

    # 4d. Ekstraksi struktur dinding horizontal dan vertikal (menghilangkan sisa objek melengkung/mobil)
    kh_len = max(18, int(round(45 * scale)))
    kv_len = max(20, int(round(50 * scale)))
    kh_thk = max(2, min(6, int(round(4 * scale))))
    walls_h = cv2.morphologyEx(
        morph_opened,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kh_len, kh_thk)),
    )
    walls_v = cv2.morphologyEx(
        morph_opened,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kh_thk, kv_len)),
    )
    walls_unbridged = cv2.bitwise_or(walls_h, walls_v)

    # 4e. Penutupan celah pintu & jendela (Two-Tier Directional Closing):
    # Tier 1: Dinding eksterior tebal (menutup bukaan garasi & jendela besar)
    k_ext = max(6, int(round(16 * scale)))
    ext_thick = cv2.morphologyEx(
        morph_opened,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_ext, k_ext)),
    )
    num_e, l_e, s_e, _ = cv2.connectedComponentsWithStats(ext_thick)
    ext_clean = np.zeros_like(ext_thick)
    for i in range(1, num_e):
        if (
            s_e[i, cv2.CC_STAT_AREA] > int(800 * scale * scale)
            and max(s_e[i, cv2.CC_STAT_WIDTH], s_e[i, cv2.CC_STAT_HEIGHT]) > int(40 * scale)
        ):
            ext_clean[l_e == i] = 255

    ext_thk_bar = max(5, int(round(15 * scale)))
    ext_h = cv2.morphologyEx(
        ext_clean,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(round(42 * scale))), ext_thk_bar)),
    )
    ext_v = cv2.morphologyEx(
        ext_clean,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (ext_thk_bar, max(25, int(round(75 * scale))))),
    )
    eg = max(40, int(round(ext_gap_px * scale)))
    ext_h_b = cv2.morphologyEx(
        ext_h,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (eg, ext_thk_bar)),
    )
    ext_v_b = cv2.morphologyEx(
        ext_v,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (ext_thk_bar, eg)),
    )
    ext_all = cv2.bitwise_or(ext_clean, cv2.bitwise_or(ext_h_b, ext_v_b))

    # Tier 2: Dinding partisi interior (menutup celah pintu antar-ruangan)
    ig = max(25, int(round(door_gap_px * scale)))
    walls_h_b = cv2.morphologyEx(
        walls_h,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (ig, kh_thk)),
    )
    walls_v_b = cv2.morphologyEx(
        walls_v,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kh_thk, ig)),
    )

    walls_bridged = cv2.bitwise_or(cv2.bitwise_or(walls_h_b, walls_v_b), ext_all)

    # Pertahankan komponen jaringan bangunan utama (terbesar) agar artefak luar hilang
    num_l2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(walls_bridged, np.ones((7, 7), np.uint8))
    )
    if num_l2 > 1:
        largest_label = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        main_network = (labels2 == largest_label).astype(np.uint8) * 255
        walls_clean = cv2.bitwise_and(walls_bridged, main_network)
        walls_unbridged = cv2.bitwise_and(walls_unbridged, main_network)
    else:
        walls_clean = walls_bridged.copy()

    return {
        "original": original,
        "gray": gray,
        "blur": blur,
        "threshold": threshold,
        "no_text": no_text,
        "morph_closed": morph_closed,
        "morph_opened": morph_opened,
        "walls_unbridged": walls_unbridged,
        "walls_clean": walls_clean,
        "scale_factor": scale,
    }


def save_preprocessing_results(
    prep_result: Dict[str, Union[np.ndarray, float]],
    output_dir: str = "results",
) -> Dict[str, str]:
    """Menyimpan tahapan hasil preprocessing ke folder results dan membuat plot perbandingan Matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    stage_files = {
        "grayscale": ("grayscale.png", prep_result["gray"]),
        "blur": ("blur.png", prep_result["blur"]),
        "threshold": ("threshold.png", prep_result["threshold"]),
        "morph_closed": ("morph_closed.png", prep_result["morph_closed"]),
        "morph_opened": ("morph_opened.png", prep_result["morph_opened"]),
        "walls_clean": ("walls_clean.png", prep_result["walls_clean"]),
    }

    for key, (filename, img_arr) in stage_files.items():
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, img_arr)
        saved_paths[key] = path

    # Buat visualisasi komparasi tahapan menggunakan Matplotlib (Bagian 5 & 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    stages_plot = [
        ("1. Gambar Asli (RGB)", cv2.cvtColor(prep_result["original"], cv2.COLOR_BGR2RGB), None),
        ("2. Grayscale (cv2.cvtColor)", prep_result["gray"], "gray"),
        ("3. Gaussian Blur (cv2.GaussianBlur)", prep_result["blur"], "gray"),
        ("4. Thresholding (cv2.threshold)", prep_result["threshold"], "gray"),
        ("5. Morphological Closing & Opening", prep_result["morph_opened"], "gray"),
        ("6. Mask Dinding Bersih (Siap Deteksi)", prep_result["walls_clean"], "gray"),
    ]

    for ax, (title, img_data, cmap) in zip(axes.flat, stages_plot):
        ax.imshow(img_data, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("SmartPlan - Tahapan Pengolahan Citra (Image Processing)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "preprocessing_stages.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["summary_plot"] = summary_path

    return saved_paths


if __name__ == "__main__":
    sample_path = os.path.join("dataset", "train", "denah1.png")
    if not os.path.exists(sample_path):
        print("Gambar tidak ditemukan:", sample_path)
    else:
        res = preprocess_floorplan(sample_path)
        paths = save_preprocessing_results(res, output_dir="results")
        print("Preprocessing berhasil!")
        for k, p in paths.items():
            print(f" - {k}: {p}")
