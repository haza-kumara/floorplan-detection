"""
Tahap 4 & Tahap 13: Pemeriksaan Dataset, Pengujian, dan Evaluasi Sistem
Mencakup:
1. Pemeriksaan otomatis dataset Kaggle (asutoshprad/floor-plan-dataset) di folder dataset/raw
2. Evaluasi perbandingan luas hasil deteksi OpenCV terhadap ukuran referensi tertulis pada denah
3. Evaluasi mask deteksi ruangan OpenCV terhadap anotasi Ground Truth dataset (folder Y)
4. Pengujian lintas beberapa denah dengan karakteristik berbeda (multi-image evaluation).
"""

import os
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_floorplan
from src.detection import detect_walls_hough, detect_rooms_floodfill_contours
from src.measurement import DATASET_SCALE_PRESETS, measure_building_dimensions


# Data ukuran referensi tertulis pada gambar denah1.png (sebagai acuan evaluasi manual Bagian 13)
DENAH1_REFERENCE_ROOMS: List[Dict[str, Any]] = [
    {"nama_acuan": "Chambre 2 (Kamar Tidur 2)", "luas_acuan_m2": 10.20, "target_centroid": (738, 330)},
    {"nama_acuan": "Chambre 3 (Kamar Tidur 3)", "luas_acuan_m2": 10.02, "target_centroid": (1030, 325)},
    {"nama_acuan": "Bains (Kamar Mandi)", "luas_acuan_m2": 6.31, "target_centroid": (740, 540)},
    {"nama_acuan": "Bureau (Ruang Kerja)", "luas_acuan_m2": 7.80, "target_centroid": (1145, 525)},
    {"nama_acuan": "Chambre 1 (Kamar Tidur 1)", "luas_acuan_m2": 10.95, "target_centroid": (697, 776)},
    {"nama_acuan": "Dressing (Ruang Ganti)", "luas_acuan_m2": 2.94, "target_centroid": (871, 869)},
    {"nama_acuan": "Salon / Sejour / Labo", "luas_acuan_m2": 44.38, "target_centroid": (1240, 700)},
    {"nama_acuan": "Garage (Garasi)", "luas_acuan_m2": 32.88, "target_centroid": (815, 1175)},
]


def inspect_dataset(raw_dir: str = os.path.join("dataset", "raw")) -> Dict[str, Any]:
    """
    Memeriksa kelengkapan dan karakteristik dataset Floor Plan (Bagian 4):
    Menghitung jumlah gambar di folder X, jumlah anotasi di folder Y,
    variasi resolusi gambar, serta contoh pasangan gambar dan anotasi.
    """
    x_dir = os.path.join(raw_dir, "X")
    y_dir = os.path.join(raw_dir, "Y")

    if not os.path.exists(x_dir):
        return {
            "available": False,
            "num_images_x": 0,
            "num_annotations_y": 0,
            "summary_df": pd.DataFrame(),
        }

    x_files = sorted([f for f in os.listdir(x_dir) if f.lower().endswith(".png")])
    y_files = sorted([f for f in os.listdir(y_dir) if f.lower().endswith(".png")]) if os.path.exists(y_dir) else []

    sample_records: List[Dict[str, Any]] = []
    for fname in x_files[:15]:
        base_id = os.path.splitext(fname)[0]
        x_path = os.path.join(x_dir, fname)
        y_expected = f"{base_id} (2).png"
        y_path = os.path.join(y_dir, y_expected)
        has_annot = os.path.exists(y_path)

        img = cv2.imread(x_path)
        if img is not None:
            h, w, c = img.shape
            sample_records.append(
                {
                    "File Denah (X)": fname,
                    "File Anotasi (Y)": y_expected if has_annot else "Tidak Ada",
                    "Resolusi (Px)": f"{w} × {h}",
                    "Channel": c,
                    "Status Anotasi": "Tersedia" if has_annot else "Tidak Lengkap",
                }
            )

    return {
        "available": True,
        "num_images_x": len(x_files),
        "num_annotations_y": len(y_files),
        "sample_df": pd.DataFrame(sample_records),
    }


def evaluate_against_reference_areas(
    measured_rooms: List[Dict[str, Any]],
    reference_rooms: List[Dict[str, Any]] = DENAH1_REFERENCE_ROOMS,
) -> Dict[str, Any]:
    """
    Membandingkan luas ruangan hasil deteksi OpenCV dengan ukuran referensi tertulis
    pada denah untuk mengetahui tingkat kesalahan pengukuran (Bagian 13).
    """
    rows: List[Dict[str, Any]] = []
    for ref in reference_rooms:
        tx, ty = ref["target_centroid"]
        best_room = None
        best_dist = float("inf")
        for r in measured_rooms:
            cx, cy = r["centroid_px"]
            dist = float(np.hypot(cx - tx, cy - ty))
            if dist < best_dist:
                best_dist = dist
                best_room = r

        if best_room is not None and best_dist < 380:
            det_area = float(best_room["luas_m2"])
            ref_area = float(ref["luas_acuan_m2"])
            err_abs = abs(det_area - ref_area)
            err_pct = (err_abs / ref_area) * 100.0 if ref_area > 0 else 0.0
            acc_pct = max(0.0, 100.0 - err_pct)
            rows.append(
                {
                    "Ruangan Acuan (Denah)": ref["nama_acuan"],
                    "ID Terdeteksi": f"R{best_room['id_ruangan']}",
                    "Luas Referensi (m²)": round(ref_area, 2),
                    "Luas Hasil OpenCV (m²)": round(det_area, 2),
                    "Selisih Absolut (m²)": round(err_abs, 2),
                    "Tingkat Kesalahan (%)": round(err_pct, 2),
                    "Akurasi (%)": round(acc_pct, 2),
                }
            )

    eval_df = pd.DataFrame(rows)
    if not eval_df.empty:
        mean_error_pct = float(eval_df["Tingkat Kesalahan (%)"].mean())
        mean_accuracy_pct = float(eval_df["Akurasi (%)"].mean())
        total_ref_m2 = float(eval_df["Luas Referensi (m²)"].sum())
        total_det_m2 = float(eval_df["Luas Hasil OpenCV (m²)"].sum())
        total_err_pct = abs(total_det_m2 - total_ref_m2) / total_ref_m2 * 100.0 if total_ref_m2 > 0 else 0.0
    else:
        mean_error_pct = 0.0
        mean_accuracy_pct = 0.0
        total_ref_m2 = 0.0
        total_det_m2 = 0.0
        total_err_pct = 0.0

    return {
        "eval_df": eval_df,
        "mean_error_pct": round(mean_error_pct, 2),
        "mean_accuracy_pct": round(mean_accuracy_pct, 2),
        "total_ref_m2": round(total_ref_m2, 2),
        "total_det_m2": round(total_det_m2, 2),
        "total_error_pct": round(total_err_pct, 2),
        "total_accuracy_pct": round(max(0.0, 100.0 - total_err_pct), 2),
    }


def evaluate_against_ground_truth_mask(
    detected_rooms_mask: np.ndarray,
    gt_mask_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Mengevaluasi mask ruangan hasil OpenCV terhadap mask Ground Truth dari dataset/raw/Y.
    Pada dataset CVC-FP (asutoshprad/floor-plan-dataset), warna ruangan di Y memiliki BGR sekitar (142, 142, 0).
    """
    if not os.path.exists(gt_mask_path):
        return None

    gt_bgr = cv2.imread(gt_mask_path)
    if gt_bgr is None:
        return None

    if gt_bgr.shape[:2] != detected_rooms_mask.shape[:2]:
        gt_bgr = cv2.resize(gt_bgr, (detected_rooms_mask.shape[1], detected_rooms_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ekstrak piksel ruangan pada ground truth Y:
    # Sebagian besar mask menggunakan warna teal BGR~(142, 142, 0), sebagian lainnya BGR~(70, 142, 255)
    b, g, r = cv2.split(gt_bgr)
    gt_room_bool = (b > 110) & (b < 175) & (g > 110) & (g < 175) & (r < 40)
    if gt_room_bool.sum() < 0.02 * (gt_bgr.shape[0] * gt_bgr.shape[1]):
        gt_room_bool = (b > 40) & (b < 100) & (g > 110) & (g < 175) & (r > 210)
    pred_room_bool = detected_rooms_mask > 0

    intersection = float(np.logical_and(gt_room_bool, pred_room_bool).sum())
    union = float(np.logical_or(gt_room_bool, pred_room_bool).sum())
    gt_pixels = float(gt_room_bool.sum())
    pred_pixels = float(pred_room_bool.sum())

    iou = (intersection / union * 100.0) if union > 0 else 0.0
    precision = (intersection / pred_pixels * 100.0) if pred_pixels > 0 else 0.0
    recall = (intersection / gt_pixels * 100.0) if gt_pixels > 0 else 0.0
    area_ratio_acc = (1.0 - abs(pred_pixels - gt_pixels) / gt_pixels) * 100.0 if gt_pixels > 0 else 0.0

    return {
        "gt_room_pixels": int(gt_pixels),
        "pred_room_pixels": int(pred_pixels),
        "iou_percent": round(iou, 2),
        "precision_percent": round(precision, 2),
        "recall_percent": round(recall, 2),
        "area_similarity_percent": round(max(0.0, area_ratio_acc), 2),
        "gt_room_mask": (gt_room_bool.astype(np.uint8) * 255),
    }


def run_multi_floorplan_benchmark(
    train_dir: str = os.path.join("dataset", "train"),
    raw_y_dir: str = os.path.join("dataset", "raw", "Y"),
) -> pd.DataFrame:
    """
    Menguji sistem pada beberapa gambar denah berbeda (denah1.png, denah2.png, denah3.png)
    sesuai Bagian 13 (Pengujian dan Evaluasi Sistem).
    """
    benchmark_cases = [
        ("denah1.png", "1 (2).png", 80.5, "Denah 1 (Asli X/1.png - 1950×1867)"),
        ("denah2.png", "2 (2).png", 54.3, "Denah 2 (Asli X/2.png - 1318×1229)"),
        ("denah3.png", "3 (2).png", 60.7, "Denah 3 (Asli X/3.png - 1298×1506)"),
        ("denah4.png", "4 (2).png", 52.9, "Denah 4 (Asli X/4.png - 1027×1369)"),
        ("denah5.png", "5 (2).png", 60.3, "Denah 5 (Asli X/5.png - 1479×1486)"),
        ("denah6.png", "8 (2).png", 60.0, "Denah 6 (Asli X/8.png - 1621×1349)"),
        ("denah7.png", "9 (2).png", 90.4, "Denah 7 (Asli X/9.png - 1943×1987)"),
        ("denah8.png", "10 (2).png", 115.0, "Denah 8 (Asli X/10.png - 3508×2480)"),
        ("denah9.png", "14 (2).png", 115.0, "Denah 9 (Asli X/14.png - 2480×3508)"),
        ("denah10.png", "13 (2).png", 115.0, "Denah 10 (Asli X/13.png - 2479×3508)"),
    ]

    records: List[Dict[str, Any]] = []
    for fname, gt_fname, scale_px_m, desc in benchmark_cases:
        fpath = os.path.join(train_dir, fname)
        if not os.path.exists(fpath):
            continue

        prep = preprocess_floorplan(fpath)
        walls = detect_walls_hough(prep["walls_unbridged"], scale_factor=prep["scale_factor"])
        rooms = detect_rooms_floodfill_contours(prep["walls_clean"], scale_factor=prep["scale_factor"])
        meas = measure_building_dimensions(rooms, walls, pixels_per_meter=scale_px_m)

        gt_path = os.path.join(raw_y_dir, gt_fname)
        gt_eval = evaluate_against_ground_truth_mask(rooms["rooms_mask"], gt_path)

        records.append(
            {
                "Nama File": fname,
                "Karakteristik Gambar": desc,
                "Skala (px/m)": scale_px_m,
                "Jumlah Ruangan Terdeteksi": meas["num_rooms"],
                "Total Luas Lantai (m²)": meas["total_floor_area_m2"],
                "Panjang Dinding (m)": meas["total_wall_length_m"],
                "IoU vs Ground Truth (%)": gt_eval["iou_percent"] if gt_eval else None,
                "Kemiripan Luas vs GT (%)": gt_eval["area_similarity_percent"] if gt_eval else None,
            }
        )

    return pd.DataFrame(records)

