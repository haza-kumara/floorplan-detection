"""
Tahap 8: Perhitungan Dimensi dan Luas Bangunan
Mengonversi ukuran piksel hasil deteksi OpenCV menjadi satuan meter (m) dan meter persegi (m2)
berdasarkan skala gambar atau ukuran referensi manual (misal: 200 piksel = 4 meter -> 50 piksel/meter).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# Preset kalibrasi skala referensi untuk gambar denah pada dataset/train (denah1 s/d denah10)
DATASET_SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "denah1.png": {
        "ref_pixels": 269.7,
        "ref_meters": 3.35,
        "pixels_per_meter": 80.5,
        "description": "Denah 1 (Asli: X/1.png) - Lebar Chambre 2 (269.7 px = 3.35 m -> 80.5 px/m)",
    },
    "denah2.png": {
        "ref_pixels": 182.0,
        "ref_meters": 3.35,
        "pixels_per_meter": 54.3,
        "description": "Denah 2 (Asli: X/2.png) - Lebar Chambre 2 resolusi sedang (182 px = 3.35 m -> 54.3 px/m)",
    },
    "denah3.png": {
        "ref_pixels": 215.0,
        "ref_meters": 3.54,
        "pixels_per_meter": 60.7,
        "description": "Denah 3 (Asli: X/3.png) - Lebar Chambre 3 (215 px = 3.54 m -> 60.7 px/m)",
    },
    "denah4.png": {
        "ref_pixels": 175.0,
        "ref_meters": 3.31,
        "pixels_per_meter": 52.9,
        "description": "Denah 4 (Asli: X/4.png) - Lebar Chambre 1 (175 px = 3.31 m -> 52.9 px/m)",
    },
    "denah5.png": {
        "ref_pixels": 238.0,
        "ref_meters": 3.95,
        "pixels_per_meter": 60.3,
        "description": "Denah 5 (Asli: X/5.png) - Lebar Chambre 1 (238 px = 3.95 m -> 60.3 px/m)",
    },
    "denah6.png": {
        "ref_pixels": 210.0,
        "ref_meters": 3.50,
        "pixels_per_meter": 60.0,
        "description": "Denah 6 (Asli: X/8.png) - Lebar ruangan utama (210 px = 3.50 m -> 60.0 px/m)",
    },
    "denah7.png": {
        "ref_pixels": 320.0,
        "ref_meters": 3.54,
        "pixels_per_meter": 90.4,
        "description": "Denah 7 (Asli: X/9.png) - Lebar Chambre 3 resolusi tinggi (320 px = 3.54 m -> 90.4 px/m)",
    },
    "denah8.png": {
        "ref_pixels": 460.0,
        "ref_meters": 4.00,
        "pixels_per_meter": 115.0,
        "description": "Denah 8 (Asli: X/10.png) - Denah dinding solid 1:100 (460 px = 4.00 m -> 115.0 px/m)",
    },
    "denah9.png": {
        "ref_pixels": 460.0,
        "ref_meters": 4.00,
        "pixels_per_meter": 115.0,
        "description": "Denah 9 (Asli: X/14.png) - Denah dinding solid 1:100 B (460 px = 4.00 m -> 115.0 px/m)",
    },
    "denah10.png": {
        "ref_pixels": 460.0,
        "ref_meters": 4.00,
        "pixels_per_meter": 115.0,
        "description": "Denah 10 (Asli: X/13.png) - Denah dinding solid vertikal (460 px = 4.00 m -> 115.0 px/m)",
    },
    "contoh_proposal": {
        "ref_pixels": 200.0,
        "ref_meters": 4.0,
        "pixels_per_meter": 50.0,
        "description": "Contoh Deskripsi Projek (Dinding 200 piksel = 4 meter -> 50 px/m)",
    },
}


def calculate_scale(ref_pixels: float, ref_meters: float) -> float:
    """
    Menghitung skala gambar dalam satuan piksel per meter (px/m).
    Contoh: 200 piksel / 4 meter = 50 piksel/meter.
    """
    if ref_pixels <= 0 or ref_meters <= 0:
        raise ValueError("Nilai piksel referensi dan ukuran meter sebenarnya harus lebih besar dari 0.")
    return float(ref_pixels) / float(ref_meters)


def measure_building_dimensions(
    room_result: Dict[str, Any],
    wall_result: Dict[str, Any],
    pixels_per_meter: float = 80.5,
    area_method: str = "contour",
) -> Dict[str, Any]:
    """
    Menghitung dimensi setiap ruangan (panjang, lebar, keliling, luas dalam m2),
    total luas lantai bangunan (m2), serta panjang total dinding (m).

    Parameter:
    - room_result: Hasil deteksi ruangan dari src.detection.detect_rooms_floodfill_contours
    - wall_result: Hasil deteksi dinding dari src.detection.detect_walls_hough
    - pixels_per_meter: Skala konversi (piksel per 1 meter)
    - area_method: 'contour' (luas kontur piksel akurat) atau 'bbox' (panjang x lebar bounding box)
    """
    if pixels_per_meter <= 0:
        raise ValueError("Skala pixels_per_meter harus lebih besar dari 0.")

    px_to_m = 1.0 / float(pixels_per_meter)
    px2_to_m2 = px_to_m ** 2

    measured_rooms: List[Dict[str, Any]] = []
    for room in room_result["rooms"]:
        w_px = float(room["width_px"])
        h_px = float(room["height_px"])
        length_px = max(w_px, h_px)
        width_px = min(w_px, h_px)

        length_m = length_px * px_to_m
        width_m = width_px * px_to_m
        perimeter_m = float(room["perimeter_px"]) * px_to_m

        area_contour_m2 = float(room["area_px"]) * px2_to_m2
        area_bbox_m2 = length_m * width_m

        chosen_area_m2 = area_bbox_m2 if area_method == "bbox" else area_contour_m2

        measured_rooms.append(
            {
                "id_ruangan": room["room_id"],
                "nama_ruangan": room.get("custom_name", room["name"]),
                "panjang_px": round(length_px, 1),
                "lebar_px": round(width_px, 1),
                "luas_px2": round(float(room["area_px"]), 1),
                "panjang_m": round(length_m, 2),
                "lebar_m": round(width_m, 2),
                "keliling_m": round(perimeter_m, 2),
                "luas_kontur_m2": round(area_contour_m2, 2),
                "luas_pxl_m2": round(area_bbox_m2, 2),
                "luas_m2": round(chosen_area_m2, 2),
                "bbox_px": room["bbox_px"],
                "centroid_px": room["centroid_px"],
            }
        )

    total_floor_area_m2 = float(np.sum([r["luas_m2"] for r in measured_rooms]))
    total_contour_area_m2 = float(np.sum([r["luas_kontur_m2"] for r in measured_rooms]))
    total_bbox_area_m2 = float(np.sum([r["luas_pxl_m2"] for r in measured_rooms]))
    total_room_perimeter_m = float(np.sum([r["keliling_m"] for r in measured_rooms]))

    # Perhitungan panjang dinding berdasarkan deteksi garis Hough (Tahap 7 & 8)
    total_wall_length_px = float(wall_result["total_wall_length_px"])
    horizontal_wall_px = float(wall_result["horizontal_length_px"])
    vertical_wall_px = float(wall_result["vertical_length_px"])

    total_wall_length_m = total_wall_length_px * px_to_m
    horizontal_wall_m = horizontal_wall_px * px_to_m
    vertical_wall_m = vertical_wall_px * px_to_m

    rooms_df = pd.DataFrame(
        [
            {
                "No": r["id_ruangan"],
                "Nama Ruangan": r["nama_ruangan"],
                "Panjang (px)": int(round(r["panjang_px"])),
                "Lebar (px)": int(round(r["lebar_px"])),
                "Luas Piksel (px²)": int(round(r["luas_px2"])),
                "Panjang (m)": r["panjang_m"],
                "Lebar (m)": r["lebar_m"],
                "Luas Kontur (m²)": r["luas_kontur_m2"],
                "Luas P×L (m²)": r["luas_pxl_m2"],
                "Luas Digunakan (m²)": r["luas_m2"],
            }
            for r in measured_rooms
        ]
    )

    disclaimer = (
        "Catatan: Hasil perhitungan dimensi dan luas merupakan estimasi awal berbasis "
        "Computer Vision yang bergantung pada ketepatan deteksi kontur/garis dinding "
        "serta skala ukuran referensi yang diberikan."
    )

    return {
        "pixels_per_meter": round(float(pixels_per_meter), 2),
        "area_method": area_method,
        "num_rooms": len(measured_rooms),
        "rooms": measured_rooms,
        "rooms_df": rooms_df,
        "total_floor_area_m2": round(total_floor_area_m2, 2),
        "total_contour_area_m2": round(total_contour_area_m2, 2),
        "total_bbox_area_m2": round(total_bbox_area_m2, 2),
        "total_room_perimeter_m": round(total_room_perimeter_m, 2),
        "total_wall_length_px": round(total_wall_length_px, 1),
        "total_wall_length_m": round(total_wall_length_m, 2),
        "horizontal_wall_m": round(horizontal_wall_m, 2),
        "vertical_wall_m": round(vertical_wall_m, 2),
        "disclaimer": disclaimer,
    }

