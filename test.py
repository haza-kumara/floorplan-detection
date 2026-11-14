"""
Unit Test & Verifikasi Sistem SmartPlan (Tahap 5 - Tahap 13)
Menguji ketersediaan library serta ketepatan perhitungan skala, deteksi OpenCV,
kebutuhan material (keramik & cat), dan estimasi biaya.
"""

import os
import cv2
import numpy as np
import matplotlib
import pandas as pd
from PIL import Image
import streamlit

from src.preprocessing import preprocess_floorplan
from src.detection import detect_walls_hough, detect_rooms_floodfill_contours
from src.measurement import calculate_scale, measure_building_dimensions
from src.material_calculator import calculate_tile_needs, calculate_paint_needs
from src.cost_estimator import estimate_material_costs


def run_tests() -> None:
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Pandas:", pd.__version__)
    print("Streamlit:", streamlit.__version__)
    print("Semua library berhasil di-import!\n")

    # 1. Uji rumus skala sesuai contoh Bagian 8: 200 piksel = 4 meter -> 50 piksel/meter
    scale_px_m = calculate_scale(200.0, 4.0)
    assert abs(scale_px_m - 50.0) < 1e-6, f"Skala seharusnya 50.0, diperoleh {scale_px_m}"
    print("[OK] Uji Skala Referensi (200 px / 4 m = 50.0 px/m) berhasil.")

    # 2. Uji rumus kebutuhan keramik sesuai Bagian 9 (luas 100 m2, keramik 40x40 cm = 0.16 m2, waste 10%)
    # 100 / 0.16 = 625 keping dasar + 10% (62.5) = 687.5 -> 688 keping
    tile_res = calculate_tile_needs(total_floor_area_m2=100.0, tile_length_cm=40.0, tile_width_cm=40.0, waste_percent=10.0)
    assert tile_res["base_tiles_count"] == 625
    assert tile_res["total_tiles_count"] == 688
    print(f"[OK] Uji Kalkulator Keramik (100 m², 40x40 cm, +10% waste = {tile_res['total_tiles_count']} keping) berhasil.")

    # 3. Uji rumus kebutuhan cat dinding sesuai Bagian 9 (panjang dinding 50m, tinggi 3m = 150m2, 2 lapis = 300m2, daya sebar 10m2/L = 30 L)
    paint_res = calculate_paint_needs(
        total_wall_length_m=50.0,
        wall_height_m=3.0,
        paint_coverage_m2_per_liter=10.0,
        paint_coats=2,
        wall_sides=1,
    )
    assert paint_res["paint_liters_needed"] == 30
    print(f"[OK] Uji Kalkulator Cat Dinding (50m × 3m, 2 lapis, 10 m²/L = {paint_res['paint_liters_needed']} liter) berhasil.")

    # 4. Uji Estimasi Biaya sesuai Bagian 10
    cost_res = estimate_material_costs(
        {"tiles": tile_res, "paint": paint_res, "include_structural": False},
        tile_unit_price=10000.0,
        paint_unit_price_per_liter=50000.0,
    )
    expected_total = (688 * 10000.0) + (30 * 50000.0)
    assert abs(cost_res["total_cost_idr"] - expected_total) < 1e-3
    print(f"[OK] Uji Estimasi Biaya ({cost_res['total_cost_formatted']}) berhasil.")

    # 5. Uji Pipeline OpenCV pada dataset/train/denah1.png
    sample_img = os.path.join("dataset", "train", "denah1.png")
    if os.path.exists(sample_img):
        prep = preprocess_floorplan(sample_img)
        walls = detect_walls_hough(prep["walls_unbridged"], scale_factor=prep["scale_factor"])
        rooms = detect_rooms_floodfill_contours(prep["walls_clean"], scale_factor=prep["scale_factor"])
        meas = measure_building_dimensions(rooms, walls, pixels_per_meter=80.5)
        assert meas["num_rooms"] >= 8, "Jumlah ruangan terdeteksi pada denah1 seharusnya >= 8"
        assert 100.0 <= meas["total_floor_area_m2"] <= 135.0, "Total luas lantai denah1 seharusnya di kisaran 100-135 m2"
        print(
            f"[OK] Uji Pipeline OpenCV pada denah1.png berhasil: "
            f"{meas['num_rooms']} ruangan, Total Luas = {meas['total_floor_area_m2']:.2f} m², "
            f"Panjang Dinding = {meas['total_wall_length_m']:.2f} m."
        )

    print("\nSeluruh pengujian unit dan integrasi SmartPlan BERHASIL!")


if __name__ == "__main__":
    run_tests()
