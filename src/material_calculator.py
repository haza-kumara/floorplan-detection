"""
Tahap 9: Perhitungan Kebutuhan Material Bangunan
Menghitung perkiraan kebutuhan material utama:
1. Keramik Lantai (berdasarkan luas lantai, ukuran keramik, dan faktor cadangan/waste 10%)
2. Cat Dinding (berdasarkan panjang dinding, tinggi dinding, pengurangan pintu/jendela,
   daya sebar cat m2/liter, dan jumlah lapisan pengecatan)
3. Material Konstruksi Dinding Lanjutan Opsional (Batu Bata, Semen, Pasir).
"""

import math
from typing import Any, Dict, List, Optional
import pandas as pd


TILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "30x30 cm": {"length_cm": 30.0, "width_cm": 30.0, "tiles_per_box": 11, "material_id": "keramik_30x30"},
    "40x40 cm": {"length_cm": 40.0, "width_cm": 40.0, "tiles_per_box": 6, "material_id": "keramik_40x40"},
    "50x50 cm": {"length_cm": 50.0, "width_cm": 50.0, "tiles_per_box": 4, "material_id": "keramik_50x50"},
    "60x60 cm": {"length_cm": 60.0, "width_cm": 60.0, "tiles_per_box": 4, "material_id": "keramik_60x60"},
    "80x80 cm": {"length_cm": 80.0, "width_cm": 80.0, "tiles_per_box": 3, "material_id": "keramik_80x80"},
}


def calculate_tile_needs(
    total_floor_area_m2: float,
    tile_length_cm: float = 40.0,
    tile_width_cm: float = 40.0,
    waste_percent: float = 10.0,
    tiles_per_box: int = 6,
    rooms: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Menghitung kebutuhan keramik lantai berdasarkan luas lantai bangunan,
    ukuran keramik (cm x cm), serta faktor cadangan (waste factor, default 10%).
    """
    if tile_length_cm <= 0 or tile_width_cm <= 0:
        raise ValueError("Ukuran panjang dan lebar keramik harus lebih besar dari 0 cm.")

    tile_area_m2 = (tile_length_cm / 100.0) * (tile_width_cm / 100.0)
    raw_tiles = total_floor_area_m2 / tile_area_m2 if total_floor_area_m2 > 0 else 0.0
    waste_multiplier = 1.0 + (max(0.0, float(waste_percent)) / 100.0)
    total_tiles_exact = raw_tiles * waste_multiplier
    total_tiles_ceil = int(math.ceil(total_tiles_exact))
    waste_tiles = max(0, total_tiles_ceil - int(math.ceil(raw_tiles)))

    boxes_needed = int(math.ceil(total_tiles_ceil / max(1, int(tiles_per_box))))

    room_tile_breakdown: List[Dict[str, Any]] = []
    if rooms:
        for r in rooms:
            r_area = float(r["luas_m2"])
            r_raw = r_area / tile_area_m2
            r_total = int(math.ceil(r_raw * waste_multiplier))
            r_boxes = int(math.ceil(r_total / max(1, int(tiles_per_box))))
            room_tile_breakdown.append(
                {
                    "No": r["id_ruangan"],
                    "Nama Ruangan": r["nama_ruangan"],
                    "Luas Lantai (m²)": round(r_area, 2),
                    "Kebutuhan Dasar (keping)": int(math.ceil(r_raw)),
                    f"Total + Cadangan {waste_percent:.0f}% (keping)": r_total,
                    "Estimasi Dus": r_boxes,
                }
            )

    return {
        "tile_size_label": f"{int(tile_length_cm)}x{int(tile_width_cm)} cm",
        "tile_length_cm": float(tile_length_cm),
        "tile_width_cm": float(tile_width_cm),
        "tile_area_m2": round(tile_area_m2, 4),
        "total_floor_area_m2": round(float(total_floor_area_m2), 2),
        "waste_percent": float(waste_percent),
        "base_tiles_count": int(math.ceil(raw_tiles)),
        "waste_tiles_count": waste_tiles,
        "total_tiles_count": total_tiles_ceil,
        "tiles_per_box": int(tiles_per_box),
        "boxes_needed": boxes_needed,
        "room_breakdown_df": pd.DataFrame(room_tile_breakdown) if room_tile_breakdown else pd.DataFrame(),
    }


def calculate_paint_needs(
    total_wall_length_m: float,
    wall_height_m: float = 3.0,
    paint_coverage_m2_per_liter: float = 10.0,
    paint_coats: int = 2,
    wall_sides: int = 1,
    door_count: int = 0,
    window_count: int = 0,
    door_area_m2: float = 1.89,
    window_area_m2: float = 1.20,
) -> Dict[str, Any]:
    """
    Menghitung kebutuhan cat dinding berdasarkan:
    - Panjang total dinding (m) dan tinggi dinding (m)
    - Jumlah sisi pengecatan dinding (1 sisi atau 2 sisi)
    - Pengurangan luas pintu dan jendela (opsional)
    - Daya sebar cat (m2 per liter, default 10 m2/liter)
    - Jumlah lapisan pengecatan (default 2 lapis).
    """
    if wall_height_m <= 0:
        raise ValueError("Tinggi dinding harus lebih besar dari 0 meter.")
    if paint_coverage_m2_per_liter <= 0:
        raise ValueError("Daya sebar cat harus lebih besar dari 0 m2/liter.")

    gross_wall_area_m2 = float(total_wall_length_m) * float(wall_height_m) * max(1, int(wall_sides))
    opening_deduction_m2 = (
        (max(0, int(door_count)) * float(door_area_m2))
        + (max(0, int(window_count)) * float(window_area_m2))
    ) * max(1, int(wall_sides))

    net_wall_area_m2 = max(0.0, gross_wall_area_m2 - opening_deduction_m2)
    total_painted_area_m2 = net_wall_area_m2 * max(1, int(paint_coats))

    liters_exact = total_painted_area_m2 / float(paint_coverage_m2_per_liter)
    liters_ceil = int(math.ceil(liters_exact))

    cans_2_5l = int(math.ceil(liters_exact / 2.5))
    pails_20l = int(math.ceil(liters_exact / 20.0))

    return {
        "total_wall_length_m": round(float(total_wall_length_m), 2),
        "wall_height_m": round(float(wall_height_m), 2),
        "wall_sides": int(wall_sides),
        "gross_wall_area_m2": round(gross_wall_area_m2, 2),
        "opening_deduction_m2": round(opening_deduction_m2, 2),
        "net_wall_area_m2": round(net_wall_area_m2, 2),
        "paint_coats": int(paint_coats),
        "paint_coverage_m2_per_liter": round(float(paint_coverage_m2_per_liter), 2),
        "total_painted_area_m2": round(total_painted_area_m2, 2),
        "paint_liters_exact": round(liters_exact, 2),
        "paint_liters_needed": liters_ceil,
        "cans_2_5l_equivalent": cans_2_5l,
        "pails_20l_equivalent": pails_20l,
    }


def calculate_structural_wall_materials(
    total_wall_length_m: float,
    wall_height_m: float = 3.0,
    door_count: int = 0,
    window_count: int = 0,
    door_area_m2: float = 1.89,
    window_area_m2: float = 1.20,
    bricks_per_m2: float = 70.0,
    cement_sacks_per_m2: float = 0.45,
    sand_m3_per_m2: float = 0.045,
) -> Dict[str, Any]:
    """
    Perhitungan material konstruksi dinding lanjutan (Bagian 9 Paragraf Akhir):
    Menghitung estimasi kebutuhan batu bata, semen, dan pasir berdasarkan luas dinding satu sisi.
    """
    single_side_gross_m2 = float(total_wall_length_m) * float(wall_height_m)
    deduction_m2 = (max(0, int(door_count)) * float(door_area_m2)) + (
        max(0, int(window_count)) * float(window_area_m2)
    )
    wall_surface_m2 = max(0.0, single_side_gross_m2 - deduction_m2)

    bricks_needed = int(math.ceil(wall_surface_m2 * bricks_per_m2))
    cement_sacks_needed = int(math.ceil(wall_surface_m2 * cement_sacks_per_m2))
    sand_m3_needed = round(wall_surface_m2 * sand_m3_per_m2, 2)

    return {
        "wall_surface_m2": round(wall_surface_m2, 2),
        "bricks_needed": bricks_needed,
        "cement_sacks_needed": cement_sacks_needed,
        "sand_m3_needed": sand_m3_needed,
    }


def calculate_all_materials(
    measurement_result: Dict[str, Any],
    tile_length_cm: float = 40.0,
    tile_width_cm: float = 40.0,
    tile_waste_percent: float = 10.0,
    tiles_per_box: int = 6,
    wall_height_m: float = 3.0,
    paint_coverage_m2_per_liter: float = 10.0,
    paint_coats: int = 2,
    wall_sides: int = 1,
    door_count: int = 0,
    window_count: int = 0,
    include_structural: bool = False,
) -> Dict[str, Any]:
    """Menjalankan perhitungan lengkap kebutuhan keramik lantai, cat dinding, dan material opsional."""
    tiles = calculate_tile_needs(
        total_floor_area_m2=measurement_result["total_floor_area_m2"],
        tile_length_cm=tile_length_cm,
        tile_width_cm=tile_width_cm,
        waste_percent=tile_waste_percent,
        tiles_per_box=tiles_per_box,
        rooms=measurement_result.get("rooms", []),
    )

    paint = calculate_paint_needs(
        total_wall_length_m=measurement_result["total_wall_length_m"],
        wall_height_m=wall_height_m,
        paint_coverage_m2_per_liter=paint_coverage_m2_per_liter,
        paint_coats=paint_coats,
        wall_sides=wall_sides,
        door_count=door_count,
        window_count=window_count,
    )

    structural = None
    if include_structural:
        structural = calculate_structural_wall_materials(
            total_wall_length_m=measurement_result["total_wall_length_m"],
            wall_height_m=wall_height_m,
            door_count=door_count,
            window_count=window_count,
        )

    return {
        "tiles": tiles,
        "paint": paint,
        "structural": structural,
        "include_structural": include_structural,
    }

