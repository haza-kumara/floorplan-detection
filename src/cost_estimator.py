"""
Tahap 10: Perhitungan Estimasi Biaya Material Bangunan
Mengelola tabel harga material menggunakan Pandas dan menghitung estimasi biaya
setiap material serta total biaya keseluruhan.
"""

import os
from typing import Any, Dict, List, Optional
import pandas as pd


DEFAULT_PRICE_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "material_prices.csv",
)


def load_material_prices(csv_path: str = DEFAULT_PRICE_CSV) -> pd.DataFrame:
    """Membaca tabel harga satuan material dari file CSV menggunakan Pandas."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # Fallback tabel default apabila file CSV belum tersedia
    default_rows = [
        ("Keramik Lantai", "keramik_30x30", "Keramik Lantai Polos 30x30 cm", "Ukuran 30x30 cm (0.09 m2/keping)", "keping", 1, 5500),
        ("Keramik Lantai", "keramik_40x40", "Keramik Lantai Standar 40x40 cm", "Ukuran 40x40 cm (0.16 m2/keping)", "keping", 1, 10500),
        ("Keramik Lantai", "keramik_50x50", "Keramik Lantai Medium 50x50 cm", "Ukuran 50x50 cm (0.25 m2/keping)", "keping", 1, 18500),
        ("Keramik Lantai", "keramik_60x60", "Granit / Keramik 60x60 cm", "Ukuran 60x60 cm (0.36 m2/keping)", "keping", 1, 32000),
        ("Keramik Lantai", "keramik_80x80", "Granit Premium 80x80 cm", "Ukuran 80x80 cm (0.64 m2/keping)", "keping", 1, 75000),
        ("Cat Dinding", "cat_standar_liter", "Cat Dinding Interior Standar (Per Liter)", "Daya sebar ~10 m2/liter", "liter", 1, 45000),
        ("Cat Dinding", "cat_premium_liter", "Cat Dinding Interior Premium (Per Liter)", "Daya sebar ~12 m2/liter", "liter", 1, 75000),
        ("Cat Dinding", "cat_ekonomis_liter", "Cat Dinding Ekonomis (Per Liter)", "Daya sebar ~8 m2/liter", "liter", 1, 28000),
        ("Material Konstruksi Lanjutan", "bata_merah", "Batu Bata Merah Press", "Kebutuhan ~70 buah/m2 dinding", "buah", 1, 900),
        ("Material Konstruksi Lanjutan", "semen_50kg", "Semen Portland (Sak 50 kg)", "Kebutuhan pasang & plester ~0.45 sak/m2", "sak", 50, 68000),
        ("Material Konstruksi Lanjutan", "pasir_m3", "Pasir Pasang / Plester", "Kebutuhan pasang & plester ~0.045 m3/m2", "m3", 1, 280000),
    ]
    return pd.DataFrame(
        default_rows,
        columns=["kategori", "id_material", "nama_material", "spesifikasi", "satuan", "isi_per_satuan", "harga_satuan"],
    )


def save_material_prices(df: pd.DataFrame, csv_path: str = DEFAULT_PRICE_CSV) -> None:
    """Menyimpan pembaruan tabel harga satuan material ke file CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


def format_rupiah(value: float) -> str:
    """Memformat angka menjadi string mata uang Rupiah (contoh: Rp 12.500.000)."""
    return f"Rp {int(round(value)):,}".replace(",", ".")


def estimate_material_costs(
    material_result: Dict[str, Any],
    tile_unit_price: float = 10500.0,
    paint_unit_price_per_liter: float = 45000.0,
    tile_material_name: str = "Keramik Lantai",
    paint_material_name: str = "Cat Dinding Interior",
    brick_unit_price: float = 900.0,
    cement_unit_price: float = 68000.0,
    sand_unit_price: float = 280000.0,
) -> Dict[str, Any]:
    """
    Menghitung estimasi biaya setiap jenis material (jumlah kebutuhan x harga satuan)
    serta menjumlahkan total estimasi biaya menggunakan Pandas DataFrame.
    """
    tiles = material_result["tiles"]
    paint = material_result["paint"]
    structural = material_result.get("structural")

    tile_qty = int(tiles["total_tiles_count"])
    tile_cost = float(tile_qty) * float(tile_unit_price)

    paint_qty = int(paint["paint_liters_needed"])
    paint_cost = float(paint_qty) * float(paint_unit_price_per_liter)

    items: List[Dict[str, Any]] = [
        {
            "Nama Material": f"{tile_material_name} ({tiles['tile_size_label']})",
            "Spesifikasi / Keterangan": (
                f"Luas lantai {tiles['total_floor_area_m2']:.2f} m², "
                f"termasuk cadangan {tiles['waste_percent']:.0f}% (~{tiles['boxes_needed']} dus)"
            ),
            "Jumlah Kebutuhan": tile_qty,
            "Satuan": "keping",
            "Harga Satuan (Rp)": round(float(tile_unit_price), 0),
            "Total Biaya (Rp)": round(tile_cost, 0),
        },
        {
            "Nama Material": paint_material_name,
            "Spesifikasi / Keterangan": (
                f"Luas dinding bersih {paint['net_wall_area_m2']:.2f} m², "
                f"{paint['paint_coats']} lapis, daya sebar {paint['paint_coverage_m2_per_liter']:.1f} m²/L "
                f"(~{paint['cans_2_5l_equivalent']} kaleng @2.5L)"
            ),
            "Jumlah Kebutuhan": paint_qty,
            "Satuan": "liter",
            "Harga Satuan (Rp)": round(float(paint_unit_price_per_liter), 0),
            "Total Biaya (Rp)": round(paint_cost, 0),
        },
    ]

    if material_result.get("include_structural") and structural is not None:
        b_qty = int(structural["bricks_needed"])
        c_qty = int(structural["cement_sacks_needed"])
        s_qty = float(structural["sand_m3_needed"])

        items.extend(
            [
                {
                    "Nama Material": "Batu Bata Merah Press",
                    "Spesifikasi / Keterangan": f"Pasangan dinding {structural['wall_surface_m2']:.2f} m² (@70 bh/m²)",
                    "Jumlah Kebutuhan": b_qty,
                    "Satuan": "buah",
                    "Harga Satuan (Rp)": round(float(brick_unit_price), 0),
                    "Total Biaya (Rp)": round(b_qty * float(brick_unit_price), 0),
                },
                {
                    "Nama Material": "Semen Portland (Sak 50 kg)",
                    "Spesifikasi / Keterangan": f"Pasangan & plesteran dinding {structural['wall_surface_m2']:.2f} m²",
                    "Jumlah Kebutuhan": c_qty,
                    "Satuan": "sak",
                    "Harga Satuan (Rp)": round(float(cement_unit_price), 0),
                    "Total Biaya (Rp)": round(c_qty * float(cement_unit_price), 0),
                },
                {
                    "Nama Material": "Pasir Pasang & Plester",
                    "Spesifikasi / Keterangan": f"Pasangan & plesteran dinding {structural['wall_surface_m2']:.2f} m²",
                    "Jumlah Kebutuhan": s_qty,
                    "Satuan": "m³",
                    "Harga Satuan (Rp)": round(float(sand_unit_price), 0),
                    "Total Biaya (Rp)": round(s_qty * float(sand_unit_price), 0),
                },
            ]
        )

    cost_df = pd.DataFrame(items)
    total_cost_idr = float(cost_df["Total Biaya (Rp)"].sum())

    # Buat versi terformat Rupiah untuk tampilan tabel laporan/UI
    display_df = cost_df.copy()
    display_df["Harga Satuan"] = display_df["Harga Satuan (Rp)"].apply(format_rupiah)
    display_df["Total Biaya"] = display_df["Total Biaya (Rp)"].apply(format_rupiah)

    scope_note = (
        "Catatan Batasan Estimasi Biaya: Estimasi biaya di atas hanya mencakup kebutuhan "
        "material bangunan yang dimasukkan ke dalam sistem. Biaya tenaga kerja/tukang, "
        "transportasi, fondasi, atap, kusen pintu/jendela, instalasi listrik, dan pipa air "
        "belum termasuk dalam perhitungan."
    )

    return {
        "cost_df": cost_df,
        "display_df": display_df[
            ["Nama Material", "Spesifikasi / Keterangan", "Jumlah Kebutuhan", "Satuan", "Harga Satuan", "Total Biaya"]
        ],
        "tile_cost_idr": round(tile_cost, 0),
        "paint_cost_idr": round(paint_cost, 0),
        "total_cost_idr": round(total_cost_idr, 0),
        "total_cost_formatted": format_rupiah(total_cost_idr),
        "scope_note": scope_note,
    }

