"""
SmartPlan – Sistem Deteksi Denah Rumah dan Estimasi Kebutuhan Material Bangunan
Berbasis Computer Vision (OpenCV)

Script utama (CLI) untuk menjalankan seluruh alur kerja sistem dari membaca denah,
preprocessing, deteksi dinding & ruangan, perhitungan dimensi & luas,
kalkulasi kebutuhan material (keramik & cat), estimasi biaya, hingga evaluasi.
"""

import argparse
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_floorplan, save_preprocessing_results
from src.detection import (
    detect_walls_hough,
    detect_rooms_floodfill_contours,
    visualize_detections,
)
from src.measurement import (
    DATASET_SCALE_PRESETS,
    calculate_scale,
    measure_building_dimensions,
)
from src.material_calculator import calculate_all_materials
from src.cost_estimator import estimate_material_costs
from src.evaluation import (
    inspect_dataset,
    evaluate_against_reference_areas,
    evaluate_against_ground_truth_mask,
    run_multi_floorplan_benchmark,
)


def run_smartplan_pipeline(
    image_path: str,
    output_dir: str = "results",
    ref_pixels: float = 269.7,
    ref_meters: float = 3.35,
    tile_length_cm: float = 40.0,
    tile_width_cm: float = 40.0,
    tile_waste_percent: float = 10.0,
    wall_height_m: float = 3.0,
    paint_coverage: float = 10.0,
    paint_coats: int = 2,
    tile_unit_price: float = 10500.0,
    paint_unit_price: float = 45000.0,
    include_structural: bool = False,
) -> None:
    print("=" * 78)
    print(" SMARTPLAN - SISTEM DETEKSI DENAH RUMAH & ESTIMASI MATERIAL (COMPUTER VISION)")
    print("=" * 78)

    # 1. Pemeriksaan Dataset (Tahap 4)
    ds_info = inspect_dataset()
    if ds_info["available"]:
        print(
            f"[Tahap 4] Dataset Kaggle (asutoshprad/floor-plan-dataset): "
            f"{ds_info['num_images_x']} gambar denah (X) & {ds_info['num_annotations_y']} anotasi (Y)"
        )

    # 2. Tahap 6: Image Processing / Preprocessing
    print(f"\n[Tahap 6] Membaca dan memproses gambar denah: {image_path}")
    prep = preprocess_floorplan(image_path)
    orig = prep["original"]
    h_img, w_img, ch_img = orig.shape
    print(f" - Dimensi Gambar : {w_img} x {h_img} piksel ({ch_img} channel)")

    saved_prep = save_preprocessing_results(prep, output_dir=output_dir)
    print(f" - Hasil tahapan preprocessing disimpan di folder '{output_dir}/'")

    # 3. Tahap 7: Deteksi Dinding dan Ruangan
    print("\n[Tahap 7] Mendeteksi garis dinding (Hough Line Transform) & area ruangan (Flood Fill + Contours)...")
    wall_res = detect_walls_hough(prep["walls_unbridged"], scale_factor=prep["scale_factor"])
    room_res = detect_rooms_floodfill_contours(prep["walls_clean"], scale_factor=prep["scale_factor"])

    print(
        f" - Garis Dinding Terdeteksi : {wall_res['num_merged_segments']} segmen utama "
        f"(Horizontal: {wall_res['num_horizontal']}, Vertikal: {wall_res['num_vertical']})"
    )
    print(f" - Area Ruangan Terdeteksi  : {room_res['num_rooms']} ruangan tertutup")

    # 4. Tahap 8: Perhitungan Dimensi dan Luas Bangunan
    pixels_per_meter = calculate_scale(ref_pixels, ref_meters)
    print(
        f"\n[Tahap 8] Perhitungan Dimensi & Luas Bangunan "
        f"(Skala Referensi: {ref_pixels:.1f} px = {ref_meters:.2f} m -> {pixels_per_meter:.2f} piksel/meter)"
    )
    meas_res = measure_building_dimensions(
        room_res, wall_res, pixels_per_meter=pixels_per_meter, area_method="contour"
    )

    print("\n--- Tabel Rincian Dimensi & Luas Ruangan ---")
    print(meas_res["rooms_df"].to_string(index=False))
    print(f"\n - Total Luas Lantai Bangunan : {meas_res['total_floor_area_m2']:.2f} m²")
    print(f" - Panjang Total Dinding      : {meas_res['total_wall_length_m']:.2f} m")
    print(f"   ({meas_res['disclaimer']})")

    # Visualisasi Deteksi & Simpan ke results/
    vis_dict = visualize_detections(orig, wall_res, room_res, pixels_per_meter=pixels_per_meter)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "walls_detected.png"), vis_dict["walls_vis"])
    cv2.imwrite(os.path.join(output_dir, "rooms_detected.png"), vis_dict["rooms_vis"])
    cv2.imwrite(os.path.join(output_dir, "combined_detection.png"), vis_dict["combined_vis"])
    meas_res["rooms_df"].to_csv(os.path.join(output_dir, "rooms_summary.csv"), index=False)

    # Buat plot perbandingan hasil deteksi menggunakan Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Denah Asli", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(vis_dict["walls_vis"], cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"2. Deteksi Dinding Hough ({meas_res['total_wall_length_m']:.1f} m)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(vis_dict["rooms_vis"], cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f"3. Deteksi {meas_res['num_rooms']} Ruangan ({meas_res['total_floor_area_m2']:.2f} m²)",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.suptitle("SmartPlan - Hasil Deteksi Dinding dan Area Ruangan", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "detection_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Tahap 9: Perhitungan Kebutuhan Material Bangunan
    print("\n[Tahap 9] Perhitungan Kebutuhan Material Bangunan...")
    mat_res = calculate_all_materials(
        measurement_result=meas_res,
        tile_length_cm=tile_length_cm,
        tile_width_cm=tile_width_cm,
        tile_waste_percent=tile_waste_percent,
        wall_height_m=wall_height_m,
        paint_coverage_m2_per_liter=paint_coverage,
        paint_coats=paint_coats,
        include_structural=include_structural,
    )
    tiles = mat_res["tiles"]
    paint = mat_res["paint"]
    print(
        f" - Kebutuhan Keramik ({tiles['tile_size_label']}) : "
        f"{tiles['base_tiles_count']} keping dasar + {tiles['waste_tiles_count']} cadangan ({tiles['waste_percent']:.0f}%) "
        f"= {tiles['total_tiles_count']} keping (~{tiles['boxes_needed']} dus)"
    )
    print(
        f" - Kebutuhan Cat Dinding (Tinggi {paint['wall_height_m']} m, {paint['paint_coats']} lapis) : "
        f"Luas {paint['net_wall_area_m2']:.2f} m² -> {paint['paint_liters_needed']} liter "
        f"(~{paint['cans_2_5l_equivalent']} kaleng @2.5L)"
    )

    # 6. Tahap 10: Perhitungan Estimasi Biaya
    print("\n[Tahap 10] Perhitungan Estimasi Biaya Material...")
    cost_res = estimate_material_costs(
        material_result=mat_res,
        tile_unit_price=tile_unit_price,
        paint_unit_price_per_liter=paint_unit_price,
    )
    print("\n--- Tabel Kebutuhan Material & Estimasi Biaya ---")
    print(cost_res["display_df"].to_string(index=False))
    print(f"\n >>> TOTAL ESTIMASI BIAYA MATERIAL : {cost_res['total_cost_formatted']} <<<")
    print(f" ({cost_res['scope_note']})")
    cost_res["cost_df"].to_csv(os.path.join(output_dir, "material_cost_summary.csv"), index=False)

    # 7. Tahap 13: Pengujian dan Evaluasi Sistem
    if os.path.basename(image_path).lower() == "denah1.png":
        print("\n[Tahap 13] Evaluasi Ketepatan Pengukuran Luas terhadap Ukuran Referensi Denah 1:")
        eval_ref = evaluate_against_reference_areas(meas_res["rooms"])
        if not eval_ref["eval_df"].empty:
            print(eval_ref["eval_df"].to_string(index=False))
            print(
                f" - Rata-rata Akurasi Per Ruangan : {eval_ref['mean_accuracy_pct']:.2f}% "
                f"(Rata-rata Error: {eval_ref['mean_error_pct']:.2f}%)"
            )
            print(
                f" - Akurasi Total Luas Ruangan    : {eval_ref['total_accuracy_pct']:.2f}% "
                f"(OpenCV: {eval_ref['total_det_m2']:.2f} m² vs Referensi Denah: {eval_ref['total_ref_m2']:.2f} m²)"
            )
            eval_ref["eval_df"].to_csv(os.path.join(output_dir, "evaluation_denah1.csv"), index=False)

        gt_path = os.path.join("dataset", "raw", "Y", "1 (2).png")
        gt_eval = evaluate_against_ground_truth_mask(room_res["rooms_mask"], gt_path)
        if gt_eval:
            print(
                f" - Evaluasi Mask terhadap Anotasi Dataset (Y/1 (2).png): "
                f"IoU = {gt_eval['iou_percent']:.2f}%, Precision = {gt_eval['precision_percent']:.2f}%, "
                f"Kemiripan Total Area = {gt_eval['area_similarity_percent']:.2f}%"
            )

    print("\n[Tahap 13] Pengujian Multi-Denah (denah1, denah2, denah3):")
    bench_df = run_multi_floorplan_benchmark()
    if not bench_df.empty:
        print(bench_df.to_string(index=False))
        bench_df.to_csv(os.path.join(output_dir, "benchmark_multi_denah.csv"), index=False)

    print("\n" + "=" * 78)
    print(f" Seluruh proses selesai! File hasil visualisasi & laporan tersimpan di '{output_dir}/'")
    print(" Untuk menjalankan aplikasi web interaktif Streamlit, jalankan perintah:")
    print("   streamlit run app.py")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartPlan - Sistem Deteksi Denah Rumah dan Estimasi Kebutuhan Material"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join("dataset", "train", "denah1.png"),
        help="Path gambar denah rumah (PNG/JPG)",
    )
    parser.add_argument("--ref-pixels", type=float, default=269.7, help="Panjang referensi dalam piksel")
    parser.add_argument("--ref-meters", type=float, default=3.35, help="Panjang referensi sebenarnya dalam meter")
    parser.add_argument("--wall-height", type=float, default=3.0, help="Tinggi dinding dalam meter")
    parser.add_argument("--tile-size", type=float, default=40.0, help="Ukuran sisi keramik dalam cm")
    parser.add_argument("--waste", type=float, default=10.0, help="Persentase cadangan/waste keramik (%%)")
    parser.add_argument(
        "--structural",
        action="store_true",
        help="Sertakan estimasi material konstruksi lanjutan (bata, semen, pasir)",
    )
    args = parser.parse_args()

    run_smartplan_pipeline(
        image_path=args.image,
        ref_pixels=args.ref_pixels,
        ref_meters=args.ref_meters,
        tile_length_cm=args.tile_size,
        tile_width_cm=args.tile_size,
        tile_waste_percent=args.waste,
        wall_height_m=args.wall_height,
        include_structural=args.structural,
    )
