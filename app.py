"""
SmartPlan – Aplikasi Web Deteksi Denah Rumah dan Estimasi Kebutuhan Material Bangunan
Berbasis Computer Vision (OpenCV & Streamlit)

Jalankan dengan perintah:
    streamlit run app.py
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

from src.preprocessing import preprocess_floorplan
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
from src.material_calculator import TILE_PRESETS, calculate_all_materials
from src.cost_estimator import (
    load_material_prices,
    save_material_prices,
    estimate_material_costs,
    format_rupiah,
)
from src.evaluation import (
    inspect_dataset,
    evaluate_against_reference_areas,
    evaluate_against_ground_truth_mask,
    run_multi_floorplan_benchmark,
)


st.set_page_config(
    page_title="SmartPlan – Deteksi Denah & Estimasi Material",
    page_icon="🏠",
    layout="wide",
)

# Header Aplikasi
st.title("🏠 SmartPlan – Sistem Deteksi Denah Rumah & Estimasi Kebutuhan Material")
st.markdown(
    """
    **Aplikasi berbasis Computer Vision Klasik (OpenCV)** untuk menganalisis gambar denah rumah 2D,
    mendeteksi garis dinding (*Hough Line Transform*) dan area ruangan (*Flood Fill & Contour Detection*),
    menghitung dimensi serta luas lantai bangunan, dan memperkirakan kebutuhan material (**Keramik Lantai** & **Cat Dinding**)
    beserta **Estimasi Biaya**.
    """
)

# Load Database Harga Material (Pandas)
price_df = load_material_prices()

# ==============================================================================
# SIDEBAR: INPUT GAMBAR, PARAMETER OPENCV, SKALA, MATERIAL, DAN HARGA
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan & Input Parameter")

    # 1. Sumber Gambar Denah
    st.subheader("1. Pilih / Unggah Gambar Denah")
    source_mode = st.radio(
        "Sumber Gambar Denah:",
        options=["Sampel Dataset (Train)", "Unggah Gambar Sendiri (PNG/JPG)", "Jelajahi Dataset Raw (Kaggle)"],
        index=0,
    )

    selected_image_bgr = None
    selected_image_name = "custom_upload.png"
    gt_mask_candidate_path = None

    if source_mode == "Sampel Dataset (Train)":
        train_dir = os.path.join("dataset", "train")
        train_files = (
            sorted(
                [f for f in os.listdir(train_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
                key=lambda x: int("".join(filter(str.isdigit, x)) or 9999),
            )
            if os.path.exists(train_dir)
            else []
        )
        if train_files:
            chosen_train = st.selectbox("Pilih Gambar Denah (10 Sampel Train):", train_files, index=0)
            selected_image_name = chosen_train
            img_path = os.path.join(train_dir, chosen_train)
            selected_image_bgr = cv2.imread(img_path)
            # Petakan ke ground truth di dataset/raw/Y jika ada
            train_to_gt = {
                "denah1.png": "1 (2).png",
                "denah2.png": "2 (2).png",
                "denah3.png": "3 (2).png",
                "denah4.png": "4 (2).png",
                "denah5.png": "5 (2).png",
                "denah6.png": "8 (2).png",
                "denah7.png": "9 (2).png",
                "denah8.png": "10 (2).png",
                "denah9.png": "14 (2).png",
                "denah10.png": "13 (2).png",
            }
            if chosen_train in train_to_gt:
                gt_mask_candidate_path = os.path.join("dataset", "raw", "Y", train_to_gt[chosen_train])
        else:
            st.warning("Folder `dataset/train` kosong. Silakan unggah gambar denah.")

    elif source_mode == "Jelajahi Dataset Raw (Kaggle)":
        raw_x_dir = os.path.join("dataset", "raw", "X")
        raw_files = (
            sorted(
                [f for f in os.listdir(raw_x_dir) if f.lower().endswith(".png")],
                key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 9999,
            )
            if os.path.exists(raw_x_dir)
            else []
        )
        if raw_files:
            chosen_raw = st.selectbox("Pilih Gambar dari `dataset/raw/X` (121 Denah):", raw_files, index=0)
            selected_image_name = chosen_raw
            selected_image_bgr = cv2.imread(os.path.join(raw_x_dir, chosen_raw))
            base_num = os.path.splitext(chosen_raw)[0]
            gt_mask_candidate_path = os.path.join("dataset", "raw", "Y", f"{base_num} (2).png")
        else:
            st.warning("Folder `dataset/raw/X` belum ditemukan. Jalankan `python download_dataset.py`.")

    else:
        uploaded_file = st.file_uploader("Unggah file denah rumah (PNG/JPG/JPEG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            selected_image_name = uploaded_file.name
            pil_img = Image.open(uploaded_file).convert("RGB")
            rgb_arr = np.array(pil_img)
            selected_image_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    st.divider()

    # 2. Kalibrasi Skala Denah (Tahap 8)
    st.subheader("2. Ukuran Referensi & Skala Denah")
    scale_mode = st.selectbox(
        "Metode Penentuan Skala:",
        options=[
            "Preset Otomatis Sesuai Gambar",
            "Input Ukuran Referensi Manual (Piksel -> Meter)",
            "Contoh Proposal (200 px = 4 m -> 50 px/m)",
        ],
        index=0,
    )

    if scale_mode == "Preset Otomatis Sesuai Gambar" and selected_image_name in DATASET_SCALE_PRESETS:
        preset = DATASET_SCALE_PRESETS[selected_image_name]
        ref_pixels = float(preset["ref_pixels"])
        ref_meters = float(preset["ref_meters"])
        st.caption(f"📌 **{preset['description']}**")
    elif scale_mode == "Contoh Proposal (200 px = 4 m -> 50 px/m)":
        preset = DATASET_SCALE_PRESETS["contoh_proposal"]
        ref_pixels = float(preset["ref_pixels"])
        ref_meters = float(preset["ref_meters"])
        st.caption(f"📌 **{preset['description']}**")
    else:
        default_px = DATASET_SCALE_PRESETS.get(selected_image_name, {}).get("ref_pixels", 200.0)
        default_m = DATASET_SCALE_PRESETS.get(selected_image_name, {}).get("ref_meters", 4.0)
        ref_pixels = st.number_input(
            "Panjang Dinding Referensi pada Gambar (piksel):",
            min_value=10.0,
            max_value=5000.0,
            value=float(default_px),
            step=5.0,
        )
        ref_meters = st.number_input(
            "Panjang Dinding Sebenarnya (meter):",
            min_value=0.1,
            max_value=100.0,
            value=float(default_m),
            step=0.1,
        )

    pixels_per_meter = calculate_scale(ref_pixels, ref_meters)
    st.info(f"**Skala Aktif:** `{pixels_per_meter:.2f} piksel / meter` (1 px ≈ {100.0/pixels_per_meter:.2f} cm)")

    area_calc_method = st.radio(
        "Metode Perhitungan Luas Ruangan:",
        options=["contour", "bbox"],
        format_func=lambda x: "Luas Kontur Piksel Akurat (Direkomendasikan)" if x == "contour" else "Panjang × Lebar (Bounding Box)",
        index=0,
    )

    st.divider()

    # 3. Parameter Spesifikasi Material (Tahap 9)
    st.subheader("3. Spesifikasi Material Bangunan")
    tile_choice = st.selectbox("Pilih Ukuran Keramik Lantai:", options=list(TILE_PRESETS.keys()) + ["Custom"], index=1)
    if tile_choice != "Custom":
        tile_len_cm = float(TILE_PRESETS[tile_choice]["length_cm"])
        tile_wid_cm = float(TILE_PRESETS[tile_choice]["width_cm"])
        tiles_per_box = int(TILE_PRESETS[tile_choice]["tiles_per_box"])
        mat_id = TILE_PRESETS[tile_choice]["material_id"]
        matched_row = price_df[price_df["id_material"] == mat_id]
        default_tile_price = float(matched_row["harga_satuan"].iloc[0]) if not matched_row.empty else 10500.0
    else:
        tile_len_cm = st.number_input("Panjang Keramik (cm):", min_value=10.0, max_value=120.0, value=40.0, step=5.0)
        tile_wid_cm = st.number_input("Lebar Keramik (cm):", min_value=10.0, max_value=120.0, value=40.0, step=5.0)
        tiles_per_box = st.number_input("Isi Keping per Dus:", min_value=1, max_value=30, value=6, step=1)
        default_tile_price = 10500.0

    tile_waste_pct = st.slider("Faktor Cadangan / Pemborosan Keramik (%):", min_value=0, max_value=25, value=10, step=1)

    wall_height_m = st.number_input("Tinggi Dinding Bangunan (meter):", min_value=1.5, max_value=10.0, value=3.0, step=0.1)
    paint_coverage = st.number_input("Daya Sebar Cat (m² / liter):", min_value=2.0, max_value=25.0, value=10.0, step=0.5)
    paint_coats = st.slider("Jumlah Lapisan Pengecatan (Lapis):", min_value=1, max_value=4, value=2, step=1)
    wall_sides = st.radio(
        "Sisi Permukaan Dinding yang Dicat:",
        options=[1, 2],
        format_func=lambda s: "1 Sisi Permukaan Dinding" if s == 1 else "2 Sisi (Bolak-balik Dalam & Luar)",
        index=0,
    )

    with st.expander("Pengurangan Pintu & Jendela (Opsional)"):
        door_count = st.number_input("Jumlah Pintu (0.9m × 2.1m):", min_value=0, max_value=50, value=0, step=1)
        window_count = st.number_input("Jumlah Jendela (1.0m × 1.2m):", min_value=0, max_value=50, value=0, step=1)

    include_structural = st.checkbox(
        "Sertakan Estimasi Material Konstruksi Dinding (Batu Bata, Semen, Pasir)",
        value=False,
    )

    st.divider()

    # 4. Input Harga Satuan Material (Tahap 10)
    st.subheader("4. Harga Satuan Material (Rp)")
    tile_unit_price = st.number_input(
        f"Harga Keramik {int(tile_len_cm)}×{int(tile_wid_cm)} cm (Rp / keping):",
        min_value=500.0,
        max_value=500000.0,
        value=float(default_tile_price),
        step=500.0,
    )
    paint_unit_price = st.number_input(
        "Harga Cat Dinding (Rp / liter):",
        min_value=5000.0,
        max_value=500000.0,
        value=45000.0,
        step=2500.0,
    )

    st.divider()

    # 5. Pengaturan Parameter Computer Vision (Tahap 6 & 7)
    with st.expander("🔧 Penyesuaian Parameter OpenCV (Lanjutan)"):
        blur_ksize = st.slider("Ukuran Kernel Gaussian Blur:", min_value=3, max_value=11, value=3, step=2)
        thresh_val = st.slider("Nilai Ambang Threshold (0-255):", min_value=100, max_value=245, value=200, step=5)
        close_ksize = st.slider("Kernel Morphological Closing:", min_value=3, max_value=17, value=9, step=2)
        open_ksize = st.slider("Kernel Morphological Opening:", min_value=3, max_value=15, value=7, step=2)
        door_gap_px = st.slider("Penutup Celah Pintu Interior (px):", min_value=40, max_value=180, value=115, step=5)
        ext_gap_px = st.slider("Penutup Bukaan Dinding Luar (px):", min_value=100, max_value=300, value=230, step=10)
        min_room_ratio = st.slider(
            "Filter Luas Minimum Ruangan (% area gambar):",
            min_value=0.05,
            max_value=2.0,
            value=0.20,
            step=0.05,
        )

# ==============================================================================
# EKSEKUSI PIPELINE SMARTPLAN
# ==============================================================================
if selected_image_bgr is None:
    st.info("👆 Silakan pilih gambar denah pada panel sebelah kiri atau unggah file gambar denah (PNG/JPG) untuk memulai analisis.")
    st.stop()

# Jalankan Preprocessing (Tahap 6)
prep_res = preprocess_floorplan(
    selected_image_bgr,
    blur_ksize=blur_ksize,
    thresh_val=thresh_val,
    close_ksize=close_ksize,
    open_ksize=open_ksize,
    door_gap_px=door_gap_px,
    ext_gap_px=ext_gap_px,
)

# Jalankan Deteksi Dinding & Ruangan (Tahap 7)
wall_res = detect_walls_hough(prep_res["walls_unbridged"], scale_factor=prep_res["scale_factor"])
room_res = detect_rooms_floodfill_contours(
    prep_res["walls_clean"],
    scale_factor=prep_res["scale_factor"],
    min_room_area_ratio=min_room_ratio / 100.0,
)

# Jalankan Perhitungan Dimensi & Luas (Tahap 8)
meas_res = measure_building_dimensions(
    room_res,
    wall_res,
    pixels_per_meter=pixels_per_meter,
    area_method=area_calc_method,
)

# Buat Visualisasi Deteksi
vis_res = visualize_detections(
    prep_res["original"],
    wall_res,
    room_res,
    pixels_per_meter=pixels_per_meter,
)

# Jalankan Perhitungan Material (Tahap 9) & Estimasi Biaya (Tahap 10)
mat_res = calculate_all_materials(
    measurement_result=meas_res,
    tile_length_cm=tile_len_cm,
    tile_width_cm=tile_wid_cm,
    tile_waste_percent=float(tile_waste_pct),
    tiles_per_box=tiles_per_box,
    wall_height_m=wall_height_m,
    paint_coverage_m2_per_liter=paint_coverage,
    paint_coats=paint_coats,
    wall_sides=wall_sides,
    door_count=door_count,
    window_count=window_count,
    include_structural=include_structural,
)

cost_res = estimate_material_costs(
    material_result=mat_res,
    tile_unit_price=tile_unit_price,
    paint_unit_price_per_liter=paint_unit_price,
)

# ==============================================================================
# RINGKASAN METRIK UTAMA (KPI CARDS)
# ==============================================================================
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
col_kpi1.metric("Jumlah Ruangan", f"{meas_res['num_rooms']} Ruangan")
col_kpi2.metric("Total Luas Lantai", f"{meas_res['total_floor_area_m2']:.2f} m²")
col_kpi3.metric("Panjang Total Dinding", f"{meas_res['total_wall_length_m']:.2f} m")
col_kpi4.metric(
    f"Kebutuhan Keramik ({mat_res['tiles']['tile_size_label']})",
    f"{mat_res['tiles']['total_tiles_count']:,} keping".replace(",", "."),
    f"~{mat_res['tiles']['boxes_needed']} dus (+{tile_waste_pct}% cadangan)",
)
col_kpi5.metric(
    "Total Estimasi Biaya",
    cost_res["total_cost_formatted"],
    f"Cat: {mat_res['paint']['paint_liters_needed']} Liter",
)

# ==============================================================================
# TABS ANTARMUKA UTAMA
# ==============================================================================
tab_utama, tab_preproc, tab_material, tab_evaluasi = st.tabs(
    [
        "🔍 Hasil Deteksi & Analisis Ruangan",
        "🖼️ Tahapan Pengolahan Citra (OpenCV)",
        "🧱 Kalkulator Material & Estimasi Biaya",
        "📊 Evaluasi Sistem & Pemeriksaan Dataset",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: HASIL DETEKSI & ANALISIS RUANGAN
# ------------------------------------------------------------------------------
with tab_utama:
    st.subheader("Visualisasi Hasil Deteksi Dinding dan Area Ruangan")
    c_img1, c_img2, c_img3 = st.columns(3)
    with c_img1:
        st.image(
            cv2.cvtColor(prep_res["original"], cv2.COLOR_BGR2RGB),
            caption=f"1. Gambar Denah Asli ({selected_image_name})",
            use_container_width=True,
        )
    with c_img2:
        st.image(
            cv2.cvtColor(vis_res["walls_vis"], cv2.COLOR_BGR2RGB),
            caption=(
                f"2. Deteksi Garis Dinding - Hough Transform "
                f"(H: {wall_res['num_horizontal']}, V: {wall_res['num_vertical']}, Total: {meas_res['total_wall_length_m']:.1f} m)"
            ),
            use_container_width=True,
        )
    with c_img3:
        st.image(
            cv2.cvtColor(vis_res["rooms_vis"], cv2.COLOR_BGR2RGB),
            caption=f"3. Segmentasi {meas_res['num_rooms']} Area Ruangan (Total: {meas_res['total_floor_area_m2']:.2f} m²)",
            use_container_width=True,
        )

    st.divider()
    st.subheader("📋 Tabel Dimensi dan Luas Masing-Masing Ruangan")
    if not meas_res["rooms_df"].empty:
        st.dataframe(meas_res["rooms_df"], use_container_width=True, hide_index=True)

        # Grafik Perbandingan Luas Setiap Ruangan (Matplotlib)
        fig_r, ax_r = plt.subplots(figsize=(10, 4))
        names = [f"R{r['id_ruangan']}" for r in meas_res["rooms"]]
        areas = [r["luas_m2"] for r in meas_res["rooms"]]
        bars = ax_r.bar(names, areas, color="#2b6cb0", edgecolor="#1a365d")
        ax_r.set_title("Perbandingan Luas Setiap Ruangan Terdeteksi (m²)", fontsize=12, fontweight="bold")
        ax_r.set_xlabel("ID Ruangan")
        ax_r.set_ylabel("Luas (m²)")
        ax_r.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, areas):
            ax_r.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.3,
                f"{val:.2f} m²",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        st.pyplot(fig_r)
        plt.close(fig_r)
    else:
        st.warning("Tidak ada area ruangan tertutup yang terdeteksi. Coba sesuaikan parameter OpenCV di sidebar.")

    st.info(f"ℹ️ {meas_res['disclaimer']}")

# ------------------------------------------------------------------------------
# TAB 2: TAHAPAN PENGOLAHAN CITRA (IMAGE PROCESSING - BAGIAN 6)
# ------------------------------------------------------------------------------
with tab_preproc:
    st.subheader("Tahapan Pengolahan Citra (Image Processing Pipeline)")
    st.markdown(
        """
        Sesuai rancangan **Tahap 6 & Tahap 7**, gambar denah diproses secara bertahap menggunakan OpenCV:
        1. **Pembacaan Gambar (`cv2.imread`)** & **Konversi ke Grayscale (`cv2.cvtColor`)**
        2. **Reduksi Noise (`cv2.GaussianBlur`)**
        3. **Binarisasi Citra (`cv2.threshold` dengan `THRESH_BINARY_INV`)**
        4. **Operasi Morfologi (`Erosion`, `Dilation`, `Opening`, `Closing`)** untuk memisahkan garis dinding dari teks/simbol tipis
        5. **Penutupan Celah Pintu/Jendela & Flood Fill (`cv2.floodFill`)** untuk memperoleh mask ruangan tertutup.
        """
    )

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.image(prep_res["gray"], caption="1. Grayscale (cv2.cvtColor)", use_container_width=True, clamp=True)
        st.image(
            prep_res["morph_closed"],
            caption="4. Morphological Closing (Penyatuan Arsiran Dinding)",
            use_container_width=True,
            clamp=True,
        )
    with p_col2:
        st.image(prep_res["blur"], caption=f"2. Gaussian Blur ({blur_ksize}×{blur_ksize})", use_container_width=True, clamp=True)
        st.image(
            prep_res["morph_opened"],
            caption="5. Morphological Opening (Penghapusan Garis Dimensi & Teks)",
            use_container_width=True,
            clamp=True,
        )
    with p_col3:
        st.image(
            prep_res["threshold"],
            caption=f"3. Thresholding Binary Inverse (T={thresh_val})",
            use_container_width=True,
            clamp=True,
        )
        st.image(
            room_res["rooms_mask"],
            caption="6. Hasil Flood Fill & Mask Area Ruangan Tertutup",
            use_container_width=True,
            clamp=True,
        )

# ------------------------------------------------------------------------------
# TAB 3: KALKULATOR KEBUTUHAN MATERIAL & ESTIMASI BIAYA (BAGIAN 9 & 10)
# ------------------------------------------------------------------------------
with tab_material:
    st.subheader("🧱 Rincian Kebutuhan Material Bangunan (Tahap 9)")

    m_col1, m_col2 = st.columns(2)
    tiles = mat_res["tiles"]
    paint = mat_res["paint"]

    with m_col1:
        st.markdown("#### 1. Perhitungan Kebutuhan Keramik Lantai")
        st.markdown(
            f"""
            - **Total Luas Lantai Bangunan:** `{tiles['total_floor_area_m2']:.2f} m²`
            - **Ukuran Keramik Dipilih:** `{tiles['tile_size_label']}` (`{tiles['tile_area_m2']:.4f} m² / keping`)
            - **Kebutuhan Dasar (Tanpa Cadangan):** `{tiles['base_tiles_count']:,} keping`
            - **Tambahan Cadangan / Waste ({tiles['waste_percent']:.0f}%):** `{tiles['waste_tiles_count']:,} keping`
            - **Total Kebutuhan Keramik:** **`{tiles['total_tiles_count']:,} keping`** *(setara ± `{tiles['boxes_needed']} dus` @{tiles['tiles_per_box']} keping/dus)*
            """
        )

    with m_col2:
        st.markdown("#### 2. Perhitungan Kebutuhan Cat Dinding")
        st.markdown(
            f"""
            - **Panjang Total Dinding Terdeteksi:** `{paint['total_wall_length_m']:.2f} m`
            - **Tinggi Dinding Input:** `{paint['wall_height_m']:.2f} m` ({paint['wall_sides']} sisi permukaan)
            - **Luas Permukaan Dinding Kotor:** `{paint['gross_wall_area_m2']:.2f} m²`
            - **Pengurangan Pintu & Jendela:** `{paint['opening_deduction_m2']:.2f} m²`
            - **Luas Dinding Bersih × {paint['paint_coats']} Lapis:** `{paint['total_painted_area_m2']:.2f} m²`
            - **Daya Sebar Cat:** `{paint['paint_coverage_m2_per_liter']:.1f} m² / liter`
            - **Total Kebutuhan Cat Dinding:** **`{paint['paint_liters_needed']} liter`** *(akurat `{paint['paint_liters_exact']:.2f} L`, setara ± `{paint['cans_2_5l_equivalent']} kaleng @2.5L`)*
            """
        )

    if not tiles["room_breakdown_df"].empty:
        with st.expander("📐 Lihat Rincian Kebutuhan Keramik per Ruangan", expanded=False):
            st.dataframe(tiles["room_breakdown_df"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("💰 Tabel Estimasi Biaya Material Bangunan (Tahap 10)")
    st.dataframe(cost_res["display_df"], use_container_width=True, hide_index=True)

    st.success(f"### Total Estimasi Biaya Material: **{cost_res['total_cost_formatted']}**")
    st.caption(cost_res["scope_note"])

    # Tombol Unduh Laporan CSV
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Unduh Tabel Dimensi Ruangan (CSV)",
            data=meas_res["rooms_df"].to_csv(index=False).encode("utf-8"),
            file_name="smartplan_dimensi_ruangan.csv",
            mime="text/csv",
        )
    with dl_col2:
        st.download_button(
            "⬇️ Unduh Tabel Estimasi Material & Biaya (CSV)",
            data=cost_res["cost_df"].to_csv(index=False).encode("utf-8"),
            file_name="smartplan_estimasi_biaya.csv",
            mime="text/csv",
        )

    st.divider()
    with st.expander("🗃️ Kelola Database Harga Satuan Material (`data/material_prices.csv`)"):
        st.markdown("Anda dapat memperbarui daftar harga referensi di bawah ini dan menyimpannya ke file database CSV:")
        edited_prices = st.data_editor(price_df, use_container_width=True, num_rows="dynamic")
        if st.button("💾 Simpan Perubahan Database Harga"):
            save_material_prices(edited_prices)
            st.success("Database harga material berhasil diperbarui di `data/material_prices.csv`!")

# ------------------------------------------------------------------------------
# TAB 4: EVALUASI SISTEM & PEMERIKSAAN DATASET (BAGIAN 4 & 13)
# ------------------------------------------------------------------------------
with tab_evaluasi:
    st.subheader("📊 Pengujian dan Evaluasi Sistem (Tahap 13)")

    # 1. Evaluasi terhadap ukuran referensi tertulis pada denah1.png
    if selected_image_name.lower() in ("denah1.png", "1.png"):
        st.markdown("#### A. Perbandingan Luas Deteksi OpenCV vs Ukuran Tertulis pada Denah")
        eval_ref = evaluate_against_reference_areas(meas_res["rooms"])
        if not eval_ref["eval_df"].empty:
            ev1, ev2, ev3 = st.columns(3)
            ev1.metric("Rata-rata Akurasi Per Ruangan", f"{eval_ref['mean_accuracy_pct']:.2f}%")
            ev2.metric("Rata-rata Tingkat Kesalahan (MAPE)", f"{eval_ref['mean_error_pct']:.2f}%")
            ev3.metric(
                "Akurasi Total Luas Ruangan",
                f"{eval_ref['total_accuracy_pct']:.2f}%",
                f"{eval_ref['total_det_m2']:.2f} m² vs {eval_ref['total_ref_m2']:.2f} m²",
            )
            st.dataframe(eval_ref["eval_df"], use_container_width=True, hide_index=True)

    # 2. Evaluasi terhadap mask anotasi Ground Truth Dataset (jika tersedia di dataset/raw/Y)
    if gt_mask_candidate_path and os.path.exists(gt_mask_candidate_path):
        st.markdown("#### B. Evaluasi Segmentasi Ruangan OpenCV vs Anotasi Ground Truth Dataset (`dataset/raw/Y`)")
        gt_eval = evaluate_against_ground_truth_mask(room_res["rooms_mask"], gt_mask_candidate_path)
        if gt_eval:
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Intersection over Union (IoU)", f"{gt_eval['iou_percent']:.2f}%")
            g2.metric("Precision Piksel Ruangan", f"{gt_eval['precision_percent']:.2f}%")
            g3.metric("Recall Piksel Ruangan", f"{gt_eval['recall_percent']:.2f}%")
            g4.metric("Kemiripan Total Area Ruangan", f"{gt_eval['area_similarity_percent']:.2f}%")

            gt_img_bgr = cv2.imread(gt_mask_candidate_path)
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(
                    cv2.cvtColor(gt_img_bgr, cv2.COLOR_BGR2RGB),
                    caption=f"Anotasi Ground Truth Dataset ({os.path.basename(gt_mask_candidate_path)})",
                    use_container_width=True,
                )
            with gc2:
                st.image(
                    room_res["rooms_mask"],
                    caption="Mask Area Ruangan Hasil Pengolahan Citra Klasik OpenCV",
                    use_container_width=True,
                    clamp=True,
                )

    st.divider()
    st.markdown("#### C. Hasil Pengujian Multi-Denah dengan Karakteristik Berbeda")
    if st.button("▶️ Jalankan Benchmark pada denah1, denah2, dan denah3"):
        with st.spinner("Menguji kemampuan OpenCV pada beberapa gambar denah..."):
            bench_df = run_multi_floorplan_benchmark()
            st.dataframe(bench_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### D. Pemeriksaan Dataset Kaggle (`asutoshprad/floor-plan-dataset` - Tahap 4)")
    ds_info = inspect_dataset()
    if ds_info["available"]:
        st.write(
            f"- **Jumlah Gambar Denah (`dataset/raw/X`):** `{ds_info['num_images_x']}` file PNG\n"
            f"- **Jumlah Mask Anotasi (`dataset/raw/Y`):** `{ds_info['num_annotations_y']}` file PNG"
        )
        st.dataframe(ds_info["sample_df"], use_container_width=True, hide_index=True)
    else:
        st.info("Folder `dataset/raw` tidak ditemukan.")

