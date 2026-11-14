"""
Tahap 7: Deteksi Dinding dan Ruangan
Menggunakan OpenCV untuk:
1. Mendeteksi garis dinding horizontal & vertikal menggunakan Hough Line Transform (cv2.HoughLinesP)
2. Mengidentifikasi area ruangan tertutup menggunakan Flood Fill (cv2.floodFill)
3. Menemukan batas dan bentuk ruangan menggunakan Contour Detection (cv2.findContours)
4. Memvisualisasikan hasil deteksi dinding dan ruangan pada gambar denah.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


# Palet warna BGR yang kontras dan ramah visual untuk ruangan
ROOM_COLORS_BGR: List[Tuple[int, int, int]] = [
    (180, 210, 120),  # Hijau mint
    (140, 180, 240),  # Oranye pastel
    (230, 180, 130),  # Biru langit
    (190, 150, 230),  # Merah muda / lavender
    (130, 220, 235),  # Kuning emas muda
    (210, 210, 130),  # Teal muda
    (160, 230, 170),  # Hijau apel
    (150, 160, 240),  # Salmon
    (225, 195, 165),  # Biru keabu-abuan
    (175, 220, 205),  # Sage
    (200, 170, 210),  # Ungu pastel
    (145, 205, 245),  # Amber muda
]


def _skeletonize(binary_img: np.ndarray) -> np.ndarray:
    """Melakukan penipisan morfologi (morphological skeletonization) pada mask biner dinding."""
    skel = np.zeros(binary_img.shape, np.uint8)
    temp_img = (binary_img > 0).astype(np.uint8) * 255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iters = 100
    iters = 0
    while iters < max_iters:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        iters += 1
        if cv2.countNonZero(temp_img) == 0:
            break
    return skel


def _merge_collinear_segments(
    lines: np.ndarray, tol_pos: float = 25.0, tol_gap: float = 25.0
) -> List[Dict[str, Any]]:
    """
    Menggabungkan segmen garis Hough horizontal dan vertikal yang sejajar/bertumpuk
    pada satu dinding yang sama agar perhitungan panjang dinding tidak terhitung ganda.
    """
    h_segs: List[Tuple[float, float, float]] = []
    v_segs: List[Tuple[float, float, float]] = []
    d_segs: List[Dict[str, Any]] = []

    for x1, y1, x2, y2 in lines:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 5:
            continue
        angle = float(np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle <= 15.0 or angle >= 165.0:
            ym = (y1 + y2) / 2.0
            h_segs.append((ym, float(min(x1, x2)), float(max(x1, x2))))
        elif 75.0 <= angle <= 105.0:
            xm = (x1 + x2) / 2.0
            v_segs.append((xm, float(min(y1, y2)), float(max(y1, y2))))
        else:
            d_segs.append(
                {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "orientation": "Diagonal",
                    "length_px": length,
                }
            )

    def _merge_1d(segs: List[Tuple[float, float, float]]) -> List[Tuple[int, int, int]]:
        if not segs:
            return []
        segs.sort(key=lambda s: s[0])
        groups: List[List[Any]] = []
        for pos, a, b in segs:
            if not groups or abs(pos - groups[-1][0]) > tol_pos:
                groups.append([pos, [(a, b)], 1])
            else:
                cnt = groups[-1][2]
                groups[-1][0] = (groups[-1][0] * cnt + pos) / (cnt + 1)
                groups[-1][1].append((a, b))
                groups[-1][2] = cnt + 1

        merged: List[Tuple[int, int, int]] = []
        for pos, intervals, _ in groups:
            intervals.sort(key=lambda iv: iv[0])
            cur_a, cur_b = intervals[0]
            for a, b in intervals[1:]:
                if a <= cur_b + tol_gap:
                    cur_b = max(cur_b, b)
                else:
                    merged.append((int(round(pos)), int(round(cur_a)), int(round(cur_b))))
                    cur_a, cur_b = a, b
            merged.append((int(round(pos)), int(round(cur_a)), int(round(cur_b))))
        return merged

    merged_list: List[Dict[str, Any]] = []
    for y_pos, x_start, x_end in _merge_1d(h_segs):
        length = float(max(0, x_end - x_start))
        if length >= 10:
            merged_list.append(
                {
                    "x1": x_start,
                    "y1": y_pos,
                    "x2": x_end,
                    "y2": y_pos,
                    "orientation": "Horizontal",
                    "length_px": length,
                }
            )

    for x_pos, y_start, y_end in _merge_1d(v_segs):
        length = float(max(0, y_end - y_start))
        if length >= 10:
            merged_list.append(
                {
                    "x1": x_pos,
                    "y1": y_start,
                    "x2": x_pos,
                    "y2": y_end,
                    "orientation": "Vertikal",
                    "length_px": length,
                }
            )

    merged_list.extend(d_segs)
    return merged_list


def detect_walls_hough(
    walls_mask: np.ndarray,
    scale_factor: float = 1.0,
    hough_threshold: int = 15,
    min_line_length: int = 30,
    max_line_gap: int = 20,
) -> Dict[str, Any]:
    """
    Mendeteksi garis dinding menggunakan Probabilistic Hough Line Transform (cv2.HoughLinesP).
    Menghasilkan daftar segmen garis mentah, segmen garis dinding tergabung (centerline),
    serta total panjang dinding dalam piksel.
    """
    close_k = max(3, int(round(15 * scale_factor)))
    solid_walls = cv2.morphologyEx(
        walls_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)),
    )
    skeleton = _skeletonize(solid_walls)

    min_len_scaled = max(10, int(round(min_line_length * scale_factor)))
    max_gap_scaled = max(5, int(round(max_line_gap * scale_factor)))

    raw_lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(5, int(hough_threshold)),
        minLineLength=min_len_scaled,
        maxLineGap=max_gap_scaled,
    )

    if raw_lines is None or len(raw_lines) == 0:
        lines_arr = np.zeros((0, 4), dtype=np.int32)
    else:
        lines_arr = raw_lines.reshape(-1, 4)

    tol = max(10.0, 26.0 * scale_factor)
    merged_segments = _merge_collinear_segments(lines_arr, tol_pos=tol, tol_gap=tol)

    horizontal_segments = [s for s in merged_segments if s["orientation"] == "Horizontal"]
    vertical_segments = [s for s in merged_segments if s["orientation"] == "Vertikal"]
    diagonal_segments = [s for s in merged_segments if s["orientation"] == "Diagonal"]

    total_length_px = float(sum(s["length_px"] for s in merged_segments))
    horizontal_length_px = float(sum(s["length_px"] for s in horizontal_segments))
    vertical_length_px = float(sum(s["length_px"] for s in vertical_segments))

    return {
        "skeleton": skeleton,
        "raw_lines": lines_arr,
        "segments": merged_segments,
        "num_raw_lines": int(len(lines_arr)),
        "num_merged_segments": len(merged_segments),
        "num_horizontal": len(horizontal_segments),
        "num_vertical": len(vertical_segments),
        "num_diagonal": len(diagonal_segments),
        "total_wall_length_px": total_length_px,
        "horizontal_length_px": horizontal_length_px,
        "vertical_length_px": vertical_length_px,
    }


def detect_rooms_floodfill_contours(
    walls_clean: np.ndarray,
    scale_factor: float = 1.0,
    min_room_area_ratio: float = 0.002,
    max_room_area_ratio: float = 0.65,
    room_open_ksize: int = 18,
) -> Dict[str, Any]:
    """
    Mendeteksi area ruangan tertutup menggunakan kombinasi:
    1. Inversi mask dinding (cv2.bitwise_not)
    2. Flood Fill (cv2.floodFill) dari titik luar denah untuk memisahkan area eksterior
    3. Morphological Opening untuk membersihkan artefak celah sempit
    4. Contour Detection (cv2.findContours) untuk mengidentifikasi batas setiap ruangan.
    """
    h_img, w_img = walls_clean.shape[:2]
    total_img_area = float(h_img * w_img)

    # Pastikan tepi gambar terbuka (0) agar floodFill dapat mengalir di seluruh area luar bangunan
    walls_bordered = walls_clean.copy()
    walls_bordered[0:2, :] = 0
    walls_bordered[-2:, :] = 0
    walls_bordered[:, 0:2] = 0
    walls_bordered[:, -2:] = 0

    inverted = cv2.bitwise_not(walls_bordered)
    flood_filled = inverted.copy()
    ff_mask = np.zeros((h_img + 2, w_img + 2), dtype=np.uint8)

    # Flood Fill dari keempat sudut eksterior gambar
    seed_points = [(0, 0), (w_img - 1, 0), (0, h_img - 1), (w_img - 1, h_img - 1)]
    for sx, sy in seed_points:
        if flood_filled[sy, sx] > 0:
            cv2.floodFill(flood_filled, ff_mask, (sx, sy), 0)

    # Bersihkan koridor sempit/artefak dinding menggunakan Morphological Opening
    rk = max(5, int(round(room_open_ksize * scale_factor)))
    rooms_mask = cv2.morphologyEx(
        flood_filled,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (rk, rk)),
    )

    contours, _ = cv2.findContours(rooms_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area_px = total_img_area * min_room_area_ratio
    max_area_px = total_img_area * max_room_area_ratio

    raw_rooms: List[Dict[str, Any]] = []
    for cnt in contours:
        area_px = float(cv2.contourArea(cnt))
        if min_area_px <= area_px <= max_area_px:
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter_px = float(cv2.arcLength(cnt, True))
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            raw_rooms.append(
                {
                    "contour": cnt,
                    "area_px": area_px,
                    "perimeter_px": perimeter_px,
                    "bbox_px": (int(x), int(y), int(w), int(h)),
                    "width_px": int(w),
                    "height_px": int(h),
                    "centroid_px": (cx, cy),
                }
            )

    # Urutkan ruangan dari atas ke bawah (berdasarkan baris y), lalu kiri ke kanan (x)
    row_band = max(50, int(h_img * 0.12))
    raw_rooms.sort(key=lambda r: (r["centroid_px"][1] // row_band, r["centroid_px"][0]))

    rooms: List[Dict[str, Any]] = []
    for idx, r in enumerate(raw_rooms, start=1):
        color_bgr = ROOM_COLORS_BGR[(idx - 1) % len(ROOM_COLORS_BGR)]
        r_copy = dict(r)
        r_copy["room_id"] = idx
        r_copy["name"] = f"Ruangan {idx}"
        r_copy["color_bgr"] = color_bgr
        rooms.append(r_copy)

    return {
        "inverted_walls": inverted,
        "flood_filled_interior": flood_filled,
        "rooms_mask": rooms_mask,
        "rooms": rooms,
        "num_rooms": len(rooms),
        "total_room_area_px": float(sum(r["area_px"] for r in rooms)),
    }


def visualize_detections(
    original_img: np.ndarray,
    wall_result: Dict[str, Any],
    room_result: Dict[str, Any],
    pixels_per_meter: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Membuat gambar visualisasi hasil deteksi:
    1. Visualisasi garis dinding Hough (Horizontal=Biru, Vertikal=Merah)
    2. Visualisasi area ruangan terdeteksi (warna transparan + kontur + label luas)
    3. Visualisasi gabungan dinding dan ruangan.
    """
    h_img, w_img = original_img.shape[:2]
    scale = max(0.5, max(h_img, w_img) / 1950.0)

    # 1. Visualisasi Garis Dinding (Hough Line Transform)
    walls_vis = original_img.copy()
    line_thickness = max(2, int(round(4 * scale)))
    for seg in wall_result["segments"]:
        x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
        if seg["orientation"] == "Horizontal":
            color = (220, 90, 20)  # Biru tua (BGR)
        elif seg["orientation"] == "Vertikal":
            color = (30, 50, 220)  # Merah (BGR)
        else:
            color = (20, 160, 20)  # Hijau (BGR)
        cv2.line(walls_vis, (x1, y1), (x2, y2), color, line_thickness)

    # 2. Visualisasi Area Ruangan (Flood Fill + Contours)
    rooms_vis = original_img.copy()
    overlay = original_img.copy()

    for room in room_result["rooms"]:
        cv2.drawContours(overlay, [room["contour"]], -1, room["color_bgr"], -1)

    rooms_vis = cv2.addWeighted(overlay, 0.45, rooms_vis, 0.55, 0)

    border_thickness = max(2, int(round(3 * scale)))
    font_scale = max(0.45, 0.68 * scale)
    font_thickness = max(1, int(round(2 * scale)))

    for room in room_result["rooms"]:
        cnt = room["contour"]
        cv2.drawContours(rooms_vis, [cnt], -1, (20, 20, 180), border_thickness)

        cx, cy = room["centroid_px"]
        rid = room["room_id"]
        if pixels_per_meter and pixels_per_meter > 0:
            area_m2 = room["area_px"] / (pixels_per_meter ** 2)
            label_line1 = f"R{rid}"
            label_line2 = f"{area_m2:.2f} m2"
        else:
            label_line1 = f"R{rid}"
            label_line2 = f"{int(room['area_px'])} px2"

        # Gambar kotak latar putih kecil agar teks mudah dibaca
        (tw1, th1), _ = cv2.getTextSize(label_line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        (tw2, th2), _ = cv2.getTextSize(label_line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, font_thickness)
        box_w = max(tw1, tw2) + int(14 * scale)
        box_h = th1 + th2 + int(18 * scale)
        bx1 = max(5, cx - box_w // 2)
        by1 = max(5, cy - box_h // 2)
        bx2 = min(w_img - 5, bx1 + box_w)
        by2 = min(h_img - 5, by1 + box_h)

        cv2.rectangle(rooms_vis, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
        cv2.rectangle(rooms_vis, (bx1, by1), (bx2, by2), (40, 40, 40), max(1, int(round(1.5 * scale))))

        cv2.putText(
            rooms_vis,
            label_line1,
            (bx1 + (box_w - tw1) // 2, by1 + th1 + int(5 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (10, 10, 140),
            font_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            rooms_vis,
            label_line2,
            (bx1 + (box_w - tw2) // 2, by1 + th1 + th2 + int(12 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85,
            (20, 20, 20),
            font_thickness,
            cv2.LINE_AA,
        )

    # 3. Visualisasi Gabungan (Ruangan + Garis Dinding Hough)
    combined_vis = rooms_vis.copy()
    for seg in wall_result["segments"]:
        x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
        cv2.line(combined_vis, (x1, y1), (x2, y2), (180, 40, 0), max(2, int(round(3 * scale))))

    return {
        "walls_vis": walls_vis,
        "rooms_vis": rooms_vis,
        "combined_vis": combined_vis,
    }

