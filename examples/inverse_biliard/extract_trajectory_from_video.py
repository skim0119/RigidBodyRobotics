"""Extract billiard ball 2D trajectories from video.
Use template matching algorithm. I don't think optical flow make sense.

Order of operation (short):
1) Load video and first frame, then load/reuse cached selections if available.
2) Select/validate table corners and three ball ROIs (white/red/yellow).
3) Build homography from image plane to table metric plane.
4) Track each ball center per frame (color + template matching), and write overlay video.
5) Convert tracked centers to meters, smooth/trim by motion, and wall-fit correction.
6) Export CSV + metadata and save trajectory/position/speed diagnostic plots.

Workflow:
1) Open first frame.
2) Click four table corners (top-left, top-right, bottom-right, bottom-left).
3) Select 3 ROIs for white/red/yellow balls.
4) Track centers across all frames.
5) Export CSV with pixel and flattened metric coordinates.

Run:
    uv run python examples/biliard/extractrejectory.py \
        --video /path/to/video.mp4
"""

from __future__ import annotations
from typing import Literal

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import plotting as bp

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenCV is required. Install with: uv add opencv-python") from exc


BALL_LABELS = ("white", "red", "yellow")
TABLE_WIDTH_M = 2.54
TABLE_HEIGHT_M = 1.27
BALL_RADIUS_M = 0.05715 / 2.0
WALL_CLEARANCE_EPS_M = 1e-4


def _order_corners_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    """Return corners ordered as TL, TR, BR, BL from 4 unordered points."""
    if points.shape != (4, 2):
        raise ValueError("Expected exactly 4 corner points.")
    pts = points.astype(np.float32)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _color_ranges_hsv(label: str) -> list[tuple[np.ndarray, np.ndarray]]:
    # Relaxed ranges for varied lighting/white balance in billiard videos.
    if label == "white":
        return [
            (
                np.array([0, 0, 115], dtype=np.uint8),
                np.array([179, 95, 255], dtype=np.uint8),
            )
        ]
    if label == "red":
        return [
            (
                np.array([0, 55, 40], dtype=np.uint8),
                np.array([15, 255, 255], dtype=np.uint8),
            ),
            (
                np.array([160, 55, 40], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            ),
        ]
    if label == "yellow":
        return [
            (
                np.array([10, 45, 50], dtype=np.uint8),
                np.array([50, 255, 255], dtype=np.uint8),
            )
        ]
    return []


def _find_ball_center_by_color(
    search_bgr: np.ndarray,
    label: str,
    min_area: float,
    preferred_center: tuple[float, float] | None = None,
    expected_radius: float | None = None,
) -> tuple[float, float, float] | None:
    if search_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _color_ranges_hsv(label):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        per = cv2.arcLength(cnt, True) + 1e-6
        circularity = float(4.0 * np.pi * area / (per * per))
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius <= 1.0:
            continue
        # Prefer roughly circular blobs with expected ball size near predicted center.
        shape_score = float(np.clip(circularity, 0.0, 1.0))
        if expected_radius is None:
            radius_score = 1.0
        else:
            radius_score = float(
                np.exp(-abs(radius - expected_radius) / (expected_radius + 1e-6))
            )
        if preferred_center is None:
            dist_score = 1.0
        else:
            dx = cx - preferred_center[0]
            dy = cy - preferred_center[1]
            dist = float(np.hypot(dx, dy))
            scale = max(
                2.0, 2.0 * (expected_radius if expected_radius is not None else radius)
            )
            dist_score = float(np.exp(-((dist / scale) ** 2)))
        score = 0.45 * shape_score + 0.35 * radius_score + 0.20 * dist_score
        if score > best_score:
            best = (float(cx), float(cy), float(score))
            best_score = score

    return best


@dataclass
class TemplateTracker:
    label: str
    template_gray: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    last_center: tuple[float, float] | None = None
    last_velocity: tuple[float, float] = (0.0, 0.0)
    filt_center: tuple[float, float] | None = None
    filt_velocity: tuple[float, float] = (0.0, 0.0)

    def update(
        self,
        frame_bgr: np.ndarray,
        frame_gray: np.ndarray,
        search_scale: float = 3.2,
        color_min_area_ratio: float = 0.08,
        template_update_rate: float = 0.08,
        max_jump_px: float = 42.0,
        smooth_alpha: float = 0.55,
        process_blend: float = 0.22,
        max_accel_px: float = 14.0,
    ) -> tuple[float, float]:
        x, y, w, h = self.bbox
        cx = x + w // 2
        cy = y + h // 2
        if self.last_center is None:
            self.last_center = (float(cx), float(cy))
        if self.filt_center is None:
            self.filt_center = self.last_center
        pred_cx = self.last_center[0] + self.last_velocity[0]
        pred_cy = self.last_center[1] + self.last_velocity[1]

        sw = max(int(w * search_scale), w + 8)
        sh = max(int(h * search_scale), h + 8)
        sx0 = max(0, int(round(pred_cx - sw / 2.0)))
        sy0 = max(0, int(round(pred_cy - sh / 2.0)))
        sx1 = min(frame_gray.shape[1], sx0 + sw)
        sy1 = min(frame_gray.shape[0], sy0 + sh)
        search_bgr = frame_bgr[sy0:sy1, sx0:sx1]
        search = frame_gray[sy0:sy1, sx0:sx1]

        if search.shape[0] < h or search.shape[1] < w:
            return float(pred_cx), float(pred_cy)

        candidate_centers: list[tuple[float, float]] = []
        used_color_lock = False

        min_area = max(8.0, color_min_area_ratio * float(w * h))
        expected_radius = 0.33 * float(min(w, h))
        by_color = _find_ball_center_by_color(
            search_bgr,
            self.label,
            min_area=min_area,
            preferred_center=(pred_cx - sx0, pred_cy - sy0),
            expected_radius=expected_radius,
        )
        if by_color is not None:
            cxx, cyy, cscore = by_color
            candidate_centers.append((float(sx0 + cxx), float(sy0 + cyy)))
            # Strong color lock: skip template branch to avoid drift onto background texture.
            if cscore >= 0.45:
                used_color_lock = True

        if not used_color_lock:
            res = cv2.matchTemplate(search, self.template_gray, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            candidate_centers.append(
                (float(sx0 + max_loc[0] + w / 2.0), float(sy0 + max_loc[1] + h / 2.0))
            )

        # Pick candidate closest to predicted center, then clamp jump.
        best_cx, best_cy = min(
            candidate_centers,
            key=lambda p: (p[0] - pred_cx) ** 2 + (p[1] - pred_cy) ** 2,
        )
        dx = best_cx - pred_cx
        dy = best_cy - pred_cy
        jump = float(np.hypot(dx, dy))
        if jump > max_jump_px:
            s = max_jump_px / (jump + 1e-12)
            best_cx = pred_cx + dx * s
            best_cy = pred_cy + dy * s

        # Smooth and commit state.
        new_cx = (1.0 - smooth_alpha) * pred_cx + smooth_alpha * best_cx
        new_cy = (1.0 - smooth_alpha) * pred_cy + smooth_alpha * best_cy

        # Alpha-beta like filter to reduce frame jitter while preserving motion.
        fcx, fcy = self.filt_center
        fvx, fvy = self.filt_velocity
        pred_fx = fcx + fvx
        pred_fy = fcy + fvy
        rx = new_cx - pred_fx
        ry = new_cy - pred_fy
        nfvx = fvx + process_blend * rx
        nfvy = fvy + process_blend * ry
        dax = nfvx - fvx
        day = nfvy - fvy
        da = float(np.hypot(dax, day))
        if da > max_accel_px:
            s = max_accel_px / (da + 1e-12)
            nfvx = fvx + dax * s
            nfvy = fvy + day * s
        nfcx = pred_fx + smooth_alpha * rx
        nfcy = pred_fy + smooth_alpha * ry

        self.filt_center = (nfcx, nfcy)
        self.filt_velocity = (nfvx, nfvy)
        self.last_velocity = (nfcx - self.last_center[0], nfcy - self.last_center[1])
        self.last_center = (nfcx, nfcy)

        nx = int(round(nfcx - w / 2.0))
        ny = int(round(nfcy - h / 2.0))
        nx = int(np.clip(nx, 0, frame_gray.shape[1] - w))
        ny = int(np.clip(ny, 0, frame_gray.shape[0] - h))
        self.bbox = (nx, ny, w, h)

        # Adapt template slowly to mild illumination changes.
        # Only adapt template when color lock is strong to avoid drifting onto table texture.
        if used_color_lock:
            patch = frame_gray[ny : ny + h, nx : nx + w]
            if patch.shape == self.template_gray.shape:
                self.template_gray = (
                    (1.0 - template_update_rate) * self.template_gray
                    + template_update_rate * patch
                ).astype(np.uint8)

        return float(nfcx), float(nfcy)


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = np.median(xp[i : i + k])
    return out


def _trim_rows_by_motion(
    rows: list[dict[str, float | int]],
    start_speed_threshold_mps: float,
    end_speed_threshold_mps: float,
    sustain_frames: int,
) -> tuple[list[dict[str, float | int]], int, int]:
    """Trim rows to [first moving frame, last moving frame]."""
    n = len(rows)
    if n < 3:
        return rows, 0, max(0, n - 1)

    t = np.array([float(r["time_s"]) for r in rows], dtype=np.float64)
    dt = np.diff(t)
    valid_dt = dt > 1e-9
    if not np.any(valid_dt):
        return rows, 0, n - 1

    any_moving_start = np.zeros(n, dtype=bool)
    any_moving_end = np.zeros(n, dtype=bool)
    for label in BALL_LABELS:
        x = np.array([float(r[f"{label}_x_m"]) for r in rows], dtype=np.float64)
        y = np.array([float(r[f"{label}_y_m"]) for r in rows], dtype=np.float64)
        speed = np.zeros(n, dtype=np.float64)
        dxy = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        speed[1:] = np.where(valid_dt, dxy / np.maximum(dt, 1e-9), 0.0)
        any_moving_start |= speed > start_speed_threshold_mps
        any_moving_end |= speed > end_speed_threshold_mps

    k = max(1, int(sustain_frames))
    start_idx = 0
    end_idx = n - 1

    for i in range(0, n - k + 1):
        if np.all(any_moving_start[i : i + k]):
            start_idx = i
            break
    else:
        nz = np.flatnonzero(any_moving_start)
        if nz.size > 0:
            start_idx = int(nz[0])

    for i in range(n - k, -1, -1):
        if np.all(any_moving_end[i : i + k]):
            end_idx = i + k - 1
            break
    else:
        nz = np.flatnonzero(any_moving_end)
        if nz.size > 0:
            end_idx = int(nz[-1])

    if end_idx < start_idx:
        return rows, 0, n - 1

    trimmed = rows[start_idx : end_idx + 1]
    if not trimmed:
        return rows, 0, n - 1

    t0 = float(trimmed[0]["time_s"])
    for i, row in enumerate(trimmed):
        row["frame"] = i
        row["time_s"] = float(row["time_s"]) - t0

    return trimmed, start_idx, end_idx


def _collect_points(
    frame: np.ndarray, window_name: str, n_points: int, prompt: str
) -> np.ndarray:
    points: list[tuple[int, int]] = []
    display = frame.copy()

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = display.copy()
        cv2.putText(
            canvas,
            prompt,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Points: {len(points)}/{n_points} (enter=confirm, r=reset)",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for i, (px, py) in enumerate(points):
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(15) & 0xFF
        if key in (13, 10) and len(points) == n_points:  # enter
            break
        if key == ord("r"):
            points.clear()
        if key == 27:  # esc
            raise RuntimeError("Point selection cancelled by user.")

    cv2.destroyWindow(window_name)
    return np.asarray(points, dtype=np.float32)


def _select_rois(
    first_frame: np.ndarray, pad_scale: float
) -> list[tuple[int, int, int, int]]:
    rois: list[tuple[int, int, int, int]] = []
    for label in BALL_LABELS:
        roi = cv2.selectROI(
            "Select ROI",
            first_frame,
            fromCenter=False,
            showCrosshair=True,
        )
        x, y, w, h = map(int, roi)
        if w <= 0 or h <= 0:
            raise RuntimeError(f"Invalid ROI for {label} ball.")
        cx = x + w / 2.0
        cy = y + h / 2.0
        pw = int(round(w * pad_scale))
        ph = int(round(h * pad_scale))
        pw = max(pw, w)
        ph = max(ph, h)
        nx = int(round(cx - pw / 2.0))
        ny = int(round(cy - ph / 2.0))
        nx = int(np.clip(nx, 0, first_frame.shape[1] - 1))
        ny = int(np.clip(ny, 0, first_frame.shape[0] - 1))
        pw = int(min(pw, first_frame.shape[1] - nx))
        ph = int(min(ph, first_frame.shape[0] - ny))
        rois.append((nx, ny, pw, ph))
    cv2.destroyWindow("Select ROI")
    return rois


def _build_homography(table_corners_px: np.ndarray) -> np.ndarray:
    dst = np.array(
        [
            [0.0, 0.0],
            [TABLE_WIDTH_M, 0.0],
            [TABLE_WIDTH_M, TABLE_HEIGHT_M],
            [0.0, TABLE_HEIGHT_M],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(table_corners_px.astype(np.float32), dst)


def _build_homography_with_orientation(
    table_corners_px: np.ndarray, orientation: Literal["auto", "horizontal", "vertical"]
) -> tuple[np.ndarray, str]:
    """Build homography while handling horizontal/vertical table orientation."""
    tl, tr, br, bl = table_corners_px.astype(np.float32)
    horiz_len = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    vert_len = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))

    if orientation == "auto":
        used = "vertical" if vert_len > horiz_len else "horizontal"
    else:
        used = orientation

    # Keep output metric coordinates as x in [0, TABLE_WIDTH], y in [0, TABLE_HEIGHT].
    # If table appears vertical in image, rotate source correspondence so long side maps to x.
    if used == "horizontal":
        src = np.array([tl, tr, br, bl], dtype=np.float32)
    else:
        src = np.array([tl, bl, br, tr], dtype=np.float32)

    dst = np.array(
        [
            [0.0, 0.0],
            [TABLE_WIDTH_M, 0.0],
            [TABLE_WIDTH_M, TABLE_HEIGHT_M],
            [0.0, TABLE_HEIGHT_M],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return H, used


def _project_points(H: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    points = points_px.reshape(-1, 1, 2).astype(np.float32)
    xy = cv2.perspectiveTransform(points, H).reshape(-1, 2)
    return xy.astype(np.float64)


def _fit_points_inside_table_by_zoom(
    points_m: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Uniformly zoom points toward table center so all centers stay inside walls."""
    xmin = BALL_RADIUS_M + WALL_CLEARANCE_EPS_M
    xmax = TABLE_WIDTH_M - BALL_RADIUS_M - WALL_CLEARANCE_EPS_M
    ymin = BALL_RADIUS_M + WALL_CLEARANCE_EPS_M
    ymax = TABLE_HEIGHT_M - BALL_RADIUS_M - WALL_CLEARANCE_EPS_M

    center = np.array([0.5 * TABLE_WIDTH_M, 0.5 * TABLE_HEIGHT_M], dtype=np.float64)
    delta = points_m - center[None, :]
    max_abs_x = float(np.max(np.abs(delta[:, 0]))) if delta.size else 0.0
    max_abs_y = float(np.max(np.abs(delta[:, 1]))) if delta.size else 0.0

    allowed_abs_x = min(center[0] - xmin, xmax - center[0])
    allowed_abs_y = min(center[1] - ymin, ymax - center[1])

    sx = 1.0 if max_abs_x <= 1e-12 else allowed_abs_x / max_abs_x
    sy = 1.0 if max_abs_y <= 1e-12 else allowed_abs_y / max_abs_y
    scale = float(min(1.0, sx, sy))
    if scale < 1.0:
        scale *= 0.999  # tiny guard margin from exact wall contact

    corrected = center[None, :] + scale * delta
    changed = int(
        np.count_nonzero(np.any(np.abs(corrected - points_m) > 1e-12, axis=1))
    )
    return corrected, scale, changed


def _load_selection_cache(
    cache_path: Path,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | tuple[None, None]:
    try:
        payload = json.loads(cache_path.read_text())
        c = np.asarray(payload["table_corners"], dtype=np.float32)
        assert c.shape == (4, 2)
        corners = c
        rois = payload["rois"]
        return corners, rois
    except Exception:
        return None, None


def _save_selection_cache(
    cache_path: Path,
    table_corners: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    frame_shape: tuple[int, int, int],
) -> None:
    payload = {
        "table_corners": np.asarray(table_corners, dtype=float).tolist(),
        "rois": [list(map(int, roi)) for roi in rois],
        "frame_shape": list(frame_shape),
        "labels": list(BALL_LABELS),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2))


@click.command(context_settings={"show_default": True})
@click.option("--video", type=str, required=True, help="Path to source video")
@click.option(
    "--search-scale",
    type=float,
    default=3.2,
    help="Search window scale around previous position",
)
@click.option(
    "--color-min-area-ratio",
    type=float,
    default=0.08,
    help="Minimum contour area as fraction of ROI area for color detection",
)
@click.option(
    "--max-jump-px",
    type=float,
    default=42.0,
    help="Maximum allowed per-frame center jump in pixels (anti-jump gate)",
)
@click.option(
    "--smooth-alpha",
    type=float,
    default=0.55,
    help="Tracker smoothing factor in [0,1], larger follows detections faster",
)
@click.option(
    "--filter-blend",
    type=float,
    default=0.22,
    help="Velocity filter blend gain (smaller = smoother, larger = more reactive)",
)
@click.option(
    "--max-accel-px",
    type=float,
    default=14.0,
    help="Maximum per-frame acceleration of tracked center in pixels",
)
@click.option(
    "--median-window",
    type=int,
    default=5,
    help="Final median filter window on tracked center series (odd preferred, 1 disables)",
)
@click.option(
    "--motion-threshold-mps",
    type=float,
    default=0.15,
    help="Speed threshold (m/s) to detect motion start",
)
@click.option(
    "--motion-end-threshold-mps",
    type=float,
    default=0.01,
    help="Speed threshold (m/s) to detect motion end (closing trim)",
)
@click.option(
    "--motion-sustain-frames",
    type=int,
    default=3,
    help="Consecutive frames required to confirm movement state",
)
@click.option(
    "--plot-table-bounds-only",
    is_flag=True,
    help="Plot trajectories with axis fixed to table bounds only",
)
@click.option(
    "--orientation",
    type=click.Choice(["auto", "horizontal", "vertical"]),
    default="auto",
    help="Table orientation in video for homography mapping",
)
def main(
    video: str,
    search_scale: float,
    color_min_area_ratio: float,
    max_jump_px: float,
    smooth_alpha: float,
    filter_blend: float,
    max_accel_px: float,
    median_window: int,
    motion_threshold_mps: float,
    motion_end_threshold_mps: float,
    motion_sustain_frames: int,
    plot_table_bounds_only: bool,
    orientation: str,
) -> None:
    # Paths
    video_path = Path(video)
    cache_path = video_path.with_suffix(".extract_cache.json")
    overlay_video_path = video_path.with_name(f"{video_path.stem}_tracked.mp4")
    output_csv = video_path.with_suffix(".csv")
    meta_path = video_path.with_name(f"{video_path.stem}_meta.txt")

    output_png = output_csv.with_suffix(".png")
    output_time_png = output_csv.with_name(f"{output_csv.stem}_position_vs_time.png")
    output_speed_png = output_csv.with_name(f"{output_csv.stem}_speed_vs_time.png")

    # Open video
    cap = cv2.VideoCapture(video_path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path.as_posix()}")
    ok, first_frame = cap.read()
    h, w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not ok:
        raise RuntimeError("Could not read first frame.")

    # Open overlaying video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(
        overlay_video_path.as_posix(),
        fourcc,
        fps,
        (w, h),
    )
    if not overlay_writer.isOpened() or overlay_writer is None:
        raise RuntimeError(f"Failed to open overlay video writer: {overlay_video_path}")

    cached_corners, cached_rois = _load_selection_cache(cache_path)
    if cached_corners is not None or cached_rois is not None:
        print(f"Loaded selection cache: {cache_path}")

    # Detect corners and board (ROI)
    if cached_corners is not None and cached_rois is not None:
        table_corners = cached_corners
        rois = cached_rois
    else:
        raw_corners = _collect_points(
            first_frame,
            window_name="Table Corners",
            n_points=4,
            prompt="Click 4 table corners (any order). Enter=confirm",
        )
        table_corners = _order_corners_tl_tr_br_bl(raw_corners)
        print("Select ROI for WHITE, then RED, then YELLOW ball.")
        rois = _select_rois(first_frame, pad_scale=1.2)

        _save_selection_cache(cache_path, table_corners, rois, first_frame.shape)
        print(f"Saved selection cache: {cache_path}")

    # Compute homography (table could be vertical or horizontal.)
    H, used_orientation = _build_homography_with_orientation(
        table_corners,
        orientation=orientation,
    )
    print(
        f"Homography orientation mode: requested={orientation}, used={used_orientation}"
    )

    # Tracking ball location
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    trackers: list[TemplateTracker] = []
    for label, (x, y, w, h) in zip(BALL_LABELS, rois):
        tmpl = first_gray[y : y + h, x : x + w].copy()
        trackers.append(
            TemplateTracker(label=label, template_gray=tmpl, bbox=(x, y, w, h))
        )

    rows: list[dict[str, float | int]] = []
    in_bounds_count = {label: 0 for label in BALL_LABELS}
    wall_fit_scale = 1.0
    wall_fit_changed_points = 0
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar = tqdm(
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        desc="Extracting trajectory",
        unit="frame",
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track balls in each frame
        centers_px = np.zeros((3, 2), dtype=np.float64)
        for i, tracker in enumerate(trackers):
            cx, cy = tracker.update(
                frame,
                gray,
                search_scale=search_scale,
                color_min_area_ratio=color_min_area_ratio,
                max_jump_px=max_jump_px,
                smooth_alpha=smooth_alpha,
                process_blend=filter_blend,
                max_accel_px=max_accel_px,
            )
            centers_px[i, :] = [cx, cy]

        centers_m = _project_points(H, centers_px)

        # Overlay
        overlay = frame.copy()
        for i, label in enumerate(BALL_LABELS):
            x = int(round(centers_px[i, 0]))
            y = int(round(centers_px[i, 1]))
            cv2.circle(overlay, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(
                overlay,
                label,
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            f"frame={frame_idx}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        overlay_writer.write(overlay)

        # Record trajectory
        row: dict[str, float | int] = {
            "frame": frame_idx,
            "time_s": frame_idx / fps,
        }
        for i, label in enumerate(BALL_LABELS):
            row[f"{label}_x_px"] = float(centers_px[i, 0])
            row[f"{label}_y_px"] = float(centers_px[i, 1])
            row[f"{label}_x_m"] = float(centers_m[i, 0])
            row[f"{label}_y_m"] = float(centers_m[i, 1])
            in_bounds = (0.0 <= centers_m[i, 0] <= TABLE_WIDTH_M) and (
                0.0 <= centers_m[i, 1] <= TABLE_HEIGHT_M
            )
            row[f"{label}_in_bounds"] = int(in_bounds)
            if in_bounds:
                in_bounds_count[label] += 1
        rows.append(row)

        frame_idx += 1
        pbar.update(1)

    # Release all
    pbar.close()
    cap.release()
    overlay_writer.release()
    cv2.destroyAllWindows()
    print(f"Saved overlay video: {overlay_video_path}")

    if not rows:
        raise RuntimeError("No tracking rows produced.")

    # Optional post-filter to suppress brief jitter bursts near contact events.
    if median_window > 1:
        for label in BALL_LABELS:
            x = np.array(
                [float(row[f"{label}_x_px"]) for row in rows], dtype=np.float64
            )
            y = np.array(
                [float(row[f"{label}_y_px"]) for row in rows], dtype=np.float64
            )
            xf = _median_filter_1d(x, median_window)
            yf = _median_filter_1d(y, median_window)
            pts_px = np.stack([xf, yf], axis=1)
            pts_m = _project_points(H, pts_px)
            for i in range(len(rows)):
                rows[i][f"{label}_x_px"] = float(xf[i])
                rows[i][f"{label}_y_px"] = float(yf[i])
                rows[i][f"{label}_x_m"] = float(pts_m[i, 0])
                rows[i][f"{label}_y_m"] = float(pts_m[i, 1])

    # Trim
    n_before = len(rows)
    rows, start_idx, end_idx = _trim_rows_by_motion(
        rows,
        start_speed_threshold_mps=float(motion_threshold_mps),
        end_speed_threshold_mps=float(motion_end_threshold_mps),
        sustain_frames=int(motion_sustain_frames),
    )
    trim_info = (n_before, len(rows), start_idx, end_idx)
    print(
        f"Motion trim: kept {len(rows)}/{n_before} rows (source frame range {start_idx}..{end_idx})"
    )

    # Wall clearance : Zoom slightly to avoid initial contact
    all_pts = []
    for row in rows:
        for label in BALL_LABELS:
            all_pts.append([float(row[f"{label}_x_m"]), float(row[f"{label}_y_m"])])
    all_pts_arr = np.asarray(all_pts, dtype=np.float64)
    fitted_pts, wall_fit_scale, wall_fit_changed_points = (
        _fit_points_inside_table_by_zoom(all_pts_arr)
    )
    idx = 0
    for row in rows:
        for label in BALL_LABELS:
            row[f"{label}_x_m"] = float(fitted_pts[idx, 0])
            row[f"{label}_y_m"] = float(fitted_pts[idx, 1])
            idx += 1
    print(
        f"Wall-fit zoom scale applied: {wall_fit_scale:.6f} "
        f"(changed points: {wall_fit_changed_points})"
    )

    # Export
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved trajectory CSV: {output_csv}")

    with meta_path.open("w") as fh:
        fh.write("table_corners_px (TL,TR,BR,BL):\n")
        fh.write(f"{table_corners.tolist()}\n")
        fh.write(f"orientation_used={used_orientation}\n")
        fh.write("homography_pixel_to_meter:\n")
        fh.write(f"{H.tolist()}\n")
        fh.write(f"fps={fps}\n")
        fh.write(f"frames={len(rows)}\n")

    # -------- Diagnostics --------- #

    # Plot flattened metric trajectories and save as PNG with the same base name.
    traj = np.zeros((len(rows), 3, 2), dtype=np.float64)
    for i, row in enumerate(rows):
        for b, label in enumerate(BALL_LABELS):
            traj[i, b, 0] = float(row[f"{label}_x_m"])
            traj[i, b, 1] = float(row[f"{label}_y_m"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted_points = bp.plot_full_xy_trajectories(ax, traj)
    bp.add_table_bounds(ax, TABLE_WIDTH_M, TABLE_HEIGHT_M)
    bp.set_xy_limits(
        ax,
        traj,
        TABLE_WIDTH_M,
        TABLE_HEIGHT_M,
        bounds_only=plot_table_bounds_only,
    )
    bp.style_xy_axis(ax)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    print(f"Saved trajectory plot: {output_png}")

    # Plot metric position over time (x/y) for each ball.
    t = np.array([float(row["time_s"]) for row in rows], dtype=np.float64)
    fig_ts, axes_ts = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    bp.plot_position_vs_time(axes_ts[0], axes_ts[1], t, traj)
    fig_ts.tight_layout()
    fig_ts.savefig(output_time_png, dpi=180)
    plt.close(fig_ts)
    print(f"Saved position-time plot: {output_time_png}")

    # Plot speed over time for each ball (from flattened XY trajectory).
    speed = bp.compute_speed_from_traj(t, traj)
    fig_v, ax_v = plt.subplots(figsize=(9, 3.8))
    bp.plot_speed_vs_time(ax_v, t, speed)
    fig_v.tight_layout()
    fig_v.savefig(output_speed_png, dpi=180)
    plt.close(fig_v)
    print(f"Saved speed-time plot: {output_speed_png}")

    print(f"Rows: {len(rows)}, FPS used: {fps:.3f}")
    print(
        "Plotted points per ball: "
        + ", ".join(f"{label}={plotted_points[label]}" for label in BALL_LABELS)
    )


if __name__ == "__main__":
    main()
