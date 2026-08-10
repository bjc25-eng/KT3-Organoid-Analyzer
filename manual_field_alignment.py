from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ManualTransform:
    scale: float
    rotation_deg: float
    shift_x_px: float
    shift_y_px: float


def affine_day7_to_day10(
    day7_shape: tuple[int, int] | tuple[int, int, int],
    day10_shape: tuple[int, int] | tuple[int, int, int],
    transform: ManualTransform,
) -> np.ndarray:
    """Return a 2x3 affine matrix mapping Day-7 pixel coordinates onto Day 10.

    The Day-7 image is scaled/rotated about its centre, then its centre is moved to
    the Day-10 image centre plus the user-specified x/y shift. This makes the shift
    controls intuitive when Day 10 has the larger field of view.
    """
    h7, w7 = day7_shape[:2]
    h10, w10 = day10_shape[:2]
    c7 = (w7 / 2.0, h7 / 2.0)
    c10 = np.array([w10 / 2.0, h10 / 2.0], dtype=float)
    matrix = cv2.getRotationMatrix2D(c7, float(transform.rotation_deg), float(transform.scale))
    matrix[:, 2] += c10 + np.array([transform.shift_x_px, transform.shift_y_px], dtype=float) - np.array(c7)
    return matrix.astype(np.float64)


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    return cv2.transform(points.reshape(-1, 1, 2).astype(np.float64), matrix).reshape(-1, 2)


def warp_day7_to_day10(day7_rgb: np.ndarray, day10_rgb: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h10, w10 = day10_rgb.shape[:2]
    warped = cv2.warpAffine(
        day7_rgb,
        matrix,
        (w10, h10),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask_src = np.full(day7_rgb.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpAffine(
        mask_src,
        matrix,
        (w10, h10),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, mask


def alpha_overlay(day10_rgb: np.ndarray, warped_day7_rgb: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    base = day10_rgb.astype(np.float32)
    over = warped_day7_rgb.astype(np.float32)
    out = base.copy()
    valid = mask > 0
    out[valid] = (1.0 - opacity) * base[valid] + opacity * over[valid]
    return np.clip(out, 0, 255).astype(np.uint8)


def green_excess(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    excess = np.clip(g - 0.5 * (r + b), 0, None)
    positive = excess[excess > 0]
    if positive.size:
        hi = float(np.percentile(positive, 99.5))
        if hi > 0:
            excess = np.clip(excess / hi, 0, 1)
    return excess


def pdo_pattern_overlay(day10_rgb: np.ndarray, warped_day7_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Show Day 10 GFP-excess in green and aligned Day 7 GFP-excess in magenta."""
    gray = cv2.cvtColor(day10_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    background = np.repeat((gray[..., None] * 0.28), 3, axis=2)
    d10 = green_excess(day10_rgb)
    d7 = green_excess(warped_day7_rgb) * (mask > 0)
    out = background
    out[..., 1] += 0.95 * d10
    out[..., 0] += 0.95 * d7
    out[..., 2] += 0.95 * d7
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def draw_markers(
    rgb: np.ndarray,
    day7_points: np.ndarray | None = None,
    day10_points: np.ndarray | None = None,
    radius: int = 8,
) -> np.ndarray:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    r = int(max(2, radius))
    if day7_points is not None:
        for x, y in np.asarray(day7_points, dtype=float):
            draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 0, 255), width=2)
    if day10_points is not None:
        for x, y in np.asarray(day10_points, dtype=float):
            draw.rectangle((x-r, y-r, x+r, y+r), outline=(0, 255, 0), width=2)
    return np.asarray(image)


def mutual_nearest_well_matches(
    day7_wells: pd.DataFrame,
    day10_wells: pd.DataFrame,
    matrix: np.ndarray,
) -> pd.DataFrame:
    """Map every mutually-nearest physical well and report geometric error.

    No QC threshold is applied here. The user chooses which errors are acceptable
    after visually aligning the whole field.
    """
    if day7_wells.empty or day10_wells.empty:
        return pd.DataFrame()

    p7 = day7_wells[["well_centre_x_px", "well_centre_y_px"]].to_numpy(float)
    p7t = transform_points(p7, matrix)
    p10 = day10_wells[["well_centre_x_px", "well_centre_y_px"]].to_numpy(float)

    tree10 = cKDTree(p10)
    dist, idx10 = tree10.query(p7t, k=1)
    tree7 = cKDTree(p7t)
    _, reverse_idx = tree7.query(p10[idx10], k=1)
    keep = reverse_idx == np.arange(len(p7t))

    rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(keep):
        j = int(idx10[i])
        r7 = day7_wells.iloc[i]
        r10 = day10_wells.iloc[j]
        umpp10 = float(r10.get("um_per_pixel", np.nan))
        error_px = float(dist[i])
        rows.append(
            {
                "day7_well_index": r7["well_index"],
                "day10_well_index": r10["well_index"],
                "day7_well_x_px": float(r7["well_centre_x_px"]),
                "day7_well_y_px": float(r7["well_centre_y_px"]),
                "day7_transformed_x_px": float(p7t[i, 0]),
                "day7_transformed_y_px": float(p7t[i, 1]),
                "day10_well_x_px": float(r10["well_centre_x_px"]),
                "day10_well_y_px": float(r10["well_centre_y_px"]),
                "match_error_px": error_px,
                "match_error_um": error_px * umpp10 if np.isfinite(umpp10) else np.nan,
                "day7_PDO_count": int(r7.get("PDO_count", 0)),
                "day10_PDO_count": int(r10.get("PDO_count", 0)),
                "day7_PSC_count": int(r7.get("PSC_like_focus_count", 0)),
                "day10_PSC_count": int(r10.get("PSC_like_focus_count", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("match_error_px").reset_index(drop=True)


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
