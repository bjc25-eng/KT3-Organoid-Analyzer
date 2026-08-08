from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import zarr
from PIL import Image, ImageDraw


def _read_scale(root) -> tuple[float, float]:
    try:
        ms = root.attrs['multiscales'][0]
        scale = ms['datasets'][0]['coordinateTransformations'][0]['scale']
        return float(scale[-1]), float(scale[-2])
    except Exception:
        return 1.0, 1.0


def _channel_labels(root, count: int) -> list[str]:
    try:
        rows = root.attrs.get('omero', {}).get('channels', [])
        labels = [str(row.get('label', f'Channel {i}')) for i, row in enumerate(rows)]
    except Exception:
        labels = []
    while len(labels) < count:
        labels.append(f'Channel {len(labels)}')
    return labels[:count]


def _intensity_max(root, channel: int, dtype) -> float:
    try:
        row = root.attrs.get('omero', {}).get('channels', [])[channel]
        end = float(row.get('window', {}).get('end'))
        if end > 0:
            return end
    except Exception:
        pass
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def _u8_absolute(arr: np.ndarray, maximum: float) -> np.ndarray:
    """Absolute fluorescence scaling using the acquisition bit-depth/window."""
    a = np.asarray(arr, dtype=np.float32)
    return np.clip(a * (255.0 / max(float(maximum), 1.0)), 0, 255).astype(np.uint8)


def _u8_local_contrast(arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> tuple[np.ndarray, float, float]:
    """Percentile-normalise a structural/DIC tile for geometry detection.

    DIC is used here only to reveal microwell edges.  Its absolute grey level is
    not a biological measurement, so stretching each tile independently is
    appropriate and avoids collapsing a narrow raw range into only a few uint8
    levels.
    """
    a = np.asarray(arr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8), 0.0, 0.0
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8), lo, hi
    out = np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return out, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description='Make a central DIC+GFP OME-Zarr QC preview without loading the whole dataset.')
    ap.add_argument('source', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('omezarr_preview'))
    ap.add_argument('--tile', type=int, default=4096)
    ap.add_argument('--gfp-channel', type=int, default=0)
    ap.add_argument('--dic-channel', type=int, default=1)
    ap.add_argument('--well-diameter-um', type=float, default=100.0)
    ap.add_argument('--radius-tolerance', type=float, default=0.20, help='Engineering Hough search margin around expected physical well radius; default ±20%%.')
    ap.add_argument('--hough-p2', type=float, default=27.0)
    args = ap.parse_args()

    source = args.source.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(source), mode='r')
    arr = root['0']
    if arr.ndim != 3:
        raise RuntimeError(f'Expected C,Y,X OME-Zarr; got shape {arr.shape}')
    c, h, w = map(int, arr.shape)
    labels = _channel_labels(root, c)
    px_x_um, px_y_um = _read_scale(root)
    px_um = (px_x_um + px_y_um) / 2.0
    expected_radius = float(args.well_diameter_um) / (2.0 * px_um)
    tol = max(0.05, float(args.radius_tolerance))
    rmin = max(2, int(round(expected_radius * (1.0 - tol))))
    rmax = max(rmin + 2, int(round(expected_radius * (1.0 + tol))))
    min_dist = max(2, int(round(expected_radius * 1.5)))

    tile = min(int(args.tile), h, w)
    y0 = max(0, h // 2 - tile // 2)
    x0 = max(0, w // 2 - tile // 2)
    y1, x1 = y0 + tile, x0 + tile

    gfp_raw = np.asarray(arr[int(args.gfp_channel), y0:y1, x0:x1])
    dic_raw = np.asarray(arr[int(args.dic_channel), y0:y1, x0:x1])

    # Preserve absolute fluorescence scaling for GFP so thresholds remain tied
    # to the acquisition intensity range.  DIC is structural only, so use a
    # local percentile stretch to expose microwell edges robustly.
    gfp = _u8_absolute(gfp_raw, _intensity_max(root, int(args.gfp_channel), arr.dtype))
    dic, dic_p_low, dic_p_high = _u8_local_contrast(dic_raw)

    blur = cv2.GaussianBlur(dic, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.15, minDist=float(min_dist),
        param1=75.0, param2=float(args.hough_p2), minRadius=rmin, maxRadius=rmax,
    )
    wells = [] if circles is None else np.round(circles[0]).astype(int).tolist()

    dic_rgb = np.stack([dic, dic, dic], axis=-1)
    overlay = dic_rgb.copy()
    overlay[..., 1] = np.maximum(overlay[..., 1], gfp)
    Image.fromarray(overlay).save(out/'central_DIC_GFP_overlay.png')
    Image.fromarray(dic).save(out/'central_DIC.png')
    Image.fromarray(gfp).save(out/'central_GFP.png')

    marked = Image.fromarray(overlay)
    draw = ImageDraw.Draw(marked)
    for x, y, r in wells:
        draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 255, 255), width=2)
    marked.save(out/'central_well_detection.png')

    summary = {
        'source': str(source),
        'shape_cyx': [c, h, w],
        'channel_labels': labels,
        'gfp_channel': int(args.gfp_channel),
        'dic_channel': int(args.dic_channel),
        'pixel_size_um': {'x': px_x_um, 'y': px_y_um},
        'preview_origin_fullres_px': {'x': x0, 'y': y0},
        'preview_size_px': tile,
        'well_diameter_um': float(args.well_diameter_um),
        'expected_well_radius_px': expected_radius,
        'hough_radius_search_px': [rmin, rmax],
        'hough_min_dist_px': min_dist,
        'hough_p2': float(args.hough_p2),
        'detected_wells_in_preview': len(wells),
        'gfp_raw_min': int(np.min(gfp_raw)),
        'gfp_raw_max': int(np.max(gfp_raw)),
        'dic_raw_min': int(np.min(dic_raw)),
        'dic_raw_max': int(np.max(dic_raw)),
        'dic_local_contrast_low_raw': dic_p_low,
        'dic_local_contrast_high_raw': dic_p_high,
        'dic_uint8_min': int(dic.min()),
        'dic_uint8_max': int(dic.max()),
    }
    (out/'preview_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Preview files written to: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
