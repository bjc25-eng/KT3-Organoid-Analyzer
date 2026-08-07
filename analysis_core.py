from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

APP_TITLE = 'KT3 PDO + PSC Microwell Analyzer'
GFP_MODE = 'GFP-labelled (green fluorescence)'
BRIGHTFIELD_MODE = 'Not GFP-labelled (brightfield / phase contrast)'
PSC_PRESENT = 'RFP-labelled PSC/stromal cells present'
PSC_ABSENT = 'No RFP-labelled PSC/stromal cells'
ML_SCHEMA_VERSION = '1.0'


@dataclass
class Settings:
    well_diameter_um: float = 100.0
    well_rmin: int = 23
    well_rmax: int = 40
    well_spacing: int = 54
    hough_p2: float = 27.0
    green_low: float = 30.0
    green_high: float = 45.0
    pdo_min_area: int = 20
    split_pdos: bool = True
    pdo_peak_distance: int = 18
    psc_peak_threshold: float = 9.0
    psc_red_minimum: float = 12.0
    psc_peak_distance: int = 4
    histogram_bins: int = 12
    organoid_mode: str = GFP_MODE
    rfp_psc_present: bool = True
    brightfield_contrast_threshold: float = 10.0
    brightfield_min_area: int = 80


def fonts(a=22, b=17):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', a), ImageFont.truetype('DejaVuSans.ttf', b)
    except Exception:
        return ImageFont.load_default(), ImageFont.load_default()


def natural_key(p: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', p.name)]


def infer_series(name: str, fallback: int):
    m = re.search(r'series\s*0*(\d+)', name, re.I)
    return int(m.group(1)) if m else fallback


def slugify(text: str):
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text).strip()).strip('_')
    return s or 'unnamed'


def stable_token(text: str, fallback='unknown'):
    s = slugify(text)
    return s if s != 'unnamed' else fallback


def detect_wells(rgb, s):
    rmin = max(1, int(round(s.well_rmin)))
    rmax = max(rmin + 1, int(round(s.well_rmax)))
    spacing = max(1, int(round(s.well_spacing)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.15, minDist=float(spacing),
        param1=75.0, param2=float(s.hough_p2),
        minRadius=int(rmin), maxRadius=int(rmax)
    )
    if circles is None:
        return np.empty((0, 3), dtype=int)
    circles = np.round(circles[0]).astype(int)
    kept = []
    for x, y, r in circles[np.argsort(circles[:, 0])]:
        if all((x-a)**2 + (y-b)**2 > 20**2 for a, b, _ in kept):
            kept.append((int(x), int(y), int(r)))
    return np.asarray(kept, dtype=int)


def cluster(vals, tol=12):
    vals = sorted(map(float, vals))
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - np.mean(groups[-1])) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(x)) for x in groups]


def grid_index(x, y, xs, ys):
    return int(np.argmin([abs(x-v) for v in xs])) + 1, int(np.argmin([abs(y-v) for v in ys])) + 1


def green_excess(rgb):
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return gaussian_filter(g - np.maximum(r, b), 0.8)


def segment_pdos(green, s):
    labs = label(green > float(s.green_low))
    out = []
    min_area = int(s.pdo_min_area)
    peak_dist = int(s.pdo_peak_distance)
    for reg in regionprops(labs, intensity_image=green):
        if reg.area < min_area or reg.intensity_max < float(s.green_high):
            continue
        y0, x0, y1, x1 = reg.bbox
        mask = labs[y0:y1, x0:x1] == reg.label
        sub = green[y0:y1, x0:x1]
        pieces = None
        if s.split_pdos:
            peaks = peak_local_max(
                sub, min_distance=peak_dist, threshold_abs=float(s.green_high),
                labels=mask.astype(np.uint8), exclude_border=False
            )
            if len(peaks) >= 2:
                markers = np.zeros_like(sub, np.int32)
                for i, (py, px) in enumerate(peaks, 1):
                    markers[py, px] = i
                ws = watershed(-sub, markers=markers, mask=mask)
                cand = [p for p in regionprops(ws, intensity_image=sub)
                        if p.area >= min_area and p.intensity_max >= float(s.green_high)]
                if len(cand) >= 2:
                    pieces = cand
            if pieces is None:
                dist = distance_transform_edt(mask)
                if float(dist.max()) > 0:
                    sp = peak_local_max(
                        dist, min_distance=12,
                        threshold_abs=max(7.0, 0.55 * float(dist.max())),
                        labels=mask.astype(np.uint8), exclude_border=False
                    )
                    if len(sp) == 2:
                        vals = [float(dist[tuple(p)]) for p in sp]
                        sep = float(np.linalg.norm(sp[0] - sp[1]))
                        if min(vals) / max(vals) >= 0.60 and sep >= 18:
                            markers = np.zeros_like(sub, np.int32)
                            for i, (py, px) in enumerate(sp, 1):
                                markers[py, px] = i
                            ws = watershed(-dist, markers=markers, mask=mask)
                            cand = [p for p in regionprops(ws, intensity_image=sub)
                                    if p.area >= max(220, min_area) and p.intensity_max >= float(s.green_high)]
                            if len(cand) == 2:
                                aa = sorted(float(p.area) for p in cand)
                                if aa[0] / aa[1] >= 0.35:
                                    pieces = cand
        if pieces:
            for p in pieces:
                cy, cx = p.centroid
                out.append({'x': x0 + float(cx), 'y': y0 + float(cy), 'area': float(p.area)})
        else:
            cy, cx = reg.centroid
            out.append({'x': float(cx), 'y': float(cy), 'area': float(reg.area)})
    return out


def _split_binary_region(mask, intensity, min_area, split_enabled):
    labs = label(mask)
    pieces = []
    for reg in regionprops(labs, intensity_image=intensity):
        if reg.area < min_area:
            continue
        y0, x0, y1, x1 = reg.bbox
        submask = labs[y0:y1, x0:x1] == reg.label
        chosen = None
        if split_enabled:
            dist = distance_transform_edt(submask)
            if float(dist.max()) > 2:
                min_dist = max(5, int(round(math.sqrt(float(reg.area)) * 0.18)))
                peaks = peak_local_max(
                    dist, min_distance=min_dist,
                    threshold_abs=max(2.0, 0.45 * float(dist.max())),
                    labels=submask.astype(np.uint8), exclude_border=False
                )
                if 2 <= len(peaks) <= 4:
                    markers = np.zeros_like(dist, dtype=np.int32)
                    for i, (py, px) in enumerate(peaks, 1):
                        markers[py, px] = i
                    ws = watershed(-dist, markers=markers, mask=submask)
                    cand = [p for p in regionprops(ws, intensity_image=intensity[y0:y1, x0:x1])
                            if p.area >= min_area]
                    if len(cand) >= 2:
                        chosen = []
                        for p in cand:
                            cy, cx = p.centroid
                            chosen.append((y0 + cy, x0 + cx, float(p.area)))
        if chosen:
            pieces.extend(chosen)
        else:
            cy, cx = reg.centroid
            pieces.append((cy, cx, float(reg.area)))
    return pieces


def _unlabelled_candidate_mask(rgb, x, y, r, s):
    R = max(10, int(round(r * 0.80)))
    x0, x1 = max(0, x-R), min(rgb.shape[1], x+R+1)
    y0, y1 = max(0, y-R), min(rgb.shape[0], y+R+1)
    sub = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(3.0, float(r) * 0.22)
    background = gaussian_filter(gray, sigma=sigma)
    contrast = gaussian_filter(np.abs(gray - background), 0.8)
    yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
    cx0, cy0 = x-x0, y-y0
    interior_r = max(5.0, float(r) * 0.70)
    interior = (xx-cx0)**2 + (yy-cy0)**2 <= interior_r**2
    binary = (contrast >= float(s.brightfield_contrast_threshold)) & interior
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
    binary = binary_fill_holes(binary)
    return binary, contrast, x0, y0, interior_r


def segment_unlabelled_pdos_in_well(rgb, x, y, r, s):
    binary, contrast, x0, y0, interior_r = _unlabelled_candidate_mask(rgb, x, y, r, s)
    min_area = max(1, int(s.brightfield_min_area))
    max_area = math.pi * (interior_r ** 2) * 0.80
    pieces = _split_binary_region(binary, contrast, min_area, bool(s.split_pdos))
    out = []
    for cy, cx, area in pieces:
        gx, gy = x0 + float(cx), y0 + float(cy)
        if (gx-x)**2 + (gy-y)**2 <= (0.68*r)**2 and area <= max_area:
            out.append({'x': gx, 'y': gy, 'area': float(area)})
    return out


def detect_psc(rgb, x, y, r, s):
    R = max(4, int(round(r * 0.86)))
    x0, x1 = max(0, x-R), min(rgb.shape[1], x+R+1)
    y0, y1 = max(0, y-R), min(rgb.shape[0], y+R+1)
    sub = rgb[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.ogrid[:sub.shape[0], :sub.shape[1]]
    mask = (xx-(x-x0))**2 + (yy-(y-y0))**2 <= R**2
    score = sub[..., 0] - sub[..., 2]
    score = score - gaussian_filter(score, 5.0)
    score[~mask] = -999
    peaks = peak_local_max(
        score, min_distance=int(s.psc_peak_distance),
        threshold_abs=float(s.psc_peak_threshold), exclude_border=False
    )
    foci = []
    for py, px in peaks:
        if sub[py, px, 0] - sub[py, px, 2] >= float(s.psc_red_minimum):
            foci.append((x0+int(px), y0+int(py), float(score[py, px])))
    return foci


def crop_square(rgb, x, y, r, scale=4):
    R = int(round(r * 1.75))
    x0, x1 = max(0, x-R), min(rgb.shape[1], x+R)
    y0, y1 = max(0, y-R), min(rgb.shape[0], y+R)
    im = Image.fromarray(rgb[y0:y1, x0:x1])
    side = max(im.size)
    c = Image.new('RGB', (side, side), 'black')
    c.paste(im, ((side-im.width)//2, (side-im.height)//2))
    return c.resize((side*scale, side*scale), Image.Resampling.NEAREST)


def labelled_crop(crop, series, well, pdo_n, psc_n, sizes):
    title, body = fonts()
    psc_text = 'not analysed' if psc_n is None or pd.isna(psc_n) else str(int(psc_n))
    lines = [
        f'Image {series:02d} | Well {well}',
        f'PDO count: {pdo_n} | PSC count: {psc_text}',
        'PDO size' + ('s' if len(sizes) != 1 else '') + ': ' + ', '.join(f'{v:.1f} µm' for v in sizes)
    ]
    dummy = Image.new('RGB', (10, 10))
    d = ImageDraw.Draw(dummy)
    widths = [d.textbbox((0, 0), t, font=(title if i == 0 else body))[2] for i, t in enumerate(lines)]
    header = 105
    W = max(crop.width, max(widths)+28)
    out = Image.new('RGB', (W, header+crop.height), 'white')
    out.paste(crop, ((W-crop.width)//2, header))
    dr = ImageDraw.Draw(out)
    dr.rectangle((0, 0, W, header), fill='black')
    ypos = 8
    for i, t in enumerate(lines):
        f = title if i == 0 else body
        dr.text((12, ypos), t, font=f, fill='white')
        ypos += 34 if i == 0 else 29
    return out


def indexed_overlay(rgb, wells):
    im = Image.fromarray(rgb).convert('RGB')
    dr = ImageDraw.Draw(im)
    _, f = fonts(18, 14)
    for w in wells:
        x, y, r = w['x'], w['y'], w['r']
        dr.ellipse((x-r, y-r, x+r, y+r), outline='yellow', width=1)
        dr.text((x-r, y-r-12), w['well'], fill='white', font=f)
    return im


def make_contact(paths, out, cols=5, gap=5):
    ims = [Image.open(p).convert('RGB') for p in paths]
    if not ims:
        return
    tw = min(620, max(i.width for i in ims))
    thumbs = []
    for im in ims:
        sc = tw / im.width
        thumbs.append(im.resize((tw, int(im.height*sc)), Image.Resampling.LANCZOS))
    ch = max(i.height for i in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new('RGB', (cols*tw+(cols+1)*gap, rows*ch+(rows+1)*gap), 'white')
    for i, im in enumerate(thumbs):
        rr, cc = divmod(i, cols)
        sheet.paste(im, (gap+cc*(tw+gap), gap+rr*(ch+gap)))
    sheet.save(out, dpi=(300, 300))


def zip_bytes(folder):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in Path(folder).rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(folder))
    return b.getvalue()


def _gfp_semantic_mask(rgb, circles, s):
    green = green_excess(rgb)
    labs = label(green > float(s.green_low))
    mask = np.zeros(green.shape, dtype=np.uint8)
    for reg in regionprops(labs, intensity_image=green):
        if reg.area >= int(s.pdo_min_area) and reg.intensity_max >= float(s.green_high):
            mask[labs == reg.label] = 255
    allowed = np.zeros_like(mask)
    for x, y, r in circles:
        cv2.circle(allowed, (int(x), int(y)), max(1, int(round(0.86*r))), 255, -1)
    return cv2.bitwise_and(mask, allowed)


def _brightfield_semantic_mask(rgb, circles, s):
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for x, y, r in circles:
        x, y, r = int(x), int(y), int(r)
        binary, contrast, x0, y0, interior_r = _unlabelled_candidate_mask(rgb, x, y, r, s)
        labs = label(binary)
        min_area = max(1, int(s.brightfield_min_area))
        max_area = math.pi * (interior_r ** 2) * 0.80
        for reg in regionprops(labs):
            if reg.area < min_area or reg.area > max_area:
                continue
            cy, cx = reg.centroid
            gx, gy = x0 + float(cx), y0 + float(cy)
            if (gx-x)**2 + (gy-y)**2 > (0.68*r)**2:
                continue
            yb0, xb0, yb1, xb1 = reg.bbox
            local_component = labs[yb0:yb1, xb0:xb1] == reg.label
            target = out[y0+yb0:y0+yb1, x0+xb0:x0+xb1]
            target[local_component] = 255
    return out


def make_training_masks(rgb, circles, psc_focus_rows, s):
    H, W = rgb.shape[:2]
    well_mask = np.zeros((H, W), dtype=np.uint8)
    for x, y, r in circles:
        cv2.circle(well_mask, (int(x), int(y)), int(r), 255, -1)
    if s.organoid_mode == GFP_MODE:
        pdo_mask = _gfp_semantic_mask(rgb, circles, s)
    else:
        pdo_mask = _brightfield_semantic_mask(rgb, circles, s)
    psc_mask = np.zeros((H, W), dtype=np.uint8)
    if s.rfp_psc_present:
        for row in psc_focus_rows:
            cv2.circle(psc_mask, (int(row['focus_x_px']), int(row['focus_y_px'])), 2, 255, -1)
    return well_mask, pdo_mask, psc_mask


def _empty_pdo_df():
    return pd.DataFrame(columns=[
        'image_series', 'well_index', 'organoid_detection_mode', 'GFP_labelled_organoids',
        'RFP_PSC_stromal_cells_present', 'PDO_number_in_well', 'PDO_count_in_well',
        'centroid_x_px', 'centroid_y_px', 'projected_area_px2',
        'equivalent_circular_diameter_um', 'PSC_like_focus_count_in_well'
    ])


def _empty_psc_df():
    return pd.DataFrame(columns=[
        'image_series', 'source_image', 'well_index', 'focus_number_in_well',
        'focus_x_px', 'focus_y_px', 'focus_score', 'qc_status'
    ])


def process(files, s, cols):
    """Process one image batch and retain raw images plus machine-readable masks."""
    root = Path(tempfile.mkdtemp(prefix='kt3_web_'))
    inp, out = root/'input', root/'results'
    for d in [
        inp, out/'csv', out/'raw_images', out/'segmentation_masks', out/'raw_crops',
        out/'labelled_crops', out/'indexed_large_images', out/'figures'
    ]:
        d.mkdir(parents=True, exist_ok=True)

    paths = []
    for uf in files:
        p = inp / Path(uf.name).name
        p.write_bytes(uf.getbuffer())
        paths.append(p)
    paths = sorted(paths, key=natural_key)

    settings_row = {
        'organoid_detection_mode': s.organoid_mode,
        'GFP_labelled_organoids': bool(s.organoid_mode == GFP_MODE),
        'RFP_PSC_stromal_cells_present': bool(s.rfp_psc_present),
        'microwell_diameter_um': float(s.well_diameter_um),
        'split_touching_PDOs': bool(s.split_pdos),
        'well_rmin_px': int(s.well_rmin), 'well_rmax_px': int(s.well_rmax),
        'well_spacing_px': int(s.well_spacing), 'hough_p2': float(s.hough_p2),
        'green_low_threshold': float(s.green_low), 'green_high_threshold': float(s.green_high),
        'pdo_min_area_px2': int(s.pdo_min_area), 'pdo_peak_distance_px': int(s.pdo_peak_distance),
        'psc_peak_threshold': float(s.psc_peak_threshold), 'psc_red_minimum': float(s.psc_red_minimum),
        'psc_peak_distance_px': int(s.psc_peak_distance),
        'brightfield_contrast_threshold': float(s.brightfield_contrast_threshold),
        'brightfield_min_area_px2': int(s.brightfield_min_area),
    }
    pd.DataFrame([settings_row]).to_csv(out/'csv'/'analysis_settings.csv', index=False)

    wells_rows, pdo_rows, psc_rows, image_rows, mask_rows, labelled_paths = [], [], [], [], [], []
    for idx, p in enumerate(paths, 1):
        series = infer_series(p.name, idx)
        raw_copy_name = f'series_{series:02d}__{p.name}'
        shutil.copy2(p, out/'raw_images'/raw_copy_name)

        rgb = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
        H, W = rgb.shape[:2]
        circles = detect_wells(rgb, s)
        circles = np.asarray([
            c for c in circles
            if c[0]-c[2] >= 2 and c[0]+c[2] < W-2 and c[1]-c[2] >= 2 and c[1]+c[2] < H-2
        ], dtype=int)
        if len(circles) == 0:
            raise RuntimeError(
                f'No fully visible wells detected in {p.name}. '
                'Try widening the well-radius range or lowering the well detection sensitivity.'
            )

        xs, ys = cluster(circles[:, 0]), cluster(circles[:, 1])
        umpp = float(s.well_diameter_um) / (2 * float(np.median(circles[:, 2])))
        gfp_pdos = segment_pdos(green_excess(rgb), s) if s.organoid_mode == GFP_MODE else None
        local = []
        current_psc_rows = []

        for x, y, r in circles:
            x, y, r = int(x), int(y), int(r)
            col, row = grid_index(x, y, xs, ys)
            well = f'{col},{row}'
            if s.organoid_mode == GFP_MODE:
                assigned = [o for o in gfp_pdos if (o['x']-x)**2 + (o['y']-y)**2 <= (0.86*r)**2]
            else:
                assigned = segment_unlabelled_pdos_in_well(rgb, x, y, r, s)

            if s.rfp_psc_present:
                foci = detect_psc(rgb, x, y, r, s)
                psc_n = len(foci)
                for focus_n, (fx, fy, score) in enumerate(foci, 1):
                    focus_row = {
                        'image_series': series, 'source_image': p.name, 'well_index': well,
                        'focus_number_in_well': focus_n, 'focus_x_px': fx, 'focus_y_px': fy,
                        'focus_score': score, 'qc_status': 'automated_not_manually_reviewed'
                    }
                    psc_rows.append(focus_row)
                    current_psc_rows.append(focus_row)
            else:
                psc_n = None

            sizes = [2 * math.sqrt(o['area']/math.pi) * umpp for o in assigned]
            local.append({'x': x, 'y': y, 'r': r, 'well': well, 'col': col, 'row': row,
                          'pdo_n': len(assigned), 'psc_n': psc_n, 'sizes': sizes})
            wells_rows.append({
                'image_series': series, 'source_image': p.name,
                'organoid_detection_mode': s.organoid_mode,
                'GFP_labelled_organoids': bool(s.organoid_mode == GFP_MODE),
                'RFP_PSC_stromal_cells_present': bool(s.rfp_psc_present),
                'well_index': well, 'well_col_index': col, 'well_row_index': row,
                'well_centre_x_px': x, 'well_centre_y_px': y, 'well_radius_px': r,
                'um_per_pixel': umpp, 'PDO_count': len(assigned),
                'PSC_like_focus_count': psc_n,
                'PDO_sizes_um': '; '.join(f'{v:.4f}' for v in sizes),
                'qc_status': 'automated_not_manually_reviewed',
                'qc_fully_visible_well': True,
                'qc_multiple_pdos_in_well': bool(len(assigned) > 1),
                'qc_no_pdo_detected': bool(len(assigned) == 0),
                'qc_brightfield_detection_requires_visual_review': bool(s.organoid_mode == BRIGHTFIELD_MODE),
            })
            for n, (obj, size) in enumerate(zip(assigned, sizes), 1):
                pdo_rows.append({
                    'image_series': series, 'well_index': well,
                    'organoid_detection_mode': s.organoid_mode,
                    'GFP_labelled_organoids': bool(s.organoid_mode == GFP_MODE),
                    'RFP_PSC_stromal_cells_present': bool(s.rfp_psc_present),
                    'PDO_number_in_well': n, 'PDO_count_in_well': len(assigned),
                    'centroid_x_px': obj['x'], 'centroid_y_px': obj['y'],
                    'projected_area_px2': obj['area'],
                    'equivalent_circular_diameter_um': size,
                    'PSC_like_focus_count_in_well': psc_n,
                    'qc_status': 'automated_not_manually_reviewed'
                })
            if assigned:
                crop = crop_square(rgb, x, y, r)
                base = f'series_{series:02d}_well_{col}_{row}'
                rp = out/'raw_crops'/f'{base}.png'
                lp = out/'labelled_crops'/f'{base}_labelled.png'
                crop.save(rp, dpi=(300, 300))
                labelled_crop(crop, series, well, len(assigned), psc_n, sizes).save(lp, dpi=(300, 300))
                labelled_paths.append(lp)

        well_mask, pdo_mask, psc_mask = make_training_masks(rgb, circles, current_psc_rows, s)
        mask_specs = [
            ('well_mask', well_mask, 'filled detected microwell mask'),
            ('pdo_semantic_mask', pdo_mask, 'automated PDO semantic foreground mask'),
            ('psc_focus_point_mask', psc_mask, 'detected PSC-like focus point markers; not cell segmentation'),
        ]
        for kind, arr, definition in mask_specs:
            mask_name = f'series_{series:02d}__{kind}.png'
            Image.fromarray(arr).save(out/'segmentation_masks'/mask_name)
            mask_rows.append({
                'image_series': series, 'source_image': p.name, 'mask_type': kind,
                'mask_file': mask_name, 'mask_definition': definition,
                'qc_status': 'automated_not_manually_reviewed'
            })

        indexed_overlay(rgb, local).save(out/'indexed_large_images'/f'series_{series:02d}_indexed.png', dpi=(300, 300))
        image_rows.append({
            'image_series': series, 'source_image': p.name, 'raw_image_export': raw_copy_name,
            'organoid_detection_mode': s.organoid_mode,
            'GFP_labelled_organoids': bool(s.organoid_mode == GFP_MODE),
            'RFP_PSC_stromal_cells_present': bool(s.rfp_psc_present),
            'fully_visible_wells': len(local),
            'PDO_containing_wells': sum(w['pdo_n'] > 0 for w in local),
            'PDO_count': sum(w['pdo_n'] for w in local),
            'PSC_like_foci_all_wells': sum(w['psc_n'] for w in local) if s.rfp_psc_present else None,
            'um_per_pixel': umpp,
            'qc_status': 'automated_not_manually_reviewed'
        })

    wdf = pd.DataFrame(wells_rows)
    pdf = pd.DataFrame(pdo_rows) if pdo_rows else _empty_pdo_df()
    pscdf = pd.DataFrame(psc_rows) if psc_rows else _empty_psc_df()
    idf = pd.DataFrame(image_rows)
    maskdf = pd.DataFrame(mask_rows)
    wdf.to_csv(out/'csv'/'well_raw_data.csv', index=False)
    pdf.to_csv(out/'csv'/'PDO_raw_data.csv', index=False)
    pscdf.to_csv(out/'csv'/'PSC_focus_raw_data.csv', index=False)
    idf.to_csv(out/'csv'/'image_summary.csv', index=False)
    maskdf.to_csv(out/'csv'/'mask_manifest.csv', index=False)

    summary_base = {
        'organoid_detection_mode': s.organoid_mode,
        'GFP_labelled_organoids': bool(s.organoid_mode == GFP_MODE),
        'RFP_PSC_stromal_cells_present': bool(s.rfp_psc_present),
        'images_processed': len(idf), 'fully_visible_wells': len(wdf),
        'qc_status': 'automated_not_manually_reviewed'
    }
    if len(pdf):
        d = pdf['equivalent_circular_diameter_um'].astype(float)
        mean = float(d.mean())
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        ax.hist(d, bins=int(s.histogram_bins), edgecolor='black')
        ax.axvline(mean, ls='--', label=f'Mean = {mean:.1f} µm')
        ax.axvline(mean-sd, ls=':', label=f'±1 SD = {sd:.1f} µm')
        ax.axvline(mean+sd, ls=':')
        ax.set(xlabel='PDO equivalent circular diameter (µm)', ylabel='Number of PDOs')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out/'figures'/'PDO_size_distribution.png', dpi=300)
        plt.close(fig)

        if s.rfp_psc_present:
            fr = pdf['PSC_like_focus_count_in_well'].value_counts().sort_index().rename_axis('PSC_like_focus_count').reset_index(name='PDO_count')
            fr['percentage_of_PDOs'] = 100 * fr.PDO_count / len(pdf)
            fr.to_csv(out/'csv'/'PSC_count_frequency_across_PDOs.csv', index=False)
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
            bars = ax.bar(fr.PSC_like_focus_count, fr.PDO_count, edgecolor='black')
            for b, c, pct in zip(bars, fr.PDO_count, fr.percentage_of_PDOs):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                        f'{int(c)}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)
            ax.set(xlabel='PSC-like fluorescent foci in the same well', ylabel='Number of PDOs')
            ax.set_xticks(fr.PSC_like_focus_count)
            fig.tight_layout()
            fig.savefig(out/'figures'/'PSC_count_frequency_across_PDOs.png', dpi=300)
            plt.close(fig)

        summary = pd.DataFrame([{**summary_base,
            'PDO_containing_wells': int((wdf.PDO_count > 0).sum()), 'PDO_count': len(pdf),
            'mean_PDO_diameter_um': mean, 'median_PDO_diameter_um': float(d.median()),
            'SD_PDO_diameter_um': sd, 'min_PDO_diameter_um': float(d.min()),
            'max_PDO_diameter_um': float(d.max()),
            'PSC_like_foci_all_detected_wells': int(wdf.PSC_like_focus_count.sum()) if s.rfp_psc_present else None,
            'PSC_like_foci_in_PDO_wells': int(wdf.loc[wdf.PDO_count > 0, 'PSC_like_focus_count'].sum()) if s.rfp_psc_present else None,
        }])
    else:
        summary = pd.DataFrame([{**summary_base,
            'PDO_containing_wells': 0, 'PDO_count': 0,
            'PSC_like_foci_all_detected_wells': int(wdf.PSC_like_focus_count.sum()) if s.rfp_psc_present else None,
            'PSC_like_foci_in_PDO_wells': 0 if s.rfp_psc_present else None,
        }])

    summary.to_csv(out/'csv'/'overall_summary.csv', index=False)
    if labelled_paths:
        make_contact(labelled_paths, out/'figures'/'PDO_well_contact_sheet_compact.png', cols=int(cols), gap=5)
    return root, out, summary, idf


def _copy_tree_contents(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob('*'):
        if p.is_file():
            q = dst / p.relative_to(src)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def _make_ids(df, meta):
    if not len(df):
        return df
    out = df.copy()
    exp = stable_token(meta.get('experiment_id', 'Experiment_001'), 'Experiment_001')
    dev = stable_token(meta.get('device_id', 'Array_001'), 'Array_001')
    lane = int(meta['condition_index'])
    tp = int(meta['timepoint_index'])
    out['experiment_id'] = str(meta.get('experiment_id', 'Experiment_001'))
    out['device_id'] = str(meta.get('device_id', 'Array_001'))
    out['biological_replicate_id'] = str(meta.get('biological_replicate_id', 'Replicate_1'))
    out['pdo_model'] = str(meta.get('pdo_model', ''))
    out['condition_index'] = lane
    out['condition'] = str(meta['condition'])
    out['timepoint_index'] = tp
    out['timepoint'] = str(meta['timepoint'])
    out['elapsed_time'] = meta.get('elapsed_time', np.nan)
    out['time_unit'] = str(meta.get('time_unit', 'days'))
    out['drug_or_therapeutic'] = str(meta.get('drug_or_therapeutic', ''))
    out['concentration'] = meta.get('concentration', np.nan)
    out['concentration_unit'] = str(meta.get('concentration_unit', ''))
    out['organoid_detection_mode'] = str(meta['organoid_detection_mode'])
    out['GFP_labelled_organoids'] = bool(meta['GFP_labelled_organoids'])
    out['RFP_PSC_stromal_cells_present'] = bool(meta['RFP_PSC_stromal_cells_present'])
    if 'image_series' in out.columns:
        out['image_uid'] = out['image_series'].map(lambda f: f'{exp}__{dev}__L{lane:02d}__T{tp:02d}__F{int(f):02d}')
    if {'image_series', 'well_col_index', 'well_row_index'}.issubset(out.columns):
        out['trajectory_id'] = out.apply(
            lambda r: f'{exp}__{dev}__L{lane:02d}__F{int(r.image_series):02d}__W{int(r.well_col_index)}_{int(r.well_row_index)}', axis=1
        )
        out['well_observation_id'] = out['trajectory_id'] + f'__T{tp:02d}'
    return out


def _longitudinal_well_table(wdf: pd.DataFrame, pdf: pd.DataFrame):
    keys = ['condition_index', 'condition', 'image_series', 'timepoint_index', 'timepoint', 'well_index']
    id_cols = [c for c in [
        'experiment_id', 'device_id', 'biological_replicate_id', 'pdo_model', 'image_uid',
        'trajectory_id', 'well_observation_id', 'elapsed_time', 'time_unit', 'drug_or_therapeutic',
        'concentration', 'concentration_unit', 'organoid_detection_mode',
        'GFP_labelled_organoids', 'RFP_PSC_stromal_cells_present'
    ] if c in wdf.columns]
    base_cols = keys + id_cols + ['PDO_count', 'PSC_like_focus_count', 'um_per_pixel']
    base = wdf[base_cols].copy() if len(wdf) else pd.DataFrame(columns=base_cols)
    if len(pdf):
        p = pdf.copy()
        p['projected_area_um2'] = p['projected_area_px2'].astype(float) * (p['um_per_pixel'].astype(float) ** 2)
        agg = p.groupby(keys, as_index=False).agg(
            mean_PDO_diameter_um=('equivalent_circular_diameter_um', 'mean'),
            max_PDO_diameter_um=('equivalent_circular_diameter_um', 'max'),
            total_PDO_projected_area_um2=('projected_area_um2', 'sum')
        )
        base = base.merge(agg, on=keys, how='left')
    else:
        base['mean_PDO_diameter_um'] = np.nan
        base['max_PDO_diameter_um'] = np.nan
        base['total_PDO_projected_area_um2'] = np.nan
    base['total_PDO_projected_area_um2'] = base['total_PDO_projected_area_um2'].fillna(0.0)
    base['PDO_present'] = base['PDO_count'].fillna(0).astype(float) > 0
    if len(base):
        trajectory_key = 'trajectory_id' if 'trajectory_id' in base.columns else None
        group_cols = [trajectory_key] if trajectory_key else ['condition_index', 'image_series', 'well_index']
        baseline = (base.sort_values('timepoint_index').groupby(group_cols, as_index=False).first()
                    [group_cols + ['total_PDO_projected_area_um2']]
                    .rename(columns={'total_PDO_projected_area_um2': 'baseline_total_PDO_area_um2'}))
        base = base.merge(baseline, on=group_cols, how='left')
        base['relative_total_PDO_area_vs_baseline'] = np.where(
            base['baseline_total_PDO_area_um2'] > 0,
            base['total_PDO_projected_area_um2'] / base['baseline_total_PDO_area_um2'], np.nan
        )
    return base


def _make_longitudinal_figures(summary: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    if not len(summary):
        return
    order = summary[['timepoint_index', 'timepoint']].drop_duplicates().sort_values('timepoint_index')
    xticks = order['timepoint_index'].to_numpy()
    xlabels = order['timepoint'].astype(str).tolist()

    def lineplot(ycol, ylabel, filename):
        if ycol not in summary.columns:
            return
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        any_line = False
        for condition, g in summary.groupby('condition', sort=False):
            g = g.sort_values('timepoint_index')
            vals = pd.to_numeric(g[ycol], errors='coerce')
            if vals.notna().any():
                ax.plot(g['timepoint_index'], vals, marker='o', label=str(condition))
                any_line = True
        if not any_line:
            plt.close(fig)
            return
        ax.set_xlabel('Time point')
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir/filename, dpi=300)
        plt.close(fig)

    lineplot('mean_PDO_diameter_um', 'Mean PDO equivalent circular diameter (µm)', 'condition_comparison_mean_PDO_diameter.png')
    lineplot('PDO_containing_well_percentage', 'PDO-containing wells (%)', 'condition_comparison_PDO_occupancy.png')
    lineplot('PDO_count', 'Detected PDO count', 'condition_comparison_PDO_count.png')
    lineplot('mean_PSC_foci_in_PDO_wells', 'Mean PSC-like foci per PDO-containing well', 'condition_comparison_PSC_foci.png')


def _sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_ml_export_package(out: Path, wdf: pd.DataFrame, pdf: pd.DataFrame, pscdf: pd.DataFrame,
                            ldf: pd.DataFrame, mapdf: pd.DataFrame, experiment_metadata: dict):
    """Create a standardised, self-describing ML / virtual-model dataset package."""
    ml = out/'machine_learning_export'
    if ml.exists():
        shutil.rmtree(ml)
    for d in [ml/'tables', ml/'assets'/'raw_images', ml/'assets'/'masks', ml/'assets'/'well_crops']:
        d.mkdir(parents=True, exist_ok=True)

    exp_row = {
        'schema_version': ML_SCHEMA_VERSION,
        'experiment_id': experiment_metadata.get('experiment_id', 'Experiment_001'),
        'device_id': experiment_metadata.get('device_id', 'Array_001'),
        'biological_replicate_id': experiment_metadata.get('biological_replicate_id', 'Replicate_1'),
        'pdo_model': experiment_metadata.get('pdo_model', ''),
        'time_unit': experiment_metadata.get('time_unit', 'days'),
        'qc_status': 'automated_not_manually_reviewed'
    }
    pd.DataFrame([exp_row]).to_csv(ml/'tables'/'experiment_metadata.csv', index=False)

    condition_cols = [c for c in [
        'condition_index', 'condition', 'drug_or_therapeutic', 'concentration', 'concentration_unit',
        'organoid_detection_mode', 'GFP_labelled_organoids', 'RFP_PSC_stromal_cells_present'
    ] if c in mapdf.columns]
    mapdf[condition_cols].drop_duplicates().sort_values('condition_index').to_csv(
        ml/'tables'/'condition_metadata.csv', index=False
    )
    time_cols = [c for c in ['timepoint_index', 'timepoint', 'elapsed_time', 'time_unit'] if c in mapdf.columns]
    mapdf[time_cols].drop_duplicates().sort_values('timepoint_index').to_csv(
        ml/'tables'/'timepoint_metadata.csv', index=False
    )
    mapdf.to_csv(ml/'tables'/'experiment_map.csv', index=False)
    wdf.to_csv(ml/'tables'/'well_observations.csv', index=False)
    pdf.to_csv(ml/'tables'/'pdo_observations.csv', index=False)
    pscdf.to_csv(ml/'tables'/'psc_focus_observations.csv', index=False)
    ldf.to_csv(ml/'tables'/'longitudinal_trajectories.csv', index=False)

    qc_cols = [c for c in [
        'experiment_id', 'device_id', 'condition_index', 'condition', 'image_series', 'timepoint_index',
        'timepoint', 'well_index', 'trajectory_id', 'well_observation_id', 'qc_status',
        'qc_fully_visible_well', 'qc_multiple_pdos_in_well', 'qc_no_pdo_detected',
        'qc_brightfield_detection_requires_visual_review'
    ] if c in wdf.columns]
    wdf[qc_cols].to_csv(ml/'tables'/'qc_flags.csv', index=False)

    asset_rows = []
    base = out/'condition_timepoint_outputs'
    for p in base.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        parts = set(rel.parts)
        asset_type = None
        if 'raw_images' in parts:
            asset_type = 'raw_image'
            dest_root = ml/'assets'/'raw_images'
        elif 'segmentation_masks' in parts:
            asset_type = 'segmentation_mask'
            dest_root = ml/'assets'/'masks'
        elif 'raw_crops' in parts:
            asset_type = 'raw_well_crop'
            dest_root = ml/'assets'/'well_crops'
        else:
            continue
        dest = dest_root/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        asset_rows.append({
            'asset_type': asset_type,
            'source_relative_path': str(rel),
            'export_relative_path': str(dest.relative_to(ml)),
            'file_size_bytes': int(dest.stat().st_size),
            'sha256': _sha256(dest)
        })
    assetdf = pd.DataFrame(asset_rows, columns=[
        'asset_type', 'source_relative_path', 'export_relative_path', 'file_size_bytes', 'sha256'
    ])
    assetdf.to_csv(ml/'tables'/'asset_manifest.csv', index=False)

    mask_manifest_files = list(base.rglob('csv/mask_manifest.csv'))
    masks = []
    for p in mask_manifest_files:
        m = pd.read_csv(p)
        prefix = p.parent.parent.relative_to(base)
        m['condition_timepoint_path'] = str(prefix)
        masks.append(m)
    pd.concat(masks, ignore_index=True).to_csv(ml/'tables'/'mask_manifest.csv', index=False) if masks else pd.DataFrame().to_csv(ml/'tables'/'mask_manifest.csv', index=False)

    schema = {
        'schema_version': ML_SCHEMA_VERSION,
        'primary_longitudinal_unit': 'trajectory_id',
        'trajectory_id_definition': 'experiment + device/array + lane + field of view + well x,y; excludes time so it remains stable longitudinally',
        'well_observation_id_definition': 'trajectory_id + timepoint',
        'image_uid_definition': 'experiment + device/array + lane + timepoint + field of view',
        'pdo_observation_id_definition': 'well_observation_id + PDO number at that timepoint; not guaranteed to be the same physical PDO after splitting/merging',
        'mask_definitions': {
            'well_mask': 'filled detected microwell interiors',
            'pdo_semantic_mask': 'automated PDO foreground; semantic rather than guaranteed instance mask',
            'psc_focus_point_mask': 'small point markers centred on detected PSC-like red fluorescent foci; not individual-cell segmentation'
        },
        'qc_definition': 'automated_not_manually_reviewed means the algorithm completed, not that a human accepted the segmentation',
        'recommended_ml_grouping': ['experiment_id', 'biological_replicate_id', 'device_id'],
        'recommended_no_leakage_key': 'trajectory_id'
    }
    (ml/'schema.json').write_text(json.dumps(schema, indent=2), encoding='utf-8')

    manifest = {
        'schema_version': ML_SCHEMA_VERSION,
        'dataset_type': 'longitudinal PDO/stromal perturbation imaging dataset',
        'experiment_id': exp_row['experiment_id'],
        'device_id': exp_row['device_id'],
        'biological_replicate_id': exp_row['biological_replicate_id'],
        'counts': {
            'well_observations': int(len(wdf)),
            'pdo_observations': int(len(pdf)),
            'psc_focus_observations': int(len(pscdf)),
            'longitudinal_rows': int(len(ldf)),
            'assets': int(len(assetdf))
        },
        'important_note': 'Automated masks and measurements require visual QC before use as ground-truth labels.'
    }
    (ml/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    readme = f'''# Machine Learning / Virtual Model Export\n\nSchema version: {ML_SCHEMA_VERSION}\n\nThis folder is a self-describing export of one longitudinal PDO experiment. It is designed so multiple experiments can later be pooled for machine-learning or virtual-model development without reconstructing the experimental metadata.\n\n## Stable identifiers\n- `image_uid`: experiment + device/array + lane + time point + field of view.\n- `trajectory_id`: experiment + device/array + lane + field of view + microwell x,y. It deliberately excludes time so the same microwell can be followed longitudinally.\n- `well_observation_id`: trajectory + time point.\n- `pdo_observation_id`: a PDO object at one time point. It must not be assumed to represent the same physical PDO after splitting or merging.\n\n## Assets\n- `assets/raw_images/`: copies of the original uploaded microscopy images.\n- `assets/masks/`: automated well, PDO semantic, and PSC-focus point masks.\n- `assets/well_crops/`: raw unannotated PDO-containing well crops.\n\n## Masks\n`pdo_semantic_mask` is a semantic foreground mask. Touching PDOs can remain connected in the semantic mask even when the measurement algorithm splits them into separate objects. `psc_focus_point_mask` marks detected PSC-like fluorescent focus locations and is **not** an individual PSC-cell segmentation mask.\n\n## Tables\n- `experiment_metadata.csv`\n- `condition_metadata.csv`\n- `timepoint_metadata.csv`\n- `experiment_map.csv`\n- `well_observations.csv`\n- `pdo_observations.csv`\n- `psc_focus_observations.csv`\n- `longitudinal_trajectories.csv`\n- `qc_flags.csv`\n- `mask_manifest.csv`\n- `asset_manifest.csv` with SHA-256 hashes for provenance.\n\n## QC and model development\nAutomated outputs are labelled `automated_not_manually_reviewed`. This is not a manual QC pass. For supervised segmentation, manually reviewed masks should be used as ground truth. When creating training/test splits, do not randomly split observations from the same `trajectory_id` across train and test. For biological generalisation, split at the biological replicate / experiment / device level where possible.\n'''
    (ml/'README.md').write_text(readme, encoding='utf-8')
    return ml


def process_experiment(entries, base_settings, cols=5, experiment_metadata=None, make_ml_export=True):
    if not entries:
        raise RuntimeError('No condition/time-point images were uploaded.')
    experiment_metadata = dict(experiment_metadata or {})
    experiment_metadata.setdefault('experiment_id', 'Experiment_001')
    experiment_metadata.setdefault('device_id', 'Array_001')
    experiment_metadata.setdefault('biological_replicate_id', 'Replicate_1')
    experiment_metadata.setdefault('pdo_model', '')
    experiment_metadata.setdefault('time_unit', 'days')

    root = Path(tempfile.mkdtemp(prefix='kt3_longitudinal_'))
    out = root/'results'
    for d in [out/'csv', out/'figures', out/'condition_timepoint_outputs']:
        d.mkdir(parents=True, exist_ok=True)

    all_wells, all_pdos, all_psc, cell_summaries, experiment_map = [], [], [], [], []
    for entry in sorted(entries, key=lambda e: (e['condition_index'], e['timepoint_index'])):
        files = entry.get('files') or []
        if not files:
            continue
        s = replace(base_settings, organoid_mode=entry['organoid_mode'], rfp_psc_present=bool(entry['rfp_psc_present']))
        _, cell_out, summary, _ = process(files, s, cols)

        meta = {
            **experiment_metadata,
            'condition_index': int(entry['condition_index']), 'condition': str(entry['condition']),
            'timepoint_index': int(entry['timepoint_index']), 'timepoint': str(entry['timepoint']),
            'elapsed_time': entry.get('elapsed_time', np.nan),
            'drug_or_therapeutic': entry.get('drug_or_therapeutic', ''),
            'concentration': entry.get('concentration', np.nan),
            'concentration_unit': entry.get('concentration_unit', ''),
            'organoid_detection_mode': str(entry['organoid_mode']),
            'GFP_labelled_organoids': bool(entry['organoid_mode'] == GFP_MODE),
            'RFP_PSC_stromal_cells_present': bool(entry['rfp_psc_present'])
        }
        experiment_map.append({**meta, 'number_of_uploaded_images': len(files)})

        w = pd.read_csv(cell_out/'csv'/'well_raw_data.csv')
        w = _make_ids(w, meta)
        all_wells.append(w)

        p = pd.read_csv(cell_out/'csv'/'PDO_raw_data.csv')
        if len(p):
            lookup_cols = ['image_series', 'well_index', 'well_col_index', 'well_row_index', 'um_per_pixel',
                           'image_uid', 'trajectory_id', 'well_observation_id']
            lookup = w[lookup_cols].drop_duplicates()
            p = p.merge(lookup, on=['image_series', 'well_index'], how='left')
            p = _make_ids(p, meta)
            p['pdo_observation_id'] = p.apply(
                lambda r: f"{r['well_observation_id']}__PDO{int(r['PDO_number_in_well']):02d}", axis=1
            )
            all_pdos.append(p)

        psc = pd.read_csv(cell_out/'csv'/'PSC_focus_raw_data.csv')
        if len(psc):
            lookup = w[['image_series', 'well_index', 'well_col_index', 'well_row_index', 'um_per_pixel',
                        'image_uid', 'trajectory_id', 'well_observation_id']].drop_duplicates()
            psc = psc.merge(lookup, on=['image_series', 'well_index'], how='left')
            psc = _make_ids(psc, meta)
            psc['psc_focus_id'] = psc.apply(
                lambda r: f"{r['well_observation_id']}__PSCFOCUS{int(r['focus_number_in_well']):03d}", axis=1
            )
            all_psc.append(psc)

        sm = summary.copy()
        for k, v in meta.items():
            sm[k] = v
        wells_n = int(sm.iloc[0].get('fully_visible_wells', 0))
        pdo_wells = int(sm.iloc[0].get('PDO_containing_wells', 0))
        sm['PDO_containing_well_percentage'] = 100.0 * pdo_wells / wells_n if wells_n else np.nan
        if entry['rfp_psc_present'] and len(w):
            denom = int((w['PDO_count'] > 0).sum())
            sm['mean_PSC_foci_in_PDO_wells'] = float(w.loc[w['PDO_count'] > 0, 'PSC_like_focus_count'].mean()) if denom else np.nan
        else:
            sm['mean_PSC_foci_in_PDO_wells'] = np.nan
        cell_summaries.append(sm)

        cell_folder = (out/'condition_timepoint_outputs'/
                       f"condition_{entry['condition_index']:02d}_{slugify(entry['condition'])}"/
                       f"time_{entry['timepoint_index']:02d}_{slugify(entry['timepoint'])}")
        _copy_tree_contents(cell_out, cell_folder)

    if not all_wells:
        raise RuntimeError('No uploaded condition/time-point cell produced analysis output.')

    wdf = pd.concat(all_wells, ignore_index=True)
    pdf = pd.concat(all_pdos, ignore_index=True) if all_pdos else _empty_pdo_df()
    pscdf = pd.concat(all_psc, ignore_index=True) if all_psc else _empty_psc_df()
    sdf = pd.concat(cell_summaries, ignore_index=True) if cell_summaries else pd.DataFrame()
    mapdf = pd.DataFrame(experiment_map)

    wdf.to_csv(out/'csv'/'longitudinal_well_raw_data.csv', index=False)
    pdf.to_csv(out/'csv'/'longitudinal_PDO_raw_data.csv', index=False)
    pscdf.to_csv(out/'csv'/'longitudinal_PSC_focus_raw_data.csv', index=False)
    sdf.to_csv(out/'csv'/'condition_timepoint_summary.csv', index=False)
    mapdf.to_csv(out/'csv'/'experiment_map.csv', index=False)

    ldf = _longitudinal_well_table(wdf, pdf)
    ldf.to_csv(out/'csv'/'well_longitudinal_tracking.csv', index=False)

    if len(sdf):
        keep = [c for c in [
            'condition_index', 'condition', 'drug_or_therapeutic', 'concentration', 'concentration_unit',
            'timepoint_index', 'timepoint', 'elapsed_time', 'time_unit', 'organoid_detection_mode',
            'RFP_PSC_stromal_cells_present', 'images_processed', 'fully_visible_wells',
            'PDO_containing_wells', 'PDO_containing_well_percentage', 'PDO_count',
            'mean_PDO_diameter_um', 'median_PDO_diameter_um', 'SD_PDO_diameter_um',
            'PSC_like_foci_in_PDO_wells', 'mean_PSC_foci_in_PDO_wells'
        ] if c in sdf.columns]
        sdf[keep].to_csv(out/'csv'/'condition_comparison_over_time.csv', index=False)
        _make_longitudinal_figures(sdf, out/'figures')

    ml_path = None
    if make_ml_export:
        ml_path = build_ml_export_package(out, wdf, pdf, pscdf, ldf, mapdf, experiment_metadata)
    return root, out, sdf, ldf, ml_path


def build_settings_from_widgets(well, rmin, rmax, spacing, hp2, gl, gh, amin, split,
                                pdist, pt, prm, ppd, bins, bf_contrast, bf_min_area,
                                organoid_mode=GFP_MODE, rfp_psc_present=True):
    return Settings(
        well_diameter_um=float(well), well_rmin=int(rmin), well_rmax=int(rmax),
        well_spacing=int(spacing), hough_p2=float(hp2), green_low=float(gl),
        green_high=float(gh), pdo_min_area=int(amin), split_pdos=bool(split),
        pdo_peak_distance=int(pdist), psc_peak_threshold=float(pt),
        psc_red_minimum=float(prm), psc_peak_distance=int(ppd),
        histogram_bins=int(bins), organoid_mode=organoid_mode,
        rfp_psc_present=bool(rfp_psc_present),
        brightfield_contrast_threshold=float(bf_contrast),
        brightfield_min_area=int(bf_min_area)
    )
