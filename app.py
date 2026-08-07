from __future__ import annotations

import io, math, re, shutil, tempfile, zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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


def detect_wells(rgb, s):
    rmin = max(1, int(round(s.well_rmin)))
    rmax = max(rmin + 1, int(round(s.well_rmax)))
    spacing = max(1, int(round(s.well_spacing)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    c = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.15, minDist=float(spacing),
        param1=75.0, param2=float(s.hough_p2),
        minRadius=int(rmin), maxRadius=int(rmax)
    )
    if c is None:
        return np.empty((0, 3), dtype=int)
    c = np.round(c[0]).astype(int)
    kept = []
    for x, y, r in c[np.argsort(c[:, 0])]:
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


def segment_unlabelled_pdos_in_well(rgb, x, y, r, s):
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
    f = []
    for py, px in peaks:
        if sub[py, px, 0] - sub[py, px, 2] >= float(s.psc_red_minimum):
            f.append((x0+int(px), y0+int(py), float(score[py, px])))
    return f


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
    y = 8
    for i, t in enumerate(lines):
        f = title if i == 0 else body
        dr.text((12, y), t, font=f, fill='white')
        y += 34 if i == 0 else 29
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
        for p in folder.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(folder))
    return b.getvalue()


def process(files, s, cols):
    """Process one batch of images with a single analysis configuration.

    Kept as a stable API for single-batch analysis and the automated test suite.
    """
    root = Path(tempfile.mkdtemp(prefix='kt3_web_'))
    inp, out = root/'input', root/'results'
    for d in [inp, out/'csv', out/'raw_crops', out/'labelled_crops', out/'indexed_large_images', out/'figures']:
        d.mkdir(parents=True, exist_ok=True)

    paths = []
    for uf in files:
        p = inp / Path(uf.name).name
        p.write_bytes(uf.getbuffer())
        paths.append(p)
    paths = sorted(paths, key=natural_key)

    pd.DataFrame([{
        'organoid_detection_mode': s.organoid_mode,
        'GFP_labelled_organoids': bool(s.organoid_mode == GFP_MODE),
        'RFP_PSC_stromal_cells_present': bool(s.rfp_psc_present),
        'microwell_diameter_um': float(s.well_diameter_um),
        'split_touching_PDOs': bool(s.split_pdos),
    }]).to_csv(out/'csv'/'analysis_settings.csv', index=False)

    wells_rows, pdo_rows, image_rows, labelled_paths = [], [], [], []
    for idx, p in enumerate(paths, 1):
        series = infer_series(p.name, idx)
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

        for x, y, r in circles:
            x, y, r = int(x), int(y), int(r)
            col, row = grid_index(x, y, xs, ys)
            well = f'{col},{row}'
            if s.organoid_mode == GFP_MODE:
                assigned = [o for o in gfp_pdos if (o['x']-x)**2 + (o['y']-y)**2 <= (0.86*r)**2]
            else:
                assigned = segment_unlabelled_pdos_in_well(rgb, x, y, r, s)

            if s.rfp_psc_present:
                psc_n = len(detect_psc(rgb, x, y, r, s))
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
                'PDO_sizes_um': '; '.join(f'{v:.4f}' for v in sizes)
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
                    'PSC_like_focus_count_in_well': psc_n
                })
            if assigned:
                crop = crop_square(rgb, x, y, r)
                base = f'series_{series:02d}_well_{col}_{row}'
                rp = out/'raw_crops'/f'{base}.png'
                lp = out/'labelled_crops'/f'{base}_labelled.png'
                crop.save(rp, dpi=(300, 300))
                labelled_crop(crop, series, well, len(assigned), psc_n, sizes).save(lp, dpi=(300, 300))
                labelled_paths.append(lp)

        indexed_overlay(rgb, local).save(out/'indexed_large_images'/f'series_{series:02d}_indexed.png', dpi=(300, 300))
        image_rows.append({
            'image_series': series, 'source_image': p.name,
            'organoid_detection_mode': s.organoid_mode,
            'GFP_labelled_organoids': bool(s.organoid_mode == GFP_MODE),
            'RFP_PSC_stromal_cells_present': bool(s.rfp_psc_present),
            'fully_visible_wells': len(local),
            'PDO_containing_wells': sum(w['pdo_n'] > 0 for w in local),
            'PDO_count': sum(w['pdo_n'] for w in local),
            'PSC_like_foci_all_wells': sum(w['psc_n'] for w in local) if s.rfp_psc_present else None,
            'um_per_pixel': umpp
        })

    wdf = pd.DataFrame(wells_rows)
    pdf = pd.DataFrame(pdo_rows)
    idf = pd.DataFrame(image_rows)
    wdf.to_csv(out/'csv'/'well_raw_data.csv', index=False)
    pdf.to_csv(out/'csv'/'PDO_raw_data.csv', index=False)
    idf.to_csv(out/'csv'/'image_summary.csv', index=False)

    summary_base = {
        'organoid_detection_mode': s.organoid_mode,
        'GFP_labelled_organoids': bool(s.organoid_mode == GFP_MODE),
        'RFP_PSC_stromal_cells_present': bool(s.rfp_psc_present),
        'images_processed': len(idf), 'fully_visible_wells': len(wdf)
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


def _longitudinal_well_table(wdf: pd.DataFrame, pdf: pd.DataFrame):
    keys = ['condition_index', 'condition', 'timepoint_index', 'timepoint', 'well_index']
    base_cols = keys + ['PDO_count', 'PSC_like_focus_count', 'um_per_pixel']
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
        baseline = (base.sort_values('timepoint_index')
                    .groupby(['condition_index', 'well_index'], as_index=False)
                    .first()[['condition_index', 'well_index', 'total_PDO_projected_area_um2']]
                    .rename(columns={'total_PDO_projected_area_um2': 'baseline_total_PDO_area_um2'}))
        base = base.merge(baseline, on=['condition_index', 'well_index'], how='left')
        base['relative_total_PDO_area_vs_baseline'] = np.where(
            base['baseline_total_PDO_area_um2'] > 0,
            base['total_PDO_projected_area_um2'] / base['baseline_total_PDO_area_um2'],
            np.nan
        )
    return base


def _make_longitudinal_figures(summary: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    if not len(summary):
        return
    order = (summary[['timepoint_index', 'timepoint']]
             .drop_duplicates().sort_values('timepoint_index'))
    xticks = order['timepoint_index'].to_numpy()
    xlabels = order['timepoint'].astype(str).tolist()

    def lineplot(ycol, ylabel, filename):
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
        ax.set_xticklabels(xlabels, rotation=0)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir/filename, dpi=300)
        plt.close(fig)

    lineplot('mean_PDO_diameter_um', 'Mean PDO equivalent circular diameter (µm)', 'condition_comparison_mean_PDO_diameter.png')
    lineplot('PDO_containing_well_percentage', 'PDO-containing wells (%)', 'condition_comparison_PDO_occupancy.png')
    lineplot('PDO_count', 'Detected PDO count', 'condition_comparison_PDO_count.png')
    if 'mean_PSC_foci_in_PDO_wells' in summary.columns:
        lineplot('mean_PSC_foci_in_PDO_wells', 'Mean PSC-like foci per PDO-containing well', 'condition_comparison_PSC_foci.png')


def process_experiment(entries, base_settings, cols=5):
    """Process a condition × timepoint experiment matrix.

    Each entry is a dict with condition/timepoint metadata, per-condition
    organoid/PSC configuration, and one or more uploaded images.
    """
    if not entries:
        raise RuntimeError('No condition/time-point images were uploaded.')

    root = Path(tempfile.mkdtemp(prefix='kt3_longitudinal_'))
    out = root/'results'
    for d in [out/'csv', out/'figures', out/'condition_timepoint_outputs']:
        d.mkdir(parents=True, exist_ok=True)

    all_wells, all_pdos, cell_summaries = [], [], []
    experiment_map = []

    for entry in sorted(entries, key=lambda e: (e['condition_index'], e['timepoint_index'])):
        files = entry.get('files') or []
        if not files:
            continue
        s = replace(
            base_settings,
            organoid_mode=entry['organoid_mode'],
            rfp_psc_present=bool(entry['rfp_psc_present'])
        )
        _, cell_out, summary, _ = process(files, s, cols)

        meta = {
            'condition_index': int(entry['condition_index']),
            'condition': str(entry['condition']),
            'timepoint_index': int(entry['timepoint_index']),
            'timepoint': str(entry['timepoint']),
            'organoid_detection_mode': str(entry['organoid_mode']),
            'GFP_labelled_organoids': bool(entry['organoid_mode'] == GFP_MODE),
            'RFP_PSC_stromal_cells_present': bool(entry['rfp_psc_present'])
        }
        experiment_map.append({**meta, 'number_of_uploaded_images': len(files)})

        w = pd.read_csv(cell_out/'csv'/'well_raw_data.csv')
        for k, v in meta.items():
            w[k] = v
        all_wells.append(w)

        pdo_path = cell_out/'csv'/'PDO_raw_data.csv'
        p = pd.read_csv(pdo_path) if pdo_path.exists() and pdo_path.stat().st_size > 0 else pd.DataFrame()
        if len(p):
            if 'um_per_pixel' not in p.columns:
                um_lookup = w[['image_series', 'well_index', 'um_per_pixel']].drop_duplicates()
                p = p.merge(um_lookup, on=['image_series', 'well_index'], how='left')
            for k, v in meta.items():
                p[k] = v
            all_pdos.append(p)

        sm = summary.copy()
        for k, v in meta.items():
            sm[k] = v
        wells_n = int(sm.iloc[0].get('fully_visible_wells', 0))
        pdo_wells = int(sm.iloc[0].get('PDO_containing_wells', 0))
        sm['PDO_containing_well_percentage'] = 100.0 * pdo_wells / wells_n if wells_n else np.nan
        if entry['rfp_psc_present'] and len(w):
            denom = int((w['PDO_count'] > 0).sum())
            sm['mean_PSC_foci_in_PDO_wells'] = (
                float(w.loc[w['PDO_count'] > 0, 'PSC_like_focus_count'].mean()) if denom else np.nan
            )
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
    pdf = pd.concat(all_pdos, ignore_index=True) if all_pdos else pd.DataFrame()
    sdf = pd.concat(cell_summaries, ignore_index=True) if cell_summaries else pd.DataFrame()
    mapdf = pd.DataFrame(experiment_map)

    wdf.to_csv(out/'csv'/'longitudinal_well_raw_data.csv', index=False)
    pdf.to_csv(out/'csv'/'longitudinal_PDO_raw_data.csv', index=False)
    sdf.to_csv(out/'csv'/'condition_timepoint_summary.csv', index=False)
    mapdf.to_csv(out/'csv'/'experiment_map.csv', index=False)

    ldf = _longitudinal_well_table(wdf, pdf)
    ldf.to_csv(out/'csv'/'well_longitudinal_tracking.csv', index=False)

    if len(sdf):
        keep = [c for c in [
            'condition_index', 'condition', 'timepoint_index', 'timepoint',
            'organoid_detection_mode', 'RFP_PSC_stromal_cells_present',
            'images_processed', 'fully_visible_wells', 'PDO_containing_wells',
            'PDO_containing_well_percentage', 'PDO_count', 'mean_PDO_diameter_um',
            'median_PDO_diameter_um', 'SD_PDO_diameter_um',
            'PSC_like_foci_in_PDO_wells', 'mean_PSC_foci_in_PDO_wells'
        ] if c in sdf.columns]
        sdf[keep].to_csv(out/'csv'/'condition_comparison_over_time.csv', index=False)
        _make_longitudinal_figures(sdf, out/'figures')

    return root, out, sdf, ldf


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


# ------------------------- Streamlit interface -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon='🔬', layout='wide')
st.title(APP_TITLE)
st.caption('Analyze GFP-labelled or unlabelled PDOs, optionally quantify RFP stromal cells, and compare multiple experimental lanes longitudinally.')

with st.expander('Measurement and tracking notes'):
    st.markdown('''- **GFP-labelled PDOs** use green-fluorescence segmentation. **Unlabelled PDOs** use local brightfield/phase-contrast segmentation within each microwell.\n- RFP PSC/stromal analysis can be switched **on or off independently for each condition**.\n- Microwell diameter is the calibration ruler (default **100 µm**).\n- PDO size is **2D equivalent circular diameter** from segmented projected area, not true 3D diameter.\n- PSC values are automated **PSC-like red fluorescent foci**, not definitive single-cell counts.\n- Longitudinal well tracking uses the **well x,y index within each lane**. It assumes the same lane is imaged in the same orientation and that the well grid is detected consistently at each time point.\n- If a well contains multiple PDOs, the longitudinal table also reports **total projected PDO area**, which is often more stable than treating one object as a persistent single organoid.\n- Always visually QC the indexed images and crops before thesis/publication use.''')

workflow = st.sidebar.selectbox(
    'Workflow', ['Longitudinal multi-condition experiment', 'Single / batch analysis'], index=0
)

st.sidebar.header('Analysis settings')
well = st.sidebar.number_input('Microwell diameter (µm)', 1.0, 1000.0, 100.0, 1.0)
split = st.sidebar.checkbox('Split touching PDOs', True)
bins = st.sidebar.slider('Histogram bins', 5, 30, 12)
cols = st.sidebar.slider('Contact-sheet columns', 3, 10, 5)

with st.sidebar.expander('Advanced thresholds'):
    rmin = st.number_input('Minimum well radius (px)', 5, 200, 23, step=1, key='rmin')
    rmax = st.number_input('Maximum well radius (px)', 6, 300, 40, step=1, key='rmax')
    spacing = st.number_input('Minimum well spacing (px)', 10, 500, 54, step=1, key='spacing')
    hp2 = st.number_input('Well detection sensitivity', 1.0, 100.0, 27.0, 1.0, key='hp2')
    gl = st.number_input('GFP PDO low threshold', 0.0, 255.0, 30.0, 1.0, key='gl')
    gh = st.number_input('GFP PDO high threshold', 0.0, 255.0, 45.0, 1.0, key='gh')
    amin = st.number_input('Minimum GFP PDO area (px²)', 1, 100000, 20, step=1, key='amin')
    pdist = st.number_input('PDO split peak distance (px)', 1, 100, 18, step=1, key='pdist')
    bf_contrast = st.number_input('Unlabelled PDO contrast threshold', 0.5, 100.0, 10.0, 0.5, key='bfcontrast')
    bf_min_area = st.number_input('Unlabelled PDO minimum area (px²)', 5, 100000, 80, step=5, key='bfarea')
    pt = st.number_input('PSC focus threshold', 0.0, 255.0, 9.0, 0.5, key='pt')
    prm = st.number_input('PSC red-minus-blue minimum', 0.0, 255.0, 12.0, 0.5, key='prm')
    ppd = st.number_input('PSC focus minimum spacing (px)', 1, 100, 4, step=1, key='ppd')

base_settings = build_settings_from_widgets(
    well, rmin, rmax, spacing, hp2, gl, gh, amin, split, pdist,
    pt, prm, ppd, bins, bf_contrast, bf_min_area
)

if workflow == 'Longitudinal multi-condition experiment':
    st.subheader('Experiment layout')
    c1, c2 = st.columns(2)
    with c1:
        n_conditions = st.selectbox('Number of lanes / conditions', list(range(1, 13)), index=5)
    with c2:
        n_timepoints = st.selectbox('Number of imaging time points', list(range(1, 13)), index=3)

    st.caption('Configure the condition names and imaging labels below. The app will create one upload box for every condition × time point.')

    with st.expander('Time-point labels', expanded=True):
        timepoints = []
        tcols = st.columns(min(4, n_timepoints))
        for t in range(n_timepoints):
            default = f'Day {t}'
            with tcols[t % len(tcols)]:
                timepoints.append(st.text_input(f'Time point {t+1}', default, key=f'time_label_{t}'))

    conditions = []
    with st.expander('Lane / condition setup', expanded=True):
        for i in range(n_conditions):
            st.markdown(f'**Lane {i+1}**')
            a, b, c = st.columns([1.4, 1.2, 1.2])
            with a:
                name = st.text_input('Condition name', f'Condition {i+1}', key=f'cond_name_{i}')
            with b:
                omode = st.selectbox('Organoid detection', [GFP_MODE, BRIGHTFIELD_MODE], key=f'cond_org_mode_{i}')
            with c:
                pmode = st.selectbox('RFP stromal cells', [PSC_PRESENT, PSC_ABSENT], key=f'cond_psc_mode_{i}')
            conditions.append({'condition_index': i+1, 'condition': name,
                               'organoid_mode': omode, 'rfp_psc_present': pmode == PSC_PRESENT})
            if i < n_conditions - 1:
                st.divider()

    st.subheader('Upload condition × time-point images')
    st.caption('Each box accepts one or more images. If a lane requires several fields of view at one time point, upload them together in the same box.')

    upload_entries = []
    for cond in conditions:
        with st.expander(f"Lane {cond['condition_index']}: {cond['condition']}", expanded=(cond['condition_index'] == 1)):
            status = 'RFP PSC analysis ON' if cond['rfp_psc_present'] else 'RFP PSC analysis OFF'
            mode_short = 'GFP PDOs' if cond['organoid_mode'] == GFP_MODE else 'Unlabelled PDOs'
            st.caption(f'{mode_short} • {status}')
            for row_start in range(0, n_timepoints, 4):
                row_items = list(range(row_start, min(row_start+4, n_timepoints)))
                ucols = st.columns(len(row_items))
                for col_widget, t in zip(ucols, row_items):
                    with col_widget:
                        files = st.file_uploader(
                            timepoints[t], type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'],
                            accept_multiple_files=True,
                            key=f"upload_c{cond['condition_index']}_t{t+1}"
                        )
                        upload_entries.append({
                            **cond, 'timepoint_index': t+1, 'timepoint': timepoints[t], 'files': files
                        })

    uploaded_cells = sum(bool(e['files']) for e in upload_entries)
    expected_cells = n_conditions * n_timepoints
    st.info(f'{uploaded_cells} of {expected_cells} condition × time-point boxes currently contain images.')

    run = st.button('Run longitudinal experiment analysis', type='primary', use_container_width=True,
                    disabled=uploaded_cells == 0)
    if run:
        missing = [f"{e['condition']} / {e['timepoint']}" for e in upload_entries if not e['files']]
        if missing:
            st.warning(f'{len(missing)} upload boxes are empty. The analysis will run on the uploaded cells only.')
        bar = st.progress(5, text='Starting longitudinal analysis…')
        try:
            root, out, summary, longitudinal = process_experiment(upload_entries, base_settings, int(cols))
            bar.progress(100, text='Complete')
            st.session_state['long_zip'] = zip_bytes(out)
            st.session_state['long_out'] = str(out)
            st.session_state['long_summary'] = summary.to_dict('records')
            st.session_state['long_tracking'] = longitudinal.to_dict('records')
            st.success('Longitudinal experiment analysis complete.')
        except Exception as e:
            bar.empty()
            st.error(f'Analysis stopped: {e}')

    if 'long_zip' in st.session_state:
        out = Path(st.session_state['long_out'])
        summary = pd.DataFrame(st.session_state['long_summary'])
        tracking = pd.DataFrame(st.session_state['long_tracking'])
        st.divider()
        st.subheader('Longitudinal results')
        st.download_button(
            'Download complete longitudinal results ZIP', st.session_state['long_zip'],
            'PDO_PSC_longitudinal_experiment_results.zip', 'application/zip',
            type='primary', use_container_width=True
        )

        fdir = out/'figures'
        figs = [
            ('Mean PDO diameter by condition', 'condition_comparison_mean_PDO_diameter.png'),
            ('PDO-containing wells by condition', 'condition_comparison_PDO_occupancy.png'),
            ('PDO count by condition', 'condition_comparison_PDO_count.png'),
            ('PSC-like foci by condition', 'condition_comparison_PSC_foci.png'),
        ]
        available = [(lab, fdir/name) for lab, name in figs if (fdir/name).exists()]
        if available:
            for start in range(0, len(available), 2):
                cc = st.columns(min(2, len(available)-start))
                for col, (lab, p) in zip(cc, available[start:start+2]):
                    col.image(str(p), caption=lab, use_container_width=True)

        with st.expander('Condition × time-point summary', expanded=True):
            show_cols = [c for c in [
                'condition', 'timepoint', 'organoid_detection_mode',
                'RFP_PSC_stromal_cells_present', 'fully_visible_wells',
                'PDO_containing_wells', 'PDO_containing_well_percentage',
                'PDO_count', 'mean_PDO_diameter_um', 'mean_PSC_foci_in_PDO_wells'
            ] if c in summary.columns]
            st.dataframe(summary[show_cols], use_container_width=True, hide_index=True)

        with st.expander('Well-by-well longitudinal tracking'):
            st.caption('Rows are matched by lane/condition + microwell x,y index across time points.')
            st.dataframe(tracking, use_container_width=True, hide_index=True)

else:
    st.subheader('Single / batch analysis')
    a, b = st.columns(2)
    with a:
        organoid_mode = st.selectbox('Are the organoids GFP-labelled?', [GFP_MODE, BRIGHTFIELD_MODE])
    with b:
        psc_mode = st.selectbox('Are RFP-labelled PSC/stromal cells present?', [PSC_PRESENT, PSC_ABSENT])
    files = st.file_uploader(
        'Upload large microscopy images', type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'],
        accept_multiple_files=True, key='single_files'
    )
    run = st.button('Run analysis', type='primary', use_container_width=True, disabled=not files)
    if run:
        s = replace(base_settings, organoid_mode=organoid_mode, rfp_psc_present=psc_mode == PSC_PRESENT)
        bar = st.progress(5, text='Starting analysis…')
        try:
            root, out, summary, idf = process(files, s, int(cols))
            bar.progress(100, text='Complete')
            st.session_state['zip'] = zip_bytes(out)
            st.session_state['out'] = str(out)
            st.session_state['summary'] = summary.to_dict('records')
            st.session_state['idf'] = idf.to_dict('records')
            st.success('Analysis complete.')
        except Exception as e:
            bar.empty()
            st.error(f'Analysis stopped: {e}')

    if 'zip' in st.session_state:
        out = Path(st.session_state['out'])
        summary = pd.DataFrame(st.session_state['summary'])
        idf = pd.DataFrame(st.session_state['idf'])
        st.divider()
        st.subheader('Results')
        if len(summary):
            r = summary.iloc[0]
            c = st.columns(5)
            c[0].metric('Images', int(r.get('images_processed', 0)))
            c[1].metric('Visible wells', int(r.get('fully_visible_wells', 0)))
            c[2].metric('PDO wells', int(r.get('PDO_containing_wells', 0)))
            c[3].metric('PDOs', int(r.get('PDO_count', 0)))
            c[4].metric('Mean PDO diameter',
                        f"{float(r['mean_PDO_diameter_um']):.1f} µm"
                        if 'mean_PDO_diameter_um' in r and pd.notna(r['mean_PDO_diameter_um']) else '—')
        st.download_button('Download complete results ZIP', st.session_state['zip'],
                           'KT3_PDO_PSC_analysis_results.zip', 'application/zip',
                           type='primary', use_container_width=True)
        fdir = out/'figures'
        plot_items = []
        for lab, name in [
            ('PDO size distribution', 'PDO_size_distribution.png'),
            ('PSC frequency', 'PSC_count_frequency_across_PDOs.png'),
            ('Cropped PDO wells', 'PDO_well_contact_sheet_compact.png')
        ]:
            p = fdir/name
            if p.exists():
                plot_items.append((lab, p))
        if plot_items:
            cc = st.columns(min(3, len(plot_items)))
            for i, (lab, p) in enumerate(plot_items):
                cc[i % len(cc)].image(str(p), caption=lab, use_container_width=True)
        with st.expander('Image-by-image summary'):
            st.dataframe(idf, use_container_width=True, hide_index=True)
        inds = sorted((out/'indexed_large_images').glob('*.png'))
        if inds:
            with st.expander('Indexed large-image QC overlays'):
                for i in range(0, len(inds), 3):
                    cs = st.columns(3)
                    for col, p in zip(cs, inds[i:i+3]):
                        col.image(str(p), caption=p.stem, use_container_width=True)
