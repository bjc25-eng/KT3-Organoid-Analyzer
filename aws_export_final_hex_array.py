from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from PIL import Image, ImageDraw, ImageFont


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _u8_local(arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, low_pct)); hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a-lo) * (255.0/(hi-lo)), 0, 255).astype(np.uint8)


def _window_end(root, channel: int, dtype) -> float:
    try:
        row = root.attrs.get('omero', {}).get('channels', [])[channel]
        end = float(row.get('window', {}).get('end'))
        if end > 0:
            return end
    except Exception:
        pass
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def _u8_absolute(arr: np.ndarray, maximum: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return np.clip(a * (255.0/max(float(maximum), 1.0)), 0, 255).astype(np.uint8)


def _composite_crop(arr, x0: int, y0: int, x1: int, y1: int, dic_channel: int, gfp_channel: int, gfp_max: float) -> Image.Image:
    dic_raw = np.asarray(arr[int(dic_channel), y0:y1, x0:x1])
    gfp_raw = np.asarray(arr[int(gfp_channel), y0:y1, x0:x1])
    dic = _u8_local(dic_raw)
    gfp = _u8_absolute(gfp_raw, gfp_max)
    rgb = np.stack([dic, dic, dic], axis=-1)
    rgb[..., 1] = np.maximum(rgb[..., 1], gfp)
    return Image.fromarray(rgb)


def _square_bounds(cx: float, cy: float, half: int, width: int, height: int):
    x0 = max(0, int(round(cx))-half); x1 = min(width, int(round(cx))+half+1)
    y0 = max(0, int(round(cy))-half); y1 = min(height, int(round(cy))+half+1)
    return x0, y0, x1, y1


def _pad_square(im: Image.Image) -> Image.Image:
    side = max(im.width, im.height)
    out = Image.new('RGB', (side, side), 'black')
    out.paste(im, ((side-im.width)//2, (side-im.height)//2))
    return out


def _assign_array_indices(wells: pd.DataFrame, pitch_px: float) -> pd.DataFrame:
    out = wells.copy()
    if out.empty:
        out['array_row_index'] = []
        out['array_col_in_row'] = []
        return out
    tol = max(8.0, 0.28*float(pitch_px))
    ys = sorted(float(v) for v in out['y_px_fullres'])
    groups = [[ys[0]]]
    for y in ys[1:]:
        if abs(y-float(np.mean(groups[-1]))) <= tol:
            groups[-1].append(y)
        else:
            groups.append([y])
    centres = np.asarray([float(np.mean(g)) for g in groups])
    out['array_row_index'] = [int(np.argmin(abs(centres-float(y))))+1 for y in out['y_px_fullres']]
    out['array_col_in_row'] = 0
    for row_i, idx in out.groupby('array_row_index').groups.items():
        ordered = out.loc[list(idx)].sort_values('x_px_fullres').index.tolist()
        for col_i, original_idx in enumerate(ordered, 1):
            out.at[original_idx, 'array_col_in_row'] = col_i
    out['array_well_key'] = [f'R{int(r):04d}_C{int(c):03d}' for r,c in zip(out['array_row_index'], out['array_col_in_row'])]
    return out


def _contact_sheet(paths: list[Path], labels: list[str], outpath: Path, cols: int = 6, thumb: int = 220):
    if not paths:
        return
    rows = int(math.ceil(len(paths)/cols))
    label_h = 28
    sheet = Image.new('RGB', (cols*thumb, rows*(thumb+label_h)), 'white')
    font = ImageFont.load_default()
    for i, (path, label) in enumerate(zip(paths, labels)):
        im = Image.open(path).convert('RGB')
        im = _pad_square(im).resize((thumb, thumb), Image.Resampling.LANCZOS)
        cell = Image.new('RGB', (thumb, thumb+label_h), 'white')
        cell.paste(im, (0, label_h))
        ImageDraw.Draw(cell).text((5, 7), label, fill='black', font=font)
        x = (i % cols)*thumb; y = (i // cols)*(thumb+label_h)
        sheet.paste(cell, (x, y))
    sheet.save(outpath, dpi=(300,300))


def _zip_folder(folder: Path, out_zip: Path):
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in folder.rglob('*'):
            if p.is_file() and p.resolve() != out_zip.resolve():
                zf.write(p, p.relative_to(folder))


def main() -> int:
    ap = argparse.ArgumentParser(description='Production export from validated dominant hexagonal microwell-array analysis. Generates all-well/PDO tables, individual PDO crops, PDO-positive well crops, figures, and a ZIP package without rerunning well detection.')
    ap.add_argument('source', type=Path, help='Converted C,Y,X OME-Zarr source.')
    ap.add_argument('refined_results_dir', type=Path, help='Directory containing refined_summary.json.')
    ap.add_argument('hex_results_dir', type=Path, help='Directory containing hex_array_well_measurements.csv and hex_array_pdo_measurements.csv.')
    ap.add_argument('--output-dir', type=Path, default=Path('final_hex_array_export'))
    ap.add_argument('--dic-channel', type=int, default=1)
    ap.add_argument('--gfp-channel', type=int, default=0)
    ap.add_argument('--standard-size', type=int, default=256)
    args = ap.parse_args()

    source = args.source.expanduser().resolve()
    refined = args.refined_results_dir.expanduser().resolve()
    hexdir = args.hex_results_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    tables = out/'tables'; pdo_full = out/'PDO_crops_fullres'; pdo_std = out/'PDO_crops_256'; well_full = out/'PDO_positive_well_crops_fullres'; well_std = out/'PDO_positive_well_crops_256'; figs = out/'figures'
    for d in [tables,pdo_full,pdo_std,well_full,well_std,figs]: d.mkdir(parents=True, exist_ok=True)

    refined_summary = json.loads((refined/'refined_summary.json').read_text(encoding='utf-8'))
    pitch_px = float(refined_summary['pitch_px'])
    wells = pd.read_csv(hexdir/'hex_array_well_measurements.csv')
    pdos = pd.read_csv(hexdir/'hex_array_pdo_measurements.csv')
    wells = _assign_array_indices(wells, pitch_px)
    well_lookup = wells.set_index('well_id').to_dict('index')

    root = zarr.open_group(str(source), mode='r'); arr = root['0']
    if arr.ndim != 3:
        raise RuntimeError(f'Expected C,Y,X OME-Zarr source; got shape {arr.shape}')
    c, height, width = map(int, arr.shape)
    for ch in [args.dic_channel, args.gfp_channel]:
        if ch < 0 or ch >= c:
            raise RuntimeError(f'Channel {ch} unavailable; source has {c} channels.')
    gfp_max = _window_end(root, int(args.gfp_channel), arr.dtype)

    positive_ids = set(int(v) for v in wells.loc[wells['PDO_count'] > 0, 'well_id'])
    well_crop_paths = {}
    for j, wid in enumerate(sorted(positive_ids), 1):
        w = well_lookup[wid]
        x = float(w['x_px_fullres']); y = float(w['y_px_fullres']); r = float(w['radius_px'])
        half = max(70, int(math.ceil(1.25*r)))
        x0,y0,x1,y1 = _square_bounds(x,y,half,width,height)
        im = _pad_square(_composite_crop(arr,x0,y0,x1,y1,args.dic_channel,args.gfp_channel,gfp_max))
        stem = f'well_{wid:05d}_{w["array_well_key"]}'
        full_path = well_full/f'{stem}.png'; std_path = well_std/f'{stem}.png'
        im.save(full_path); im.resize((args.standard_size,args.standard_size),Image.Resampling.LANCZOS).save(std_path)
        well_crop_paths[wid] = (full_path, std_path)
        if j % 50 == 0 or j == len(positive_ids): print(f'PDO-positive well crops: {j}/{len(positive_ids)}', flush=True)

    pdo_records = []
    contact_paths = []; contact_labels = []
    for i, row in pdos.iterrows():
        wid = int(row['well_id']); w = well_lookup[wid]
        pdo_n = int(row['pdo_number_in_well'])
        cx = float(row['centroid_x_px_fullres']); cy = float(row['centroid_y_px_fullres'])
        area_px = float(row['projected_area_px2']); eq_r = math.sqrt(max(area_px,1.0)/math.pi); well_r = float(w['radius_px'])
        half = int(math.ceil(max(28.0, 1.9*eq_r, 0.42*well_r)))
        half = min(half, int(math.ceil(1.20*well_r)))
        x0,y0,x1,y1 = _square_bounds(cx,cy,half,width,height)
        im = _pad_square(_composite_crop(arr,x0,y0,x1,y1,args.dic_channel,args.gfp_channel,gfp_max))
        stem = f'well_{wid:05d}_{w["array_well_key"]}_PDO_{pdo_n:02d}'
        full_path = pdo_full/f'{stem}.png'; std_path = pdo_std/f'{stem}.png'
        im.save(full_path); im.resize((args.standard_size,args.standard_size),Image.Resampling.LANCZOS).save(std_path)
        rec = row.to_dict()
        rec.update({
            'array_row_index': int(w['array_row_index']), 'array_col_in_row': int(w['array_col_in_row']), 'array_well_key': w['array_well_key'],
            'well_x_px_fullres': int(w['x_px_fullres']), 'well_y_px_fullres': int(w['y_px_fullres']),
            'PDO_crop_x0_px_fullres': x0, 'PDO_crop_y0_px_fullres': y0, 'PDO_crop_width_px': x1-x0, 'PDO_crop_height_px': y1-y0,
            'PDO_crop_fullres_png': str(full_path.relative_to(out)), 'PDO_crop_256_png': str(std_path.relative_to(out)),
            'PDO_positive_well_crop_fullres_png': str(well_crop_paths[wid][0].relative_to(out)),
            'PDO_positive_well_crop_256_png': str(well_crop_paths[wid][1].relative_to(out)),
            'qc_status': 'automated_dominant_hex_array_not_manually_reviewed'
        })
        pdo_records.append(rec); contact_paths.append(std_path); contact_labels.append(f'W{wid} PDO{pdo_n}')
        if (i+1) % 50 == 0 or i == len(pdos)-1: print(f'Individual PDO crops: {i+1}/{len(pdos)}', flush=True)

    pdo_df = pd.DataFrame(pdo_records)
    wells['PDO_present'] = wells['PDO_count'] > 0
    wells['PDO_positive_well_crop_fullres_png'] = [str(well_crop_paths[int(wid)][0].relative_to(out)) if int(wid) in well_crop_paths else '' for wid in wells['well_id']]
    wells['PDO_positive_well_crop_256_png'] = [str(well_crop_paths[int(wid)][1].relative_to(out)) if int(wid) in well_crop_paths else '' for wid in wells['well_id']]
    wells['qc_status'] = 'automated_dominant_hex_array_not_manually_reviewed'

    wells.to_csv(tables/'well_data.csv', index=False)
    pdo_df.to_csv(tables/'PDO_data.csv', index=False)
    freq = wells.groupby('PDO_count').size().reset_index(name='well_count').sort_values('PDO_count')
    freq['percent_of_accepted_wells'] = 100.0*freq['well_count']/len(wells)
    freq.to_csv(tables/'PDO_count_frequency.csv', index=False)

    positive_wells = int((wells['PDO_count'] > 0).sum())
    multi_wells = int((wells['PDO_count'] > 1).sum())
    summary = {
        'source': str(source),
        'analysis_scope': 'Detected wells belonging to the dominant connected 100-um hexagonal microwell lattice only; disconnected capture-well/non-array regions excluded.',
        'accepted_hex_array_wells': int(len(wells)),
        'PDO_positive_wells': positive_wells,
        'PDO_negative_wells': int(len(wells)-positive_wells),
        'PDO_objects': int(len(pdo_df)),
        'wells_with_multiple_PDOs': multi_wells,
        'PDO_positive_well_fraction': float(positive_wells/len(wells)) if len(wells) else None,
        'mean_equivalent_circular_diameter_um': float(pdo_df['equivalent_circular_diameter_um'].mean()) if len(pdo_df) else None,
        'median_equivalent_circular_diameter_um': float(pdo_df['equivalent_circular_diameter_um'].median()) if len(pdo_df) else None,
        'mean_projected_area_um2': float(pdo_df['projected_area_um2'].mean()) if len(pdo_df) else None,
        'pixel_size_um_from_refined_analysis': refined_summary.get('pixel_size_um'),
        'inferred_well_pitch_px': pitch_px,
        'inferred_well_pitch_um': refined_summary.get('pitch_um'),
        'individual_PDO_crop_count': int(len(pdo_df)),
        'PDO_positive_well_crop_count': positive_wells,
        'PDO_crop_policy': 'Composite DIC + GFP context crop centred on automated PDO centroid; both full-resolution context and 256x256 standardized PNG exported.',
        'well_ID_note': 'array_row_index/array_col_in_row are source-image indexing for traceability; longitudinal physical-well registration across separate acquisitions requires registration and should not assume these IDs are stable by themselves.',
        'qc_status': 'automated final export from visually validated dominant-component method; biological calls remain automated unless manually reviewed.'
    }
    (out/'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # Thesis/QC plots, using measured values only.
    plt.figure(figsize=(6.2,4.2))
    plt.bar(freq['PDO_count'].astype(str), freq['well_count'])
    plt.xlabel('PDO count per accepted microwell'); plt.ylabel('Number of microwells'); plt.title('PDO occupancy across dominant hexagonal array')
    plt.tight_layout(); plt.savefig(figs/'PDO_count_per_well_distribution.png', dpi=300); plt.close()
    if len(pdo_df):
        plt.figure(figsize=(6.2,4.2)); plt.hist(pdo_df['equivalent_circular_diameter_um'], bins='auto')
        plt.xlabel('Equivalent circular diameter (µm)'); plt.ylabel('PDO count'); plt.title('PDO size distribution')
        plt.tight_layout(); plt.savefig(figs/'PDO_size_distribution.png', dpi=300); plt.close()

    _contact_sheet(contact_paths, contact_labels, figs/'all_PDO_crops_contact_sheet.png', cols=6, thumb=220)

    readme = f'''FINAL HEX-ARRAY PDO EXPORT\n\nThis package contains the production export for one whole-array image.\n\nSelection rule:\n- detect 100 µm microwells\n- infer local lattice spacing\n- keep the dominant connected hexagonal microwell lattice\n- exclude disconnected large bead-capture/non-array regions\n- no synthetic/predicted wells\n- no global DIC-intensity structural cutoff\n\nOutputs:\n- tables/well_data.csv: every accepted analysis microwell and PDO count\n- tables/PDO_data.csv: one row per automated PDO object with size/area/full-resolution coordinates and crop paths\n- tables/PDO_count_frequency.csv: occupancy distribution\n- PDO_crops_fullres/: individual PDO context crops at native crop resolution\n- PDO_crops_256/: standardized 256x256 PDO crops\n- PDO_positive_well_crops_fullres/: full well context for every PDO-positive well\n- PDO_positive_well_crops_256/: standardized positive-well crops\n- figures/: occupancy, size distribution, and all-PDO contact sheet\n- analysis_summary.json: run-level summary\n\nImportant: PDO diameter is 2D equivalent circular diameter from segmented projected area, not a true 3D diameter. Automated results should retain QC status until manually reviewed.\n'''
    (out/'README.txt').write_text(readme, encoding='utf-8')

    zip_path = out.parent/f'{out.name}.zip'
    _zip_folder(out, zip_path)
    print(json.dumps(summary, indent=2))
    print(f'Export directory: {out}')
    print(f'ZIP package: {zip_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
