from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


EXPORT_VERSION = 1
EXPECTED_PIXEL_SIZE_UM = 0.732915056602578
CHANNELS = {'gfp': 0, 'rfp': 1, 'dic': 2}
CONDITIONS = {
    'K3T_PSC_RMC6236_Lane_1_DMSO': {'condition_name': 'DMSO', 'dose': '0 nM'},
    'K3T_PSC_RMC6236_5nm_Lane_2': {'condition_name': '5 nM RMC6236', 'dose': '5 nM'},
    'K3T_PSC_RMC6236_25nm_Lane_3': {'condition_name': '25 nM RMC6236', 'dose': '25 nM'},
    'K3T_PSC_RMC6236_50nm_Lane_1': {'condition_name': '50 nM RMC6236', 'dose': '50 nM'},
    'K3T_PSC_RMC6236_100nm_Lane_5': {'condition_name': '100 nM RMC6236', 'dose': '100 nM'},
    'K3T_PSC_RMC6236_150nm_Lane_6': {'condition_name': '150 nM RMC6236', 'dose': '150 nM'},
}
MANIFEST_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'pdo_status', 'PDO_count',
    'total_PDO_projected_area_um2', 'equivalent_circular_diameters_um',
    'x_px_fullres', 'y_px_fullres', 'radius_px', 'qc_status', 'manual_review_status',
    'gfp_channel', 'rfp_channel', 'dic_channel', 'pixel_size_x_um', 'pixel_size_y_um',
    'omezarr_source', 'crop_left_px_fullres', 'crop_top_px_fullres', 'crop_side_px',
    'crop_radius_scale', 'display_ranges_json', 'overlay_provenance', 'restart_signature',
    'labelled_crop', 'raw_dic_crop', 'raw_gfp_crop', 'raw_rfp_crop',
    'raw_composite_crop', 'export_status', 'error',
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def _atomic_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def _number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid or missing numeric field '{key}' in row {row}.") from exc


def _normalise_well_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r'[+-]?\d+\.0+', text):
        return str(int(float(text)))
    if not text:
        raise RuntimeError('Encountered an empty well_id.')
    return text


def _filename(condition_id: str, well_id: str, x: float, y: float) -> str:
    if re.fullmatch(r'\d+', well_id):
        well = f'well_{int(well_id):06d}'
    else:
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', well_id).strip('_')
        well = f'well_{safe}'
    return f'{condition_id}__{well}__x_{int(round(x))}__y_{int(round(y))}.png'


def _signature(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _condition_inputs(folder: Path) -> tuple[dict, list[dict], list[dict]]:
    summary_path = folder / 'condition_summary.json'
    well_path = folder / 'well_measurements.csv'
    pdo_path = folder / 'pdo_measurements.csv'
    for path in (summary_path, well_path, pdo_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required final condition output is missing: {path}')
    summary = _read_json(summary_path)
    if summary.get('completion_status') != 'completed':
        raise RuntimeError(f'Condition is not marked completed in {summary_path}.')
    return summary, _read_csv(well_path), _read_csv(pdo_path)


def resolve_omezarr(condition_id: str, summary: dict, batch_status: dict,
                    cache_root: Path | None) -> Path:
    recorded: list[str] = []
    status_row = (batch_status.get('conditions') or {}).get(condition_id) or {}
    for value in (status_row.get('omezarr'), summary.get('omezarr'),
                  (summary.get('benchmark') or {}).get('source')):
        if value:
            recorded.append(str(value))
    for value in recorded:
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
    if cache_root is None:
        raise FileNotFoundError(
            f'No recorded OME-Zarr path exists for {condition_id}; recorded={recorded}. '
            'Provide --cache-root if the cache moved.'
        )
    root = cache_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'OME-Zarr cache root does not exist: {root}')
    exact_names = {f'{condition_id}.ome.zarr'.lower()}
    source_name = str(summary.get('condition_name') or '').strip()
    if source_name:
        exact_names.add(f'{source_name}.ome.zarr'.lower())
    candidates = sorted(
        p.resolve() for p in root.rglob('*')
        if p.is_dir() and p.name.lower().endswith('.ome.zarr')
        and (p.name.lower() in exact_names or condition_id.lower() in p.name.lower())
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f'Expected exactly one moved OME-Zarr for {condition_id} below {root}; '
            f'found {len(candidates)}: {[str(p) for p in candidates]}'
        )
    return candidates[0]


def _channel_label_matches(label: str, kind: str) -> bool:
    value = ' '.join(str(label).lower().replace('_', ' ').replace('-', ' ').split())
    terms = {
        'gfp': ('gfp', 'egfp', 'green', 'fitc', '488'),
        'rfp': ('rfp', 'red', '561', '568', '594'),
        'dic': ('dic', 'brightfield', 'bright field', 'transmitted', 'transmission', 'dia'),
    }[kind]
    return any(term in value for term in terms)


def validate_omezarr(meta: dict, summary: dict, expected_pixel_size_um: float) -> dict:
    shape = tuple(int(v) for v in meta.get('shape') or ())
    axes = [str(v).upper() for v in meta.get('axes') or ()]
    if len(shape) != len(axes):
        raise RuntimeError(f'OME-Zarr axes/shape mismatch: axes={axes}, shape={shape}.')
    for axis in ('C', 'Y', 'X'):
        if axes.count(axis) != 1:
            raise RuntimeError(f'OME-Zarr must contain exactly one {axis} axis; axes={axes}.')
    for axis in ('T', 'Z'):
        if axis in axes and shape[axes.index(axis)] != 1:
            raise RuntimeError(f'OME-Zarr {axis} axis is not singleton: shape={shape}, axes={axes}.')
    if shape[axes.index('C')] != 3:
        raise RuntimeError(f'Expected exactly three channels GFP/RFP/DIC; got shape={shape}.')
    rows = list(meta.get('channel_metadata') or [])
    if len(rows) < 3:
        raise RuntimeError(f'OME-Zarr is missing three channel metadata rows: {rows}.')
    names = []
    for kind, expected_index in CHANNELS.items():
        row = rows[expected_index]
        actual_index = int(row.get('index', expected_index))
        name = str(row.get('name', '')).strip()
        if actual_index != expected_index or not _channel_label_matches(name, kind):
            raise RuntimeError(
                f'Cannot validate {kind.upper()}={expected_index} from OME metadata: '
                f'index={actual_index}, name={name!r}.'
            )
        names.append(name)
    voxel = meta.get('voxel_size_um') or {}
    try:
        px_x, px_y = float(voxel['x']), float(voxel['y'])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'OME-Zarr lacks valid X/Y physical calibration: {voxel}.') from exc
    if px_x <= 0 or px_y <= 0:
        raise RuntimeError(f'OME-Zarr physical calibration must be positive: {voxel}.')
    tolerance = max(1e-9, abs(expected_pixel_size_um) * 1e-6)
    if abs(px_x - expected_pixel_size_um) > tolerance or abs(px_y - expected_pixel_size_um) > tolerance:
        raise RuntimeError(
            f'OME-Zarr calibration ({px_x}, {px_y}) µm/px does not match expected '
            f'{expected_pixel_size_um} µm/px.'
        )
    recorded = summary.get('pixel_size_um') or {}
    for axis, actual in (('x', px_x), ('y', px_y)):
        if recorded.get(axis) is not None and abs(float(recorded[axis]) - actual) > tolerance:
            raise RuntimeError(
                f'Condition summary {axis.upper()} calibration {recorded[axis]} disagrees '
                f'with OME-Zarr value {actual}.'
            )
    mapping = summary.get('channel_mapping') or {}
    if mapping:
        if int(mapping.get('gfp_channel', -1)) != 0 or int(mapping.get('dic_channel', -1)) != 2:
            raise RuntimeError(f'Condition summary channel mapping disagrees with GFP=0/DIC=2: {mapping}.')
    return {'shape': list(shape), 'axes': axes, 'channel_names': names,
            'pixel_size_um': {'x': px_x, 'y': px_y}}


def _metadata_window(meta: dict, channel: int) -> tuple[float, float] | None:
    try:
        window = meta['channel_metadata'][channel].get('window') or {}
        lo, hi = float(window['start']), float(window['end'])
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return lo, hi
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return None


def _condition_display_range(planes: SingletonTZCYX, channel: int, width: int, height: int,
                             sample_size: int = 256, grid: int = 4) -> tuple[float, float]:
    samples = []
    size = max(16, int(sample_size))
    for gy in range(max(1, int(grid))):
        cy = int(round((gy + 0.5) * height / max(1, int(grid))))
        y0, y1 = max(0, cy - size // 2), min(height, cy + (size + 1) // 2)
        for gx in range(max(1, int(grid))):
            cx = int(round((gx + 0.5) * width / max(1, int(grid))))
            x0, x1 = max(0, cx - size // 2), min(width, cx + (size + 1) // 2)
            tile = planes.read(channel, slice(y0, y1), slice(x0, x1))
            finite = np.asarray(tile)[np.isfinite(tile)]
            if finite.size:
                samples.append(finite.reshape(-1))
    if not samples:
        raise RuntimeError(f'Could not sample finite display values for channel {channel}.')
    values = np.concatenate(samples).astype(np.float64, copy=False)
    lo, hi = (float(v) for v in np.percentile(values, (0.5, 99.5)))
    if hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def display_ranges(meta: dict, planes: SingletonTZCYX, width: int, height: int,
                   sample_size: int = 256, grid: int = 4) -> dict:
    result = {}
    for kind in ('gfp', 'rfp'):
        channel = CHANNELS[kind]
        window = _metadata_window(meta, channel)
        if window is None:
            window = _condition_display_range(planes, channel, width, height, sample_size, grid)
            source = 'condition_wide_sample_percentiles_0.5_99.5'
        else:
            source = 'ome_omero_channel_window'
        result[kind] = {'channel': channel, 'minimum': window[0], 'maximum': window[1],
                        'source': source}
    result['dic'] = {'channel': CHANNELS['dic'], 'source': 'per_crop_local_percentiles_0.5_99.5'}
    return result


def _u8_range(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    return np.clip((data - lo) * (255.0 / max(hi - lo, 1e-12)), 0, 255).astype(np.uint8)


def _u8_local(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if not finite.size:
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = (float(v) for v in np.percentile(finite, (0.5, 99.5)))
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    return np.zeros(data.shape, dtype=np.uint8) if hi <= lo else _u8_range(data, lo, hi)


def _read_padded(planes: SingletonTZCYX, channel: int, cx: float, cy: float,
                 half: int, width: int, height: int) -> tuple[np.ndarray, int, int]:
    left, top = int(round(cx)) - half, int(round(cy)) - half
    side = 2 * half + 1
    x0, x1 = max(0, left), min(width, left + side)
    y0, y1 = max(0, top), min(height, top + side)
    tile = planes.read(channel, slice(y0, y1), slice(x0, x1))
    out = np.zeros((side, side), dtype=tile.dtype)
    out[y0 - top:y1 - top, x0 - left:x1 - left] = tile
    return out, left, top


def _fonts(title_size: int = 22, body_size: int = 16):
    try:
        return (ImageFont.truetype('DejaVuSans-Bold.ttf', title_size),
                ImageFont.truetype('DejaVuSans.ttf', body_size))
    except Exception:
        return ImageFont.load_default(), ImageFont.load_default()


def _raw_images(dic: np.ndarray, gfp: np.ndarray, rfp: np.ndarray,
                ranges: dict) -> dict[str, Image.Image]:
    du8 = _u8_local(dic)
    gu8 = _u8_range(gfp, ranges['gfp']['minimum'], ranges['gfp']['maximum'])
    ru8 = _u8_range(rfp, ranges['rfp']['minimum'], ranges['rfp']['maximum'])
    dic_rgb = np.stack([du8, du8, du8], axis=-1)
    gfp_rgb = np.zeros_like(dic_rgb); gfp_rgb[..., 1] = gu8
    rfp_rgb = np.zeros_like(dic_rgb); rfp_rgb[..., 0] = ru8
    composite = dic_rgb.copy()
    composite[..., 1] = np.maximum(composite[..., 1], gu8)
    composite[..., 0] = np.maximum(composite[..., 0], ru8)
    return {name: Image.fromarray(value) for name, value in
            [('dic', dic_rgb), ('gfp', gfp_rgb), ('rfp', rfp_rgb), ('composite', composite)]}


def _overlay_panel(image: Image.Image, *, panel_size: int, well_x: float, well_y: float,
                   well_radius: float, left: int, top: int, pdos: list[dict],
                   pixel_size_um: float) -> Image.Image:
    source_side = image.width
    out = image.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(out)
    scale = panel_size / source_side
    cx, cy, radius = (well_x - left) * scale, (well_y - top) * scale, well_radius * scale
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                 outline=(255, 255, 0), width=max(2, panel_size // 180))
    for row in pdos:
        px = (_number(row, 'centroid_x_px_fullres') - left) * scale
        py = (_number(row, 'centroid_y_px_fullres') - top) * scale
        diameter_um = _number(row, 'equivalent_circular_diameter_um')
        pr = (diameter_um / (2.0 * pixel_size_um)) * scale
        width_px = max(2, panel_size // 220)
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), outline=(0, 255, 255), width=width_px)
        arm = max(3, panel_size // 100)
        draw.line((px - arm, py, px + arm, py), fill=(255, 0, 255), width=width_px)
        draw.line((px, py - arm, px, py + arm), fill=(255, 0, 255), width=width_px)
    return out


def labelled_four_panel(raw: dict[str, Image.Image], *, condition_name: str, dose: str,
                        well: dict, pdos: list[dict], validation: dict,
                        left: int, top: int, panel_size: int) -> Image.Image:
    gap, title_h = 8, 28
    width = 2 * panel_size + 3 * gap
    title_font, body_font = _fonts()
    well_id = _normalise_well_id(well['well_id'])
    count = int(round(_number(well, 'PDO_count')))
    area = _number(well, 'total_PDO_projected_area_um2')
    diameters = [_number(row, 'equivalent_circular_diameter_um') for row in pdos]
    channel_names = validation['channel_names']
    lines = [
        f'{condition_name} / {dose} | Final well {well_id} | PDO POSITIVE',
        f'PDO count: {count} | Total projected area: {area:.1f} µm² | Diameters: ' +
        ', '.join(f'{value:.1f} µm' for value in diameters),
        f"Full-res x/y: {_number(well, 'x_px_fullres'):.1f}, {_number(well, 'y_px_fullres'):.1f} | "
        f"Well radius: {_number(well, 'radius_px'):.1f} px",
        'Final dominant-array accepted well | Automated QC; not manually reviewed',
        f'Channels: GFP=0 ({channel_names[0]}); RFP=1 ({channel_names[1]}); DIC=2 ({channel_names[2]})',
        'RFP / PSC channel displayed | PSC-like foci not analysed',
    ]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=max(50, width // 9), break_long_words=False) or [''])
    header_h = 14 + 30 + max(0, len(wrapped) - 1) * 23 + 10
    canvas = Image.new('RGB', (width, header_h + 2 * (panel_size + title_h) + 3 * gap), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, header_h), fill='black')
    y = 8
    for index, line in enumerate(wrapped):
        font = title_font if index == 0 else body_font
        draw.text((12, y), line, fill='white', font=font)
        y += 30 if index == 0 else 23
    px_um = (validation['pixel_size_um']['x'] + validation['pixel_size_um']['y']) / 2.0
    panel_specs = [('dic', 'DIC'), ('gfp', 'GFP'), ('rfp', 'RFP / PSC'),
                   ('composite', 'Composite')]
    for index, (key, label) in enumerate(panel_specs):
        col, row = index % 2, index // 2
        x = gap + col * (panel_size + gap)
        py = header_h + gap + row * (panel_size + title_h + gap)
        draw.rectangle((x, py, x + panel_size, py + title_h), fill='black')
        draw.text((x + 7, py + 5), label, fill='white', font=body_font)
        panel = _overlay_panel(
            raw[key], panel_size=panel_size, well_x=_number(well, 'x_px_fullres'),
            well_y=_number(well, 'y_px_fullres'), well_radius=_number(well, 'radius_px'),
            left=left, top=top, pdos=pdos, pixel_size_um=px_um,
        )
        canvas.paste(panel, (x, py + title_h))
    return canvas


def _contact_sheets(paths: list[Path], output: Path, per_page: int = 16) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    # Contact sheets are derived presentation artifacts. Remove only pages made
    # by this exporter so a smaller restarted export cannot leave stale pages.
    for stale in output.glob('page_*.png'):
        stale.unlink()
    generated = []
    for page_index, start in enumerate(range(0, len(paths), per_page), 1):
        page_paths = paths[start:start + per_page]
        opened = []
        try:
            for path in page_paths:
                with Image.open(path) as source:
                    image = source.convert('RGB')
                    scale = min(1.0, 420.0 / image.width)
                    opened.append(image.resize((int(image.width * scale), int(image.height * scale)),
                                               Image.Resampling.LANCZOS))
            cols, gap = 4, 8
            rows = math.ceil(len(opened) / cols)
            cell_w = max(im.width for im in opened)
            cell_h = max(im.height for im in opened)
            sheet = Image.new('RGB', (cols * cell_w + (cols + 1) * gap,
                                      rows * cell_h + (rows + 1) * gap), 'white')
            for index, image in enumerate(opened):
                row, col = divmod(index, cols)
                sheet.paste(image, (gap + col * (cell_w + gap), gap + row * (cell_h + gap)))
            path = output / f'page_{page_index:03d}.png'
            sheet.save(path, dpi=(300, 300))
            generated.append(str(path))
        finally:
            for image in opened:
                image.close()
    return generated


def _relative_outputs(folder: Path, filename: str) -> dict[str, str]:
    base = folder / 'pdo_positive_crops'
    return {
        'labelled_crop': str(base / 'labelled_crops' / filename),
        'raw_dic_crop': str(base / 'raw_crops' / 'dic' / filename),
        'raw_gfp_crop': str(base / 'raw_crops' / 'gfp' / filename),
        'raw_rfp_crop': str(base / 'raw_crops' / 'rfp' / filename),
        'raw_composite_crop': str(base / 'raw_crops' / 'composite' / filename),
    }


def _restart_valid(row: dict | None, signature: str) -> bool:
    return bool(row and row.get('export_status') == 'completed'
                and row.get('restart_signature') == signature
                and all(Path(row[key]).is_file() for key in
                        ('labelled_crop', 'raw_dic_crop', 'raw_gfp_crop',
                         'raw_rfp_crop', 'raw_composite_crop')))


def _upload_additive(client, local_root: Path, bucket: str, prefix: str) -> dict:
    uploaded = skipped = conflicts = 0
    conflict_keys = []
    for path in sorted(p for p in local_root.rglob('*') if p.is_file()):
        rel = path.relative_to(local_root).as_posix()
        key = '/'.join(value.strip('/') for value in (prefix, rel) if value.strip('/'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, 'response', {}) or {}
            if (response.get('Error') or {}).get('Code') not in {'404', 'NoSuchKey', 'NotFound'}:
                raise
        else:
            if (head.get('Metadata') or {}).get('sha256') == digest:
                skipped += 1
            else:
                conflicts += 1; conflict_keys.append(key)
            continue
        client.upload_file(str(path), bucket, key, ExtraArgs={'Metadata': {'sha256': digest}})
        uploaded += 1
    return {'bucket': bucket, 'prefix': prefix, 'uploaded_files': uploaded,
            'skipped_matching_files': skipped, 'conflicting_files': conflicts,
            'conflict_keys': conflict_keys}


def export_condition(condition_id: str, folder: Path, args: argparse.Namespace, batch_status: dict,
                     *, probe: Callable = probe_omezarr, open_group: Callable = zarr.open_group,
                     s3_client=None) -> dict:
    started = _now()
    mapping = CONDITIONS[condition_id]
    summary, wells, pdos = _condition_inputs(folder)
    original_name = str(summary.get('condition_name') or condition_id)
    zarr_path = resolve_omezarr(condition_id, summary, batch_status, args.cache_root)
    meta = probe(zarr_path)
    validation = validate_omezarr(meta, summary, args.expected_pixel_size_um)
    root = open_group(str(zarr_path), mode='r')
    array = root[meta['level0_array_path']]
    planes = SingletonTZCYX(array, meta['axes'])
    channels, height, width = planes.shape_cyx
    if channels != 3:
        raise RuntimeError(f'Validated metadata and opened array disagree: {channels} channels.')
    ranges = display_ranges(meta, planes, width, height, args.display_sample_size,
                            args.display_sample_grid)
    positive = []
    accepted_ids = set()
    for row in wells:
        well_id = _normalise_well_id(row.get('well_id'))
        if well_id in accepted_ids:
            raise RuntimeError(f'Duplicate well_id in final accepted table: {well_id}.')
        accepted_ids.add(well_id)
        if _truthy(row.get('PDO_present')):
            positive.append(row)
    by_well: dict[str, list[dict]] = {}
    for row in pdos:
        well_id = _normalise_well_id(row.get('well_id'))
        if well_id not in accepted_ids:
            raise RuntimeError(f'PDO row refers to non-final well_id {well_id}.')
        by_well.setdefault(well_id, []).append(row)
    for row in positive:
        well_id = _normalise_well_id(row['well_id'])
        expected = int(round(_number(row, 'PDO_count')))
        actual = len(by_well.get(well_id, []))
        if expected <= 0 or expected != actual:
            raise RuntimeError(
                f'Final PDO count mismatch for well {well_id}: well CSV={expected}, PDO rows={actual}.'
            )
    output = folder / 'pdo_positive_crops'
    manifest_path = output / 'pdo_positive_crop_manifest.csv'
    prior_rows = {row['well_id']: row for row in _read_csv(manifest_path)} if manifest_path.is_file() else {}
    rows: list[dict] = []
    output.mkdir(parents=True, exist_ok=True)
    ordered_positive = sorted(positive, key=lambda row: (_number(row, 'y_px_fullres'),
                                                         _number(row, 'x_px_fullres')))
    if not ordered_positive:
        _atomic_csv(manifest_path, rows)
    for well_index, well in enumerate(ordered_positive):
        well_id = _normalise_well_id(well['well_id'])
        objects = sorted(by_well[well_id], key=lambda row: int(float(row.get('pdo_number_in_well', 0))))
        x, y = _number(well, 'x_px_fullres'), _number(well, 'y_px_fullres')
        radius = _number(well, 'radius_px')
        half = max(1, int(round(radius * args.crop_radius_scale)))
        filename = _filename(condition_id, well_id, x, y)
        files = _relative_outputs(folder, filename)
        sig_payload = {
            'export_version': EXPORT_VERSION, 'condition_id': condition_id, 'well_id': well_id,
            'x': x, 'y': y, 'radius': radius, 'pdo_rows': objects,
            'omezarr': str(zarr_path), 'shape': validation['shape'], 'axes': validation['axes'],
            'channels': CHANNELS, 'pixel_size_um': validation['pixel_size_um'],
            'crop_radius_scale': args.crop_radius_scale, 'panel_size': args.panel_size,
            'display_ranges': ranges,
        }
        signature = _signature(sig_payload)
        if _restart_valid(prior_rows.get(well_id), signature):
            rows.append(prior_rows[well_id])
            future_ids = [_normalise_well_id(item['well_id'])
                          for item in ordered_positive[well_index + 1:]]
            _atomic_csv(manifest_path, rows + [prior_rows[item] for item in future_ids
                                               if item in prior_rows])
            continue
        try:
            raw_arrays = {}
            left = top = 0
            for kind in ('dic', 'gfp', 'rfp'):
                raw_arrays[kind], left, top = _read_padded(
                    planes, CHANNELS[kind], x, y, half, width, height)
            images = _raw_images(raw_arrays['dic'], raw_arrays['gfp'], raw_arrays['rfp'], ranges)
            for key, manifest_key in (('dic', 'raw_dic_crop'), ('gfp', 'raw_gfp_crop'),
                                      ('rfp', 'raw_rfp_crop'), ('composite', 'raw_composite_crop')):
                path = Path(files[manifest_key]); path.parent.mkdir(parents=True, exist_ok=True)
                images[key].save(path)
            labelled = labelled_four_panel(
                images, condition_name=mapping['condition_name'], dose=mapping['dose'], well=well,
                pdos=objects, validation=validation, left=left, top=top,
                panel_size=args.panel_size,
            )
            labelled_path = Path(files['labelled_crop']); labelled_path.parent.mkdir(parents=True, exist_ok=True)
            labelled.save(labelled_path, dpi=(300, 300))
            for image in images.values():
                image.close()
            diameters = [_number(row, 'equivalent_circular_diameter_um') for row in objects]
            row = {
                'condition_id': condition_id, 'condition_name': original_name,
                'dose': mapping['dose'], 'well_id': well_id, 'pdo_status': 'PDO POSITIVE',
                'PDO_count': int(round(_number(well, 'PDO_count'))),
                'total_PDO_projected_area_um2': _number(well, 'total_PDO_projected_area_um2'),
                'equivalent_circular_diameters_um': ';'.join(f'{v:.12g}' for v in diameters),
                'x_px_fullres': x, 'y_px_fullres': y, 'radius_px': radius,
                'qc_status': 'Final dominant-array accepted well',
                'manual_review_status': 'Automated QC; not manually reviewed',
                'gfp_channel': 0, 'rfp_channel': 1, 'dic_channel': 2,
                'pixel_size_x_um': validation['pixel_size_um']['x'],
                'pixel_size_y_um': validation['pixel_size_um']['y'],
                'omezarr_source': str(zarr_path), 'crop_left_px_fullres': left,
                'crop_top_px_fullres': top, 'crop_side_px': 2 * half + 1,
                'crop_radius_scale': args.crop_radius_scale,
                'display_ranges_json': json.dumps(ranges, sort_keys=True),
                'overlay_provenance': ('Reconstructed measurement overlay: PDO centroid and '
                                       'equivalent-diameter circle; not a retained segmentation mask.'),
                'restart_signature': signature, **files, 'export_status': 'completed', 'error': '',
            }
        except Exception as exc:
            row = {
                'condition_id': condition_id, 'condition_name': original_name,
                'dose': mapping['dose'], 'well_id': well_id, 'restart_signature': signature,
                **files, 'export_status': 'failed',
                'error': f'{type(exc).__name__}: {exc}',
            }
        rows.append(row)
        future_ids = [_normalise_well_id(item['well_id'])
                      for item in ordered_positive[well_index + 1:]]
        _atomic_csv(manifest_path, rows + [prior_rows[item] for item in future_ids
                                           if item in prior_rows])
    completed_ids = {row['well_id'] for row in rows if row.get('export_status') == 'completed'}
    expected_ids = {_normalise_well_id(row['well_id']) for row in positive}
    qc_ok = completed_ids == expected_ids
    contact_paths = _contact_sheets(
        [Path(row['labelled_crop']) for row in rows if row.get('export_status') == 'completed'],
        output / 'contact_sheets', args.contact_sheet_size,
    )
    export_summary = {
        'export_version': EXPORT_VERSION, 'condition_id': condition_id,
        'condition_name': original_name, 'display_name': mapping['condition_name'],
        'dose': mapping['dose'], 'started_at': started, 'completed_at': _now(),
        'status': 'completed' if qc_ok else 'failed_qc',
        'expected_pdo_positive_wells': len(expected_ids),
        'unique_exported_well_ids': len(completed_ids), 'crop_count_qc_passed': qc_ok,
        'omezarr_source': str(zarr_path), 'omezarr_validation': validation,
        'channel_mapping': CHANNELS, 'display_ranges': ranges,
        'visualization_notice': ('8-bit PNG crops are display products only; the OME-Zarr is the '
                                 'quantitative source of truth.'),
        'dic_scaling_notice': 'DIC uses per-crop local contrast for morphology/QC visibility.',
        'overlay_provenance': ('PDO overlays are reconstructed from final centroids and equivalent '
                               'diameters; original segmentation masks were not retained.'),
        'psc_notice': 'RFP / PSC channel displayed; PSC-like foci not analysed.',
        'crop_settings': {'crop_radius_scale': args.crop_radius_scale,
                          'panel_size': args.panel_size,
                          'contact_sheet_wells_per_page': args.contact_sheet_size},
        'contact_sheets': contact_paths,
        'failed_wells': [row['well_id'] for row in rows if row.get('export_status') != 'completed'],
    }
    # Write the scientific-source/QC summary before any optional upload so the
    # first additive upload includes it. The upload report is then added only to
    # the local copy; a later differing S3 summary is reported as a conflict.
    _atomic_json(output / 'crop_export_summary.json', export_summary)
    if args.upload_s3:
        if s3_client is None:
            from nd2_s3_stage import get_s3_client
            s3_client = get_s3_client(region_name=args.region)
        prefix = '/'.join(value.strip('/') for value in
                          (args.results_s3_prefix, condition_id, 'pdo_positive_crops') if value.strip('/'))
        export_summary['s3_upload'] = _upload_additive(s3_client, output, args.bucket, prefix)
        if export_summary['s3_upload']['conflicting_files']:
            export_summary['status'] = 'completed_with_s3_conflicts' if qc_ok else 'failed_qc'
    _atomic_json(output / 'crop_export_summary.json', export_summary)
    return export_summary


def combine_manifests(result_root: Path) -> list[dict]:
    rows = []
    for condition_id in CONDITIONS:
        path = result_root / condition_id / 'pdo_positive_crops' / 'pdo_positive_crop_manifest.csv'
        if path.is_file():
            rows.extend(_read_csv(path))
    _atomic_csv(result_root / 'all_conditions_pdo_positive_crop_manifest.csv', rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Headless post-processing exporter for final PDO-positive KT3 wells. '
                    'Does not rerun detection or segmentation.')
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--condition-id', action='append', choices=tuple(CONDITIONS), default=[])
    parser.add_argument('--expected-pixel-size-um', type=float, default=EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--crop-radius-scale', type=float, default=1.75)
    parser.add_argument('--panel-size', type=int, default=384)
    parser.add_argument('--contact-sheet-size', type=int, default=16)
    parser.add_argument('--display-sample-size', type=int, default=256)
    parser.add_argument('--display-sample-grid', type=int, default=4)
    parser.add_argument('--upload-s3', action='store_true')
    parser.add_argument('--bucket', default='')
    parser.add_argument('--results-s3-prefix', default='')
    parser.add_argument('--region', default='eu-west-2')
    return parser


def run(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
        open_group: Callable = zarr.open_group, s3_client=None) -> int:
    if args.crop_radius_scale <= 0 or args.panel_size < 64 or args.contact_sheet_size < 1:
        raise ValueError('Crop radius scale, panel size, and contact-sheet size must be positive.')
    if args.upload_s3 and (not args.bucket or not args.results_s3_prefix):
        raise ValueError('--upload-s3 requires --bucket and --results-s3-prefix.')
    result_root = args.result_root.expanduser().resolve()
    status_path = result_root / 'batch_status.json'
    batch_status = _read_json(status_path) if status_path.is_file() else {}
    selected = args.condition_id or list(CONDITIONS)
    failures = 0
    for condition_id in selected:
        try:
            summary = export_condition(
                condition_id, result_root / condition_id, args, batch_status,
                probe=probe, open_group=open_group, s3_client=s3_client,
            )
            if summary['status'] == 'failed_qc':
                failures += 1
            print(f"{condition_id}: {summary['status']} "
                  f"({summary['unique_exported_well_ids']}/"
                  f"{summary['expected_pdo_positive_wells']} wells)", flush=True)
        except Exception as exc:
            failures += 1
            output = result_root / condition_id / 'pdo_positive_crops'
            _atomic_json(output / 'crop_export_summary.json', {
                'export_version': EXPORT_VERSION, 'condition_id': condition_id,
                'condition_name': CONDITIONS[condition_id]['condition_name'],
                'dose': CONDITIONS[condition_id]['dose'], 'status': 'failed',
                'failed_at': _now(), 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            })
            print(f'{condition_id}: FAILED: {type(exc).__name__}: {exc}', flush=True)
        finally:
            combine_manifests(result_root)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())

