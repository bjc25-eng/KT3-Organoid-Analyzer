from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, gaussian_filter

from aws_export_pdo_positive_crops import (
    CHANNELS,
    CONDITIONS,
    EXPECTED_PIXEL_SIZE_UM,
    _condition_inputs,
    _contact_sheets,
    _normalise_well_id,
    _number,
    _raw_images,
    _read_csv,
    _read_json,
    _read_padded,
    _truthy,
    _u8_range,
    _upload_additive,
    display_ranges,
    resolve_omezarr,
    validate_omezarr,
)
from aws_psc_validation_round2 import (
    DIAGNOSTIC_RADII,
    MANIFEST_FIELDS,
    MAX_WELLS_PER_CONDITION,
    OBJECT_FIELDS,
    WELL_FIELDS,
    _resolve_mask_path,
)
from aws_quantify_psc_like_objects import DETECTION_GAUSSIAN_SIGMA_PX
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


ROUND2_CROP_VERSION = 1
COLORS = {
    'normal_candidate': (255, 255, 255),
    'wall_proximity_candidate': (255, 0, 255),
    'PDO_overlap_candidate': (0, 255, 255),
    'PDO_overlap_and_wall_candidate': (128, 255, 0),
    'unresolved_cluster': (255, 128, 0),
    'well': (255, 255, 0),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def _fonts(title_size: int = 20, body_size: int = 14):
    try:
        return (ImageFont.truetype('DejaVuSans-Bold.ttf', title_size),
                ImageFont.truetype('DejaVuSans.ttf', body_size))
    except Exception:
        return ImageFont.load_default(), ImageFont.load_default()


def _format(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'NaN'
    return f'{number:.{digits}f}' if math.isfinite(number) else 'NaN'


def _round2_inputs(folder: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    root = folder / 'psc_object_quantification' / 'validation_round2'
    paths = [root / 'diagnostic_manifest.csv', root / 'object_qc_measurements.csv',
             root / 'well_diagnostic_summary.csv', root / 'round2_summary.json']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required Round-2 diagnostic output is missing: {path}')
    summary = _read_json(paths[3])
    if summary.get('completion_status') != 'round2_diagnostics_completed':
        raise RuntimeError(f'Round-2 diagnostics are not completed: {paths[3]}')
    if summary.get('validation_round2_only') is not True:
        raise RuntimeError('Round-2 summary lacks the mandatory diagnostic-only gate.')
    if summary.get('full_well_processing_available') is not False:
        raise RuntimeError('Round-2 summary permits forbidden full-well processing.')
    return _read_csv(paths[0]), _read_csv(paths[1]), _read_csv(paths[2]), summary


def _read_canonical_labels(folder: Path, well_id: str, summary: dict) -> tuple[np.ndarray, int, int]:
    path = _resolve_mask_path(folder, well_id, summary['canonical_mask_path'])
    with np.load(path) as payload:
        return np.asarray(payload['labels'], dtype=np.int32), int(payload['left']), int(payload['top'])


def _embed(source: np.ndarray, source_left: int, source_top: int, shape: tuple[int, int],
           destination_left: int, destination_top: int, *, fill=0) -> np.ndarray:
    result = np.full((*shape, *source.shape[2:]), fill, dtype=source.dtype)
    x0, y0 = max(source_left, destination_left), max(source_top, destination_top)
    x1 = min(source_left + source.shape[1], destination_left + shape[1])
    y1 = min(source_top + source.shape[0], destination_top + shape[0])
    if x1 > x0 and y1 > y0:
        result[y0 - destination_top:y1 - destination_top,
               x0 - destination_left:x1 - destination_left, ...] = source[
                   y0 - source_top:y1 - source_top, x0 - source_left:x1 - source_left, ...]
    return result


def locally_enhanced_rfp(rfp: np.ndarray, well: dict, left: int, top: int,
                         image_width: int | None = None,
                         image_height: int | None = None) -> tuple[Image.Image, dict]:
    x, y, radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
    yy, xx = np.ogrid[:rfp.shape[0], :rfp.shape[1]]
    interior = ((xx + left - x) ** 2 + (yy + top - y) ** 2 <= (0.86 * radius) ** 2)
    if image_width is not None and image_height is not None:
        interior &= ((xx + left >= 0) & (xx + left < image_width)
                     & (yy + top >= 0) & (yy + top < image_height))
    values = np.asarray(rfp[interior], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        raise RuntimeError('No finite RFP pixels exist inside the 0.86r QC display region.')
    lo, hi = (float(value) for value in np.percentile(values, (0.5, 99.5)))
    if hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo: hi = lo + 1.0
    u8 = _u8_range(rfp, lo, hi)
    rgb = np.zeros((*u8.shape, 3), dtype=np.uint8); rgb[..., 0] = u8
    return Image.fromarray(rgb), {
        'lower_percentile': 0.5, 'upper_percentile': 99.5,
        'minimum_detector_value': lo, 'maximum_detector_value': hi,
        'source_region': 'raw RFP channel 1 pixels inside 0.86r',
        'use': 'QC DISPLAY ONLY; NOT USED FOR SEGMENTATION OR MEASUREMENT',
    }


def detection_display(canonical_rfp: np.ndarray, background_median: float,
                      threshold_corrected: float) -> tuple[Image.Image, np.ndarray, dict]:
    corrected = np.asarray(canonical_rfp, dtype=np.float64) - float(background_median)
    detection = gaussian_filter(corrected, sigma=DETECTION_GAUSSIAN_SIGMA_PX, mode='nearest')
    finite = detection[np.isfinite(detection)]
    hi = float(np.percentile(finite, 99.5)) if finite.size else threshold_corrected
    hi = max(hi, float(threshold_corrected), 1e-12)
    u8 = _u8_range(detection, 0.0, hi)
    rgb = np.stack([u8, u8, u8], axis=-1)
    exceeded = detection > float(threshold_corrected)
    rgb[exceeded, 0] = np.maximum(rgb[exceeded, 0], 160)
    rgb[exceeded, 1] = (rgb[exceeded, 1] * 0.35).astype(np.uint8)
    rgb[exceeded, 2] = (rgb[exceeded, 2] * 0.35).astype(np.uint8)
    return Image.fromarray(rgb), exceeded, {
        'background_median_detector_value': background_median,
        'gaussian_sigma_px': DETECTION_GAUSSIAN_SIGMA_PX,
        'threshold_corrected_RFP': threshold_corrected,
        'threshold_exceeded_pixel_count': int(np.count_nonzero(exceeded)),
        'display_range_corrected_RFP': [0.0, hi],
        'red_tint': 'pixels strictly greater than the detection threshold',
    }


def _draw_well_and_pdos(image: Image.Image, well: dict, pdos: list[dict], left: int, top: int,
                        panel_size: int, pixel_size_um: float,
                        source_side: int | None = None) -> Image.Image:
    output = image.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(output); scale = panel_size / (source_side or image.width)
    x, y, radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
    cx, cy = (x - left) * scale, (y - top) * scale
    width = max(2, panel_size // 220)
    draw.ellipse((cx - radius * scale, cy - radius * scale,
                  cx + radius * scale, cy + radius * scale),
                 outline=COLORS['well'], width=max(2, panel_size // 180))
    for row in pdos:
        px = (_number(row, 'centroid_x_px_fullres') - left) * scale
        py = (_number(row, 'centroid_y_px_fullres') - top) * scale
        pr = _number(row, 'equivalent_circular_diameter_um') / (2 * pixel_size_um) * scale
        draw.ellipse((px - pr, py - pr, px + pr, py + pr),
                     outline=(0, 255, 255), width=width)
        arm = max(3, panel_size // 100)
        draw.line((px - arm, py, px + arm, py), fill=(0, 255, 255), width=width)
        draw.line((px, py - arm, px, py + arm), fill=(0, 255, 255), width=width)
    return output


def _short_id(row: dict) -> str:
    value = str(row['canonical_object_id'])
    return value.rsplit('__', 1)[-1].replace('PSCLIKE', 'C')


def overlay_candidates(image: Image.Image, labels: np.ndarray, objects: list[dict],
                       *, panel_size: int, label_filter: np.ndarray | None = None) -> Image.Image:
    output = image.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(output).copy()
    for row in objects:
        mask = labels == int(float(row['canonical_mask_label']))
        if label_filter is not None: mask &= label_filter
        if not np.any(mask): continue
        boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        scaled = np.asarray(Image.fromarray(boundary.astype(np.uint8) * 255).resize(
            (panel_size, panel_size), Image.Resampling.NEAREST)) > 0
        pixels[scaled] = COLORS[row['round2_candidate_status']]
    output = Image.fromarray(pixels); draw = ImageDraw.Draw(output)
    _, font = _fonts(17, 13); scale = panel_size / labels.shape[1]
    for row in objects:
        mask = labels == int(float(row['canonical_mask_label']))
        if label_filter is not None: mask &= label_filter
        if not np.any(mask): continue
        ys, xs = np.nonzero(mask)
        px, py = float(np.mean(xs) * scale), float(np.mean(ys) * scale)
        color = COLORS[row['round2_candidate_status']]
        draw.text((px + 2, py + 2), _short_id(row), fill=color,
                  stroke_width=2, stroke_fill=(0, 0, 0), font=font)
    return output


def radial_strip(local_rfp: Image.Image, labels: np.ndarray, objects: list[dict], well: dict,
                 left: int, top: int, panel_size: int) -> Image.Image:
    gap, title_h = 8, 28
    strip = Image.new('RGB', (3 * panel_size + 4 * gap, panel_size + title_h + 2 * gap), 'white')
    draw = ImageDraw.Draw(strip); _, font = _fonts(18, 14)
    x, y, well_radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
    yy, xx = np.ogrid[:labels.shape[0], :labels.shape[1]]
    radial = np.sqrt((xx + left - x) ** 2 + (yy + top - y) ** 2) / well_radius
    radius_prefixes = {0.75: '0_75', 0.80: '0_80', 0.86: '0_86'}
    for index, radius in enumerate(DIAGNOSTIC_RADII):
        x0 = gap + index * (panel_size + gap)
        draw.rectangle((x0, gap, x0 + panel_size, gap + title_h), fill='black')
        candidate_count = sum(int(float(row[f'radius_{radius_prefixes[radius]}_component_count']))
                              for row in objects)
        draw.text((x0 + 6, gap + 5), f'{radius:.2f}r | components: {candidate_count}',
                  fill='white', font=font)
        panel = overlay_candidates(local_rfp, labels, objects, panel_size=panel_size,
                                   label_filter=radial <= radius)
        panel_draw = ImageDraw.Draw(panel); scale = panel_size / labels.shape[1]
        cx, cy = (x - left) * scale, (y - top) * scale
        panel_draw.ellipse((cx - radius * well_radius * scale, cy - radius * well_radius * scale,
                            cx + radius * well_radius * scale, cy + radius * well_radius * scale),
                           outline=(255, 255, 0), width=max(2, panel_size // 180))
        strip.paste(panel, (x0, gap + title_h)); panel.close()
    return strip


def _legend_lines() -> list[str]:
    return [
        'WHITE = normal candidate | MAGENTA = wall-proximity candidate | '
        'CYAN = PDO-overlap candidate | LIME = PDO + wall candidate',
        'ORANGE = unresolved cluster | YELLOW = final microwell boundary | '
        'CYAN circle/cross = reconstructed PDO extent (not original PDO mask)',
        'Candidate categories are diagnostic QC only; no combined or true PSC count is shown.',
    ]


def labelled_round2_crop(raw: dict[str, Image.Image], local_rfp: Image.Image,
                         detection: Image.Image, labels: np.ndarray, objects: list[dict],
                         well: dict, well_summary: dict, pdos: list[dict], left: int, top: int,
                         panel_size: int, pixel_size_um: float) -> tuple[Image.Image, Image.Image]:
    gap, title_h = 8, 28; width = 3 * panel_size + 4 * gap
    title_font, body_font = _fonts()
    pdo_status = 'POSITIVE' if _truthy(well.get('PDO_present')) else 'NEGATIVE'
    lines = [
        f"{CONDITIONS[well_summary['condition_id']]['condition_name']} | "
        f"{CONDITIONS[well_summary['condition_id']]['dose']} | Final well "
        f"{_normalise_well_id(well['well_id'])} | PDO {pdo_status} | ROUND-2 DIAGNOSTIC ONLY",
        f"Unflagged resolved: {well_summary['PSC_like_unflagged_resolved_count']} | "
        f"PDO overlap: {well_summary['PSC_like_PDO_overlap_candidate_count']} | "
        f"Wall proximity: {well_summary['PSC_like_wall_proximity_candidate_count']} | "
        f"PDO + wall: {well_summary['PSC_like_PDO_overlap_and_wall_candidate_count']} | "
        f"Unresolved: {well_summary['unresolved_PSC_like_cluster_count']}",
        f"Detection threshold: corrected {_format(well_summary['threshold_corrected_RFP'])} "
        f"detector units | Background QC: {well_summary['background_qc']}",
        'RFP LOCALLY ENHANCED - QC DISPLAY ONLY | NOT USED FOR SEGMENTATION OR MEASUREMENT',
        *_legend_lines(),
    ]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=max(80, width // 9),
                                     break_long_words=False) or [''])
    header_h = 12 + 28 + max(0, len(wrapped) - 1) * 21 + 8
    radial = radial_strip(local_rfp, labels, objects, well, left, top, panel_size)
    canvas_h = header_h + 2 * (panel_size + title_h + gap) + radial.height + 2 * gap
    canvas = Image.new('RGB', (width, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas); draw.rectangle((0, 0, width, header_h), fill='black')
    cursor = 7
    for index, line in enumerate(wrapped):
        draw.text((10, cursor), line, fill='white', font=title_font if index == 0 else body_font)
        cursor += 28 if index == 0 else 21
    specs = [
        ('dic', 'DIC', raw['dic'], False), ('gfp', 'GFP', raw['gfp'], False),
        ('rfp', 'RFP condition-consistent', raw['rfp'], False),
        ('local', 'RFP LOCALLY ENHANCED - QC DISPLAY ONLY', local_rfp, True),
        ('detection', 'Background-corrected RFP detection image', detection, True),
        ('composite', 'Composite - GFP green, RFP red', raw['composite'], False),
    ]
    for index, (_, title, image, candidates) in enumerate(specs):
        row, col = divmod(index, 3); x0 = gap + col * (panel_size + gap)
        y0 = header_h + gap + row * (panel_size + title_h + gap)
        draw.rectangle((x0, y0, x0 + panel_size, y0 + title_h), fill='black')
        draw.text((x0 + 5, y0 + 5), title, fill='white', font=body_font)
        panel = _draw_well_and_pdos(image, well, pdos, left, top, panel_size, pixel_size_um)
        if candidates:
            candidate_panel = overlay_candidates(image, labels, objects, panel_size=panel_size)
            # Candidate outlines replace only their coloured boundary/labels; well/PDO geometry is
            # drawn again afterwards so both provenance layers remain visible.
            panel.close(); panel = _draw_well_and_pdos(
                candidate_panel, well, pdos, left, top, panel_size, pixel_size_um,
                source_side=image.width)
            candidate_panel.close()
        canvas.paste(panel, (x0, y0 + title_h)); panel.close()
    canvas.paste(radial, (0, header_h + 2 * (panel_size + title_h + gap) + gap))
    return canvas, radial


def export_condition(condition_id: str, folder: Path, args: argparse.Namespace, batch_status: dict,
                     *, probe: Callable = probe_omezarr,
                     open_group: Callable = zarr.open_group, s3_client=None) -> dict:
    condition_summary, final_wells, pdos = _condition_inputs(folder)
    manifest, object_rows, well_rows, round2_summary = _round2_inputs(folder)
    selected_ids = {_normalise_well_id(row['well_id']) for row in manifest}
    summary_by_id = {_normalise_well_id(row['well_id']): row for row in well_rows}
    if len(selected_ids) != len(manifest) or set(summary_by_id) != selected_ids:
        raise RuntimeError('Round-2 crop inputs do not have an exact diagnostic well-set match.')
    if len(selected_ids) > MAX_WELLS_PER_CONDITION:
        raise RuntimeError('Round-2 crop export exceeds the 12-well hard maximum.')
    final_by_id = {_normalise_well_id(row['well_id']): row for row in final_wells}
    pdo_by_id: dict[str, list[dict]] = {}
    for row in pdos: pdo_by_id.setdefault(_normalise_well_id(row['well_id']), []).append(row)
    objects_by_id: dict[str, list[dict]] = {}
    for row in object_rows:
        objects_by_id.setdefault(_normalise_well_id(row['well_id']), []).append(row)
    zarr_path = resolve_omezarr(condition_id, condition_summary, batch_status, args.cache_root)
    meta = probe(zarr_path); validation = validate_omezarr(
        meta, condition_summary, args.expected_pixel_size_um)
    root = open_group(str(zarr_path), mode='r'); array = root[meta['level0_array_path']]
    planes = SingletonTZCYX(array, meta['axes']); _, height, width = planes.shape_cyx
    ranges = display_ranges(meta, planes, width, height, args.display_sample_size,
                            args.display_sample_grid)
    output = folder / 'psc_object_quantification' / 'validation_round2'
    completed = []
    for manifest_row in sorted(manifest, key=lambda row: int(float(row['selection_rank']))):
        well_id = _normalise_well_id(manifest_row['well_id']); well = final_by_id[well_id]
        well_summary = summary_by_id[well_id]
        objects = sorted(objects_by_id.get(well_id, []),
                         key=lambda row: int(float(row['canonical_mask_label'])))
        well_pdos = sorted(pdo_by_id.get(well_id, []),
                           key=lambda row: int(float(row.get('pdo_number_in_well', 0))))
        labels_source, mask_left, mask_top = _read_canonical_labels(folder, well_id, well_summary)
        x, y, radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
        half = max(1, int(round(radius * args.crop_radius_scale)))
        arrays = {}; left = top = 0
        for kind in ('dic', 'gfp', 'rfp'):
            arrays[kind], left, top = _read_padded(
                planes, CHANNELS[kind], x, y, half, width, height)
        labels = _embed(labels_source, mask_left, mask_top, arrays['rfp'].shape, left, top)
        raw = _raw_images(arrays['dic'], arrays['gfp'], arrays['rfp'], ranges)
        local, local_settings = locally_enhanced_rfp(
            arrays['rfp'], well, left, top, image_width=width, image_height=height)
        canonical_rfp = np.asarray(planes.read(
            CHANNELS['rfp'], slice(mask_top, mask_top + labels_source.shape[0]),
            slice(mask_left, mask_left + labels_source.shape[1])))
        background_median = (_number(well_summary, 'threshold_detector_RFP')
                             - _number(well_summary, 'threshold_corrected_RFP'))
        detection_source, exceeded_source, detection_settings = detection_display(
            canonical_rfp, background_median, _number(well_summary, 'threshold_corrected_RFP'))
        detection_array = _embed(np.asarray(detection_source), mask_left, mask_top,
                                 arrays['rfp'].shape, left, top)
        detection = Image.fromarray(detection_array); detection_source.close()
        labelled, radial = labelled_round2_crop(
            raw, local, detection, labels, objects, well, well_summary, well_pdos,
            left, top, args.panel_size,
            (validation['pixel_size_um']['x'] + validation['pixel_size_um']['y']) / 2.0)
        filename = f'{condition_id}__well_{well_id}__round2.png'
        labelled_path = output / 'labelled_crops' / filename
        radial_path = output / 'radial_comparisons' / filename
        labelled_path.parent.mkdir(parents=True, exist_ok=True)
        radial_path.parent.mkdir(parents=True, exist_ok=True)
        labelled.save(labelled_path, dpi=(300, 300)); radial.save(radial_path, dpi=(300, 300))
        labelled.close(); radial.close(); local.close(); detection.close()
        for image in raw.values(): image.close()
        manifest_row.update({
            'labelled_crop': str(labelled_path), 'radial_comparison': str(radial_path),
            'crop_export_status': 'completed', 'crop_error': '',
        })
        completed.append(well_id)
        _atomic_csv(output / 'diagnostic_manifest.csv', manifest)
        # Per-well display provenance is intentionally stored beside the summary rather than in
        # the quantitative object table.
        display_path = output / 'display_settings' / f'well_{well_id}.json'
        _atomic_json(display_path, {
            'condition_id': condition_id, 'well_id': well_id,
            'condition_consistent_ranges': ranges, 'local_RFP_QC_display': local_settings,
            'detection_display': detection_settings,
            'threshold_exceeded_pixels_in_canonical_region': int(np.count_nonzero(exceeded_source)),
            'segmentation_or_measurement_use_of_local_stretch': False,
        })
    completed_ids = set(completed); qc_passed = completed_ids == selected_ids
    contacts = _contact_sheets(
        [Path(row['labelled_crop']) for row in manifest if row.get('crop_export_status') == 'completed'],
        output / 'contact_sheets', args.contact_sheet_size)
    round2_summary.update({
        'crop_export_status': 'round2_crops_completed' if qc_passed else 'failed_qc',
        'crop_export_completed_at': _now(), 'crop_well_set_qc_passed': qc_passed,
        'exported_crop_count': len(completed_ids), 'condition_consistent_display_ranges': ranges,
        'local_RFP_display_rule': ('Per-well 0.5th–99.5th percentile linear stretch inside 0.86r; '
                                   'QC display only; never used for segmentation or measurement.'),
        'six_panel_layout': ['DIC', 'GFP', 'RFP condition-consistent',
                             'RFP locally enhanced QC', 'RFP detection image', 'Composite'],
        'radial_strip': [0.75, 0.80, 0.86], 'contact_sheets': contacts,
        'full_well_processing_available': False, 'final_crop_regeneration_available': False,
    })
    _atomic_json(output / 'round2_summary.json', round2_summary)
    if args.upload_s3:
        if s3_client is None:
            from nd2_s3_stage import get_s3_client
            s3_client = get_s3_client(region_name=args.region)
        prefix = '/'.join(value.strip('/') for value in
                          (args.results_s3_prefix, condition_id,
                           'psc_object_quantification', 'validation_round2') if value.strip('/'))
        round2_summary['s3_upload'] = _upload_additive(s3_client, output, args.bucket, prefix)
        _atomic_json(output / 'round2_summary.json', round2_summary)
    if not qc_passed:
        raise RuntimeError('Round-2 crop IDs do not match the selected diagnostic IDs.')
    return round2_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='ROUND 2 DIAGNOSTIC ONLY: six-panel and radial-comparison crops.')
    parser.add_argument('--validation-round2-only', action='store_true', required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--condition-id', action='append', choices=tuple(CONDITIONS), default=[])
    parser.add_argument('--expected-pixel-size-um', type=float, default=EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--crop-radius-scale', type=float, default=1.75)
    parser.add_argument('--panel-size', type=int, default=384)
    parser.add_argument('--contact-sheet-size', type=int, default=6)
    parser.add_argument('--display-sample-size', type=int, default=256)
    parser.add_argument('--display-sample-grid', type=int, default=4)
    parser.add_argument('--upload-s3', action='store_true')
    parser.add_argument('--bucket', default='')
    parser.add_argument('--results-s3-prefix', default='')
    parser.add_argument('--region', default='eu-west-2')
    return parser


def run(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
        open_group: Callable = zarr.open_group, s3_client=None) -> int:
    if not args.validation_round2_only:
        raise RuntimeError('--validation-round2-only is mandatory; final/full processing is blocked.')
    if args.crop_radius_scale <= 0 or args.panel_size < 64 or args.contact_sheet_size < 1:
        raise ValueError('Crop scale, panel size, and contact-sheet size must be positive.')
    if args.upload_s3 and (not args.bucket or not args.results_s3_prefix):
        raise ValueError('--upload-s3 requires --bucket and --results-s3-prefix.')
    result_root = args.result_root.expanduser().resolve()
    batch_status_path = result_root / 'batch_status.json'
    batch_status = _read_json(batch_status_path) if batch_status_path.is_file() else {}
    failures = 0; selected = args.condition_id or list(CONDITIONS)
    for condition_id in selected:
        output = result_root / condition_id / 'psc_object_quantification' / 'validation_round2'
        try:
            summary = export_condition(
                condition_id, result_root / condition_id, args, batch_status,
                probe=probe, open_group=open_group, s3_client=s3_client)
            print(f"{condition_id}: Round-2 crops completed ({summary['exported_crop_count']} wells)",
                  flush=True)
        except Exception as exc:
            failures += 1
            print(f'{condition_id}: Round-2 crop failure: {type(exc).__name__}: {exc}', flush=True)
            prior = _read_json(output / 'round2_summary.json') if (output / 'round2_summary.json').is_file() else {}
            prior.update({'crop_export_status': 'failed', 'validation_round2_only': True,
                          'full_well_processing_available': False,
                          'final_crop_regeneration_available': False,
                          'error': f'{type(exc).__name__}: {exc}',
                          'traceback': traceback.format_exc()})
            _atomic_json(output / 'round2_summary.json', prior)
    if not failures:
        print(f'PSC Round-2 diagnostic crops completed: {len(selected)}/{len(selected)} conditions; '
              'full processing and final crop regeneration remain blocked.', flush=True)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
