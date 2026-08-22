from __future__ import annotations

import argparse
import csv
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
from scipy.ndimage import binary_erosion

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
    _upload_additive,
    display_ranges,
    resolve_omezarr,
    validate_omezarr,
)
from aws_quantify_psc_like_objects import CORE_SAMPLE_SIZE
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


VALIDATION_CROP_VERSION = 1
MANIFEST_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'lane', 'well_id', 'sample_type',
    'sample_reasons', 'PDO_status', 'PDO_count', 'PDO_sizes_um', 'total_PDO_area_um2',
    'PSC_like_resolved_object_count', 'unresolved_PSC_like_cluster_count',
    'background_corrected_RFP_signal', 'PSC_segmentation_status', 'background_qc',
    'threshold_corrected_RFP', 'threshold_detector_RFP', 'labelled_validation_crop',
    'display_ranges_json', 'PDO_overlay_provenance', 'PSC_overlay_provenance',
    'export_status', 'error',
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_csv(path: Path, rows: Iterable[dict], fields: tuple[str, ...] = MANIFEST_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def _lane(condition_id: str) -> str:
    match = re.search(r'(?:^|_)Lane_(\d+)(?:_|$)', condition_id, re.IGNORECASE)
    if not match:
        raise RuntimeError(f'Cannot derive lane number from condition_id {condition_id!r}.')
    return match.group(1)


def _fonts(title_size: int = 21, body_size: int = 15):
    try:
        return (ImageFont.truetype('DejaVuSans-Bold.ttf', title_size),
                ImageFont.truetype('DejaVuSans.ttf', body_size))
    except Exception:
        return ImageFont.load_default(), ImageFont.load_default()


def _format_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'NaN'
    return f'{number:.{digits}f}' if math.isfinite(number) else 'NaN'


def validation_header_lines(condition_id: str, well: dict, pdos: list[dict],
                            well_summary: dict) -> list[str]:
    well_id = _normalise_well_id(well['well_id'])
    pdo_status = 'POSITIVE' if _truthy(well.get('PDO_present')) else 'NEGATIVE'
    diameters = [_number(row, 'equivalent_circular_diameter_um') for row in pdos]
    sizes = ', '.join(f'{value:.1f}' for value in diameters) if diameters else 'none'
    resolved = _format_number(well_summary['PSC_like_resolved_object_count'], 0)
    unresolved = _format_number(well_summary['unresolved_PSC_like_cluster_count'], 0)
    return [
        f"Lane {_lane(condition_id)} | RMC6236 {CONDITIONS[condition_id]['dose']} | "
        f'Final well {well_id} | PDO {pdo_status}',
        f"PDO count: {int(_number(well, 'PDO_count'))} | PDO size(s): {sizes} µm | "
        f"Total PDO area: {_number(well, 'total_PDO_projected_area_um2'):.1f} µm²",
        f'PSC-like resolved objects: {resolved} | Unresolved clusters: {unresolved}',
        f"Background-corrected RFP signal: "
        f"{_format_number(well_summary.get('RFP_background_corrected_mean'))} detector units",
        f"PSC segmentation: {well_summary['PSC_segmentation_status']}",
        f"Threshold: corrected {_format_number(well_summary.get('threshold_corrected_RFP'))}; "
        f"detector {_format_number(well_summary.get('threshold_detector_RFP'))} | "
        f"Background QC: {well_summary['background_qc']} | VALIDATION ONLY",
    ]


def _load_label_mask(well_summary: dict, crop_shape: tuple[int, int], crop_left: int,
                     crop_top: int) -> np.ndarray:
    path = Path(well_summary['mask_path'])
    if not path.is_file():
        raise FileNotFoundError(f'PSC validation mask is missing: {path}')
    with np.load(path) as payload:
        labels = np.asarray(payload['labels'], dtype=np.int32)
        left, top = int(payload['left']), int(payload['top'])
    out = np.zeros(crop_shape, dtype=np.int32)
    x0, y0 = max(crop_left, left), max(crop_top, top)
    x1 = min(crop_left + crop_shape[1], left + labels.shape[1])
    y1 = min(crop_top + crop_shape[0], top + labels.shape[0])
    if x1 > x0 and y1 > y0:
        out[y0 - crop_top:y1 - crop_top, x0 - crop_left:x1 - crop_left] = \
            labels[y0 - top:y1 - top, x0 - left:x1 - left]
    return out


def _overlay(image: Image.Image, *, panel_size: int, well: dict, pdos: list[dict],
             objects: list[dict], labels: np.ndarray, left: int, top: int,
             pixel_size_um: float) -> Image.Image:
    source_side = image.width
    out = image.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(out)
    scale = panel_size / source_side
    x, y, radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
    cx, cy = (x - left) * scale, (y - top) * scale
    width_px = max(2, panel_size // 220)
    draw.ellipse((cx - radius * scale, cy - radius * scale,
                  cx + radius * scale, cy + radius * scale),
                 outline=(255, 255, 0), width=max(2, panel_size // 180))
    for row in pdos:
        px = (_number(row, 'centroid_x_px_fullres') - left) * scale
        py = (_number(row, 'centroid_y_px_fullres') - top) * scale
        pr = (_number(row, 'equivalent_circular_diameter_um') / (2 * pixel_size_um)) * scale
        draw.ellipse((px - pr, py - pr, px + pr, py + pr),
                     outline=(0, 255, 255), width=width_px)
        arm = max(3, panel_size // 100)
        draw.line((px - arm, py, px + arm, py), fill=(255, 0, 255), width=width_px)
        draw.line((px, py - arm, px, py + arm), fill=(255, 0, 255), width=width_px)

    overlay = np.asarray(out).copy()
    for row in objects:
        mask = labels == int(float(row['mask_label']))
        if not np.any(mask):
            raise RuntimeError(f"Mask label {row['mask_label']} is absent for {row['object_id']}.")
        boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        boundary_image = Image.fromarray((boundary.astype(np.uint8) * 255)).resize(
            (panel_size, panel_size), Image.Resampling.NEAREST)
        boundary_scaled = np.asarray(boundary_image) > 0
        color = ((255, 128, 0) if row['object_status'] == 'unresolved_cluster'
                 else (255, 255, 255))
        overlay[boundary_scaled] = color
    out = Image.fromarray(overlay)
    draw = ImageDraw.Draw(out)
    _, body = _fonts(18, 14)
    for resolved_number, row in enumerate(
            [item for item in objects if item['object_status'] == 'resolved'], 1):
        px = (_number(row, 'centroid_x_px_fullres') - left) * scale
        py = (_number(row, 'centroid_y_px_fullres') - top) * scale
        draw.text((px + 3, py + 2), f'R{resolved_number}', fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0), font=body)
    for cluster_number, row in enumerate(
            [item for item in objects if item['object_status'] == 'unresolved_cluster'], 1):
        px = (_number(row, 'centroid_x_px_fullres') - left) * scale
        py = (_number(row, 'centroid_y_px_fullres') - top) * scale
        draw.text((px + 3, py + 2), f'U{cluster_number}', fill=(255, 128, 0),
                  stroke_width=2, stroke_fill=(0, 0, 0), font=body)
    return out


def labelled_validation_crop(raw: dict[str, Image.Image], *, condition_id: str, well: dict,
                             pdos: list[dict], objects: list[dict], well_summary: dict,
                             labels: np.ndarray, validation: dict, left: int, top: int,
                             panel_size: int) -> Image.Image:
    gap, title_h = 8, 28
    width = 2 * panel_size + 3 * gap
    title_font, body_font = _fonts()
    lines = validation_header_lines(condition_id, well, pdos, well_summary)
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=max(55, width // 9),
                                     break_long_words=False) or [''])
    header_h = 14 + 29 + max(0, len(wrapped) - 1) * 22 + 10
    canvas = Image.new('RGB', (width, header_h + 2 * (panel_size + title_h) + 3 * gap), 'white')
    draw = ImageDraw.Draw(canvas); draw.rectangle((0, 0, width, header_h), fill='black')
    cursor_y = 8
    for index, line in enumerate(wrapped):
        draw.text((12, cursor_y), line, fill='white',
                  font=title_font if index == 0 else body_font)
        cursor_y += 29 if index == 0 else 22
    px_um = (validation['pixel_size_um']['x'] + validation['pixel_size_um']['y']) / 2.0
    specs = [('dic', 'DIC'), ('gfp', 'GFP'),
             ('rfp', 'RFP / PSC-like objects'), ('composite', 'Composite — GFP green, RFP red')]
    for index, (key, title) in enumerate(specs):
        col, row = index % 2, index // 2
        x = gap + col * (panel_size + gap)
        y = header_h + gap + row * (panel_size + title_h + gap)
        draw.rectangle((x, y, x + panel_size, y + title_h), fill='black')
        draw.text((x + 7, y + 5), title, fill='white', font=body_font)
        panel = _overlay(raw[key], panel_size=panel_size, well=well, pdos=pdos,
                         objects=objects, labels=labels, left=left, top=top,
                         pixel_size_um=px_um)
        canvas.paste(panel, (x, y + title_h)); panel.close()
    return canvas


def _validation_inputs(folder: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    root = folder / 'psc_object_quantification'
    paths = [root / 'validation_sample_manifest.csv', root / 'psc_well_object_summary.csv',
             root / 'psc_object_measurements.csv', root / 'segmentation_summary.json']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required validation segmentation output is missing: {path}')
    summary = _read_json(paths[3])
    if summary.get('completion_status') != 'validation_sample_completed':
        raise RuntimeError(f'Validation segmentation is not completed: {paths[3]}')
    if summary.get('validation_only') is not True or summary.get('full_well_processing_available') is not False:
        raise RuntimeError('Segmentation summary does not contain the validation-only safety gate.')
    return _read_csv(paths[0]), _read_csv(paths[1]), _read_csv(paths[2]), summary


def export_condition(condition_id: str, folder: Path, args: argparse.Namespace, batch_status: dict,
                     *, probe: Callable = probe_omezarr,
                     open_group: Callable = zarr.open_group, s3_client=None) -> dict:
    condition_summary, wells, pdos = _condition_inputs(folder)
    sample, well_summaries, objects, segmentation_summary = _validation_inputs(folder)
    primary_count = sum(row['sample_type'] == 'primary' for row in sample)
    if primary_count != CORE_SAMPLE_SIZE:
        raise RuntimeError(f'Validation crop export requires exactly {CORE_SAMPLE_SIZE} primary wells.')
    sample_ids = {_normalise_well_id(row['well_id']) for row in sample}
    summary_by_id = {_normalise_well_id(row['well_id']): row for row in well_summaries}
    if len(sample_ids) != len(sample) or set(summary_by_id) != sample_ids:
        raise RuntimeError('Validation crop inputs do not have an exact sampled-well-set match.')
    well_by_id = {_normalise_well_id(row['well_id']): row for row in wells}
    if not sample_ids.issubset(well_by_id):
        raise RuntimeError('Validation sample contains a non-final well ID.')
    pdo_by_id: dict[str, list[dict]] = {}
    for row in pdos:
        pdo_by_id.setdefault(_normalise_well_id(row['well_id']), []).append(row)
    object_by_id: dict[str, list[dict]] = {}
    for row in objects:
        object_by_id.setdefault(_normalise_well_id(row['well_id']), []).append(row)

    zarr_path = resolve_omezarr(condition_id, condition_summary, batch_status, args.cache_root)
    meta = probe(zarr_path); validation = validate_omezarr(
        meta, condition_summary, args.expected_pixel_size_um)
    root = open_group(str(zarr_path), mode='r'); array = root[meta['level0_array_path']]
    planes = SingletonTZCYX(array, meta['axes'])
    channels, height, width = planes.shape_cyx
    if channels != 3:
        raise RuntimeError(f'Validated metadata and opened array disagree: {channels} channels.')
    ranges = display_ranges(meta, planes, width, height, args.display_sample_size,
                            args.display_sample_grid)
    output = folder / 'psc_object_quantification' / 'validation_crops'
    manifest_path = output / 'manifest.csv'; rows = []
    ordered = sorted(sample, key=lambda row: int(float(row['selection_rank'])))
    for sample_row in ordered:
        well_id = _normalise_well_id(sample_row['well_id']); well = well_by_id[well_id]
        well_summary = summary_by_id[well_id]
        well_pdos = sorted(pdo_by_id.get(well_id, []),
                           key=lambda row: int(float(row.get('pdo_number_in_well', 0))))
        well_objects = sorted(object_by_id.get(well_id, []),
                              key=lambda row: int(float(row['object_number_in_well'])))
        x, y, radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
        half = max(1, int(round(radius * args.crop_radius_scale)))
        filename = f'{condition_id}__well_{well_id}__validation.png'
        path = output / 'labelled_crops' / filename
        try:
            arrays = {}; left = top = 0
            for kind in ('dic', 'gfp', 'rfp'):
                arrays[kind], left, top = _read_padded(
                    planes, CHANNELS[kind], x, y, half, width, height)
            images = _raw_images(arrays['dic'], arrays['gfp'], arrays['rfp'], ranges)
            labels = _load_label_mask(well_summary, arrays['rfp'].shape, left, top)
            crop = labelled_validation_crop(
                images, condition_id=condition_id, well=well, pdos=well_pdos,
                objects=well_objects, well_summary=well_summary, labels=labels,
                validation=validation, left=left, top=top, panel_size=args.panel_size)
            path.parent.mkdir(parents=True, exist_ok=True); crop.save(path, dpi=(300, 300))
            crop.close()
            for image in images.values(): image.close()
            row = {
                'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
                'dose': CONDITIONS[condition_id]['dose'], 'lane': _lane(condition_id),
                'well_id': well_id, 'sample_type': sample_row['sample_type'],
                'sample_reasons': well_summary['sample_reasons'],
                'PDO_status': 'POSITIVE' if _truthy(well.get('PDO_present')) else 'NEGATIVE',
                'PDO_count': int(_number(well, 'PDO_count')),
                'PDO_sizes_um': ';'.join(f"{_number(item, 'equivalent_circular_diameter_um'):.12g}"
                                         for item in well_pdos),
                'total_PDO_area_um2': _number(well, 'total_PDO_projected_area_um2'),
                'PSC_like_resolved_object_count': well_summary['PSC_like_resolved_object_count'],
                'unresolved_PSC_like_cluster_count': well_summary['unresolved_PSC_like_cluster_count'],
                'background_corrected_RFP_signal': well_summary['RFP_background_corrected_mean'],
                'PSC_segmentation_status': well_summary['PSC_segmentation_status'],
                'background_qc': well_summary['background_qc'],
                'threshold_corrected_RFP': well_summary['threshold_corrected_RFP'],
                'threshold_detector_RFP': well_summary['threshold_detector_RFP'],
                'labelled_validation_crop': str(path),
                'display_ranges_json': json.dumps(ranges, sort_keys=True),
                'PDO_overlay_provenance': ('Reconstructed centroid/equivalent-diameter overlay; '
                                           'not an original PDO segmentation mask.'),
                'PSC_overlay_provenance': ('Actual provisional connected-component outlines from '
                                           'the validation segmentation mask.'),
                'export_status': 'completed', 'error': '',
            }
        except Exception as exc:
            row = {'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
                   'dose': CONDITIONS[condition_id]['dose'], 'lane': _lane(condition_id),
                   'well_id': well_id, 'sample_type': sample_row['sample_type'],
                   'sample_reasons': sample_row['sample_reasons'],
                   'labelled_validation_crop': str(path), 'export_status': 'failed',
                   'error': f'{type(exc).__name__}: {exc}'}
        rows.append(row); _atomic_csv(manifest_path, rows)
    completed_ids = {row['well_id'] for row in rows if row['export_status'] == 'completed'}
    qc_passed = completed_ids == sample_ids
    contact_sheets = _contact_sheets(
        [Path(row['labelled_validation_crop']) for row in rows if row['export_status'] == 'completed'],
        output / 'contact_sheets', args.contact_sheet_size)
    result = {
        'completion_status': 'validation_crops_completed' if qc_passed else 'failed_qc',
        'validation_crop_version': VALIDATION_CROP_VERSION, 'validation_only': True,
        'full_PDO_positive_crop_regeneration_available': False,
        'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
        'dose': CONDITIONS[condition_id]['dose'], 'lane': _lane(condition_id),
        'completed_at': _now(), 'expected_sampled_wells': len(sample_ids),
        'exported_sampled_wells': len(completed_ids), 'crop_well_set_qc_passed': qc_passed,
        'primary_validation_wells': primary_count,
        'supplemental_QC_wells': len(sample_ids) - primary_count,
        'display_ranges': ranges, 'channel_mapping': CHANNELS,
        'segmentation_parameters': segmentation_summary['segmentation_parameters'],
        'visualization_notice': ('Crops are validation display products only. Segmentation used '
                                 'original quantitative OME-Zarr RFP channel 1 values.'),
        'object_terminology': ('PSC-like resolved objects and unresolved clusters; not true PSC '
                               'cell counts.'),
        'PDO_overlay_provenance': ('Reconstructed measurement overlay; original PDO segmentation '
                                   'masks were not retained.'),
        'PSC_overlay_provenance': 'Actual provisional connected-component mask outlines.',
        'contact_sheets': contact_sheets,
        'failed_wells': [row['well_id'] for row in rows if row['export_status'] != 'completed'],
    }
    _atomic_json(output / 'validation_crop_summary.json', result)
    if args.upload_s3:
        if s3_client is None:
            from nd2_s3_stage import get_s3_client
            s3_client = get_s3_client(region_name=args.region)
        prefix = '/'.join(value.strip('/') for value in
                          (args.results_s3_prefix, condition_id, 'psc_object_quantification')
                          if value.strip('/'))
        result['s3_upload'] = _upload_additive(
            s3_client, folder / 'psc_object_quantification', args.bucket, prefix)
        _atomic_json(output / 'validation_crop_summary.json', result)
    if not qc_passed:
        raise RuntimeError('Validation crop well IDs do not match the selected validation sample.')
    return result


def combine_manifests(result_root: Path) -> None:
    rows = []
    for condition_id in CONDITIONS:
        folder = result_root / condition_id / 'psc_object_quantification' / 'validation_crops'
        try:
            if _read_json(folder / 'validation_crop_summary.json').get(
                    'completion_status') != 'validation_crops_completed':
                continue
        except Exception:
            continue
        path = folder / 'manifest.csv'
        if path.is_file(): rows.extend(_read_csv(path))
    _atomic_csv(result_root / 'all_conditions_psc_validation_crop_manifest.csv', rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='VALIDATION ONLY: four-panel crops for sampled PSC-like segmentations.')
    parser.add_argument('--validation-only', action='store_true', required=True,
                        help='Required safety acknowledgement; final crop regeneration is blocked.')
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--condition-id', action='append', choices=tuple(CONDITIONS), default=[])
    parser.add_argument('--expected-pixel-size-um', type=float, default=EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--crop-radius-scale', type=float, default=1.75)
    parser.add_argument('--panel-size', type=int, default=512)
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
    if not args.validation_only:
        raise RuntimeError('--validation-only is mandatory; final crop regeneration is blocked.')
    if args.crop_radius_scale <= 0 or args.panel_size < 64 or args.contact_sheet_size < 1:
        raise ValueError('Crop scale, panel size, and contact-sheet size must be positive.')
    if args.upload_s3 and (not args.bucket or not args.results_s3_prefix):
        raise ValueError('--upload-s3 requires --bucket and --results-s3-prefix.')
    result_root = args.result_root.expanduser().resolve()
    status_path = result_root / 'batch_status.json'
    batch_status = _read_json(status_path) if status_path.is_file() else {}
    failures = 0; selected = args.condition_id or list(CONDITIONS)
    for condition_id in selected:
        output = result_root / condition_id / 'psc_object_quantification' / 'validation_crops'
        try:
            summary = export_condition(condition_id, result_root / condition_id, args,
                                       batch_status, probe=probe, open_group=open_group,
                                       s3_client=s3_client)
            print(f"{condition_id}: {summary['exported_sampled_wells']} validation crops completed",
                  flush=True)
        except Exception as exc:
            failures += 1
            _atomic_json(output / 'validation_crop_summary.json', {
                'completion_status': 'failed', 'validation_only': True,
                'full_PDO_positive_crop_regeneration_available': False,
                'condition_id': condition_id, 'failed_at': _now(),
                'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(),
            })
        finally:
            try: combine_manifests(result_root)
            except Exception as exc:
                failures += 1
                print(f'{condition_id}: combined manifest failure: {type(exc).__name__}: {exc}',
                      flush=True)
    if not failures:
        print(f'PSC-like validation crops completed: {len(selected)}/{len(selected)} conditions; '
              'final PDO-positive crop regeneration remains blocked.', flush=True)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
