from __future__ import annotations

import argparse
import csv
import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import zarr
from scipy.ndimage import label

from aws_export_pdo_positive_crops import (
    CHANNELS,
    CONDITIONS,
    EXPECTED_PIXEL_SIZE_UM,
    _condition_inputs,
    _normalise_well_id,
    _number,
    _read_csv,
    _read_json,
    _truthy,
    _upload_additive,
    resolve_omezarr,
    validate_omezarr,
)
from aws_quantify_psc_like_objects import (
    DETECTION_GAUSSIAN_SIGMA_PX,
    INTERIOR_RADIUS_FRACTION,
    MIN_EQUIVALENT_DIAMETER_UM,
    THRESHOLD_K,
    UNRESOLVED_EQUIVALENT_DIAMETER_UM,
)
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


ROUND2_VERSION = 1
TARGET_WELLS_PER_CONDITION = 10
MAX_WELLS_PER_CONDITION = 12
DIAGNOSTIC_RADII = (0.75, 0.80, 0.86)
DMSO_CONDITION = 'K3T_PSC_RMC6236_Lane_1_DMSO'
EXPLICIT_FORCED = {DMSO_CONDITION: ('11163', '15470')}
LOCATE_UNIQUE_ROUND1_IDS = ('6350', '19515')

MANIFEST_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'selection_rank', 'forced',
    'selection_reasons', 'PDO_state', 'round1_sample_type', 'round1_sample_reasons',
    'round1_resolved_object_count', 'round1_unresolved_cluster_count',
    'RFP_background_corrected_mean', 'x_px_fullres', 'y_px_fullres', 'radius_px',
    'labelled_crop', 'radial_comparison', 'crop_export_status', 'crop_error',
)
OBJECT_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'canonical_object_id',
    'canonical_mask_label', 'canonical_round1_object_status', 'round2_candidate_status',
    'touches_wall_QC_zone', 'wall_QC_zone_pixels', 'wall_QC_zone_fraction',
    'maximum_radial_fraction', 'PDO_overlap_pixels', 'PDO_overlap_fraction',
    'GFP_mean_intensity', 'GFP_median_intensity', 'GFP_max_intensity',
    'RFP_mean_intensity', 'RFP_median_intensity', 'RFP_max_intensity',
    'RFP_integrated_intensity', 'background_corrected_RFP_mean',
    'raw_mean_RFP_to_mean_GFP_ratio', 'corrected_RFP_to_mean_GFP_ratio',
    'centroid_x_px_fullres', 'centroid_y_px_fullres', 'area_px2', 'area_um2',
    'equivalent_diameter_um', 'threshold_corrected_RFP', 'threshold_detector_RFP',
    'radius_0_75_status', 'radius_0_75_overlap_pixels', 'radius_0_75_overlap_fraction',
    'radius_0_75_component_count', 'radius_0_75_component_areas_px2',
    'radius_0_80_status', 'radius_0_80_overlap_pixels', 'radius_0_80_overlap_fraction',
    'radius_0_80_component_count', 'radius_0_80_component_areas_px2',
    'radius_0_86_status', 'radius_0_86_overlap_pixels', 'radius_0_86_overlap_fraction',
    'radius_0_86_component_count', 'radius_0_86_component_areas_px2',
    'PDO_mask_provenance', 'scientific_status',
)
WELL_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'selection_rank', 'forced',
    'selection_reasons', 'PDO_present', 'PDO_count', 'total_PDO_projected_area_um2',
    'RFP_background_corrected_mean', 'background_qc', 'threshold_corrected_RFP',
    'threshold_detector_RFP', 'PSC_like_unflagged_resolved_count',
    'PSC_like_PDO_overlap_candidate_count', 'PSC_like_wall_proximity_candidate_count',
    'PSC_like_PDO_overlap_and_wall_candidate_count', 'unresolved_PSC_like_cluster_count',
    'canonical_candidate_count', 'radius_0_75_candidate_count',
    'radius_0_75_resolved_count', 'radius_0_75_unresolved_count',
    'radius_0_75_retained_count', 'radius_0_75_truncated_count',
    'radius_0_75_removed_count', 'radius_0_75_split_count',
    'radius_0_80_candidate_count', 'radius_0_80_resolved_count',
    'radius_0_80_unresolved_count', 'radius_0_80_retained_count',
    'radius_0_80_truncated_count', 'radius_0_80_removed_count',
    'radius_0_80_split_count', 'radius_0_86_candidate_count',
    'radius_0_86_resolved_count', 'radius_0_86_unresolved_count',
    'radius_0_86_retained_count', 'radius_0_86_truncated_count',
    'radius_0_86_removed_count', 'radius_0_86_split_count',
    'canonical_mask_path', 'PSC_round2_status',
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_csv(path: Path, rows: Iterable[dict], fields: tuple[str, ...]) -> None:
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


def _finite(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ''))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _well_key(row: dict) -> tuple:
    well_id = _normalise_well_id(row['well_id'])
    try:
        return 0, int(well_id)
    except ValueError:
        return 1, well_id


def _round1_inputs(folder: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    root = folder / 'psc_object_quantification'
    paths = [root / 'validation_sample_manifest.csv', root / 'psc_well_object_summary.csv',
             root / 'psc_object_measurements.csv', root / 'segmentation_summary.json']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required Round-1 validation output is missing: {path}')
    summary = _read_json(paths[3])
    if summary.get('completion_status') != 'validation_sample_completed':
        raise RuntimeError(f'Round-1 validation is not completed: {paths[3]}')
    if summary.get('validation_only') is not True or summary.get('full_well_processing_available') is not False:
        raise RuntimeError('Round-1 summary does not retain its validation-only safety gate.')
    sample, wells, objects = _read_csv(paths[0]), _read_csv(paths[1]), _read_csv(paths[2])
    sample_ids = {_normalise_well_id(row['well_id']) for row in sample}
    well_ids = {_normalise_well_id(row['well_id']) for row in wells}
    if len(sample_ids) != len(sample) or sample_ids != well_ids:
        raise RuntimeError('Round-1 sample and well summary do not have an exact well-set match.')
    return sample, wells, objects, summary


def locate_forced_wells(result_root: Path) -> tuple[dict[str, set[str]], dict]:
    forced = {condition: set(ids) for condition, ids in EXPLICIT_FORCED.items()}
    locations: dict[str, list[str]] = {well_id: [] for well_id in LOCATE_UNIQUE_ROUND1_IDS}
    missing_explicit = []
    for condition_id in CONDITIONS:
        path = result_root / condition_id / 'psc_object_quantification' / 'validation_sample_manifest.csv'
        try:
            ids = {_normalise_well_id(row['well_id']) for row in _read_csv(path)}
        except Exception:
            ids = set()
        for well_id in LOCATE_UNIQUE_ROUND1_IDS:
            if well_id in ids:
                locations[well_id].append(condition_id)
        for well_id in EXPLICIT_FORCED.get(condition_id, ()):
            if well_id not in ids:
                missing_explicit.append({'condition_id': condition_id, 'well_id': well_id})
    unique, ambiguous, missing = {}, {}, []
    for well_id, condition_ids in locations.items():
        if len(condition_ids) == 1:
            forced.setdefault(condition_ids[0], set()).add(well_id)
            unique[well_id] = condition_ids[0]
        elif len(condition_ids) > 1:
            ambiguous[well_id] = condition_ids
        else:
            missing.append(well_id)
    return forced, {'unique_round1_locations': unique, 'ambiguous_round1_locations': ambiguous,
                    'missing_round1_ids': missing, 'missing_explicit_wells': missing_explicit}


def select_diagnostic_wells(condition_id: str, sample: list[dict], round1_wells: list[dict],
                            forced_ids: set[str], *, target: int = TARGET_WELLS_PER_CONDITION,
                            maximum: int = MAX_WELLS_PER_CONDITION) -> list[dict]:
    if target > maximum or maximum > MAX_WELLS_PER_CONDITION:
        raise ValueError(f'Round-2 selection cannot exceed {MAX_WELLS_PER_CONDITION} wells.')
    sample_by_id = {_normalise_well_id(row['well_id']): row for row in sample}
    well_by_id = {_normalise_well_id(row['well_id']): row for row in round1_wells}
    if set(sample_by_id) != set(well_by_id):
        raise RuntimeError('Round-2 selection requires an exact Round-1 sampled-well set.')
    selected: dict[str, dict] = {}
    rank = 0

    def add(candidates: list[dict], reason: str, limit: int) -> None:
        nonlocal rank
        credited = 0
        for row in candidates:
            if credited >= limit:
                break
            well_id = _normalise_well_id(row['well_id'])
            if well_id in selected:
                if reason not in selected[well_id]['reasons']:
                    selected[well_id]['reasons'].append(reason)
                credited += 1
                continue
            if len(selected) >= target:
                continue
            rank += 1; credited += 1
            selected[well_id] = {'row': row, 'reasons': [reason], 'rank': rank,
                                 'forced': well_id in forced_ids}

    for well_id in sorted(forced_ids, key=lambda value: (len(value), value)):
        if well_id in well_by_id:
            add([well_by_id[well_id]], 'forced_known_problematic_well', 1)
    highest = sorted(round1_wells, key=lambda row: (
        -(_finite(row, 'PSC_like_resolved_object_count') or 0), _well_key(row)))
    add(highest, 'highest_round1_resolved_object_count', 2)
    zeros = [row for row in round1_wells
             if (_finite(row, 'PSC_like_resolved_object_count') or 0) == 0
             and (_finite(row, 'unresolved_PSC_like_cluster_count') or 0) == 0]
    for pdo_state in (True, False):
        add(sorted([row for row in zeros if _truthy(row.get('PDO_present')) == pdo_state],
                   key=_well_key), f'zero_object_PDO_{"positive" if pdo_state else "negative"}', 1)
    finite_rfp = [row for row in round1_wells
                  if _finite(row, 'RFP_background_corrected_mean') is not None]
    for pdo_state in (True, False):
        group = [row for row in finite_rfp if _truthy(row.get('PDO_present')) == pdo_state]
        add(sorted(group, key=lambda row: (-float(_finite(row, 'RFP_background_corrected_mean')),
                                           _well_key(row))),
            f'high_RFP_PDO_{"positive" if pdo_state else "negative"}', 1)
        add(sorted(group, key=lambda row: (float(_finite(row, 'RFP_background_corrected_mean')),
                                           _well_key(row))),
            f'low_RFP_PDO_{"positive" if pdo_state else "negative"}', 1)
    add(sorted(round1_wells, key=lambda row: (
        _truthy(row.get('PDO_present')), _number(row, 'y_px_fullres'), _well_key(row))),
        'deterministic_balance_fill', target)
    if len(selected) > maximum:
        raise RuntimeError(f'Internal error: selected {len(selected)} > hard maximum {maximum}.')
    if len(round1_wells) >= 8 and len(selected) < 8:
        raise RuntimeError('Round-2 diagnostic selection could not reach the minimum of 8 wells.')
    output = []
    for item in sorted(selected.values(), key=lambda value: value['rank']):
        row = item['row']; well_id = _normalise_well_id(row['well_id'])
        sample_row = sample_by_id[well_id]
        output.append({
            'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
            'dose': CONDITIONS[condition_id]['dose'], 'well_id': well_id,
            'selection_rank': item['rank'], 'forced': item['forced'],
            'selection_reasons': ';'.join(item['reasons']),
            'PDO_state': 'positive' if _truthy(row.get('PDO_present')) else 'negative',
            'round1_sample_type': sample_row.get('sample_type', ''),
            'round1_sample_reasons': sample_row.get('sample_reasons', ''),
            'round1_resolved_object_count': row.get('PSC_like_resolved_object_count', ''),
            'round1_unresolved_cluster_count': row.get('unresolved_PSC_like_cluster_count', ''),
            'RFP_background_corrected_mean': row.get('RFP_background_corrected_mean', ''),
            'x_px_fullres': row.get('x_px_fullres', ''),
            'y_px_fullres': row.get('y_px_fullres', ''), 'radius_px': row.get('radius_px', ''),
        })
    return output


def _resolve_mask_path(folder: Path, well_id: str, recorded: object) -> Path:
    path = Path(str(recorded))
    if path.is_file():
        return path.resolve()
    fallback = folder / 'psc_object_quantification' / 'segmentation_masks' / f'well_{well_id}.npz'
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f'Canonical Round-1 mask is missing for well {well_id}: {path}')


def _pdo_qc_mask(shape: tuple[int, int], left: int, top: int, pdos: list[dict],
                  pixel_size_um: float) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    gx, gy = xx + left, yy + top
    result = np.zeros(shape, dtype=bool)
    for row in pdos:
        radius = _number(row, 'equivalent_circular_diameter_um') / (2.0 * pixel_size_um)
        result |= ((gx - _number(row, 'centroid_x_px_fullres')) ** 2
                   + (gy - _number(row, 'centroid_y_px_fullres')) ** 2 <= radius ** 2)
    return result


def radial_comparison(candidate_mask: np.ndarray, radial_fraction: np.ndarray, radius: float,
                      minimum_area_px: float) -> dict:
    restricted = candidate_mask & (radial_fraction <= float(radius))
    components, count = label(restricted, structure=np.ones((3, 3), dtype=np.uint8))
    areas = [int(np.count_nonzero(components == component)) for component in range(1, count + 1)]
    valid = [area for area in areas if area >= minimum_area_px]
    overlap = int(sum(valid)); canonical_area = int(np.count_nonzero(candidate_mask))
    if not valid:
        status = 'removed'
    elif len(valid) > 1:
        status = 'split'
    elif overlap == canonical_area:
        status = 'retained'
    else:
        status = 'truncated'
    return {'status': status, 'overlap_pixels': overlap,
            'overlap_fraction': float(overlap / canonical_area) if canonical_area else 0.0,
            'component_count': len(valid), 'component_areas_px2': valid}


def candidate_qc_rows(condition_id: str, well: dict, canonical_objects: list[dict],
                      labels: np.ndarray, left: int, top: int, rfp: np.ndarray, gfp: np.ndarray,
                      pdos: list[dict], pixel_size_x_um: float,
                      pixel_size_y_um: float, threshold_corrected: float,
                      threshold_detector: float) -> tuple[list[dict], dict]:
    if labels.shape != rfp.shape or labels.shape != gfp.shape:
        raise RuntimeError('Canonical mask, RFP, and GFP regions do not share a shape.')
    x, y, radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
    yy, xx = np.ogrid[:labels.shape[0], :labels.shape[1]]
    radial = np.sqrt((xx + left - x) ** 2 + (yy + top - y) ** 2) / radius
    pixel_area_um2 = pixel_size_x_um * pixel_size_y_um
    px_um = (pixel_size_x_um + pixel_size_y_um) / 2.0
    min_area_px = math.pi * (MIN_EQUIVALENT_DIAMETER_UM / 2.0) ** 2 / pixel_area_um2
    unresolved_area_px = math.pi * (UNRESOLVED_EQUIVALENT_DIAMETER_UM / 2.0) ** 2 / pixel_area_um2
    pdo_mask = _pdo_qc_mask(labels.shape, left, top, pdos, px_um)
    output = []
    category_counts = {
        'normal_candidate': 0, 'PDO_overlap_candidate': 0,
        'wall_proximity_candidate': 0, 'PDO_overlap_and_wall_candidate': 0,
        'unresolved_cluster': 0,
    }
    radial_totals = {value: {'candidate_count': 0, 'resolved_count': 0,
                             'unresolved_count': 0, 'retained_count': 0,
                             'truncated_count': 0, 'removed_count': 0, 'split_count': 0}
                     for value in DIAGNOSTIC_RADII}
    for canonical in sorted(canonical_objects,
                            key=lambda row: int(float(row['object_number_in_well']))):
        mask_label = int(float(canonical['mask_label']))
        mask = labels == mask_label
        if not np.any(mask):
            raise RuntimeError(f'Canonical mask label {mask_label} is absent for '
                               f"{canonical['object_id']}.")
        area = int(np.count_nonzero(mask)); area_um2 = area * pixel_area_um2
        diameter_um = 2.0 * math.sqrt(area_um2 / math.pi)
        overlap_pixels = int(np.count_nonzero(mask & pdo_mask))
        overlap_fraction = overlap_pixels / area
        wall_pixels = int(np.count_nonzero(mask & (radial >= 0.75) & (radial <= 0.86 + 1e-9)))
        wall_fraction = wall_pixels / area
        is_unresolved = (canonical.get('object_status') == 'unresolved_cluster'
                         or diameter_um > UNRESOLVED_EQUIVALENT_DIAMETER_UM)
        if is_unresolved:
            status = 'unresolved_cluster'
        elif overlap_pixels and wall_pixels:
            status = 'PDO_overlap_and_wall_candidate'
        elif overlap_pixels:
            status = 'PDO_overlap_candidate'
        elif wall_pixels:
            status = 'wall_proximity_candidate'
        else:
            status = 'normal_candidate'
        category_counts[status] += 1
        raw_rfp = np.asarray(rfp[mask], dtype=np.float64)
        raw_gfp = np.asarray(gfp[mask], dtype=np.float64)
        gfp_mean = float(np.mean(raw_gfp)); rfp_mean = float(np.mean(raw_rfp))
        corrected = rfp_mean - (threshold_detector - threshold_corrected)
        raw_ratio = rfp_mean / gfp_mean if math.isfinite(gfp_mean) and gfp_mean > 0 else float('nan')
        corrected_ratio = corrected / gfp_mean if math.isfinite(gfp_mean) and gfp_mean > 0 else float('nan')
        comparisons = {}
        for diagnostic_radius in DIAGNOSTIC_RADII:
            comparison = radial_comparison(mask, radial, diagnostic_radius, min_area_px)
            comparisons[diagnostic_radius] = comparison
            total = radial_totals[diagnostic_radius]
            total[comparison['status'] + '_count'] += 1
            total['candidate_count'] += comparison['component_count']
            for child_area in comparison['component_areas_px2']:
                child_diameter = 2.0 * math.sqrt(child_area * pixel_area_um2 / math.pi)
                if child_area > unresolved_area_px:
                    total['unresolved_count'] += 1
                else:
                    total['resolved_count'] += 1
        ys, xs = np.nonzero(mask)
        row = {
            'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
            'dose': CONDITIONS[condition_id]['dose'], 'well_id': _normalise_well_id(well['well_id']),
            'canonical_object_id': canonical['object_id'], 'canonical_mask_label': mask_label,
            'canonical_round1_object_status': canonical['object_status'],
            'round2_candidate_status': status, 'touches_wall_QC_zone': bool(wall_pixels),
            'wall_QC_zone_pixels': wall_pixels, 'wall_QC_zone_fraction': wall_fraction,
            'maximum_radial_fraction': float(np.max(radial[mask])),
            'PDO_overlap_pixels': overlap_pixels, 'PDO_overlap_fraction': overlap_fraction,
            'GFP_mean_intensity': gfp_mean, 'GFP_median_intensity': float(np.median(raw_gfp)),
            'GFP_max_intensity': float(np.max(raw_gfp)), 'RFP_mean_intensity': rfp_mean,
            'RFP_median_intensity': float(np.median(raw_rfp)),
            'RFP_max_intensity': float(np.max(raw_rfp)),
            'RFP_integrated_intensity': float(np.sum(raw_rfp, dtype=np.float64)),
            'background_corrected_RFP_mean': corrected,
            'raw_mean_RFP_to_mean_GFP_ratio': raw_ratio,
            'corrected_RFP_to_mean_GFP_ratio': corrected_ratio,
            'centroid_x_px_fullres': float(np.mean(xs) + left),
            'centroid_y_px_fullres': float(np.mean(ys) + top), 'area_px2': area,
            'area_um2': area_um2, 'equivalent_diameter_um': diameter_um,
            'threshold_corrected_RFP': threshold_corrected,
            'threshold_detector_RFP': threshold_detector,
            'PDO_mask_provenance': ('Reconstructed centroid/equivalent-diameter circles; not the '
                                    'original PDO segmentation mask.'),
            'scientific_status': 'Round-2 diagnostic candidate; not a true PSC cell count.',
        }
        for diagnostic_radius, prefix in ((0.75, 'radius_0_75'), (0.80, 'radius_0_80'),
                                          (0.86, 'radius_0_86')):
            comparison = comparisons[diagnostic_radius]
            row.update({
                f'{prefix}_status': comparison['status'],
                f'{prefix}_overlap_pixels': comparison['overlap_pixels'],
                f'{prefix}_overlap_fraction': comparison['overlap_fraction'],
                f'{prefix}_component_count': comparison['component_count'],
                f'{prefix}_component_areas_px2': ';'.join(map(str, comparison['component_areas_px2'])),
            })
        output.append(row)
    well_summary = {
        'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
        'dose': CONDITIONS[condition_id]['dose'], 'well_id': _normalise_well_id(well['well_id']),
        'PDO_present': _truthy(well.get('PDO_present')),
        'PDO_count': int(_number(well, 'PDO_count')),
        'total_PDO_projected_area_um2': _number(well, 'total_PDO_projected_area_um2'),
        'PSC_like_unflagged_resolved_count': category_counts['normal_candidate'],
        'PSC_like_PDO_overlap_candidate_count': category_counts['PDO_overlap_candidate'],
        'PSC_like_wall_proximity_candidate_count': category_counts['wall_proximity_candidate'],
        'PSC_like_PDO_overlap_and_wall_candidate_count': category_counts['PDO_overlap_and_wall_candidate'],
        'unresolved_PSC_like_cluster_count': category_counts['unresolved_cluster'],
        'canonical_candidate_count': len(output), 'PSC_round2_status': 'diagnostic_QC_only',
    }
    for diagnostic_radius, prefix in ((0.75, 'radius_0_75'), (0.80, 'radius_0_80'),
                                      (0.86, 'radius_0_86')):
        for key, value in radial_totals[diagnostic_radius].items():
            well_summary[f'{prefix}_{key}'] = value
    return output, well_summary


def _read_mask(path: Path) -> tuple[np.ndarray, int, int]:
    with np.load(path) as payload:
        return (np.asarray(payload['labels'], dtype=np.int32),
                int(payload['left']), int(payload['top']))


def diagnose_condition(condition_id: str, folder: Path, selected: list[dict],
                       args: argparse.Namespace, batch_status: dict,
                       *, probe: Callable = probe_omezarr,
                       open_group: Callable = zarr.open_group,
                       s3_client=None, forced_qc: dict | None = None) -> dict:
    condition_summary, final_wells, pdos = _condition_inputs(folder)
    _, round1_wells, round1_objects, _ = _round1_inputs(folder)
    final_by_id = {_normalise_well_id(row['well_id']): row for row in final_wells}
    round1_by_id = {_normalise_well_id(row['well_id']): row for row in round1_wells}
    pdo_by_id: dict[str, list[dict]] = {}
    for row in pdos: pdo_by_id.setdefault(_normalise_well_id(row['well_id']), []).append(row)
    object_by_id: dict[str, list[dict]] = {}
    for row in round1_objects:
        object_by_id.setdefault(_normalise_well_id(row['well_id']), []).append(row)
    selected_ids = {_normalise_well_id(row['well_id']) for row in selected}
    if not selected_ids.issubset(final_by_id) or not selected_ids.issubset(round1_by_id):
        raise RuntimeError('Round-2 selection contains a non-final or non-Round-1 well ID.')
    if len(selected_ids) > MAX_WELLS_PER_CONDITION:
        raise RuntimeError(f'Round-2 hard maximum exceeded: {len(selected_ids)} wells.')
    zarr_path = resolve_omezarr(condition_id, condition_summary, batch_status, args.cache_root)
    meta = probe(zarr_path); validation = validate_omezarr(
        meta, condition_summary, args.expected_pixel_size_um)
    root = open_group(str(zarr_path), mode='r'); array = root[meta['level0_array_path']]
    planes = SingletonTZCYX(array, meta['axes'])
    output = folder / 'psc_object_quantification' / 'validation_round2'
    _atomic_csv(output / 'diagnostic_manifest.csv', selected, MANIFEST_FIELDS)
    object_rows, well_rows = [], []
    for selection in selected:
        well_id = _normalise_well_id(selection['well_id']); well = final_by_id[well_id]
        round1_well = round1_by_id[well_id]
        mask_path = _resolve_mask_path(folder, well_id, round1_well['mask_path'])
        labels, left, top = _read_mask(mask_path)
        y_slice, x_slice = slice(top, top + labels.shape[0]), slice(left, left + labels.shape[1])
        rfp = np.asarray(planes.read(CHANNELS['rfp'], y_slice, x_slice))
        gfp = np.asarray(planes.read(CHANNELS['gfp'], y_slice, x_slice))
        objects, summary = candidate_qc_rows(
            condition_id, well, object_by_id.get(well_id, []), labels, left, top, rfp, gfp,
            pdo_by_id.get(well_id, []), validation['pixel_size_um']['x'],
            validation['pixel_size_um']['y'], _number(round1_well, 'threshold_corrected_RFP'),
            _number(round1_well, 'threshold_detector_RFP'))
        summary.update({
            'selection_rank': selection['selection_rank'], 'forced': selection['forced'],
            'selection_reasons': selection['selection_reasons'],
            'RFP_background_corrected_mean': round1_well['RFP_background_corrected_mean'],
            'background_qc': round1_well['background_qc'],
            'threshold_corrected_RFP': round1_well['threshold_corrected_RFP'],
            'threshold_detector_RFP': round1_well['threshold_detector_RFP'],
            'canonical_mask_path': str(mask_path),
        })
        object_rows.extend(objects); well_rows.append(summary)
        _atomic_csv(output / 'object_qc_measurements.csv', object_rows, OBJECT_FIELDS)
        _atomic_csv(output / 'well_diagnostic_summary.csv', well_rows, WELL_FIELDS)
    measured_ids = {row['well_id'] for row in well_rows}
    result = {
        'completion_status': 'round2_diagnostics_completed', 'round2_version': ROUND2_VERSION,
        'validation_round2_only': True, 'full_well_processing_available': False,
        'final_crop_regeneration_available': False, 'condition_id': condition_id,
        'condition_name': CONDITIONS[condition_id]['condition_name'],
        'dose': CONDITIONS[condition_id]['dose'], 'completed_at': _now(),
        'selected_well_count': len(selected_ids), 'selected_well_ids': sorted(selected_ids),
        'diagnostic_well_set_qc_passed': selected_ids == measured_ids,
        'canonical_baseline': ('Round-1 saved 0.86r connected-component mask and stable object IDs; '
                               'not regenerated or renumbered.'),
        'radial_comparisons': [0.75, 0.80, 0.86],
        'radial_matching': ('Each smaller-radius component is matched to its canonical 0.86r '
                            'candidate by direct pixel overlap within that canonical mask.'),
        'fixed_parameters': {
            'threshold_k': THRESHOLD_K,
            'detection_gaussian_sigma_px': DETECTION_GAUSSIAN_SIGMA_PX,
            'minimum_equivalent_diameter_um': MIN_EQUIVALENT_DIAMETER_UM,
            'unresolved_equivalent_diameter_um': UNRESOLVED_EQUIVALENT_DIAMETER_UM,
            'canonical_interior_radius_fraction': INTERIOR_RADIUS_FRACTION,
            'touching_object_splitting': False,
        },
        'PDO_mask_provenance': ('Reconstructed final PDO centroid/equivalent-diameter circles; '
                                'not the original PDO segmentation masks.'),
        'scientific_status': ('Diagnostic candidate categories only. No combined biological PSC '
                              'count or true PSC count is calculated.'),
        'omezarr_source': str(zarr_path), 'omezarr_validation': validation,
        'forced_well_resolution': forced_qc or {},
    }
    if not result['diagnostic_well_set_qc_passed']:
        raise RuntimeError('Round-2 diagnostic output IDs do not match the selected IDs.')
    _atomic_json(output / 'round2_summary.json', result)
    if args.upload_s3:
        if s3_client is None:
            from nd2_s3_stage import get_s3_client
            s3_client = get_s3_client(region_name=args.region)
        prefix = '/'.join(value.strip('/') for value in
                          (args.results_s3_prefix, condition_id,
                           'psc_object_quantification', 'validation_round2') if value.strip('/'))
        result['s3_upload'] = _upload_additive(s3_client, output, args.bucket, prefix)
        _atomic_json(output / 'round2_summary.json', result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='ROUND 2 DIAGNOSTIC ONLY: bounded PSC-like candidate QC.')
    parser.add_argument('--validation-round2-only', action='store_true', required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--condition-id', action='append', choices=tuple(CONDITIONS), default=[])
    parser.add_argument('--expected-pixel-size-um', type=float, default=EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--upload-s3', action='store_true')
    parser.add_argument('--bucket', default='')
    parser.add_argument('--results-s3-prefix', default='')
    parser.add_argument('--region', default='eu-west-2')
    return parser


def run(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
        open_group: Callable = zarr.open_group, s3_client=None) -> int:
    if not args.validation_round2_only:
        raise RuntimeError('--validation-round2-only is mandatory; full processing is blocked.')
    if args.upload_s3 and (not args.bucket or not args.results_s3_prefix):
        raise ValueError('--upload-s3 requires --bucket and --results-s3-prefix.')
    result_root = args.result_root.expanduser().resolve()
    batch_status_path = result_root / 'batch_status.json'
    batch_status = _read_json(batch_status_path) if batch_status_path.is_file() else {}
    forced, forced_qc = locate_forced_wells(result_root)
    failures = 0; selected_conditions = args.condition_id or list(CONDITIONS)
    for condition_id in selected_conditions:
        output = result_root / condition_id / 'psc_object_quantification' / 'validation_round2'
        try:
            sample, round1_wells, _, _ = _round1_inputs(result_root / condition_id)
            selected = select_diagnostic_wells(
                condition_id, sample, round1_wells, forced.get(condition_id, set()))
            summary = diagnose_condition(
                condition_id, result_root / condition_id, selected, args, batch_status,
                probe=probe, open_group=open_group, s3_client=s3_client,
                forced_qc=forced_qc)
            _atomic_json(output / 'round2_summary.json', summary)
            print(f"{condition_id}: Round-2 diagnostics completed "
                  f"({summary['selected_well_count']} wells)", flush=True)
        except Exception as exc:
            failures += 1
            _atomic_json(output / 'round2_summary.json', {
                'completion_status': 'failed', 'validation_round2_only': True,
                'full_well_processing_available': False,
                'final_crop_regeneration_available': False, 'condition_id': condition_id,
                'failed_at': _now(), 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            })
    if not failures:
        print(f'PSC Round-2 diagnostics completed: {len(selected_conditions)}/'
              f'{len(selected_conditions)} conditions; full processing remains blocked.', flush=True)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
