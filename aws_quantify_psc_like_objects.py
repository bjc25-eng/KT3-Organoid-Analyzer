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
from scipy.ndimage import gaussian_filter, label
from scipy.spatial import cKDTree

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
from aws_quantify_psc_rfp import (
    BACKGROUND_INNER_RADIUS_FRACTION,
    BACKGROUND_OUTER_RADIUS_FRACTION,
    INTERIOR_RADIUS_FRACTION,
    NEIGHBOUR_EXCLUSION_RADIUS_FRACTION,
    RunLogger,
)
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


SEGMENTATION_VERSION = 1
CORE_SAMPLE_SIZE = 30
RFP_STRATA = 3
Y_BANDS = 5
THRESHOLD_K = 3.0
DETECTION_GAUSSIAN_SIGMA_PX = 0.75
MIN_EQUIVALENT_DIAMETER_UM = 3.0
UNRESOLVED_EQUIVALENT_DIAMETER_UM = 30.0
TOUCHING_OBJECT_SPLITTING = False

SAMPLE_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'sample_type', 'sample_reasons',
    'RFP_stratum', 'PDO_state', 'Y_band', 'x_px_fullres', 'y_px_fullres',
    'background_qc', 'selection_rank',
)
OBJECT_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'object_id', 'object_number_in_well',
    'object_status', 'centroid_x_px_fullres', 'centroid_y_px_fullres', 'area_px2', 'area_um2',
    'equivalent_diameter_um', 'mean_RFP_intensity', 'median_RFP_intensity',
    'max_RFP_intensity', 'integrated_RFP_intensity', 'background_corrected_mean_RFP',
    'background_median_RFP', 'background_robust_sigma_RFP',
    'threshold_corrected_RFP', 'threshold_detector_RFP', 'peak_excess_over_threshold_RFP',
    'peak_excess_sigma', 'touches_well_interior_boundary', 'saturated_pixel_fraction',
    'mask_label', 'segmentation_qc_status',
)
WELL_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'sample_type', 'sample_reasons',
    'x_px_fullres', 'y_px_fullres', 'radius_px', 'PDO_present', 'PDO_count',
    'total_PDO_projected_area_um2', 'RFP_channel', 'background_qc',
    'RFP_background_corrected_mean', 'background_robust_sigma_RFP',
    'threshold_corrected_RFP', 'threshold_detector_RFP',
    'PSC_like_resolved_object_count', 'unresolved_PSC_like_cluster_count',
    'rejected_small_component_count', 'PSC_like_total_area_px2', 'PSC_like_total_area_um2',
    'PSC_like_median_object_area_um2', 'PSC_like_median_object_diameter_um',
    'unresolved_cluster_total_area_um2', 'PSC_segmentation_status',
    'mask_path', 'mask_left_px_fullres', 'mask_top_px_fullres',
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_csv(path: Path, rows: Iterable[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
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


def _stable_well_key(row: dict) -> tuple:
    well_id = _normalise_well_id(row['well_id'])
    try:
        return (0, int(well_id))
    except ValueError:
        return (1, well_id)


def _rank_groups(rows: list[dict], key: str, groups: int) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (_number(row, key), _stable_well_key(row)))
    count = len(ordered)
    return {
        _normalise_well_id(row['well_id']): min(groups - 1, index * groups // max(1, count))
        for index, row in enumerate(ordered)
    }


def select_validation_sample(condition_id: str, wells: list[dict], psc_rows: list[dict],
                             *, qc_per_category: int = 2,
                             core_size: int = CORE_SAMPLE_SIZE) -> tuple[list[dict], dict]:
    """Select a deterministic validation-only sample; never expands to all final wells."""
    if core_size != CORE_SAMPLE_SIZE:
        raise ValueError(f'Production core sample size is fixed at {CORE_SAMPLE_SIZE}.')
    psc_by_id = {_normalise_well_id(row['well_id']): row for row in psc_rows}
    well_by_id = {_normalise_well_id(row['well_id']): row for row in wells}
    if len(psc_by_id) != len(psc_rows) or len(well_by_id) != len(wells):
        raise RuntimeError('Duplicate well_id in final or PSC well table.')
    if set(psc_by_id) != set(well_by_id):
        raise RuntimeError('PSC validation sampling requires an exact final-well-set match.')

    joined = []
    for well_id, well in well_by_id.items():
        row = {**well, **psc_by_id[well_id], 'well_id': well_id}
        row['_pdo_state'] = 'positive' if _truthy(well.get('PDO_present')) else 'negative'
        joined.append(row)
    eligible = [row for row in joined
                if row.get('background_qc') == 'valid_local_background'
                and _finite(row, 'RFP_background_corrected_mean') is not None]
    if len(eligible) < core_size:
        raise RuntimeError(f'{condition_id} has only {len(eligible)} background-valid wells; '
                           f'{core_size} are required for the primary validation sample.')
    rfp_group = _rank_groups(eligible, 'RFP_background_corrected_mean', RFP_STRATA)
    y_group = _rank_groups(eligible, 'y_px_fullres', Y_BANDS)
    for row in eligible:
        well_id = row['well_id']
        row['_rfp_stratum'] = rfp_group[well_id]
        row['_y_band'] = y_group[well_id]

    # Reserve QC examples before constructing the core so they are always additional and retain
    # an explicit qc_supplement label instead of being absorbed into the primary 30.
    reserved: dict[str, dict] = {}
    def reserve(candidates: list[dict], reason: str) -> None:
        added = 0
        for row in candidates:
            if added >= max(0, int(qc_per_category)):
                break
            well_id = row['well_id']
            if well_id in reserved:
                continue
            reserved[well_id] = {'row': row, 'reasons': [reason]}
            added += 1

    reserve(sorted(
        [row for row in joined if (_finite(row, 'RFP_saturated_pixel_count') or 0) > 0],
        key=lambda row: (-(_finite(row, 'RFP_saturated_pixel_fraction') or 0), _stable_well_key(row))),
        'saturated_RFP')
    reserve(sorted(
        [row for row in joined if row.get('background_qc') != 'valid_local_background'],
        key=lambda row: ((_finite(row, 'background_valid_fraction') or 0), _stable_well_key(row))),
        'insufficient_background')
    positive_area = sorted(
        [row for row in joined if _finite(row, 'exploratory_RFP_positive_area_um2') is not None],
        key=lambda row: (-float(_finite(row, 'exploratory_RFP_positive_area_um2')), _stable_well_key(row)))
    reserve(positive_area, 'highest_RFP_positive_area')
    reserve(positive_area, 'very_large_candidate_screen')
    core_eligible = [row for row in eligible if row['well_id'] not in reserved]
    if len(core_eligible) < core_size:
        raise RuntimeError(
            f'{condition_id} cannot provide {core_size} primary wells plus separately labelled QC '
            f'supplements; only {len(core_eligible)} non-supplement background-valid wells remain.')

    selected: dict[str, dict] = {}
    empty_cells = []
    rank = 0
    for rfp in range(RFP_STRATA):
        for pdo in ('negative', 'positive'):
            for y_band in range(Y_BANDS):
                candidates = [row for row in core_eligible
                              if row['_rfp_stratum'] == rfp and row['_pdo_state'] == pdo
                              and row['_y_band'] == y_band]
                if not candidates:
                    empty_cells.append({'RFP_stratum': rfp, 'PDO_state': pdo, 'Y_band': y_band})
                    continue
                signal_mid = float(np.median([_number(row, 'RFP_background_corrected_mean')
                                              for row in candidates]))
                y_mid = float(np.median([_number(row, 'y_px_fullres') for row in candidates]))
                chosen = min(candidates, key=lambda row: (
                    abs(_number(row, 'RFP_background_corrected_mean') - signal_mid),
                    abs(_number(row, 'y_px_fullres') - y_mid), _stable_well_key(row)))
                rank += 1
                selected[chosen['well_id']] = {
                    'row': chosen, 'sample_type': 'primary',
                    'reasons': [f'core_rfp_{rfp}_pdo_{pdo}_y_band_{y_band + 1}'], 'rank': rank,
                }

    # Empty factorial cells do not reduce the fixed primary sample size. Fill deterministically
    # from unused eligible wells while favouring the least represented factor levels.
    while len(selected) < core_size:
        remaining = [row for row in core_eligible if row['well_id'] not in selected]
        if not remaining:
            raise RuntimeError(f'Could not fill {core_size} unique primary validation wells.')
        counts_rfp = {value: 0 for value in range(RFP_STRATA)}
        counts_y = {value: 0 for value in range(Y_BANDS)}
        counts_pdo = {value: 0 for value in ('negative', 'positive')}
        for item in selected.values():
            row = item['row']; counts_rfp[row['_rfp_stratum']] += 1
            counts_y[row['_y_band']] += 1; counts_pdo[row['_pdo_state']] += 1
        chosen = min(remaining, key=lambda row: (
            counts_rfp[row['_rfp_stratum']] + counts_y[row['_y_band']]
            + counts_pdo[row['_pdo_state']],
            counts_rfp[row['_rfp_stratum']], counts_pdo[row['_pdo_state']],
            counts_y[row['_y_band']], _stable_well_key(row)))
        rank += 1
        selected[chosen['well_id']] = {
            'row': chosen, 'sample_type': 'primary',
            'reasons': ['core_fill_for_empty_factor_cell'], 'rank': rank,
        }

    for item in reserved.values():
        rank += 1
        selected[item['row']['well_id']] = {
            'row': item['row'], 'sample_type': 'qc_supplement',
            'reasons': item['reasons'], 'rank': rank,
        }

    mapping = CONDITIONS[condition_id]
    output = []
    for item in sorted(selected.values(), key=lambda value: value['rank']):
        row = item['row']
        output.append({
            'condition_id': condition_id, 'condition_name': mapping['condition_name'],
            'dose': mapping['dose'], 'well_id': row['well_id'],
            'sample_type': item['sample_type'], 'sample_reasons': ';'.join(item['reasons']),
            'RFP_stratum': row.get('_rfp_stratum', ''), 'PDO_state': row['_pdo_state'],
            'Y_band': (row.get('_y_band', -1) + 1 if '_y_band' in row else ''),
            'x_px_fullres': _number(row, 'x_px_fullres'),
            'y_px_fullres': _number(row, 'y_px_fullres'),
            'background_qc': row.get('background_qc', ''), 'selection_rank': item['rank'],
        })
    return output, {'primary_count': sum(row['sample_type'] == 'primary' for row in output),
                    'supplement_count': sum(row['sample_type'] == 'qc_supplement' for row in output),
                    'empty_core_factor_cells': empty_cells}


def _read_well_region(planes: SingletonTZCYX, channel: int, well: dict,
                      width: int, height: int) -> tuple[np.ndarray, int, int]:
    x, y, radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
    margin = int(math.ceil(radius * BACKGROUND_OUTER_RADIUS_FRACTION)) + 2
    left, right = max(0, int(math.floor(x)) - margin), min(width, int(math.floor(x)) + margin + 1)
    top, bottom = max(0, int(math.floor(y)) - margin), min(height, int(math.floor(y)) + margin + 1)
    return np.asarray(planes.read(channel, slice(top, bottom), slice(left, right))), left, top


def _masks(shape: tuple[int, int], left: int, top: int, well: dict, all_wells: list[dict],
           tree: cKDTree, max_radius: float) -> tuple[np.ndarray, np.ndarray]:
    x, y, radius = (_number(well, key) for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    gx, gy = xx + left, yy + top
    distance2 = (gx - x) ** 2 + (gy - y) ** 2
    interior = distance2 <= (INTERIOR_RADIUS_FRACTION * radius) ** 2
    background = ((distance2 >= (BACKGROUND_INNER_RADIUS_FRACTION * radius) ** 2)
                  & (distance2 <= (BACKGROUND_OUTER_RADIUS_FRACTION * radius) ** 2))
    nearby = tree.query_ball_point(
        [x, y], BACKGROUND_OUTER_RADIUS_FRACTION * radius
        + NEIGHBOUR_EXCLUSION_RADIUS_FRACTION * max_radius)
    well_id = _normalise_well_id(well['well_id'])
    for index in nearby:
        other = all_wells[index]
        if _normalise_well_id(other['well_id']) == well_id:
            continue
        ox, oy, other_radius = (_number(other, key)
                                for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'))
        background &= ((gx - ox) ** 2 + (gy - oy) ** 2
                       > (NEIGHBOUR_EXCLUSION_RADIUS_FRACTION * other_radius) ** 2)
    return interior, background


def _save_mask(path: Path, labels: np.ndarray, left: int, top: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, labels=np.asarray(labels, dtype=np.int32),
                        left=np.int64(left), top=np.int64(top))
    os.replace(tmp, path)


def segment_validation_well(condition_id: str, well: dict, psc_row: dict,
                            all_wells: list[dict], tree: cKDTree, tile: np.ndarray,
                            left: int, top: int, pixel_size_x_um: float,
                            pixel_size_y_um: float, source_dtype,
                            *, threshold_k: float = THRESHOLD_K,
                            gaussian_sigma_px: float = DETECTION_GAUSSIAN_SIGMA_PX,
                            min_diameter_um: float = MIN_EQUIVALENT_DIAMETER_UM,
                            unresolved_diameter_um: float = UNRESOLVED_EQUIVALENT_DIAMETER_UM,
                            max_radius: float | None = None) -> tuple[list[dict], dict, np.ndarray]:
    well_id = _normalise_well_id(well['well_id'])
    base = {
        'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
        'dose': CONDITIONS[condition_id]['dose'], 'well_id': well_id,
        'x_px_fullres': _number(well, 'x_px_fullres'),
        'y_px_fullres': _number(well, 'y_px_fullres'), 'radius_px': _number(well, 'radius_px'),
        'PDO_present': _truthy(well.get('PDO_present')), 'PDO_count': int(_number(well, 'PDO_count')),
        'total_PDO_projected_area_um2': _number(well, 'total_PDO_projected_area_um2'),
        'RFP_channel': CHANNELS['rfp'], 'background_qc': psc_row.get('background_qc', ''),
        'RFP_background_corrected_mean': _finite(psc_row, 'RFP_background_corrected_mean'),
        'mask_left_px_fullres': left, 'mask_top_px_fullres': top,
    }
    empty_labels = np.zeros(tile.shape, dtype=np.int32)
    if psc_row.get('background_qc') != 'valid_local_background':
        return [], {**base, 'background_robust_sigma_RFP': float('nan'),
                    'threshold_corrected_RFP': float('nan'), 'threshold_detector_RFP': float('nan'),
                    'PSC_like_resolved_object_count': float('nan'),
                    'unresolved_PSC_like_cluster_count': float('nan'),
                    'rejected_small_component_count': float('nan'),
                    'PSC_like_total_area_px2': float('nan'), 'PSC_like_total_area_um2': float('nan'),
                    'PSC_like_median_object_area_um2': float('nan'),
                    'PSC_like_median_object_diameter_um': float('nan'),
                    'unresolved_cluster_total_area_um2': float('nan'),
                    'PSC_segmentation_status': 'insufficient_local_background'}, empty_labels

    actual_max_radius = (max_radius if max_radius is not None else
                         max(_number(row, 'radius_px') for row in all_wells))
    interior, background_mask = _masks(tile.shape, left, top, well, all_wells, tree,
                                       float(actual_max_radius))
    background = np.asarray(tile[background_mask], dtype=np.float64)
    if not background.size:
        return [], {**base, 'background_robust_sigma_RFP': float('nan'),
                    'threshold_corrected_RFP': float('nan'), 'threshold_detector_RFP': float('nan'),
                    'PSC_like_resolved_object_count': float('nan'),
                    'unresolved_PSC_like_cluster_count': float('nan'),
                    'rejected_small_component_count': float('nan'),
                    'PSC_like_total_area_px2': float('nan'), 'PSC_like_total_area_um2': float('nan'),
                    'PSC_like_median_object_area_um2': float('nan'),
                    'PSC_like_median_object_diameter_um': float('nan'),
                    'unresolved_cluster_total_area_um2': float('nan'),
                    'PSC_segmentation_status': 'insufficient_local_background'}, empty_labels
    background_median = float(psc_row['RFP_background_median'])
    background_p99 = float(psc_row['RFP_background_p99'])
    mad = float(np.median(np.abs(background - np.median(background))))
    robust_sigma = 1.4826 * mad
    threshold_corrected = max(background_p99 - background_median,
                              float(threshold_k) * robust_sigma)
    threshold_detector = background_median + threshold_corrected
    corrected = np.asarray(tile, dtype=np.float64) - background_median
    detection = gaussian_filter(corrected, sigma=float(gaussian_sigma_px), mode='nearest')
    binary = interior & (detection > threshold_corrected)
    component_labels, count = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    pixel_area_um2 = float(pixel_size_x_um) * float(pixel_size_y_um)
    min_area_px = math.pi * (float(min_diameter_um) / 2.0) ** 2 / pixel_area_um2
    unresolved_area_px = math.pi * (float(unresolved_diameter_um) / 2.0) ** 2 / pixel_area_um2

    candidates = []
    for component in range(1, int(count) + 1):
        mask = component_labels == component
        area = int(np.count_nonzero(mask))
        if not area:
            continue
        ys, xs = np.nonzero(mask)
        candidates.append((float(np.mean(ys) + top), float(np.mean(xs) + left), component, mask,
                           area, 'rejected_small' if area < min_area_px else
                           'unresolved_cluster' if area > unresolved_area_px else 'resolved'))
    candidates.sort(key=lambda item: (item[0], item[1]))
    output_labels = np.zeros(tile.shape, dtype=np.int32)
    objects = []
    resolved_areas, unresolved_areas = [], []
    rejected = 0
    interior_boundary = interior & ~np.asarray(
        gaussian_filter(interior.astype(np.float32), 1, mode='constant') > 0.999,
        dtype=bool)
    accepted_number = 0
    for _, _, component, mask, area, status in candidates:
        if status == 'rejected_small':
            rejected += 1
            continue
        accepted_number += 1
        output_labels[mask] = accepted_number
        raw = np.asarray(tile[mask], dtype=np.float64)
        ys, xs = np.nonzero(mask)
        area_um2 = area * pixel_area_um2
        diameter_um = 2.0 * math.sqrt(area_um2 / math.pi)
        if status == 'resolved':
            resolved_areas.append((area, area_um2, diameter_um))
        else:
            unresolved_areas.append(area_um2)
        peak_excess = float(np.max(raw) - threshold_detector)
        peak_sigma = (float((np.max(raw) - background_median) / robust_sigma)
                      if robust_sigma > 0 else float('inf'))
        saturated_fraction = 0.0
        if np.issubdtype(np.dtype(source_dtype), np.integer):
            saturated_fraction = float(np.count_nonzero(raw >= np.iinfo(source_dtype).max) / area)
        objects.append({
            'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
            'dose': CONDITIONS[condition_id]['dose'], 'well_id': well_id,
            'object_id': f'{condition_id}__W{well_id}__PSCLIKE{accepted_number:03d}',
            'object_number_in_well': accepted_number, 'object_status': status,
            'centroid_x_px_fullres': float(np.mean(xs) + left),
            'centroid_y_px_fullres': float(np.mean(ys) + top), 'area_px2': area,
            'area_um2': area_um2, 'equivalent_diameter_um': diameter_um,
            'mean_RFP_intensity': float(np.mean(raw)), 'median_RFP_intensity': float(np.median(raw)),
            'max_RFP_intensity': float(np.max(raw)),
            'integrated_RFP_intensity': float(np.sum(raw, dtype=np.float64)),
            'background_corrected_mean_RFP': float(np.mean(raw) - background_median),
            'background_median_RFP': background_median,
            'background_robust_sigma_RFP': robust_sigma,
            'threshold_corrected_RFP': threshold_corrected,
            'threshold_detector_RFP': threshold_detector,
            'peak_excess_over_threshold_RFP': peak_excess, 'peak_excess_sigma': peak_sigma,
            'touches_well_interior_boundary': bool(np.any(mask & interior_boundary)),
            'saturated_pixel_fraction': saturated_fraction, 'mask_label': accepted_number,
            'segmentation_qc_status': ('unresolved_large_component' if status == 'unresolved_cluster'
                                       else 'provisional_resolved_object'),
        })
    resolved_count = len(resolved_areas); unresolved_count = len(unresolved_areas)
    status = 'unresolved_cluster_present' if unresolved_count else 'completed_validation_provisional'
    summary = {
        **base, 'background_robust_sigma_RFP': robust_sigma,
        'threshold_corrected_RFP': threshold_corrected, 'threshold_detector_RFP': threshold_detector,
        'PSC_like_resolved_object_count': resolved_count,
        'unresolved_PSC_like_cluster_count': unresolved_count,
        'rejected_small_component_count': rejected,
        'PSC_like_total_area_px2': sum(item[0] for item in resolved_areas),
        'PSC_like_total_area_um2': sum(item[1] for item in resolved_areas),
        'PSC_like_median_object_area_um2': (float(np.median([item[1] for item in resolved_areas]))
                                            if resolved_areas else float('nan')),
        'PSC_like_median_object_diameter_um': (float(np.median([item[2] for item in resolved_areas]))
                                                if resolved_areas else float('nan')),
        'unresolved_cluster_total_area_um2': sum(unresolved_areas),
        'PSC_segmentation_status': status,
    }
    return objects, summary, output_labels


def _final_inputs(folder: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    condition_summary, wells, pdos = _condition_inputs(folder)
    psc_summary_path = folder / 'psc_quantification' / 'psc_summary.json'
    psc_path = folder / 'psc_quantification' / 'psc_well_measurements.csv'
    for path in (psc_summary_path, psc_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required completed PSC quantification output is missing: {path}')
    psc_summary = _read_json(psc_summary_path)
    if psc_summary.get('completion_status') != 'completed' or psc_summary.get('well_set_qc_passed') is not True:
        raise RuntimeError(f'PSC quantification is not completed with well-set QC: {psc_summary_path}')
    return condition_summary, wells, pdos, _read_csv(psc_path)


def quantify_condition(condition_id: str, folder: Path, args: argparse.Namespace, batch_status: dict,
                       *, probe: Callable = probe_omezarr,
                       open_group: Callable = zarr.open_group, s3_client=None) -> dict:
    output = folder / 'psc_object_quantification'
    logger = RunLogger(condition_id, output / 'validation_segmentation.log')
    condition_summary, wells, _, psc_rows = _final_inputs(folder)
    sample, sampling_qc = select_validation_sample(
        condition_id, wells, psc_rows, qc_per_category=args.qc_per_category)
    if sampling_qc['primary_count'] != CORE_SAMPLE_SIZE:
        raise RuntimeError(f'Primary sample count is not exactly {CORE_SAMPLE_SIZE}.')
    zarr_path = resolve_omezarr(condition_id, condition_summary, batch_status, args.cache_root)
    meta = probe(zarr_path)
    validation = validate_omezarr(meta, condition_summary, args.expected_pixel_size_um)
    root = open_group(str(zarr_path), mode='r')
    array = root[meta['level0_array_path']]
    planes = SingletonTZCYX(array, meta['axes'])
    channels, height, width = planes.shape_cyx
    if channels != 3:
        raise RuntimeError(f'Validated metadata and opened array disagree: {channels} channels.')
    well_by_id = {_normalise_well_id(row['well_id']): row for row in wells}
    psc_by_id = {_normalise_well_id(row['well_id']): row for row in psc_rows}
    points = np.asarray([[_number(row, 'x_px_fullres'), _number(row, 'y_px_fullres')]
                         for row in wells], dtype=float)
    tree = cKDTree(points)
    max_radius = max(_number(row, 'radius_px') for row in wells)
    px_x, px_y = validation['pixel_size_um']['x'], validation['pixel_size_um']['y']
    object_rows, well_rows = [], []
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output / 'validation_sample_manifest.csv', sample, SAMPLE_FIELDS)
    for index, sample_row in enumerate(sample, 1):
        well_id = sample_row['well_id']; well = well_by_id[well_id]; psc_row = psc_by_id[well_id]
        tile, left, top = _read_well_region(planes, CHANNELS['rfp'], well, width, height)
        objects, summary, labels = segment_validation_well(
            condition_id, well, psc_row, wells, tree, tile, left, top, px_x, px_y, array.dtype,
            threshold_k=args.threshold_k, gaussian_sigma_px=args.detection_gaussian_sigma_px,
            min_diameter_um=args.minimum_equivalent_diameter_um,
            unresolved_diameter_um=args.unresolved_equivalent_diameter_um,
            max_radius=max_radius)
        mask_path = output / 'segmentation_masks' / f'well_{well_id}.npz'
        _save_mask(mask_path, labels, left, top)
        summary.update({'sample_type': sample_row['sample_type'],
                        'sample_reasons': sample_row['sample_reasons'], 'mask_path': str(mask_path)})
        if summary['unresolved_PSC_like_cluster_count'] not in (0, 0.0) and math.isfinite(
                float(summary['unresolved_PSC_like_cluster_count'])):
            reasons = set(filter(None, summary['sample_reasons'].split(';')))
            reasons.add('very_large_candidate_component')
            summary['sample_reasons'] = ';'.join(sorted(reasons))
        object_rows.extend(objects); well_rows.append(summary)
        _atomic_csv(output / 'psc_object_measurements.csv', object_rows, OBJECT_FIELDS)
        _atomic_csv(output / 'psc_well_object_summary.csv', well_rows, WELL_FIELDS)
        logger.event('segment_validation', 'validation_well_completed', wells_processed=index,
                     wells_total=len(sample), well_id=well_id,
                     resolved_objects=summary['PSC_like_resolved_object_count'],
                     unresolved_clusters=summary['unresolved_PSC_like_cluster_count'])

    selected_ids = {row['well_id'] for row in sample}
    measured_ids = {row['well_id'] for row in well_rows}
    result = {
        'completion_status': 'validation_sample_completed',
        'segmentation_version': SEGMENTATION_VERSION, 'condition_id': condition_id,
        'condition_name': CONDITIONS[condition_id]['condition_name'],
        'dose': CONDITIONS[condition_id]['dose'], 'completed_at': _now(),
        'validation_only': True, 'full_well_processing_available': False,
        'final_crop_regeneration_available': False,
        'selected_validation_wells': len(selected_ids),
        'primary_validation_wells': sampling_qc['primary_count'],
        'supplemental_QC_wells': sampling_qc['supplement_count'],
        'validation_well_set_qc_passed': selected_ids == measured_ids,
        'empty_core_factor_cells': sampling_qc['empty_core_factor_cells'],
        'resolved_objects': sum(int(row['PSC_like_resolved_object_count']) for row in well_rows
                                if math.isfinite(float(row['PSC_like_resolved_object_count']))),
        'unresolved_clusters': sum(int(row['unresolved_PSC_like_cluster_count']) for row in well_rows
                                   if math.isfinite(float(row['unresolved_PSC_like_cluster_count']))),
        'segmentation_parameters': {
            'rfp_channel': 1, 'interior_radius_fraction': INTERIOR_RADIUS_FRACTION,
            'background_inner_radius_fraction': BACKGROUND_INNER_RADIUS_FRACTION,
            'background_outer_radius_fraction': BACKGROUND_OUTER_RADIUS_FRACTION,
            'neighbour_exclusion_radius_fraction': NEIGHBOUR_EXCLUSION_RADIUS_FRACTION,
            'robust_sigma': '1.4826 * median absolute deviation of valid local background pixels',
            'threshold_rule': 'max(background_p99 - background_median, k * robust_sigma)',
            'threshold_k': args.threshold_k,
            'detection_gaussian_sigma_px': args.detection_gaussian_sigma_px,
            'measurements_source': 'unsmoothed original OME-Zarr RFP channel 1 detector values',
            'minimum_equivalent_diameter_um': args.minimum_equivalent_diameter_um,
            'unresolved_above_equivalent_diameter_um': args.unresolved_equivalent_diameter_um,
            'touching_object_splitting': False,
        },
        'scientific_status': ('PSC-like resolved objects and unresolved clusters are provisional '
                              'validation outputs, not true PSC cell counts.'),
        'omezarr_source': str(zarr_path), 'omezarr_validation': validation,
    }
    if not result['validation_well_set_qc_passed']:
        raise RuntimeError('Validation sample output well IDs do not match the selected IDs.')
    _atomic_json(output / 'segmentation_summary.json', result)
    if args.upload_s3:
        if s3_client is None:
            from nd2_s3_stage import get_s3_client
            s3_client = get_s3_client(region_name=args.region)
        prefix = '/'.join(value.strip('/') for value in
                          (args.results_s3_prefix, condition_id, 'psc_object_quantification')
                          if value.strip('/'))
        result['s3_upload'] = _upload_additive(s3_client, output, args.bucket, prefix)
        _atomic_json(output / 'segmentation_summary.json', result)
    return result


def combine_outputs(result_root: Path) -> None:
    specs = [
        ('psc_object_measurements.csv', 'all_conditions_psc_object_measurements.csv', OBJECT_FIELDS),
        ('psc_well_object_summary.csv', 'all_conditions_psc_well_object_summary.csv', WELL_FIELDS),
        ('validation_sample_manifest.csv', 'all_conditions_psc_validation_sample_manifest.csv', SAMPLE_FIELDS),
    ]
    for local_name, combined_name, fields in specs:
        rows = []
        for condition_id in CONDITIONS:
            folder = result_root / condition_id / 'psc_object_quantification'
            summary_path = folder / 'segmentation_summary.json'
            try:
                if _read_json(summary_path).get('completion_status') != 'validation_sample_completed':
                    continue
            except Exception:
                continue
            path = folder / local_name
            if path.is_file():
                rows.extend(_read_csv(path))
        _atomic_csv(result_root / combined_name, rows, fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='VALIDATION ONLY: provisional PSC-like object segmentation on sampled wells.')
    parser.add_argument('--validation-only', action='store_true', required=True,
                        help='Required safety acknowledgement; no full-well mode exists.')
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--condition-id', action='append', choices=tuple(CONDITIONS), default=[])
    parser.add_argument('--expected-pixel-size-um', type=float, default=EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--threshold-k', type=float, default=THRESHOLD_K)
    parser.add_argument('--detection-gaussian-sigma-px', type=float,
                        default=DETECTION_GAUSSIAN_SIGMA_PX)
    parser.add_argument('--minimum-equivalent-diameter-um', type=float,
                        default=MIN_EQUIVALENT_DIAMETER_UM)
    parser.add_argument('--unresolved-equivalent-diameter-um', type=float,
                        default=UNRESOLVED_EQUIVALENT_DIAMETER_UM)
    parser.add_argument('--qc-per-category', type=int, default=2)
    parser.add_argument('--upload-s3', action='store_true')
    parser.add_argument('--bucket', default='')
    parser.add_argument('--results-s3-prefix', default='')
    parser.add_argument('--region', default='eu-west-2')
    return parser


def run(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
        open_group: Callable = zarr.open_group, s3_client=None) -> int:
    if not args.validation_only:
        raise RuntimeError('--validation-only is mandatory; full-well processing is blocked.')
    if args.threshold_k <= 0 or args.detection_gaussian_sigma_px < 0:
        raise ValueError('Threshold k must be positive and Gaussian sigma cannot be negative.')
    if args.minimum_equivalent_diameter_um <= 0:
        raise ValueError('Minimum equivalent diameter must be positive.')
    if args.unresolved_equivalent_diameter_um <= args.minimum_equivalent_diameter_um:
        raise ValueError('Unresolved diameter must exceed the minimum diameter.')
    if args.qc_per_category < 0:
        raise ValueError('--qc-per-category cannot be negative.')
    if args.upload_s3 and (not args.bucket or not args.results_s3_prefix):
        raise ValueError('--upload-s3 requires --bucket and --results-s3-prefix.')
    result_root = args.result_root.expanduser().resolve()
    status_path = result_root / 'batch_status.json'
    batch_status = _read_json(status_path) if status_path.is_file() else {}
    failures = 0
    selected = args.condition_id or list(CONDITIONS)
    for condition_id in selected:
        output = result_root / condition_id / 'psc_object_quantification'
        try:
            summary = quantify_condition(condition_id, result_root / condition_id, args,
                                         batch_status, probe=probe, open_group=open_group,
                                         s3_client=s3_client)
            print(f"{condition_id}: validation sample completed "
                  f"({summary['primary_validation_wells']} primary + "
                  f"{summary['supplemental_QC_wells']} QC supplements)", flush=True)
        except Exception as exc:
            failures += 1
            tb = traceback.format_exc()
            _atomic_json(output / 'segmentation_summary.json', {
                'completion_status': 'failed', 'validation_only': True,
                'full_well_processing_available': False,
                'final_crop_regeneration_available': False,
                'condition_id': condition_id, 'failed_at': _now(),
                'error': f'{type(exc).__name__}: {exc}', 'traceback': tb,
            })
        finally:
            try:
                combine_outputs(result_root)
            except Exception as exc:
                failures += 1
                print(f'{condition_id}: combined output failure: {type(exc).__name__}: {exc}', flush=True)
    if not failures:
        print(f'PSC-like validation segmentation completed: {len(selected)}/{len(selected)} conditions; '
              'full-well processing remains blocked.', flush=True)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
