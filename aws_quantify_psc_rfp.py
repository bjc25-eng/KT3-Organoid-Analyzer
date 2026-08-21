from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import zarr
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from aws_export_pdo_positive_crops import (
    CONDITIONS,
    EXPECTED_PIXEL_SIZE_UM,
    _upload_additive,
    resolve_omezarr,
    validate_omezarr,
)
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


QUANTIFICATION_VERSION = 1
RFP_CHANNEL = 1
INTERIOR_RADIUS_FRACTION = 0.86
BACKGROUND_INNER_RADIUS_FRACTION = 1.15
BACKGROUND_OUTER_RADIUS_FRACTION = 1.45
NEIGHBOUR_EXCLUSION_RADIUS_FRACTION = 1.05
MIN_BACKGROUND_VALID_PIXELS = 512
MIN_BACKGROUND_VALID_FRACTION = 0.10
SPATIAL_BIN_MM = 5.0

PSC_FIELDS = (
    'condition_id', 'condition_name', 'dose_nM', 'well_id',
    'x_px_fullres', 'y_px_fullres', 'radius_px', 'x_mm', 'y_mm',
    'RFP_channel', 'RFP_source_dtype',
    'interior_radius_fraction', 'interior_radius_px', 'interior_pixel_count',
    'background_inner_radius_fraction', 'background_outer_radius_fraction',
    'neighbour_exclusion_radius_fraction', 'background_valid_pixel_count',
    'background_expected_pixel_count', 'background_valid_fraction', 'background_qc',
    'RFP_mean_intensity', 'RFP_median_intensity', 'RFP_max_intensity',
    'RFP_integrated_intensity', 'RFP_p90', 'RFP_p95', 'RFP_p99',
    'RFP_saturated_pixel_count', 'RFP_saturated_pixel_fraction',
    'RFP_background_mean', 'RFP_background_median', 'RFP_background_p95',
    'RFP_background_p99', 'RFP_background_corrected_mean',
    'RFP_background_corrected_integrated_intensity',
    'RFP_positive_only_excess_integrated_intensity',
    'exploratory_RFP_threshold_intensity', 'exploratory_RFP_positive_area_px2',
    'exploratory_RFP_positive_area_um2', 'exploratory_RFP_positive_fraction',
    'quantification_status', 'error',
)

SPATIAL_FIELDS = (
    'condition_id', 'condition_name', 'dose_nM', 'y_bin_index',
    'y_bin_start_mm', 'y_bin_end_mm', 'accepted_well_count',
    'wells_with_valid_background', 'wells_with_insufficient_background',
    'median_background_valid_fraction', 'PDO_positive_well_count',
    'PDO_positive_fraction', 'median_total_PDO_projected_area_um2',
    'median_PDO_equivalent_diameter_um', 'median_RFP_mean_intensity',
    'median_RFP_integrated_intensity', 'median_RFP_background_corrected_mean',
    'median_RFP_background_corrected_integrated_intensity', 'median_RFP_p95',
    'median_exploratory_RFP_positive_area_um2',
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _fields(rows: Iterable[dict], required: tuple[str, ...] = ()) -> list[str]:
    result, seen = list(required), set(required)
    for row in rows:
        for key in row:
            if key not in seen:
                result.append(key); seen.add(key)
    return result


def _atomic_csv(path: Path, rows: list[dict], required: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _fields(rows, required)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        if fields:
            writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)


def _normalise_well_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    if not text:
        raise RuntimeError('Encountered an empty final well_id.')
    return text


def _number(row: dict, key: str) -> float:
    try:
        result = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid or missing numeric field '{key}' in row {row}.") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite numeric field '{key}' in row {row}.")
    return result


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def _nan() -> float:
    return float('nan')


def _nanmedian(values: Iterable[object]) -> float:
    parsed = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parsed.append(number)
    return float(np.median(parsed)) if parsed else _nan()


def _signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rss_mib() -> float | None:
    try:
        resident = int(Path('/proc/self/statm').read_text(encoding='ascii').split()[1])
        return resident * int(os.sysconf('SC_PAGE_SIZE')) / 1024 ** 2
    except Exception:
        try:
            import psutil  # type: ignore
            return float(psutil.Process().memory_info().rss) / 1024 ** 2
        except Exception:
            return None


class RunLogger:
    def __init__(self, condition_id: str, path: Path):
        self.condition_id = condition_id
        self.path = path
        self.started = time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, phase: str, message: str, *, wells_processed: int = 0,
              wells_total: int = 0, **extra: object) -> None:
        rss = _rss_mib()
        row = {
            'timestamp': _now(), 'condition': self.condition_id, 'phase': phase,
            'wells_processed': int(wells_processed), 'wells_total': int(wells_total),
            'elapsed_seconds': round(time.monotonic() - self.started, 3),
            'pid': os.getpid(), 'rss_mib': None if rss is None else round(rss, 1),
            'message': message, **extra,
        }
        line = json.dumps(row, default=str)
        print(line, flush=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')


def _final_inputs(condition_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    paths = [condition_dir / 'condition_summary.json', condition_dir / 'well_measurements.csv',
             condition_dir / 'pdo_measurements.csv']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required completed analysis output is missing: {path}')
    summary = _read_json(paths[0])
    if summary.get('completion_status') != 'completed':
        raise RuntimeError(f'Condition is not marked completed in {paths[0]}.')
    wells, pdos = _read_csv(paths[1]), _read_csv(paths[2])
    seen = set()
    for row in wells:
        well_id = _normalise_well_id(row.get('well_id'))
        if well_id in seen:
            raise RuntimeError(f'Duplicate final accepted well_id: {well_id}.')
        seen.add(well_id)
        for key in ('x_px_fullres', 'y_px_fullres', 'radius_px'):
            _number(row, key)
    for row in pdos:
        well_id = _normalise_well_id(row.get('well_id'))
        if well_id not in seen:
            raise RuntimeError(f'Final PDO row references non-final well_id {well_id}.')
    return summary, wells, pdos


def _background_expected_count(radius: float, x_fraction: float = 0.0,
                               y_fraction: float = 0.0) -> int:
    outer = int(math.ceil(radius * BACKGROUND_OUTER_RADIUS_FRACTION))
    yy, xx = np.ogrid[-outer:outer + 1, -outer:outer + 1]
    xx = xx - float(x_fraction); yy = yy - float(y_fraction)
    distance2 = xx * xx + yy * yy
    return int(np.count_nonzero(
        (distance2 >= (BACKGROUND_INNER_RADIUS_FRACTION * radius) ** 2)
        & (distance2 <= (BACKGROUND_OUTER_RADIUS_FRACTION * radius) ** 2)
    ))


def quantify_well(tile: np.ndarray, tile_left: int, tile_top: int, well: dict,
                  all_wells: list[dict], neighbour_tree: cKDTree, pixel_size_um: float,
                  source_dtype, *, min_background_pixels: int = MIN_BACKGROUND_VALID_PIXELS,
                  min_background_fraction: float = MIN_BACKGROUND_VALID_FRACTION,
                  max_neighbour_radius: float | None = None) -> dict:
    well_id = _normalise_well_id(well['well_id'])
    x, y, radius = (_number(well, key) for key in
                    ('x_px_fullres', 'y_px_fullres', 'radius_px'))
    tile_cx, tile_cy = x - tile_left, y - tile_top
    outer = int(math.ceil(radius * BACKGROUND_OUTER_RADIUS_FRACTION)) + 1
    sx0 = max(0, int(math.floor(tile_cx)) - outer)
    sx1 = min(tile.shape[1], int(math.floor(tile_cx)) + outer + 1)
    sy0 = max(0, int(math.floor(tile_cy)) - outer)
    sy1 = min(tile.shape[0], int(math.floor(tile_cy)) + outer + 1)
    local = np.asarray(tile[sy0:sy1, sx0:sx1])
    local_left, local_top = tile_left + sx0, tile_top + sy0
    cx, cy = x - local_left, y - local_top
    yy, xx = np.ogrid[:local.shape[0], :local.shape[1]]
    distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
    interior_radius = INTERIOR_RADIUS_FRACTION * radius
    interior_mask = distance2 <= interior_radius ** 2
    background_mask = (
        (distance2 >= (BACKGROUND_INNER_RADIUS_FRACTION * radius) ** 2)
        & (distance2 <= (BACKGROUND_OUTER_RADIUS_FRACTION * radius) ** 2)
    )
    max_radius = (float(max_neighbour_radius) if max_neighbour_radius is not None else
                  max(_number(row, 'radius_px') for row in all_wells) if all_wells else radius)
    nearby = neighbour_tree.query_ball_point(
        [x, y], BACKGROUND_OUTER_RADIUS_FRACTION * radius
        + NEIGHBOUR_EXCLUSION_RADIUS_FRACTION * max_radius)
    for index in nearby:
        other = all_wells[index]
        other_id = _normalise_well_id(other['well_id'])
        if other_id == well_id:
            continue
        ox, oy, other_radius = (_number(other, key) for key in
                                ('x_px_fullres', 'y_px_fullres', 'radius_px'))
        exclusion2 = (xx - (ox - local_left)) ** 2 + (yy - (oy - local_top)) ** 2
        background_mask &= exclusion2 > (NEIGHBOUR_EXCLUSION_RADIUS_FRACTION * other_radius) ** 2

    interior = np.asarray(local[interior_mask], dtype=np.float64)
    if not interior.size:
        raise RuntimeError(f'Final well {well_id} has no interior pixels in its assigned tile read.')
    background = np.asarray(local[background_mask], dtype=np.float64)
    expected = _background_expected_count(radius, x - math.floor(x), y - math.floor(y))
    valid_count = int(background.size)
    valid_fraction = float(valid_count / expected) if expected else 0.0
    background_ok = (valid_count >= int(min_background_pixels)
                     and valid_fraction >= float(min_background_fraction))

    mean = float(np.mean(interior))
    median = float(np.median(interior))
    maximum = float(np.max(interior))
    integrated = float(np.sum(interior, dtype=np.float64))
    p90, p95, p99 = (float(v) for v in np.percentile(interior, (90, 95, 99)))
    saturated_count = 0
    if np.issubdtype(np.dtype(source_dtype), np.integer):
        saturated_count = int(np.count_nonzero(interior >= np.iinfo(source_dtype).max))

    result = {
        'well_id': well_id, 'x_px_fullres': x, 'y_px_fullres': y, 'radius_px': radius,
        'x_mm': x * pixel_size_um / 1000.0, 'y_mm': y * pixel_size_um / 1000.0,
        'RFP_channel': RFP_CHANNEL, 'RFP_source_dtype': str(np.dtype(source_dtype)),
        'interior_radius_fraction': INTERIOR_RADIUS_FRACTION,
        'interior_radius_px': interior_radius, 'interior_pixel_count': int(interior.size),
        'background_inner_radius_fraction': BACKGROUND_INNER_RADIUS_FRACTION,
        'background_outer_radius_fraction': BACKGROUND_OUTER_RADIUS_FRACTION,
        'neighbour_exclusion_radius_fraction': NEIGHBOUR_EXCLUSION_RADIUS_FRACTION,
        'background_valid_pixel_count': valid_count,
        'background_expected_pixel_count': expected,
        'background_valid_fraction': valid_fraction,
        'background_qc': 'valid_local_background' if background_ok else 'insufficient_local_background',
        'RFP_mean_intensity': mean, 'RFP_median_intensity': median,
        'RFP_max_intensity': maximum, 'RFP_integrated_intensity': integrated,
        'RFP_p90': p90, 'RFP_p95': p95, 'RFP_p99': p99,
        'RFP_saturated_pixel_count': saturated_count,
        'RFP_saturated_pixel_fraction': float(saturated_count / interior.size),
        'quantification_status': 'completed', 'error': '',
    }
    background_fields = {
        'RFP_background_mean': _nan(), 'RFP_background_median': _nan(),
        'RFP_background_p95': _nan(), 'RFP_background_p99': _nan(),
        'RFP_background_corrected_mean': _nan(),
        'RFP_background_corrected_integrated_intensity': _nan(),
        'RFP_positive_only_excess_integrated_intensity': _nan(),
        'exploratory_RFP_threshold_intensity': _nan(),
        'exploratory_RFP_positive_area_px2': _nan(),
        'exploratory_RFP_positive_area_um2': _nan(),
        'exploratory_RFP_positive_fraction': _nan(),
    }
    if background_ok:
        bg_mean = float(np.mean(background)); bg_median = float(np.median(background))
        bg_p95, bg_p99 = (float(v) for v in np.percentile(background, (95, 99)))
        corrected_mean = mean - bg_median
        corrected_integrated = float(np.sum(interior - bg_median, dtype=np.float64))
        positive_excess = float(np.sum(np.maximum(interior - bg_median, 0.0), dtype=np.float64))
        positive_count = int(np.count_nonzero(interior > bg_p99))
        background_fields.update({
            'RFP_background_mean': bg_mean, 'RFP_background_median': bg_median,
            'RFP_background_p95': bg_p95, 'RFP_background_p99': bg_p99,
            'RFP_background_corrected_mean': corrected_mean,
            'RFP_background_corrected_integrated_intensity': corrected_integrated,
            'RFP_positive_only_excess_integrated_intensity': positive_excess,
            'exploratory_RFP_threshold_intensity': bg_p99,
            'exploratory_RFP_positive_area_px2': positive_count,
            'exploratory_RFP_positive_area_um2': positive_count * pixel_size_um ** 2,
            'exploratory_RFP_positive_fraction': float(positive_count / interior.size),
        })
    result.update(background_fields)
    return result


def _tile_groups(wells: list[dict], tile_size: int) -> list[list[dict]]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in wells:
        key = (int(_number(row, 'x_px_fullres')) // tile_size,
               int(_number(row, 'y_px_fullres')) // tile_size)
        groups.setdefault(key, []).append(row)
    return [groups[key] for key in sorted(groups, key=lambda value: (value[1], value[0]))]


def _read_group_tile(planes: SingletonTZCYX, group: list[dict], width: int,
                     height: int) -> tuple[np.ndarray, int, int]:
    margins = [int(math.ceil(_number(row, 'radius_px') * BACKGROUND_OUTER_RADIUS_FRACTION)) + 2
               for row in group]
    left = max(0, min(int(math.floor(_number(row, 'x_px_fullres'))) - margin
                      for row, margin in zip(group, margins)))
    right = min(width, max(int(math.ceil(_number(row, 'x_px_fullres'))) + margin + 1
                           for row, margin in zip(group, margins)))
    top = max(0, min(int(math.floor(_number(row, 'y_px_fullres'))) - margin
                     for row, margin in zip(group, margins)))
    bottom = min(height, max(int(math.ceil(_number(row, 'y_px_fullres'))) + margin + 1
                            for row, margin in zip(group, margins)))
    return planes.read(RFP_CHANNEL, slice(top, bottom), slice(left, right)), left, top


def _pdo_aggregates(pdos: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in pdos:
        grouped.setdefault(_normalise_well_id(row['well_id']), []).append(row)
    result = {}
    for well_id, rows in grouped.items():
        diameters = [_number(row, 'equivalent_circular_diameter_um') for row in rows]
        result[well_id] = {
            'PDO_equivalent_diameter_count': len(diameters),
            'PDO_equivalent_diameter_mean_um': float(np.mean(diameters)),
            'PDO_equivalent_diameter_median_um': float(np.median(diameters)),
            'PDO_equivalent_diameter_min_um': float(np.min(diameters)),
            'PDO_equivalent_diameter_max_um': float(np.max(diameters)),
            'PDO_equivalent_diameters_um': ';'.join(f'{value:.12g}' for value in diameters),
        }
    return result


def integrate_pdo_psc(condition_id: str, condition_name: str, dose_nm: float,
                      wells: list[dict], pdos: list[dict], psc_rows: list[dict]) -> list[dict]:
    psc_by_id = {_normalise_well_id(row['well_id']): row for row in psc_rows}
    final_ids = {_normalise_well_id(row['well_id']) for row in wells}
    if set(psc_by_id) != final_ids:
        raise RuntimeError(
            f'PSC/final well-set mismatch: missing={sorted(final_ids - set(psc_by_id))}, '
            f'extra={sorted(set(psc_by_id) - final_ids)}'
        )
    aggregates = _pdo_aggregates(pdos)
    result = []
    empty_pdo = {
        'PDO_equivalent_diameter_count': 0, 'PDO_equivalent_diameter_mean_um': _nan(),
        'PDO_equivalent_diameter_median_um': _nan(), 'PDO_equivalent_diameter_min_um': _nan(),
        'PDO_equivalent_diameter_max_um': _nan(), 'PDO_equivalent_diameters_um': '',
    }
    for well in wells:
        well_id = _normalise_well_id(well['well_id'])
        row = {
            'condition_id': condition_id, 'condition_name': condition_name, 'dose_nM': dose_nm,
            **well, **(aggregates.get(well_id) or empty_pdo), **psc_by_id[well_id],
        }
        row['well_id'] = well_id
        row['PDO_present'] = _truthy(well.get('PDO_present'))
        result.append(row)
    return result


def spatial_qc(integrated: list[dict], pdos: list[dict], bin_mm: float = SPATIAL_BIN_MM) -> list[dict]:
    if not integrated:
        return []
    pdo_diameters: dict[str, list[float]] = {}
    for row in pdos:
        pdo_diameters.setdefault(_normalise_well_id(row['well_id']), []).append(
            _number(row, 'equivalent_circular_diameter_um'))
    groups: dict[int, list[dict]] = {}
    for row in integrated:
        index = int(math.floor(float(row['y_mm']) / bin_mm))
        groups.setdefault(index, []).append(row)
    output = []
    for index in sorted(groups):
        rows = groups[index]
        positive = [row for row in rows if _truthy(row.get('PDO_present'))]
        valid = [row for row in rows if row.get('background_qc') == 'valid_local_background']
        diameters = [value for row in rows for value in pdo_diameters.get(str(row['well_id']), [])]
        first = rows[0]
        output.append({
            'condition_id': first['condition_id'], 'condition_name': first['condition_name'],
            'dose_nM': first['dose_nM'], 'y_bin_index': index,
            'y_bin_start_mm': index * bin_mm, 'y_bin_end_mm': (index + 1) * bin_mm,
            'accepted_well_count': len(rows), 'wells_with_valid_background': len(valid),
            'wells_with_insufficient_background': len(rows) - len(valid),
            'median_background_valid_fraction': _nanmedian(row['background_valid_fraction'] for row in rows),
            'PDO_positive_well_count': len(positive),
            'PDO_positive_fraction': float(len(positive) / len(rows)),
            'median_total_PDO_projected_area_um2': _nanmedian(
                row.get('total_PDO_projected_area_um2') for row in positive),
            'median_PDO_equivalent_diameter_um': _nanmedian(diameters),
            'median_RFP_mean_intensity': _nanmedian(row['RFP_mean_intensity'] for row in rows),
            'median_RFP_integrated_intensity': _nanmedian(row['RFP_integrated_intensity'] for row in rows),
            'median_RFP_background_corrected_mean': _nanmedian(
                row['RFP_background_corrected_mean'] for row in rows),
            'median_RFP_background_corrected_integrated_intensity': _nanmedian(
                row['RFP_background_corrected_integrated_intensity'] for row in rows),
            'median_RFP_p95': _nanmedian(row['RFP_p95'] for row in rows),
            'median_exploratory_RFP_positive_area_um2': _nanmedian(
                row['exploratory_RFP_positive_area_um2'] for row in rows),
        })
    return output


def _descriptive_relation(rows: list[dict], x_field: str, y_field: str) -> dict:
    pairs = []
    for row in rows:
        try:
            x, y = float(row[x_field]), float(row[y_field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 2:
        return {'n': len(pairs), 'spearman_rho': _nan(), 'slope': _nan()}
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        rho = _nan()
    else:
        rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    slope = _nan() if np.all(x == x[0]) else float(np.polyfit(x, y, 1)[0])
    return {'n': len(pairs), 'spearman_rho': rho, 'slope': slope,
            'p_value_calculated': False}


def _completion_valid(summary_path: Path, signature: str, final_ids: set[str]) -> bool:
    try:
        summary = _read_json(summary_path)
        rows = _read_csv(summary_path.parent / 'psc_well_measurements.csv')
        integrated = _read_csv(summary_path.parent / 'integrated_pdo_psc_well_measurements.csv')
        row_ids = {_normalise_well_id(row['well_id']) for row in rows}
        integrated_ids = {_normalise_well_id(row['well_id']) for row in integrated}
        return (summary.get('completion_status') == 'completed'
                and summary.get('analysis_signature') == signature
                and summary.get('well_set_qc_passed') is True
                and row_ids == final_ids and integrated_ids == final_ids
                and len(rows) == len(final_ids)
                and len(integrated) == len(final_ids)
                and (summary_path.parent / 'psc_spatial_QC_5mm.csv').is_file())
    except Exception:
        return False


def _available_acquisition_metadata(root, meta: dict, condition_summary: dict,
                                    batch_state: dict) -> dict:
    attrs = dict(root.attrs)
    omero = attrs.get('omero') or {}
    channels = omero.get('channels') or []
    return {
        'availability_notice': ('Only metadata present in the completed OME-Zarr and batch records is '
                                'reported; identical acquisition settings are not assumed.'),
        'rfp_omero_channel': channels[RFP_CHANNEL] if len(channels) > RFP_CHANNEL else None,
        'omero_rendering_defaults': omero.get('rdefs'),
        'omezarr_channel_metadata': meta.get('channel_metadata'),
        'source_object': condition_summary.get('source_object'),
        'batch_channel_mapping_evidence': batch_state.get('channel_mapping_evidence'),
        'batch_voxel_size_um': batch_state.get('voxel_size_um'),
    }


def export_condition(condition_id: str, condition_dir: Path, args: argparse.Namespace,
                     batch_status: dict, *, probe: Callable = probe_omezarr,
                     open_group: Callable = zarr.open_group, s3_client=None) -> dict:
    output = condition_dir / 'psc_quantification'
    logger = RunLogger(condition_id, output / 'psc_run.log')
    mapping = CONDITIONS[condition_id]
    condition_summary, wells, pdos = _final_inputs(condition_dir)
    final_ids = {_normalise_well_id(row['well_id']) for row in wells}
    condition_name = str(condition_summary.get('condition_name') or condition_id)
    dose_nm = float(str(mapping['dose']).split()[0])
    batch_state = (batch_status.get('conditions') or {}).get(condition_id) or {}
    zarr_path = resolve_omezarr(condition_id, condition_summary, batch_status, args.cache_root)
    meta = probe(zarr_path)
    validation = validate_omezarr(meta, condition_summary, args.expected_pixel_size_um)
    signature = _signature({
        'version': QUANTIFICATION_VERSION, 'omezarr': str(zarr_path),
        'omezarr_metadata': {'shape': meta.get('shape'), 'axes': meta.get('axes'),
                             'dtype': meta.get('dtype'), 'channel_metadata': meta.get('channel_metadata'),
                             'voxel_size_um': meta.get('voxel_size_um')},
        'source_object': condition_summary.get('source_object'),
        'final_wells': [{key: row.get(key) for key in
                         ('well_id', 'x_px_fullres', 'y_px_fullres', 'radius_px')} for row in wells],
        'settings': {'rfp_channel': RFP_CHANNEL,
                     'interior_radius_fraction': INTERIOR_RADIUS_FRACTION,
                     'background_inner_radius_fraction': BACKGROUND_INNER_RADIUS_FRACTION,
                     'background_outer_radius_fraction': BACKGROUND_OUTER_RADIUS_FRACTION,
                     'neighbour_exclusion_radius_fraction': NEIGHBOUR_EXCLUSION_RADIUS_FRACTION,
                     'minimum_background_valid_pixels': MIN_BACKGROUND_VALID_PIXELS,
                     'minimum_background_valid_fraction': MIN_BACKGROUND_VALID_FRACTION,
                     'exploratory_threshold': 'valid local-background p99',
                     'spatial_bin_mm': SPATIAL_BIN_MM, 'tile_size_px': args.tile},
    })
    summary_path = output / 'psc_summary.json'
    if _completion_valid(summary_path, signature, final_ids):
        result = _read_json(summary_path)
        result['skipped_existing'] = True
        logger.event('complete', 'completed_condition_reused', wells_processed=len(wells),
                     wells_total=len(wells))
        return result

    logger.event('validate', 'validated_final_inputs_and_omezarr', wells_total=len(wells),
                 omezarr=str(zarr_path), shape=validation['shape'])
    root = open_group(str(zarr_path), mode='r')
    array = root[meta['level0_array_path']]
    planes = SingletonTZCYX(array, meta['axes'])
    channels, height, width = planes.shape_cyx
    if channels != 3:
        raise RuntimeError(f'Opened OME-Zarr has {channels} channels after metadata validation.')
    pixel_size_um = (validation['pixel_size_um']['x'] + validation['pixel_size_um']['y']) / 2.0
    partial_path = output / '.psc_well_measurements.partial.csv'
    checkpoint_path = output / '.psc_checkpoint.json'
    prior_rows = []
    try:
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get('analysis_signature') == signature and partial_path.is_file():
            prior_rows = _read_csv(partial_path)
    except Exception:
        prior_rows = []
    completed_ids = {_normalise_well_id(row['well_id']) for row in prior_rows}
    if len(prior_rows) != len(completed_ids) or not completed_ids.issubset(final_ids):
        prior_rows, completed_ids = [], set()
    rows = list(prior_rows)
    remaining = [row for row in wells if _normalise_well_id(row['well_id']) not in completed_ids]
    all_centres = np.asarray([[_number(row, 'x_px_fullres'), _number(row, 'y_px_fullres')]
                              for row in wells], dtype=float)
    neighbour_tree = cKDTree(all_centres)
    max_neighbour_radius = max(_number(row, 'radius_px') for row in wells)
    logger.event('quantify', 'quantification_start', wells_processed=len(rows), wells_total=len(wells),
                 resumed_wells=len(rows), tile_size=args.tile)
    for group in _tile_groups(remaining, args.tile):
        tile, left, top = _read_group_tile(planes, group, width, height)
        group_rows = []
        for well in group:
            measured = quantify_well(tile, left, top, well, wells, neighbour_tree,
                                     pixel_size_um, array.dtype,
                                     max_neighbour_radius=max_neighbour_radius)
            measured.update(condition_id=condition_id, condition_name=condition_name,
                            dose_nM=dose_nm)
            group_rows.append(measured)
        rows.extend(group_rows)
        rows.sort(key=lambda row: (_number(row, 'y_px_fullres'), _number(row, 'x_px_fullres')))
        _atomic_csv(partial_path, rows, PSC_FIELDS)
        _atomic_json(checkpoint_path, {'analysis_signature': signature, 'updated_at': _now(),
                                      'completed_well_ids': [row['well_id'] for row in rows]})
        logger.event('quantify', 'tile_complete', wells_processed=len(rows), wells_total=len(wells))

    measured_ids = {_normalise_well_id(row['well_id']) for row in rows}
    if len(rows) != len(final_ids) or measured_ids != final_ids:
        raise RuntimeError(
            f'PSC well-set QC failed: final={len(final_ids)}, rows={len(rows)}, '
            f'missing={sorted(final_ids - measured_ids)}, extra={sorted(measured_ids - final_ids)}'
        )
    psc_path = output / 'psc_well_measurements.csv'
    _atomic_copy(partial_path, psc_path)
    integrated = integrate_pdo_psc(condition_id, condition_name, dose_nm, wells, pdos, rows)
    integrated_path = output / 'integrated_pdo_psc_well_measurements.csv'
    _atomic_csv(integrated_path, integrated, ('condition_id', 'condition_name', 'dose_nM', 'well_id'))
    spatial = spatial_qc(integrated, pdos)
    spatial_path = output / 'psc_spatial_QC_5mm.csv'
    _atomic_csv(spatial_path, spatial, SPATIAL_FIELDS)
    valid_background = [row for row in rows if row['background_qc'] == 'valid_local_background']
    positive = [row for row in integrated if _truthy(row.get('PDO_present'))]
    acquisition = _available_acquisition_metadata(root, meta, condition_summary, batch_state)
    result = {
        'completion_status': 'completed', 'quantification_version': QUANTIFICATION_VERSION,
        'analysis_signature': signature, 'condition_id': condition_id,
        'condition_name': condition_name, 'display_name': mapping['condition_name'],
        'dose_nM': dose_nm, 'completed_at': _now(), 'skipped_existing': False,
        'omezarr_source': str(zarr_path), 'omezarr_shape': validation['shape'],
        'omezarr_axes': validation['axes'], 'omezarr_dtype': str(array.dtype),
        'rfp_channel': RFP_CHANNEL, 'validated_channel_names': validation['channel_names'],
        'pixel_size_um': validation['pixel_size_um'],
        'available_acquisition_metadata': acquisition,
        'acquisition_comparability_notice': (
            'RFP values are detector intensity units, not absolute PSC abundance. Between-condition '
            'comparisons require consistent acquisition settings; consistency is not assumed when '
            'metadata are absent.'),
        'geometry': {
            'interior_radius_fraction': INTERIOR_RADIUS_FRACTION,
            'background_inner_radius_fraction': BACKGROUND_INNER_RADIUS_FRACTION,
            'background_outer_radius_fraction': BACKGROUND_OUTER_RADIUS_FRACTION,
            'neighbour_exclusion_radius_fraction': NEIGHBOUR_EXCLUSION_RADIUS_FRACTION,
        },
        'background_qc_criterion': {
            'minimum_valid_pixels': MIN_BACKGROUND_VALID_PIXELS,
            'minimum_valid_fraction': MIN_BACKGROUND_VALID_FRACTION,
            'failure_action': ('background-derived and exploratory metrics are NaN; raw interior '
                               'metrics remain available'),
        },
        'background_correction': ('Signed interior mean minus local-background median; integrated '
                                  'value is sum(interior pixel minus background median).'),
        'exploratory_area_rule': ('Interior pixels strictly greater than that well valid local-'
                                  'background p99; not PSC-positive area or a PSC classification.'),
        'psc_focus_analysis': 'omitted; existing heuristic detect_psc was not run',
        'total_final_wells': len(final_ids), 'wells_successfully_quantified': len(rows),
        'missing_well_ids': sorted(final_ids - measured_ids),
        'well_set_qc_passed': len(rows) == len(final_ids) and measured_ids == final_ids,
        'wells_with_valid_background': len(valid_background),
        'wells_with_insufficient_background': len(rows) - len(valid_background),
        'RFP_median_intensity': _nanmedian(row['RFP_median_intensity'] for row in rows),
        'RFP_background_median': _nanmedian(row['RFP_background_median'] for row in rows),
        'RFP_background_corrected_mean_median': _nanmedian(
            row['RFP_background_corrected_mean'] for row in rows),
        'PDO_positive_wells': len(positive),
        'PDO_positive_fraction': float(len(positive) / len(rows)) if rows else _nan(),
        'PDO_size_summary': {
            'median_total_projected_area_um2_in_positive_wells': _nanmedian(
                row.get('total_PDO_projected_area_um2') for row in positive),
            'median_equivalent_diameter_um': _nanmedian(
                row.get('PDO_equivalent_diameter_median_um') for row in positive),
        },
        'descriptive_spatial_RFP_trends': {
            'RFP_mean_intensity_vs_y_mm': _descriptive_relation(rows, 'y_mm', 'RFP_mean_intensity'),
            'RFP_background_corrected_mean_vs_y_mm': _descriptive_relation(
                rows, 'y_mm', 'RFP_background_corrected_mean'),
        },
        'descriptive_RFP_PDO_associations': {
            'RFP_background_corrected_mean_vs_total_PDO_area_um2': _descriptive_relation(
                integrated, 'RFP_background_corrected_mean', 'total_PDO_projected_area_um2'),
            'RFP_background_corrected_mean_vs_median_PDO_diameter_um': _descriptive_relation(
                integrated, 'RFP_background_corrected_mean', 'PDO_equivalent_diameter_median_um'),
        },
        'statistical_safeguard': ('Descriptive measurement/QC only. Wells are subsamples from one '
                                  'image/lane per concentration; no treatment p-values or '
                                  'significance claims were calculated.'),
        'performance': {'tile_size_px': args.tile, 'pid': os.getpid(), 'rss_mib': _rss_mib(),
                        'elapsed_seconds': round(time.monotonic() - logger.started, 3)},
        'outputs': [psc_path.name, integrated_path.name, spatial_path.name, summary_path.name],
    }
    _atomic_json(summary_path, result)
    if args.upload_s3:
        if s3_client is None:
            from nd2_s3_stage import get_s3_client
            s3_client = get_s3_client(region_name=args.region)
        prefix = '/'.join(value.strip('/') for value in
                          (args.results_s3_prefix, condition_id, 'psc_quantification') if value.strip('/'))
        result['s3_upload'] = _upload_additive(s3_client, output, args.bucket, prefix)
        _atomic_json(summary_path, result)
    logger.event('complete', 'condition_complete', wells_processed=len(rows), wells_total=len(wells),
                 valid_background_wells=len(valid_background))
    return result


def combine_outputs(result_root: Path) -> None:
    specifications = [
        ('psc_well_measurements.csv', 'all_conditions_psc_well_measurements.csv', PSC_FIELDS),
        ('integrated_pdo_psc_well_measurements.csv',
         'all_conditions_integrated_pdo_psc_well_measurements.csv',
         ('condition_id', 'condition_name', 'dose_nM', 'well_id')),
        ('psc_spatial_QC_5mm.csv', 'all_conditions_pdo_psc_spatial_QC_5mm.csv', SPATIAL_FIELDS),
    ]
    for local_name, combined_name, required in specifications:
        rows = []
        for condition_id in CONDITIONS:
            summary_path = result_root / condition_id / 'psc_quantification' / 'psc_summary.json'
            try:
                if _read_json(summary_path).get('completion_status') != 'completed':
                    continue
            except Exception:
                continue
            path = result_root / condition_id / 'psc_quantification' / local_name
            if path.is_file():
                rows.extend(_read_csv(path))
        _atomic_csv(result_root / combined_name, rows, required)
    summaries = []
    for condition_id in CONDITIONS:
        path = result_root / condition_id / 'psc_quantification' / 'psc_summary.json'
        try:
            summary = _read_json(path)
        except Exception:
            continue
        if summary.get('completion_status') != 'completed':
            continue
        summaries.append({key: json.dumps(value, sort_keys=True, default=str)
                          if isinstance(value, (dict, list)) else value
                          for key, value in summary.items()})
    _atomic_csv(result_root / 'all_conditions_psc_summary.csv', summaries,
                ('condition_id', 'condition_name', 'dose_nM', 'completion_status'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Quantitative, restartable raw-RFP second pass over final accepted KT3 wells.')
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--condition-id', action='append', choices=tuple(CONDITIONS), default=[])
    parser.add_argument('--expected-pixel-size-um', type=float, default=EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--tile', type=int, default=2048)
    parser.add_argument('--upload-s3', action='store_true')
    parser.add_argument('--bucket', default='')
    parser.add_argument('--results-s3-prefix', default='')
    parser.add_argument('--region', default='eu-west-2')
    return parser


def run(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
        open_group: Callable = zarr.open_group, s3_client=None) -> int:
    if args.tile < 256:
        raise ValueError('--tile must be at least 256 pixels.')
    if args.upload_s3 and (not args.bucket or not args.results_s3_prefix):
        raise ValueError('--upload-s3 requires --bucket and --results-s3-prefix.')
    result_root = args.result_root.expanduser().resolve()
    status_path = result_root / 'batch_status.json'
    batch_status = _read_json(status_path) if status_path.is_file() else {}
    failures = 0
    selected_conditions = args.condition_id or list(CONDITIONS)
    completed_conditions = 0
    for condition_id in selected_conditions:
        output = result_root / condition_id / 'psc_quantification'
        try:
            summary = export_condition(condition_id, result_root / condition_id, args,
                                       batch_status, probe=probe, open_group=open_group,
                                       s3_client=s3_client)
            if summary.get('well_set_qc_passed') is not True:
                raise RuntimeError('Completed condition did not pass exact final-well-set QC.')
            completed_conditions += 1
            print(f"{condition_id}: completed ({summary['wells_successfully_quantified']}/"
                  f"{summary['total_final_wells']} wells)", flush=True)
        except Exception as exc:
            failures += 1
            tb = traceback.format_exc()
            _atomic_json(output / 'psc_summary.json', {
                'completion_status': 'failed', 'quantification_version': QUANTIFICATION_VERSION,
                'condition_id': condition_id, 'condition_name': CONDITIONS[condition_id]['condition_name'],
                'dose_nM': float(str(CONDITIONS[condition_id]['dose']).split()[0]),
                'failed_at': _now(), 'error': f'{type(exc).__name__}: {exc}', 'traceback': tb,
            })
            RunLogger(condition_id, output / 'psc_run.log').event(
                'failed', 'condition_failed', error=f'{type(exc).__name__}: {exc}', traceback=tb)
        finally:
            try:
                combine_outputs(result_root)
            except Exception as exc:
                failures += 1
                print(f'{condition_id}: combined output failure: {type(exc).__name__}: {exc}', flush=True)
    if not failures:
        print(f'PSC/RFP quantification batch completed: {completed_conditions}/'
              f'{len(selected_conditions)} conditions completed; all condition well-set QC checks passed.',
              flush=True)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
