from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import textwrap
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import zarr
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops

import aws_export_pdo_positive_crops as crop_base
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


QC_VERSION = 1
OUTPUT_DIRECTORY = 'pdo_full_component_qc'
GEOMETRY_PROVENANCE = 'true_pixel_mask_of_matched_unmasked_GFP_connected_component'
QC_STATUS = 'diagnostic_measurement_only_no_exclusion_rule'
EXPECTED_PIXEL_SIZE_UM = crop_base.EXPECTED_PIXEL_SIZE_UM
CONDITIONS = dict(crop_base.CONDITIONS)
CHANNELS = dict(crop_base.CHANNELS)
DMSO_CONDITION = 'K3T_PSC_RMC6236_Lane_1_DMSO'
KNOWN_VISUAL_FAILURE_WELLS = {(DMSO_CONDITION, '606'), (DMSO_CONDITION, '624')}

GREEN_LOW = 30.0
GREEN_HIGH = 45.0
GAUSSIAN_SIGMA_PX = 0.8
PRODUCTION_INTERIOR_FRACTION = 0.86
PRODUCTION_CROP_FRACTION = 0.95
EXPANSION_FACTORS = (2.0, 3.0, 4.5, 6.0)
MAX_CROP_HALF_WIDTH_PX = 2048
CROP_EDGE_GUARD_PX = 4
CENTROID_TOLERANCE_PX = 1e-6
SECONDARY_TOLERANCE = 1e-6

MATCH_FAILURE_REASONS = {
    '', 'clipped_pixels_became_background', 'clipped_pixels_map_to_multiple_components',
    'no_unmasked_component', 'crop_extent_incomplete', 'cross_crop_identity_conflict', 'other',
}

TRUSTED_NUMERIC_FIELDS = (
    'full_component_area_px2', 'full_component_area_um2',
    'full_component_pixels_inside_final_well',
    'full_component_pixels_outside_final_well',
    'full_component_fraction_inside_final_well',
    'full_component_fraction_outside_final_well',
    'full_component_centroid_x_px', 'full_component_centroid_y_px',
    'full_component_fraction_inside_production_0p86r',
)

PDO_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'pdo_number_in_well',
    'known_visual_failure', 'hex_array_member', 'lattice_degree',
    'well_x_px_fullres', 'well_y_px_fullres', 'well_radius_px',
    'pixel_size_x_um', 'pixel_size_y_um', 'gfp_channel',
    'green_low_uint8', 'green_high_uint8', 'pdo_min_area_px',
    'gaussian_sigma_px', 'split_pdos', 'production_interior_radius_fraction',
    'production_crop_radius_fraction',
    'production_PDO_centroid_x_px_fullres', 'production_PDO_centroid_y_px_fullres',
    'production_PDO_projected_area_px2', 'production_PDO_projected_area_um2',
    'production_PDO_equivalent_circular_diameter_um',
    'production_component_reproduction_status',
    'production_component_reproduction_failure_reason',
    'production_centroid_x_difference_px', 'production_centroid_y_difference_px',
    'production_projected_area_difference_px2',
    'production_projected_area_difference_um2',
    'production_equivalent_diameter_difference_um',
    'clipped_component_id', 'clipped_component_area_px2',
    'unmasked_component_id', 'clipped_pixels_matching_unmasked_component',
    'clipped_component_overlap_fraction', 'unmasked_component_match_status',
    'unmasked_component_match_failure_reason',
    'requested_crop_bounds', 'actual_source_clipped_crop_bounds',
    'crop_half_width_px', 'crop_expansion_count', 'crop_edge_guard_px',
    'component_touches_crop_edge', 'full_component_extent_status',
    *TRUSTED_NUMERIC_FIELDS,
    'full_component_centroid_inside_final_well',
    'full_component_touches_final_well_boundary',
    'full_component_crosses_final_well_boundary',
    'unmasked_component_production_PDO_count', 'associated_production_well_ids',
    'associated_production_PDO_numbers',
    'many_production_PDOs_to_one_unmasked_component',
    'containment_geometry_provenance', 'containment_qc_status',
)

COMPONENT_FIELDS = (
    'condition_id', 'unmasked_component_id', 'fullres_bbox_x0', 'fullres_bbox_y0',
    'fullres_bbox_x1_exclusive', 'fullres_bbox_y1_exclusive',
    'full_component_area_px2', 'full_component_area_um2',
    'full_component_centroid_x_px', 'full_component_centroid_y_px',
    'unmasked_component_production_PDO_count', 'associated_production_well_ids',
    'associated_production_PDO_numbers',
    'many_production_PDOs_to_one_unmasked_component',
    'full_component_extent_status', 'component_hash_provenance',
    'containment_geometry_provenance', 'containment_qc_status',
)

DIAGNOSTIC_FIELDS = (*PDO_FIELDS, 'diagnostic_sampling_reasons',
                     'labelled_diagnostic', 'export_status', 'error')


@dataclass
class Component:
    label_id: int
    coords_yx_fullres: np.ndarray
    area_px2: int
    centroid_x_fullres: float
    centroid_y_fullres: float
    intensity_max: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Iterable[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def _well_id(value: object) -> str:
    return crop_base._normalise_well_id(value)


def _finite(row: dict, key: str) -> float:
    value = crop_base._number(row, key)
    if not math.isfinite(value):
        raise RuntimeError(f"Field '{key}' must be finite.")
    return value


def _nan_trusted(row: dict) -> None:
    for field in TRUSTED_NUMERIC_FIELDS:
        row[field] = math.nan
    row['full_component_centroid_inside_final_well'] = ''
    row['full_component_touches_final_well_boundary'] = ''
    row['full_component_crosses_final_well_boundary'] = ''


def _u8_absolute(arr: np.ndarray, maximum: float) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    return np.clip(data * (255.0 / max(float(maximum), 1.0)), 0, 255).astype(np.uint8)


def quantitative_window_end(root, channel: int, dtype) -> tuple[float, str]:
    try:
        end = float(root.attrs.get('omero', {}).get('channels', [])[channel]
                    .get('window', {}).get('end'))
        if end > 0:
            return end, 'ome_omero_channel_window_end'
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max), 'source_dtype_maximum_fallback'
    return 1.0, 'floating_source_unit_maximum_fallback'


def validate_scientific_settings(summary: dict) -> int:
    settings = summary.get('scientific_settings') or {}
    required = ('green_low', 'green_high', 'pdo_min_area')
    missing = [key for key in required if key not in settings]
    if missing:
        raise RuntimeError(f'Condition summary lacks recorded PDO settings: {missing}.')
    if float(settings['green_low']) != GREEN_LOW or float(settings['green_high']) != GREEN_HIGH:
        raise RuntimeError(
            f'Recorded GFP thresholds differ from approved production values: {settings}.'
        )
    minimum = int(settings['pdo_min_area'])
    if minimum < 1:
        raise RuntimeError(f'Invalid recorded pdo_min_area={minimum}.')
    mapping = summary.get('channel_mapping') or {}
    if int(mapping.get('gfp_channel', -1)) != CHANNELS['gfp']:
        raise RuntimeError(f'Recorded GFP channel provenance is invalid: {mapping}.')
    return minimum


def segment_components(signal: np.ndarray, minimum_area: int, *, left: int = 0,
                       top: int = 0) -> tuple[np.ndarray, list[Component]]:
    labels = label(signal > GREEN_LOW)
    components = []
    accepted_labels = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for region in regionprops(labels, intensity_image=signal):
        if int(region.area) < int(minimum_area) or float(region.intensity_max) < GREEN_HIGH:
            continue
        coords = region.coords.astype(np.int64, copy=True)
        accepted_labels[coords[:, 0], coords[:, 1]] = next_id
        coords[:, 0] += int(top)
        coords[:, 1] += int(left)
        cy, cx = region.centroid
        components.append(Component(
            next_id, coords, int(region.area), float(left + cx), float(top + cy),
            float(region.intensity_max),
        ))
        next_id += 1
    return accepted_labels, components


def production_crop_bounds(well: dict, width: int, height: int) -> tuple[int, int, int, int]:
    wx, wy, radius = _finite(well, 'x_px_fullres'), _finite(well, 'y_px_fullres'), _finite(well, 'radius_px')
    cr = int(math.ceil(PRODUCTION_CROP_FRACTION * radius))
    cx, cy = int(wx), int(wy)
    return max(0, cx-cr), max(0, cy-cr), min(width, cx+cr+1), min(height, cy+cr+1)


def reproduce_production_components(planes: SingletonTZCYX, well: dict, width: int,
                                    height: int, maximum: float,
                                    minimum_area: int) -> list[Component]:
    left, top, right, bottom = production_crop_bounds(well, width, height)
    raw = planes.read(CHANNELS['gfp'], slice(top, bottom), slice(left, right))
    sub = _u8_absolute(raw, maximum).astype(np.float32)
    wx, wy, radius = _finite(well, 'x_px_fullres'), _finite(well, 'y_px_fullres'), _finite(well, 'radius_px')
    scx, scy = int(wx) - left, int(wy) - top
    yy, xx = np.ogrid[:sub.shape[0], :sub.shape[1]]
    interior = (xx-scx)**2 + (yy-scy)**2 <= (PRODUCTION_INTERIOR_FRACTION*radius)**2
    signal = gaussian_filter(np.where(interior, sub, 0.0), GAUSSIAN_SIGMA_PX)
    _, objects = segment_components(signal, minimum_area, left=left, top=top)
    return [obj for obj in objects
            if (obj.centroid_x_fullres-wx)**2 + (obj.centroid_y_fullres-wy)**2
            <= (PRODUCTION_INTERIOR_FRACTION*radius)**2]


def match_production_row(pdo: dict, reproduced: list[Component], pixel_size_um: float) -> tuple[Component | None, dict]:
    target_area = _finite(pdo, 'projected_area_px2')
    target_x = _finite(pdo, 'centroid_x_px_fullres')
    target_y = _finite(pdo, 'centroid_y_px_fullres')
    target_number = int(round(_finite(pdo, 'pdo_number_in_well')))
    candidates = []
    for number, component in enumerate(reproduced, 1):
        if (number == target_number and component.area_px2 == int(target_area)
                and abs(component.centroid_x_fullres-target_x) <= CENTROID_TOLERANCE_PX
                and abs(component.centroid_y_fullres-target_y) <= CENTROID_TOLERANCE_PX):
            candidates.append(component)
    details = {
        'production_centroid_x_difference_px': math.nan,
        'production_centroid_y_difference_px': math.nan,
        'production_projected_area_difference_px2': math.nan,
        'production_projected_area_difference_um2': math.nan,
        'production_equivalent_diameter_difference_um': math.nan,
    }
    if len(candidates) != 1:
        return None, details
    component = candidates[0]
    reproduced_um2 = component.area_px2 * pixel_size_um**2
    reproduced_diameter = 2.0 * math.sqrt(component.area_px2/math.pi) * pixel_size_um
    details.update({
        'production_centroid_x_difference_px': component.centroid_x_fullres-target_x,
        'production_centroid_y_difference_px': component.centroid_y_fullres-target_y,
        'production_projected_area_difference_px2': component.area_px2-target_area,
        'production_projected_area_difference_um2': reproduced_um2-_finite(pdo, 'projected_area_um2'),
        'production_equivalent_diameter_difference_um': reproduced_diameter-_finite(pdo, 'equivalent_circular_diameter_um'),
    })
    if (abs(details['production_projected_area_difference_um2']) > SECONDARY_TOLERANCE
            or abs(details['production_equivalent_diameter_difference_um']) > SECONDARY_TOLERANCE):
        return None, details
    return component, details


def _requested_bounds(wx: float, wy: float, half: int) -> tuple[int, int, int, int]:
    cx, cy = int(wx), int(wy)
    return cx-half, cy-half, cx+half+1, cy+half+1


def _actual_bounds(requested: tuple[int, int, int, int], width: int,
                   height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = requested
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def overlap_match(clipped: Component, accepted_labels: np.ndarray, left: int,
                  top: int) -> tuple[int | None, int, float, str]:
    yy = clipped.coords_yx_fullres[:, 0] - int(top)
    xx = clipped.coords_yx_fullres[:, 1] - int(left)
    valid = ((yy >= 0) & (yy < accepted_labels.shape[0])
             & (xx >= 0) & (xx < accepted_labels.shape[1]))
    if not np.all(valid):
        return None, 0, 0.0, 'no_unmasked_component'
    values = accepted_labels[yy, xx]
    positive = values[values > 0]
    if positive.size == 0:
        return None, 0, 0.0, 'no_unmasked_component'
    if positive.size != values.size:
        return None, int(positive.size), positive.size/values.size, 'clipped_pixels_became_background'
    unique = np.unique(positive)
    if unique.size != 1:
        return None, int(positive.size), 1.0, 'clipped_pixels_map_to_multiple_components'
    return int(unique[0]), int(values.size), 1.0, ''


def _component_in_guard(component: Component, bounds: tuple[int, int, int, int],
                        guard: int = CROP_EDGE_GUARD_PX) -> tuple[bool, set[str]]:
    x0, y0, x1, y1 = bounds
    ys, xs = component.coords_yx_fullres[:, 0], component.coords_yx_fullres[:, 1]
    sides = set()
    if np.any(xs < x0+guard): sides.add('left')
    if np.any(xs >= x1-guard): sides.add('right')
    if np.any(ys < y0+guard): sides.add('top')
    if np.any(ys >= y1-guard): sides.add('bottom')
    return bool(sides), sides


def recover_unmasked_component(planes: SingletonTZCYX, clipped: Component, well: dict,
                               width: int, height: int, maximum: float,
                               minimum_area: int) -> tuple[Component | None, dict]:
    wx, wy, radius = _finite(well, 'x_px_fullres'), _finite(well, 'y_px_fullres'), _finite(well, 'radius_px')
    maximum_half = min(int(math.ceil(EXPANSION_FACTORS[-1]*radius)), MAX_CROP_HALF_WIDTH_PX)
    halves = []
    for factor in EXPANSION_FACTORS:
        half = min(int(math.ceil(factor*radius)), maximum_half)
        if not halves or half != halves[-1]: halves.append(half)
    result = {
        'unmasked_component_match_status': 'failed',
        'unmasked_component_match_failure_reason': 'other',
        'clipped_pixels_matching_unmasked_component': 0,
        'clipped_component_overlap_fraction': 0.0,
        'requested_crop_bounds': '', 'actual_source_clipped_crop_bounds': '',
        'crop_half_width_px': math.nan, 'crop_expansion_count': 0,
        'crop_edge_guard_px': CROP_EDGE_GUARD_PX,
        'component_touches_crop_edge': '', 'full_component_extent_status': 'not_evaluated',
    }
    for expansion, half in enumerate(halves):
        requested = _requested_bounds(wx, wy, half)
        actual = _actual_bounds(requested, width, height)
        left, top, right, bottom = actual
        raw = planes.read(CHANNELS['gfp'], slice(top, bottom), slice(left, right))
        signal = gaussian_filter(_u8_absolute(raw, maximum).astype(np.float32), GAUSSIAN_SIGMA_PX)
        labels, components = segment_components(signal, minimum_area, left=left, top=top)
        component_id, overlap, fraction, failure = overlap_match(clipped, labels, left, top)
        result.update({
            'clipped_pixels_matching_unmasked_component': overlap,
            'clipped_component_overlap_fraction': fraction,
            'requested_crop_bounds': json.dumps(requested),
            'actual_source_clipped_crop_bounds': json.dumps(actual),
            'crop_half_width_px': half, 'crop_expansion_count': expansion,
        })
        if failure:
            result['unmasked_component_match_failure_reason'] = failure
            return None, result
        matched = next((item for item in components if item.label_id == component_id), None)
        if matched is None:
            result['unmasked_component_match_failure_reason'] = 'no_unmasked_component'
            return None, result
        edge, sides = _component_in_guard(matched, actual)
        source_sides = set()
        if actual[0] != requested[0]: source_sides.add('left')
        if actual[1] != requested[1]: source_sides.add('top')
        if actual[2] != requested[2]: source_sides.add('right')
        if actual[3] != requested[3]: source_sides.add('bottom')
        result['component_touches_crop_edge'] = edge
        if edge and sides & source_sides:
            result.update(full_component_extent_status='incomplete_at_source_boundary',
                          unmasked_component_match_failure_reason='crop_extent_incomplete')
            return None, result
        if edge and half == maximum_half:
            result.update(full_component_extent_status='incomplete_at_maximum_crop',
                          unmasked_component_match_failure_reason='crop_extent_incomplete')
            return None, result
        if edge:
            continue
        result.update(unmasked_component_match_status='trusted_complete_match',
                      unmasked_component_match_failure_reason='',
                      full_component_extent_status='complete')
        return matched, result
    result.update(full_component_extent_status='incomplete_at_maximum_crop',
                  unmasked_component_match_failure_reason='crop_extent_incomplete')
    return None, result


def component_hash(condition_id: str, component: Component) -> tuple[str, tuple[int, int, int, int], np.ndarray]:
    coords = component.coords_yx_fullres
    y0, x0 = np.min(coords, axis=0)
    y1, x1 = np.max(coords, axis=0) + 1
    mask = np.zeros((int(y1-y0), int(x1-x0)), dtype=np.uint8)
    mask[coords[:, 0]-y0, coords[:, 1]-x0] = 1
    bbox = (int(x0), int(y0), int(x1), int(y1))
    header = json.dumps({'condition_id': condition_id, 'bbox_xyxy_exclusive': bbox},
                        sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(header + b'\0' + np.packbits(mask, axis=None, bitorder='big').tobytes()).hexdigest()
    return f'gfpcomp_{digest}', bbox, mask


def canonical_masks_overlap(first_bbox: tuple[int, int, int, int], first_mask: np.ndarray,
                            second_bbox: tuple[int, int, int, int], second_mask: np.ndarray) -> bool:
    x0=max(first_bbox[0],second_bbox[0]); y0=max(first_bbox[1],second_bbox[1])
    x1=min(first_bbox[2],second_bbox[2]); y1=min(first_bbox[3],second_bbox[3])
    if x0>=x1 or y0>=y1:
        return False
    first=first_mask[y0-first_bbox[1]:y1-first_bbox[1],x0-first_bbox[0]:x1-first_bbox[0]]
    second=second_mask[y0-second_bbox[1]:y1-second_bbox[1],x0-second_bbox[0]:x1-second_bbox[0]]
    return bool(np.any(first & second))


def mask_containment(component: Component, well: dict, px_x: float, px_y: float) -> dict:
    wx, wy, radius = _finite(well, 'x_px_fullres'), _finite(well, 'y_px_fullres'), _finite(well, 'radius_px')
    ys = component.coords_yx_fullres[:, 0].astype(float)
    xs = component.coords_yx_fullres[:, 1].astype(float)
    d2 = (xs-wx)**2 + (ys-wy)**2
    inside = d2 <= radius**2
    inside86 = d2 <= (PRODUCTION_INTERIOR_FRACTION*radius)**2
    count = component.area_px2
    inside_count = int(np.count_nonzero(inside))
    outside_count = count-inside_count
    distances = np.sqrt(d2)
    # A pixel square intersects the analytic circle when its minimum and maximum
    # distance from the well centre straddle the radius.
    dx = np.abs(xs-wx); dy = np.abs(ys-wy)
    min_d = np.hypot(np.maximum(dx-0.5, 0.0), np.maximum(dy-0.5, 0.0))
    max_d = np.hypot(dx+0.5, dy+0.5)
    touches = bool(np.any((min_d <= radius) & (max_d >= radius)))
    return {
        'full_component_area_px2': count,
        'full_component_area_um2': count*px_x*px_y,
        'full_component_pixels_inside_final_well': inside_count,
        'full_component_pixels_outside_final_well': outside_count,
        'full_component_fraction_inside_final_well': inside_count/count,
        'full_component_fraction_outside_final_well': outside_count/count,
        'full_component_centroid_x_px': component.centroid_x_fullres,
        'full_component_centroid_y_px': component.centroid_y_fullres,
        'full_component_centroid_inside_final_well': (
            (component.centroid_x_fullres-wx)**2+(component.centroid_y_fullres-wy)**2 <= radius**2),
        'full_component_touches_final_well_boundary': touches,
        'full_component_crosses_final_well_boundary': inside_count > 0 and outside_count > 0,
        'full_component_fraction_inside_production_0p86r': int(np.count_nonzero(inside86))/count,
    }


def _base_row(condition_id: str, well: dict, pdo: dict, settings: dict,
              px_x: float, px_y: float) -> dict:
    mapping = CONDITIONS[condition_id]
    well_id = _well_id(well['well_id'])
    row = {
        'condition_id': condition_id, 'condition_name': mapping['condition_name'],
        'dose': mapping.get('dose', mapping['condition_name']), 'well_id': well_id,
        'pdo_number_in_well': int(round(_finite(pdo, 'pdo_number_in_well'))),
        'known_visual_failure': (condition_id, well_id) in KNOWN_VISUAL_FAILURE_WELLS,
        'hex_array_member': True, 'lattice_degree': well.get('lattice_degree', ''),
        'well_x_px_fullres': _finite(well, 'x_px_fullres'),
        'well_y_px_fullres': _finite(well, 'y_px_fullres'),
        'well_radius_px': _finite(well, 'radius_px'),
        'pixel_size_x_um': px_x, 'pixel_size_y_um': px_y,
        'gfp_channel': CHANNELS['gfp'], 'green_low_uint8': GREEN_LOW,
        'green_high_uint8': GREEN_HIGH, 'pdo_min_area_px': settings['pdo_min_area'],
        'gaussian_sigma_px': GAUSSIAN_SIGMA_PX, 'split_pdos': False,
        'production_interior_radius_fraction': PRODUCTION_INTERIOR_FRACTION,
        'production_crop_radius_fraction': PRODUCTION_CROP_FRACTION,
        'production_PDO_centroid_x_px_fullres': _finite(pdo, 'centroid_x_px_fullres'),
        'production_PDO_centroid_y_px_fullres': _finite(pdo, 'centroid_y_px_fullres'),
        'production_PDO_projected_area_px2': _finite(pdo, 'projected_area_px2'),
        'production_PDO_projected_area_um2': _finite(pdo, 'projected_area_um2'),
        'production_PDO_equivalent_circular_diameter_um': _finite(pdo, 'equivalent_circular_diameter_um'),
        'containment_geometry_provenance': GEOMETRY_PROVENANCE,
        'containment_qc_status': QC_STATUS,
    }
    _nan_trusted(row)
    return row


def quantify_condition_components(condition_id: str, wells: list[dict], pdos: list[dict],
                                  planes: SingletonTZCYX, width: int, height: int,
                                  maximum: float, minimum_area: int,
                                  px_x: float, px_y: float) -> tuple[list[dict], list[dict]]:
    well_by_id = {}
    for well in wells:
        wid = _well_id(well.get('well_id'))
        if wid in well_by_id: raise RuntimeError(f'Duplicate final well_id {wid}.')
        if 'hex_array_member' not in well or not _truthy(well.get('hex_array_member')):
            raise RuntimeError(f'Final well {wid} lacks truthy hex_array_member provenance.')
        well_by_id[wid] = well
    grouped: dict[str, list[dict]] = {}
    for pdo in pdos:
        wid = _well_id(pdo.get('well_id'))
        if wid not in well_by_id: raise RuntimeError(f'PDO row refers to non-final well {wid}.')
        grouped.setdefault(wid, []).append(pdo)
    for wid, well in well_by_id.items():
        if len(grouped.get(wid, [])) != int(round(_finite(well, 'PDO_count'))):
            raise RuntimeError(f'Final PDO count mismatch for well {wid}.')
    output = []
    settings = {'pdo_min_area': minimum_area}
    reproduced_cache = {}
    internal_components: dict[str, tuple[Component, tuple[int, int, int, int], np.ndarray]] = {}
    seen_components: dict[str, tuple[Component, tuple[int, int, int, int], np.ndarray]] = {}
    conflicted_component_ids: set[str] = set()
    for pdo in sorted(pdos, key=lambda q: (_well_id(q['well_id']), int(float(q['pdo_number_in_well'])))):
        wid = _well_id(pdo['well_id']); well = well_by_id[wid]
        row = _base_row(condition_id, well, pdo, settings, px_x, px_y)
        reproduced = reproduced_cache.setdefault(
            wid, reproduce_production_components(planes, well, width, height, maximum, minimum_area))
        clipped, differences = match_production_row(pdo, reproduced, math.sqrt(px_x*px_y))
        row.update(differences)
        if clipped is None:
            row.update(production_component_reproduction_status='production_component_reproduction_failed',
                       production_component_reproduction_failure_reason='strict_identity_mismatch',
                       clipped_component_id='', clipped_component_area_px2=math.nan,
                       unmasked_component_id='', unmasked_component_match_status='not_evaluated',
                       unmasked_component_match_failure_reason='other')
            _nan_trusted(row); output.append(row); continue
        row.update(production_component_reproduction_status='reproduced_exactly',
                   production_component_reproduction_failure_reason='',
                   clipped_component_id=f'{wid}:PDO{row["pdo_number_in_well"]}:label{clipped.label_id}',
                   clipped_component_area_px2=clipped.area_px2)
        full, match = recover_unmasked_component(
            planes, clipped, well, width, height, maximum, minimum_area)
        row.update(match)
        if full is None:
            row['unmasked_component_id'] = ''
            _nan_trusted(row); output.append(row); continue
        stable_id, bbox, mask = component_hash(condition_id, full)
        row['unmasked_component_id'] = stable_id
        row.update(mask_containment(full, well, px_x, px_y))
        row['_full_component'] = full
        row['_clipped_component'] = clipped
        conflicts=[candidate_id for candidate_id,(_,candidate_bbox,candidate_mask) in seen_components.items()
                   if candidate_id!=stable_id and canonical_masks_overlap(bbox,mask,candidate_bbox,candidate_mask)]
        if conflicts or stable_id in conflicted_component_ids:
            conflicted_component_ids.update(conflicts); conflicted_component_ids.add(stable_id)
            row.update(unmasked_component_match_status='failed',
                       unmasked_component_match_failure_reason='cross_crop_identity_conflict')
            _nan_trusted(row)
            for prior_row in output:
                if prior_row.get('unmasked_component_id') in conflicts:
                    prior_row.update(unmasked_component_match_status='failed',
                                     unmasked_component_match_failure_reason='cross_crop_identity_conflict')
                    _nan_trusted(prior_row)
            for candidate_id in conflicts:
                internal_components.pop(candidate_id,None)
            seen_components[stable_id]=(full,bbox,mask)
            output.append(row)
            continue
        prior = internal_components.get(stable_id)
        if prior is not None and (prior[1] != bbox or not np.array_equal(prior[2], mask)):
            row.update(unmasked_component_match_status='failed',
                       unmasked_component_match_failure_reason='cross_crop_identity_conflict')
            _nan_trusted(row)
        else:
            internal_components[stable_id] = (full, bbox, mask)
            seen_components[stable_id] = (full, bbox, mask)
        output.append(row)
    groups: dict[str, list[dict]] = {}
    for row in output:
        if row.get('unmasked_component_match_status') == 'trusted_complete_match':
            groups.setdefault(row['unmasked_component_id'], []).append(row)
    component_rows = []
    for component_id, rows in groups.items():
        full, bbox, _ = internal_components[component_id]
        associations=sorted(rows,key=lambda row: (_well_id(row['well_id']),int(row['pdo_number_in_well'])))
        well_ids=[_well_id(row['well_id']) for row in associations]
        pdo_numbers=[str(int(row['pdo_number_in_well'])) for row in associations]
        for row in rows:
            row.update(unmasked_component_production_PDO_count=len(rows),
                       associated_production_well_ids=';'.join(well_ids),
                       associated_production_PDO_numbers=';'.join(pdo_numbers),
                       many_production_PDOs_to_one_unmasked_component=len(rows) > 1)
        component_rows.append({
            'condition_id': condition_id, 'unmasked_component_id': component_id,
            'fullres_bbox_x0': bbox[0], 'fullres_bbox_y0': bbox[1],
            'fullres_bbox_x1_exclusive': bbox[2], 'fullres_bbox_y1_exclusive': bbox[3],
            'full_component_area_px2': full.area_px2,
            'full_component_area_um2': full.area_px2*px_x*px_y,
            'full_component_centroid_x_px': full.centroid_x_fullres,
            'full_component_centroid_y_px': full.centroid_y_fullres,
            'unmasked_component_production_PDO_count': len(rows),
            'associated_production_well_ids': ';'.join(well_ids),
            'associated_production_PDO_numbers': ';'.join(pdo_numbers),
            'many_production_PDOs_to_one_unmasked_component': len(rows) > 1,
            'full_component_extent_status': 'complete',
            'component_hash_provenance': ('sha256(condition_id + canonical full-resolution '
                                          'bbox + row-major packed binary bbox mask)'),
            'containment_geometry_provenance': GEOMETRY_PROVENANCE,
            'containment_qc_status': QC_STATUS,
        })
    for row in output:
        if not row.get('unmasked_component_production_PDO_count'):
            row.update(unmasked_component_production_PDO_count=math.nan,
                       associated_production_well_ids='', associated_production_PDO_numbers='',
                       many_production_PDOs_to_one_unmasked_component='')
    return output, sorted(component_rows, key=lambda q: q['unmasked_component_id'])


def select_diagnostics(condition_id: str, rows: list[dict]) -> list[dict]:
    selected: dict[tuple[str, int], dict] = {}; reasons: dict[tuple[str, int], set[str]] = {}
    def key(row): return (_well_id(row['well_id']), int(row['pdo_number_in_well']))
    def add(items, reason, limit=None):
        for row in list(items)[:limit]: selected[key(row)] = row; reasons.setdefault(key(row), set()).add(reason)
    trusted = [row for row in rows if row.get('unmasked_component_match_status') == 'trusted_complete_match'
               and not _truthy(row.get('known_visual_failure'))]
    tie = lambda row: (int(row['well_id']) if str(row['well_id']).isdigit() else str(row['well_id']), int(row['pdo_number_in_well']))
    add(sorted(trusted, key=lambda r: (r['full_component_fraction_inside_final_well'], tie(r))), 'lowest_fraction_inside', 5)
    crossing = sorted([r for r in trusted if _truthy(r['full_component_crosses_final_well_boundary'])],
                      key=lambda r: (r['full_component_fraction_inside_final_well'], tie(r)))
    if crossing:
        indices = sorted(set(int(round(q*(len(crossing)-1))) for q in np.linspace(0, 1, 5)))
        add([crossing[i] for i in indices], 'boundary_crossing_distribution')
    contained = [r for r in trusted if r['full_component_fraction_inside_final_well'] == 1.0]
    radial = lambda r: math.hypot(r['full_component_centroid_x_px']-r['well_x_px_fullres'],
                                  r['full_component_centroid_y_px']-r['well_y_px_fullres'])/r['well_radius_px']
    add(sorted(contained, key=lambda r: (-radial(r), tie(r))), 'fully_contained_near_wall_control', 3)
    add(sorted(contained, key=lambda r: (radial(r), tie(r))), 'fully_contained_central_control', 3)
    add(sorted(trusted, key=lambda r: (r['full_component_area_px2'], tie(r))), 'smallest_complete_component', 2)
    add(sorted(trusted, key=lambda r: (-r['full_component_area_px2'], tie(r))), 'largest_complete_component', 2)
    add([r for r in trusted if _truthy(r['many_production_PDOs_to_one_unmasked_component'])], 'many_to_one_component')
    failures = [r for r in rows if r.get('unmasked_component_match_status') != 'trusted_complete_match'
                and not _truthy(r.get('known_visual_failure'))]
    add(sorted(failures, key=tie), 'bounded_failure_supplement', 6)
    forced = [r for r in rows if _truthy(r.get('known_visual_failure'))]
    if condition_id == DMSO_CONDITION:
        required = {'606', '624'}; present = {_well_id(r['well_id']) for r in forced}
        if present != required:
            raise RuntimeError(f'Mandatory known visual failure PDO wells missing: {sorted(required-present)}.')
    add(forced, 'known_visual_failure_mandatory')
    result = []
    for identity in sorted(selected, key=lambda k: (int(k[0]) if k[0].isdigit() else k[0], k[1])):
        row = dict(selected[identity]); row['diagnostic_sampling_reasons'] = ';'.join(sorted(reasons[identity])); result.append(row)
    return result


def _outline_from_component(component: Component, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    coords = component.coords_yx_fullres
    yy, xx = coords[:, 0]-y0, coords[:, 1]-x0
    valid = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    mask[yy[valid], xx[valid]] = 255
    return mask


def _draw_dashed_circle(array: np.ndarray, centre: tuple[int, int], radius: int,
                        colour: tuple[int, int, int], width: int = 1) -> None:
    for start in range(0,360,20):
        cv2.ellipse(array,centre,(radius,radius),0,start,min(start+11,360),colour,width,lineType=cv2.LINE_AA)


def diagnostic_image(row: dict, well: dict, dic: np.ndarray, gfp: np.ndarray,
                     bounds: tuple[int, int, int, int], display_range: tuple[float, float],
                     panel_size: int) -> Image.Image:
    x0, y0, _, _ = bounds
    dic8 = crop_base._u8_local(dic); gfp8 = crop_base._u8_range(gfp, *display_range)
    clipped = row.get('_clipped_component'); full = row.get('_full_component')
    clip_mask = _outline_from_component(clipped, bounds) if clipped else np.zeros(dic8.shape, np.uint8)
    full_mask = _outline_from_component(full, bounds) if full else np.zeros(dic8.shape, np.uint8)
    mask_rgb = np.zeros((*dic8.shape, 3), np.uint8); mask_rgb[..., 1] = full_mask
    dic_rgb = np.stack([dic8]*3, axis=-1); gfp_rgb = np.zeros_like(dic_rgb); gfp_rgb[..., 1] = gfp8
    composite = dic_rgb.copy(); composite[..., 1] = np.maximum(composite[..., 1], gfp8)
    panels = [dic_rgb, gfp_rgb, mask_rgb, composite]
    wx = (_finite(well, 'x_px_fullres')-x0); wy = (_finite(well, 'y_px_fullres')-y0); radius = _finite(well, 'radius_px')
    for array in panels:
        cv2.circle(array, (int(round(wx)), int(round(wy))), int(round(radius)), (255,255,0), 2)
        _draw_dashed_circle(array,(int(round(wx)),int(round(wy))),
                            int(round(PRODUCTION_INTERIOR_FRACTION*radius)),(255,255,0),1)
        if clipped is not None:
            contours, _ = cv2.findContours(clip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(array, contours, -1, (0,255,255), 2)
        if full is not None:
            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(array, contours, -1, (255,0,255), 2)
        px = int(round(row['production_PDO_centroid_x_px_fullres']-x0)); py = int(round(row['production_PDO_centroid_y_px_fullres']-y0))
        cv2.drawMarker(array, (px, py), (255,255,255), cv2.MARKER_CROSS, 9, 2)
    def fmt(field):
        value = row.get(field, math.nan)
        return 'not quantified' if not isinstance(value, (int,float)) or not math.isfinite(float(value)) else f'{float(value):.4f}'
    failure = ' | KNOWN VISUAL FAILURE' if _truthy(row['known_visual_failure']) else ''
    lines = [
        f"{row['condition_name']} | {row['dose']} | well {row['well_id']} | PDO {row['pdo_number_in_well']}{failure}",
        f"Existing PDO diameter: {row['production_PDO_equivalent_circular_diameter_um']:.2f} µm",
        f"Production PDO area: {row['production_PDO_projected_area_px2']:.0f} px² | Complete unmasked GFP-component area: {fmt('full_component_area_px2')} px²",
        f"Production reproduction: {row['production_component_reproduction_status']} | Match: {row['unmasked_component_match_status']}",
        f"Fraction inside final well: {fmt('full_component_fraction_inside_final_well')} | outside: {fmt('full_component_fraction_outside_final_well')}",
        f"Fraction inside production 0.86r: {fmt('full_component_fraction_inside_production_0p86r')} | overlap: {row.get('clipped_component_overlap_fraction', math.nan)}",
        f"Extent: {row.get('full_component_extent_status')} | many-to-one: {row.get('many_production_PDOs_to_one_unmasked_component')}",
        f"Sampling: {row.get('diagnostic_sampling_reasons', '')}",
    ]
    gap=8; title_h=25; width=4*panel_size+5*gap; _, font=crop_base._fonts(); wrapped=[]
    for line in lines: wrapped.extend(textwrap.wrap(line, width=max(80,width//9), break_long_words=False) or [''])
    header=12+len(wrapped)*23; canvas=Image.new('RGB',(width,header+title_h+panel_size+2*gap),'white'); draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,width,header),fill='black'); yy=6
    for line in wrapped: draw.text((10,yy),line,fill='white',font=font); yy+=23
    for i,(array,title) in enumerate(zip(panels,('DIC','GFP','GFP mask','Composite'))):
        x=gap+i*(panel_size+gap); py=header+gap; draw.rectangle((x,py,x+panel_size,py+title_h),fill='black'); draw.text((x+6,py+3),title,fill='white',font=font)
        image=Image.fromarray(array).resize((panel_size,panel_size),Image.Resampling.LANCZOS); canvas.paste(image,(x,py+title_h)); image.close()
    return canvas


def export_condition(condition_id: str, folder: Path, args: argparse.Namespace,
                     batch_status: dict, *, probe: Callable = probe_omezarr,
                     open_group: Callable = zarr.open_group) -> dict:
    summary_path=folder/'condition_summary.json'; well_path=folder/'well_measurements.csv'; pdo_path=folder/'pdo_measurements.csv'
    for path in (summary_path,well_path,pdo_path):
        if not path.is_file(): raise FileNotFoundError(f'Required completed source is missing: {path}')
    summary=_read_json(summary_path); wells=_read_csv(well_path); pdos=_read_csv(pdo_path)
    minimum_area=validate_scientific_settings(summary)
    zarr_path=crop_base.resolve_omezarr(condition_id,summary,batch_status,args.cache_root)
    metadata=probe(zarr_path); validation=crop_base.validate_omezarr(metadata,summary,args.expected_pixel_size_um)
    px_x=float(validation['pixel_size_um']['x']); px_y=float(validation['pixel_size_um']['y'])
    root=open_group(str(zarr_path),mode='r'); array=root[metadata['level0_array_path']]; planes=SingletonTZCYX(array,metadata['axes'])
    channels,height,width=planes.shape_cyx
    if channels != 3: raise RuntimeError(f'Validated OME-Zarr channel count changed after open: {channels}.')
    maximum,window_source=quantitative_window_end(root,CHANNELS['gfp'],array.dtype)
    rows,components=quantify_condition_components(condition_id,wells,pdos,planes,width,height,maximum,minimum_area,px_x,px_y)
    diagnostics=select_diagnostics(condition_id,rows)
    output=folder/OUTPUT_DIRECTORY; output.mkdir(parents=True,exist_ok=True)
    _atomic_csv(output/'pdo_full_component_measurements.csv',rows,PDO_FIELDS)
    _atomic_csv(output/'unmasked_component_summary.csv',components,COMPONENT_FIELDS)
    well_by_id={_well_id(w['well_id']):w for w in wells}; manifest=[]
    display=crop_base._metadata_window(metadata,CHANNELS['gfp']) or (0.0,maximum)
    for row in diagnostics:
        item=dict(row); wid=_well_id(row['well_id']); well=well_by_id[wid]
        try:
            half=max(int(math.ceil(2.0*_finite(well,'radius_px'))),1); requested=_requested_bounds(_finite(well,'x_px_fullres'),_finite(well,'y_px_fullres'),half); bounds=_actual_bounds(requested,width,height)
            x0,y0,x1,y1=bounds; dic=planes.read(CHANNELS['dic'],slice(y0,y1),slice(x0,x1)); gfp=planes.read(CHANNELS['gfp'],slice(y0,y1),slice(x0,x1))
            image=diagnostic_image(row,well,dic,gfp,bounds,display,args.panel_size)
            filename=f'{condition_id}__well_{wid}__PDO_{int(row["pdo_number_in_well"]):02d}.png'; path=output/'labelled_diagnostics'/filename; path.parent.mkdir(parents=True,exist_ok=True); image.save(path,dpi=(300,300)); image.close()
            item.update(labelled_diagnostic=str(path),export_status='completed',error='')
        except Exception as exc:
            item.update(labelled_diagnostic='',export_status='failed',error=f'{type(exc).__name__}: {exc}')
        manifest.append(item); _atomic_csv(output/'diagnostic_manifest.csv',manifest,DIAGNOSTIC_FIELDS)
    completed=[Path(r['labelled_diagnostic']) for r in manifest if r.get('export_status')=='completed' and Path(r['labelled_diagnostic']).is_file()]
    sheets=crop_base._contact_sheets(completed,output/'contact_sheets',args.contact_sheet_size)
    known={_well_id(r['well_id']):r for r in rows if _truthy(r.get('known_visual_failure'))}
    known_status={wid:{'reproduction':r['production_component_reproduction_status'],'match':r['unmasked_component_match_status'],'extent':r.get('full_component_extent_status'),'fraction_inside':r.get('full_component_fraction_inside_final_well')} for wid,r in known.items()}
    qc_ok=len(completed)==len(diagnostics) and (condition_id!=DMSO_CONDITION or set(known)=={'606','624'})
    result={
        'qc_version':QC_VERSION,'completion_status':'completed' if qc_ok else 'failed_qc','condition_id':condition_id,'completed_at':_now(),
        'omezarr_source':str(zarr_path),'omezarr_validation':validation,'gfp_quantitative_channel':CHANNELS['gfp'],'gfp_uint8_window_end':maximum,'gfp_uint8_window_source':window_source,
        'scientific_settings':{'green_low':GREEN_LOW,'green_high':GREEN_HIGH,'pdo_min_area':minimum_area,'gaussian_sigma_px':GAUSSIAN_SIGMA_PX,'split_pdos':False,'production_interior_radius_fraction':PRODUCTION_INTERIOR_FRACTION},
        'crop_expansion':{'radius_factors':EXPANSION_FACTORS,'maximum_half_width_px':MAX_CROP_HALF_WIDTH_PX,'edge_guard_px':CROP_EDGE_GUARD_PX},
        'component_hash_provenance':'SHA-256 of condition_id, full-resolution XYXY-exclusive bounding box, NUL delimiter, and row-major big-bitorder packed binary bounding-box mask.',
        'containment_geometry_provenance':GEOMETRY_PROVENANCE,'containment_qc_status':QC_STATUS,'exclusion_rule':None,
        'production_PDO_rows':len(pdos),'trusted_complete_matches':sum(r.get('unmasked_component_match_status')=='trusted_complete_match' for r in rows),'unique_complete_components':len(components),
        'known_visual_failure_results':known_status,'diagnostics_expected':len(diagnostics),'diagnostics_exported':len(completed),'contact_sheets':sheets,
        'source_files':['well_measurements.csv','pdo_measurements.csv','condition_summary.json','batch_status.json where available','validated OME-Zarr GFP/DIC channels'],
        'outputs':['pdo_full_component_measurements.csv','unmasked_component_summary.csv','diagnostic_manifest.csv','labelled_diagnostics','contact_sheets'],
    }
    _atomic_json(output/'full_component_qc_summary.json',result); return result


def combine_outputs(result_root: Path) -> None:
    pdos=[];components=[]
    for condition_id in CONDITIONS:
        out=result_root/condition_id/OUTPUT_DIRECTORY; summary=out/'full_component_qc_summary.json'
        if summary.is_file() and _read_json(summary).get('completion_status')=='completed':
            pdos.extend(_read_csv(out/'pdo_full_component_measurements.csv')); components.extend(_read_csv(out/'unmasked_component_summary.csv'))
    _atomic_csv(result_root/'all_conditions_pdo_full_component_measurements.csv',pdos,PDO_FIELDS)
    _atomic_csv(result_root/'all_conditions_unmasked_component_summary.csv',components,COMPONENT_FIELDS)


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description='Diagnostic true unmasked GFP-component containment replay; no exclusion rule.')
    parser.add_argument('--result-root',type=Path,required=True); parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--condition-id',action='append',choices=tuple(CONDITIONS),default=[])
    parser.add_argument('--expected-pixel-size-um',type=float,default=EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--panel-size',type=int,default=384); parser.add_argument('--contact-sheet-size',type=int,default=12)
    return parser


def run(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
        open_group: Callable = zarr.open_group) -> int:
    if args.panel_size<64 or args.contact_sheet_size<1: raise ValueError('Invalid diagnostic presentation settings.')
    root=args.result_root.expanduser().resolve(); batch_path=root/'batch_status.json'; batch=_read_json(batch_path) if batch_path.is_file() else {}
    selected=args.condition_id or list(CONDITIONS); failures=0
    for condition_id in selected:
        try:
            result=export_condition(condition_id,root/condition_id,args,batch,probe=probe,open_group=open_group)
            if result['completion_status']!='completed': failures+=1
            print(f"{condition_id}: {result['completion_status']} ({result['production_PDO_rows']} production PDOs; {result['trusted_complete_matches']} trusted complete matches; {result['unique_complete_components']} unique components)",flush=True)
        except Exception as exc:
            failures+=1; out=root/condition_id/OUTPUT_DIRECTORY
            _atomic_json(out/'full_component_qc_summary.json',{'qc_version':QC_VERSION,'completion_status':'failed','condition_id':condition_id,'failed_at':_now(),'containment_geometry_provenance':GEOMETRY_PROVENANCE,'containment_qc_status':QC_STATUS,'exclusion_rule':None,'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
            print(f'{condition_id}: FAILED: {type(exc).__name__}: {exc}',flush=True)
        finally: combine_outputs(root)
    if not failures:
        print(f'PDO full-component diagnostic QC completed: {len(selected)}/{len(selected)} conditions; true unmasked GFP-component masks measured; no exclusion rule applied.',flush=True)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__=='__main__':
    raise SystemExit(main())
