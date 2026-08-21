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
from PIL import Image, ImageDraw

import aws_export_pdo_positive_crops as crop_base
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


QC_VERSION = 1
OUTPUT_DIRECTORY = 'pdo_containment_qc'
GEOMETRY_PROVENANCE = 'equivalent_circular_diameter_reconstructed_circle'
QC_STATUS = 'diagnostic_measurement_only_no_exclusion_rule'
EXPECTED_PIXEL_SIZE_UM = crop_base.EXPECTED_PIXEL_SIZE_UM
CHANNELS = {'gfp': 0, 'dic': 2}
CONDITIONS = dict(crop_base.CONDITIONS)
DMSO_CONDITION = 'K3T_PSC_RMC6236_Lane_1_DMSO'
KNOWN_VISUAL_FAILURE_WELLS = {(DMSO_CONDITION, '606'), (DMSO_CONDITION, '624')}

OBJECT_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'pdo_number_in_well',
    'known_visual_failure', 'hex_array_member', 'lattice_degree',
    'well_x_px_fullres', 'well_y_px_fullres', 'well_radius_px', 'pixel_size_um',
    'PDO_centroid_x_px_fullres', 'PDO_centroid_y_px_fullres',
    'PDO_centroid_dx_px', 'PDO_centroid_dy_px',
    'PDO_centroid_distance_px', 'PDO_centroid_distance_um',
    'normalized_PDO_centroid_radius', 'PDO_centroid_inside_well',
    'PDO_projected_area_px2', 'PDO_projected_area_um2',
    'PDO_equivalent_circular_diameter_um',
    'reconstructed_PDO_radius_px', 'reconstructed_PDO_radius_um',
    'reconstructed_PDO_area_px2', 'reconstructed_PDO_area_um2',
    'PDO_well_overlap_area_px2', 'PDO_well_overlap_area_um2',
    'PDO_fraction_inside_well', 'PDO_fraction_outside_well',
    'PDO_boundary_intersection', 'PDO_edge_clearance_px', 'PDO_edge_clearance_um',
    'normalized_PDO_edge_clearance', 'containment_geometry_provenance',
    'containment_qc_status',
)

WELL_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'well_id', 'known_visual_failure',
    'hex_array_member', 'lattice_degree', 'well_x_px_fullres', 'well_y_px_fullres',
    'well_radius_px', 'pixel_size_um', 'PDO_count',
    'total_PDO_projected_area_px2', 'total_PDO_projected_area_um2',
    'minimum_PDO_diameter_um', 'maximum_PDO_diameter_um',
    'minimum_PDO_fraction_inside_well', 'maximum_PDO_fraction_outside_well',
    'maximum_normalized_PDO_centroid_radius', 'any_PDO_centroid_outside_well',
    'any_PDO_boundary_intersection', 'minimum_PDO_edge_clearance_px',
    'minimum_PDO_edge_clearance_um', 'minimum_normalized_PDO_edge_clearance',
    'diagnostic_selection', 'diagnostic_sampling_reasons',
    'containment_geometry_provenance', 'containment_qc_status',
)

DIAGNOSTIC_FIELDS = (*OBJECT_FIELDS, 'diagnostic_sampling_reasons', 'restart_signature',
                     'labelled_diagnostic', 'export_status', 'error')


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


def _normalise_well_id(value: object) -> str:
    return crop_base._normalise_well_id(value)


def _number(row: dict, key: str) -> float:
    value = crop_base._number(row, key)
    if not math.isfinite(value):
        raise RuntimeError(f"Field '{key}' must be finite for containment geometry.")
    return value


def _object_key(row: dict) -> tuple[str, int]:
    return (_normalise_well_id(row['well_id']), int(round(float(row['pdo_number_in_well']))))


def circle_overlap_area(well_radius: float, pdo_radius: float, distance: float) -> float:
    R, r, d = float(well_radius), float(pdo_radius), float(distance)
    if not all(math.isfinite(value) for value in (R, r, d)) or R <= 0 or r <= 0 or d < 0:
        raise ValueError(f'Invalid circle geometry R={R}, r={r}, d={d}.')
    if d >= R + r:
        return 0.0
    if d <= abs(R - r):
        return math.pi * min(R, r) ** 2
    first_argument = (d * d + r * r - R * R) / (2.0 * d * r)
    second_argument = (d * d + R * R - r * r) / (2.0 * d * R)

    def numerical_acos(value: float) -> float:
        if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
            raise RuntimeError(f'Numerically invalid circle-overlap acos argument {value}.')
        if value < -1.0:
            value = -1.0
        elif value > 1.0:
            value = 1.0
        return math.acos(value)

    radical = (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)
    if radical < -1e-9:
        raise RuntimeError(f'Numerically invalid circle-overlap radical {radical}.')
    radical = 0.0 if radical < 0.0 else radical
    return (r * r * numerical_acos(first_argument)
            + R * R * numerical_acos(second_argument)
            - 0.5 * math.sqrt(radical))


def containment_geometry(well_radius_px: float, pdo_radius_px: float,
                         dx_px: float, dy_px: float, pixel_size_um: float) -> dict:
    R, r, scale = float(well_radius_px), float(pdo_radius_px), float(pixel_size_um)
    if R <= 0 or r <= 0 or scale <= 0:
        raise ValueError(f'Radii and calibration must be positive: R={R}, r={r}, scale={scale}.')
    d = math.hypot(float(dx_px), float(dy_px))
    overlap = circle_overlap_area(R, r, d)
    reconstructed_area = math.pi * r * r
    fraction_inside = overlap / reconstructed_area
    # Fractions alone are clamped to protect against floating-point rounding.
    fraction_inside = min(1.0, max(0.0, fraction_inside))
    fraction_outside = min(1.0, max(0.0, 1.0 - fraction_inside))
    clearance = R - (d + r)
    return {
        'PDO_centroid_distance_px': d,
        'PDO_centroid_distance_um': d * scale,
        'normalized_PDO_centroid_radius': d / R,
        'PDO_centroid_inside_well': d <= R,
        'reconstructed_PDO_radius_px': r,
        'reconstructed_PDO_radius_um': r * scale,
        'reconstructed_PDO_area_px2': reconstructed_area,
        'reconstructed_PDO_area_um2': reconstructed_area * scale * scale,
        'PDO_well_overlap_area_px2': overlap,
        'PDO_well_overlap_area_um2': overlap * scale * scale,
        'PDO_fraction_inside_well': fraction_inside,
        'PDO_fraction_outside_well': fraction_outside,
        'PDO_boundary_intersection': abs(R - r) <= d <= R + r,
        'PDO_edge_clearance_px': clearance,
        'PDO_edge_clearance_um': clearance * scale,
        'normalized_PDO_edge_clearance': clearance / R,
    }


def calculate_object_rows(condition_id: str, condition_name: str, dose: str,
                          wells: list[dict], pdos: list[dict], pixel_size_um: float) -> list[dict]:
    well_by_id = {}
    for well in wells:
        well_id = _normalise_well_id(well.get('well_id'))
        if well_id in well_by_id:
            raise RuntimeError(f'Duplicate final well_id {well_id}.')
        if 'hex_array_member' not in well or not str(well.get('hex_array_member', '')).strip():
            raise RuntimeError(f'Final well {well_id} lacks required hex_array_member provenance.')
        if not _truthy(well['hex_array_member']):
            raise RuntimeError(f'Final well {well_id} is not a truthy hex_array_member.')
        well_by_id[well_id] = well
    grouped: dict[str, list[dict]] = {}
    seen_objects = set()
    output = []
    for pdo in pdos:
        well_id = _normalise_well_id(pdo.get('well_id'))
        if well_id not in well_by_id:
            raise RuntimeError(f'PDO row refers to non-final well_id {well_id}.')
        pdo_number = int(round(_number(pdo, 'pdo_number_in_well')))
        key = (well_id, pdo_number)
        if key in seen_objects:
            raise RuntimeError(f'Duplicate PDO key well={well_id}, PDO={pdo_number}.')
        seen_objects.add(key)
        grouped.setdefault(well_id, []).append(pdo)
        well = well_by_id[well_id]
        wx, wy = _number(well, 'x_px_fullres'), _number(well, 'y_px_fullres')
        px = _number(pdo, 'centroid_x_px_fullres')
        py = _number(pdo, 'centroid_y_px_fullres')
        diameter_um = _number(pdo, 'equivalent_circular_diameter_um')
        if diameter_um <= 0:
            raise RuntimeError(f'PDO diameter must be positive for well {well_id}, PDO {pdo_number}.')
        radius_px = (diameter_um / 2.0) / pixel_size_um
        geometry = containment_geometry(
            _number(well, 'radius_px'), radius_px, px - wx, py - wy, pixel_size_um
        )
        output.append({
            'condition_id': condition_id, 'condition_name': condition_name, 'dose': dose,
            'well_id': well_id, 'pdo_number_in_well': pdo_number,
            'known_visual_failure': (condition_id, well_id) in KNOWN_VISUAL_FAILURE_WELLS,
            'hex_array_member': True, 'lattice_degree': well.get('lattice_degree', ''),
            'well_x_px_fullres': wx, 'well_y_px_fullres': wy,
            'well_radius_px': _number(well, 'radius_px'), 'pixel_size_um': pixel_size_um,
            'PDO_centroid_x_px_fullres': px, 'PDO_centroid_y_px_fullres': py,
            'PDO_centroid_dx_px': px - wx, 'PDO_centroid_dy_px': py - wy,
            'PDO_projected_area_px2': pdo.get('projected_area_px2', ''),
            'PDO_projected_area_um2': pdo.get('projected_area_um2', ''),
            'PDO_equivalent_circular_diameter_um': diameter_um,
            **geometry, 'containment_geometry_provenance': GEOMETRY_PROVENANCE,
            'containment_qc_status': QC_STATUS,
        })
    for well_id, well in well_by_id.items():
        expected = int(round(_number(well, 'PDO_count')))
        actual = len(grouped.get(well_id, []))
        if actual != expected:
            raise RuntimeError(
                f'Final PDO count mismatch for well {well_id}: well CSV={expected}, PDO rows={actual}.'
            )
    return sorted(output, key=lambda row: (_number(row, 'well_y_px_fullres'),
                                           _number(row, 'well_x_px_fullres'),
                                           int(row['pdo_number_in_well'])))


def aggregate_wells(condition_id: str, condition_name: str, dose: str,
                    wells: list[dict], objects: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in objects:
        grouped.setdefault(_normalise_well_id(row['well_id']), []).append(row)
    output = []
    for well in wells:
        well_id = _normalise_well_id(well['well_id'])
        rows = grouped.get(well_id, [])
        if not rows:
            continue
        output.append({
            'condition_id': condition_id, 'condition_name': condition_name, 'dose': dose,
            'well_id': well_id,
            'known_visual_failure': (condition_id, well_id) in KNOWN_VISUAL_FAILURE_WELLS,
            'hex_array_member': True, 'lattice_degree': well.get('lattice_degree', ''),
            'well_x_px_fullres': well['x_px_fullres'], 'well_y_px_fullres': well['y_px_fullres'],
            'well_radius_px': well['radius_px'], 'pixel_size_um': rows[0]['pixel_size_um'],
            'PDO_count': len(rows),
            'total_PDO_projected_area_px2': well.get('total_PDO_projected_area_px2', ''),
            'total_PDO_projected_area_um2': well.get('total_PDO_projected_area_um2', ''),
            'minimum_PDO_diameter_um': min(_number(row, 'PDO_equivalent_circular_diameter_um')
                                           for row in rows),
            'maximum_PDO_diameter_um': max(_number(row, 'PDO_equivalent_circular_diameter_um')
                                           for row in rows),
            'minimum_PDO_fraction_inside_well': min(_number(row, 'PDO_fraction_inside_well')
                                                    for row in rows),
            'maximum_PDO_fraction_outside_well': max(_number(row, 'PDO_fraction_outside_well')
                                                    for row in rows),
            'maximum_normalized_PDO_centroid_radius': max(
                _number(row, 'normalized_PDO_centroid_radius') for row in rows),
            'any_PDO_centroid_outside_well': any(not _truthy(row['PDO_centroid_inside_well'])
                                                 for row in rows),
            'any_PDO_boundary_intersection': any(_truthy(row['PDO_boundary_intersection'])
                                                 for row in rows),
            'minimum_PDO_edge_clearance_px': min(_number(row, 'PDO_edge_clearance_px')
                                                 for row in rows),
            'minimum_PDO_edge_clearance_um': min(_number(row, 'PDO_edge_clearance_um')
                                                 for row in rows),
            'minimum_normalized_PDO_edge_clearance': min(
                _number(row, 'normalized_PDO_edge_clearance') for row in rows),
            'diagnostic_selection': False, 'diagnostic_sampling_reasons': '',
            'containment_geometry_provenance': GEOMETRY_PROVENANCE,
            'containment_qc_status': QC_STATUS,
        })
    return output


def select_diagnostics(condition_id: str, objects: list[dict]) -> list[dict]:
    ordinary = [row for row in objects
                if (condition_id, _normalise_well_id(row['well_id']))
                not in KNOWN_VISUAL_FAILURE_WELLS]
    selected: dict[tuple[str, int], dict] = {}
    reasons: dict[tuple[str, int], set[str]] = {}

    def add(rows: list[dict], reason: str, limit: int | None = None) -> None:
        for row in rows[:limit]:
            key = _object_key(row)
            selected[key] = row
            reasons.setdefault(key, set()).add(reason)

    tie = lambda row: (_normalise_well_id(row['well_id']), int(row['pdo_number_in_well']))
    add(sorted(ordinary, key=lambda row: (_number(row, 'PDO_fraction_inside_well'), tie(row))),
        'lowest_fraction_inside', 3)
    negative = [row for row in ordinary if _number(row, 'PDO_edge_clearance_px') < 0]
    add(sorted(negative, key=lambda row: (_number(row, 'PDO_edge_clearance_px'), tie(row))),
        'most_negative_edge_clearance', 3)
    outside = [row for row in ordinary if not _truthy(row['PDO_centroid_inside_well'])]
    add(sorted(outside, key=lambda row: (-_number(row, 'normalized_PDO_centroid_radius'), tie(row))),
        'centroid_outside', 2)
    intersecting = [row for row in ordinary if _truthy(row['PDO_boundary_intersection'])]
    add(sorted(intersecting,
               key=lambda row: (-_number(row, 'PDO_fraction_inside_well'), tie(row))),
        'boundary_intersection_nearest_full_containment', 2)
    contained = [row for row in ordinary
                 if _number(row, 'PDO_fraction_inside_well') >= 1.0
                 and _number(row, 'PDO_edge_clearance_px') >= 0]
    add(sorted(contained, key=lambda row: (_number(row, 'PDO_edge_clearance_px'), tie(row))),
        'fully_contained_near_wall_control', 1)
    add(sorted(contained,
               key=lambda row: (_number(row, 'normalized_PDO_centroid_radius'), tie(row))),
        'fully_contained_central_control', 1)
    add(sorted(ordinary,
               key=lambda row: (_number(row, 'PDO_equivalent_circular_diameter_um'), tie(row))),
        'smallest_PDO', 1)
    add(sorted(ordinary,
               key=lambda row: (-_number(row, 'PDO_equivalent_circular_diameter_um'), tie(row))),
        'largest_PDO', 1)
    counts: dict[str, int] = {}
    for row in ordinary:
        well_id = _normalise_well_id(row['well_id'])
        counts[well_id] = counts.get(well_id, 0) + 1
    multi_wells = sorted(well_id for well_id, count in counts.items() if count > 1)
    if multi_wells:
        add([row for row in ordinary if _normalise_well_id(row['well_id']) == multi_wells[0]],
            'multi_PDO_well')

    forced = [row for row in objects
              if (condition_id, _normalise_well_id(row['well_id']))
              in KNOWN_VISUAL_FAILURE_WELLS]
    if condition_id == DMSO_CONDITION:
        present = {_normalise_well_id(row['well_id']) for row in forced}
        required = {well_id for candidate_condition, well_id in KNOWN_VISUAL_FAILURE_WELLS
                    if candidate_condition == condition_id}
        if present != required:
            raise RuntimeError(
                f'Mandatory known visual failure PDO wells missing: {sorted(required-present)}.'
            )
    add(forced, 'known_visual_failure_mandatory')
    output = []
    for key in sorted(selected, key=lambda value: (int(value[0]) if value[0].isdigit() else value[0],
                                                   value[1])):
        row = dict(selected[key])
        row['diagnostic_sampling_reasons'] = ';'.join(sorted(reasons[key]))
        output.append(row)
    return output


def apply_diagnostic_selection(well_rows: list[dict], diagnostics: list[dict]) -> None:
    reasons: dict[str, set[str]] = {}
    for row in diagnostics:
        reasons.setdefault(_normalise_well_id(row['well_id']), set()).update(
            str(row['diagnostic_sampling_reasons']).split(';'))
    for row in well_rows:
        well_id = _normalise_well_id(row['well_id'])
        row['diagnostic_selection'] = well_id in reasons
        row['diagnostic_sampling_reasons'] = ';'.join(sorted(reasons.get(well_id, set())))


def _display_images(dic: np.ndarray, gfp: np.ndarray, gfp_range: dict) -> dict[str, Image.Image]:
    dic_u8 = crop_base._u8_local(dic)
    gfp_u8 = crop_base._u8_range(gfp, gfp_range['minimum'], gfp_range['maximum'])
    dic_rgb = np.stack([dic_u8, dic_u8, dic_u8], axis=-1)
    gfp_rgb = np.zeros_like(dic_rgb)
    gfp_rgb[..., 1] = gfp_u8
    composite = dic_rgb.copy()
    composite[..., 1] = np.maximum(composite[..., 1], gfp_u8)
    return {key: Image.fromarray(value) for key, value in
            (('dic', dic_rgb), ('gfp', gfp_rgb), ('composite', composite))}


def _overlay(image: Image.Image, *, well: dict, objects: list[dict], left: int, top: int,
             panel_size: int) -> Image.Image:
    output = image.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(output)
    scale = panel_size / image.width
    wx = (_number(well, 'x_px_fullres') - left) * scale
    wy = (_number(well, 'y_px_fullres') - top) * scale
    R = _number(well, 'radius_px') * scale
    width = max(2, panel_size // 180)
    draw.ellipse((wx - R, wy - R, wx + R, wy + R), outline=(255, 255, 0), width=width)
    _, font = crop_base._fonts(body_size=max(12, panel_size // 24))
    for row in objects:
        px = (_number(row, 'PDO_centroid_x_px_fullres') - left) * scale
        py = (_number(row, 'PDO_centroid_y_px_fullres') - top) * scale
        radius = _number(row, 'reconstructed_PDO_radius_px') * scale
        draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                     outline=(0, 255, 255), width=width)
        arm = max(3, panel_size // 100)
        draw.line((px - arm, py, px + arm, py), fill=(255, 0, 255), width=width)
        draw.line((px, py - arm, px, py + arm), fill=(255, 0, 255), width=width)
        draw.text((px + arm + 2, py - arm - 2), str(row['pdo_number_in_well']),
                  fill=(255, 0, 255), font=font)
    return output


def diagnostic_header(row: dict) -> list[str]:
    failure = ' | KNOWN VISUAL FAILURE' if _truthy(row['known_visual_failure']) else ''
    return [
        f"{row['condition_name']} | {row['dose']} | Final well {row['well_id']} | "
        f"Focal PDO {row['pdo_number_in_well']}{failure}",
        'Containment geometry: reconstructed equivalent-area circle',
        f"PDO diameter: {_number(row, 'PDO_equivalent_circular_diameter_um'):.2f} µm | "
        f"Centroid distance: {_number(row, 'PDO_centroid_distance_px'):.2f} px / "
        f"{_number(row, 'PDO_centroid_distance_um'):.2f} µm",
        f"Normalized centroid radius: {_number(row, 'normalized_PDO_centroid_radius'):.4f}",
        f"Reconstructed-circle fraction inside: {_number(row, 'PDO_fraction_inside_well'):.4f} | "
        f"outside: {_number(row, 'PDO_fraction_outside_well'):.4f}",
        f"Edge clearance: {_number(row, 'PDO_edge_clearance_um'):.2f} µm | "
        f"Normalized edge clearance: {_number(row, 'normalized_PDO_edge_clearance'):.4f}",
        f"Centroid inside: {row['PDO_centroid_inside_well']} | "
        f"Boundary intersection: {row['PDO_boundary_intersection']}",
        f"Sampling: {row['diagnostic_sampling_reasons']}",
    ]


def labelled_diagnostic(images: dict[str, Image.Image], *, row: dict, well: dict,
                        objects: list[dict], left: int, top: int, panel_size: int) -> Image.Image:
    gap, title_height = 8, 28
    width = 3 * panel_size + 4 * gap
    title_font, body_font = crop_base._fonts()
    wrapped = []
    for line in diagnostic_header(row):
        wrapped.extend(textwrap.wrap(line, width=max(70, width // 9),
                                     break_long_words=False) or [''])
    header_height = 14 + 30 + max(0, len(wrapped) - 1) * 23 + 10
    canvas = Image.new('RGB', (width, header_height + panel_size + title_height + 2 * gap), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, header_height), fill='black')
    y = 8
    for index, line in enumerate(wrapped):
        draw.text((12, y), line, fill='white', font=title_font if index == 0 else body_font)
        y += 30 if index == 0 else 23
    for index, (kind, label) in enumerate((('dic', 'DIC'), ('gfp', 'GFP'),
                                           ('composite', 'Composite'))):
        x = gap + index * (panel_size + gap)
        panel_y = header_height + gap
        draw.rectangle((x, panel_y, x + panel_size, panel_y + title_height), fill='black')
        draw.text((x + 7, panel_y + 5), label, fill='white', font=body_font)
        panel = _overlay(images[kind], well=well, objects=objects, left=left, top=top,
                         panel_size=panel_size)
        canvas.paste(panel, (x, panel_y + title_height))
        panel.close()
    return canvas


def export_condition(condition_id: str, folder: Path, args: argparse.Namespace,
                     batch_status: dict, *, probe: Callable = probe_omezarr,
                     open_group: Callable = zarr.open_group) -> dict:
    condition_summary_path = folder / 'condition_summary.json'
    well_path = folder / 'well_measurements.csv'
    pdo_path = folder / 'pdo_measurements.csv'
    for path in (condition_summary_path, well_path, pdo_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required completed source is missing: {path}')
    summary = _read_json(condition_summary_path)
    wells, pdos = _read_csv(well_path), _read_csv(pdo_path)
    zarr_path = crop_base.resolve_omezarr(condition_id, summary, batch_status, args.cache_root)
    metadata = probe(zarr_path)
    validation = crop_base.validate_omezarr(metadata, summary, args.expected_pixel_size_um)
    px_x, px_y = validation['pixel_size_um']['x'], validation['pixel_size_um']['y']
    if not math.isclose(px_x, px_y, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f'Containment circle geometry requires isotropic pixels: x={px_x}, y={px_y}.')
    pixel_size_um = (px_x + px_y) / 2.0
    mapping = CONDITIONS[condition_id]
    condition_name = mapping['condition_name']
    objects = calculate_object_rows(
        condition_id, condition_name, mapping['dose'], wells, pdos, pixel_size_um
    )
    well_rows = aggregate_wells(condition_id, condition_name, mapping['dose'], wells, objects)
    diagnostics = select_diagnostics(condition_id, objects)
    apply_diagnostic_selection(well_rows, diagnostics)
    output = folder / OUTPUT_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output / 'pdo_containment_measurements.csv', objects, OBJECT_FIELDS)
    _atomic_csv(output / 'well_containment_summary.csv', well_rows, WELL_FIELDS)

    root = open_group(str(zarr_path), mode='r')
    array = root[metadata['level0_array_path']]
    planes = SingletonTZCYX(array, metadata['axes'])
    channels, height, width = planes.shape_cyx
    if channels != 3:
        raise RuntimeError(f'Validated metadata and opened array disagree: {channels} channels.')
    gfp_window = crop_base._metadata_window(metadata, CHANNELS['gfp'])
    if gfp_window is None:
        gfp_window = crop_base._condition_display_range(
            planes, CHANNELS['gfp'], width, height,
            args.display_sample_size, args.display_sample_grid,
        )
        gfp_source = 'condition_wide_sample_percentiles_0.5_99.5'
    else:
        gfp_source = 'ome_omero_channel_window'
    gfp_range = {'minimum': gfp_window[0], 'maximum': gfp_window[1], 'source': gfp_source}
    well_by_id = {_normalise_well_id(row['well_id']): row for row in wells}
    objects_by_well: dict[str, list[dict]] = {}
    for row in objects:
        objects_by_well.setdefault(_normalise_well_id(row['well_id']), []).append(row)
    manifest_path = output / 'diagnostic_manifest.csv'
    prior = {_object_key(row): row for row in _read_csv(manifest_path)} \
        if manifest_path.is_file() else {}
    manifest = []
    for index, diagnostic in enumerate(diagnostics):
        key = _object_key(diagnostic)
        well = well_by_id[key[0]]
        filename = (f'{condition_id}__well_{int(key[0]):06d}__PDO_{key[1]:02d}.png'
                    if key[0].isdigit() else f'{condition_id}__well_{key[0]}__PDO_{key[1]:02d}.png')
        path = output / 'labelled_diagnostics' / filename
        signature = crop_base._signature({
            'version': QC_VERSION, 'diagnostic': diagnostic,
            'well_objects': objects_by_well[key[0]], 'omezarr': str(zarr_path),
            'display': gfp_range, 'crop_radius_scale': args.crop_radius_scale,
            'panel_size': args.panel_size,
        })
        old = prior.get(key)
        if (old and old.get('export_status') == 'completed'
                and old.get('restart_signature') == signature
                and Path(old.get('labelled_diagnostic', '')).is_file()):
            manifest.append(old)
        else:
            row = dict(diagnostic)
            try:
                wx, wy = _number(well, 'x_px_fullres'), _number(well, 'y_px_fullres')
                half = max(1, int(round(_number(well, 'radius_px') * args.crop_radius_scale)))
                dic, left, top = crop_base._read_padded(
                    planes, CHANNELS['dic'], wx, wy, half, width, height
                )
                gfp, _, _ = crop_base._read_padded(
                    planes, CHANNELS['gfp'], wx, wy, half, width, height
                )
                images = _display_images(dic, gfp, gfp_range)
                labelled = labelled_diagnostic(
                    images, row=diagnostic, well=well, objects=objects_by_well[key[0]],
                    left=left, top=top, panel_size=args.panel_size,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                labelled.save(path, dpi=(300, 300))
                labelled.close()
                for image in images.values():
                    image.close()
                row.update({'restart_signature': signature, 'labelled_diagnostic': str(path),
                            'export_status': 'completed', 'error': ''})
            except Exception as exc:
                row.update({'restart_signature': signature, 'labelled_diagnostic': str(path),
                            'export_status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            manifest.append(row)
        future = [_object_key(row) for row in diagnostics[index + 1:]]
        _atomic_csv(manifest_path, manifest + [prior[key] for key in future if key in prior],
                    DIAGNOSTIC_FIELDS)
    expected_keys = {_object_key(row) for row in diagnostics}
    completed_keys = {_object_key(row) for row in manifest
                      if row.get('export_status') == 'completed'
                      and Path(row.get('labelled_diagnostic', '')).is_file()}
    contact_sheets = crop_base._contact_sheets(
        [Path(row['labelled_diagnostic']) for row in manifest
         if row.get('export_status') == 'completed'],
        output / 'contact_sheets', args.contact_sheet_size,
    )
    qc_ok = completed_keys == expected_keys and len(manifest) == len(expected_keys)
    result = {
        'qc_version': QC_VERSION, 'completion_status': 'completed' if qc_ok else 'failed_qc',
        'condition_id': condition_id, 'condition_name': condition_name, 'dose': mapping['dose'],
        'completed_at': _now(), 'omezarr_source': str(zarr_path),
        'omezarr_validation': validation, 'pixel_size_um': pixel_size_um,
        'containment_geometry_provenance': GEOMETRY_PROVENANCE,
        'geometry_notice': ('All containment values describe the reconstructed equivalent-area '
                            'PDO circle, not the original segmented PDO boundary.'),
        'exclusion_rule': None,
        'containment_qc_status': QC_STATUS,
        'PDO_objects_measured': len(objects), 'PDO_positive_wells_measured': len(well_rows),
        'diagnostic_objects_expected': len(expected_keys),
        'diagnostic_objects_exported': len(completed_keys),
        'diagnostic_object_set_qc_passed': completed_keys == expected_keys,
        'known_visual_failure_wells': sorted(
            {row['well_id'] for row in diagnostics if _truthy(row['known_visual_failure'])}),
        'gfp_display_range': gfp_range,
        'diagnostic_crop_radius_scale': args.crop_radius_scale,
        'contact_sheets': contact_sheets,
        'source_files': ['well_measurements.csv', 'pdo_measurements.csv'],
        'outputs': ['pdo_containment_measurements.csv', 'well_containment_summary.csv',
                    'diagnostic_manifest.csv', 'labelled_diagnostics', 'contact_sheets'],
    }
    _atomic_json(output / 'containment_qc_summary.json', result)
    return result


def combine_outputs(result_root: Path) -> None:
    object_rows, well_rows = [], []
    for condition_id in CONDITIONS:
        output = result_root / condition_id / OUTPUT_DIRECTORY
        summary_path = output / 'containment_qc_summary.json'
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        if summary.get('completion_status') != 'completed':
            continue
        object_rows.extend(_read_csv(output / 'pdo_containment_measurements.csv'))
        well_rows.extend(_read_csv(output / 'well_containment_summary.csv'))
    _atomic_csv(result_root / 'all_conditions_pdo_containment_measurements.csv',
                object_rows, OBJECT_FIELDS)
    _atomic_csv(result_root / 'all_conditions_pdo_containment_well_summary.csv',
                well_rows, WELL_FIELDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Diagnostic reconstructed-circle PDO containment QC; no exclusion rule.'
    )
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--condition-id', action='append', choices=tuple(CONDITIONS), default=[])
    parser.add_argument('--expected-pixel-size-um', type=float, default=EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--crop-radius-scale', type=float, default=2.25)
    parser.add_argument('--panel-size', type=int, default=384)
    parser.add_argument('--contact-sheet-size', type=int, default=12)
    parser.add_argument('--display-sample-size', type=int, default=256)
    parser.add_argument('--display-sample-grid', type=int, default=4)
    return parser


def run(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
        open_group: Callable = zarr.open_group) -> int:
    if args.crop_radius_scale < 1.5 or args.panel_size < 64 or args.contact_sheet_size < 1:
        raise ValueError('Diagnostic crop margin, panel size, and contact-sheet size are invalid.')
    result_root = args.result_root.expanduser().resolve()
    batch_path = result_root / 'batch_status.json'
    batch_status = _read_json(batch_path) if batch_path.is_file() else {}
    selected = args.condition_id or list(CONDITIONS)
    failures = 0
    for condition_id in selected:
        try:
            summary = export_condition(
                condition_id, result_root / condition_id, args, batch_status,
                probe=probe, open_group=open_group,
            )
            if summary['completion_status'] != 'completed':
                failures += 1
            print(f"{condition_id}: {summary['completion_status']} "
                  f"({summary['PDO_objects_measured']} PDO objects; "
                  f"{summary['diagnostic_objects_exported']} diagnostics)", flush=True)
        except Exception as exc:
            failures += 1
            output = result_root / condition_id / OUTPUT_DIRECTORY
            _atomic_json(output / 'containment_qc_summary.json', {
                'qc_version': QC_VERSION, 'completion_status': 'failed',
                'condition_id': condition_id, 'failed_at': _now(),
                'containment_geometry_provenance': GEOMETRY_PROVENANCE,
                'exclusion_rule': None, 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            })
            print(f'{condition_id}: FAILED: {type(exc).__name__}: {exc}', flush=True)
        finally:
            combine_outputs(result_root)
    if not failures:
        print(f'PDO containment diagnostic QC completed: {len(selected)}/{len(selected)} '
              'conditions; no exclusion rule applied.', flush=True)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
