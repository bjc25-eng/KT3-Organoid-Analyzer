#!/usr/bin/env python3
"""Apply the approved final PDO-object QC rule and export presentation crops.

This is a table finalizer and presentation-only crop exporter.  It consumes the
completed full-component QC, production well/PDO tables, continuous RFP table,
and validated OME-Zarr.  It never performs segmentation or quantification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw
import zarr

import aws_export_final_pdo_rfp_crops as presentation
import aws_export_pdo_positive_crops as crop_base
from omezarr_cyx import SingletonTZCYX


FINALIZATION_VERSION = 'final-pdo-qc-v1'
OUTPUT_DIRECTORY = 'final_analysis_qc'
AUTHORITATIVE_COMPONENT_CSV = 'all_conditions_pdo_full_component_measurements.csv'
EXPECTED_AUTHORITATIVE_PDO_ROWS = 2568
FRACTION_INSIDE_THRESHOLD = 0.60
FINAL_ARRAY_QC_STATUS = 'final_dominant_hex_array_accepted'
PSC_CELL_COUNT_STATUS = 'NOT VALIDATED'

FAILURE_REASONS = (
    'containment_below_0p60',
    'full_component_centroid_outside_well',
    'many_to_one_unmasked_component',
    'untrusted_component_match',
    'incomplete_component_extent',
)
KNOWN_FAILURES = (
    ('K3T_PSC_RMC6236_Lane_1_DMSO', '606'),
    ('K3T_PSC_RMC6236_Lane_1_DMSO', '624'),
)

ORIGINAL_AUDIT_FIELDS = (
    'original_PDO_present', 'original_PDO_count',
    'original_total_PDO_projected_area_px2',
    'original_total_PDO_projected_area_um2',
)
FINAL_WELL_EXTRA_FIELDS = (
    *ORIGINAL_AUDIT_FIELDS,
    'retained_PDO_numbers', 'retained_PDO_equivalent_circular_diameters_um',
    'final_array_qc_status', 'PSC_cell_count_status',
)
OBJECT_EXTRA_FIELDS = ('final_PDO_QC_pass', 'final_PDO_QC_failure_reasons')
SUMMARY_FIELDS = (
    'condition_id', 'condition_name', 'dose', 'dose_nM',
    'total_final_dominant_array_wells', 'original_PDO_positive_wells',
    'original_PDO_objects', 'retained_PDO_positive_wells', 'retained_PDO_objects',
    'PDO_objects_removed', 'wells_converted_PDO_positive_to_negative',
    'final_PDO_positive_fraction', 'removed_containment_below_0p60',
    'removed_full_component_centroid_outside_well',
    'removed_many_to_one_unmasked_component', 'removed_untrusted_component_match',
    'removed_incomplete_component_extent', 'retained_multi_PDO_wells',
)
CROP_EXTRA_FIELDS = ('retained_PDO_identities', 'final_PDO_QC_rule')
CROP_MANIFEST_FIELDS = (*presentation.MANIFEST_FIELDS, *CROP_EXTRA_FIELDS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Required source CSV is missing: {path}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f'Source CSV has no header: {path}')
        return list(reader), list(reader.fieldnames)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Required source JSON is missing: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Iterable[dict], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _ordered_union(*groups: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for field in group:
            if field not in seen:
                output.append(field)
                seen.add(field)
    return output


def _well_id(value: object) -> str:
    return crop_base._normalise_well_id(value)


def _pdo_number(value: object) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f'Invalid pdo_number_in_well {value!r}.') from exc
    if not math.isfinite(number) or not number.is_integer() or number < 1:
        raise RuntimeError(f'Invalid pdo_number_in_well {value!r}.')
    return int(number)


def _identity(condition_id: str, row: dict) -> tuple[str, str, int]:
    return condition_id, _well_id(row.get('well_id')), _pdo_number(row.get('pdo_number_in_well'))


def _finite(row: dict, field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid or missing numeric field '{field}'.") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Field '{field}' must be finite.")
    return value


def _optional_finite(row: dict, field: str) -> float | None:
    value = str(row.get(field, '')).strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"Malformed numeric field '{field}': {value!r}.") from exc
    if not math.isfinite(parsed):
        return None
    return parsed


def _optional_bool(row: dict, field: str) -> bool | None:
    value = str(row.get(field, '')).strip().lower()
    if not value:
        return None
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise RuntimeError(f"Malformed boolean field '{field}': {row.get(field)!r}.")


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def final_object_decision(row: dict) -> tuple[bool, list[str]]:
    """Return the frozen final decision and all independently applicable reasons."""
    match = str(row.get('unmasked_component_match_status', '')).strip()
    extent = str(row.get('full_component_extent_status', '')).strip()
    fraction = _optional_finite(row, 'full_component_fraction_inside_final_well')
    centroid_inside = _optional_bool(row, 'full_component_centroid_inside_final_well')
    many_to_one = _optional_bool(row, 'many_production_PDOs_to_one_unmasked_component')

    if match == 'trusted_complete_match' and extent == 'complete':
        missing = [name for name, value in (
            ('full_component_fraction_inside_final_well', fraction),
            ('full_component_centroid_inside_final_well', centroid_inside),
            ('many_production_PDOs_to_one_unmasked_component', many_to_one),
        ) if value is None]
        if missing:
            raise RuntimeError('Trusted complete PDO row lacks required final-QC values: '
                               + ', '.join(missing) + '.')

    reasons: list[str] = []
    if fraction is not None and fraction < FRACTION_INSIDE_THRESHOLD:
        reasons.append('containment_below_0p60')
    if centroid_inside is False:
        reasons.append('full_component_centroid_outside_well')
    if many_to_one is True:
        reasons.append('many_to_one_unmasked_component')
    if match != 'trusted_complete_match':
        reasons.append('untrusted_component_match')
    if extent != 'complete':
        reasons.append('incomplete_component_extent')

    if not reasons and (fraction is None or centroid_inside is not True or many_to_one is not False):
        raise RuntimeError('PDO row could not be classified because required final-QC values are absent.')
    return not reasons, reasons


def _condition_sources(result_root: Path, condition_id: str) -> dict:
    folder = result_root / condition_id
    wells, well_fields = _read_csv(folder / 'well_measurements.csv')
    pdos, pdo_fields = _read_csv(folder / 'pdo_measurements.csv')
    rfp, rfp_fields = _read_csv(folder / 'psc_quantification' / 'psc_well_measurements.csv')
    return {
        'folder': folder, 'wells': wells, 'well_fields': well_fields,
        'pdos': pdos, 'pdo_fields': pdo_fields, 'rfp': rfp, 'rfp_fields': rfp_fields,
    }


def validate_and_prepare(result_root: Path, *, expected_pdo_rows: int = EXPECTED_AUTHORITATIVE_PDO_ROWS) -> dict:
    authoritative, authoritative_fields = _read_csv(result_root / AUTHORITATIVE_COMPONENT_CSV)
    if len(authoritative) != expected_pdo_rows:
        raise RuntimeError(f'Authoritative full-component PDO row count is {len(authoritative)}; '
                           f'expected {expected_pdo_rows}.')

    conditions = {condition_id: _condition_sources(result_root, condition_id)
                  for condition_id in presentation.CONDITIONS}
    well_maps: dict[str, dict[str, dict]] = {}
    pdo_maps: dict[tuple[str, str, int], dict] = {}
    rfp_maps: dict[str, dict[str, dict]] = {}
    all_well_fields: list[str] = []
    all_rfp_fields: list[str] = []

    for condition_id, source in conditions.items():
        well_map: dict[str, dict] = {}
        for well in source['wells']:
            well_id = _well_id(well.get('well_id'))
            if well_id in well_map:
                raise RuntimeError(f'Duplicate final well {condition_id}/{well_id}.')
            if 'hex_array_member' not in well or not _truthy(well.get('hex_array_member')):
                raise RuntimeError(f'Final well {condition_id}/{well_id} lacks truthy '
                                   'hex_array_member provenance.')
            well_map[well_id] = well
        well_maps[condition_id] = well_map

        grouped: dict[str, list[dict]] = defaultdict(list)
        for pdo in source['pdos']:
            key = _identity(condition_id, pdo)
            if key in pdo_maps:
                raise RuntimeError(f'Duplicate original production PDO identity {key}.')
            if key[1] not in well_map:
                raise RuntimeError(f'Original PDO refers to nonexistent final well {key[:2]}.')
            pdo_maps[key] = pdo
            grouped[key[1]].append(pdo)
        for well_id, well in well_map.items():
            actual = len(grouped.get(well_id, []))
            expected = int(round(_finite(well, 'PDO_count')))
            if actual != expected:
                raise RuntimeError(f'Original PDO_count mismatch for {condition_id}/{well_id}: '
                                   f'well={expected}, rows={actual}.')
            if _truthy(well.get('PDO_present')) != (actual > 0):
                raise RuntimeError(f'Original PDO_present mismatch for {condition_id}/{well_id}.')

        rfp_map: dict[str, dict] = {}
        for row in source['rfp']:
            well_id = _well_id(row.get('well_id'))
            if well_id in rfp_map:
                raise RuntimeError(f'Duplicate continuous-RFP well {condition_id}/{well_id}.')
            if str(row.get('condition_id', '')).strip() != condition_id:
                raise RuntimeError(f'Continuous-RFP condition mismatch for {condition_id}/{well_id}.')
            rfp_map[well_id] = row
        if set(rfp_map) != set(well_map):
            raise RuntimeError(f'Continuous-RFP/final-well identity mismatch for {condition_id}: '
                               f'missing={sorted(set(well_map)-set(rfp_map))}, '
                               f'extra={sorted(set(rfp_map)-set(well_map))}.')
        for well_id, well in well_map.items():
            for field in ('x_px_fullres', 'y_px_fullres', 'radius_px'):
                if not math.isclose(_finite(well, field), _finite(rfp_map[well_id], field),
                                    rel_tol=0.0, abs_tol=1e-6):
                    raise RuntimeError(f'Continuous-RFP {field} mismatch for '
                                       f'{condition_id}/{well_id}.')
        rfp_maps[condition_id] = rfp_map
        all_well_fields = _ordered_union(all_well_fields, source['well_fields'])
        all_rfp_fields = _ordered_union(all_rfp_fields, source['rfp_fields'])

    authoritative_map: dict[tuple[str, str, int], dict] = {}
    for row in authoritative:
        condition_id = str(row.get('condition_id', '')).strip()
        if condition_id not in conditions:
            raise RuntimeError(f'Unknown authoritative condition_id {condition_id!r}.')
        key = _identity(condition_id, row)
        if key in authoritative_map:
            raise RuntimeError(f'Duplicate authoritative PDO identity {key}.')
        if key[1] not in well_maps[condition_id]:
            raise RuntimeError(f'Authoritative PDO refers to nonexistent final well {key[:2]}.')
        authoritative_map[key] = row
    if set(authoritative_map) != set(pdo_maps):
        raise RuntimeError('Authoritative and original production PDO identities differ: '
                           f'missing={sorted(set(pdo_maps)-set(authoritative_map))}, '
                           f'extra={sorted(set(authoritative_map)-set(pdo_maps))}.')

    return {
        'authoritative': authoritative, 'authoritative_fields': authoritative_fields,
        'authoritative_map': authoritative_map, 'conditions': conditions,
        'well_maps': well_maps, 'pdo_maps': pdo_maps, 'rfp_maps': rfp_maps,
        'well_fields': all_well_fields, 'rfp_fields': all_rfp_fields,
    }


def apply_object_rule(prepared: dict) -> tuple[list[dict], set[tuple[str, str, int]]]:
    rows: list[dict] = []
    passing: set[tuple[str, str, int]] = set()
    for source in prepared['authoritative']:
        condition_id = str(source['condition_id']).strip()
        key = _identity(condition_id, source)
        accepted, reasons = final_object_decision(source)
        row = dict(source)
        row['final_PDO_QC_pass'] = accepted
        row['final_PDO_QC_failure_reasons'] = ';'.join(reasons)
        rows.append(row)
        if accepted:
            passing.add(key)
    if len(passing) + (len(rows) - len(passing)) != len(rows):
        raise RuntimeError('Retained/rejected PDO row reconciliation failed.')
    by_key = {_identity(str(row['condition_id']).strip(), row): row for row in rows}
    for key in KNOWN_FAILURES:
        candidates = [row for identity, row in by_key.items() if identity[:2] == key]
        if not candidates:
            raise RuntimeError(f'Mandatory known failure {key[0]}/well {key[1]} is absent.')
        for row in candidates:
            if _truthy(row['final_PDO_QC_pass']):
                raise RuntimeError(f'Mandatory known failure {key[0]}/well {key[1]} unexpectedly passed.')
            reasons = row['final_PDO_QC_failure_reasons'].split(';')
            if 'containment_below_0p60' not in reasons:
                raise RuntimeError(f'Mandatory known failure {key[0]}/well {key[1]} lacks '
                                   'containment_below_0p60.')
    return rows, passing


def recompute_wells(prepared: dict, passing: set[tuple[str, str, int]]) -> tuple[list[dict], dict]:
    output: list[dict] = []
    final_by_condition: dict[str, dict[str, dict]] = {}
    for condition_id in presentation.CONDITIONS:
        source = prepared['conditions'][condition_id]
        final_map: dict[str, dict] = {}
        for well in source['wells']:
            well_id = _well_id(well['well_id'])
            original = [pdo for key, pdo in prepared['pdo_maps'].items()
                        if key[:2] == (condition_id, well_id)]
            retained = sorted(
                [(key, pdo) for key, pdo in prepared['pdo_maps'].items()
                 if key[:2] == (condition_id, well_id) and key in passing],
                key=lambda item: item[0][2],
            )
            row = dict(well)
            audit = {
                'original_PDO_present': well.get('PDO_present', ''),
                'original_PDO_count': well.get('PDO_count', ''),
                'original_total_PDO_projected_area_px2':
                    well.get('total_PDO_projected_area_px2', ''),
                'original_total_PDO_projected_area_um2':
                    well.get('total_PDO_projected_area_um2', ''),
            }
            # Update first so every RFP source cell, including key/geometry cells, is verbatim.
            row.update(prepared['rfp_maps'][condition_id][well_id])
            row.update(audit)
            row['PDO_count'] = len(retained)
            row['PDO_present'] = bool(retained)
            row['total_PDO_projected_area_px2'] = sum(
                _finite(pdo, 'projected_area_px2') for _, pdo in retained)
            row['total_PDO_projected_area_um2'] = sum(
                _finite(pdo, 'projected_area_um2') for _, pdo in retained)
            row['retained_PDO_numbers'] = ';'.join(str(key[2]) for key, _ in retained)
            row['retained_PDO_equivalent_circular_diameters_um'] = ';'.join(
                str(pdo.get('equivalent_circular_diameter_um', '')) for _, pdo in retained)
            row['final_array_qc_status'] = FINAL_ARRAY_QC_STATUS
            row['PSC_cell_count_status'] = PSC_CELL_COUNT_STATUS
            output.append(row)
            final_map[well_id] = row
            if len(original) != int(round(_finite(well, 'PDO_count'))):
                raise RuntimeError(f'Original PDO denominator changed for {condition_id}/{well_id}.')
        final_by_condition[condition_id] = final_map
    return output, final_by_condition


def condition_summaries(prepared: dict, object_rows: list[dict], final_maps: dict) -> list[dict]:
    rows: list[dict] = []
    for condition_id, mapping in presentation.CONDITIONS.items():
        wells = prepared['conditions'][condition_id]['wells']
        objects = [row for row in object_rows if str(row['condition_id']).strip() == condition_id]
        retained = [row for row in objects if _truthy(row['final_PDO_QC_pass'])]
        final_wells = list(final_maps[condition_id].values())
        original_positive = {_well_id(row['well_id']) for row in wells if _truthy(row.get('PDO_present'))}
        retained_positive = {_well_id(row['well_id']) for row in final_wells if _truthy(row.get('PDO_present'))}
        row = {
            'condition_id': condition_id, 'condition_name': mapping['condition_name'],
            'dose': mapping['dose'], 'dose_nM': mapping['dose_nM'],
            'total_final_dominant_array_wells': len(wells),
            'original_PDO_positive_wells': len(original_positive),
            'original_PDO_objects': len(objects),
            'retained_PDO_positive_wells': len(retained_positive),
            'retained_PDO_objects': len(retained),
            'PDO_objects_removed': len(objects) - len(retained),
            'wells_converted_PDO_positive_to_negative': len(original_positive-retained_positive),
            'final_PDO_positive_fraction': len(retained_positive) / len(wells) if wells else math.nan,
            'retained_multi_PDO_wells': sum(int(float(well['PDO_count'])) > 1 for well in final_wells),
        }
        for reason in FAILURE_REASONS:
            row[f'removed_{reason}'] = sum(
                reason in str(item['final_PDO_QC_failure_reasons']).split(';') for item in objects)
        rows.append(row)

    combined = {
        'condition_id': 'ALL_CONDITIONS', 'condition_name': 'All conditions',
        'dose': '', 'dose_nM': '',
    }
    additive = [field for field in SUMMARY_FIELDS
                if field not in {'condition_id', 'condition_name', 'dose', 'dose_nM',
                                 'final_PDO_positive_fraction'}]
    for field in additive:
        combined[field] = sum(int(row[field]) for row in rows)
    denominator = combined['total_final_dominant_array_wells']
    combined['final_PDO_positive_fraction'] = (
        combined['retained_PDO_positive_wells'] / denominator if denominator else math.nan)
    rows.append(combined)
    return rows


def _header_lines(condition_id: str, well: dict, pdos: list[dict], rfp: dict) -> list[str]:
    mapping = presentation.CONDITIONS[condition_id]
    diameters = [_finite(pdo, 'equivalent_circular_diameter_um') for pdo in pdos]
    lines = [
        f"Lane {mapping['lane']} | RMC6236 {mapping['dose']} | Final well {_well_id(well['well_id'])} | PDO POSITIVE",
        f"Final PDO count: {int(float(well['PDO_count']))}",
        'Retained PDO size(s): ' + ', '.join(f'{value:.1f}' for value in diameters) + ' µm',
        f"Retained total PDO projected area: {float(well['total_PDO_projected_area_um2']):.1f} µm²",
    ]
    if rfp.get('background_qc') == 'insufficient_local_background':
        lines.extend([
            'PSC/RFP background-corrected signal: not quantified',
            f"Raw RFP p95: {presentation._display_number(_finite(rfp, 'RFP_p95'))} detector units",
            'Exploratory RFP-positive area: not quantified',
            'Background QC: insufficient_local_background',
        ])
    else:
        lines.extend([
            'PSC/RFP background-corrected mean: '
            f"{presentation._display_number(_finite(rfp, 'RFP_background_corrected_mean'))} detector units",
            'PSC/RFP background-corrected integrated intensity: '
            f"{presentation._display_number(_finite(rfp, 'RFP_background_corrected_integrated_intensity'))} detector units·pixels",
            f"Raw RFP p95: {presentation._display_number(_finite(rfp, 'RFP_p95'))} detector units",
            'Exploratory RFP-positive area: '
            f"{presentation._display_number(_finite(rfp, 'exploratory_RFP_positive_area_um2'))} µm²",
            f"Background QC: {rfp.get('background_qc')}",
        ])
    lines.extend([
        'PSC cell count: NOT VALIDATED',
        f"Full-resolution x/y: {float(well['x_px_fullres']):.1f}, {float(well['y_px_fullres']):.1f}",
        f"Well radius: {float(well['radius_px']):.1f} px",
        f'Final-array QC status: {FINAL_ARRAY_QC_STATUS}',
    ])
    return lines


def _labelled_crop(raw: dict[str, Image.Image], *, condition_id: str, well: dict,
                   pdos: list[dict], rfp: dict, validation: dict,
                   left: int, top: int, panel_size: int) -> Image.Image:
    gap, panel_title_height = 8, 28
    width = 2 * panel_size + 3 * gap
    title_font, body_font = crop_base._fonts()
    wrapped: list[str] = []
    for line in _header_lines(condition_id, well, pdos, rfp):
        wrapped.extend(textwrap.wrap(line, width=max(50, width // 9),
                                     break_long_words=False) or [''])
    header_height = 14 + 30 + max(0, len(wrapped)-1) * 23 + 10
    canvas = Image.new('RGB', (width, header_height + 2*(panel_size+panel_title_height)+3*gap), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, header_height), fill='black')
    y = 8
    for index, line in enumerate(wrapped):
        draw.text((12, y), line, fill='white', font=title_font if index == 0 else body_font)
        y += 30 if index == 0 else 23
    pixel_size_um = (validation['pixel_size_um']['x'] + validation['pixel_size_um']['y']) / 2.0
    for index, (kind, label) in enumerate((('dic', 'DIC'), ('gfp', 'GFP'),
                                           ('rfp', 'RFP'), ('composite', 'Composite'))):
        column, panel_row = index % 2, index // 2
        x = gap + column*(panel_size+gap)
        panel_y = header_height + gap + panel_row*(panel_size+panel_title_height+gap)
        draw.rectangle((x, panel_y, x+panel_size, panel_y+panel_title_height), fill='black')
        draw.text((x+7, panel_y+5), label, fill='white', font=body_font)
        panel = crop_base._overlay_panel(
            raw[kind], panel_size=panel_size, well_x=float(well['x_px_fullres']),
            well_y=float(well['y_px_fullres']), well_radius=float(well['radius_px']),
            left=left, top=top, pdos=pdos, pixel_size_um=pixel_size_um)
        canvas.paste(panel, (x, panel_y+panel_title_height))
        panel.close()
    return canvas


def _crop_restart_valid(row: dict | None, signature: str) -> bool:
    return bool(row and row.get('export_status') == 'completed'
                and row.get('restart_signature') == signature
                and Path(row.get('labelled_crop', '')).is_file())


def export_final_crops(result_root: Path, final_root: Path, prepared: dict,
                       final_maps: dict, passing: set[tuple[str, str, int]], args,
                       *, probe: Callable = presentation.probe_omezarr,
                       open_group: Callable = zarr.open_group) -> dict:
    batch_path = result_root / 'batch_status.json'
    batch_status = json.loads(batch_path.read_text(encoding='utf-8')) if batch_path.is_file() else {}
    crop_checks = {}
    for condition_id in presentation.CONDITIONS:
        folder = result_root / condition_id
        condition_summary = _read_json(folder / 'condition_summary.json')
        zarr_path = crop_base.resolve_omezarr(condition_id, condition_summary,
                                              batch_status, args.cache_root)
        metadata = probe(zarr_path)
        validation = crop_base.validate_omezarr(metadata, condition_summary,
                                                args.expected_pixel_size_um)
        root = open_group(str(zarr_path), mode='r')
        array = root[metadata['level0_array_path']]
        planes = SingletonTZCYX(array, metadata['axes'])
        channels, height, width = planes.shape_cyx
        if channels != 3:
            raise RuntimeError(f'Validated OME-Zarr for {condition_id} does not contain 3 channels.')
        ranges = crop_base.display_ranges(metadata, planes, width, height,
                                          args.display_sample_size, args.display_sample_grid)
        output = final_root / 'pdo_positive_crops' / condition_id
        manifest_path = output / 'manifest.csv'
        prior_rows = {}
        if manifest_path.is_file():
            prior, _ = _read_csv(manifest_path)
            prior_rows = {_well_id(row['well_id']): row for row in prior}
        positives = [well for well in final_maps[condition_id].values() if _truthy(well['PDO_present'])]
        positives.sort(key=lambda row: (float(row['y_px_fullres']), float(row['x_px_fullres'])))
        expected_ids = {_well_id(well['well_id']) for well in positives}
        filenames = [crop_base._filename(condition_id, _well_id(well['well_id']),
                                          float(well['x_px_fullres']), float(well['y_px_fullres']))
                     for well in positives]
        if len(filenames) != len(set(filenames)):
            raise RuntimeError(f'Duplicate crop filename for {condition_id}.')
        rows: list[dict] = []
        for index, (well, filename) in enumerate(zip(positives, filenames)):
            well_id = _well_id(well['well_id'])
            retained = sorted(
                [(key, pdo) for key, pdo in prepared['pdo_maps'].items()
                 if key[:2] == (condition_id, well_id) and key in passing],
                key=lambda item: item[0][2])
            objects = [pdo for _, pdo in retained]
            identities = ';'.join(f'{key[0]}|{key[1]}|{key[2]}' for key, _ in retained)
            if len(objects) != int(float(well['PDO_count'])):
                raise RuntimeError(f'Crop PDO-count mismatch for {condition_id}/{well_id}.')
            rfp = prepared['rfp_maps'][condition_id][well_id]
            x, y, radius = (float(well['x_px_fullres']), float(well['y_px_fullres']),
                            float(well['radius_px']))
            half = max(1, int(round(radius*args.crop_radius_scale)))
            labelled_path = output / 'labelled_crops' / filename
            source_row = presentation._source_manifest_row(condition_id, well, objects, rfp)
            source_row['retained_PDO_identities'] = identities
            source_row['final_PDO_QC_rule'] = (
                'trusted_complete_match;complete;fraction_inside>=0.60;'
                'centroid_inside;not_many_to_one')
            signature = crop_base._signature({
                'finalization_version': FINALIZATION_VERSION, 'condition_id': condition_id,
                'well': well, 'passing_PDO_identities': identities, 'pdo_rows': objects,
                'rfp_row': rfp, 'omezarr': str(zarr_path), 'shape': validation['shape'],
                'axes': validation['axes'], 'channels': presentation.CHANNELS,
                'pixel_size_um': validation['pixel_size_um'], 'display_ranges': ranges,
                'crop_radius_scale': args.crop_radius_scale, 'panel_size': args.panel_size,
            })
            if _crop_restart_valid(prior_rows.get(well_id), signature):
                row = prior_rows[well_id]
            else:
                row = dict(source_row)
                try:
                    arrays = {}
                    left = top = 0
                    for kind in ('dic', 'gfp', 'rfp'):
                        arrays[kind], left, top = crop_base._read_padded(
                            planes, presentation.CHANNELS[kind], x, y, half, width, height)
                    images = crop_base._raw_images(arrays['dic'], arrays['gfp'], arrays['rfp'], ranges)
                    labelled = _labelled_crop(images, condition_id=condition_id, well=well,
                                               pdos=objects, rfp=rfp, validation=validation,
                                               left=left, top=top, panel_size=args.panel_size)
                    labelled_path.parent.mkdir(parents=True, exist_ok=True)
                    labelled.save(labelled_path, dpi=(300, 300))
                    labelled.close()
                    for image in images.values():
                        image.close()
                    row.update({
                        'pixel_size_x_um': validation['pixel_size_um']['x'],
                        'pixel_size_y_um': validation['pixel_size_um']['y'],
                        'omezarr_source': str(zarr_path), 'crop_left_px_fullres': left,
                        'crop_top_px_fullres': top, 'crop_side_px': 2*half+1,
                        'crop_radius_scale': args.crop_radius_scale,
                        'display_ranges_json': json.dumps(ranges, sort_keys=True),
                        'restart_signature': signature, 'labelled_crop': str(labelled_path),
                        'export_status': 'completed', 'export_error': '',
                    })
                except Exception as exc:
                    row.update({'restart_signature': signature, 'labelled_crop': str(labelled_path),
                                'export_status': 'failed',
                                'export_error': f'{type(exc).__name__}: {exc}'})
            rows.append(row)
            future_ids = [_well_id(item['well_id']) for item in positives[index+1:]]
            _atomic_csv(manifest_path, rows + [prior_rows[item] for item in future_ids
                                               if item in prior_rows], CROP_MANIFEST_FIELDS)
        if not positives:
            _atomic_csv(manifest_path, [], CROP_MANIFEST_FIELDS)
        completed = [row for row in rows if row.get('export_status') == 'completed'
                     and Path(row.get('labelled_crop', '')).is_file()]
        completed_ids = {_well_id(row['well_id']) for row in completed}
        manifest_ids = {_well_id(row['well_id']) for row in rows
                        if row.get('export_status') == 'completed'}
        overlay_identity_ok = all(
            row.get('retained_PDO_identities', '') == ';'.join(
                f'{key[0]}|{key[1]}|{key[2]}' for key in sorted(passing)
                if key[:2] == (condition_id, _well_id(row['well_id'])))
            for row in completed)
        if completed_ids != expected_ids or manifest_ids != expected_ids or not overlay_identity_ok:
            raise RuntimeError(f'Final crop-set/overlay identity QC failed for {condition_id}.')
        sheets = crop_base._contact_sheets([Path(row['labelled_crop']) for row in completed],
                                           output / 'contact_sheets', args.contact_sheet_size)
        if expected_ids and not sheets:
            raise RuntimeError(f'No contact sheets were generated for {condition_id}.')
        crop_checks[condition_id] = {
            'expected_positive_wells': len(expected_ids), 'completed_crops': len(completed_ids),
            'crop_ids_equal_final_positive_ids': completed_ids == expected_ids,
            'manifest_ids_equal_final_positive_ids': manifest_ids == expected_ids,
            'crop_overlay_identities_equal_passing_PDO_identities': overlay_identity_ok,
            'display_ranges': ranges, 'contact_sheets': sheets,
        }
    return crop_checks


def _verify_rfp_verbatim(path: Path, prepared: dict) -> dict:
    rows, _ = _read_csv(path)
    output = {(str(row.get('condition_id', '')).strip(), _well_id(row.get('well_id'))): row
              for row in rows}
    checked_cells = 0
    for condition_id in presentation.CONDITIONS:
        fields = prepared['conditions'][condition_id]['rfp_fields']
        for well_id, source in prepared['rfp_maps'][condition_id].items():
            target = output.get((condition_id, well_id))
            if target is None:
                raise RuntimeError(f'Final well table lacks RFP row {condition_id}/{well_id}.')
            for field in fields:
                if target.get(field, '') != source.get(field, ''):
                    raise RuntimeError(f'Verbatim RFP mismatch for {condition_id}/{well_id}/{field}: '
                                       f'{source.get(field)!r} != {target.get(field)!r}.')
                checked_cells += 1
    return {'passed': True, 'compared_source_cells': checked_cells}


def _verify_summary(rows: list[dict]) -> bool:
    if len(rows) != len(presentation.CONDITIONS)+1 or rows[-1]['condition_id'] != 'ALL_CONDITIONS':
        return False
    combined = rows[-1]
    additive = [field for field in SUMMARY_FIELDS
                if field not in {'condition_id', 'condition_name', 'dose', 'dose_nM',
                                 'final_PDO_positive_fraction'}]
    if any(int(combined[field]) != sum(int(row[field]) for row in rows[:-1]) for field in additive):
        return False
    denominator = int(combined['total_final_dominant_array_wells'])
    expected = int(combined['retained_PDO_positive_wells'])/denominator if denominator else math.nan
    return math.isclose(float(combined['final_PDO_positive_fraction']), expected,
                        rel_tol=0.0, abs_tol=1e-15)


def finalize(result_root: Path, args, *, expected_pdo_rows: int = EXPECTED_AUTHORITATIVE_PDO_ROWS,
             crop_exporter: Callable = export_final_crops,
             probe: Callable = presentation.probe_omezarr,
             open_group: Callable = zarr.open_group) -> dict:
    result_root = result_root.expanduser().resolve()
    final_root = result_root / OUTPUT_DIRECTORY
    summary_path = final_root / 'final_qc_summary.json'
    _atomic_json(summary_path, {
        'finalization_version': FINALIZATION_VERSION, 'completion_status': 'running',
        'started_at': _now(), 'final_PDO_QC_threshold_inclusive': FRACTION_INSIDE_THRESHOLD,
    })
    try:
        prepared = validate_and_prepare(result_root, expected_pdo_rows=expected_pdo_rows)
        object_rows, passing = apply_object_rule(prepared)
        final_wells, final_maps = recompute_wells(prepared, passing)
        summaries = condition_summaries(prepared, object_rows, final_maps)

        object_path = final_root / 'final_pdo_object_qc.csv'
        well_path = final_root / 'final_well_measurements.csv'
        summary_csv_path = final_root / 'final_condition_summary.csv'
        object_fields = _ordered_union(prepared['authoritative_fields'], OBJECT_EXTRA_FIELDS)
        well_fields = _ordered_union(prepared['well_fields'], prepared['rfp_fields'],
                                     FINAL_WELL_EXTRA_FIELDS)
        _atomic_csv(object_path, object_rows, object_fields)
        _atomic_csv(well_path, final_wells, well_fields)
        _atomic_csv(summary_csv_path, summaries, SUMMARY_FIELDS)

        rfp_check = _verify_rfp_verbatim(well_path, prepared)
        crop_checks = crop_exporter(result_root, final_root, prepared, final_maps, passing, args,
                                    probe=probe, open_group=open_group)
        source_well_count = sum(len(item['wells']) for item in prepared['conditions'].values())
        final_keys = [(str(row['condition_id']).strip(), _well_id(row['well_id']))
                      for row in final_wells]
        original_negative = {(condition_id, _well_id(well['well_id']))
                             for condition_id, source in prepared['conditions'].items()
                             for well in source['wells'] if not _truthy(well.get('PDO_present'))}
        final_negative = {(condition_id, well_id) for condition_id, mapping in final_maps.items()
                          for well_id, well in mapping.items() if not _truthy(well.get('PDO_present'))}
        retained_count = len(passing)
        rejected_count = len(object_rows)-retained_count
        known_crop_ids = {condition_id: {
            _well_id(row['well_id']) for row in final_maps[condition_id].values()
            if _truthy(row['PDO_present'])} for condition_id in presentation.CONDITIONS}
        integrity = {
            'authoritative_PDO_rows_equal_expected_2568': len(object_rows) == expected_pdo_rows,
            'retained_plus_rejected_equals_expected_2568':
                retained_count+rejected_count == expected_pdo_rows,
            'authoritative_identities_equal_original_production_identities': True,
            'every_source_final_well_appears_once':
                len(final_keys) == source_well_count and len(final_keys) == len(set(final_keys)),
            'all_final_wells_dominant_hex_array_members': all(
                _truthy(well.get('hex_array_member'))
                for source in prepared['conditions'].values() for well in source['wells']),
            'original_PDO_negative_wells_preserved': original_negative <= final_negative,
            'RFP_source_output_verbatim_match': rfp_check,
            'condition_summaries_reconcile_to_ALL_CONDITIONS': _verify_summary(summaries),
            'crop_checks': crop_checks,
            'DMSO_606_absent_from_retained_crops': '606' not in known_crop_ids[KNOWN_FAILURES[0][0]],
            'DMSO_624_absent_from_retained_crops': '624' not in known_crop_ids[KNOWN_FAILURES[1][0]],
        }
        required = [value for key, value in integrity.items()
                    if key not in {'RFP_source_output_verbatim_match', 'crop_checks'}]
        if not all(required) or not rfp_check['passed'] or not all(
                all(value for key, value in check.items()
                    if key.endswith('_ids') or key.startswith('crop_'))
                for check in crop_checks.values()):
            raise RuntimeError('One or more final integrity checks failed.')
        result = {
            'finalization_version': FINALIZATION_VERSION, 'completion_status': 'completed',
            'started_at': json.loads(summary_path.read_text(encoding='utf-8'))['started_at'],
            'completed_at': _now(),
            'final_PDO_QC_rule': {
                'unmasked_component_match_status': 'trusted_complete_match',
                'full_component_extent_status': 'complete',
                'full_component_fraction_inside_final_well_minimum_inclusive': 0.60,
                'full_component_centroid_inside_final_well': True,
                'many_production_PDOs_to_one_unmasked_component': False,
                'crosses_or_touches_boundary_is_not_an_exclusion': True,
            },
            'authoritative_PDO_rows': len(object_rows), 'retained_PDO_rows': retained_count,
            'rejected_PDO_rows': rejected_count, 'total_final_wells': len(final_wells),
            'condition_summary': summaries, 'integrity_checks': integrity,
            'psc_cell_count_status': PSC_CELL_COUNT_STATUS,
            'scientific_analysis_performed': False,
            'outputs': [str(object_path), str(well_path), str(summary_csv_path),
                        str(final_root/'pdo_positive_crops')],
        }
        _atomic_json(summary_path, result)
        return result
    except Exception as exc:
        _atomic_json(summary_path, {
            'finalization_version': FINALIZATION_VERSION, 'completion_status': 'failed',
            'failed_at': _now(), 'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc(), 'scientific_analysis_performed': False,
        })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Apply the frozen final PDO QC rule and regenerate retained-positive '
                    'presentation crops without scientific image analysis.')
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--expected-pixel-size-um', type=float,
                        default=presentation.EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--crop-radius-scale', type=float, default=1.75)
    parser.add_argument('--panel-size', type=int, default=384)
    parser.add_argument('--contact-sheet-size', type=int, default=16)
    parser.add_argument('--display-sample-size', type=int, default=256)
    parser.add_argument('--display-sample-grid', type=int, default=4)
    return parser


def run(args) -> int:
    if args.crop_radius_scale <= 0 or args.panel_size < 64 or args.contact_sheet_size < 1:
        raise ValueError('Crop radius scale, panel size, and contact-sheet size must be positive.')
    try:
        result = finalize(args.result_root, args)
    except Exception as exc:
        print(f'Final PDO analysis QC FAILED: {type(exc).__name__}: {exc}', flush=True)
        return 1
    retained_wells = result['condition_summary'][-1]['retained_PDO_positive_wells']
    print(f"Final PDO analysis QC completed: {result['retained_PDO_rows']}/"
          f"{result['authoritative_PDO_rows']} PDO objects retained; {retained_wells}/"
          f"{result['total_final_wells']} wells PDO-positive; all final integrity checks passed; "
          'PSC cell count remains NOT VALIDATED.', flush=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
