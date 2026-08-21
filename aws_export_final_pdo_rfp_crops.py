from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import zarr
from PIL import Image, ImageDraw

import aws_export_pdo_positive_crops as crop_base
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


EXPORT_VERSION = 1
OUTPUT_DIRECTORY = 'pdo_positive_crops_final_pdo_rfp'
COMBINED_MANIFEST = 'all_conditions_final_pdo_rfp_crop_manifest.csv'
FINAL_ARRAY_QC_STATUS = 'final_dominant_hex_array_accepted'
EXPECTED_PIXEL_SIZE_UM = crop_base.EXPECTED_PIXEL_SIZE_UM
CHANNELS = dict(crop_base.CHANNELS)
CONDITIONS = {
    'K3T_PSC_RMC6236_Lane_1_DMSO': {
        'condition_name': 'DMSO', 'lane': '1', 'dose': '0 nM', 'dose_nM': 0.0,
    },
    'K3T_PSC_RMC6236_5nm_Lane_2': {
        'condition_name': '5 nM RMC6236', 'lane': '2', 'dose': '5 nM', 'dose_nM': 5.0,
    },
    'K3T_PSC_RMC6236_25nm_Lane_3': {
        'condition_name': '25 nM RMC6236', 'lane': '3', 'dose': '25 nM', 'dose_nM': 25.0,
    },
    'K3T_PSC_RMC6236_50nm_Lane_1': {
        'condition_name': '50 nM RMC6236', 'lane': '1', 'dose': '50 nM', 'dose_nM': 50.0,
    },
    'K3T_PSC_RMC6236_100nm_Lane_5': {
        'condition_name': '100 nM RMC6236', 'lane': '5', 'dose': '100 nM', 'dose_nM': 100.0,
    },
    'K3T_PSC_RMC6236_150nm_Lane_6': {
        'condition_name': '150 nM RMC6236', 'lane': '6', 'dose': '150 nM', 'dose_nM': 150.0,
    },
}

# Complete schema of the already-generated continuous RFP table. These values are
# copied to the final manifest; none is recalculated by this presentation exporter.
RFP_SOURCE_FIELDS = (
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
_RFP_COLLISIONS = {
    'condition_id', 'condition_name', 'dose_nM', 'well_id',
    'x_px_fullres', 'y_px_fullres', 'radius_px', 'error',
}
RFP_MANIFEST_FIELDS = tuple(field for field in RFP_SOURCE_FIELDS if field not in _RFP_COLLISIONS)
MANIFEST_FIELDS = (
    'condition_id', 'condition_name', 'lane', 'dose', 'dose_nM', 'well_id',
    'PDO_present', 'PDO_count', 'total_PDO_projected_area_px2',
    'total_PDO_projected_area_um2', 'PDO_measurement_row_count',
    'PDO_equivalent_circular_diameters_um',
    'x_px_fullres', 'y_px_fullres', 'radius_px', 'final_array_qc_status',
    'lattice_degree',
    *RFP_MANIFEST_FIELDS, 'RFP_quantification_error',
    'gfp_channel', 'rfp_channel', 'dic_channel',
    'pixel_size_x_um', 'pixel_size_y_um', 'omezarr_source',
    'crop_left_px_fullres', 'crop_top_px_fullres', 'crop_side_px',
    'crop_radius_scale', 'display_ranges_json', 'overlay_provenance',
    'psc_cell_count_status', 'restart_signature', 'labelled_crop',
    'export_status', 'export_error',
)
BACKGROUND_DERIVED_HEADER_FIELDS = (
    'RFP_background_corrected_mean',
    'RFP_background_corrected_integrated_intensity',
    'exploratory_RFP_positive_area_um2',
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _atomic_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def _number(row: dict, key: str) -> float:
    return crop_base._number(row, key)


def _normalise_well_id(value: object) -> str:
    return crop_base._normalise_well_id(value)


def _finite_number(row: dict, key: str) -> float:
    value = _number(row, key)
    if not math.isfinite(value):
        raise RuntimeError(f"Field '{key}' must be finite for final crop display.")
    return value


def _unique_by_well(rows: list[dict], source: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        well_id = _normalise_well_id(row.get('well_id'))
        if well_id in result:
            raise RuntimeError(f'Duplicate well_id {well_id} in {source}.')
        result[well_id] = row
    return result


def _condition_inputs(folder: Path) -> tuple[list[dict], list[dict], list[dict]]:
    paths = (
        folder / 'well_measurements.csv',
        folder / 'pdo_measurements.csv',
        folder / 'psc_quantification' / 'psc_well_measurements.csv',
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required approved source CSV is missing: {path}')
    return tuple(_read_csv(path) for path in paths)  # type: ignore[return-value]


def _validate_sources(condition_id: str, wells: list[dict], pdos: list[dict],
                      rfp_rows: list[dict]) -> tuple[dict[str, dict], dict[str, list[dict]],
                                                     dict[str, dict], list[dict]]:
    mapping = CONDITIONS[condition_id]
    well_by_id = _unique_by_well(wells, 'well_measurements.csv')
    rfp_by_id = _unique_by_well(rfp_rows, 'psc_quantification/psc_well_measurements.csv')
    final_ids = set(well_by_id)
    if set(rfp_by_id) != final_ids:
        raise RuntimeError(
            f'Continuous-RFP/final well-set mismatch: missing={sorted(final_ids-set(rfp_by_id))}, '
            f'extra={sorted(set(rfp_by_id)-final_ids)}.'
        )
    pdo_by_id: dict[str, list[dict]] = {}
    for row in pdos:
        well_id = _normalise_well_id(row.get('well_id'))
        if well_id not in final_ids:
            raise RuntimeError(f'PDO row refers to non-final well_id {well_id}.')
        pdo_by_id.setdefault(well_id, []).append(row)
    for well_id, well in well_by_id.items():
        if 'hex_array_member' not in well or not str(well.get('hex_array_member', '')).strip():
            raise RuntimeError(
                f'Final well {well_id} lacks required hex_array_member schema/provenance field.'
            )
        if not _truthy(well['hex_array_member']):
            raise RuntimeError(
                f'Final well {well_id} has hex_array_member={well["hex_array_member"]!r}; '
                'expected truthy dominant-array membership.'
            )
        actual = len(pdo_by_id.get(well_id, []))
        expected = int(round(_finite_number(well, 'PDO_count')))
        if actual != expected:
            raise RuntimeError(
                f'Final PDO count mismatch for well {well_id}: well CSV={expected}, PDO rows={actual}.'
            )
        rfp = rfp_by_id[well_id]
        for field in RFP_SOURCE_FIELDS:
            if field not in rfp:
                raise RuntimeError(f'Continuous-RFP row for well {well_id} lacks field {field}.')
        if str(rfp.get('condition_id')).strip() != condition_id:
            raise RuntimeError(f'Continuous-RFP condition mismatch for well {well_id}.')
        if not math.isclose(_finite_number(rfp, 'dose_nM'), mapping['dose_nM'], abs_tol=1e-9):
            raise RuntimeError(f'Continuous-RFP dose mismatch for well {well_id}.')
        for coordinate in ('x_px_fullres', 'y_px_fullres', 'radius_px'):
            if not math.isclose(_finite_number(well, coordinate), _finite_number(rfp, coordinate),
                                rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f'Continuous-RFP {coordinate} mismatch for well {well_id}.')
        if rfp.get('background_qc') == 'insufficient_local_background':
            finite = [field for field in BACKGROUND_DERIVED_HEADER_FIELDS
                      if math.isfinite(_number(rfp, field))]
            if finite:
                raise RuntimeError(
                    f'Insufficient-background well {well_id} has finite background-derived '
                    f'values instead of NaN: {finite}.'
                )
    positive = [well for well in wells if _truthy(well.get('PDO_present'))]
    for well in positive:
        well_id = _normalise_well_id(well['well_id'])
        if int(round(_finite_number(well, 'PDO_count'))) <= 0:
            raise RuntimeError(f'PDO-positive well {well_id} does not have PDO_count > 0.')
    return well_by_id, pdo_by_id, rfp_by_id, positive


def _display_number(value: float) -> str:
    return f'{value:,.3f}'


def header_lines(condition_id: str, well: dict, pdos: list[dict], rfp: dict) -> list[str]:
    mapping = CONDITIONS[condition_id]
    well_id = _normalise_well_id(well['well_id'])
    diameters = [_finite_number(row, 'equivalent_circular_diameter_um') for row in pdos]
    lines = [
        f"Lane {mapping['lane']} | RMC6236 {mapping['dose']} | Final well {well_id} | PDO POSITIVE",
        f"PDO count: {int(round(_finite_number(well, 'PDO_count')))}",
        'PDO size(s): ' + ', '.join(f'{diameter:.1f}' for diameter in diameters) + ' µm',
        f"Total PDO projected area: {_finite_number(well, 'total_PDO_projected_area_um2'):.1f} µm²",
    ]
    if rfp.get('background_qc') == 'insufficient_local_background':
        lines.extend([
            'PSC/RFP background-corrected signal: not quantified',
            f"Raw RFP p95: {_display_number(_finite_number(rfp, 'RFP_p95'))} detector units",
            'Exploratory RFP-positive area: not quantified',
            'Background QC: insufficient_local_background',
        ])
    else:
        lines.extend([
            'PSC/RFP background-corrected mean: '
            f"{_display_number(_finite_number(rfp, 'RFP_background_corrected_mean'))} detector units",
            'PSC/RFP background-corrected integrated intensity: '
            f"{_display_number(_finite_number(rfp, 'RFP_background_corrected_integrated_intensity'))} "
            'detector units·pixels',
            f"Raw RFP p95: {_display_number(_finite_number(rfp, 'RFP_p95'))} detector units",
            'Exploratory RFP-positive area: '
            f"{_display_number(_finite_number(rfp, 'exploratory_RFP_positive_area_um2'))} µm²",
            f"Background QC: {rfp.get('background_qc')}",
        ])
    lines.extend([
        'PSC cell count: NOT VALIDATED',
        f"Full-resolution x/y: {_finite_number(well, 'x_px_fullres'):.1f}, "
        f"{_finite_number(well, 'y_px_fullres'):.1f}",
        f"Well radius: {_finite_number(well, 'radius_px'):.1f} px",
        f'Final-array QC status: {FINAL_ARRAY_QC_STATUS}',
    ])
    return lines


def labelled_four_panel(raw: dict[str, Image.Image], *, condition_id: str, well: dict,
                        pdos: list[dict], rfp: dict, validation: dict,
                        left: int, top: int, panel_size: int) -> Image.Image:
    gap, panel_title_height = 8, 28
    width = 2 * panel_size + 3 * gap
    title_font, body_font = crop_base._fonts()
    wrapped = []
    for line in header_lines(condition_id, well, pdos, rfp):
        wrapped.extend(textwrap.wrap(line, width=max(50, width // 9),
                                     break_long_words=False) or [''])
    header_height = 14 + 30 + max(0, len(wrapped) - 1) * 23 + 10
    canvas = Image.new(
        'RGB', (width, header_height + 2 * (panel_size + panel_title_height) + 3 * gap), 'white'
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, header_height), fill='black')
    y = 8
    for index, line in enumerate(wrapped):
        draw.text((12, y), line, fill='white', font=title_font if index == 0 else body_font)
        y += 30 if index == 0 else 23
    pixel_size_um = (validation['pixel_size_um']['x'] + validation['pixel_size_um']['y']) / 2.0
    for index, (kind, label) in enumerate((('dic', 'DIC'), ('gfp', 'GFP'),
                                           ('rfp', 'RFP'), ('composite', 'Composite'))):
        column, row = index % 2, index // 2
        x = gap + column * (panel_size + gap)
        panel_y = header_height + gap + row * (panel_size + panel_title_height + gap)
        draw.rectangle((x, panel_y, x + panel_size, panel_y + panel_title_height), fill='black')
        draw.text((x + 7, panel_y + 5), label, fill='white', font=body_font)
        panel = crop_base._overlay_panel(
            raw[kind], panel_size=panel_size,
            well_x=_finite_number(well, 'x_px_fullres'),
            well_y=_finite_number(well, 'y_px_fullres'),
            well_radius=_finite_number(well, 'radius_px'), left=left, top=top,
            pdos=pdos, pixel_size_um=pixel_size_um,
        )
        canvas.paste(panel, (x, panel_y + panel_title_height))
        panel.close()
    return canvas


def _source_manifest_row(condition_id: str, well: dict, pdos: list[dict], rfp: dict) -> dict:
    mapping = CONDITIONS[condition_id]
    diameters = [_finite_number(row, 'equivalent_circular_diameter_um') for row in pdos]
    row = {
        'condition_id': condition_id, 'condition_name': rfp['condition_name'],
        'lane': mapping['lane'], 'dose': mapping['dose'], 'dose_nM': rfp['dose_nM'],
        'well_id': _normalise_well_id(well['well_id']), 'PDO_present': True,
        'PDO_count': int(round(_finite_number(well, 'PDO_count'))),
        'total_PDO_projected_area_px2': well.get('total_PDO_projected_area_px2', ''),
        'total_PDO_projected_area_um2': well['total_PDO_projected_area_um2'],
        'PDO_measurement_row_count': len(pdos),
        'PDO_equivalent_circular_diameters_um': ';'.join(f'{value:.12g}' for value in diameters),
        'x_px_fullres': well['x_px_fullres'], 'y_px_fullres': well['y_px_fullres'],
        'radius_px': well['radius_px'], 'final_array_qc_status': FINAL_ARRAY_QC_STATUS,
        'lattice_degree': well.get('lattice_degree', ''),
        'RFP_quantification_error': rfp.get('error', ''),
        'gfp_channel': CHANNELS['gfp'], 'rfp_channel': CHANNELS['rfp'],
        'dic_channel': CHANNELS['dic'], 'psc_cell_count_status': 'NOT VALIDATED',
        'overlay_provenance': ('Reconstructed PDO centroid/equivalent-diameter overlay; '
                               'not the original segmentation mask.'),
    }
    row.update({field: rfp.get(field, '') for field in RFP_MANIFEST_FIELDS})
    return row


def _restart_valid(row: dict | None, signature: str) -> bool:
    return bool(row and row.get('export_status') == 'completed'
                and row.get('restart_signature') == signature
                and Path(row.get('labelled_crop', '')).is_file())


def export_condition(condition_id: str, folder: Path, args: argparse.Namespace,
                     batch_status: dict,
                     *, probe: Callable = probe_omezarr,
                     open_group: Callable = zarr.open_group, s3_client=None) -> dict:
    started = _now()
    condition_summary_path = folder / 'condition_summary.json'
    if not condition_summary_path.is_file():
        raise FileNotFoundError(
            f'Required OME-Zarr provenance summary is missing: {condition_summary_path}'
        )
    condition_summary = _read_json(condition_summary_path)
    wells, pdos, rfp_rows = _condition_inputs(folder)
    _, pdo_by_id, rfp_by_id, positive = _validate_sources(condition_id, wells, pdos, rfp_rows)
    expected_ids = {_normalise_well_id(row['well_id']) for row in positive}
    zarr_path = crop_base.resolve_omezarr(
        condition_id, condition_summary, batch_status, args.cache_root
    )
    metadata = probe(zarr_path)
    validation = crop_base.validate_omezarr(
        metadata, condition_summary, args.expected_pixel_size_um
    )
    root = open_group(str(zarr_path), mode='r')
    array = root[metadata['level0_array_path']]
    planes = SingletonTZCYX(array, metadata['axes'])
    channels, height, width = planes.shape_cyx
    if channels != 3:
        raise RuntimeError(f'Validated metadata and opened array disagree: {channels} channels.')
    ranges = crop_base.display_ranges(
        metadata, planes, width, height, args.display_sample_size, args.display_sample_grid
    )
    output = folder / OUTPUT_DIRECTORY
    manifest_path = output / 'manifest.csv'
    output.mkdir(parents=True, exist_ok=True)
    prior_rows = {_normalise_well_id(row['well_id']): row for row in _read_csv(manifest_path)} \
        if manifest_path.is_file() else {}
    ordered = sorted(positive, key=lambda row: (_finite_number(row, 'y_px_fullres'),
                                                _finite_number(row, 'x_px_fullres')))
    filenames = [crop_base._filename(condition_id, _normalise_well_id(row['well_id']),
                                      _finite_number(row, 'x_px_fullres'),
                                      _finite_number(row, 'y_px_fullres')) for row in ordered]
    if len(filenames) != len(set(filenames)):
        raise RuntimeError('Duplicate labelled-crop output filenames were generated.')
    rows = []
    if not ordered:
        _atomic_csv(manifest_path, rows)
    for index, (well, filename) in enumerate(zip(ordered, filenames)):
        well_id = _normalise_well_id(well['well_id'])
        objects = sorted(pdo_by_id.get(well_id, []),
                         key=lambda row: int(round(_finite_number(row, 'pdo_number_in_well'))))
        rfp = rfp_by_id[well_id]
        x, y = _finite_number(well, 'x_px_fullres'), _finite_number(well, 'y_px_fullres')
        radius = _finite_number(well, 'radius_px')
        half = max(1, int(round(radius * args.crop_radius_scale)))
        labelled_path = output / 'labelled_crops' / filename
        source_row = _source_manifest_row(condition_id, well, objects, rfp)
        signature = crop_base._signature({
            'export_version': EXPORT_VERSION, 'condition_id': condition_id, 'well': well,
            'pdo_rows': objects, 'rfp_row': rfp, 'omezarr': str(zarr_path),
            'shape': validation['shape'], 'axes': validation['axes'], 'channels': CHANNELS,
            'pixel_size_um': validation['pixel_size_um'], 'display_ranges': ranges,
            'crop_radius_scale': args.crop_radius_scale, 'panel_size': args.panel_size,
        })
        if _restart_valid(prior_rows.get(well_id), signature):
            rows.append(prior_rows[well_id])
        else:
            row = dict(source_row)
            try:
                arrays = {}
                left = top = 0
                for kind in ('dic', 'gfp', 'rfp'):
                    arrays[kind], left, top = crop_base._read_padded(
                        planes, CHANNELS[kind], x, y, half, width, height
                    )
                images = crop_base._raw_images(arrays['dic'], arrays['gfp'], arrays['rfp'], ranges)
                labelled = labelled_four_panel(
                    images, condition_id=condition_id, well=well, pdos=objects, rfp=rfp,
                    validation=validation, left=left, top=top, panel_size=args.panel_size,
                )
                labelled_path.parent.mkdir(parents=True, exist_ok=True)
                labelled.save(labelled_path, dpi=(300, 300))
                labelled.close()
                for image in images.values():
                    image.close()
                row.update({
                    'pixel_size_x_um': validation['pixel_size_um']['x'],
                    'pixel_size_y_um': validation['pixel_size_um']['y'],
                    'omezarr_source': str(zarr_path), 'crop_left_px_fullres': left,
                    'crop_top_px_fullres': top, 'crop_side_px': 2 * half + 1,
                    'crop_radius_scale': args.crop_radius_scale,
                    'display_ranges_json': json.dumps(ranges, sort_keys=True),
                    'restart_signature': signature, 'labelled_crop': str(labelled_path),
                    'export_status': 'completed', 'export_error': '',
                })
            except Exception as exc:
                row.update({
                    'restart_signature': signature, 'labelled_crop': str(labelled_path),
                    'export_status': 'failed', 'export_error': f'{type(exc).__name__}: {exc}',
                })
            rows.append(row)
        future_ids = [_normalise_well_id(item['well_id']) for item in ordered[index + 1:]]
        _atomic_csv(manifest_path, rows + [prior_rows[item] for item in future_ids
                                          if item in prior_rows])
    completed_ids = {
        _normalise_well_id(row['well_id']) for row in rows
        if row.get('export_status') == 'completed' and Path(row.get('labelled_crop', '')).is_file()
    }
    manifest_ids = {_normalise_well_id(row['well_id']) for row in rows
                    if row.get('export_status') == 'completed'}
    qc_ok = completed_ids == expected_ids and manifest_ids == expected_ids and len(rows) == len(expected_ids)
    contact_paths = crop_base._contact_sheets(
        [Path(row['labelled_crop']) for row in rows if row.get('export_status') == 'completed'],
        output / 'contact_sheets', args.contact_sheet_size,
    )
    summary = {
        'export_version': EXPORT_VERSION, 'condition_id': condition_id,
        'condition_name': CONDITIONS[condition_id]['condition_name'],
        'lane': CONDITIONS[condition_id]['lane'], 'dose': CONDITIONS[condition_id]['dose'],
        'started_at': started, 'completed_at': _now(),
        'status': 'completed' if qc_ok else 'failed_qc',
        'expected_pdo_positive_wells_from_csv': len(expected_ids),
        'unique_exported_well_ids': len(completed_ids),
        'exported_ids_equal_csv_pdo_positive_ids': completed_ids == expected_ids,
        'completed_manifest_ids_equal_expected_ids': manifest_ids == expected_ids,
        'all_labelled_crop_files_exist': completed_ids == manifest_ids,
        'continuous_rfp_well_set_equals_all_final_wells': True,
        'no_duplicate_output_filenames': len(filenames) == len(set(filenames)),
        'omezarr_source': str(zarr_path), 'omezarr_validation': validation,
        'channel_mapping': CHANNELS, 'display_ranges': ranges,
        'display_scaling_notice': ('GFP and RFP use one condition-consistent range; no local '
                                   'Round-2 enhancement is used.'),
        'psc_cell_count_status': 'NOT VALIDATED',
        'psc_candidate_notice': ('No Round-1 or Round-2 PSC candidates, outlines, object counts, '
                                 'or inferred PSC cell counts are used.'),
        'overlay_provenance': ('PDO overlays are reconstructed from validated centroids and '
                               'equivalent diameters; not original segmentation masks.'),
        'source_files': ['well_measurements.csv', 'pdo_measurements.csv',
                         'psc_quantification/psc_well_measurements.csv'],
        'crop_settings': {'crop_radius_scale': args.crop_radius_scale,
                          'panel_size': args.panel_size,
                          'contact_sheet_wells_per_page': args.contact_sheet_size},
        'contact_sheets': contact_paths,
        'failed_wells': [row['well_id'] for row in rows if row.get('export_status') != 'completed'],
    }
    _atomic_json(output / 'crop_export_summary.json', summary)
    if args.upload_s3:
        if s3_client is None:
            from nd2_s3_stage import get_s3_client
            s3_client = get_s3_client(region_name=args.region)
        prefix = '/'.join(value.strip('/') for value in
                          (args.results_s3_prefix, condition_id, OUTPUT_DIRECTORY)
                          if value.strip('/'))
        summary['s3_upload'] = crop_base._upload_additive(s3_client, output, args.bucket, prefix)
        if summary['s3_upload']['conflicting_files']:
            summary['status'] = 'completed_with_s3_conflicts' if qc_ok else 'failed_qc'
        _atomic_json(output / 'crop_export_summary.json', summary)
    return summary


def combine_manifests(result_root: Path) -> list[dict]:
    rows = []
    for condition_id in CONDITIONS:
        output = result_root / condition_id / OUTPUT_DIRECTORY
        path = output / 'manifest.csv'
        summary_path = output / 'crop_export_summary.json'
        if not path.is_file() or not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if (summary.get('status') not in {'completed', 'completed_with_s3_conflicts'}
                or not summary.get('exported_ids_equal_csv_pdo_positive_ids')
                or not summary.get('completed_manifest_ids_equal_expected_ids')):
            continue
        rows.extend(row for row in _read_csv(path) if row.get('export_status') == 'completed')
    _atomic_csv(result_root / COMBINED_MANIFEST, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Presentation-only exporter for validated PDO and continuous-RFP results. '
                    'Does not run image analysis or PSC object counting.'
    )
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
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
    batch_status_path = result_root / 'batch_status.json'
    batch_status = _read_json(batch_status_path) if batch_status_path.is_file() else {}
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
                  f"{summary['expected_pdo_positive_wells_from_csv']} wells)", flush=True)
        except Exception as exc:
            failures += 1
            output = result_root / condition_id / OUTPUT_DIRECTORY
            _atomic_json(output / 'crop_export_summary.json', {
                'export_version': EXPORT_VERSION, 'condition_id': condition_id,
                'condition_name': CONDITIONS[condition_id]['condition_name'],
                'lane': CONDITIONS[condition_id]['lane'], 'dose': CONDITIONS[condition_id]['dose'],
                'status': 'failed', 'failed_at': _now(),
                'psc_cell_count_status': 'NOT VALIDATED',
                'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(),
            })
            print(f'{condition_id}: FAILED: {type(exc).__name__}: {exc}', flush=True)
        finally:
            combine_manifests(result_root)
    if not failures:
        print(f'Final PDO/RFP crop export completed: {len(selected)}/{len(selected)} conditions; '
              'all exported well-ID QC checks passed; PSC cell count remains NOT VALIDATED.',
              flush=True)
    return 1 if failures else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
