import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest

import aws_finalize_pdo_analysis_qc as finalizer


CONDITIONS = tuple(finalizer.presentation.CONDITIONS)
DMSO = CONDITIONS[0]


def write_csv(path: Path, rows: list[dict], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def authoritative(condition, well_id, number, fraction, *, centroid=True, many=False,
                  match='trusted_complete_match', extent='complete', crossing=False,
                  touching=False, area=20.0):
    return {
        'condition_id': condition, 'condition_name': condition, 'dose': 'fixture',
        'well_id': str(well_id), 'pdo_number_in_well': str(number),
        'production_PDO_projected_area_px2': str(area),
        'production_PDO_projected_area_um2': str(area / 2),
        'production_PDO_equivalent_circular_diameter_um': '5.5',
        'unmasked_component_match_status': match,
        'full_component_extent_status': extent,
        'full_component_fraction_inside_final_well': '' if fraction is None else str(fraction),
        'full_component_centroid_inside_final_well': '' if centroid is None else str(centroid),
        'many_production_PDOs_to_one_unmasked_component': '' if many is None else str(many),
        'full_component_crosses_final_well_boundary': str(crossing),
        'full_component_touches_final_well_boundary': str(touching),
        'hex_array_member': 'True', 'lattice_degree': '6',
    }


def production_pdo(well_id, number, area=20.0):
    return {
        'well_id': str(well_id), 'pdo_number_in_well': str(number),
        'centroid_x_px_fullres': str(50 + number),
        'centroid_y_px_fullres': str(50 + number),
        'projected_area_px2': str(area), 'projected_area_um2': str(area / 2),
        'equivalent_circular_diameter_um': '5.5',
    }


def well(well_id, pdo_count, area=20.0):
    return {
        'well_id': str(well_id), 'x_px_fullres': '50.0', 'y_px_fullres': '50.0',
        'radius_px': '15.0', 'PDO_present': str(pdo_count > 0), 'PDO_count': str(pdo_count),
        'total_PDO_projected_area_px2': str(area if pdo_count else 0),
        'total_PDO_projected_area_um2': str(area / 2 if pdo_count else 0),
        'hex_array_member': 'True', 'lattice_degree': '6',
    }


def rfp(condition, well_id, *, nan='NaN', blank=''):
    return {
        'condition_id': condition, 'well_id': str(well_id),
        'x_px_fullres': '50.0', 'y_px_fullres': '50.0', 'radius_px': '15.0',
        'background_qc': 'sufficient_local_background',
        'RFP_background_corrected_mean': '0012.340000',
        'RFP_background_corrected_integrated_intensity': '123456.7890000000',
        'RFP_p95': '98.765432100', 'exploratory_RFP_positive_area_um2': '4.5000',
        'nan_semantics_fixture': nan, 'blank_semantics_fixture': blank,
    }


def build_fixture(root: Path):
    auth = []
    layout = {
        DMSO: [
            (well(606, 1), [production_pdo(606, 1)],
             [authoritative(DMSO, 606, 1, 0.08143293627950228)]),
            (well(624, 1), [production_pdo(624, 1)],
             [authoritative(DMSO, 624, 1, 0.39954833281672053)]),
            (well(700, 2, 50), [production_pdo(700, 1, 20), production_pdo(700, 2, 30)],
             [authoritative(DMSO, 700, 1, 0.80, area=20),
              authoritative(DMSO, 700, 2, 0.50, area=30)]),
            (well(701, 0, 0), [], []),
        ],
    }
    failure_variants = [
        {'fraction': .60, 'crossing': True, 'touching': True},
        {'fraction': .80, 'centroid': False},
        {'fraction': .80, 'many': True},
        {'fraction': None, 'centroid': None, 'many': None, 'match': 'failed',
         'extent': 'not_evaluated'},
        {'fraction': None, 'centroid': None, 'many': None, 'match': 'failed',
         'extent': 'incomplete_at_maximum_crop'},
    ]
    for index, condition in enumerate(CONDITIONS[1:]):
        wid = 100 + index
        settings = failure_variants[index]
        layout[condition] = [
            (well(wid, 1), [production_pdo(wid, 1)],
             [authoritative(condition, wid, 1, **settings)]),
            (well(wid + 1000, 0, 0), [], []),
        ]
    for condition, entries in layout.items():
        wells, pdos, rfps = [], [], []
        for source_well, source_pdos, source_auth in entries:
            wells.append(source_well)
            pdos.extend(source_pdos)
            rfps.append(rfp(condition, source_well['well_id']))
            auth.extend(source_auth)
        folder = root / condition
        write_csv(folder / 'well_measurements.csv', wells)
        # Keep a stable PDO schema even for a hypothetical empty condition.
        pdo_fields = list(production_pdo(1, 1))
        write_csv(folder / 'pdo_measurements.csv', pdos, pdo_fields)
        write_csv(folder / 'psc_quantification' / 'psc_well_measurements.csv', rfps)
    write_csv(root / finalizer.AUTHORITATIVE_COMPONENT_CSV, auth)
    return auth


def args(root):
    return Namespace(
        result_root=root, cache_root=root / 'cache', expected_pixel_size_um=0.733,
        crop_radius_scale=1.75, panel_size=128, contact_sheet_size=16,
        display_sample_size=32, display_sample_grid=2,
    )


def fake_crop_exporter(result_root, final_root, prepared, final_maps, passing, args,
                       *, probe=None, open_group=None):
    checks = {}
    for condition in CONDITIONS:
        output = final_root / 'pdo_positive_crops' / condition
        positive = [row for row in final_maps[condition].values()
                    if finalizer._truthy(row['PDO_present'])]
        manifest = []
        for source_well in positive:
            well_id = finalizer._well_id(source_well['well_id'])
            identities = sorted(key for key in passing if key[:2] == (condition, well_id))
            path = output / 'labelled_crops' / f'{condition}_{well_id}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
            manifest.append({
                'condition_id': condition, 'well_id': well_id,
                'retained_PDO_identities': ';'.join('|'.join(map(str, key)) for key in identities),
                'labelled_crop': str(path), 'export_status': 'completed',
            })
        finalizer._atomic_csv(output / 'manifest.csv', manifest, finalizer.CROP_MANIFEST_FIELDS)
        sheets = output / 'contact_sheets'
        sheets.mkdir(parents=True, exist_ok=True)
        if positive:
            (sheets / 'page_001.png').write_bytes(b'fixture')
        checks[condition] = {
            'crop_ids_equal_final_positive_ids': True,
            'manifest_ids_equal_final_positive_ids': True,
            'crop_overlay_identities_equal_passing_PDO_identities': True,
            'expected_positive_wells': len(positive), 'completed_crops': len(positive),
        }
    return checks


def test_frozen_rule_is_inclusive_and_crossing_touching_do_not_reject():
    row = authoritative(DMSO, 1, 1, .60, crossing=True, touching=True)
    assert finalizer.final_object_decision(row) == (True, [])
    below = authoritative(DMSO, 1, 1, .599999)
    assert finalizer.final_object_decision(below) == (False, ['containment_below_0p60'])


def test_failure_reasons_are_independent_and_deterministically_ordered():
    row = authoritative(DMSO, 1, 1, .2, centroid=False, many=True,
                        match='failed', extent='incomplete_at_maximum_crop')
    accepted, reasons = finalizer.final_object_decision(row)
    assert accepted is False
    assert reasons == list(finalizer.FAILURE_REASONS)


def test_malformed_required_trusted_values_are_integrity_errors():
    row = authoritative(DMSO, 1, 1, None)
    with pytest.raises(RuntimeError, match='lacks required final-QC values'):
        finalizer.final_object_decision(row)
    row = authoritative(DMSO, 1, 1, .8)
    row['full_component_centroid_inside_final_well'] = 'perhaps'
    with pytest.raises(RuntimeError, match='Malformed boolean'):
        finalizer.final_object_decision(row)


def test_authoritative_identity_and_dominant_array_integrity(tmp_path):
    auth = build_fixture(tmp_path)
    prepared = finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth))
    assert len(prepared['authoritative_map']) == len(auth)
    first = CONDITIONS[0]
    rows = read_csv(tmp_path / first / 'well_measurements.csv')
    rows[0]['hex_array_member'] = 'False'
    write_csv(tmp_path / first / 'well_measurements.csv', rows)
    with pytest.raises(RuntimeError, match='truthy hex_array_member'):
        finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth))


def test_authoritative_and_original_pdo_identity_sets_must_match(tmp_path):
    auth = build_fixture(tmp_path)
    path = tmp_path / DMSO / 'pdo_measurements.csv'
    rows = read_csv(path)
    rows[0]['pdo_number_in_well'] = '99'
    write_csv(path, rows)
    with pytest.raises(RuntimeError, match='Original PDO_count mismatch|identities differ'):
        finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth))


def test_known_failures_606_and_624_fail_with_containment_reason(tmp_path):
    auth = build_fixture(tmp_path)
    prepared = finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth))
    rows, passing = finalizer.apply_object_rule(prepared)
    selected = {(row['well_id'], row['final_PDO_QC_pass'], row['final_PDO_QC_failure_reasons'])
                for row in rows if row['condition_id'] == DMSO and row['well_id'] in {'606', '624'}}
    assert selected == {
        ('606', False, 'containment_below_0p60'),
        ('624', False, 'containment_below_0p60'),
    }
    assert not any(key[0] == DMSO and key[1] in {'606', '624'} for key in passing)


def test_missing_known_failure_aborts(tmp_path):
    auth = build_fixture(tmp_path)
    rows = read_csv(tmp_path / finalizer.AUTHORITATIVE_COMPONENT_CSV)
    rows = [row for row in rows if not (row['condition_id'] == DMSO and row['well_id'] == '624')]
    write_csv(tmp_path / finalizer.AUTHORITATIVE_COMPONENT_CSV, rows)
    # Remove its original row and reconcile the source count so absence reaches the mandatory check.
    pdo_path = tmp_path / DMSO / 'pdo_measurements.csv'
    pdos = [row for row in read_csv(pdo_path) if row['well_id'] != '624']
    write_csv(pdo_path, pdos, list(production_pdo(1, 1)))
    well_path = tmp_path / DMSO / 'well_measurements.csv'
    wells = read_csv(well_path)
    for source in wells:
        if source['well_id'] == '624':
            source.update(PDO_present='False', PDO_count='0',
                          total_PDO_projected_area_px2='0', total_PDO_projected_area_um2='0')
    write_csv(well_path, wells)
    prepared = finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth)-1)
    with pytest.raises(RuntimeError, match='well 624 is absent'):
        finalizer.apply_object_rule(prepared)


def test_mixed_pass_well_retains_passing_sibling_and_full_denominator(tmp_path):
    auth = build_fixture(tmp_path)
    prepared = finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth))
    objects, passing = finalizer.apply_object_rule(prepared)
    wells, maps = finalizer.recompute_wells(prepared, passing)
    mixed = maps[DMSO]['700']
    assert mixed['PDO_present'] is True
    assert mixed['PDO_count'] == 1
    assert mixed['total_PDO_projected_area_px2'] == 20
    assert mixed['total_PDO_projected_area_um2'] == 10
    assert mixed['original_PDO_count'] == '2'
    assert len(wells) == sum(len(source['wells']) for source in prepared['conditions'].values())
    assert maps[DMSO]['701']['PDO_present'] is False
    assert maps[DMSO]['701']['original_PDO_present'] == 'False'


def test_rfp_source_cells_are_preserved_verbatim_including_nan_and_blank(tmp_path):
    auth = build_fixture(tmp_path)
    prepared = finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth))
    _, passing = finalizer.apply_object_rule(prepared)
    wells, _ = finalizer.recompute_wells(prepared, passing)
    fields = finalizer._ordered_union(prepared['well_fields'], prepared['rfp_fields'],
                                      finalizer.FINAL_WELL_EXTRA_FIELDS)
    output = tmp_path / 'verbatim.csv'
    finalizer._atomic_csv(output, wells, fields)
    result = finalizer._verify_rfp_verbatim(output, prepared)
    assert result['passed'] is True
    row = next(row for row in read_csv(output)
               if row['condition_id'] == DMSO and row['well_id'] == '606')
    assert row['RFP_background_corrected_mean'] == '0012.340000'
    assert row['nan_semantics_fixture'] == 'NaN'
    assert row['blank_semantics_fixture'] == ''


def test_rfp_post_write_mismatch_fails(tmp_path):
    auth = build_fixture(tmp_path)
    prepared = finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth))
    _, passing = finalizer.apply_object_rule(prepared)
    wells, _ = finalizer.recompute_wells(prepared, passing)
    fields = finalizer._ordered_union(prepared['well_fields'], prepared['rfp_fields'],
                                      finalizer.FINAL_WELL_EXTRA_FIELDS)
    output = tmp_path / 'verbatim.csv'
    finalizer._atomic_csv(output, wells, fields)
    rows = read_csv(output)
    rows[0]['RFP_p95'] = '98.7654321'
    write_csv(output, rows, fields)
    with pytest.raises(RuntimeError, match='Verbatim RFP mismatch'):
        finalizer._verify_rfp_verbatim(output, prepared)


def test_condition_summary_has_six_plus_combined_and_reconciles(tmp_path):
    auth = build_fixture(tmp_path)
    prepared = finalizer.validate_and_prepare(tmp_path, expected_pdo_rows=len(auth))
    objects, passing = finalizer.apply_object_rule(prepared)
    _, maps = finalizer.recompute_wells(prepared, passing)
    summaries = finalizer.condition_summaries(prepared, objects, maps)
    assert len(summaries) == 7
    assert summaries[-1]['condition_id'] == 'ALL_CONDITIONS'
    assert finalizer._verify_summary(summaries)
    combined = summaries[-1]
    assert combined['final_PDO_positive_fraction'] == (
        combined['retained_PDO_positive_wells'] / combined['total_final_dominant_array_wells'])


def test_finalization_writes_only_new_tree_and_crop_sets_match(tmp_path):
    auth = build_fixture(tmp_path)
    upstream = tmp_path / 'upstream_sentinel.txt'
    upstream.write_text('unchanged', encoding='utf-8')
    result = finalizer.finalize(tmp_path, args(tmp_path), expected_pdo_rows=len(auth),
                                crop_exporter=fake_crop_exporter)
    output = tmp_path / finalizer.OUTPUT_DIRECTORY
    assert result['completion_status'] == 'completed'
    assert upstream.read_text(encoding='utf-8') == 'unchanged'
    assert {path.name for path in output.iterdir()} == {
        'final_pdo_object_qc.csv', 'final_well_measurements.csv',
        'final_condition_summary.csv', 'final_qc_summary.json', 'pdo_positive_crops',
    }
    assert result['integrity_checks']['DMSO_606_absent_from_retained_crops']
    assert result['integrity_checks']['DMSO_624_absent_from_retained_crops']
    for check in result['integrity_checks']['crop_checks'].values():
        assert check['crop_ids_equal_final_positive_ids']
        assert check['crop_overlay_identities_equal_passing_PDO_identities']


def test_summary_is_not_completed_when_crop_export_fails(tmp_path):
    auth = build_fixture(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError('crop fixture failed')
    with pytest.raises(RuntimeError, match='crop fixture failed'):
        finalizer.finalize(tmp_path, args(tmp_path), expected_pdo_rows=len(auth),
                           crop_exporter=fail)
    summary = json.loads((tmp_path / finalizer.OUTPUT_DIRECTORY /
                          'final_qc_summary.json').read_text(encoding='utf-8'))
    assert summary['completion_status'] == 'failed'


def test_crop_restart_reuse_requires_matching_identity_and_existing_file(tmp_path):
    crop = tmp_path / 'crop.png'
    crop.write_bytes(b'fixture')
    row = {'export_status': 'completed', 'restart_signature': 'exact-state',
           'labelled_crop': str(crop)}
    assert finalizer._crop_restart_valid(row, 'exact-state')
    assert not finalizer._crop_restart_valid(row, 'changed-state')
    crop.unlink()
    assert not finalizer._crop_restart_valid(row, 'exact-state')


def test_final_header_uses_cleaned_pdo_and_validated_psc_wording():
    source_well = well(1, 1)
    source_rfp = rfp(DMSO, 1)
    lines = finalizer._header_lines(DMSO, source_well, [production_pdo(1, 1)], source_rfp)
    assert 'Final PDO count: 1' in lines
    assert 'Retained PDO size(s): 5.5 µm' in lines
    assert 'Retained total PDO projected area: 10.0 µm²' in lines
    assert any(line.startswith('Raw RFP p95:') for line in lines)
    assert any('detector units·pixels' in line for line in lines)
    assert 'PSC cell count: NOT VALIDATED' in lines


def test_insufficient_background_header_never_substitutes_zero():
    source_rfp = rfp(DMSO, 1)
    source_rfp.update(background_qc='insufficient_local_background',
                      RFP_background_corrected_mean='NaN',
                      RFP_background_corrected_integrated_intensity='',
                      exploratory_RFP_positive_area_um2='NaN')
    lines = finalizer._header_lines(DMSO, well(1, 1), [production_pdo(1, 1)], source_rfp)
    assert 'PSC/RFP background-corrected signal: not quantified' in lines
    assert 'Exploratory RFP-positive area: not quantified' in lines
    assert 'Background QC: insufficient_local_background' in lines
    assert not any('0 detector units' in line for line in lines)


def test_source_contains_no_scientific_analysis_calls_or_extra_thresholds():
    source = Path(finalizer.__file__).read_text(encoding='utf-8')
    forbidden = (
        'detect_psc(', 'bioformats2raw', 'quantify_condition_components(',
        'reproduce_production_components(', 'recover_unmasked_component(',
        'match_production_row(', 'mask_containment(', 'hough_circle', 'watershed(',
    )
    assert not any(token in source for token in forbidden)
    assert finalizer.FRACTION_INSIDE_THRESHOLD == .60
    assert 'full_component_crosses_final_well_boundary' not in source.split(
        'def final_object_decision', 1)[1].split('def _condition_sources', 1)[0]
    assert 'full_component_touches_final_well_boundary' not in source.split(
        'def final_object_decision', 1)[1].split('def _condition_sources', 1)[0]


def test_cli_has_no_partial_condition_or_analysis_route():
    parser = finalizer.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert 'condition_id' not in destinations
    assert 'threshold' not in destinations
    assert 'upload_s3' not in destinations
