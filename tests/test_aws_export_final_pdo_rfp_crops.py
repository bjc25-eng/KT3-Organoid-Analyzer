from __future__ import annotations

import ast
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

import aws_export_final_pdo_rfp_crops as exporter


PIXEL_SIZE = exporter.EXPECTED_PIXEL_SIZE_UM


def _write_dict_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FinalPdoRfpCropExporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = self.root / 'results'
        self.cache = self.root / 'cache'

    def tearDown(self):
        self.temporary.cleanup()

    def _make_zarr(self, condition_id: str, *, labels=None, windows=True) -> Path:
        path = self.cache / f'{condition_id}.ome.zarr'
        root = zarr.open_group(str(path), mode='w')
        data = np.zeros((1, 3, 80, 84), dtype=np.uint16)
        yy, xx = np.indices(data.shape[-2:])
        data[0, 0] = xx * 6 + yy
        data[0, 1] = xx * 3 + yy * 2
        data[0, 2] = xx + yy * 4
        root.create_dataset('0', data=data, chunks=(1, 1, 20, 21))
        channel_rows = []
        for index, label in enumerate(labels or ['GFP 488', 'RFP 561', 'DIC transmitted']):
            row = {'label': label}
            if windows:
                row['window'] = {'start': 0, 'end': 900 + index * 100}
            channel_rows.append(row)
        root.attrs['omero'] = {'channels': channel_rows}
        root.attrs['multiscales'] = [{
            'axes': [{'name': 't'}, {'name': 'c'},
                     {'name': 'y', 'unit': 'micrometer'},
                     {'name': 'x', 'unit': 'micrometer'}],
            'datasets': [{'path': '0', 'coordinateTransformations': [{
                'type': 'scale', 'scale': [1, 1, PIXEL_SIZE, PIXEL_SIZE],
            }]}],
        }]
        return path

    def _rfp_row(self, condition_id: str, well: dict, *, insufficient=False) -> dict:
        mapping = exporter.CONDITIONS[condition_id]
        row = {field: 1 for field in exporter.RFP_SOURCE_FIELDS}
        row.update({
            'condition_id': condition_id, 'condition_name': f'{condition_id}_original',
            'dose_nM': mapping['dose_nM'], 'well_id': str(well['well_id']),
            'x_px_fullres': well['x_px_fullres'], 'y_px_fullres': well['y_px_fullres'],
            'radius_px': well['radius_px'], 'x_mm': 0.01, 'y_mm': 0.02,
            'RFP_channel': 1, 'RFP_source_dtype': 'uint16',
            'interior_radius_fraction': 0.86, 'interior_radius_px': 5.16,
            'interior_pixel_count': 84, 'background_inner_radius_fraction': 1.15,
            'background_outer_radius_fraction': 1.45,
            'neighbour_exclusion_radius_fraction': 1.05,
            'background_valid_pixel_count': 700, 'background_expected_pixel_count': 800,
            'background_valid_fraction': 0.875,
            'background_qc': ('insufficient_local_background' if insufficient
                              else 'valid_local_background'),
            'RFP_mean_intensity': 30.0, 'RFP_median_intensity': 29.0,
            'RFP_max_intensity': 110.0, 'RFP_integrated_intensity': 2520.0,
            'RFP_p90': 70.0, 'RFP_p95': 90.0, 'RFP_p99': 105.0,
            'RFP_saturated_pixel_count': 0, 'RFP_saturated_pixel_fraction': 0.0,
            'RFP_background_mean': 18.0, 'RFP_background_median': 17.5,
            'RFP_background_p95': 25.0, 'RFP_background_p99': 28.0,
            'RFP_background_corrected_mean': 12.5,
            'RFP_background_corrected_integrated_intensity': 1050.0,
            'RFP_positive_only_excess_integrated_intensity': 1100.0,
            'exploratory_RFP_threshold_intensity': 28.0,
            'exploratory_RFP_positive_area_px2': 6,
            'exploratory_RFP_positive_area_um2': 3.25,
            'exploratory_RFP_positive_fraction': 6 / 84,
            'quantification_status': 'completed', 'error': '',
        })
        if insufficient:
            for field in (
                'RFP_background_mean', 'RFP_background_median', 'RFP_background_p95',
                'RFP_background_p99', 'RFP_background_corrected_mean',
                'RFP_background_corrected_integrated_intensity',
                'RFP_positive_only_excess_integrated_intensity',
                'exploratory_RFP_threshold_intensity', 'exploratory_RFP_positive_area_px2',
                'exploratory_RFP_positive_area_um2', 'exploratory_RFP_positive_fraction',
            ):
                row[field] = float('nan')
        return row

    def _make_condition(self, condition_id: str, *, positive_count=1, total_wells=2,
                        insufficient_ids=(), omit_rfp_ids=(), duplicate_rfp_id=None,
                        bad_pdo_count=False, windows=True, labels=None) -> Path:
        self._make_zarr(condition_id, windows=windows, labels=labels)
        folder = self.results / condition_id
        folder.mkdir(parents=True, exist_ok=True)
        wells = []
        pdos = []
        rfp_rows = []
        for index in range(total_wells):
            well_id = index + 1
            positive = index < positive_count
            well = {
                'well_id': str(well_id), 'x_px_fullres': 18 + index * 18,
                'y_px_fullres': 20 + index * 14, 'radius_px': 6,
                'PDO_count': 1 if positive else 0, 'PDO_present': positive,
                'total_PDO_projected_area_px2': 20 if positive else 0,
                'total_PDO_projected_area_um2': 10.743 if positive else 0,
                'qc_status': 'automated_dominant_hex_array_not_manually_reviewed',
            }
            wells.append(well)
            if positive:
                pdos.append({
                    'well_id': str(well_id), 'pdo_number_in_well': 1,
                    'centroid_x_px_fullres': well['x_px_fullres'] + 0.5,
                    'centroid_y_px_fullres': well['y_px_fullres'] + 0.5,
                    'projected_area_px2': 20, 'projected_area_um2': 10.743,
                    'equivalent_circular_diameter_um': 3.7,
                })
            if well_id not in omit_rfp_ids:
                rfp_rows.append(self._rfp_row(
                    condition_id, well, insufficient=well_id in insufficient_ids
                ))
        if bad_pdo_count and pdos:
            pdos.append({**pdos[0], 'pdo_number_in_well': 2})
        if duplicate_rfp_id is not None:
            source = next(row for row in rfp_rows if int(row['well_id']) == duplicate_rfp_id)
            rfp_rows.append(dict(source))
        _write_dict_csv(folder / 'well_measurements.csv', list(wells[0]), wells)
        _write_dict_csv(folder / 'pdo_measurements.csv', [
            'well_id', 'pdo_number_in_well', 'centroid_x_px_fullres',
            'centroid_y_px_fullres', 'projected_area_px2', 'projected_area_um2',
            'equivalent_circular_diameter_um',
        ], pdos)
        _write_dict_csv(folder / 'psc_quantification' / 'psc_well_measurements.csv',
                        exporter.RFP_SOURCE_FIELDS, rfp_rows)
        return folder

    def _args(self, *extra: str):
        return exporter.build_parser().parse_args([
            '--result-root', str(self.results), '--cache-root', str(self.cache),
            '--panel-size', '96', '--display-sample-size', '16',
            '--display-sample-grid', '2', *extra,
        ])

    def test_exact_valid_background_header_and_units(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition)
        wells, pdos, rfp = exporter._condition_inputs(folder)
        lines = exporter.header_lines(condition, wells[0], pdos, rfp[0])
        self.assertEqual(lines[0],
                         'Lane 1 | RMC6236 0 nM | Final well 1 | PDO POSITIVE')
        self.assertIn('PDO count: 1', lines)
        self.assertIn('PDO size(s): 3.7 µm', lines)
        self.assertIn('Total PDO projected area: 10.7 µm²', lines)
        self.assertIn('PSC/RFP background-corrected mean: 12.500 detector units', lines)
        self.assertIn('PSC/RFP background-corrected integrated intensity: '
                      '1,050.000 detector units·pixels', lines)
        self.assertIn('Raw RFP p95: 90.000 detector units', lines)
        self.assertIn('Exploratory RFP-positive area: 3.250 µm²', lines)
        self.assertIn('PSC cell count: NOT VALIDATED', lines)
        self.assertTrue(any(line.startswith('Full-resolution x/y:') for line in lines))
        self.assertTrue(any(line.startswith('Well radius:') for line in lines))
        self.assertTrue(any(line.startswith('Final-array QC status:') for line in lines))

    def test_insufficient_background_is_not_quantified_never_zero(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, insufficient_ids={1})
        wells, pdos, rfp = exporter._condition_inputs(folder)
        lines = exporter.header_lines(condition, wells[0], pdos, rfp[0])
        self.assertIn('PSC/RFP background-corrected signal: not quantified', lines)
        self.assertIn('Exploratory RFP-positive area: not quantified', lines)
        self.assertIn('Background QC: insufficient_local_background', lines)
        self.assertIn('Raw RFP p95: 90.000 detector units', lines)
        self.assertFalse(any('background-corrected mean: 0' in line for line in lines))
        self.assertFalse(any('RFP-positive area: 0' in line for line in lines))

    def test_insufficient_background_with_finite_zero_is_rejected(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, insufficient_ids={1})
        path = folder / 'psc_quantification' / 'psc_well_measurements.csv'
        rows = exporter._read_csv(path)
        rows[0]['RFP_background_corrected_mean'] = 0
        _write_dict_csv(path, exporter.RFP_SOURCE_FIELDS, rows)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 1)
        summary = json.loads((folder / exporter.OUTPUT_DIRECTORY /
                              'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertIn('finite background-derived values instead of NaN', summary['error'])

    def test_exports_exact_dynamic_positive_set_and_complete_manifest_schema(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, positive_count=2, total_wells=3)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 0)
        manifest_path = folder / exporter.OUTPUT_DIRECTORY / 'manifest.csv'
        with manifest_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), exporter.MANIFEST_FIELDS)
        self.assertEqual({row['well_id'] for row in rows}, {'1', '2'})
        self.assertTrue(all(row['psc_cell_count_status'] == 'NOT VALIDATED' for row in rows))
        self.assertTrue(all(Path(row['labelled_crop']).is_file() for row in rows))
        self.assertTrue(all(row['RFP_p95'] == '90.0' for row in rows))
        summary = json.loads((folder / exporter.OUTPUT_DIRECTORY /
                              'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['expected_pdo_positive_wells_from_csv'], 2)
        self.assertTrue(summary['exported_ids_equal_csv_pdo_positive_ids'])
        self.assertTrue(summary['completed_manifest_ids_equal_expected_ids'])
        self.assertTrue(summary['all_labelled_crop_files_exist'])
        self.assertEqual(summary['psc_cell_count_status'], 'NOT VALIDATED')
        self.assertEqual(set(path.name for path in (folder / exporter.OUTPUT_DIRECTORY).iterdir()),
                         {'labelled_crops', 'contact_sheets', 'manifest.csv',
                          'crop_export_summary.json'})
        with Image.open(rows[0]['labelled_crop']) as image:
            self.assertGreater(image.height, image.width)

    def test_fixture_counts_are_derived_and_combined(self):
        first, second = list(exporter.CONDITIONS)[:2]
        first_folder = self._make_condition(first, positive_count=2, total_wells=3)
        second_folder = self._make_condition(second, positive_count=1, total_wells=2)
        self.assertEqual(exporter.run(self._args('--condition-id', first,
                                                '--condition-id', second)), 0)
        counts = {}
        for condition, folder in ((first, first_folder), (second, second_folder)):
            summary = json.loads((folder / exporter.OUTPUT_DIRECTORY /
                                  'crop_export_summary.json').read_text(encoding='utf-8'))
            counts[condition] = summary['expected_pdo_positive_wells_from_csv']
        self.assertEqual(counts, {first: 2, second: 1})
        combined = exporter._read_csv(self.results / exporter.COMBINED_MANIFEST)
        self.assertEqual(len(combined), 3)
        self.assertEqual({row['condition_id'] for row in combined}, {first, second})

    def test_continuous_rfp_must_match_all_final_wells_and_batch_continues(self):
        first, second = list(exporter.CONDITIONS)[:2]
        first_folder = self._make_condition(first, omit_rfp_ids={2})
        second_folder = self._make_condition(second)
        self.assertEqual(exporter.run(self._args('--condition-id', first,
                                                '--condition-id', second)), 1)
        failed = json.loads((first_folder / exporter.OUTPUT_DIRECTORY /
                             'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertIn('Continuous-RFP/final well-set mismatch', failed['error'])
        passed = json.loads((second_folder / exporter.OUTPUT_DIRECTORY /
                             'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(passed['status'], 'completed')

    def test_duplicate_rfp_row_is_rejected(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, duplicate_rfp_id=1)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 1)
        summary = json.loads((folder / exporter.OUTPUT_DIRECTORY /
                              'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertIn('Duplicate well_id 1', summary['error'])

    def test_pdo_count_must_equal_pdo_rows(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, bad_pdo_count=True)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 1)
        summary = json.loads((folder / exporter.OUTPUT_DIRECTORY /
                              'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertIn('Final PDO count mismatch', summary['error'])

    def test_final_array_qc_status_is_required(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition)
        path = folder / 'well_measurements.csv'
        rows = exporter._read_csv(path)
        rows[0]['qc_status'] = ''
        _write_dict_csv(path, list(rows[0]), rows)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 1)
        summary = json.loads((folder / exporter.OUTPUT_DIRECTORY /
                              'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertIn('lacks a final-array qc_status', summary['error'])

    def test_restart_reuses_matching_completed_crop(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition)
        args = self._args('--condition-id', condition)
        self.assertEqual(exporter.run(args), 0)
        row = exporter._read_csv(folder / exporter.OUTPUT_DIRECTORY / 'manifest.csv')[0]
        crop = Path(row['labelled_crop'])
        modified = crop.stat().st_mtime_ns
        self.assertEqual(exporter.run(args), 0)
        self.assertEqual(crop.stat().st_mtime_ns, modified)

    def test_fluorescence_display_range_is_condition_consistent_not_local(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, positive_count=2, total_wells=2, windows=False)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 0)
        rows = exporter._read_csv(folder / exporter.OUTPUT_DIRECTORY / 'manifest.csv')
        self.assertEqual(len({row['display_ranges_json'] for row in rows}), 1)
        ranges = json.loads(rows[0]['display_ranges_json'])
        self.assertEqual(ranges['gfp']['source'], 'condition_wide_sample_percentiles_0.5_99.5')
        self.assertEqual(ranges['rfp']['source'], 'condition_wide_sample_percentiles_0.5_99.5')
        summary = json.loads((folder / exporter.OUTPUT_DIRECTORY /
                              'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertIn('no local Round-2 enhancement', summary['display_scaling_notice'])

    def test_channel_mapping_is_validated_without_guessing(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, labels=['RFP', 'GFP', 'DIC'])
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 1)
        summary = json.loads((folder / exporter.OUTPUT_DIRECTORY /
                              'crop_export_summary.json').read_text(encoding='utf-8'))
        self.assertIn('Cannot validate GFP=0', summary['error'])

    def test_failed_current_summary_excludes_stale_manifest_from_combined(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition)
        args = self._args('--condition-id', condition)
        self.assertEqual(exporter.run(args), 0)
        self.assertEqual(len(exporter._read_csv(self.results / exporter.COMBINED_MANIFEST)), 1)
        rfp_path = folder / 'psc_quantification' / 'psc_well_measurements.csv'
        rows = exporter._read_csv(rfp_path)
        rows.pop()
        _write_dict_csv(rfp_path, exporter.RFP_SOURCE_FIELDS, rows)
        self.assertEqual(exporter.run(args), 1)
        self.assertEqual(exporter._read_csv(self.results / exporter.COMBINED_MANIFEST), [])

    def test_upload_is_opt_in_and_legacy_outputs_are_untouched(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition)
        sentinels = []
        for name in ('pdo_positive_crops', 'pdo_positive_crops_with_psc',
                     'psc_object_quantification'):
            path = folder / name / 'sentinel.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('unchanged', encoding='utf-8')
            sentinels.append(path)

        class ForbiddenClient:
            def __getattr__(self, name):
                raise AssertionError(f'automatic S3 use attempted: {name}')

        self.assertEqual(exporter.run(self._args('--condition-id', condition),
                                      s3_client=ForbiddenClient()), 0)
        self.assertTrue(all(path.read_text(encoding='utf-8') == 'unchanged'
                            for path in sentinels))

    def test_no_analysis_or_psc_candidate_call_route_exists(self):
        source_path = Path(exporter.__file__)
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        prohibited = {
            'detect_psc', 'detect_wells', 'hough_circle', 'segment_pdos',
            'quantify_well', 'convert_nd2', 'refine_lattice',
        }
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        called.update({
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        })
        self.assertFalse(called & prohibited)
        self.assertNotIn('PSC_like_resolved_object_count', source)
        self.assertNotIn('locally enhanced', source.lower())
        self.assertEqual(exporter.OUTPUT_DIRECTORY, 'pdo_positive_crops_final_pdo_rfp')


if __name__ == '__main__':
    unittest.main()
