from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

import aws_export_pdo_positive_crops as exporter


PIXEL_SIZE = exporter.EXPECTED_PIXEL_SIZE_UM


def _write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


class CropExporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / 'results'
        self.cache = self.root / 'cache'

    def tearDown(self):
        self.temp.cleanup()

    def _make_zarr(self, condition_id: str, *, labels=None, windows=True,
                   shape=(1, 3, 48, 52)) -> Path:
        path = self.cache / f'{condition_id}.ome.zarr'
        root = zarr.open_group(str(path), mode='w')
        data = np.zeros(shape, dtype=np.uint16)
        yy, xx = np.indices(shape[-2:])
        data[0, 0] = xx * 10 + yy
        data[0, 1] = xx * 5 + yy * 2
        data[0, 2] = xx + yy * 3
        root.create_dataset('0', data=data, chunks=(1, 1, 16, 16))
        channel_labels = labels or ['GFP 488', 'RFP 561', 'DIC transmitted']
        channel_rows = []
        for index, label in enumerate(channel_labels):
            row = {'label': label}
            if windows:
                row['window'] = {'start': 0, 'end': 1000 + 100 * index}
            channel_rows.append(row)
        root.attrs['omero'] = {'channels': channel_rows}
        root.attrs['multiscales'] = [{
            'axes': [{'name': 't'}, {'name': 'c'},
                     {'name': 'y', 'unit': 'micrometer'},
                     {'name': 'x', 'unit': 'micrometer'}],
            'datasets': [{'path': '0', 'coordinateTransformations': [{
                'type': 'scale', 'scale': [1, 1, PIXEL_SIZE, PIXEL_SIZE]}]}],
        }]
        return path

    def _make_condition(self, condition_id: str, *, zarr_path: Path | None = None,
                        positive=True, bad_count=False, second_positive=False) -> Path:
        folder = self.results / condition_id
        folder.mkdir(parents=True, exist_ok=True)
        zarr_path = zarr_path or self._make_zarr(condition_id)
        summary = {
            'completion_status': 'completed', 'condition_id': condition_id,
            'condition_name': f'{condition_id}_original',
            'pixel_size_um': {'x': PIXEL_SIZE, 'y': PIXEL_SIZE},
            'channel_mapping': {'gfp_channel': 0, 'dic_channel': 2},
            'benchmark': {'source': str(zarr_path)},
        }
        (folder / 'condition_summary.json').write_text(json.dumps(summary), encoding='utf-8')
        well_rows = [[1, 20, 20, 6, 1 if positive else 0, positive,
                      20 if positive else 0, 10.743 if positive else 0]]
        if second_positive:
            well_rows.append([2, 3, 4, 5, 1, True, 15, 8.057])
        _write_csv(folder / 'well_measurements.csv', [
            'well_id', 'x_px_fullres', 'y_px_fullres', 'radius_px', 'PDO_count',
            'PDO_present', 'total_PDO_projected_area_px2',
            'total_PDO_projected_area_um2'], well_rows)
        pdo_rows = []
        if positive:
            pdo_rows.append([1, 1, 20.5, 20.5, 20, 10.743, 3.7])
            if bad_count:
                pdo_rows.append([1, 2, 21.5, 20.5, 10, 5.37, 2.6])
        if second_positive:
            pdo_rows.append([2, 1, 3.0, 4.0, 15, 8.057, 3.2])
        _write_csv(folder / 'pdo_measurements.csv', [
            'well_id', 'pdo_number_in_well', 'centroid_x_px_fullres',
            'centroid_y_px_fullres', 'projected_area_px2', 'projected_area_um2',
            'equivalent_circular_diameter_um'], pdo_rows)
        return folder

    def _args(self, *extra: str):
        return exporter.build_parser().parse_args([
            '--result-root', str(self.results), '--cache-root', str(self.cache),
            '--panel-size', '96', '--display-sample-size', '16',
            '--display-sample-grid', '2', *extra])

    def test_explicit_six_condition_mapping(self):
        self.assertEqual(list(exporter.CONDITIONS), [
            'K3T_PSC_RMC6236_Lane_1_DMSO', 'K3T_PSC_RMC6236_5nm_Lane_2',
            'K3T_PSC_RMC6236_25nm_Lane_3', 'K3T_PSC_RMC6236_50nm_Lane_1',
            'K3T_PSC_RMC6236_100nm_Lane_5', 'K3T_PSC_RMC6236_150nm_Lane_6'])
        self.assertEqual(exporter.CONDITIONS['K3T_PSC_RMC6236_Lane_1_DMSO']['dose'], '0 nM')

    def test_exports_only_final_pdo_present_wells_and_all_panels(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition)
        code = exporter.run(self._args('--condition-id', condition))
        self.assertEqual(code, 0)
        manifest = exporter._read_csv(folder / 'pdo_positive_crops' /
                                      'pdo_positive_crop_manifest.csv')
        self.assertEqual([row['well_id'] for row in manifest], ['1'])
        self.assertEqual(manifest[0]['condition_name'], f'{condition}_original')
        self.assertEqual(manifest[0]['dose'], '0 nM')
        for key in ('labelled_crop', 'raw_dic_crop', 'raw_gfp_crop',
                    'raw_rfp_crop', 'raw_composite_crop'):
            self.assertTrue(Path(manifest[0][key]).is_file())
        with Image.open(manifest[0]['labelled_crop']) as image:
            self.assertGreater(image.height, image.width)

    def test_zero_positive_wells_produces_header_only_manifest_and_passes_qc(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, positive=False)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 0)
        manifest = folder / 'pdo_positive_crops' / 'pdo_positive_crop_manifest.csv'
        self.assertEqual(exporter._read_csv(manifest), [])
        summary = exporter._read_json(folder / 'pdo_positive_crops' / 'crop_export_summary.json')
        self.assertEqual(summary['expected_pdo_positive_wells'], 0)
        self.assertTrue(summary['crop_count_qc_passed'])

    def test_ome_windows_are_used_for_both_fluorescence_channels(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 0)
        summary = exporter._read_json(folder / 'pdo_positive_crops' / 'crop_export_summary.json')
        self.assertEqual(summary['display_ranges']['gfp']['source'], 'ome_omero_channel_window')
        self.assertEqual(summary['display_ranges']['gfp']['maximum'], 1000)
        self.assertEqual(summary['display_ranges']['rfp']['maximum'], 1100)

    def test_missing_windows_get_one_condition_wide_range(self):
        condition = next(iter(exporter.CONDITIONS))
        zarr_path = self._make_zarr(condition, windows=False)
        folder = self._make_condition(condition, zarr_path=zarr_path, second_positive=True)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 0)
        summary = exporter._read_json(folder / 'pdo_positive_crops' / 'crop_export_summary.json')
        self.assertEqual(summary['display_ranges']['gfp']['source'],
                         'condition_wide_sample_percentiles_0.5_99.5')
        rows = exporter._read_csv(folder / 'pdo_positive_crops' /
                                  'pdo_positive_crop_manifest.csv')
        self.assertEqual(len({row['display_ranges_json'] for row in rows}), 1)

    def test_restart_skips_completed_matching_crop(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition)
        args = self._args('--condition-id', condition)
        self.assertEqual(exporter.run(args), 0)
        row = exporter._read_csv(folder / 'pdo_positive_crops' /
                                 'pdo_positive_crop_manifest.csv')[0]
        labelled = Path(row['labelled_crop'])
        before = labelled.stat().st_mtime_ns
        self.assertEqual(exporter.run(args), 0)
        self.assertEqual(labelled.stat().st_mtime_ns, before)

    def test_count_mismatch_fails_one_condition_and_continues(self):
        first, second = list(exporter.CONDITIONS)[:2]
        self._make_condition(first, bad_count=True)
        second_folder = self._make_condition(second)
        code = exporter.run(self._args('--condition-id', first, '--condition-id', second))
        self.assertEqual(code, 1)
        failed = exporter._read_json(self.results / first / 'pdo_positive_crops' /
                                     'crop_export_summary.json')
        self.assertEqual(failed['status'], 'failed')
        self.assertTrue((second_folder / 'pdo_positive_crops' /
                         'pdo_positive_crop_manifest.csv').is_file())

    def test_combined_manifest_contains_completed_conditions(self):
        first, second = list(exporter.CONDITIONS)[:2]
        self._make_condition(first)
        self._make_condition(second)
        self.assertEqual(exporter.run(self._args('--condition-id', first,
                                                '--condition-id', second)), 0)
        rows = exporter._read_csv(self.results /
                                  'all_conditions_pdo_positive_crop_manifest.csv')
        self.assertEqual({row['condition_id'] for row in rows}, {first, second})

    def test_wrong_channel_metadata_fails_without_guessing(self):
        condition = next(iter(exporter.CONDITIONS))
        zarr_path = self._make_zarr(condition, labels=['RFP', 'GFP', 'DIC'])
        self._make_condition(condition, zarr_path=zarr_path)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 1)
        summary = exporter._read_json(self.results / condition / 'pdo_positive_crops' /
                                      'crop_export_summary.json')
        self.assertIn('Cannot validate GFP=0', summary['error'])

    def test_edge_well_crop_is_fixed_square_and_black_padded(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, second_positive=True)
        self.assertEqual(exporter.run(self._args('--condition-id', condition)), 0)
        rows = exporter._read_csv(folder / 'pdo_positive_crops' /
                                  'pdo_positive_crop_manifest.csv')
        edge = next(row for row in rows if row['well_id'] == '2')
        with Image.open(edge['raw_dic_crop']) as image:
            self.assertEqual(image.size, (19, 19))
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

    def test_contact_sheets_are_paginated(self):
        condition = next(iter(exporter.CONDITIONS))
        folder = self._make_condition(condition, second_positive=True)
        self.assertEqual(exporter.run(self._args('--condition-id', condition,
                                                '--contact-sheet-size', '1')), 0)
        pages = sorted((folder / 'pdo_positive_crops' / 'contact_sheets').glob('page_*.png'))
        self.assertEqual([path.name for path in pages], ['page_001.png', 'page_002.png'])

    def test_s3_upload_is_additive_and_reports_conflicts(self):
        local = self.root / 'upload'; local.mkdir()
        crop = local / 'crop.png'; crop.write_bytes(b'first')

        class Missing(Exception):
            def __init__(self):
                self.response = {'Error': {'Code': '404'}}

        class Client:
            def __init__(self):
                self.objects = {}; self.upload_calls = 0
            def head_object(self, Bucket, Key):
                if Key not in self.objects:
                    raise Missing()
                return {'Metadata': self.objects[Key]}
            def upload_file(self, path, bucket, key, ExtraArgs):
                self.upload_calls += 1
                self.objects[key] = ExtraArgs['Metadata']

        client = Client()
        first = exporter._upload_additive(client, local, 'bucket', 'condition/pdo_positive_crops')
        second = exporter._upload_additive(client, local, 'bucket', 'condition/pdo_positive_crops')
        crop.write_bytes(b'changed')
        third = exporter._upload_additive(client, local, 'bucket', 'condition/pdo_positive_crops')
        self.assertEqual(first['uploaded_files'], 1)
        self.assertEqual(second['skipped_matching_files'], 1)
        self.assertEqual(third['conflicting_files'], 1)
        self.assertEqual(client.upload_calls, 1)


if __name__ == '__main__':
    unittest.main()

