from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

import analysis_core
import aws_export_pdo_positive_crops_with_psc as cropper
import aws_quantify_psc_like_objects as segmenter


CONDITION = next(iter(segmenter.CONDITIONS))
PIXEL_SIZE = segmenter.EXPECTED_PIXEL_SIZE_UM


def _csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)


class PSCValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / 'results'; self.cache = self.root / 'cache'
        folder = self.results / CONDITION; folder.mkdir(parents=True)
        zarr_path = self.cache / f'{CONDITION}.ome.zarr'
        group = zarr.open_group(str(zarr_path), mode='w')
        height, width = 420, 600
        data = np.zeros((1, 3, height, width), dtype=np.uint16)
        yy, xx = np.indices((height, width))
        data[0, 1] = 9 + ((xx + yy) % 3)
        data[0, 0] = 15; data[0, 2] = 100
        wells, psc_rows, pdo_rows = [], [], []
        for index in range(40):
            row, col = divmod(index, 8)
            x, y = 45 + col * 70, 55 + row * 75
            positive = bool(index % 2)
            signal = 45 + index
            object_radius = 22 if index == 39 else 3
            data[0, 1][(xx - x) ** 2 + (yy - y) ** 2 <= object_radius ** 2] = signal
            wells.append([index + 1, x, y, 30, int(positive), positive,
                          20 if positive else 0, 20 * PIXEL_SIZE ** 2 if positive else 0])
            psc_rows.append([index + 1, x, y, 30, y * PIXEL_SIZE / 1000,
                             'valid_local_background', .9, 10, 11,
                             signal - 10, 0, 0, (index + 1) * PIXEL_SIZE ** 2])
            if positive:
                pdo_rows.append([index + 1, 1, x, y, 20, 20 * PIXEL_SIZE ** 2, 5.0])
        group.create_dataset('0', data=data, chunks=(1, 1, 64, 64))
        group.attrs['omero'] = {'channels': [
            {'label': 'GFP 488', 'window': {'start': 0, 'end': 200}},
            {'label': 'RFP 561', 'window': {'start': 0, 'end': 200}},
            {'label': 'DIC1', 'window': {'start': 0, 'end': 200}},
        ]}
        group.attrs['multiscales'] = [{
            'axes': [{'name': 't'}, {'name': 'c'}, {'name': 'y', 'unit': 'micrometer'},
                     {'name': 'x', 'unit': 'micrometer'}],
            'datasets': [{'path': '0', 'coordinateTransformations': [
                {'type': 'scale', 'scale': [1, 1, PIXEL_SIZE, PIXEL_SIZE]}]}],
        }]
        (folder / 'condition_summary.json').write_text(json.dumps({
            'completion_status': 'completed', 'condition_id': CONDITION,
            'condition_name': f'{CONDITION}_source',
            'pixel_size_um': {'x': PIXEL_SIZE, 'y': PIXEL_SIZE},
            'channel_mapping': {'gfp_channel': 0, 'dic_channel': 2},
            'benchmark': {'source': str(zarr_path)},
        }), encoding='utf-8')
        _csv(folder / 'well_measurements.csv', [
            'well_id', 'x_px_fullres', 'y_px_fullres', 'radius_px', 'PDO_count',
            'PDO_present', 'total_PDO_projected_area_px2', 'total_PDO_projected_area_um2'], wells)
        _csv(folder / 'pdo_measurements.csv', [
            'well_id', 'pdo_number_in_well', 'centroid_x_px_fullres',
            'centroid_y_px_fullres', 'projected_area_px2', 'projected_area_um2',
            'equivalent_circular_diameter_um'], pdo_rows)
        psc_folder = folder / 'psc_quantification'; psc_folder.mkdir()
        (psc_folder / 'psc_summary.json').write_text(json.dumps({
            'completion_status': 'completed', 'well_set_qc_passed': True,
        }), encoding='utf-8')
        _csv(psc_folder / 'psc_well_measurements.csv', [
            'well_id', 'x_px_fullres', 'y_px_fullres', 'radius_px', 'y_mm', 'background_qc',
            'background_valid_fraction', 'RFP_background_median', 'RFP_background_p99',
            'RFP_background_corrected_mean', 'RFP_saturated_pixel_count',
            'RFP_saturated_pixel_fraction', 'exploratory_RFP_positive_area_um2'], psc_rows)

    def tearDown(self):
        self.temp.cleanup()

    def test_validation_only_segmentation_then_crop_export(self):
        segment_args = segmenter.build_parser().parse_args([
            '--validation-only', '--result-root', str(self.results), '--cache-root', str(self.cache),
            '--condition-id', CONDITION, '--qc-per-category', '1'])
        with patch.object(analysis_core, 'detect_psc', side_effect=AssertionError('detect_psc called')), \
             patch.object(analysis_core, 'detect_wells', side_effect=AssertionError('Hough called')), \
             patch.object(analysis_core, 'segment_pdos', side_effect=AssertionError('PDO called')):
            self.assertEqual(segmenter.run(segment_args), 0)
        output = self.results / CONDITION / 'psc_object_quantification'
        well_rows = segmenter._read_csv(output / 'psc_well_object_summary.csv')
        primary_rows = [row for row in well_rows if row['sample_type'] == 'primary']
        supplement_rows = [row for row in well_rows if row['sample_type'] == 'qc_supplement']
        self.assertEqual(len(primary_rows), 30)
        self.assertTrue(supplement_rows)
        self.assertTrue({row['well_id'] for row in primary_rows}.isdisjoint(
            {row['well_id'] for row in supplement_rows}))
        object_rows = segmenter._read_csv(output / 'psc_object_measurements.csv')
        unresolved = [row for row in object_rows if row['object_status'] == 'unresolved_cluster']
        self.assertTrue(unresolved)
        unresolved_well = next(row for row in well_rows
                               if row['well_id'] == unresolved[0]['well_id'])
        self.assertEqual(float(unresolved_well['PSC_like_resolved_object_count']), 0)
        self.assertEqual(float(unresolved_well['unresolved_PSC_like_cluster_count']), 1)
        self.assertEqual(unresolved_well['PSC_segmentation_status'],
                         'unresolved_cluster_present')
        summary = segmenter._read_json(output / 'segmentation_summary.json')
        self.assertTrue(summary['validation_only'])
        self.assertFalse(summary['full_well_processing_available'])
        self.assertEqual(summary['primary_validation_wells'], 30)

        crop_args = cropper.build_parser().parse_args([
            '--validation-only', '--result-root', str(self.results), '--cache-root', str(self.cache),
            '--condition-id', CONDITION, '--panel-size', '64', '--contact-sheet-size', '10'])
        self.assertEqual(cropper.run(crop_args), 0)
        manifest = cropper._read_csv(output / 'validation_crops' / 'manifest.csv')
        self.assertEqual(len(manifest), len(well_rows))
        self.assertTrue(all(Path(row['labelled_validation_crop']).is_file() for row in manifest))
        crop_summary = cropper._read_json(output / 'validation_crops' /
                                          'validation_crop_summary.json')
        self.assertTrue(crop_summary['crop_well_set_qc_passed'])
        self.assertFalse(crop_summary['full_PDO_positive_crop_regeneration_available'])
        self.assertTrue((self.results / 'all_conditions_psc_validation_crop_manifest.csv').is_file())


if __name__ == '__main__':
    unittest.main()
