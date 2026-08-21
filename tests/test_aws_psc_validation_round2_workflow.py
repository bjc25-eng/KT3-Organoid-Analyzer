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
import aws_export_psc_validation_round2 as cropper
import aws_psc_validation_round2 as round2


CONDITION = round2.DMSO_CONDITION
PIXEL_SIZE = round2.EXPECTED_PIXEL_SIZE_UM


def _csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)


class Round2WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.results = self.root / 'results'; self.cache = self.root / 'cache'
        self.folder = self.results / CONDITION; self.folder.mkdir(parents=True)
        width, height = 520, 420
        zarr_path = self.cache / f'{CONDITION}.ome.zarr'
        group = zarr.open_group(str(zarr_path), mode='w')
        data = np.zeros((1, 3, height, width), dtype=np.uint16)
        yy, xx = np.indices((height, width))
        data[0, 0] = 5; data[0, 1] = 10; data[0, 2] = 100
        well_ids = ['11163', '15470', '6350', '19515', '5', '6', '7', '8', '9', '10']
        final_rows, pdo_rows, sample_rows, round1_wells, object_rows = [], [], [], [], []
        for index, well_id in enumerate(well_ids):
            row, col = divmod(index, 5); x, y = 55 + col * 95, 90 + row * 150
            positive = index % 2 == 0
            final_rows.append([well_id, x, y, 35, int(positive), positive,
                               30 if positive else 0, 30 * PIXEL_SIZE ** 2 if positive else 0])
            if positive:
                pdo_rows.append([well_id, 1, x, y, 30, 30 * PIXEL_SIZE ** 2, 8.0])
            sample_rows.append([CONDITION, well_id, 'primary', 'round1_core', index + 1])
            labels = np.zeros((105, 105), dtype=np.int32)
            # Well 8 is a zero-object example. Well 11163 is near the wall; well 15470
            # overlaps a reconstructed PDO; all IDs and masks remain canonical Round-1 assets.
            if well_id != '8':
                cx = 80 if well_id == '11163' else 52
                cy = 52
                labels[(np.indices(labels.shape)[1] - cx) ** 2
                       + (np.indices(labels.shape)[0] - cy) ** 2 <= 4 ** 2] = 1
                object_rows.append([CONDITION, well_id,
                                    f'{CONDITION}__W{well_id}__PSCLIKE001', 1, 1, 'resolved'])
                gx, gy = x + cx - 52, y + cy - 52
                data[0, 1][(xx - gx) ** 2 + (yy - gy) ** 2 <= 4 ** 2] = 80 + index
                data[0, 0][(xx - gx) ** 2 + (yy - gy) ** 2 <= 4 ** 2] = 40
            mask_path = self.folder / 'psc_object_quantification' / 'segmentation_masks' / f'well_{well_id}.npz'
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, labels=labels, left=x - 52, top=y - 52)
            round1_wells.append([
                CONDITION, well_id, x, y, 35, positive, int(positive),
                30 * PIXEL_SIZE ** 2 if positive else 0,
                0 if well_id == '8' else 1, 0, float(index - 3),
                'valid_local_background', 4.0, 14.0, str(mask_path),
            ])
        group.create_dataset('0', data=data, chunks=(1, 1, 64, 64))
        group.attrs['omero'] = {'channels': [
            {'label': 'GFP 488', 'window': {'start': 0, 'end': 120}},
            {'label': 'RFP 561', 'window': {'start': 0, 'end': 120}},
            {'label': 'DIC1', 'window': {'start': 0, 'end': 150}},
        ]}
        group.attrs['multiscales'] = [{
            'axes': [{'name': 't'}, {'name': 'c'}, {'name': 'y', 'unit': 'micrometer'},
                     {'name': 'x', 'unit': 'micrometer'}],
            'datasets': [{'path': '0', 'coordinateTransformations': [
                {'type': 'scale', 'scale': [1, 1, PIXEL_SIZE, PIXEL_SIZE]}]}],
        }]
        (self.folder / 'condition_summary.json').write_text(json.dumps({
            'completion_status': 'completed', 'condition_id': CONDITION,
            'condition_name': f'{CONDITION}_source',
            'pixel_size_um': {'x': PIXEL_SIZE, 'y': PIXEL_SIZE},
            'channel_mapping': {'gfp_channel': 0, 'dic_channel': 2},
            'benchmark': {'source': str(zarr_path)},
        }), encoding='utf-8')
        _csv(self.folder / 'well_measurements.csv', [
            'well_id', 'x_px_fullres', 'y_px_fullres', 'radius_px', 'PDO_count',
            'PDO_present', 'total_PDO_projected_area_px2', 'total_PDO_projected_area_um2'],
            final_rows)
        _csv(self.folder / 'pdo_measurements.csv', [
            'well_id', 'pdo_number_in_well', 'centroid_x_px_fullres',
            'centroid_y_px_fullres', 'projected_area_px2', 'projected_area_um2',
            'equivalent_circular_diameter_um'], pdo_rows)
        round1 = self.folder / 'psc_object_quantification'
        _csv(round1 / 'validation_sample_manifest.csv', [
            'condition_id', 'well_id', 'sample_type', 'sample_reasons', 'selection_rank'], sample_rows)
        _csv(round1 / 'psc_well_object_summary.csv', [
            'condition_id', 'well_id', 'x_px_fullres', 'y_px_fullres', 'radius_px',
            'PDO_present', 'PDO_count', 'total_PDO_projected_area_um2',
            'PSC_like_resolved_object_count', 'unresolved_PSC_like_cluster_count',
            'RFP_background_corrected_mean', 'background_qc', 'threshold_corrected_RFP',
            'threshold_detector_RFP', 'mask_path'], round1_wells)
        _csv(round1 / 'psc_object_measurements.csv', [
            'condition_id', 'well_id', 'object_id', 'object_number_in_well',
            'mask_label', 'object_status'], object_rows)
        (round1 / 'segmentation_summary.json').write_text(json.dumps({
            'completion_status': 'validation_sample_completed', 'validation_only': True,
            'full_well_processing_available': False,
        }), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_round2_diagnostics_and_crops_are_bounded_and_non_destructive(self):
        args = round2.build_parser().parse_args([
            '--validation-round2-only', '--result-root', str(self.results),
            '--cache-root', str(self.cache), '--condition-id', CONDITION])
        with patch.object(analysis_core, 'detect_psc', side_effect=AssertionError('detect_psc called')), \
             patch.object(analysis_core, 'detect_wells', side_effect=AssertionError('Hough called')), \
             patch.object(analysis_core, 'segment_pdos', side_effect=AssertionError('PDO called')):
            self.assertEqual(round2.run(args), 0)
        output = self.folder / 'psc_object_quantification' / 'validation_round2'
        manifest = round2._read_csv(output / 'diagnostic_manifest.csv')
        self.assertEqual(len(manifest), 10)
        self.assertLessEqual(len(manifest), 12)
        selected = {row['well_id'] for row in manifest}
        self.assertTrue({'11163', '15470'}.issubset(selected))
        objects = round2._read_csv(output / 'object_qc_measurements.csv')
        self.assertTrue(all(row['canonical_object_id'] for row in objects))
        wall = next(row for row in objects if row['well_id'] == '11163')
        self.assertIn(wall['round2_candidate_status'],
                      {'wall_proximity_candidate', 'PDO_overlap_and_wall_candidate'})
        summary = round2._read_json(output / 'round2_summary.json')
        self.assertFalse(summary['full_well_processing_available'])
        self.assertFalse(summary['final_crop_regeneration_available'])
        self.assertEqual(summary['fixed_parameters']['threshold_k'], 3.0)

        crop_args = cropper.build_parser().parse_args([
            '--validation-round2-only', '--result-root', str(self.results),
            '--cache-root', str(self.cache), '--condition-id', CONDITION,
            '--panel-size', '64', '--contact-sheet-size', '5'])
        self.assertEqual(cropper.run(crop_args), 0)
        manifest = cropper._read_csv(output / 'diagnostic_manifest.csv')
        self.assertTrue(all(Path(row['labelled_crop']).is_file() for row in manifest))
        self.assertTrue(all(Path(row['radial_comparison']).is_file() for row in manifest))
        final_summary = cropper._read_json(output / 'round2_summary.json')
        self.assertTrue(final_summary['crop_well_set_qc_passed'])
        self.assertFalse(final_summary['full_well_processing_available'])
        self.assertFalse(final_summary['final_crop_regeneration_available'])
        self.assertTrue((self.folder / 'psc_object_quantification' /
                         'validation_sample_manifest.csv').is_file())


if __name__ == '__main__':
    unittest.main()
