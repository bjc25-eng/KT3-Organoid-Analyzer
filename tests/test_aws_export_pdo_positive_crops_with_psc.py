from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import aws_export_pdo_positive_crops_with_psc as exporter


CONDITION = next(iter(exporter.CONDITIONS))
PIXEL_SIZE = exporter.EXPECTED_PIXEL_SIZE_UM


def _well(positive=True):
    return {
        'well_id': '17', 'x_px_fullres': 20, 'y_px_fullres': 20, 'radius_px': 15,
        'PDO_present': positive, 'PDO_count': 1 if positive else 0,
        'total_PDO_projected_area_um2': 22.25 if positive else 0,
    }


def _summary(mask_path='mask.npz'):
    return {
        'well_id': '17', 'PSC_like_resolved_object_count': 4,
        'unresolved_PSC_like_cluster_count': 1,
        'RFP_background_corrected_mean': -2.5,
        'PSC_segmentation_status': 'unresolved_cluster_present',
        'background_qc': 'valid_local_background', 'threshold_corrected_RFP': 4.45,
        'threshold_detector_RFP': 14.45, 'mask_path': mask_path,
    }


def _objects():
    return [
        {'object_id': 'resolved', 'object_status': 'resolved', 'mask_label': 1,
         'centroid_x_px_fullres': 12, 'centroid_y_px_fullres': 20,
         'object_number_in_well': 1},
        {'object_id': 'cluster', 'object_status': 'unresolved_cluster', 'mask_label': 2,
         'centroid_x_px_fullres': 28, 'centroid_y_px_fullres': 20,
         'object_number_in_well': 2},
    ]


class ValidationCropTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_header_separates_resolved_and_unresolved(self):
        pdos = [{'equivalent_circular_diameter_um': 6.2}]
        lines = exporter.validation_header_lines(CONDITION, _well(), pdos, _summary())
        self.assertEqual(lines[0], 'Lane 1 | RMC6236 0 nM | Final well 17 | PDO POSITIVE')
        self.assertIn('PDO count: 1 | PDO size(s): 6.2 µm | Total PDO area: 22.2 µm²', lines)
        self.assertIn('PSC-like resolved objects: 4 | Unresolved clusters: 1', lines)
        self.assertNotIn('PSC-like object count: 5', '\n'.join(lines))
        self.assertIn('Background-corrected RFP signal: -2.50 detector units', lines)
        self.assertIn('PSC segmentation: unresolved_cluster_present', lines)
        self.assertIn('Background QC: valid_local_background | VALIDATION ONLY', lines[-1])

    def test_negative_well_header_is_supported(self):
        summary = _summary(); summary.update({
            'PSC_like_resolved_object_count': float('nan'),
            'unresolved_PSC_like_cluster_count': float('nan'),
            'PSC_segmentation_status': 'insufficient_local_background',
            'background_qc': 'insufficient_local_background',
            'RFP_background_corrected_mean': float('nan'),
        })
        lines = exporter.validation_header_lines(CONDITION, _well(False), [], summary)
        self.assertIn('PDO NEGATIVE', lines[0])
        self.assertIn('PSC-like resolved objects: NaN | Unresolved clusters: NaN', lines)

    def test_mask_is_positioned_in_larger_display_crop(self):
        path = self.root / 'mask.npz'
        labels = np.zeros((11, 11), dtype=np.int32); labels[4:7, 4:7] = 3
        np.savez_compressed(path, labels=labels, left=15, top=15)
        result = exporter._load_label_mask({'mask_path': str(path)}, (31, 31), 5, 5)
        self.assertEqual(int(result[14, 14]), 3)
        self.assertEqual(np.count_nonzero(result == 3), 9)

    def test_overlay_uses_distinct_resolved_and_cluster_colours(self):
        image = Image.new('RGB', (41, 41), 'black')
        labels = np.zeros((41, 41), dtype=np.int32)
        yy, xx = np.indices(labels.shape)
        labels[(xx - 12) ** 2 + (yy - 20) ** 2 <= 3 ** 2] = 1
        labels[(xx - 28) ** 2 + (yy - 20) ** 2 <= 4 ** 2] = 2
        result = exporter._overlay(image, panel_size=164, well=_well(), pdos=[],
                                   objects=_objects(), labels=labels, left=0, top=0,
                                   pixel_size_um=PIXEL_SIZE)
        colors = set(map(tuple, np.asarray(result).reshape(-1, 3)))
        self.assertIn((255, 255, 255), colors)
        self.assertIn((255, 128, 0), colors)
        result.close(); image.close()

    def test_four_panel_validation_crop_renders(self):
        raw = {key: Image.new('RGB', (41, 41), 'black')
               for key in ('dic', 'gfp', 'rfp', 'composite')}
        labels = np.zeros((41, 41), dtype=np.int32)
        labels[18:23, 10:15] = 1; labels[16:25, 25:34] = 2
        validation = {'pixel_size_um': {'x': PIXEL_SIZE, 'y': PIXEL_SIZE}}
        crop = exporter.labelled_validation_crop(
            raw, condition_id=CONDITION, well=_well(),
            pdos=[{'centroid_x_px_fullres': 20, 'centroid_y_px_fullres': 20,
                   'equivalent_circular_diameter_um': 6.2}],
            objects=_objects(), well_summary=_summary(), labels=labels,
            validation=validation, left=0, top=0, panel_size=128)
        self.assertGreater(crop.width, 256)
        self.assertGreater(crop.height, 256)
        crop.close()
        for image in raw.values(): image.close()

    def test_validation_input_rejects_non_gated_summary(self):
        folder = self.root / 'condition'; output = folder / 'psc_object_quantification'
        output.mkdir(parents=True)
        for name in ('validation_sample_manifest.csv', 'psc_well_object_summary.csv',
                     'psc_object_measurements.csv'):
            (output / name).write_text('well_id\n17\n', encoding='utf-8')
        (output / 'segmentation_summary.json').write_text(json.dumps({
            'completion_status': 'validation_sample_completed', 'validation_only': False,
            'full_well_processing_available': True,
        }), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'validation-only safety gate'):
            exporter._validation_inputs(folder)

    def test_cli_has_no_final_regeneration_mode(self):
        help_text = exporter.build_parser().format_help()
        self.assertNotIn('--regenerate-all', help_text)
        self.assertNotIn('--all-pdo-positive', help_text)
        args = exporter.build_parser().parse_args([
            '--validation-only', '--result-root', str(self.root)])
        args.validation_only = False
        with self.assertRaisesRegex(RuntimeError, 'final crop regeneration is blocked'):
            exporter.run(args)


if __name__ == '__main__':
    unittest.main()
