from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import aws_export_psc_validation_round2 as exporter


CONDITION = next(iter(exporter.CONDITIONS))
PIXEL_SIZE = exporter.EXPECTED_PIXEL_SIZE_UM


def _well():
    return {'well_id': '17', 'x_px_fullres': 50, 'y_px_fullres': 50, 'radius_px': 40,
            'PDO_present': True, 'PDO_count': 1, 'total_PDO_projected_area_um2': 20}


def _objects():
    statuses = ['normal_candidate', 'wall_proximity_candidate', 'PDO_overlap_candidate',
                'PDO_overlap_and_wall_candidate', 'unresolved_cluster']
    output = []
    for index, status in enumerate(statuses, 1):
        row = {'canonical_object_id': f'condition__W17__PSCLIKE{index:03d}',
               'canonical_mask_label': index, 'round2_candidate_status': status}
        for radius in ('0_75', '0_80', '0_86'):
            row[f'radius_{radius}_component_count'] = 1
        output.append(row)
    return output


def _summary():
    return {
        'condition_id': CONDITION, 'well_id': '17', 'PSC_like_unflagged_resolved_count': 1,
        'PSC_like_PDO_overlap_candidate_count': 1,
        'PSC_like_wall_proximity_candidate_count': 1,
        'PSC_like_PDO_overlap_and_wall_candidate_count': 1,
        'unresolved_PSC_like_cluster_count': 1, 'threshold_corrected_RFP': 4.45,
        'threshold_detector_RFP': 14.45, 'background_qc': 'valid_local_background',
    }


class Round2DisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_local_rfp_stretch_is_display_only_and_does_not_mutate_input(self):
        rfp = np.arange(101 * 101, dtype=np.uint16).reshape(101, 101)
        before = rfp.copy()
        image, settings = exporter.locally_enhanced_rfp(rfp, _well(), 0, 0)
        self.assertTrue(np.array_equal(rfp, before))
        self.assertEqual(settings['lower_percentile'], .5)
        self.assertEqual(settings['upper_percentile'], 99.5)
        self.assertIn('NOT USED FOR SEGMENTATION OR MEASUREMENT', settings['use'])
        image.close()

    def test_detection_display_uses_fixed_sigma_and_marks_exceeded_pixels(self):
        rfp = np.full((41, 41), 10, dtype=np.uint16); rfp[18:23, 18:23] = 100
        image, exceeded, settings = exporter.detection_display(rfp, 10, 5)
        self.assertEqual(settings['gaussian_sigma_px'], .75)
        self.assertEqual(settings['threshold_corrected_RFP'], 5)
        self.assertTrue(exceeded[20, 20])
        self.assertGreater(settings['threshold_exceeded_pixel_count'], 0)
        image.close()

    def test_local_rfp_stretch_excludes_out_of_image_black_padding(self):
        rfp = np.zeros((25, 25), dtype=np.uint16); rfp[:, 10:] = 100
        edge_well = {**_well(), 'x_px_fullres': 2, 'y_px_fullres': 12,
                     'radius_px': 12}
        image, settings = exporter.locally_enhanced_rfp(
            rfp, edge_well, left=-10, top=0, image_width=15, image_height=25)
        self.assertEqual(settings['minimum_detector_value'], 100)
        image.close()

    def test_all_candidate_statuses_have_distinct_approved_colours(self):
        statuses = ['normal_candidate', 'wall_proximity_candidate', 'PDO_overlap_candidate',
                    'PDO_overlap_and_wall_candidate', 'unresolved_cluster']
        self.assertEqual(len({exporter.COLORS[value] for value in statuses}), len(statuses))
        self.assertEqual(exporter.COLORS['well'], (255, 255, 0))

    def test_six_panel_crop_and_three_panel_radial_strip_render(self):
        raw = {key: Image.new('RGB', (101, 101), 'black')
               for key in ('dic', 'gfp', 'rfp', 'composite')}
        local = Image.new('RGB', (101, 101), 'black')
        detection = Image.new('RGB', (101, 101), 'black')
        labels = np.zeros((101, 101), dtype=np.int32)
        centres = [(35, 50), (70, 50), (50, 35), (50, 70), (20, 20)]
        yy, xx = np.indices(labels.shape)
        for index, (x, y) in enumerate(centres, 1):
            labels[(xx - x) ** 2 + (yy - y) ** 2 <= 3 ** 2] = index
        pdos = [{'centroid_x_px_fullres': 50, 'centroid_y_px_fullres': 50,
                 'equivalent_circular_diameter_um': 8}]
        crop, radial = exporter.labelled_round2_crop(
            raw, local, detection, labels, _objects(), _well(), _summary(), pdos,
            0, 0, 96, PIXEL_SIZE)
        self.assertEqual(radial.width, 3 * 96 + 4 * 8)
        self.assertGreater(crop.width, 3 * 96)
        self.assertGreater(crop.height, 3 * 96)
        crop.close(); radial.close(); local.close(); detection.close()
        for image in raw.values(): image.close()

    def test_round2_input_rejects_missing_safety_gate(self):
        folder = self.root / 'condition'; output = folder / 'psc_object_quantification' / 'validation_round2'
        output.mkdir(parents=True)
        for name in ('diagnostic_manifest.csv', 'object_qc_measurements.csv',
                     'well_diagnostic_summary.csv'):
            (output / name).write_text('well_id\n17\n', encoding='utf-8')
        (output / 'round2_summary.json').write_text(json.dumps({
            'completion_status': 'round2_diagnostics_completed',
            'validation_round2_only': False, 'full_well_processing_available': True,
        }), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'mandatory diagnostic-only gate'):
            exporter._round2_inputs(folder)

    def test_cli_cannot_regenerate_final_crops_or_run_full_wells(self):
        help_text = exporter.build_parser().format_help()
        self.assertNotIn('--all-wells', help_text)
        self.assertNotIn('--regenerate', help_text)
        self.assertNotIn('--all-pdo-positive', help_text)
        args = exporter.build_parser().parse_args([
            '--validation-round2-only', '--result-root', str(self.root)])
        args.validation_round2_only = False
        with self.assertRaisesRegex(RuntimeError, 'final/full processing is blocked'):
            exporter.run(args)


if __name__ == '__main__':
    unittest.main()
