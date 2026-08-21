from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

import analysis_core
import aws_quantify_psc_like_objects as segmenter


PIXEL_SIZE = segmenter.EXPECTED_PIXEL_SIZE_UM
CONDITION = next(iter(segmenter.CONDITIONS))


def _well(well_id: int, x=65, y=65, radius=60, positive=False) -> dict:
    return {
        'well_id': str(well_id), 'x_px_fullres': x, 'y_px_fullres': y,
        'radius_px': radius, 'PDO_present': positive, 'PDO_count': 1 if positive else 0,
        'total_PDO_projected_area_um2': 12.5 if positive else 0,
    }


def _psc(well_id: int, signal: float, y: float, *, background='valid_local_background',
         saturated=0, area=5) -> dict:
    return {
        'well_id': str(well_id), 'y_px_fullres': y,
        'RFP_background_corrected_mean': signal if background == 'valid_local_background' else 'nan',
        'RFP_background_median': 10, 'RFP_background_p99': 11,
        'background_qc': background, 'background_valid_fraction': .8 if background.startswith('valid') else .01,
        'RFP_saturated_pixel_count': saturated,
        'RFP_saturated_pixel_fraction': .01 if saturated else 0,
        'exploratory_RFP_positive_area_um2': area if background.startswith('valid') else 'nan',
    }


class ValidationSamplingTests(unittest.TestCase):
    def _tables(self, count=90):
        wells, psc = [], []
        for index in range(count):
            wells.append(_well(index + 1, x=100 + index, y=100 + index * 10,
                               radius=20, positive=bool(index % 2)))
            psc.append(_psc(index + 1, signal=float(index), y=100 + index * 10, area=index + 1))
        return wells, psc

    def test_primary_sample_is_exactly_30_and_deterministic(self):
        wells, psc = self._tables()
        first, qc1 = segmenter.select_validation_sample(CONDITION, wells, psc, qc_per_category=0)
        second, qc2 = segmenter.select_validation_sample(CONDITION, wells, psc, qc_per_category=0)
        self.assertEqual(first, second)
        self.assertEqual(qc1, qc2)
        self.assertEqual(len(first), 30)
        self.assertEqual({row['sample_type'] for row in first}, {'primary'})
        self.assertEqual(len({row['well_id'] for row in first}), 30)

    def test_qc_supplements_are_additional_and_labelled(self):
        wells, psc = self._tables(100)
        psc[95].update({'RFP_saturated_pixel_count': 2, 'RFP_saturated_pixel_fraction': .2})
        psc[96].update({'background_qc': 'insufficient_local_background',
                        'RFP_background_corrected_mean': 'nan', 'background_valid_fraction': .01,
                        'exploratory_RFP_positive_area_um2': 'nan'})
        rows, qc = segmenter.select_validation_sample(CONDITION, wells, psc, qc_per_category=1)
        primary = {row['well_id'] for row in rows if row['sample_type'] == 'primary'}
        supplements = [row for row in rows if row['sample_type'] == 'qc_supplement']
        self.assertEqual(len(primary), 30)
        self.assertTrue(supplements)
        self.assertTrue(primary.isdisjoint({row['well_id'] for row in supplements}))
        reasons = ';'.join(row['sample_reasons'] for row in supplements)
        self.assertIn('saturated_RFP', reasons)
        self.assertIn('insufficient_background', reasons)
        self.assertIn('highest_RFP_positive_area', reasons)

    def test_exact_final_well_set_is_required(self):
        wells, psc = self._tables(35)
        with self.assertRaisesRegex(RuntimeError, 'exact final-well-set match'):
            segmenter.select_validation_sample(CONDITION, wells, psc[:-1])

    def test_less_than_30_valid_background_wells_fails(self):
        wells, psc = self._tables(30)
        psc[0]['background_qc'] = 'insufficient_local_background'
        psc[0]['RFP_background_corrected_mean'] = 'nan'
        with self.assertRaisesRegex(RuntimeError, '30 are required'):
            segmenter.select_validation_sample(CONDITION, wells, psc)


class SegmentationTests(unittest.TestCase):
    def _segment(self, *, background='valid_local_background'):
        well = _well(1)
        psc = _psc(1, 3, 65, background=background)
        tile = np.fromfunction(lambda yy, xx: 9 + ((xx + yy) % 3), (131, 131), dtype=int).astype(np.uint16)
        yy, xx = np.indices(tile.shape)
        tile[(xx - 30) ** 2 + (yy - 65) ** 2 <= 3 ** 2] = 100
        tile[(xx - 82) ** 2 + (yy - 65) ** 2 <= 22 ** 2] = 120
        tree = cKDTree(np.asarray([[65.0, 65.0]]))
        result = segmenter.segment_validation_well(
            CONDITION, well, psc, [well], tree, tile, 0, 0, PIXEL_SIZE, PIXEL_SIZE,
            tile.dtype, max_radius=60)
        return (*result, tile)

    def test_resolved_and_unresolved_are_separate(self):
        objects, summary, labels, _ = self._segment()
        self.assertEqual(summary['PSC_like_resolved_object_count'], 1)
        self.assertEqual(summary['unresolved_PSC_like_cluster_count'], 1)
        self.assertEqual(summary['PSC_segmentation_status'], 'unresolved_cluster_present')
        self.assertEqual({row['object_status'] for row in objects},
                         {'resolved', 'unresolved_cluster'})
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertNotEqual(summary['PSC_like_resolved_object_count'] +
                            summary['unresolved_PSC_like_cluster_count'],
                            summary['PSC_like_resolved_object_count'])

    def test_threshold_and_measurements_use_original_detector_values(self):
        objects, summary, labels, tile = self._segment()
        self.assertAlmostEqual(summary['background_robust_sigma_RFP'], 1.4826, places=3)
        self.assertAlmostEqual(summary['threshold_corrected_RFP'], 3 * 1.4826, places=3)
        resolved = next(row for row in objects if row['object_status'] == 'resolved')
        raw_object_pixels = tile[labels == int(resolved['mask_label'])].astype(float)
        self.assertEqual(resolved['mean_RFP_intensity'], float(np.mean(raw_object_pixels)))
        self.assertEqual(resolved['median_RFP_intensity'], float(np.median(raw_object_pixels)))
        self.assertEqual(resolved['max_RFP_intensity'], 100)
        self.assertEqual(resolved['integrated_RFP_intensity'],
                         float(np.sum(raw_object_pixels)))
        self.assertEqual(resolved['background_corrected_mean_RFP'],
                         float(np.mean(raw_object_pixels)) - 10)

    def test_insufficient_background_is_nan_not_zero(self):
        objects, summary, labels, _ = self._segment(background='insufficient_local_background')
        self.assertEqual(objects, [])
        self.assertTrue(math.isnan(summary['PSC_like_resolved_object_count']))
        self.assertTrue(math.isnan(summary['unresolved_PSC_like_cluster_count']))
        self.assertEqual(summary['PSC_segmentation_status'], 'insufficient_local_background')
        self.assertFalse(np.any(labels))

    def test_large_component_is_not_silently_counted_as_resolved(self):
        objects, summary, _, _ = self._segment()
        cluster = next(row for row in objects if row['object_status'] == 'unresolved_cluster')
        self.assertGreater(cluster['equivalent_diameter_um'], 30)
        self.assertNotIn(cluster['area_px2'], [row['area_px2'] for row in objects
                                               if row['object_status'] == 'resolved'])
        self.assertLess(summary['PSC_like_total_area_um2'], cluster['area_um2'])

    def test_no_old_detection_or_pdo_segmentation_is_called(self):
        with patch.object(analysis_core, 'detect_psc', side_effect=AssertionError('old PSC called')), \
             patch.object(analysis_core, 'detect_wells', side_effect=AssertionError('Hough called')), \
             patch.object(analysis_core, 'segment_pdos', side_effect=AssertionError('PDO called')):
            objects, summary, _, _ = self._segment()
        self.assertTrue(objects)
        self.assertEqual(summary['PSC_segmentation_status'], 'unresolved_cluster_present')

    def test_cli_has_no_full_well_mode_and_requires_validation_gate(self):
        options = segmenter.build_parser().format_help()
        self.assertNotIn('--all-wells', options)
        args = segmenter.build_parser().parse_args([
            '--validation-only', '--result-root', str(Path.cwd())])
        args.validation_only = False
        with self.assertRaisesRegex(RuntimeError, 'full-well processing is blocked'):
            segmenter.run(args)


if __name__ == '__main__':
    unittest.main()
