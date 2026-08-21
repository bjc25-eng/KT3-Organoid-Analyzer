from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import aws_psc_validation_round2 as round2


CONDITION = next(iter(round2.CONDITIONS))
PIXEL_SIZE = round2.EXPECTED_PIXEL_SIZE_UM


def _sample(well_id: str) -> dict:
    return {'well_id': well_id, 'sample_type': 'primary', 'sample_reasons': 'core'}


def _round1_well(index: int, well_id: str | None = None) -> dict:
    return {
        'well_id': well_id or str(index), 'x_px_fullres': 50 + index,
        'y_px_fullres': 100 + index * 10, 'radius_px': 30,
        'PDO_present': bool(index % 2), 'PDO_count': int(bool(index % 2)),
        'total_PDO_projected_area_um2': 10 if index % 2 else 0,
        'PSC_like_resolved_object_count': index % 4,
        'unresolved_PSC_like_cluster_count': 0,
        'RFP_background_corrected_mean': float(index),
    }


class Round2SelectionTests(unittest.TestCase):
    def test_forced_wells_and_diagnostic_categories_are_bounded(self):
        wells = [_round1_well(i + 1) for i in range(20)]
        wells[3]['well_id'] = '11163'; wells[7]['well_id'] = '15470'
        sample = [_sample(row['well_id']) for row in wells]
        selected = round2.select_diagnostic_wells(
            CONDITION, sample, wells, {'11163', '15470'})
        ids = {row['well_id'] for row in selected}
        self.assertEqual(len(selected), 10)
        self.assertLessEqual(len(selected), 12)
        self.assertTrue({'11163', '15470'}.issubset(ids))
        forced = {row['well_id'] for row in selected if row['forced'] in (True, 'True')}
        self.assertEqual(forced, {'11163', '15470'})
        reasons = ';'.join(row['selection_reasons'] for row in selected)
        self.assertIn('highest_round1_resolved_object_count', reasons)
        self.assertIn('zero_object_', reasons)
        self.assertIn('high_RFP_', reasons)
        self.assertIn('low_RFP_', reasons)

    def test_selection_rejects_more_than_hard_maximum(self):
        wells = [_round1_well(i + 1) for i in range(20)]
        sample = [_sample(row['well_id']) for row in wells]
        with self.assertRaisesRegex(ValueError, 'cannot exceed 12'):
            round2.select_diagnostic_wells(CONDITION, sample, wells, set(), target=13, maximum=13)

    def test_unique_location_is_used_but_ambiguous_id_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            conditions = list(round2.CONDITIONS)[:2]
            for condition in conditions:
                folder = root / condition / 'psc_object_quantification'
                folder.mkdir(parents=True)
                ids = ['19515'] if condition == conditions[0] else []
                ids += ['6350']
                with (folder / 'validation_sample_manifest.csv').open(
                        'w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['well_id'])
                    writer.writeheader(); writer.writerows({'well_id': value} for value in ids)
            forced, qc = round2.locate_forced_wells(root)
            self.assertIn('19515', forced[conditions[0]])
            self.assertNotIn('6350', forced.get(conditions[0], set()))
            self.assertEqual(qc['ambiguous_round1_locations']['6350'], conditions)


class RadialComparisonTests(unittest.TestCase):
    def test_retained_truncated_removed_and_split(self):
        mask = np.ones((3, 9), dtype=bool)
        retained = round2.radial_comparison(mask, np.zeros(mask.shape), .75, 1)
        self.assertEqual(retained['status'], 'retained')
        removed = round2.radial_comparison(mask, np.ones(mask.shape), .75, 1)
        self.assertEqual(removed['status'], 'removed')
        radial = np.zeros(mask.shape); radial[:, -2:] = 1
        truncated = round2.radial_comparison(mask, radial, .75, 1)
        self.assertEqual(truncated['status'], 'truncated')
        split_mask = np.zeros((3, 9), dtype=bool)
        split_mask[:, :3] = True; split_mask[:, 6:] = True; split_mask[1, 3:6] = True
        split_radial = np.zeros(split_mask.shape); split_radial[1, 3:6] = 1
        split = round2.radial_comparison(split_mask, split_radial, .75, 1)
        self.assertEqual(split['status'], 'split')
        self.assertEqual(split['component_count'], 2)

    def test_smaller_radius_never_creates_new_canonical_identity(self):
        mask = np.ones((5, 5), dtype=bool)
        comparison = round2.radial_comparison(mask, np.zeros(mask.shape), .8, 1)
        self.assertNotIn('object_id', comparison)
        self.assertEqual(comparison['component_count'], 1)


class CandidateQCTests(unittest.TestCase):
    def test_exclusive_candidate_categories_and_metrics(self):
        labels = np.zeros((120, 120), dtype=np.int32)
        labels[57:62, 57:62] = 1       # normal
        labels[57:62, 42:47] = 2       # PDO overlap
        labels[57:62, 98:103] = 3      # wall zone
        labels[97:102, 57:62] = 4      # PDO + wall
        labels[2:42, 2:42] = 5         # unresolved (precedence over wall)
        canonical = [
            {'object_id': f'stable_{index}', 'object_number_in_well': index,
             'mask_label': index, 'object_status': 'unresolved_cluster' if index == 5 else 'resolved'}
            for index in range(1, 6)
        ]
        rfp = np.full(labels.shape, 30, dtype=np.uint16)
        gfp = np.full(labels.shape, 5, dtype=np.uint16)
        well = {'well_id': '9', 'x_px_fullres': 60, 'y_px_fullres': 60, 'radius_px': 50,
                'PDO_present': True, 'PDO_count': 2, 'total_PDO_projected_area_um2': 50}
        pdos = [
            {'centroid_x_px_fullres': 44, 'centroid_y_px_fullres': 59,
             'equivalent_circular_diameter_um': 8},
            {'centroid_x_px_fullres': 59, 'centroid_y_px_fullres': 99,
             'equivalent_circular_diameter_um': 8},
        ]
        rows, summary = round2.candidate_qc_rows(
            CONDITION, well, canonical, labels, 0, 0, rfp, gfp, pdos,
            PIXEL_SIZE, PIXEL_SIZE, 4, 14)
        by_id = {row['canonical_object_id']: row for row in rows}
        self.assertEqual(by_id['stable_1']['round2_candidate_status'], 'normal_candidate')
        self.assertEqual(by_id['stable_2']['round2_candidate_status'], 'PDO_overlap_candidate')
        self.assertEqual(by_id['stable_3']['round2_candidate_status'], 'wall_proximity_candidate')
        self.assertEqual(by_id['stable_4']['round2_candidate_status'],
                         'PDO_overlap_and_wall_candidate')
        self.assertEqual(by_id['stable_5']['round2_candidate_status'], 'unresolved_cluster')
        self.assertGreater(by_id['stable_2']['PDO_overlap_fraction'], 0)
        self.assertEqual(by_id['stable_1']['GFP_mean_intensity'], 5)
        self.assertEqual(by_id['stable_1']['RFP_mean_intensity'], 30)
        self.assertEqual(by_id['stable_1']['RFP_integrated_intensity'], 30 * 25)
        self.assertEqual(by_id['stable_1']['background_corrected_RFP_mean'], 20)
        self.assertEqual(by_id['stable_1']['raw_mean_RFP_to_mean_GFP_ratio'], 6)
        self.assertEqual(by_id['stable_1']['corrected_RFP_to_mean_GFP_ratio'], 4)
        self.assertEqual(summary['PSC_like_unflagged_resolved_count'], 1)
        self.assertEqual(summary['PSC_like_PDO_overlap_candidate_count'], 1)
        self.assertEqual(summary['PSC_like_wall_proximity_candidate_count'], 1)
        self.assertEqual(summary['PSC_like_PDO_overlap_and_wall_candidate_count'], 1)
        self.assertEqual(summary['unresolved_PSC_like_cluster_count'], 1)
        self.assertNotIn('PSC_like_object_count', summary)
        self.assertIn('not the original PDO segmentation mask',
                      by_id['stable_2']['PDO_mask_provenance'])

    def test_fixed_parameters_and_no_full_route(self):
        self.assertEqual(round2.THRESHOLD_K, 3.0)
        self.assertEqual(round2.DETECTION_GAUSSIAN_SIGMA_PX, .75)
        self.assertEqual(round2.MIN_EQUIVALENT_DIAMETER_UM, 3.0)
        self.assertEqual(round2.UNRESOLVED_EQUIVALENT_DIAMETER_UM, 30.0)
        self.assertEqual(round2.INTERIOR_RADIUS_FRACTION, .86)
        help_text = round2.build_parser().format_help()
        self.assertNotIn('--all-wells', help_text)
        args = round2.build_parser().parse_args([
            '--validation-round2-only', '--result-root', str(Path.cwd())])
        args.validation_round2_only = False
        with self.assertRaisesRegex(RuntimeError, 'full processing is blocked'):
            round2.run(args)


if __name__ == '__main__':
    unittest.main()
