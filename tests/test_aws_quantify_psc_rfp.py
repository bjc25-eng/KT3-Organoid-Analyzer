from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import zarr
from scipy.spatial import cKDTree

import analysis_core
import aws_quantify_psc_rfp as psc


PIXEL_SIZE = psc.EXPECTED_PIXEL_SIZE_UM


def _write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)


def _well(well_id=1, x=60, y=60, radius=20):
    return {'well_id': str(well_id), 'x_px_fullres': x,
            'y_px_fullres': y, 'radius_px': radius}


class PSCQuantificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / 'results'
        self.cache = self.root / 'cache'

    def tearDown(self):
        self.temp.cleanup()

    def _zarr(self, condition_id: str, *, wrong_channels=False, low=10, high=80) -> Path:
        path = self.cache / f'{condition_id}.ome.zarr'
        root = zarr.open_group(str(path), mode='w')
        data = np.zeros((1, 3, 128, 128), dtype=np.uint16)
        data[0, 1] = low
        yy, xx = np.indices((128, 128))
        data[0, 1][(xx - 40) ** 2 + (yy - 40) ** 2 <= (0.86 * 20) ** 2] = high
        data[0, 1][(xx - 90) ** 2 + (yy - 90) ** 2 <= (0.86 * 20) ** 2] = low
        root.create_dataset('0', data=data, chunks=(1, 1, 32, 32))
        labels = ['RFP', 'GFP', 'DIC'] if wrong_channels else ['GFP 488', 'RFP 561', 'DIC1']
        root.attrs['omero'] = {'channels': [
            {'label': label, 'window': {'start': 0, 'end': 4095}} for label in labels]}
        root.attrs['multiscales'] = [{
            'axes': [{'name': 't'}, {'name': 'c'},
                     {'name': 'y', 'unit': 'micrometer'},
                     {'name': 'x', 'unit': 'micrometer'}],
            'datasets': [{'path': '0', 'coordinateTransformations': [{
                'type': 'scale', 'scale': [1, 1, PIXEL_SIZE, PIXEL_SIZE]}]}],
        }]
        return path

    def _condition(self, condition_id: str, *, wrong_channels=False) -> Path:
        folder = self.results / condition_id; folder.mkdir(parents=True, exist_ok=True)
        source = self._zarr(condition_id, wrong_channels=wrong_channels)
        (folder / 'condition_summary.json').write_text(json.dumps({
            'completion_status': 'completed', 'condition_id': condition_id,
            'condition_name': f'{condition_id}_original',
            'pixel_size_um': {'x': PIXEL_SIZE, 'y': PIXEL_SIZE},
            'channel_mapping': {'gfp_channel': 0, 'dic_channel': 2},
            'benchmark': {'source': str(source)}, 'source_object': {'etag': 'mock'},
        }), encoding='utf-8')
        _write_csv(folder / 'well_measurements.csv', [
            'well_id', 'x_px_fullres', 'y_px_fullres', 'radius_px', 'PDO_count',
            'PDO_present', 'total_PDO_projected_area_px2',
            'total_PDO_projected_area_um2'], [
                [11, 40, 40, 20, 1, True, 25, 25 * PIXEL_SIZE ** 2],
                [27, 90, 90, 20, 0, False, 0, 0],
            ])
        _write_csv(folder / 'pdo_measurements.csv', [
            'well_id', 'pdo_number_in_well', 'centroid_x_px_fullres',
            'centroid_y_px_fullres', 'projected_area_px2', 'projected_area_um2',
            'equivalent_circular_diameter_um'], [
                [11, 1, 40, 40, 25, 25 * PIXEL_SIZE ** 2, 4.5],
            ])
        return folder

    def _args(self, *extra):
        return psc.build_parser().parse_args([
            '--result-root', str(self.results), '--cache-root', str(self.cache),
            '--tile', '256', *extra])

    def _quantify(self, tile, wells=None, index=0, **kwargs):
        wells = wells or [_well()]
        tree = cKDTree(np.asarray([[float(row['x_px_fullres']), float(row['y_px_fullres'])]
                                  for row in wells]))
        return psc.quantify_well(tile, 0, 0, wells[index], wells, tree, PIXEL_SIZE,
                                 tile.dtype, **kwargs)

    def test_interior_mask_and_raw_quantitation(self):
        tile = np.full((121, 121), 10, dtype=np.uint16)
        yy, xx = np.indices(tile.shape)
        tile[(xx - 60) ** 2 + (yy - 60) ** 2 <= (0.86 * 20) ** 2] = 30
        row = self._quantify(tile)
        self.assertEqual(row['RFP_mean_intensity'], 30)
        self.assertEqual(row['RFP_median_intensity'], 30)
        self.assertEqual(row['RFP_integrated_intensity'], 30 * row['interior_pixel_count'])
        self.assertEqual(row['interior_radius_fraction'], 0.86)

    def test_local_background_and_signed_subtraction(self):
        tile = np.full((121, 121), 20, dtype=np.uint16)
        yy, xx = np.indices(tile.shape)
        tile[(xx - 60) ** 2 + (yy - 60) ** 2 <= (0.86 * 20) ** 2] = 5
        row = self._quantify(tile)
        self.assertEqual(row['background_qc'], 'valid_local_background')
        self.assertEqual(row['RFP_background_median'], 20)
        self.assertEqual(row['RFP_background_corrected_mean'], -15)
        self.assertEqual(row['RFP_background_corrected_integrated_intensity'],
                         -15 * row['interior_pixel_count'])
        self.assertEqual(row['RFP_positive_only_excess_integrated_intensity'], 0)

    def test_insufficient_background_produces_nan_derived_metrics(self):
        tile = np.full((31, 31), 10, dtype=np.uint16)
        row = self._quantify(tile, wells=[_well(x=15, y=15, radius=5)])
        self.assertEqual(row['background_qc'], 'insufficient_local_background')
        self.assertTrue(math.isnan(row['RFP_background_median']))
        self.assertTrue(math.isnan(row['RFP_background_corrected_mean']))
        self.assertTrue(math.isnan(row['exploratory_RFP_positive_area_um2']))

    def test_exploratory_area_uses_local_background_p99(self):
        tile = np.full((121, 121), 10, dtype=np.uint16)
        yy, xx = np.indices(tile.shape)
        signal = ((xx - 60) ** 2 + (yy - 60) ** 2 <= 4 ** 2)
        tile[signal] = 40
        row = self._quantify(tile)
        self.assertEqual(row['exploratory_RFP_threshold_intensity'], 10)
        self.assertEqual(row['exploratory_RFP_positive_area_px2'], int(signal.sum()))
        self.assertAlmostEqual(row['exploratory_RFP_positive_area_um2'],
                               signal.sum() * PIXEL_SIZE ** 2)

    def test_neighbouring_well_pixels_are_excluded_from_background(self):
        tile = np.full((140, 140), 10, dtype=np.uint16)
        wells = [_well(1, 50, 70, 20), _well(2, 90, 70, 20)]
        yy, xx = np.indices(tile.shape)
        tile[(xx - 90) ** 2 + (yy - 70) ** 2 <= (1.05 * 20) ** 2] = 1000
        row = self._quantify(tile, wells=wells, index=0, min_background_pixels=100)
        self.assertEqual(row['RFP_background_median'], 10)
        self.assertLess(row['background_valid_fraction'], 1)

    def test_physical_position_conversion(self):
        row = self._quantify(np.full((121, 121), 10, dtype=np.uint16))
        self.assertAlmostEqual(row['x_mm'], 60 * PIXEL_SIZE / 1000)
        self.assertAlmostEqual(row['y_mm'], 60 * PIXEL_SIZE / 1000)

    def test_empty_and_high_rfp_wells(self):
        low = self._quantify(np.zeros((121, 121), dtype=np.uint16))
        high = self._quantify(np.full((121, 121), 4000, dtype=np.uint16))
        self.assertEqual(low['RFP_integrated_intensity'], 0)
        self.assertGreater(high['RFP_integrated_intensity'], low['RFP_integrated_intensity'])

    def test_integrated_join_reuses_exact_final_well_ids(self):
        wells = [
            {**_well(11), 'PDO_present': True, 'PDO_count': 1,
             'total_PDO_projected_area_um2': 12},
            {**_well(27, 60, 6000), 'PDO_present': False, 'PDO_count': 0,
             'total_PDO_projected_area_um2': 0},
        ]
        pdos = [{'well_id': '11', 'equivalent_circular_diameter_um': 4.5}]
        psc_rows = [{**row, 'well_id': row['well_id'], 'RFP_mean_intensity': 10}
                    for row in wells]
        integrated = psc.integrate_pdo_psc('condition', 'name', 5, wells, pdos, psc_rows)
        self.assertEqual([row['well_id'] for row in integrated], ['11', '27'])
        self.assertEqual(integrated[0]['PDO_equivalent_diameter_median_um'], 4.5)
        self.assertTrue(math.isnan(integrated[1]['PDO_equivalent_diameter_median_um']))

    def test_exact_well_set_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'well-set mismatch'):
            psc.integrate_pdo_psc('c', 'n', 0, [_well(1), _well(2)], [],
                                  [{**_well(1), 'RFP_mean_intensity': 1}])

    def test_five_mm_spatial_bins_and_background_counts(self):
        rows = [
            {'condition_id': 'c', 'condition_name': 'n', 'dose_nM': 0, 'well_id': '1',
             'y_mm': 4.999, 'PDO_present': True, 'total_PDO_projected_area_um2': 10,
             'background_qc': 'valid_local_background', 'background_valid_fraction': .8,
             'RFP_mean_intensity': 10, 'RFP_integrated_intensity': 100,
             'RFP_background_corrected_mean': 2,
             'RFP_background_corrected_integrated_intensity': 20, 'RFP_p95': 12,
             'exploratory_RFP_positive_area_um2': 3},
            {'condition_id': 'c', 'condition_name': 'n', 'dose_nM': 0, 'well_id': '2',
             'y_mm': 5.0, 'PDO_present': False, 'total_PDO_projected_area_um2': 0,
             'background_qc': 'insufficient_local_background', 'background_valid_fraction': .05,
             'RFP_mean_intensity': 20, 'RFP_integrated_intensity': 200,
             'RFP_background_corrected_mean': float('nan'),
             'RFP_background_corrected_integrated_intensity': float('nan'), 'RFP_p95': 22,
             'exploratory_RFP_positive_area_um2': float('nan')},
        ]
        bins = psc.spatial_qc(rows, [{'well_id': '1', 'equivalent_circular_diameter_um': 4}])
        self.assertEqual([row['y_bin_index'] for row in bins], [0, 1])
        self.assertEqual(bins[0]['y_bin_end_mm'], 5)
        self.assertEqual(bins[1]['wells_with_insufficient_background'], 1)

    def test_singleton_t_and_z_axes_remain_lazy(self):
        class FakeArray:
            shape = (1, 3, 1, 100, 80)
            def __init__(self): self.selector = None
            def __getitem__(self, selector):
                self.selector = selector
                return np.zeros((5, 7), dtype=np.uint16)
        array = FakeArray(); planes = psc.SingletonTZCYX(array, ['T', 'C', 'Z', 'Y', 'X'])
        planes.read(1, slice(10, 15), slice(20, 27))
        self.assertEqual(array.selector, (0, 1, 0, slice(10, 15), slice(20, 27)))

    def test_full_run_uses_rfp_one_and_never_calls_detection_or_segmentation(self):
        condition = next(iter(psc.CONDITIONS)); folder = self._condition(condition)
        output = io.StringIO()
        with patch.object(analysis_core, 'detect_wells', side_effect=AssertionError('Hough called')), \
             patch.object(analysis_core, 'segment_pdos', side_effect=AssertionError('PDO segmentation called')), \
             redirect_stdout(output):
            code = psc.run(self._args('--condition-id', condition))
        self.assertEqual(code, 0)
        self.assertIn('1/1 conditions completed; all condition well-set QC checks passed.',
                      output.getvalue())
        rows = psc._read_csv(folder / 'psc_quantification' / 'psc_well_measurements.csv')
        self.assertEqual({row['well_id'] for row in rows}, {'11', '27'})
        self.assertEqual({row['RFP_channel'] for row in rows}, {'1'})
        self.assertGreater(float(next(row for row in rows if row['well_id'] == '11')['RFP_mean_intensity']),
                           float(next(row for row in rows if row['well_id'] == '27')['RFP_mean_intensity']))

    def test_restart_skips_completed_condition(self):
        condition = next(iter(psc.CONDITIONS)); self._condition(condition)
        args = self._args('--condition-id', condition)
        self.assertEqual(psc.run(args), 0)
        def no_open(*_args, **_kwargs):
            raise AssertionError('completed condition reopened OME-Zarr')
        self.assertEqual(psc.run(args, open_group=no_open), 0)

    def test_one_condition_failure_does_not_stop_next(self):
        first, second = list(psc.CONDITIONS)[:2]
        self._condition(first, wrong_channels=True)
        second_folder = self._condition(second)
        code = psc.run(self._args('--condition-id', first, '--condition-id', second))
        self.assertEqual(code, 1)
        failed = psc._read_json(self.results / first / 'psc_quantification' / 'psc_summary.json')
        self.assertEqual(failed['completion_status'], 'failed')
        self.assertTrue((second_folder / 'psc_quantification' /
                         'psc_well_measurements.csv').is_file())


if __name__ == '__main__':
    unittest.main()
