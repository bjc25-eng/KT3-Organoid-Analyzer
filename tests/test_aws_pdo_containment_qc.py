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

import aws_pdo_containment_qc as qc


PIXEL_SIZE = qc.EXPECTED_PIXEL_SIZE_UM


def _write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sample_object(well_id: str, pdo_number: int, *, fraction: float, clearance: float,
                   centroid_inside=True, boundary=False, normalized=0.2,
                   diameter=5.0, known=False) -> dict:
    return {
        'well_id': well_id, 'pdo_number_in_well': pdo_number,
        'known_visual_failure': known, 'PDO_fraction_inside_well': fraction,
        'PDO_fraction_outside_well': 1.0 - fraction,
        'PDO_edge_clearance_px': clearance, 'PDO_edge_clearance_um': clearance * PIXEL_SIZE,
        'normalized_PDO_edge_clearance': clearance / 10,
        'PDO_centroid_inside_well': centroid_inside,
        'PDO_boundary_intersection': boundary,
        'normalized_PDO_centroid_radius': normalized,
        'PDO_equivalent_circular_diameter_um': diameter,
    }


class ContainmentGeometryTests(unittest.TestCase):
    def test_fully_inside(self):
        row = qc.containment_geometry(10, 2, 3, 0, 0.5)
        self.assertEqual(row['PDO_fraction_inside_well'], 1.0)
        self.assertEqual(row['PDO_fraction_outside_well'], 0.0)
        self.assertEqual(row['PDO_edge_clearance_px'], 5.0)
        self.assertFalse(row['PDO_boundary_intersection'])

    def test_tangent_internally(self):
        row = qc.containment_geometry(10, 2, 8, 0, 0.5)
        self.assertEqual(row['PDO_fraction_inside_well'], 1.0)
        self.assertEqual(row['PDO_edge_clearance_px'], 0.0)
        self.assertEqual(row['normalized_PDO_edge_clearance'], 0.0)
        self.assertTrue(row['PDO_boundary_intersection'])

    def test_partial_overlap(self):
        row = qc.containment_geometry(10, 4, 8, 0, 0.5)
        self.assertGreater(row['PDO_fraction_inside_well'], 0)
        self.assertLess(row['PDO_fraction_inside_well'], 1)
        self.assertTrue(row['PDO_boundary_intersection'])

    def test_centroid_outside_with_overlap(self):
        row = qc.containment_geometry(10, 4, 11, 0, 0.5)
        self.assertFalse(row['PDO_centroid_inside_well'])
        self.assertGreater(row['PDO_fraction_inside_well'], 0)
        self.assertTrue(row['PDO_boundary_intersection'])

    def test_no_overlap(self):
        row = qc.containment_geometry(10, 2, 13, 0, 0.5)
        self.assertEqual(row['PDO_fraction_inside_well'], 0.0)
        self.assertEqual(row['PDO_fraction_outside_well'], 1.0)
        self.assertFalse(row['PDO_boundary_intersection'])

    def test_pdo_larger_than_well(self):
        row = qc.containment_geometry(5, 10, 1, 0, 0.5)
        self.assertAlmostEqual(row['PDO_fraction_inside_well'], 0.25)
        self.assertFalse(row['PDO_boundary_intersection'])
        self.assertLess(row['normalized_PDO_edge_clearance'], 0)

    def test_concentric_circles(self):
        row = qc.containment_geometry(10, 3, 0, 0, 0.5)
        self.assertEqual(row['PDO_centroid_distance_px'], 0.0)
        self.assertEqual(row['normalized_PDO_centroid_radius'], 0.0)
        self.assertEqual(row['PDO_fraction_inside_well'], 1.0)

    def test_numerical_near_tangency_is_finite_and_bounded(self):
        row = qc.containment_geometry(10, 2, 12 - 1e-10, 0, 0.5)
        self.assertTrue(math.isfinite(row['PDO_fraction_inside_well']))
        self.assertGreaterEqual(row['PDO_fraction_inside_well'], 0.0)
        self.assertLessEqual(row['PDO_fraction_inside_well'], 1.0)
        self.assertTrue(row['PDO_boundary_intersection'])


class DiagnosticSamplingTests(unittest.TestCase):
    def _ordinary_fixture(self) -> list[dict]:
        return [
            _sample_object('1', 1, fraction=.10, clearance=-5, centroid_inside=False,
                           boundary=True, normalized=1.2, diameter=5),
            _sample_object('2', 1, fraction=.20, clearance=-4, centroid_inside=False,
                           boundary=True, normalized=1.1, diameter=6),
            _sample_object('3', 1, fraction=.30, clearance=-3, boundary=True,
                           normalized=.9, diameter=7),
            _sample_object('4', 1, fraction=.99, clearance=-.1, boundary=True,
                           normalized=.8, diameter=8),
            _sample_object('5', 1, fraction=1, clearance=.1, normalized=.7, diameter=9),
            _sample_object('6', 1, fraction=1, clearance=5, normalized=.1, diameter=10),
            _sample_object('7', 1, fraction=1, clearance=4, normalized=.2, diameter=1),
            _sample_object('8', 1, fraction=1, clearance=3, normalized=.3, diameter=30),
            _sample_object('9', 1, fraction=1, clearance=2, normalized=.4, diameter=4),
            _sample_object('9', 2, fraction=1, clearance=1, normalized=.5, diameter=5),
        ]

    def test_strengthened_deterministic_sampling_exact_fixture(self):
        condition = list(qc.CONDITIONS)[1]
        rows = qc.select_diagnostics(condition, self._ordinary_fixture())
        self.assertEqual([qc._object_key(row) for row in rows],
                         [('1', 1), ('2', 1), ('3', 1), ('4', 1), ('5', 1),
                          ('6', 1), ('7', 1), ('8', 1), ('9', 1), ('9', 2)])
        reasons = {qc._object_key(row): set(row['diagnostic_sampling_reasons'].split(';'))
                   for row in rows}
        self.assertIn('lowest_fraction_inside', reasons[('1', 1)])
        self.assertIn('most_negative_edge_clearance', reasons[('1', 1)])
        self.assertIn('centroid_outside', reasons[('1', 1)])
        self.assertIn('boundary_intersection_nearest_full_containment', reasons[('4', 1)])
        self.assertIn('fully_contained_near_wall_control', reasons[('5', 1)])
        self.assertIn('fully_contained_central_control', reasons[('6', 1)])
        self.assertIn('smallest_PDO', reasons[('7', 1)])
        self.assertIn('largest_PDO', reasons[('8', 1)])
        self.assertEqual(reasons[('9', 1)], {'multi_PDO_well'})
        self.assertEqual(reasons[('9', 2)], {'multi_PDO_well'})

    def test_categories_do_not_impose_fixed_total_and_reasons_are_deduplicated(self):
        condition = list(qc.CONDITIONS)[1]
        one = [_sample_object('1', 1, fraction=1, clearance=4, diameter=5)]
        rows = qc.select_diagnostics(condition, one)
        self.assertEqual(len(rows), 1)
        reasons = rows[0]['diagnostic_sampling_reasons'].split(';')
        self.assertEqual(len(reasons), len(set(reasons)))

    def test_wells_606_and_624_are_mandatory_additional_examples(self):
        ordinary = self._ordinary_fixture()
        known = [
            _sample_object('606', 1, fraction=.4, clearance=-2, known=True),
            _sample_object('624', 1, fraction=.5, clearance=-1, known=True),
            _sample_object('624', 2, fraction=.6, clearance=-.5, known=True),
        ]
        rows = qc.select_diagnostics(qc.DMSO_CONDITION, ordinary + known)
        keys = {qc._object_key(row) for row in rows}
        self.assertTrue({('606', 1), ('624', 1), ('624', 2)}.issubset(keys))
        for row in rows:
            if row['well_id'] in {'606', '624'}:
                self.assertIn('known_visual_failure_mandatory',
                              row['diagnostic_sampling_reasons'])
        self.assertTrue({qc._object_key(row) for row in ordinary}.issubset(keys))

    def test_missing_known_failure_well_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'Mandatory known visual failure PDO wells missing'):
            qc.select_diagnostics(
                qc.DMSO_CONDITION,
                [_sample_object('606', 1, fraction=.4, clearance=-2, known=True)],
            )


class ContainmentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = self.root / 'results'
        self.cache = self.root / 'cache'
        self.condition = list(qc.CONDITIONS)[1]

    def tearDown(self):
        self.temporary.cleanup()

    def _make_zarr(self) -> Path:
        path = self.cache / 'Original retained acquisition name.ome.zarr'
        root = zarr.open_group(str(path), mode='w')
        data = np.zeros((1, 3, 80, 84), dtype=np.uint16)
        yy, xx = np.indices((80, 84))
        data[0, 0] = xx * 5 + yy
        data[0, 1] = xx * 2 + yy
        data[0, 2] = xx + yy * 3
        root.create_dataset('0', data=data, chunks=(1, 1, 20, 21))
        root.attrs['omero'] = {'channels': [
            {'label': 'GFP 488', 'window': {'start': 0, 'end': 800}},
            {'label': 'RFP 561', 'window': {'start': 0, 'end': 900}},
            {'label': 'DIC transmitted', 'window': {'start': 0, 'end': 1000}},
        ]}
        root.attrs['multiscales'] = [{
            'axes': [{'name': 't'}, {'name': 'c'},
                     {'name': 'y', 'unit': 'micrometer'},
                     {'name': 'x', 'unit': 'micrometer'}],
            'datasets': [{'path': '0', 'coordinateTransformations': [{
                'type': 'scale', 'scale': [1, 1, PIXEL_SIZE, PIXEL_SIZE],
            }]}],
        }]
        return path

    def _make_inputs(self) -> tuple[Path, Path, Path]:
        zarr_path = self._make_zarr()
        folder = self.results / self.condition
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'condition_summary.json').write_text(json.dumps({
            'completion_status': 'completed', 'condition_id': self.condition,
            'condition_name': 'Original retained acquisition name',
            'omezarr': str(zarr_path),
            'pixel_size_um': {'x': PIXEL_SIZE, 'y': PIXEL_SIZE},
            'channel_mapping': {'gfp_channel': 0, 'dic_channel': 2},
        }), encoding='utf-8')
        wells = [
            {'well_id': '1', 'x_px_fullres': 30, 'y_px_fullres': 30, 'radius_px': 10,
             'PDO_count': 2, 'PDO_present': True, 'total_PDO_projected_area_px2': 30,
             'total_PDO_projected_area_um2': 16.1, 'hex_array_member': True,
             'lattice_degree': 6},
            {'well_id': '2', 'x_px_fullres': 60, 'y_px_fullres': 60, 'radius_px': 10,
             'PDO_count': 0, 'PDO_present': False, 'total_PDO_projected_area_px2': 0,
             'total_PDO_projected_area_um2': 0, 'hex_array_member': True,
             'lattice_degree': 5},
        ]
        pdos = [
            {'well_id': '1', 'pdo_number_in_well': 1, 'centroid_x_px_fullres': 30,
             'centroid_y_px_fullres': 30, 'projected_area_px2': 10,
             'projected_area_um2': 5.37, 'equivalent_circular_diameter_um': 3.0},
            {'well_id': '1', 'pdo_number_in_well': 2, 'centroid_x_px_fullres': 39,
             'centroid_y_px_fullres': 30, 'projected_area_px2': 20,
             'projected_area_um2': 10.74, 'equivalent_circular_diameter_um': 5.0},
        ]
        well_path, pdo_path = folder / 'well_measurements.csv', folder / 'pdo_measurements.csv'
        _write_csv(well_path, list(wells[0]), wells)
        _write_csv(pdo_path, list(pdos[0]), pdos)
        return folder, well_path, pdo_path

    def _args(self):
        return qc.build_parser().parse_args([
            '--result-root', str(self.results), '--cache-root', str(self.cache),
            '--condition-id', self.condition, '--panel-size', '128',
            '--display-sample-size', '16', '--display-sample-grid', '2',
        ])

    def test_object_and_well_aggregation_preserve_proxy_provenance(self):
        folder, _, _ = self._make_inputs()
        wells = qc._read_csv(folder / 'well_measurements.csv')
        pdos = qc._read_csv(folder / 'pdo_measurements.csv')
        objects = qc.calculate_object_rows(self.condition, '5 nM RMC6236', '5 nM',
                                           wells, pdos, PIXEL_SIZE)
        summaries = qc.aggregate_wells(self.condition, '5 nM RMC6236', '5 nM', wells, objects)
        self.assertEqual(len(objects), 2)
        self.assertTrue(all(row['containment_geometry_provenance'] == qc.GEOMETRY_PROVENANCE
                            for row in objects))
        self.assertEqual(summaries[0]['minimum_normalized_PDO_edge_clearance'],
                         min(row['normalized_PDO_edge_clearance'] for row in objects))
        self.assertEqual(summaries[0]['maximum_PDO_fraction_outside_well'],
                         max(row['PDO_fraction_outside_well'] for row in objects))

    def test_workflow_writes_only_new_qc_tree_and_combined_measurements(self):
        folder, well_path, pdo_path = self._make_inputs()
        before_wells, before_pdos = well_path.read_bytes(), pdo_path.read_bytes()
        sentinels = []
        for name in ('pdo_positive_crops_final_pdo_rfp', 'psc_object_quantification'):
            sentinel = folder / name / 'sentinel.txt'
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text('unchanged', encoding='utf-8')
            sentinels.append(sentinel)
        self.assertEqual(qc.run(self._args()), 0)
        output = folder / qc.OUTPUT_DIRECTORY
        self.assertEqual({path.name for path in output.iterdir()}, {
            'pdo_containment_measurements.csv', 'well_containment_summary.csv',
            'diagnostic_manifest.csv', 'labelled_diagnostics', 'contact_sheets',
            'containment_qc_summary.json',
        })
        objects = qc._read_csv(output / 'pdo_containment_measurements.csv')
        manifest = qc._read_csv(output / 'diagnostic_manifest.csv')
        summary = qc._read_json(output / 'containment_qc_summary.json')
        self.assertEqual(len(objects), 2)
        self.assertEqual(summary['exclusion_rule'], None)
        self.assertEqual(summary['containment_geometry_provenance'], qc.GEOMETRY_PROVENANCE)
        self.assertTrue(all(row['containment_geometry_provenance'] == qc.GEOMETRY_PROVENANCE
                            for row in manifest))
        self.assertTrue(all(Path(row['labelled_diagnostic']).is_file() for row in manifest))
        with Image.open(manifest[0]['labelled_diagnostic']) as image:
            self.assertGreater(image.width, 3 * 100)
        self.assertTrue((self.results / 'all_conditions_pdo_containment_measurements.csv').is_file())
        self.assertTrue((self.results / 'all_conditions_pdo_containment_well_summary.csv').is_file())
        self.assertEqual(well_path.read_bytes(), before_wells)
        self.assertEqual(pdo_path.read_bytes(), before_pdos)
        self.assertTrue(all(path.read_text(encoding='utf-8') == 'unchanged'
                            for path in sentinels))

    def test_header_names_reconstructed_geometry_and_known_failure(self):
        row = {**_sample_object('606', 1, fraction=.4, clearance=-2, known=True),
               'condition_name': 'DMSO', 'dose': '0 nM',
               'PDO_centroid_distance_px': 11, 'PDO_centroid_distance_um': 8,
               'PDO_edge_clearance_um': -1.5, 'diagnostic_sampling_reasons':
               'known_visual_failure_mandatory'}
        lines = qc.diagnostic_header(row)
        self.assertTrue(any('KNOWN VISUAL FAILURE' in line for line in lines))
        self.assertIn('Containment geometry: reconstructed equivalent-area circle', lines)
        self.assertTrue(any('Reconstructed-circle fraction inside:' in line for line in lines))

    def test_no_analysis_exclusion_or_source_mutation_call_route(self):
        source = Path(qc.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        prohibited = {
            'detect_psc', 'detect_wells', 'hough_circle', 'segment_pdos',
            'quantify_well', 'convert_nd2', 'refine_lattice', 'bioformats2raw',
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
        self.assertNotIn('exclusion_threshold', source)
        self.assertNotIn('PSC_like_resolved_object_count', source)
        self.assertEqual(qc.QC_STATUS, 'diagnostic_measurement_only_no_exclusion_rule')


if __name__ == '__main__':
    unittest.main()
