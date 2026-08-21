import math
from pathlib import Path

import numpy as np
import pytest

import aws_pdo_full_component_qc as qc


class FakePlanes:
    def __init__(self, gfp):
        self.data = np.zeros((3, *gfp.shape), dtype=np.uint8)
        self.data[0] = gfp
        self.shape_cyx = self.data.shape

    def read(self, channel, ys, xs):
        return self.data[channel, ys, xs]


def well(well_id='1', x=50, y=50, radius=10, count=1):
    return {
        'well_id': well_id, 'x_px_fullres': x, 'y_px_fullres': y,
        'radius_px': radius, 'PDO_count': count, 'PDO_present': count > 0,
        'hex_array_member': True, 'lattice_degree': 6,
    }


def pdo_from_component(well_id, number, component, pixel_size=1.0):
    area = component.area_px2
    return {
        'well_id': well_id, 'pdo_number_in_well': number,
        'centroid_x_px_fullres': component.centroid_x_fullres,
        'centroid_y_px_fullres': component.centroid_y_fullres,
        'projected_area_px2': area,
        'projected_area_um2': area * pixel_size**2,
        'equivalent_circular_diameter_um': 2 * math.sqrt(area / math.pi) * pixel_size,
    }


def production_rows(planes, target_well, minimum=3):
    return qc.reproduce_production_components(
        planes, target_well, planes.data.shape[2], planes.data.shape[1], 255, minimum
    )


def test_strict_production_reproduction_identity_and_tolerances():
    gfp = np.zeros((101, 101), np.uint8)
    gfp[47:53, 47:53] = 255
    planes = FakePlanes(gfp)
    components = production_rows(planes, well())
    assert len(components) == 1
    row = pdo_from_component('1', 1, components[0])
    matched, differences = qc.match_production_row(row, components, 1.0)
    assert matched is components[0]
    assert differences['production_projected_area_difference_px2'] == 0

    row['centroid_x_px_fullres'] += 1.1e-6
    assert qc.match_production_row(row, components, 1.0)[0] is None
    row['centroid_x_px_fullres'] -= 1.1e-6
    row['projected_area_px2'] += 1
    assert qc.match_production_row(row, components, 1.0)[0] is None


def test_reproduction_uses_original_0p86r_mask_and_no_splitting():
    gfp = np.zeros((101, 101), np.uint8)
    gfp[48:52, 48:70] = 255
    planes = FakePlanes(gfp)
    components = production_rows(planes, well(radius=10))
    assert len(components) == 1
    assert np.max(components[0].coords_yx_fullres[:, 1]) <= 59


def test_pixel_overlap_requires_all_pixels_in_one_component():
    coords = np.array([[5, 5], [5, 6], [5, 7]], dtype=np.int64)
    clipped = qc.Component(1, coords, 3, 6, 5, 255)
    labels = np.zeros((12, 12), np.int32)
    labels[5, 5:8] = 4
    assert qc.overlap_match(clipped, labels, 0, 0) == (4, 3, 1.0, '')

    labels[5, 6] = 0
    assert qc.overlap_match(clipped, labels, 0, 0)[3] == 'clipped_pixels_became_background'
    labels[5, 6] = 5
    assert qc.overlap_match(clipped, labels, 0, 0)[3] == 'clipped_pixels_map_to_multiple_components'
    labels[:] = 0
    assert qc.overlap_match(clipped, labels, 0, 0)[3] == 'no_unmasked_component'


def test_true_mask_containment_uses_foreground_pixels_not_equivalent_circle():
    coords = np.array([[50, 50], [50, 51], [50, 60], [50, 61]], dtype=np.int64)
    component = qc.Component(1, coords, 4, 55.5, 50, 255)
    result = qc.mask_containment(component, well(radius=10), 0.5, 0.5)
    assert result['full_component_area_px2'] == 4
    assert result['full_component_area_um2'] == 1
    assert result['full_component_pixels_inside_final_well'] == 3
    assert result['full_component_pixels_outside_final_well'] == 1
    assert result['full_component_fraction_inside_final_well'] == 0.75
    assert result['full_component_crosses_final_well_boundary'] is True
    assert result['full_component_touches_final_well_boundary'] is True


def test_component_hash_is_canonical_for_overlapping_crop_recoveries():
    coords = np.array([[10, 20], [10, 21], [11, 20]], dtype=np.int64)
    first = qc.Component(1, coords, 3, 20.333333, 10.333333, 255)
    second = qc.Component(8, coords[[2, 0, 1]], 3, 20.333333, 10.333333, 255)
    one = qc.component_hash('condition', first)
    two = qc.component_hash('condition', second)
    assert one[0] == two[0]
    assert one[1] == (20, 10, 22, 12)
    assert np.array_equal(one[2], two[2])
    assert qc.component_hash('different-condition', first)[0] != one[0]


def test_overlapping_nonidentical_canonical_masks_are_detected_as_conflict():
    assert qc.canonical_masks_overlap((0, 0, 3, 2), np.array([[1, 1, 0], [0, 0, 0]], np.uint8),
                                      (1, 0, 4, 2), np.array([[1, 0, 0], [0, 0, 0]], np.uint8))
    assert not qc.canonical_masks_overlap((0, 0, 2, 2), np.ones((2, 2), np.uint8),
                                          (3, 3, 5, 5), np.ones((2, 2), np.uint8))


def test_crop_expands_until_component_clears_four_pixel_guard():
    gfp = np.zeros((201, 201), np.uint8)
    gfp[98:103, 78:126] = 255  # touches 2r crop guard but clears 3r crop
    planes = FakePlanes(gfp)
    target = well(x=100, y=100, radius=10)
    clipped = production_rows(planes, target)[0]
    full, status = qc.recover_unmasked_component(planes, clipped, target, 201, 201, 255, 3)
    assert full is not None
    assert status['crop_expansion_count'] == 1
    assert status['crop_half_width_px'] == 30
    assert status['full_component_extent_status'] == 'complete'
    assert status['clipped_component_overlap_fraction'] == 1.0


def test_component_at_maximum_crop_is_untrusted():
    gfp = np.zeros((201, 201), np.uint8)
    gfp[98:103, 25:176] = 255
    planes = FakePlanes(gfp)
    target = well(x=100, y=100, radius=10)
    clipped = production_rows(planes, target)[0]
    full, status = qc.recover_unmasked_component(planes, clipped, target, 201, 201, 255, 3)
    assert full is None
    assert status['full_component_extent_status'] == 'incomplete_at_maximum_crop'
    assert status['unmasked_component_match_failure_reason'] == 'crop_extent_incomplete'

    row = pdo_from_component('1', 1, clipped)
    rows, components = qc.quantify_condition_components(
        'K3T_PSC_RMC6236_5nm_Lane_2', [target], [row], planes, 201, 201, 255, 3, 1, 1
    )
    assert components == []
    assert rows[0]['full_component_extent_status'] == 'incomplete_at_maximum_crop'
    assert all(math.isnan(rows[0][field]) for field in qc.TRUSTED_NUMERIC_FIELDS)


def test_component_at_source_boundary_is_untrusted_and_bounds_are_recorded():
    gfp = np.zeros((101, 101), np.uint8)
    gfp[8:13, 0:18] = 255
    planes = FakePlanes(gfp)
    target = well(x=10, y=10, radius=10)
    clipped = production_rows(planes, target)[0]
    full, status = qc.recover_unmasked_component(planes, clipped, target, 101, 101, 255, 3)
    assert full is None
    assert status['full_component_extent_status'] == 'incomplete_at_source_boundary'
    assert status['requested_crop_bounds'] != status['actual_source_clipped_crop_bounds']


def test_untrusted_reproduction_has_nan_not_zero_containment():
    gfp = np.zeros((101, 101), np.uint8)
    gfp[47:53, 47:53] = 255
    planes = FakePlanes(gfp)
    target = well()
    component = production_rows(planes, target)[0]
    row = pdo_from_component('1', 1, component)
    row['projected_area_px2'] += 1
    rows, components = qc.quantify_condition_components(
        'K3T_PSC_RMC6236_5nm_Lane_2', [target], [row], planes, 101, 101, 255, 3, 1, 1
    )
    assert components == []
    assert rows[0]['production_component_reproduction_status'] == 'production_component_reproduction_failed'
    for field in qc.TRUSTED_NUMERIC_FIELDS:
        assert math.isnan(rows[0][field])


def test_many_production_pdos_share_one_component_summary_record():
    gfp = np.zeros((101, 101), np.uint8)
    gfp[48:53, 22:43] = 255
    planes = FakePlanes(gfp)
    wells = [well('1', x=30, y=50, radius=10), well('2', x=35, y=50, radius=10)]
    pdos = [pdo_from_component(str(index), 1, production_rows(planes, item)[0])
            for index, item in enumerate(wells, 1)]
    rows, components = qc.quantify_condition_components(
        'K3T_PSC_RMC6236_5nm_Lane_2', wells, pdos, planes, 101, 101, 255, 3, 1, 1
    )
    assert len(components) == 1
    assert components[0]['unmasked_component_production_PDO_count'] == 2
    assert components[0]['many_production_PDOs_to_one_unmasked_component'] is True
    assert all(row['unmasked_component_id'] == components[0]['unmasked_component_id'] for row in rows)


def _diagnostic_row(well_id, known=False, trusted=True):
    row = {field: math.nan for field in qc.PDO_FIELDS}
    row.update({
        'well_id': str(well_id), 'pdo_number_in_well': 1,
        'known_visual_failure': known,
        'unmasked_component_match_status': 'trusted_complete_match' if trusted else 'failed',
        'full_component_fraction_inside_final_well': 0.5,
        'full_component_crosses_final_well_boundary': True,
        'full_component_area_px2': 20,
        'full_component_centroid_x_px': 0,
        'full_component_centroid_y_px': 0,
        'well_x_px_fullres': 0, 'well_y_px_fullres': 0, 'well_radius_px': 10,
        'many_production_PDOs_to_one_unmasked_component': False,
    })
    return row


def test_dmso_606_and_624_are_mandatory_additional_diagnostics():
    ordinary = [_diagnostic_row(index) for index in range(1, 12)]
    known = [_diagnostic_row(606, True), _diagnostic_row(624, True)]
    selected = qc.select_diagnostics(qc.DMSO_CONDITION, ordinary + known)
    selected_ids = {row['well_id'] for row in selected}
    assert {'606', '624'} <= selected_ids
    assert selected_ids - {'606', '624'}
    assert all('known_visual_failure_mandatory' in row['diagnostic_sampling_reasons']
               for row in selected if row['well_id'] in {'606', '624'})

    with pytest.raises(RuntimeError, match='Mandatory known visual failure'):
        qc.select_diagnostics(qc.DMSO_CONDITION, ordinary + known[:1])


def test_scientific_settings_and_gfp_provenance_are_strict():
    summary = {
        'scientific_settings': {'green_low': 30, 'green_high': 45, 'pdo_min_area': 20},
        'channel_mapping': {'gfp_channel': 0, 'dic_channel': 2},
    }
    assert qc.validate_scientific_settings(summary) == 20
    changed = {**summary, 'scientific_settings': {**summary['scientific_settings'], 'green_low': 29}}
    with pytest.raises(RuntimeError, match='thresholds differ'):
        qc.validate_scientific_settings(changed)
    changed = {**summary, 'channel_mapping': {'gfp_channel': 1}}
    with pytest.raises(RuntimeError, match='GFP channel provenance'):
        qc.validate_scientific_settings(changed)

    class Root:
        attrs = {'omero': {'channels': [{'window': {'end': 4095}}]}}
    assert qc.quantitative_window_end(Root(), 0, np.dtype('uint16')) == (4095, 'ome_omero_channel_window_end')
    Root.attrs = {}
    assert qc.quantitative_window_end(Root(), 0, np.dtype('uint16')) == (65535, 'source_dtype_maximum_fallback')


def test_status_vocabulary_and_output_scope_are_fixed():
    assert qc.MATCH_FAILURE_REASONS == {
        '', 'clipped_pixels_became_background', 'clipped_pixels_map_to_multiple_components',
        'no_unmasked_component', 'crop_extent_incomplete', 'cross_crop_identity_conflict', 'other',
    }
    assert qc.QC_STATUS == 'diagnostic_measurement_only_no_exclusion_rule'
    assert qc.GEOMETRY_PROVENANCE == 'true_pixel_mask_of_matched_unmasked_GFP_connected_component'
    assert qc.OUTPUT_DIRECTORY == 'pdo_full_component_qc'
    source = Path(qc.__file__).read_text(encoding='utf-8')
    forbidden_calls = ('detect_psc(', 'HoughCircles(', 'bioformats2raw', 'segment_pdos(')
    assert all(token not in source for token in forbidden_calls)


def test_diagnostic_layout_has_four_requested_panels():
    component = qc.Component(1, np.array([[49, 49], [49, 50], [50, 49], [50, 50]]),
                             4, 49.5, 49.5, 255)
    row = _diagnostic_row(1)
    row.update({
        'condition_name': 'test', 'dose': '5 nM',
        'production_PDO_equivalent_circular_diameter_um': 2.0,
        'production_PDO_projected_area_px2': 4.0,
        'production_PDO_centroid_x_px_fullres': 49.5,
        'production_PDO_centroid_y_px_fullres': 49.5,
        'production_component_reproduction_status': 'reproduced_exactly',
        'clipped_component_overlap_fraction': 1.0,
        'full_component_fraction_outside_final_well': 0.0,
        'full_component_fraction_inside_production_0p86r': 1.0,
        'full_component_extent_status': 'complete',
        '_clipped_component': component, '_full_component': component,
        'diagnostic_sampling_reasons': 'central_control',
    })
    image = qc.diagnostic_image(row, well(), np.zeros((41, 41), np.uint8),
                                np.zeros((41, 41), np.uint8), (30, 30, 71, 71),
                                (0, 255), 64)
    assert image.width == 4 * 64 + 5 * 8
    assert image.height > 64
    image.close()
