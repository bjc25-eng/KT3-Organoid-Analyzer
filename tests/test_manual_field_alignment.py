import numpy as np
import pandas as pd

from manual_field_alignment import (
    ManualTransform,
    affine_day7_to_day10,
    mutual_nearest_well_matches,
    transform_points,
)


def test_affine_centres_day7_in_day10_and_applies_shift():
    m = affine_day7_to_day10((100, 100, 3), (200, 200, 3), ManualTransform(1.0, 0.0, 10.0, -5.0))
    mapped = transform_points(np.array([[50.0, 50.0]]), m)[0]
    np.testing.assert_allclose(mapped, [110.0, 95.0], atol=1e-6)


def test_affine_respects_scale_around_day7_centre():
    m = affine_day7_to_day10((100, 100, 3), (200, 200, 3), ManualTransform(0.5, 0.0, 0.0, 0.0))
    mapped = transform_points(np.array([[70.0, 50.0]]), m)[0]
    np.testing.assert_allclose(mapped, [110.0, 100.0], atol=1e-6)


def test_mutual_nearest_matches_correct_offset_wells():
    d7 = pd.DataFrame({
        "well_index": ["1,1", "2,1", "3,1"],
        "well_centre_x_px": [10.0, 30.0, 50.0],
        "well_centre_y_px": [10.0, 10.0, 10.0],
        "PDO_count": [1, 0, 1],
        "PSC_like_focus_count": [0, 1, 2],
        "um_per_pixel": [1.0, 1.0, 1.0],
    })
    d10 = pd.DataFrame({
        "well_index": ["11,9", "12,9", "13,9", "14,9"],
        "well_centre_x_px": [110.0, 130.0, 150.0, 170.0],
        "well_centre_y_px": [90.0, 90.0, 90.0, 90.0],
        "PDO_count": [1, 0, 1, 0],
        "PSC_like_focus_count": [0, 1, 2, 0],
        "um_per_pixel": [2.0, 2.0, 2.0, 2.0],
    })
    m = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 80.0]])
    out = mutual_nearest_well_matches(d7, d10, m)
    assert out["day10_well_index"].tolist() == ["11,9", "12,9", "13,9"]
    np.testing.assert_allclose(out["match_error_px"], 0.0)
    np.testing.assert_allclose(out["match_error_um"], 0.0)
