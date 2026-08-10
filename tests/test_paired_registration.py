import numpy as np
import pandas as pd

from paired_registration import RegistrationSettings, match_wells_one_to_one
from registered_tracking import build_registered_tracking


def _grid(nx=8, ny=7, pitch=30.0):
    pts = []
    for y in range(ny):
        for x in range(nx):
            pts.append([x * pitch + (0.5 * pitch if y % 2 else 0.0), y * pitch * 0.866])
    return np.asarray(pts, dtype=float)


def test_affine_well_matching_recovers_shifted_scaled_grid():
    base = _grid()
    angle = np.deg2rad(2.0)
    scale = 1.08
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([
        [scale * c, -scale * s, 42.0],
        [scale * s, scale * c, 18.0],
    ])
    homog = np.c_[base, np.ones(len(base))]
    follow = homog @ matrix.T
    table, metrics = match_wells_one_to_one(
        base,
        follow,
        matrix,
        RegistrationSettings(min_matched_wells=20, min_match_fraction=0.25),
    )
    assert metrics["registration_status"] == "accepted"
    assert metrics["matched_well_count"] == len(base)
    assert table["match_confident"].all()
    assert table["match_error_px"].max() < 1e-6


def test_registered_tracking_uses_physical_ids_not_local_well_labels():
    matched = pd.DataFrame([
        {
            "image_series": 1,
            "physical_trajectory_id": "F01__PHYSWELL0001",
            "baseline_well_index": "4,8",
            "followup_well_index": "7,10",
            "baseline_well_observation_id": "old_a",
            "followup_well_observation_id": "old_b",
            "baseline_PDO_count": 1,
            "followup_PDO_count": 1,
            "baseline_PSC_like_focus_count": 2,
            "followup_PSC_like_focus_count": 3,
            "baseline_total_PDO_area_um2": 100.0,
            "followup_total_PDO_area_um2": 150.0,
            "match_error_px": 1.2,
            "match_error_pitch_fraction": 0.04,
        }
    ])
    tracking = build_registered_tracking(matched, 3.0, "Friday", "Monday", "KT3 + PSC")
    assert len(tracking) == 2
    assert tracking["trajectory_id"].nunique() == 1
    assert set(tracking["well_index"]) == {"4,8", "7,10"}
    monday = tracking.loc[tracking["timepoint_index"] == 2].iloc[0]
    assert monday["relative_total_PDO_area_vs_baseline"] == 1.5
