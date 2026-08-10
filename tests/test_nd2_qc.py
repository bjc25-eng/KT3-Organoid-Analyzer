import cv2
import numpy as np
import pytest

from analysis_core import Settings
from nd2_qc import _physical_pixel_size, classify_nd2_gfp_candidates


def _dic_ring(size=220, cx=110, cy=110, radius=68):
    yy, xx = np.indices((size, size))
    rr = np.hypot(xx - cx, yy - cy)
    img = np.full((size, size), 170.0, dtype=np.float32)
    img[(rr >= radius - 5) & (rr <= radius + 5)] = 35.0
    return np.stack([img, img, img], axis=-1).astype(np.uint8)


def _dic_linear(size=220):
    img = np.full((size, size), 170, dtype=np.uint8)
    img[:, 155:165] = 35
    return np.stack([img, img, img], axis=-1)


def _gfp_object(size=220, centre=(110, 110), radius=18):
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(rgb, centre, radius, (0, 180, 0), -1)
    return rgb


def _settings():
    return Settings(
        well_diameter_um=100.0,
        green_low=30.0,
        green_high=45.0,
        pdo_min_area=20,
        split_pdos=True,
        rfp_psc_present=False,
    )


def test_nd2_pixel_size_uses_native_metadata():
    umpp, radius = _physical_pixel_size(
        {"voxel_size_um": {"x": 0.5, "y": 0.5}}, _settings()
    )
    assert umpp == pytest.approx(0.5)
    assert radius == pytest.approx(100.0)


def test_nd2_pixel_size_does_not_infer_missing_calibration():
    with pytest.raises(Exception, match="physical pixel size"):
        _physical_pixel_size({"voxel_size_um": {"x": None, "y": None}}, _settings())


def test_central_pdo_in_real_well_is_retained():
    kept, qc, validity = classify_nd2_gfp_candidates(
        _gfp_object(), _dic_ring(), 110, 110, 68.0, 67.0, _settings()
    )
    assert validity["well_validity_status"] == "accepted"
    assert len(kept) == 1
    assert len(qc) == 1
    assert qc[0]["membership_status"] == "accepted"
    assert qc[0]["included_in_quantitative_output"] is True
    assert qc[0]["component_inside_detected_well_fraction"] == pytest.approx(1.0)


def test_real_gfp_on_false_detected_well_is_rejected():
    kept, qc, validity = classify_nd2_gfp_candidates(
        _gfp_object(), _dic_linear(), 110, 110, 68.0, 67.0, _settings()
    )
    assert validity["well_validity_status"] == "rejected_false_well"
    assert len(kept) == 0
    assert len(qc) == 1
    assert qc[0]["membership_status"] == "rejected_false_well"
    assert qc[0]["included_in_quantitative_output"] is False
