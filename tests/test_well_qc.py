import numpy as np

from well_qc import assess_microwell_boundary


def _ring_image(size=220, cx=110, cy=110, radius=68):
    yy, xx = np.indices((size, size))
    rr = np.hypot(xx - cx, yy - cy)
    img = np.full((size, size), 170.0, dtype=np.float32)
    img[(rr >= radius - 5) & (rr <= radius + 5)] = 35.0
    return np.stack([img, img, img], axis=-1).astype(np.uint8)


def _linear_feature_image(size=220):
    img = np.full((size, size), 170, dtype=np.uint8)
    img[:, 155:165] = 35
    return np.stack([img, img, img], axis=-1)


def test_real_circular_wall_is_accepted():
    result = assess_microwell_boundary(_ring_image(), 110, 110, 68)
    assert result["well_validity_status"] == "accepted"
    assert result["well_wall_evidence_score"] > result["well_wall_evidence_threshold"]


def test_linear_nonwell_feature_is_rejected():
    result = assess_microwell_boundary(_linear_feature_image(), 110, 110, 68)
    assert result["well_validity_status"] == "rejected_false_well"
    assert result["well_wall_evidence_score"] <= result["well_wall_evidence_threshold"]
