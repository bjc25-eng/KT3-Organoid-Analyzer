from __future__ import annotations

"""Conservative DIC validation for detected KT3 microwells.

The Hough/hex-array detector can occasionally place a mathematically plausible
circle on a non-well image feature. This module measures whether the DIC image
actually contains a circumferential dark-wall trough around the calibrated well
radius before a PDO assigned to that circle is accepted.

The default threshold is data-derived from the manually reviewed
KT3_day_10005 OME-Zarr rerun: the two confirmed false wells (9,8 and 9,86)
scored 0.300 and 0.311, while the lowest-scoring visually confirmed real well
(30,144) scored 0.364. The midpoint, 0.337, is therefore used as a conservative
separator for this KT3 dataset rather than an arbitrary image-intensity cutoff.
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter

WELL_VALIDITY_SCORE_THRESHOLD = 0.337
WELL_VALIDITY_ANGLE_COUNT = 72
WELL_VALIDITY_RADIAL_SAMPLE_COUNT = 100


def _setting(settings, name: str, default):
    return getattr(settings, name, default) if settings is not None else default


def assess_microwell_boundary(
    rgb: np.ndarray,
    well_x: float,
    well_y: float,
    well_r: float,
    settings=None,
) -> dict:
    """Assess whether DIC supports a real microwell boundary.

    A radial DIC profile is sampled around the full circumference. For each
    angular ray, the darkest trough in the expected wall band (0.72-1.10 x the
    calibrated radius) is compared with the mean of inner and outer local DIC
    baselines. The median normalized trough depth across angles is the final
    well-validity score.
    """
    threshold = float(
        _setting(settings, "well_validity_score_threshold", WELL_VALIDITY_SCORE_THRESHOLD)
    )
    if well_r <= 0 or rgb.size == 0:
        return {
            "well_validity_status": "not_evaluable",
            "well_validity_reason": "invalid well radius or empty DIC crop",
            "well_wall_evidence_score": float("nan"),
            "well_wall_evidence_threshold": threshold,
        }

    a = np.asarray(rgb)
    if a.ndim == 3 and a.shape[2] >= 3:
        dic = (a[..., 0].astype(np.float32) + a[..., 2].astype(np.float32)) / 2.0
    else:
        dic = a.astype(np.float32)
    dic = gaussian_filter(dic, 1.0)

    yy, xx = np.indices(dic.shape)
    rr = np.hypot(xx - float(well_x), yy - float(well_y))
    local = dic[rr <= 1.35 * float(well_r)]
    if local.size < 20:
        return {
            "well_validity_status": "not_evaluable",
            "well_validity_reason": "insufficient DIC pixels around detected well",
            "well_wall_evidence_score": float("nan"),
            "well_wall_evidence_threshold": threshold,
        }

    p10, p90 = np.percentile(local, [10.0, 90.0])
    robust_range = max(float(p90 - p10), 1.0)
    angle_count = max(
        24,
        int(_setting(settings, "well_validity_angle_count", WELL_VALIDITY_ANGLE_COUNT)),
    )
    radial_count = max(
        40,
        int(
            _setting(
                settings,
                "well_validity_radial_sample_count",
                WELL_VALIDITY_RADIAL_SAMPLE_COUNT,
            )
        ),
    )

    radii = np.linspace(0.45 * float(well_r), 1.25 * float(well_r), radial_count)
    inner_mask = (radii >= 0.45 * float(well_r)) & (radii <= 0.65 * float(well_r))
    wall_mask = (radii >= 0.72 * float(well_r)) & (radii <= 1.10 * float(well_r))
    outer_mask = (radii >= 1.12 * float(well_r)) & (radii <= 1.25 * float(well_r))

    depths: list[float] = []
    for theta in np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False):
        xs = float(well_x) + radii * math.cos(float(theta))
        ys = float(well_y) + radii * math.sin(float(theta))
        xi = np.clip(np.rint(xs).astype(int), 0, dic.shape[1] - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, dic.shape[0] - 1)
        profile = dic[yi, xi]
        inner = float(np.median(profile[inner_mask]))
        outer = float(np.median(profile[outer_mask]))
        wall = float(np.min(profile[wall_mask]))
        depths.append((((inner + outer) / 2.0) - wall) / robust_range)

    score = float(np.median(np.asarray(depths, dtype=float))) if depths else float("nan")
    if np.isfinite(score) and score <= threshold:
        status = "rejected_false_well"
        reason = "DIC lacks sufficient circumferential microwell-wall evidence"
    else:
        status = "accepted"
        reason = "DIC supports a genuine microwell boundary"

    return {
        "well_validity_status": status,
        "well_validity_reason": reason,
        "well_wall_evidence_score": score,
        "well_wall_evidence_threshold": threshold,
    }
