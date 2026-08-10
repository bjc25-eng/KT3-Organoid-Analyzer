from __future__ import annotations

"""Register paired microwell fields and construct physical longitudinal trajectories.

The array lattice is highly repetitive, so the lattice alone is not sufficient to
resolve an integer-well translation unambiguously. Registration therefore uses DIC
image texture/features first and then snaps the transformed baseline well centres
to detected follow-up well centres with mutual one-to-one nearest-neighbour
matching.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RegistrationSettings:
    ratio_test: float = 0.75
    ransac_reproj_threshold_px: float = 4.0
    max_match_pitch_fraction: float = 0.35
    good_median_error_pitch_fraction: float = 0.20
    min_feature_inliers: int = 8
    min_matched_wells: int = 20
    min_match_fraction: float = 0.25


def _gray_u8(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 3:
        # In the paired-ND2 composite B is the unmodified DIC baseline.
        a = a[..., 2]
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    lo, hi = np.nanpercentile(a, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _feature_affine(baseline: np.ndarray, followup: np.ndarray,
                    settings: RegistrationSettings) -> tuple[np.ndarray, dict]:
    """Estimate baseline->follow-up affine transform from DIC features."""
    a = _gray_u8(baseline)
    b = _gray_u8(followup)

    # CLAHE improves local well-wall/texture features while preserving geometry.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    a = clahe.apply(a)
    b = clahe.apply(b)

    detector_name = "SIFT"
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
        norm = cv2.NORM_L2
    else:
        detector_name = "AKAZE"
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING

    ka, da = detector.detectAndCompute(a, None)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        raise RuntimeError("Insufficient DIC features for field registration.")

    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(da, db, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < settings.ratio_test * pair[1].distance:
            good.append(pair[0])
    if len(good) < settings.min_feature_inliers:
        raise RuntimeError(
            f"Only {len(good)} reliable DIC feature matches were found; "
            "field registration is not sufficiently constrained."
        )

    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(settings.ransac_reproj_threshold_px),
        maxIters=10000,
        confidence=0.995,
        refineIters=50,
    )
    if matrix is None or inlier_mask is None:
        raise RuntimeError("RANSAC could not estimate a DIC affine transform.")
    inliers = int(np.asarray(inlier_mask).ravel().sum())
    if inliers < settings.min_feature_inliers:
        raise RuntimeError(
            f"DIC affine transform has only {inliers} inliers; registration rejected."
        )

    linear = matrix[:, :2].astype(float)
    det = float(np.linalg.det(linear))
    scale = float(np.sqrt(abs(det))) if det != 0 else np.nan
    rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    return matrix.astype(float), {
        "feature_detector": detector_name,
        "feature_matches_after_ratio_test": int(len(good)),
        "feature_inliers": inliers,
        "feature_inlier_fraction": float(inliers / len(good)),
        "estimated_scale": scale,
        "estimated_rotation_deg": rotation,
        "estimated_translation_x_px": float(matrix[0, 2]),
        "estimated_translation_y_px": float(matrix[1, 2]),
    }


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 2)
    homog = np.c_[pts, np.ones(len(pts), dtype=float)]
    return homog @ np.asarray(matrix, dtype=float).T


def median_well_pitch(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 2:
        return np.nan
    tree = cKDTree(pts)
    distances, _ = tree.query(pts, k=2)
    vals = distances[:, 1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if len(vals) else np.nan


def match_wells_one_to_one(
    baseline_points: np.ndarray,
    followup_points: np.ndarray,
    matrix: np.ndarray,
    settings: RegistrationSettings = RegistrationSettings(),
) -> tuple[pd.DataFrame, dict]:
    """Mutual-nearest-neighbour well matching after image registration."""
    base = np.asarray(baseline_points, dtype=float)
    follow = np.asarray(followup_points, dtype=float)
    transformed = transform_points(base, matrix)
    pitch = median_well_pitch(follow)
    if not np.isfinite(pitch) or pitch <= 0:
        raise RuntimeError("Could not estimate follow-up microwell pitch.")
    max_distance = float(settings.max_match_pitch_fraction * pitch)

    tf_tree = cKDTree(follow)
    bf_dist, bf_idx = tf_tree.query(transformed, k=1)
    reverse_tree = cKDTree(transformed)
    _, reverse_idx = reverse_tree.query(follow, k=1)

    rows = []
    for i, (distance, j) in enumerate(zip(bf_dist, bf_idx)):
        j = int(j)
        mutual = int(reverse_idx[j]) == i
        accepted = bool(mutual and np.isfinite(distance) and distance <= max_distance)
        rows.append({
            "baseline_well_row": int(i),
            "followup_well_row": j,
            "baseline_x_px": float(base[i, 0]),
            "baseline_y_px": float(base[i, 1]),
            "transformed_baseline_x_px": float(transformed[i, 0]),
            "transformed_baseline_y_px": float(transformed[i, 1]),
            "followup_x_px": float(follow[j, 0]),
            "followup_y_px": float(follow[j, 1]),
            "match_error_px": float(distance),
            "match_error_pitch_fraction": float(distance / pitch),
            "mutual_nearest_neighbour": bool(mutual),
            "match_confident": accepted,
        })
    table = pd.DataFrame(rows)
    accepted = table[table["match_confident"]].copy()
    matched_n = int(len(accepted))
    base_fraction = float(matched_n / len(base)) if len(base) else 0.0
    follow_fraction = float(matched_n / len(follow)) if len(follow) else 0.0
    median_error = float(accepted["match_error_px"].median()) if matched_n else np.nan
    median_error_fraction = float(accepted["match_error_pitch_fraction"].median()) if matched_n else np.nan
    status = "accepted"
    if matched_n < settings.min_matched_wells or base_fraction < settings.min_match_fraction:
        status = "rejected_low_overlap_or_match_count"
    elif not np.isfinite(median_error_fraction) or median_error_fraction > settings.good_median_error_pitch_fraction:
        status = "rejected_high_registration_error"
    metrics = {
        "baseline_well_count": int(len(base)),
        "followup_well_count": int(len(follow)),
        "matched_well_count": matched_n,
        "match_fraction_baseline": base_fraction,
        "match_fraction_followup": follow_fraction,
        "median_well_pitch_px": float(pitch),
        "max_match_distance_px": max_distance,
        "median_match_error_px": median_error,
        "median_match_error_pitch_fraction": median_error_fraction,
        "registration_status": status,
    }
    return table, metrics


def register_field_pair(
    baseline_image: np.ndarray,
    followup_image: np.ndarray,
    baseline_wells: pd.DataFrame,
    followup_wells: pd.DataFrame,
    settings: RegistrationSettings = RegistrationSettings(),
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    matrix, feature_metrics = _feature_affine(baseline_image, followup_image, settings)
    base_xy = baseline_wells[["well_centre_x_px", "well_centre_y_px"]].to_numpy(float)
    follow_xy = followup_wells[["well_centre_x_px", "well_centre_y_px"]].to_numpy(float)
    matches, well_metrics = match_wells_one_to_one(base_xy, follow_xy, matrix, settings)
    metrics = {**feature_metrics, **well_metrics}
    return matches, metrics, matrix


def registration_overlay(
    followup_image: np.ndarray,
    matches: pd.DataFrame,
    matrix: np.ndarray,
) -> Image.Image:
    """QC overlay: follow-up wells, transformed baseline wells and match links."""
    arr = np.asarray(followup_image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for _, row in matches.iterrows():
        tx, ty = float(row["transformed_baseline_x_px"]), float(row["transformed_baseline_y_px"])
        mx, my = float(row["followup_x_px"]), float(row["followup_y_px"])
        confident = bool(row["match_confident"])
        # Use simple high-contrast fixed RGB annotation colours for QC assets.
        link = (255, 215, 0) if confident else (180, 180, 180)
        draw.line((tx, ty, mx, my), fill=link, width=1)
        draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), outline=(0, 220, 255), width=2)
        draw.ellipse((mx - 3, my - 3, mx + 3, my + 3), outline=(255, 80, 80), width=2)
    return img


def build_registered_longitudinal_tables(
    well_raw: pd.DataFrame,
    pdo_raw: pd.DataFrame,
    baseline_images: dict[int, np.ndarray],
    followup_images: dict[int, np.ndarray],
    output_dir: Path,
    elapsed_days: float,
    settings: RegistrationSettings = RegistrationSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Register all selected fields and return matched-well and growth tables.

    Only field pairs whose registration passes QC contribute to the biological
    growth table. Baseline-zero wells are retained in the matched-well table but
    have undefined fold/log growth because a relative growth rate has no finite
    baseline denominator.
    """
    output_dir = Path(output_dir)
    overlay_dir = output_dir / "registration_qc"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    w = well_raw.copy()
    p = pdo_raw.copy()
    for df in (w, p):
        if "image_series" in df.columns:
            df["image_series"] = pd.to_numeric(df["image_series"], errors="coerce").astype("Int64")
        if "timepoint_index" in df.columns:
            df["timepoint_index"] = pd.to_numeric(df["timepoint_index"], errors="coerce").astype("Int64")

    registration_rows = []
    matched_rows = []
    fields = sorted(set(baseline_images).intersection(followup_images))
    for series in fields:
        bw = w[(w["image_series"] == series) & (w["timepoint_index"] == 1)].reset_index(drop=False)
        fw = w[(w["image_series"] == series) & (w["timepoint_index"] == 2)].reset_index(drop=False)
        if bw.empty or fw.empty:
            registration_rows.append({
                "image_series": int(series),
                "registration_status": "rejected_missing_wells",
                "baseline_well_count": int(len(bw)),
                "followup_well_count": int(len(fw)),
            })
            continue
        try:
            matches, metrics, matrix = register_field_pair(
                baseline_images[series], followup_images[series], bw, fw, settings
            )
            registration_rows.append({"image_series": int(series), **metrics})
            overlay = registration_overlay(followup_images[series], matches, matrix)
            overlay.save(overlay_dir / f"field_{int(series):02d}_registration_overlay.png")
            if metrics["registration_status"] != "accepted":
                continue
            confident = matches[matches["match_confident"]].copy()
            for match_n, (_, row) in enumerate(confident.iterrows(), start=1):
                bi = int(row["baseline_well_row"])
                fi = int(row["followup_well_row"])
                b = bw.iloc[bi]
                f = fw.iloc[fi]
                physical_id = f"F{int(series):02d}__PHYSWELL{match_n:04d}"
                matched_rows.append({
                    "image_series": int(series),
                    "physical_trajectory_id": physical_id,
                    "baseline_well_index": str(b["well_index"]),
                    "followup_well_index": str(f["well_index"]),
                    "baseline_original_trajectory_id": b.get("trajectory_id", ""),
                    "followup_original_trajectory_id": f.get("trajectory_id", ""),
                    "baseline_well_observation_id": b.get("well_observation_id", ""),
                    "followup_well_observation_id": f.get("well_observation_id", ""),
                    "baseline_x_px": float(b["well_centre_x_px"]),
                    "baseline_y_px": float(b["well_centre_y_px"]),
                    "followup_x_px": float(f["well_centre_x_px"]),
                    "followup_y_px": float(f["well_centre_y_px"]),
                    "transformed_baseline_x_px": float(row["transformed_baseline_x_px"]),
                    "transformed_baseline_y_px": float(row["transformed_baseline_y_px"]),
                    "match_error_px": float(row["match_error_px"]),
                    "match_error_pitch_fraction": float(row["match_error_pitch_fraction"]),
                    "baseline_PDO_count": int(b.get("PDO_count", 0)),
                    "followup_PDO_count": int(f.get("PDO_count", 0)),
                    "baseline_PSC_like_focus_count": b.get("PSC_like_focus_count", np.nan),
                    "followup_PSC_like_focus_count": f.get("PSC_like_focus_count", np.nan),
                })
        except Exception as exc:
            registration_rows.append({
                "image_series": int(series),
                "registration_status": "rejected_registration_exception",
                "registration_error": str(exc),
                "baseline_well_count": int(len(bw)),
                "followup_well_count": int(len(fw)),
            })

    registration_df = pd.DataFrame(registration_rows)
    matched_df = pd.DataFrame(matched_rows)
    if matched_df.empty:
        return registration_df, matched_df, pd.DataFrame()

    area = p.copy()
    area["projected_area_um2"] = pd.to_numeric(area["projected_area_px2"], errors="coerce") * (
        pd.to_numeric(area["um_per_pixel"], errors="coerce") ** 2
    )
    by_obs = area.groupby("well_observation_id", as_index=True)["projected_area_um2"].sum()
    matched_df["baseline_total_PDO_area_um2"] = matched_df["baseline_well_observation_id"].map(by_obs).fillna(0.0)
    matched_df["followup_total_PDO_area_um2"] = matched_df["followup_well_observation_id"].map(by_obs).fillna(0.0)

    growth = matched_df.copy()
    a0 = pd.to_numeric(growth["baseline_total_PDO_area_um2"], errors="coerce")
    a1 = pd.to_numeric(growth["followup_total_PDO_area_um2"], errors="coerce")
    valid = (a0 > 0) & (a1 >= 0) & np.isfinite(float(elapsed_days)) & (float(elapsed_days) > 0)
    growth["fold_change_total_PDO_area"] = np.where(valid, a1 / a0, np.nan)
    growth["percent_change_total_PDO_area"] = np.where(valid, 100.0 * (a1 / a0 - 1.0), np.nan)
    positive = valid & (a1 > 0)
    growth["log_area_growth_rate_per_day"] = np.where(
        positive, np.log(a1 / a0) / float(elapsed_days), np.nan
    )
    growth["elapsed_days"] = float(elapsed_days)
    growth["growth_status"] = np.where(
        a0 <= 0,
        "undefined_no_baseline_PDO",
        np.where(a1 <= 0, "followup_zero_or_lost", "quantified"),
    )
    return registration_df, matched_df, growth
