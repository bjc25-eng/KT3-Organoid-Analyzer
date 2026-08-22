from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GRID_QC_SCHEMA_VERSION = 'nd2-global-grid-qc-v1'
GRID_NEIGHBOUR_K = 8
GRID_PITCH_TOLERANCE_FRACTION = 0.16
GRID_ANGLE_TOLERANCE_DEG = 12.0
GRID_MIN_SUPPORT = 2
GRID_MAX_REMOVAL_FRACTION = 0.20
GRID_MIN_WELLS = 12


def _angle_distance_pi(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % math.pi
    return min(d, math.pi - d)


def _estimate_pitch_and_orientations(wells: np.ndarray, settings):
    arr = np.asarray(wells, dtype=float).reshape((-1, 3))
    n = len(arr)
    if n < GRID_MIN_WELLS:
        return None, [], [], []

    pts = arr[:, :2]
    rref = 0.5 * (float(settings.well_rmin) + float(settings.well_rmax))
    dmin = max(1.30 * rref, 0.90 * float(settings.well_spacing))
    dmax = max(dmin + 10.0, 3.40 * rref)

    tree = cKDTree(pts)
    k = min(GRID_NEIGHBOUR_K + 1, n)
    distances, neighbours = tree.query(pts, k=k)

    seen = set()
    pair_rows = []
    for i in range(n):
        for rank in range(1, k):
            j = int(neighbours[i, rank])
            if j < 0 or j >= n or j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            dx = float(pts[j, 0] - pts[i, 0])
            dy = float(pts[j, 1] - pts[i, 1])
            dist = float(math.hypot(dx, dy))
            if dmin <= dist <= dmax:
                pair_rows.append((i, j, dx, dy, dist))

    if len(pair_rows) < max(12, n // 2):
        return None, [], pair_rows, []

    pair_distances = np.asarray([q[4] for q in pair_rows], dtype=float)
    bin_width = max(2.0, 0.045 * rref)
    bins = np.arange(dmin, dmax + bin_width, bin_width)
    if len(bins) < 3:
        return None, [], pair_rows, []
    hist, edges = np.histogram(pair_distances, bins=bins)
    peak_idx = int(np.argmax(hist))
    pitch0 = 0.5 * (float(edges[peak_idx]) + float(edges[peak_idx + 1]))
    refine_tol = max(2.0 * bin_width, 0.10 * pitch0)
    close = pair_distances[np.abs(pair_distances - pitch0) <= refine_tol]
    pitch = float(np.median(close)) if close.size else float(pitch0)

    pitch_tol = GRID_PITCH_TOLERANCE_FRACTION * pitch
    pitch_pairs = [q for q in pair_rows if abs(float(q[4]) - pitch) <= pitch_tol]
    if len(pitch_pairs) < max(10, n // 3):
        return pitch, [], pair_rows, pitch_pairs

    angles = np.asarray([(math.atan2(q[3], q[2]) % math.pi) for q in pitch_pairs], dtype=float)
    angle_bins = np.linspace(0.0, math.pi, 37)
    ahist, aedges = np.histogram(angles, bins=angle_bins)
    order = np.argsort(ahist)[::-1]
    chosen = []
    min_sep = math.radians(22.0)
    peak_floor = max(3, int(round(float(np.max(ahist)) * 0.20)))
    for idx in order:
        if int(ahist[idx]) < peak_floor:
            break
        centre = 0.5 * (float(aedges[idx]) + float(aedges[idx + 1]))
        if all(_angle_distance_pi(centre, old) >= min_sep for old in chosen):
            chosen.append(centre)
        if len(chosen) >= 3:
            break

    refined = []
    refine_angle = math.radians(10.0)
    for centre in chosen:
        nearby = np.asarray([a for a in angles if _angle_distance_pi(a, centre) <= refine_angle])
        if nearby.size:
            # Circular mean for an orientation, i.e. angle modulo pi.
            c = float(np.mean(np.cos(2.0 * nearby)))
            s = float(np.mean(np.sin(2.0 * nearby)))
            refined_angle = 0.5 * math.atan2(s, c)
            if refined_angle < 0:
                refined_angle += math.pi
            refined.append(refined_angle)
        else:
            refined.append(centre)

    return pitch, refined, pair_rows, pitch_pairs


def grid_filter_mask(wells: np.ndarray, settings):
    """Return a conservative whole-array grid-consistency mask.

    The filter learns the dominant nearest-neighbour pitch and orientations from
    the detected well centres themselves. A well is retained when it participates
    in at least two neighbour relationships consistent with the dominant array
    geometry. The cleanup is abandoned rather than over-pruning if it would
    remove more than 20% of candidate wells.
    """
    arr = np.asarray(wells, dtype=float).reshape((-1, 3))
    n = len(arr)
    mask = np.ones(n, dtype=bool)
    support = np.zeros(n, dtype=int)
    qc = {
        'schema_version': GRID_QC_SCHEMA_VERSION,
        'candidate_wells': int(n),
        'retained_wells': int(n),
        'rejected_wells': 0,
        'applied': False,
        'reason': '',
        'estimated_pitch_px': None,
        'dominant_orientations_deg': [],
        'minimum_grid_support': GRID_MIN_SUPPORT,
        'max_removal_fraction': GRID_MAX_REMOVAL_FRACTION,
    }

    if n < GRID_MIN_WELLS:
        qc['reason'] = 'too_few_wells_for_global_grid_fit'
        return mask, support, qc

    pitch, orientations, pair_rows, pitch_pairs = _estimate_pitch_and_orientations(arr, settings)
    if pitch is None or len(orientations) < 2:
        qc['reason'] = 'could_not_resolve_two_dominant_grid_directions'
        qc['estimated_pitch_px'] = None if pitch is None else float(pitch)
        return mask, support, qc

    pitch_tol = GRID_PITCH_TOLERANCE_FRACTION * float(pitch)
    angle_tol = math.radians(GRID_ANGLE_TOLERANCE_DEG)
    for i, j, dx, dy, dist in pitch_pairs:
        if abs(float(dist) - float(pitch)) > pitch_tol:
            continue
        angle = math.atan2(float(dy), float(dx)) % math.pi
        if any(_angle_distance_pi(angle, ori) <= angle_tol for ori in orientations):
            support[int(i)] += 1
            support[int(j)] += 1

    proposed = support >= GRID_MIN_SUPPORT
    retained = int(np.sum(proposed))
    rejected = int(n - retained)
    removal_fraction = float(rejected / max(1, n))

    qc.update({
        'estimated_pitch_px': float(pitch),
        'dominant_orientations_deg': [float(math.degrees(q)) for q in orientations],
        'retained_wells': retained,
        'rejected_wells': rejected,
        'removal_fraction': removal_fraction,
        'median_grid_support': float(np.median(support)) if n else 0.0,
        'max_grid_support': int(np.max(support)) if n else 0,
    })

    if retained < max(6, int(round(0.70 * n))):
        qc['reason'] = 'grid_fit_would_remove_too_many_candidates'
        qc['retained_wells'] = int(n)
        qc['rejected_wells'] = 0
        return mask, support, qc
    if removal_fraction > GRID_MAX_REMOVAL_FRACTION:
        qc['reason'] = 'grid_fit_exceeded_conservative_removal_limit'
        qc['retained_wells'] = int(n)
        qc['rejected_wells'] = 0
        return mask, support, qc

    mask = proposed
    qc['applied'] = True
    qc['reason'] = 'global_grid_consistency_filter_applied'
    return mask, support, qc


def _filter_by_well_ids(df: pd.DataFrame, keep_ids: set[str]) -> pd.DataFrame:
    if df is None or df.empty or 'well_observation_id' not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    return df[df['well_observation_id'].astype(str).isin(keep_ids)].copy().reset_index(drop=True)


def _filter_tracking(df: pd.DataFrame, keep_ids: set[str], keep_trajectories: set[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if 'well_observation_id' in df.columns:
        return df[df['well_observation_id'].astype(str).isin(keep_ids)].copy().reset_index(drop=True)
    if 'trajectory_id' in df.columns:
        return df[df['trajectory_id'].astype(str).isin(keep_trajectories)].copy().reset_index(drop=True)
    return df.copy()


def apply_global_grid_qc_to_result(result, settings):
    """Apply whole-array grid cleanup and rewrite exported result tables.

    This is deliberately a final geometry QC layer. The successful 2048-pixel DIC
    detector remains unchanged; only isolated centres inconsistent with the global
    microwell array are removed from the scientific result tables and downloads.
    """
    root, out, manifest, wdf, pdf, pscdf, tracking, run_status, ml_path = result
    out = Path(out)
    root = Path(root)

    if wdf is None or wdf.empty:
        return result

    audit_parts = []
    keep_ids: set[str] = set()
    qc_by_source = {}
    group_col = 'source_uid' if 'source_uid' in wdf.columns else None
    groups = wdf.groupby(group_col, sort=False) if group_col else [('all', wdf)]

    for source_key, group in groups:
        coords = group[['well_centre_x_px_fullres', 'well_centre_y_px_fullres', 'well_radius_px']].to_numpy(dtype=float)
        mask, support, qc = grid_filter_mask(coords, settings)
        qc_by_source[str(source_key)] = qc
        ids = group['well_observation_id'].astype(str).to_numpy()
        keep_ids.update(ids[mask].tolist())

        audit = group[['well_observation_id', 'well_centre_x_px_fullres', 'well_centre_y_px_fullres', 'well_radius_px']].copy()
        audit['source_uid'] = str(source_key)
        audit['global_grid_support'] = support
        audit['global_grid_retained'] = mask
        audit['global_grid_pitch_px'] = qc.get('estimated_pitch_px')
        audit['global_grid_qc_applied'] = bool(qc.get('applied', False))
        audit['global_grid_qc_reason'] = str(qc.get('reason', ''))
        audit_parts.append(audit)

    audit_df = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    wdf2 = _filter_by_well_ids(wdf, keep_ids)
    pdf2 = _filter_by_well_ids(pdf, keep_ids)
    pscdf2 = _filter_by_well_ids(pscdf, keep_ids)
    keep_trajectories = set(wdf2['trajectory_id'].astype(str)) if 'trajectory_id' in wdf2.columns else set()
    tracking2 = _filter_tracking(tracking, keep_ids, keep_trajectories)

    csv_dir = out / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    wdf2.to_csv(csv_dir / 'large_well_observations.csv', index=False)
    pdf2.to_csv(csv_dir / 'large_PDO_observations.csv', index=False)
    pscdf2.to_csv(csv_dir / 'large_PSC_focus_observations.csv', index=False)
    tracking2.to_csv(csv_dir / 'well_longitudinal_tracking.csv', index=False)
    audit_df.to_csv(csv_dir / 'global_microwell_grid_QC.csv', index=False)

    candidate_qc_path = csv_dir / 'PDO_candidate_QC.csv'
    if candidate_qc_path.exists():
        try:
            qdf = pd.read_csv(candidate_qc_path)
            _filter_by_well_ids(qdf, keep_ids).to_csv(candidate_qc_path, index=False)
        except Exception:
            pass

    summary_path = csv_dir / 'ND2_QC_summary.csv'
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            if len(summary):
                summary.loc[:, 'fully_visible_wells'] = int(len(wdf2))
                summary.loc[:, 'PDO_containing_wells'] = int((pd.to_numeric(wdf2.get('PDO_count', 0), errors='coerce').fillna(0) > 0).sum()) if not wdf2.empty else 0
                summary.loc[:, 'PDO_count'] = int(len(pdf2))
                summary.loc[:, 'global_grid_rejected_wells'] = int(len(wdf) - len(wdf2))
                summary.loc[:, 'global_grid_qc_schema_version'] = GRID_QC_SCHEMA_VERSION
                summary.to_csv(summary_path, index=False)
        except Exception:
            pass

    if isinstance(run_status, dict):
        run_status = dict(run_status)
        run_status['global_grid_qc_schema_version'] = GRID_QC_SCHEMA_VERSION
        run_status['global_grid_qc_by_source'] = qc_by_source
        run_status['global_grid_candidates'] = int(len(wdf))
        run_status['global_grid_retained'] = int(len(wdf2))
        run_status['global_grid_rejected'] = int(len(wdf) - len(wdf2))
        try:
            (out / 'run_status.json').write_text(json.dumps(run_status, indent=2, default=str), encoding='utf-8')
        except Exception:
            pass

    config_path = root / 'run_configuration.json'
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding='utf-8'))
            final_qc = cfg.setdefault('nd2_final_qc', {})
            final_qc['global_grid_qc_schema_version'] = GRID_QC_SCHEMA_VERSION
            final_qc['global_grid_qc'] = qc_by_source
            config_path.write_text(json.dumps(cfg, indent=2, default=str), encoding='utf-8')
        except Exception:
            pass

    if ml_path is not None:
        tables = Path(ml_path) / 'tables'
        tables.mkdir(parents=True, exist_ok=True)
        wdf2.to_csv(tables / 'well_observations.csv', index=False)
        pdf2.to_csv(tables / 'pdo_observations.csv', index=False)
        pscdf2.to_csv(tables / 'psc_focus_observations.csv', index=False)
        tracking2.to_csv(tables / 'longitudinal_trajectories.csv', index=False)
        audit_df.to_csv(tables / 'global_microwell_grid_QC.csv', index=False)
        qc_cols = [c for c in wdf2.columns if c.startswith('qc_') or c in {'trajectory_id', 'well_observation_id', 'source_uid'}]
        if qc_cols:
            wdf2[qc_cols].to_csv(tables / 'qc_flags.csv', index=False)
        candidate_ml = tables / 'PDO_candidate_QC.csv'
        if candidate_ml.exists():
            try:
                qdf = pd.read_csv(candidate_ml)
                _filter_by_well_ids(qdf, keep_ids).to_csv(candidate_ml, index=False)
            except Exception:
                pass

    return root, out, manifest, wdf2, pdf2, pscdf2, tracking2, run_status, ml_path
