from __future__ import annotations

import numpy as np
import pandas as pd


def build_registered_tracking(matched: pd.DataFrame, elapsed_days: float,
                              baseline_label: str, followup_label: str,
                              condition: str) -> pd.DataFrame:
    """Return two rows per confidently matched physical microwell for analysis."""
    if matched is None or matched.empty:
        return pd.DataFrame()
    rows = []
    for _, r in matched.iterrows():
        tid = str(r["physical_trajectory_id"])
        base_area = float(r.get("baseline_total_PDO_area_um2", 0.0))
        follow_area = float(r.get("followup_total_PDO_area_um2", 0.0))
        base_psc = r.get("baseline_PSC_like_focus_count", np.nan)
        follow_psc = r.get("followup_PSC_like_focus_count", np.nan)
        base_index = str(r.get("baseline_well_index", ""))
        follow_index = str(r.get("followup_well_index", ""))
        series = int(r.get("image_series", 0))
        common = {
            "condition_index": 1,
            "condition": condition,
            "image_series": series,
            "trajectory_id": tid,
            "physical_trajectory_id": tid,
            "registration_match_error_px": float(r.get("match_error_px", np.nan)),
            "registration_match_error_pitch_fraction": float(r.get("match_error_pitch_fraction", np.nan)),
            "baseline_total_PDO_area_um2": base_area,
        }
        rows.append({
            **common,
            "timepoint_index": 1,
            "timepoint": baseline_label,
            "elapsed_time": 0.0,
            "well_index": base_index,
            "local_well_index": base_index,
            "well_observation_id": r.get("baseline_well_observation_id", ""),
            "PDO_count": int(r.get("baseline_PDO_count", 0)),
            "PSC_like_focus_count": base_psc,
            "total_PDO_projected_area_um2": base_area,
            "relative_total_PDO_area_vs_baseline": 1.0 if base_area > 0 else np.nan,
        })
        rows.append({
            **common,
            "timepoint_index": 2,
            "timepoint": followup_label,
            "elapsed_time": float(elapsed_days),
            "well_index": follow_index,
            "local_well_index": follow_index,
            "well_observation_id": r.get("followup_well_observation_id", ""),
            "PDO_count": int(r.get("followup_PDO_count", 0)),
            "PSC_like_focus_count": follow_psc,
            "total_PDO_projected_area_um2": follow_area,
            "relative_total_PDO_area_vs_baseline": (follow_area / base_area) if base_area > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["image_series", "physical_trajectory_id", "timepoint_index"]).reset_index(drop=True)
