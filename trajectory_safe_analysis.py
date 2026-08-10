from __future__ import annotations

"""Trajectory-safe wrapper around advanced longitudinal analysis.

The legacy advanced-analysis module groups by condition + well_index. That is
ambiguous when several microscope fields each contain the same local well index.
For longitudinal datasets carrying trajectory_id, this wrapper substitutes the
stable trajectory ID as the analysis well key while preserving the original
local well label in `local_well_index`.
"""

import pandas as pd

from advanced_analysis import run_selected_analyses as _run_selected_analyses


def make_trajectory_safe(ldf: pd.DataFrame) -> pd.DataFrame:
    out = ldf.copy()
    if "trajectory_id" in out.columns:
        ids = out["trajectory_id"].astype(str)
        if ids.eq("").any() or ids.isna().any():
            raise ValueError("trajectory_id contains missing values; cannot safely pair fields across time.")
        if "well_index" in out.columns and "local_well_index" not in out.columns:
            out["local_well_index"] = out["well_index"]
        out["well_index"] = ids
    return out


def run_selected_analyses(*args, **kwargs):
    if not args:
        raise TypeError("A longitudinal DataFrame is required.")
    safe = make_trajectory_safe(args[0])
    return _run_selected_analyses(safe, *args[1:], **kwargs)
