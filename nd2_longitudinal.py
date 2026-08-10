from __future__ import annotations

"""Paired native-ND2 adapter for the longitudinal experiment workflow.

The adapter materialises only the selected XY positions from each ND2 source,
constructs an analysis RGB composite in which DIC is the common baseline and
GFP/RFP are added to green/red respectively, and then delegates tracking and
longitudinal aggregation to the existing experiment engine.
"""

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analysis_core import GFP_MODE, process_experiment
from nd2_large_source import ND2LargeImageReader, ND2_SOURCE_LABEL, probe_nd2_source


def _balanced_rgb(dic: np.ndarray, gfp: np.ndarray, rfp: np.ndarray | None) -> np.ndarray:
    """Create RGB preserving DIC structure and fluorescence contrast.

    DIC contributes equally to R/G/B. GFP is added only to G and RFP only to R,
    so existing green-excess and red-minus-blue logic remains meaningful while
    well detection still sees the DIC structure.
    """
    dic = np.asarray(dic, dtype=np.uint8)
    gfp = np.asarray(gfp, dtype=np.uint8)
    rfp = np.zeros_like(dic) if rfp is None else np.asarray(rfp, dtype=np.uint8)
    base = dic.astype(np.uint16)
    rgb = np.stack([
        np.clip(base + rfp.astype(np.uint16), 0, 255),
        np.clip(base + gfp.astype(np.uint16), 0, 255),
        base,
    ], axis=-1).astype(np.uint8)
    return rgb


def _xy_um_per_pixel(meta: dict) -> float:
    voxel = meta.get("voxel_size_um") or {}
    vals = []
    for key in ("x", "y"):
        try:
            value = float(voxel.get(key))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0:
            vals.append(value)
    if len(vals) != 2:
        raise ValueError("ND2 metadata does not contain valid X/Y physical pixel size.")
    return float(np.mean(vals))


def _materialise_position(
    uri: str,
    position_index: int,
    out_path: Path,
    gfp_channel: int,
    dic_channel: int,
    rfp_channel: int | None,
) -> dict:
    with ND2LargeImageReader(
        uri,
        source_type=ND2_SOURCE_LABEL,
        series_index=int(position_index),
    ) as reader:
        gfp = reader.read_channel_region(0, 0, reader.width, reader.height, int(gfp_channel))
        dic = reader.read_channel_region(0, 0, reader.width, reader.height, int(dic_channel))
        rfp = None
        if rfp_channel is not None and int(rfp_channel) >= 0:
            rfp = reader.read_channel_region(0, 0, reader.width, reader.height, int(rfp_channel))
        rgb = _balanced_rgb(dic, gfp, rfp)
        Image.fromarray(rgb).save(out_path)
        return {
            "voxel_size_um": reader.voxel_size_um,
            "width_px": reader.width,
            "height_px": reader.height,
            "position_index": int(position_index),
        }


class _UploadLike:
    def __init__(self, path: Path, name: str):
        self._path = Path(path)
        self.name = str(name)

    def getbuffer(self):
        return memoryview(self._path.read_bytes())


def probe_pair(friday_uri: str, monday_uri: str) -> dict:
    a = probe_nd2_source(friday_uri, 0)
    b = probe_nd2_source(monday_uri, 0)
    return {
        "friday": a,
        "monday": b,
        "matching_position_count": min(int(a.get("position_count", 1)), int(b.get("position_count", 1))),
        "same_position_count": int(a.get("position_count", 1)) == int(b.get("position_count", 1)),
        "same_dimensions": (
            int(a.get("width_px", -1)) == int(b.get("width_px", -2))
            and int(a.get("height_px", -1)) == int(b.get("height_px", -2))
        ),
        "friday_um_per_pixel": _xy_um_per_pixel(a),
        "monday_um_per_pixel": _xy_um_per_pixel(b),
    }


def _recalibrate_tracking(ldf: pd.DataFrame, scales_by_timepoint: dict[int, float]) -> pd.DataFrame:
    """Replace Hough-derived physical scale with native ND2 calibration."""
    out = ldf.copy()
    if out.empty or "timepoint_index" not in out.columns:
        return out
    target_scale = out["timepoint_index"].map(scales_by_timepoint).astype(float)
    old_scale = pd.to_numeric(out.get("um_per_pixel"), errors="coerce")
    valid = target_scale.notna() & (target_scale > 0) & old_scale.notna() & (old_scale > 0)
    if "total_PDO_projected_area_um2" in out.columns:
        px_area = np.where(valid, pd.to_numeric(out["total_PDO_projected_area_um2"], errors="coerce") / (old_scale ** 2), np.nan)
        out.loc[valid, "total_PDO_projected_area_um2"] = px_area[valid] * (target_scale[valid] ** 2)
    for col in ("mean_PDO_diameter_um", "max_PDO_diameter_um"):
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            out.loc[valid, col] = vals[valid] * target_scale[valid] / old_scale[valid]
    out.loc[valid, "um_per_pixel"] = target_scale[valid]

    if "trajectory_id" in out.columns and "total_PDO_projected_area_um2" in out.columns:
        baseline = (
            out.sort_values("timepoint_index")
            .groupby("trajectory_id", as_index=False)
            .first()[["trajectory_id", "total_PDO_projected_area_um2"]]
            .rename(columns={"total_PDO_projected_area_um2": "baseline_total_PDO_area_um2"})
        )
        out = out.drop(columns=["baseline_total_PDO_area_um2", "relative_total_PDO_area_vs_baseline"], errors="ignore")
        out = out.merge(baseline, on="trajectory_id", how="left", validate="many_to_one")
        base = pd.to_numeric(out["baseline_total_PDO_area_um2"], errors="coerce")
        cur = pd.to_numeric(out["total_PDO_projected_area_um2"], errors="coerce")
        out["relative_total_PDO_area_vs_baseline"] = np.where(base > 0, cur / base, np.nan)
    return out


def process_paired_nd2_longitudinal(
    friday_uri: str,
    monday_uri: str,
    position_indices: list[int],
    settings,
    cols: int,
    experiment_metadata: dict,
    condition_name: str,
    friday_label: str,
    monday_label: str,
    elapsed_days: float,
    gfp_channel: int,
    dic_channel: int,
    rfp_channel: int | None,
    make_ml_export: bool = True,
):
    if not position_indices:
        raise ValueError("Select at least one matched XY position.")
    temp = Path(tempfile.mkdtemp(prefix="kt3_nd2_longitudinal_"))
    friday_files, monday_files = [], []
    friday_scales, monday_scales = [], []

    for pos in position_indices:
        series = int(pos) + 1
        fpath = temp / f"Friday_series_{series:02d}.png"
        mpath = temp / f"Monday_series_{series:02d}.png"
        fm = _materialise_position(friday_uri, pos, fpath, gfp_channel, dic_channel, rfp_channel)
        mm = _materialise_position(monday_uri, pos, mpath, gfp_channel, dic_channel, rfp_channel)
        friday_scales.append(_xy_um_per_pixel(fm))
        monday_scales.append(_xy_um_per_pixel(mm))
        friday_files.append(_UploadLike(fpath, f"KT3_PSC_series_{series:02d}.png"))
        monday_files.append(_UploadLike(mpath, f"KT3_PSC_series_{series:02d}.png"))

    friday_umpp = float(np.mean(friday_scales))
    monday_umpp = float(np.mean(monday_scales))
    expected_umpp = float(np.mean([friday_umpp, monday_umpp]))

    local_settings = replace(settings, organoid_mode=GFP_MODE, rfp_psc_present=rfp_channel is not None and int(rfp_channel) >= 0)
    expected_r = float(local_settings.well_diameter_um) / (2.0 * expected_umpp)
    local_settings.well_rmin = max(5, int(round(0.80 * expected_r)))
    local_settings.well_rmax = max(local_settings.well_rmin + 1, int(round(1.20 * expected_r)))
    local_settings.well_spacing = max(10, int(round(1.50 * expected_r)))

    condition = {
        "condition_index": 1,
        "condition": condition_name,
        "organoid_mode": GFP_MODE,
        "rfp_psc_present": local_settings.rfp_psc_present,
        "drug_or_therapeutic": "",
        "concentration": 0.0,
        "concentration_unit": "",
    }
    entries = [
        {
            **condition,
            "timepoint_index": 1,
            "timepoint": friday_label,
            "elapsed_time": 0.0,
            "time_unit": "days",
            "files": friday_files,
        },
        {
            **condition,
            "timepoint_index": 2,
            "timepoint": monday_label,
            "elapsed_time": float(elapsed_days),
            "time_unit": "days",
            "files": monday_files,
        },
    ]
    root, out, summary, tracking, ml_path = process_experiment(
        entries, local_settings, int(cols), experiment_metadata, make_ml_export=make_ml_export
    )
    tracking = _recalibrate_tracking(tracking, {1: friday_umpp, 2: monday_umpp})
    tracking.to_csv(out / "csv" / "well_longitudinal_tracking.csv", index=False)
    pd.DataFrame([
        {
            "baseline_um_per_pixel": friday_umpp,
            "followup_um_per_pixel": monday_umpp,
            "physical_scale_source": "native ND2 X/Y voxel_size_um metadata",
            "growth_measurement": "total PDO projected area per trajectory",
        }
    ]).to_csv(out / "csv" / "ND2_physical_calibration.csv", index=False)
    if ml_path is not None:
        tables = Path(ml_path) / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        tracking.to_csv(tables / "longitudinal_trajectories.csv", index=False)
    return root, out, summary, tracking, ml_path
