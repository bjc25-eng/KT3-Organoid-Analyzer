from __future__ import annotations

"""Paired native-ND2 adapter with automatic DIC registration.

Selected XY positions are materialised from each ND2 source, analysed independently,
then registered from DIC image content. Detected microwells are matched one-to-one
after registration so longitudinal growth is calculated from the same physical
microwells even when the Friday and Monday crops differ.
"""

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analysis_core import GFP_MODE, process_experiment
from nd2_large_source import ND2LargeImageReader, ND2_SOURCE_LABEL, probe_nd2_source
from paired_registration import build_registered_longitudinal_tables
from registered_tracking import build_registered_tracking


def _balanced_rgb(dic: np.ndarray, gfp: np.ndarray, rfp: np.ndarray | None) -> np.ndarray:
    dic = np.asarray(dic, dtype=np.uint8)
    gfp = np.asarray(gfp, dtype=np.uint8)
    rfp = np.zeros_like(dic) if rfp is None else np.asarray(rfp, dtype=np.uint8)
    base = dic.astype(np.uint16)
    return np.stack([
        np.clip(base + rfp.astype(np.uint16), 0, 255),
        np.clip(base + gfp.astype(np.uint16), 0, 255),
        base,
    ], axis=-1).astype(np.uint8)


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


def _materialise_position(uri: str, position_index: int, out_path: Path,
                          gfp_channel: int, dic_channel: int,
                          rfp_channel: int | None) -> dict:
    with ND2LargeImageReader(
        uri, source_type=ND2_SOURCE_LABEL, series_index=int(position_index)
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


def _apply_native_scale(df: pd.DataFrame, scales_by_timepoint: dict[int, float]) -> pd.DataFrame:
    out = df.copy()
    if out.empty or "timepoint_index" not in out.columns:
        return out
    tp = pd.to_numeric(out["timepoint_index"], errors="coerce")
    target = tp.map(scales_by_timepoint)
    old = pd.to_numeric(out.get("um_per_pixel"), errors="coerce") if "um_per_pixel" in out.columns else pd.Series(np.nan, index=out.index)
    if "equivalent_circular_diameter_um" in out.columns:
        vals = pd.to_numeric(out["equivalent_circular_diameter_um"], errors="coerce")
        valid = target.notna() & old.notna() & (old > 0)
        out.loc[valid, "equivalent_circular_diameter_um"] = vals[valid] * target[valid] / old[valid]
    out["um_per_pixel"] = target
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
    baseline_images: dict[int, np.ndarray] = {}
    followup_images: dict[int, np.ndarray] = {}

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
        baseline_images[series] = np.asarray(Image.open(fpath).convert("RGB"))
        followup_images[series] = np.asarray(Image.open(mpath).convert("RGB"))

    friday_umpp = float(np.mean(friday_scales))
    monday_umpp = float(np.mean(monday_scales))
    expected_umpp = float(np.mean([friday_umpp, monday_umpp]))

    local_settings = replace(
        settings,
        organoid_mode=GFP_MODE,
        rfp_psc_present=rfp_channel is not None and int(rfp_channel) >= 0,
    )
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
        {**condition, "timepoint_index": 1, "timepoint": friday_label,
         "elapsed_time": 0.0, "time_unit": "days", "files": friday_files},
        {**condition, "timepoint_index": 2, "timepoint": monday_label,
         "elapsed_time": float(elapsed_days), "time_unit": "days", "files": monday_files},
    ]

    root, out, summary, _, ml_path = process_experiment(
        entries, local_settings, int(cols), experiment_metadata,
        make_ml_export=make_ml_export,
    )

    csv_dir = Path(out) / "csv"
    well_raw = pd.read_csv(csv_dir / "longitudinal_well_raw_data.csv")
    pdo_raw = pd.read_csv(csv_dir / "longitudinal_PDO_raw_data.csv")
    scales = {1: friday_umpp, 2: monday_umpp}
    well_raw = _apply_native_scale(well_raw, scales)
    pdo_raw = _apply_native_scale(pdo_raw, scales)
    well_raw.to_csv(csv_dir / "longitudinal_well_raw_data.csv", index=False)
    pdo_raw.to_csv(csv_dir / "longitudinal_PDO_raw_data.csv", index=False)

    registration_df, matched_df, growth_df = build_registered_longitudinal_tables(
        well_raw,
        pdo_raw,
        baseline_images,
        followup_images,
        Path(out),
        float(elapsed_days),
    )
    registration_df.to_csv(csv_dir / "field_registration_summary.csv", index=False)
    matched_df.to_csv(csv_dir / "matched_physical_wells.csv", index=False)
    growth_df.to_csv(csv_dir / "matched_well_growth.csv", index=False)

    if matched_df.empty:
        accepted = int((registration_df.get("registration_status", pd.Series(dtype=str)) == "accepted").sum())
        raise RuntimeError(
            "No confidently matched physical microwells were produced. "
            f"Accepted registered field pairs: {accepted}. Review field_registration_summary.csv."
        )

    tracking = build_registered_tracking(
        matched_df,
        float(elapsed_days),
        friday_label,
        monday_label,
        condition_name,
    )
    tracking.to_csv(csv_dir / "well_longitudinal_tracking.csv", index=False)

    pd.DataFrame([{
        "baseline_um_per_pixel": friday_umpp,
        "followup_um_per_pixel": monday_umpp,
        "physical_scale_source": "native ND2 X/Y voxel_size_um metadata",
        "registration_source": "DIC image features + RANSAC affine + mutual nearest detected wells",
        "growth_measurement": "total PDO projected area per confidently matched physical microwell",
        "registered_fields_accepted": int((registration_df["registration_status"] == "accepted").sum()) if "registration_status" in registration_df.columns else 0,
        "matched_physical_wells": int(len(matched_df)),
    }]).to_csv(csv_dir / "ND2_physical_calibration_and_registration.csv", index=False)

    if ml_path is not None:
        tables = Path(ml_path) / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        tracking.to_csv(tables / "longitudinal_trajectories.csv", index=False)
        registration_df.to_csv(tables / "field_registration_summary.csv", index=False)
        matched_df.to_csv(tables / "matched_physical_wells.csv", index=False)
        growth_df.to_csv(tables / "matched_well_growth.csv", index=False)

    return root, out, summary, tracking, ml_path
