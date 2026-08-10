from __future__ import annotations

"""Paired native-ND2 adapter for the longitudinal experiment workflow.

The adapter materialises only the selected XY positions from each ND2 source,
constructs an analysis RGB composite in which DIC is the common baseline and
GFP/RFP are added to green/red respectively, and then delegates tracking and
longitudinal aggregation to the existing experiment engine.
"""

import io
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
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
    }


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
    pixel_sizes = []

    for pos in position_indices:
        series = int(pos) + 1
        fpath = temp / f"Friday_series_{series:02d}.png"
        mpath = temp / f"Monday_series_{series:02d}.png"
        fm = _materialise_position(friday_uri, pos, fpath, gfp_channel, dic_channel, rfp_channel)
        mm = _materialise_position(monday_uri, pos, mpath, gfp_channel, dic_channel, rfp_channel)
        for meta in (fm, mm):
            vx = float((meta.get("voxel_size_um") or {}).get("x") or np.nan)
            vy = float((meta.get("voxel_size_um") or {}).get("y") or np.nan)
            if np.isfinite(vx) and np.isfinite(vy) and vx > 0 and vy > 0:
                pixel_sizes.extend([vx, vy])
        friday_files.append(_UploadLike(fpath, f"KT3_PSC_series_{series:02d}.png"))
        monday_files.append(_UploadLike(mpath, f"KT3_PSC_series_{series:02d}.png"))

    local_settings = replace(settings, organoid_mode=GFP_MODE, rfp_psc_present=rfp_channel is not None and int(rfp_channel) >= 0)
    if pixel_sizes:
        umpp = float(np.mean(pixel_sizes))
        expected_r = float(local_settings.well_diameter_um) / (2.0 * umpp)
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
    return process_experiment(entries, local_settings, int(cols), experiment_metadata, make_ml_export=make_ml_export)
