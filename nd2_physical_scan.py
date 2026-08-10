from __future__ import annotations

"""Physical-calibration wrapper for native ND2 tiled microwell detection."""

import copy
import hashlib

import numpy as np

import large_data_core as ldc

_ORIGINAL_SCAN = None


def physical_scan_settings(reader, settings):
    """Clone Settings with the validated physical-radius Hough geometry.

    The factors match the existing whole-array benchmark:
    - minimum Hough radius = 0.80 × expected physical radius
    - maximum Hough radius = 1.20 × expected physical radius
    - minimum centre spacing = 1.50 × expected physical radius
    """
    metadata = reader.metadata()
    voxel = metadata.get("voxel_size_um") or {}
    vals = []
    for key in ("x", "y"):
        try:
            value = float(voxel.get(key))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0:
            vals.append(value)
    if len(vals) != 2:
        raise ldc.LargeSourceError(
            "ND2 metadata does not contain valid X/Y physical pixel size; "
            "physical-radius microwell detection cannot be configured."
        )

    um_per_pixel = float(np.mean(vals))
    expected_radius = float(settings.well_diameter_um) / (2.0 * um_per_pixel)
    derived = copy.copy(settings)
    derived.well_rmin = max(2, int(round(expected_radius * 0.80)))
    derived.well_rmax = max(derived.well_rmin + 2, int(round(expected_radius * 1.20)))
    derived.well_spacing = max(2, int(round(expected_radius * 1.50)))
    return derived, expected_radius, um_per_pixel


def install_nd2_physical_well_scan():
    """Install an idempotent wrapper around the existing resumable tile scan."""
    global _ORIGINAL_SCAN
    if getattr(ldc.scan_wells_tiled, "_nd2_physical_scan_wrapper", False):
        return
    _ORIGINAL_SCAN = ldc.scan_wells_tiled

    def wrapped(reader, settings, config, work_dir, source_fingerprint,
                organoid_mode, tile_size=ldc.DEFAULT_TILE_SIZE,
                progress_callback=None):
        # Only alter native Nikon ND2 sources. TIFF/Zarr behaviour is unchanged.
        if str(getattr(reader, "format", "")) != "Nikon ND2":
            return _ORIGINAL_SCAN(
                reader, settings, config, work_dir, source_fingerprint,
                organoid_mode, tile_size, progress_callback,
            )

        scan_settings, expected_radius, umpp = physical_scan_settings(reader, settings)
        signature = (
            f"nd2-physical-scan-v1|umpp={umpp:.12g}|radius={expected_radius:.12g}|"
            f"rmin={scan_settings.well_rmin}|rmax={scan_settings.well_rmax}|"
            f"spacing={scan_settings.well_spacing}|p2={float(scan_settings.hough_p2):.12g}"
        )
        scan_fingerprint = hashlib.sha256(
            (str(source_fingerprint) + "|" + signature).encode("utf-8")
        ).hexdigest()
        return _ORIGINAL_SCAN(
            reader, scan_settings, config, work_dir, scan_fingerprint,
            organoid_mode, tile_size, progress_callback,
        )

    wrapped._nd2_physical_scan_wrapper = True
    ldc.scan_wells_tiled = wrapped
