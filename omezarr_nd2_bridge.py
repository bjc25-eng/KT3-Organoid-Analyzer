from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np

import large_data_core as ldc
from nd2_qc import process_large_experiment_qc

_ORIGINAL_METADATA = None
_ORIGINAL_READ_CHANNEL = None
_ORIGINAL_DETECT_WELLS = None
SCALING_SCHEMA_VERSION = 'nd2-omezarr-window-scaling-v2'
DETECTOR_SCHEMA_VERSION = 'nd2-omezarr-local-contrast-hough-v1'


def _unit_to_um(value: float, unit: str | None) -> float | None:
    unit_norm = str(unit or '').strip().lower().replace('µ', 'u').replace('μ', 'u')
    factors = {
        'um': 1.0,
        'micrometer': 1.0,
        'micrometre': 1.0,
        'nm': 1e-3,
        'nanometer': 1e-3,
        'nanometre': 1e-3,
        'mm': 1e3,
        'millimeter': 1e3,
        'millimetre': 1e3,
        'm': 1e6,
        'meter': 1e6,
        'metre': 1e6,
    }
    factor = factors.get(unit_norm)
    return None if factor is None else float(value) * factor


def _ome_channels(reader) -> list[dict]:
    root = getattr(reader, '_root', None)
    if root is None:
        return []
    try:
        attrs = dict(root.attrs)
        channels = (attrs.get('omero') or {}).get('channels') or []
        return [dict(row or {}) for row in channels]
    except Exception:
        return []


def _display_window(reader, channel: int) -> tuple[float, float] | None:
    channels = _ome_channels(reader)
    idx = int(channel)
    if idx < 0 or idx >= len(channels):
        return None
    window = channels[idx].get('window') or {}
    if not isinstance(window, dict):
        return None

    low = window.get('start', window.get('min'))
    high = window.get('end', window.get('max'))
    try:
        low_f = float(low)
        high_f = float(high)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(low_f) or not np.isfinite(high_f) or high_f <= low_f:
        return None
    return low_f, high_f


def install_omezarr_physical_metadata() -> None:
    """Expose OME-NGFF calibration/channel metadata to the final-QC engine."""
    global _ORIGINAL_METADATA
    if getattr(ldc.LargeImageReader.metadata, '_omezarr_physical_metadata_wrapper', False):
        return

    _ORIGINAL_METADATA = ldc.LargeImageReader.metadata

    def wrapped(self):
        payload = _ORIGINAL_METADATA(self)
        if str(getattr(self, 'format', '')) != 'OME-Zarr' or getattr(self, '_root', None) is None:
            return payload

        try:
            attrs = dict(self._root.attrs)
            multiscales = attrs.get('multiscales') or []
            if not multiscales:
                return payload
            ms = multiscales[0]
            datasets = ms.get('datasets') or []
            if not datasets:
                return payload

            level = min(max(0, int(getattr(self, 'level', 0))), len(datasets) - 1)
            dataset = datasets[level]
            axes_meta = ms.get('axes') or []
            axes = list(getattr(self, 'axes', []))

            scale = None
            for transform in dataset.get('coordinateTransformations') or []:
                if str(transform.get('type', '')).lower() == 'scale':
                    values = transform.get('scale')
                    if isinstance(values, (list, tuple)) and len(values) == len(axes):
                        scale = [float(v) for v in values]
                        break

            voxel_size_um = {}
            if scale is not None:
                for axis_name in ('X', 'Y'):
                    if axis_name not in axes:
                        continue
                    idx = axes.index(axis_name)
                    axis_meta = (
                        axes_meta[idx]
                        if idx < len(axes_meta) and isinstance(axes_meta[idx], dict)
                        else {}
                    )
                    value_um = _unit_to_um(scale[idx], axis_meta.get('unit'))
                    if value_um is not None and np.isfinite(value_um) and value_um > 0:
                        voxel_size_um[axis_name.lower()] = float(value_um)
            if voxel_size_um:
                payload['voxel_size_um'] = voxel_size_um

            omero_channels = (attrs.get('omero') or {}).get('channels') or []
            if omero_channels:
                payload['channel_metadata'] = [
                    {
                        'index': int(i),
                        'name': str(row.get('label', f'Channel {i}')),
                        'color': row.get('color'),
                        'window': row.get('window'),
                    }
                    for i, row in enumerate(omero_channels)
                ]

            # These schema tokens deliberately invalidate earlier tile-scan
            # checkpoints whenever converted-ND2 intensity handling or the
            # microwell detector changes.
            payload['nd2_omezarr_scaling_schema'] = SCALING_SCHEMA_VERSION
            payload['nd2_omezarr_detector_schema'] = DETECTOR_SCHEMA_VERSION
            payload_no_fp = dict(payload)
            payload_no_fp.pop('reference_fingerprint_sha256', None)
            payload['reference_fingerprint_sha256'] = ldc._json_fingerprint(payload_no_fp)
        except Exception:
            pass
        return payload

    wrapped._omezarr_physical_metadata_wrapper = True
    ldc.LargeImageReader.metadata = wrapped


def install_omezarr_window_scaled_reads() -> None:
    """Scale ND2-derived OME-Zarr planes with each OME display window."""
    global _ORIGINAL_READ_CHANNEL
    if getattr(ldc.LargeImageReader.read_channel_region, '_omezarr_window_scaled_wrapper', False):
        return

    _ORIGINAL_READ_CHANNEL = ldc.LargeImageReader.read_channel_region

    def wrapped(self, x0: int, y0: int, width: int, height: int, channel: int = 0,
                z_index: int = 0, t_index: int = 0):
        if str(getattr(self, 'format', '')) != 'OME-Zarr':
            return _ORIGINAL_READ_CHANNEL(
                self, x0, y0, width, height, channel, z_index, t_index
            )

        x0_i = max(0, int(x0))
        y0_i = max(0, int(y0))
        x1_i = min(self.width, x0_i + max(1, int(width)))
        y1_i = min(self.height, y0_i + max(1, int(height)))
        ch = 0 if self.channel_count == 1 else int(channel)
        selector, remaining = self._selector(
            x0_i, y0_i, x1_i, y1_i, ch, z_index, t_index
        )
        if selector is None:
            return np.zeros((y1_i - y0_i, x1_i - x0_i), dtype=np.uint8)

        arr = np.asarray(self.array[selector])
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ldc.LargeSourceError(
                f'Region read returned shape {arr.shape}; expected a 2D plane.'
            )
        if remaining == ['X', 'Y']:
            arr = arr.T

        window = _display_window(self, ch)
        if window is None:
            return ldc._to_uint8(arr)

        low, high = window
        work = arr.astype(np.float32, copy=False)
        scaled = (work - float(low)) * (255.0 / float(high - low))
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    wrapped._omezarr_window_scaled_wrapper = True
    ldc.LargeImageReader.read_channel_region = wrapped


def local_contrast_dic(rgb: np.ndarray) -> np.ndarray:
    """Locally stretch a DIC tile for edge detection only.

    The original DIC pixels are left untouched for every downstream QC step.
    Percentile stretching is calculated independently per scan tile so faint
    microwell walls are not suppressed by the much wider global OME window.
    """
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or float(high) <= float(low) + 1.0:
        return gray
    work = gray.astype(np.float32)
    work = (work - float(low)) * (255.0 / float(high - low))
    return np.clip(work, 0.0, 255.0).astype(np.uint8)


def detect_wells_converted_dic(rgb: np.ndarray, settings) -> np.ndarray:
    """Physical-radius Hough detection after local DIC contrast normalisation."""
    rmin = max(1, int(round(settings.well_rmin)))
    rmax = max(rmin + 1, int(round(settings.well_rmax)))
    spacing = max(1, int(round(settings.well_spacing)))

    gray = local_contrast_dic(rgb)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=float(spacing),
        param1=75.0,
        param2=float(settings.hough_p2),
        minRadius=int(rmin),
        maxRadius=int(rmax),
    )
    if circles is None:
        return np.empty((0, 3), dtype=int)

    circles = np.round(circles[0]).astype(int)
    kept = []
    for x, y, radius in circles[np.argsort(circles[:, 0])]:
        if all((x - a) ** 2 + (y - b) ** 2 > 20 ** 2 for a, b, _ in kept):
            kept.append((int(x), int(y), int(radius)))
    return np.asarray(kept, dtype=int)


def install_converted_nd2_bridge() -> None:
    install_omezarr_physical_metadata()
    install_omezarr_window_scaled_reads()


def calibrated_scan_settings(settings, converted_meta: dict):
    """Return Settings using physical 100-µm microwell Hough geometry."""
    voxel = converted_meta.get('voxel_size_um') or {}
    values = []
    for key in ('x', 'y'):
        try:
            value = float(voxel.get(key))
        except (TypeError, ValueError):
            value = float('nan')
        if np.isfinite(value) and value > 0:
            values.append(value)
    if len(values) != 2:
        raise ValueError(
            'Converted OME-Zarr does not contain valid X/Y physical pixel calibration.'
        )

    um_per_pixel = float(np.mean(values))
    expected_radius_px = float(settings.well_diameter_um) / (2.0 * um_per_pixel)
    derived = copy.copy(settings)
    derived.well_rmin = max(2, int(round(expected_radius_px * 0.80)))
    derived.well_rmax = max(derived.well_rmin + 2, int(round(expected_radius_px * 1.20)))
    derived.well_spacing = max(2, int(round(expected_radius_px * 1.50)))
    return derived, expected_radius_px, um_per_pixel


def process_converted_nd2_omezarr_qc(*args, **kwargs):
    """Run final QC on an ND2-derived OME-Zarr with a route-local DIC detector."""
    global _ORIGINAL_DETECT_WELLS
    install_converted_nd2_bridge()

    old_detector = ldc.detect_wells
    _ORIGINAL_DETECT_WELLS = old_detector
    ldc.detect_wells = detect_wells_converted_dic
    try:
        result = process_large_experiment_qc(*args, **kwargs)
    finally:
        ldc.detect_wells = old_detector

    root = Path(result[0])
    config_path = root / 'run_configuration.json'
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding='utf-8'))
            final_qc = cfg.setdefault('nd2_final_qc', {})
            final_qc['physical_pixel_size_source'] = (
                'OME-NGFF X/Y coordinateTransformations preserved from original Nikon ND2'
            )
            final_qc['analysis_source'] = 'ND2-derived OME-Zarr'
            final_qc['channel_scaling'] = (
                'per-channel OME display windows mapped to uint8 before image processing'
            )
            final_qc['channel_scaling_schema'] = SCALING_SCHEMA_VERSION
            final_qc['well_detector'] = (
                'physical-radius Hough detection after per-tile 1st–99th percentile DIC contrast normalisation'
            )
            final_qc['well_detector_schema'] = DETECTOR_SCHEMA_VERSION
            config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        except Exception:
            pass
    return result


def infer_channel_indices(converted_meta: dict) -> dict[str, int]:
    names = {
        str(row.get('name', '')).strip().lower(): int(row.get('index', i))
        for i, row in enumerate(converted_meta.get('channel_metadata') or [])
    }

    def first_match(tokens, default):
        for name, index in names.items():
            if any(token in name for token in tokens):
                return int(index)
        return int(default)

    return {
        'gfp': first_match(('gfp', 'green'), 0),
        'rfp': first_match(('rfp', 'red'), 1),
        'dic': first_match(('dic', 'brightfield', 'bf'), 2),
    }
