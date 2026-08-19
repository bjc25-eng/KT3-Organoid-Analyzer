from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

import large_data_core as ldc
from nd2_qc import process_large_experiment_qc

_ORIGINAL_METADATA = None


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


def install_omezarr_physical_metadata() -> None:
    """Teach LargeImageReader.metadata() to expose OME-NGFF physical calibration.

    This is intentionally additive and only changes OME-Zarr metadata payloads.
    TIFF and native ND2 behaviour are untouched.
    """
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
                    axis_meta = axes_meta[idx] if idx < len(axes_meta) and isinstance(axes_meta[idx], dict) else {}
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
                    }
                    for i, row in enumerate(omero_channels)
                ]

            # Calibration is analysis-critical, so include it in the checkpoint fingerprint.
            payload_no_fp = dict(payload)
            payload_no_fp.pop('reference_fingerprint_sha256', None)
            payload['reference_fingerprint_sha256'] = ldc._json_fingerprint(payload_no_fp)
        except Exception:
            # Leave the base metadata intact. The final-QC physical calibration check
            # will fail explicitly rather than silently inventing calibration.
            pass
        return payload

    wrapped._omezarr_physical_metadata_wrapper = True
    ldc.LargeImageReader.metadata = wrapped


def calibrated_scan_settings(settings, converted_meta: dict):
    """Return a copy of Settings using physical 100-µm microwell geometry."""
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
        raise ValueError('Converted OME-Zarr does not contain valid X/Y physical pixel calibration.')

    um_per_pixel = float(np.mean(values))
    expected_radius_px = float(settings.well_diameter_um) / (2.0 * um_per_pixel)
    derived = copy.copy(settings)
    derived.well_rmin = max(2, int(round(expected_radius_px * 0.80)))
    derived.well_rmax = max(derived.well_rmin + 2, int(round(expected_radius_px * 1.20)))
    derived.well_spacing = max(2, int(round(expected_radius_px * 1.50)))
    return derived, expected_radius_px, um_per_pixel


def process_converted_nd2_omezarr_qc(*args, **kwargs):
    """Run the validated final-QC engine on an ND2-derived OME-Zarr source."""
    install_omezarr_physical_metadata()
    result = process_large_experiment_qc(*args, **kwargs)
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
