from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import fsspec
import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image

from analysis_core import (
    BRIGHTFIELD_MODE,
    GFP_MODE,
    Settings,
    cluster,
    detect_psc,
    detect_wells,
    green_excess,
    grid_index,
    make_training_masks,
    segment_pdos,
    segment_unlabelled_pdos_in_well,
    slugify,
    stable_token,
)

LARGE_SCHEMA_VERSION = '1.0'
DEFAULT_TILE_SIZE = 2048
DEFAULT_STANDARD_CROP_SIZE = 256


class LargeSourceError(RuntimeError):
    pass


def _protocol(fs) -> str:
    p = getattr(fs, 'protocol', '')
    if isinstance(p, (list, tuple)):
        p = p[0] if p else ''
    return str(p or '')


def _normalise_info(info: dict | None) -> dict:
    info = dict(info or {})
    keys = ['size', 'etag', 'ETag', 'last_modified', 'LastModified', 'mtime', 'version_id', 'VersionId']
    out = {}
    for key in keys:
        if key in info and info[key] is not None:
            value = info[key]
            if hasattr(value, 'isoformat'):
                value = value.isoformat()
            out[key] = str(value) if not isinstance(value, (int, float, bool)) else value
    return out


def _json_fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _safe_extract_zip(data: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise LargeSourceError('Unsafe path in resume bundle.')
        zf.extractall(destination)


def make_resume_bundle(work_root: str | Path) -> bytes:
    work_root = Path(work_root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in work_root.rglob('*'):
            if path.is_file():
                zf.write(path, path.relative_to(work_root))
    return buf.getvalue()


def restore_resume_bundle(data: bytes) -> Path:
    root = Path(tempfile.mkdtemp(prefix='kt3_large_resume_'))
    _safe_extract_zip(data, root)
    return root


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        work = arr.astype(np.float32)
        if info.min < 0:
            work = work - float(info.min)
            denom = float(info.max - info.min)
        else:
            denom = float(info.max) if info.max else 1.0
        return np.clip(work * (255.0 / denom), 0, 255).astype(np.uint8)
    work = arr.astype(np.float32)
    finite = work[np.isfinite(work)]
    if finite.size == 0:
        return np.zeros(work.shape, dtype=np.uint8)
    if float(np.nanmax(finite)) <= 1.0 and float(np.nanmin(finite)) >= 0.0:
        work = work * 255.0
    return np.clip(work, 0, 255).astype(np.uint8)


def _axis_names_from_ngff(multiscale: dict, ndim: int) -> list[str]:
    axes = multiscale.get('axes') or []
    names = []
    for axis in axes:
        if isinstance(axis, dict):
            names.append(str(axis.get('name', '')).upper())
        else:
            names.append(str(axis).upper())
    if len(names) != ndim:
        if ndim == 2:
            return ['Y', 'X']
        if ndim == 3:
            return ['C', 'Y', 'X']
        return [f'D{i}' for i in range(ndim-2)] + ['Y', 'X']
    return names


class LargeImageReader:
    """Region reader for tiled TIFF/OME-TIFF/BigTIFF and OME-Zarr.

    The object exposes full-resolution dimensions but reads only requested
    regions. TIFF is exposed through tifffile's Zarr interface, which permits
    chunk/tile access instead of materialising the complete image array.
    """

    def __init__(self, uri: str, source_type: str = 'auto', series_index: int = 0, level: int = 0):
        self.uri = str(uri).strip()
        self.source_type_requested = source_type
        self.series_index = int(series_index)
        self.level = int(level)
        self._file = None
        self._tiff = None
        self._store = None
        self._root = None
        self.array = None
        self.axes: list[str] = []
        self.shape = ()
        self.dtype = None
        self.format = None
        self.is_bigtiff = False
        self.is_ome = False
        self.source_info = {}
        self._open()

    def _source_type(self):
        requested = str(self.source_type_requested or 'auto').lower()
        if requested in {'ome-zarr', 'zarr'}:
            return 'zarr'
        if requested in {'ome-tiff', 'bigtiff', 'tiff'}:
            return 'tiff'
        lower = self.uri.lower().split('?', 1)[0].rstrip('/')
        return 'zarr' if lower.endswith('.zarr') else 'tiff'

    def _open(self):
        kind = self._source_type()
        if kind == 'zarr':
            self._open_zarr()
        else:
            self._open_tiff()
        if 'Y' not in self.axes or 'X' not in self.axes:
            raise LargeSourceError(f'Source axes {self.axes} do not contain X and Y dimensions.')

    def _open_tiff(self):
        try:
            fs, path = fsspec.core.url_to_fs(self.uri)
            try:
                self.source_info = _normalise_info(fs.info(path))
            except Exception:
                self.source_info = {}
            proto = _protocol(fs)
            if proto in {'http', 'https'}:
                self._file = fs.open(path, 'rb', block_size=8 * 1024 * 1024, cache_type='readahead')
            else:
                self._file = fs.open(path, 'rb')
            self._tiff = tifffile.TiffFile(self._file)
            self.is_bigtiff = bool(self._tiff.is_bigtiff)
            self.is_ome = bool(self._tiff.ome_metadata)
            if self.series_index >= len(self._tiff.series):
                raise LargeSourceError(f'TIFF has {len(self._tiff.series)} series; series {self.series_index} is unavailable.')
            series = self._tiff.series[self.series_index]
            if hasattr(series, 'levels') and self.level >= len(series.levels):
                raise LargeSourceError(f'TIFF series has {len(series.levels)} pyramid levels; level {self.level} is unavailable.')
            self._store = series.aszarr(level=self.level)
            self.array = zarr.open(self._store, mode='r')
            axes = getattr(series, 'axes', '')
            self.axes = [str(a).upper() for a in axes]
            self.shape = tuple(int(v) for v in self.array.shape)
            if len(self.axes) != len(self.shape):
                self.axes = ['Y', 'X'] if len(self.shape) == 2 else ['C', 'Y', 'X'][-len(self.shape):]
            self.dtype = np.dtype(self.array.dtype)
            if self.is_ome and self.is_bigtiff:
                self.format = 'OME-BigTIFF'
            elif self.is_ome:
                self.format = 'OME-TIFF'
            elif self.is_bigtiff:
                self.format = 'BigTIFF'
            else:
                self.format = 'TIFF'
        except Exception as exc:
            self.close()
            raise LargeSourceError(f'Could not open TIFF source: {exc}') from exc

    def _open_zarr(self):
        try:
            mapper = fsspec.get_mapper(self.uri)
            self._root = zarr.open_group(store=mapper, mode='r')
            attrs = dict(self._root.attrs)
            multiscales = attrs.get('multiscales') or []
            if multiscales:
                ms = multiscales[0]
                datasets = ms.get('datasets') or []
                if not datasets:
                    raise LargeSourceError('OME-Zarr multiscales metadata contains no datasets.')
                if self.level >= len(datasets):
                    raise LargeSourceError(f'OME-Zarr has {len(datasets)} pyramid levels; level {self.level} is unavailable.')
                path = str(datasets[self.level].get('path', '0'))
                self.array = self._root[path]
                self.axes = _axis_names_from_ngff(ms, self.array.ndim)
            else:
                arrays = list(self._root.arrays())
                if not arrays:
                    raise LargeSourceError('Zarr group contains no arrays.')
                _, self.array = arrays[0]
                self.axes = ['Y', 'X'] if self.array.ndim == 2 else ['C', 'Y', 'X'][-self.array.ndim:]
            self.shape = tuple(int(v) for v in self.array.shape)
            self.dtype = np.dtype(self.array.dtype)
            self.format = 'OME-Zarr' if multiscales else 'Zarr'
            self.is_ome = bool(multiscales)
            self.is_bigtiff = False
            try:
                fs, path = fsspec.core.url_to_fs(self.uri)
                self.source_info = _normalise_info(fs.info(path))
            except Exception:
                self.source_info = {}
        except Exception as exc:
            self.close()
            raise LargeSourceError(f'Could not open OME-Zarr source: {exc}') from exc

    @property
    def width(self) -> int:
        return int(self.shape[self.axes.index('X')])

    @property
    def height(self) -> int:
        return int(self.shape[self.axes.index('Y')])

    @property
    def channel_count(self) -> int:
        for axis in ('C', 'S'):
            if axis in self.axes:
                return int(self.shape[self.axes.index(axis)])
        return 1

    def metadata(self) -> dict:
        payload = {
            'uri': self.uri,
            'format': self.format,
            'is_bigtiff': self.is_bigtiff,
            'is_ome': self.is_ome,
            'shape': list(self.shape),
            'axes': self.axes,
            'dtype': str(self.dtype),
            'width_px': self.width,
            'height_px': self.height,
            'channel_count': self.channel_count,
            'series_index': self.series_index,
            'pyramid_level': self.level,
            **self.source_info,
        }
        payload['reference_fingerprint_sha256'] = _json_fingerprint(payload)
        return payload

    def _selector(self, x0: int, y0: int, x1: int, y1: int, channel: int | None,
                  z_index: int = 0, t_index: int = 0):
        selector = []
        remaining = []
        for axis, size in zip(self.axes, self.shape):
            if axis == 'Y':
                selector.append(slice(int(y0), int(y1)))
                remaining.append('Y')
            elif axis == 'X':
                selector.append(slice(int(x0), int(x1)))
                remaining.append('X')
            elif axis in {'C', 'S'}:
                idx = 0 if channel is None else int(channel)
                if idx < 0 or idx >= int(size):
                    return None, []
                selector.append(idx)
            elif axis == 'Z':
                selector.append(min(max(0, int(z_index)), int(size)-1))
            elif axis == 'T':
                selector.append(min(max(0, int(t_index)), int(size)-1))
            else:
                selector.append(0)
        return tuple(selector), remaining

    def read_channel_region(self, x0: int, y0: int, width: int, height: int, channel: int = 0,
                            z_index: int = 0, t_index: int = 0) -> np.ndarray:
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(self.width, x0 + max(1, int(width)))
        y1 = min(self.height, y0 + max(1, int(height)))
        if self.channel_count == 1:
            channel = 0
        selector, remaining = self._selector(x0, y0, x1, y1, channel, z_index, t_index)
        if selector is None:
            return np.zeros((y1-y0, x1-x0), dtype=np.uint8)
        arr = np.asarray(self.array[selector])
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise LargeSourceError(f'Region read returned shape {arr.shape}; expected a 2D plane.')
        if remaining == ['X', 'Y']:
            arr = arr.T
        return _to_uint8(arr)

    def read_rgb_region(self, x0: int, y0: int, width: int, height: int,
                        red_channel: int = 0, green_channel: int = 1, blue_channel: int = 2,
                        z_index: int = 0, t_index: int = 0) -> np.ndarray:
        if self.channel_count == 1:
            plane = self.read_channel_region(x0, y0, width, height, 0, z_index, t_index)
            return np.stack([plane, plane, plane], axis=-1)
        shape = (min(self.height, y0+height)-max(0,y0), min(self.width, x0+width)-max(0,x0))
        channels = []
        for ch in (red_channel, green_channel, blue_channel):
            if ch is None or int(ch) < 0 or int(ch) >= self.channel_count:
                channels.append(np.zeros(shape, dtype=np.uint8))
            else:
                channels.append(self.read_channel_region(x0, y0, width, height, int(ch), z_index, t_index))
        return np.stack(channels, axis=-1)

    def close(self):
        for obj in [self._store, self._tiff, self._file]:
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._store = self._tiff = self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def probe_large_source(uri: str, source_type: str = 'auto', series_index: int = 0, level: int = 0) -> dict:
    with LargeImageReader(uri, source_type=source_type, series_index=series_index, level=level) as reader:
        return reader.metadata()


def compute_streaming_sha256(uri: str, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream a file-like source to SHA-256 without holding it in memory.

    This is intentionally optional: hashing a 5 GB remote TIFF requires reading
    the full 5 GB over the network. OME-Zarr directories should use their
    reference fingerprint or an externally supplied dataset checksum instead.
    """
    fs, path = fsspec.core.url_to_fs(uri)
    info = fs.info(path)
    if str(info.get('type', '')).lower() == 'directory':
        raise LargeSourceError('Full SHA-256 streaming is only supported for file-like sources, not Zarr directories.')
    h = hashlib.sha256()
    with fs.open(path, 'rb') as f:
        while True:
            chunk = f.read(int(chunk_bytes))
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _channel_config(config: dict) -> dict:
    return {
        'red_channel': int(config.get('red_channel', 0)),
        'green_channel': int(config.get('green_channel', 1)),
        'blue_channel': int(config.get('blue_channel', 2)),
        'brightfield_channel': int(config.get('brightfield_channel', -1)),
        'well_detection_channel': int(config.get('well_detection_channel', -1)),
        'z_index': int(config.get('z_index', 0)),
        'internal_t_index': int(config.get('internal_t_index', 0)),
    }


def _read_analysis_region(reader: LargeImageReader, x0: int, y0: int, w: int, h: int,
                          config: dict, organoid_mode: str):
    cfg = _channel_config(config)
    rgb = reader.read_rgb_region(
        x0, y0, w, h,
        cfg['red_channel'], cfg['green_channel'], cfg['blue_channel'],
        cfg['z_index'], cfg['internal_t_index']
    )
    if organoid_mode == BRIGHTFIELD_MODE and cfg['brightfield_channel'] >= 0:
        bf = reader.read_channel_region(
            x0, y0, w, h, cfg['brightfield_channel'], cfg['z_index'], cfg['internal_t_index']
        )
        pdo_rgb = np.stack([bf, bf, bf], axis=-1)
    else:
        pdo_rgb = rgb
    if cfg['well_detection_channel'] >= 0:
        wd = reader.read_channel_region(
            x0, y0, w, h, cfg['well_detection_channel'], cfg['z_index'], cfg['internal_t_index']
        )
        well_rgb = np.stack([wd, wd, wd], axis=-1)
    elif organoid_mode == BRIGHTFIELD_MODE and cfg['brightfield_channel'] >= 0:
        well_rgb = pdo_rgb
    else:
        well_rgb = rgb
    return rgb, pdo_rgb, well_rgb


def _dedupe_wells(wells, distance_px=20):
    kept = []
    for x, y, r in sorted(wells, key=lambda q: (q[1], q[0])):
        if all((x-a)**2 + (y-b)**2 > distance_px**2 for a, b, _ in kept):
            kept.append((int(x), int(y), int(r)))
    return kept


def scan_wells_tiled(reader: LargeImageReader, settings: Settings, config: dict, work_dir: Path,
                     source_fingerprint: str, organoid_mode: str, tile_size: int = DEFAULT_TILE_SIZE,
                     progress_callback=None):
    """Detect wells by scanning small expanded tiles and save a resumable checkpoint."""
    tile_size = max(512, int(tile_size))
    overlap = max(int(settings.well_rmax) * 3, 96)
    checkpoint_path = work_dir/'tile_scan_checkpoint.json'
    checkpoint = _read_json(checkpoint_path, default={}) or {}
    if checkpoint.get('source_fingerprint') != source_fingerprint:
        checkpoint = {'source_fingerprint': source_fingerprint, 'completed_tiles': [], 'wells': []}
    completed = set(checkpoint.get('completed_tiles', []))
    wells = [tuple(map(int, q)) for q in checkpoint.get('wells', [])]

    tiles = []
    for y0 in range(0, reader.height, tile_size):
        for x0 in range(0, reader.width, tile_size):
            tiles.append((x0, y0, min(tile_size, reader.width-x0), min(tile_size, reader.height-y0)))

    for tile_i, (core_x, core_y, core_w, core_h) in enumerate(tiles):
        tile_id = f'{core_x}_{core_y}'
        if tile_id in completed:
            if progress_callback:
                progress_callback(tile_i+1, len(tiles), 'well_scan')
            continue
        rx0 = max(0, core_x-overlap); ry0 = max(0, core_y-overlap)
        rx1 = min(reader.width, core_x+core_w+overlap); ry1 = min(reader.height, core_y+core_h+overlap)
        _, _, well_rgb = _read_analysis_region(reader, rx0, ry0, rx1-rx0, ry1-ry0, config, organoid_mode)
        local = detect_wells(well_rgb, settings)
        for lx, ly, r in local:
            gx, gy = int(rx0+lx), int(ry0+ly)
            if not (core_x <= gx < core_x+core_w and core_y <= gy < core_y+core_h):
                continue
            if gx-r < 2 or gx+r >= reader.width-2 or gy-r < 2 or gy+r >= reader.height-2:
                continue
            wells.append((gx, gy, int(r)))
        completed.add(tile_id)
        checkpoint = {
            'source_fingerprint': source_fingerprint,
            'tile_size': tile_size,
            'overlap_px': overlap,
            'completed_tiles': sorted(completed),
            'wells': [list(q) for q in _dedupe_wells(wells)],
        }
        _atomic_json(checkpoint_path, checkpoint)
        wells = [tuple(q) for q in checkpoint['wells']]
        if progress_callback:
            progress_callback(tile_i+1, len(tiles), 'well_scan')
    return np.asarray(_dedupe_wells(wells), dtype=int), checkpoint_path


def _save_standard_crop(arr: np.ndarray, path: Path, size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).resize((int(size), int(size)), Image.Resampling.LANCZOS).save(path)


def _save_standard_mask(arr: np.ndarray, path: Path, size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).resize((int(size), int(size)), Image.Resampling.NEAREST).save(path)


def _write_incremental_tables(work_dir: Path, wells_rows, pdo_rows, psc_rows):
    pd.DataFrame(wells_rows).to_csv(work_dir/'well_observations_partial.csv', index=False)
    pd.DataFrame(pdo_rows).to_csv(work_dir/'pdo_observations_partial.csv', index=False)
    pd.DataFrame(psc_rows).to_csv(work_dir/'psc_observations_partial.csv', index=False)


def _read_partial(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict('records')
    except pd.errors.EmptyDataError:
        return []


def analyse_large_source(source: dict, settings: Settings, config: dict, work_dir: Path,
                         tile_size: int = DEFAULT_TILE_SIZE, standard_crop_size: int = DEFAULT_STANDARD_CROP_SIZE,
                         progress_callback=None) -> dict:
    """Analyse one huge source using tile scanning plus one small read per well."""
    work_dir.mkdir(parents=True, exist_ok=True)
    uri = str(source['source_uri'])
    with LargeImageReader(
        uri, source_type=source.get('source_type', 'auto'),
        series_index=int(source.get('series_index', 0)), level=int(source.get('pyramid_level', 0))
    ) as reader:
        metadata = reader.metadata()
        fingerprint = metadata['reference_fingerprint_sha256']
        if source.get('source_sha256'):
            metadata['source_sha256'] = str(source['source_sha256'])
        if bool(source.get('compute_full_sha256', False)) and reader.format != 'OME-Zarr':
            metadata['source_sha256'] = compute_streaming_sha256(uri)
        _atomic_json(work_dir/'source_metadata.json', metadata)

        if reader.channel_count == 1 and source['organoid_mode'] == GFP_MODE and source.get('rfp_psc_present'):
            raise LargeSourceError('A single-channel source cannot quantify both GFP-labelled PDOs and RFP PSCs. Use a multichannel source or disable RFP PSC analysis.')

        circles, _ = scan_wells_tiled(
            reader, settings, config, work_dir, fingerprint, source['organoid_mode'], tile_size, progress_callback
        )
        if len(circles) == 0:
            raise LargeSourceError('No fully visible microwells were detected in this source.')

        xs, ys = cluster(circles[:, 0]), cluster(circles[:, 1])
        umpp = float(settings.well_diameter_um) / (2.0 * float(np.median(circles[:, 2])))

        analysis_checkpoint_path = work_dir/'well_analysis_checkpoint.json'
        state = _read_json(analysis_checkpoint_path, default={}) or {}
        if state.get('source_fingerprint') != fingerprint:
            state = {'source_fingerprint': fingerprint, 'completed_wells': []}
        completed = set(state.get('completed_wells', []))
        wells_rows = _read_partial(work_dir/'well_observations_partial.csv')
        pdo_rows = _read_partial(work_dir/'pdo_observations_partial.csv')
        psc_rows = _read_partial(work_dir/'psc_observations_partial.csv')

        exp = stable_token(source.get('experiment_id', 'Experiment_001'), 'Experiment_001')
        dev = stable_token(source.get('device_id', 'Array_001'), 'Array_001')
        lane = int(source['condition_index'])
        tp = int(source['timepoint_index'])
        field_id = stable_token(source.get('field_id', f"F{int(source.get('field_index',1)):02d}"), 'F01')
        source_uid = f'{exp}__{dev}__L{lane:02d}__T{tp:02d}__{field_id}'

        for i, (x, y, r) in enumerate(circles):
            col, row = grid_index(int(x), int(y), xs, ys)
            well_index = f'{col},{row}'
            trajectory_id = f'{exp}__{dev}__L{lane:02d}__{field_id}__W{col}_{row}'
            obs_id = f'{trajectory_id}__T{tp:02d}'
            if obs_id in completed:
                if progress_callback:
                    progress_callback(i+1, len(circles), 'well_analysis')
                continue

            crop_r = max(int(math.ceil(float(r) * 1.75)), int(settings.well_rmax) + 8)
            x0 = max(0, int(x)-crop_r); y0 = max(0, int(y)-crop_r)
            x1 = min(reader.width, int(x)+crop_r); y1 = min(reader.height, int(y)+crop_r)
            rgb, pdo_rgb, _ = _read_analysis_region(reader, x0, y0, x1-x0, y1-y0, config, source['organoid_mode'])
            cx, cy = int(x)-x0, int(y)-y0

            if source['organoid_mode'] == GFP_MODE:
                candidates = segment_pdos(green_excess(pdo_rgb), settings)
                assigned = [o for o in candidates if (o['x']-cx)**2 + (o['y']-cy)**2 <= (0.86*r)**2]
            else:
                assigned = segment_unlabelled_pdos_in_well(pdo_rgb, cx, cy, int(r), settings)

            if source.get('rfp_psc_present'):
                foci = detect_psc(rgb, cx, cy, int(r), settings)
            else:
                foci = []
            psc_n = len(foci) if source.get('rfp_psc_present') else None
            sizes = [2.0 * math.sqrt(float(o['area'])/math.pi) * umpp for o in assigned]

            focus_rows_local = []
            for focus_n, (fx, fy, score) in enumerate(foci, 1):
                gx, gy = x0+int(fx), y0+int(fy)
                focus_id = f'{obs_id}__PSCFOCUS{focus_n:03d}'
                row_focus = {
                    'source_uid': source_uid, 'trajectory_id': trajectory_id, 'well_observation_id': obs_id,
                    'psc_focus_id': focus_id, 'focus_number_in_well': focus_n,
                    'focus_x_px_fullres': gx, 'focus_y_px_fullres': gy, 'focus_score': float(score),
                    'condition_index': lane, 'condition': source['condition'], 'timepoint_index': tp,
                    'timepoint': source['timepoint'], 'elapsed_time': source.get('elapsed_time', np.nan),
                    'qc_status': 'automated_not_manually_reviewed'
                }
                psc_rows.append(row_focus)
                focus_rows_local.append({'focus_x_px': int(fx), 'focus_y_px': int(fy)})

            for pdo_n, (obj, size_um) in enumerate(zip(assigned, sizes), 1):
                pdo_rows.append({
                    'source_uid': source_uid, 'trajectory_id': trajectory_id, 'well_observation_id': obs_id,
                    'pdo_observation_id': f'{obs_id}__PDO{pdo_n:02d}', 'PDO_number_in_well': pdo_n,
                    'PDO_count_in_well': len(assigned),
                    'centroid_x_px_fullres': x0 + float(obj['x']), 'centroid_y_px_fullres': y0 + float(obj['y']),
                    'projected_area_px2': float(obj['area']), 'projected_area_um2': float(obj['area']) * umpp**2,
                    'equivalent_circular_diameter_um': float(size_um),
                    'PSC_like_focus_count_in_well': psc_n,
                    'condition_index': lane, 'condition': source['condition'], 'timepoint_index': tp,
                    'timepoint': source['timepoint'], 'elapsed_time': source.get('elapsed_time', np.nan),
                    'qc_status': 'automated_not_manually_reviewed'
                })

            crop_dir = work_dir/'ml_crops'
            full_dir = work_dir/'fullres_crops'
            mask_dir = work_dir/'ml_masks'
            full_dir.mkdir(parents=True, exist_ok=True)
            raw_name = f'{obs_id}__fullres.png'
            Image.fromarray(rgb).save(full_dir/raw_name)
            std_name = f'{obs_id}__rgb_256.png'
            _save_standard_crop(rgb, crop_dir/std_name, standard_crop_size)
            if source['organoid_mode'] == BRIGHTFIELD_MODE and int(config.get('brightfield_channel', -1)) >= 0:
                bf = pdo_rgb[..., 0]
                _save_standard_crop(np.stack([bf,bf,bf], axis=-1), crop_dir/f'{obs_id}__brightfield_256.png', standard_crop_size)

            one_circle = np.asarray([[cx, cy, int(r)]], dtype=int)
            well_mask, pdo_mask, psc_mask = make_training_masks(rgb if source['organoid_mode']==GFP_MODE else pdo_rgb, one_circle, focus_rows_local, settings)
            _save_standard_mask(well_mask, mask_dir/f'{obs_id}__well_mask_256.png', standard_crop_size)
            _save_standard_mask(pdo_mask, mask_dir/f'{obs_id}__pdo_mask_256.png', standard_crop_size)
            _save_standard_mask(psc_mask, mask_dir/f'{obs_id}__psc_focus_mask_256.png', standard_crop_size)

            wells_rows.append({
                'source_uid': source_uid, 'source_uri': uri, 'source_format': metadata['format'],
                'experiment_id': source.get('experiment_id','Experiment_001'), 'device_id': source.get('device_id','Array_001'),
                'biological_replicate_id': source.get('biological_replicate_id','Replicate_1'),
                'pdo_model': source.get('pdo_model',''), 'condition_index': lane, 'condition': source['condition'],
                'field_id': field_id, 'timepoint_index': tp, 'timepoint': source['timepoint'],
                'elapsed_time': source.get('elapsed_time', np.nan), 'time_unit': source.get('time_unit','days'),
                'drug_or_therapeutic': source.get('drug_or_therapeutic',''), 'concentration': source.get('concentration', np.nan),
                'concentration_unit': source.get('concentration_unit',''), 'organoid_detection_mode': source['organoid_mode'],
                'GFP_labelled_organoids': bool(source['organoid_mode']==GFP_MODE),
                'RFP_PSC_stromal_cells_present': bool(source.get('rfp_psc_present')),
                'well_index': well_index, 'well_col_index': col, 'well_row_index': row,
                'well_centre_x_px_fullres': int(x), 'well_centre_y_px_fullres': int(y), 'well_radius_px': int(r),
                'crop_x0_px_fullres': x0, 'crop_y0_px_fullres': y0, 'crop_width_px': int(x1-x0), 'crop_height_px': int(y1-y0),
                'um_per_pixel': umpp, 'PDO_count': len(assigned), 'PSC_like_focus_count': psc_n,
                'trajectory_id': trajectory_id, 'well_observation_id': obs_id,
                'standard_rgb_crop': str((crop_dir/std_name).relative_to(work_dir)),
                'fullres_rgb_crop': str((full_dir/raw_name).relative_to(work_dir)),
                'qc_status': 'automated_not_manually_reviewed', 'qc_fully_visible_well': True,
                'qc_multiple_pdos_in_well': bool(len(assigned)>1), 'qc_no_pdo_detected': bool(len(assigned)==0),
                'qc_brightfield_detection_requires_visual_review': bool(source['organoid_mode']==BRIGHTFIELD_MODE)
            })

            completed.add(obs_id)
            state = {'source_fingerprint': fingerprint, 'completed_wells': sorted(completed), 'source_uid': source_uid}
            _atomic_json(analysis_checkpoint_path, state)
            if (i+1) % 20 == 0 or i == len(circles)-1:
                _write_incremental_tables(work_dir, wells_rows, pdo_rows, psc_rows)
            if progress_callback:
                progress_callback(i+1, len(circles), 'well_analysis')

        _write_incremental_tables(work_dir, wells_rows, pdo_rows, psc_rows)
        pd.DataFrame(circles, columns=['x_px_fullres','y_px_fullres','radius_px']).to_csv(work_dir/'detected_wells_fullres.csv', index=False)
        return {
            'source_uid': source_uid, 'metadata': metadata, 'well_count': len(wells_rows),
            'pdo_count': len(pdo_rows), 'psc_focus_count': len(psc_rows),
            'work_dir': str(work_dir), 'complete': len(completed) >= len(circles)
        }


def _copy_asset_tree(src: Path, dst: Path):
    if not src.exists():
        return
    for p in src.rglob('*'):
        if p.is_file():
            q = dst/p.relative_to(src)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def _concat_csv(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists() and p.stat().st_size:
            try:
                frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _longitudinal_large_table(wdf: pd.DataFrame, pdf: pd.DataFrame) -> pd.DataFrame:
    if wdf.empty:
        return pd.DataFrame()
    out = wdf.copy()
    if not pdf.empty:
        agg = pdf.groupby(['well_observation_id'], as_index=False).agg(
            mean_PDO_diameter_um=('equivalent_circular_diameter_um','mean'),
            max_PDO_diameter_um=('equivalent_circular_diameter_um','max'),
            total_PDO_projected_area_um2=('projected_area_um2','sum')
        )
        out = out.merge(agg, on='well_observation_id', how='left')
    else:
        out['mean_PDO_diameter_um'] = np.nan
        out['max_PDO_diameter_um'] = np.nan
        out['total_PDO_projected_area_um2'] = 0.0
    out['total_PDO_projected_area_um2'] = out['total_PDO_projected_area_um2'].fillna(0.0)
    first = (out.sort_values('timepoint_index').groupby('trajectory_id', as_index=False).first()
             [['trajectory_id','total_PDO_projected_area_um2']]
             .rename(columns={'total_PDO_projected_area_um2':'baseline_total_PDO_area_um2'}))
    out = out.merge(first, on='trajectory_id', how='left')
    out['relative_total_PDO_area_vs_baseline'] = np.where(
        out['baseline_total_PDO_area_um2']>0,
        out['total_PDO_projected_area_um2']/out['baseline_total_PDO_area_um2'], np.nan
    )
    out['percent_area_change_vs_baseline'] = np.where(
        out['baseline_total_PDO_area_um2']>0,
        100.0*(out['relative_total_PDO_area_vs_baseline']-1.0), np.nan
    )
    return out


def _reference_manifest_row(source: dict, source_result: dict | None, status: str, error: str | None=''):
    meta = (source_result or {}).get('metadata', {})
    return {
        'source_uid': (source_result or {}).get('source_uid',''),
        'source_uri': source.get('source_uri',''),
        'field_id': source.get('field_id',''),
        'condition_index': source.get('condition_index'), 'condition': source.get('condition',''),
        'timepoint_index': source.get('timepoint_index'), 'timepoint': source.get('timepoint',''),
        'source_format': meta.get('format',''), 'is_ome': meta.get('is_ome'), 'is_bigtiff': meta.get('is_bigtiff'),
        'width_px_fullres': meta.get('width_px'), 'height_px_fullres': meta.get('height_px'),
        'shape': json.dumps(meta.get('shape')), 'axes': json.dumps(meta.get('axes')), 'dtype': meta.get('dtype',''),
        'remote_size_bytes': meta.get('size'), 'etag': meta.get('etag') or meta.get('ETag'),
        'last_modified': meta.get('last_modified') or meta.get('LastModified') or meta.get('mtime'),
        'reference_fingerprint_sha256': meta.get('reference_fingerprint_sha256',''),
        'source_sha256': meta.get('source_sha256') or source.get('source_sha256',''),
        'analysis_status': status, 'error': error or '',
        'raw_image_copied_into_export': False,
    }


def build_large_ml_export(out: Path, source_manifest: pd.DataFrame, wdf: pd.DataFrame,
                          pdf: pd.DataFrame, pscdf: pd.DataFrame, ldf: pd.DataFrame, work_root: Path):
    ml = out/'machine_learning_export'
    if ml.exists():
        shutil.rmtree(ml)
    for d in [ml/'tables', ml/'assets'/'well_crops_256', ml/'assets'/'masks_256']:
        d.mkdir(parents=True, exist_ok=True)
    source_manifest.to_csv(ml/'tables'/'raw_source_manifest.csv', index=False)
    wdf.to_csv(ml/'tables'/'well_observations.csv', index=False)
    pdf.to_csv(ml/'tables'/'pdo_observations.csv', index=False)
    pscdf.to_csv(ml/'tables'/'psc_focus_observations.csv', index=False)
    ldf.to_csv(ml/'tables'/'longitudinal_trajectories.csv', index=False)
    qc_cols = [c for c in wdf.columns if c.startswith('qc_') or c in {'trajectory_id','well_observation_id','source_uid'}]
    wdf[qc_cols].to_csv(ml/'tables'/'qc_flags.csv', index=False)

    asset_rows = []
    for source_dir in (work_root/'sources').glob('*') if (work_root/'sources').exists() else []:
        for sub, target_name in [('ml_crops','well_crops_256'),('ml_masks','masks_256')]:
            src = source_dir/sub
            if not src.exists():
                continue
            for p in src.rglob('*'):
                if not p.is_file():
                    continue
                q = ml/'assets'/target_name/source_dir.name/p.name
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
                h = hashlib.sha256(q.read_bytes()).hexdigest()
                asset_rows.append({
                    'asset_type': target_name, 'source_work_folder': source_dir.name,
                    'export_relative_path': str(q.relative_to(ml)), 'file_size_bytes': q.stat().st_size,
                    'sha256': h
                })
    pd.DataFrame(asset_rows).to_csv(ml/'tables'/'asset_manifest.csv', index=False)

    schema = {
        'schema_version': LARGE_SCHEMA_VERSION,
        'dataset_type': 'large-source longitudinal PDO/stromal imaging dataset',
        'raw_source_policy': 'Raw 3-5 GB source files are referenced, not copied into this export.',
        'raw_source_identifier': 'reference_fingerprint_sha256 is a hash of URI + remote metadata + image metadata; it is not claimed to be a content SHA-256.',
        'content_hash': 'source_sha256 is populated only when supplied or explicitly streamed/calculated.',
        'coordinate_system': 'All *_fullres coordinates refer to level-0/full-resolution source pixel coordinates.',
        'trajectory_id': 'experiment + array/device + lane + field + well x,y; stable across time.',
        'crop_policy': '256x256 standardized RGB crops are model-ready derivatives. Full-resolution source remains external.',
        'mask_policy': 'Masks are automated and not manually reviewed ground truth.',
    }
    (ml/'schema.json').write_text(json.dumps(schema, indent=2), encoding='utf-8')
    (ml/'README.md').write_text(
        '# Large Dataset ML / Virtual Model Export\n\n'
        'This export intentionally does **not** duplicate the original multi-gigabyte microscopy files. '
        'Use `tables/raw_source_manifest.csv` to resolve each derivative back to its source. Full-resolution '
        'coordinates are retained in the observation tables. `reference_fingerprint_sha256` fingerprints the '
        'source reference and metadata; it is not a content hash unless `source_sha256` is populated.\n\n'
        'Standardized 256x256 per-well crops and masks are stored under `assets/`. Automated masks require visual QC before supervised training.\n',
        encoding='utf-8'
    )
    return ml


def process_large_experiment(sources: list[dict], settings: Settings, config: dict,
                             tile_size: int = DEFAULT_TILE_SIZE,
                             standard_crop_size: int = DEFAULT_STANDARD_CROP_SIZE,
                             work_root: str | Path | None = None,
                             progress_callback=None,
                             make_ml_export: bool = True):
    """Run or resume a whole-array experiment from remote/local references.

    Source failures are recorded instead of destroying completed work. Calling
    this function again with the same work_root and source references resumes
    tile and well checkpoints.
    """
    if not sources:
        raise LargeSourceError('No large-dataset sources were provided.')
    root = Path(work_root) if work_root else Path(tempfile.mkdtemp(prefix='kt3_large_'))
    root.mkdir(parents=True, exist_ok=True)
    source_root = root/'sources'; out = root/'results'
    source_root.mkdir(parents=True, exist_ok=True); (out/'csv').mkdir(parents=True, exist_ok=True)

    run_config = {
        'large_schema_version': LARGE_SCHEMA_VERSION,
        'tile_size': int(tile_size), 'standard_crop_size': int(standard_crop_size),
        'channel_config': _channel_config(config), 'analysis_settings': asdict(settings),
    }
    _atomic_json(root/'run_configuration.json', run_config)

    manifest_rows = []
    statuses = []
    for source_index, source in enumerate(sources, 1):
        field_id = stable_token(source.get('field_id', f"F{int(source.get('field_index',1)):02d}"), 'F01')
        source_key = f"L{int(source['condition_index']):02d}_T{int(source['timepoint_index']):02d}_{field_id}"
        work_dir = source_root/source_key
        try:
            result = analyse_large_source(
                source, settings, config, work_dir, tile_size, standard_crop_size, progress_callback
            )
            status = 'complete' if result.get('complete') else 'partial'
            manifest_rows.append(_reference_manifest_row(source, result, status))
            statuses.append(status)
        except Exception as exc:
            metadata = _read_json(work_dir/'source_metadata.json', default={}) or {}
            pseudo = {'source_uid': '', 'metadata': metadata}
            manifest_rows.append(_reference_manifest_row(source, pseudo, 'error', str(exc)))
            statuses.append('error')

    source_dirs = list(source_root.glob('*'))
    wdf = _concat_csv([p/'well_observations_partial.csv' for p in source_dirs])
    pdf = _concat_csv([p/'pdo_observations_partial.csv' for p in source_dirs])
    pscdf = _concat_csv([p/'psc_observations_partial.csv' for p in source_dirs])
    source_manifest = pd.DataFrame(manifest_rows)
    ldf = _longitudinal_large_table(wdf, pdf)

    source_manifest.to_csv(out/'csv'/'raw_source_manifest.csv', index=False)
    wdf.to_csv(out/'csv'/'large_well_observations.csv', index=False)
    pdf.to_csv(out/'csv'/'large_PDO_observations.csv', index=False)
    pscdf.to_csv(out/'csv'/'large_PSC_focus_observations.csv', index=False)
    ldf.to_csv(out/'csv'/'well_longitudinal_tracking.csv', index=False)
    run_status = {
        'sources_total': len(sources), 'sources_complete': statuses.count('complete'),
        'sources_partial': statuses.count('partial'), 'sources_error': statuses.count('error'),
        'all_complete': bool(statuses) and all(s == 'complete' for s in statuses),
    }
    _atomic_json(out/'run_status.json', run_status)

    ml_path = None
    if make_ml_export:
        ml_path = build_large_ml_export(out, source_manifest, wdf, pdf, pscdf, ldf, root)
    return root, out, source_manifest, wdf, pdf, pscdf, ldf, run_status, ml_path
