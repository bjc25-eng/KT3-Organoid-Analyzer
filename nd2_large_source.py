from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fsspec
import nd2
import numpy as np

from large_data_core import LargeImageReader as TiffZarrLargeImageReader

ND2_SOURCE_LABEL = 'Nikon ND2'


def _protocol(fs) -> str:
    value = getattr(fs, 'protocol', '')
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ''
    return str(value or '')


def _normalise_info(info: dict | None) -> dict:
    info = dict(info or {})
    result = {}
    for key in ('size', 'etag', 'ETag', 'last_modified', 'LastModified', 'mtime', 'version_id', 'VersionId'):
        if key not in info or info[key] is None:
            continue
        value = info[key]
        if hasattr(value, 'isoformat'):
            value = value.isoformat()
        result[key] = value if isinstance(value, (int, float, bool)) else str(value)
    return result


def _fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _safe_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        try:
            return asdict(value)
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): _safe_dataclass(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_dataclass(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _channel_records(metadata: Any) -> list[dict]:
    channels = getattr(metadata, 'channels', None)
    if channels is None and isinstance(metadata, dict):
        channels = metadata.get('channels', [])
    rows = []
    for fallback, item in enumerate(channels or []):
        channel = getattr(item, 'channel', item)
        if isinstance(channel, dict):
            name = channel.get('name', f'Channel {fallback}')
            index = channel.get('index', fallback)
            emission = channel.get('emissionLambdaNm')
            excitation = channel.get('excitationLambdaNm')
            color = channel.get('color') or channel.get('colorRGB')
        else:
            name = getattr(channel, 'name', f'Channel {fallback}')
            index = getattr(channel, 'index', fallback)
            emission = getattr(channel, 'emissionLambdaNm', None)
            excitation = getattr(channel, 'excitationLambdaNm', None)
            color = getattr(channel, 'color', None)
        rows.append({
            'index': int(index) if index is not None else fallback,
            'name': str(name or f'Channel {fallback}'),
            'emission_nm': None if emission is None else float(emission),
            'excitation_nm': None if excitation is None else float(excitation),
            'display_color': repr(color) if color is not None else '',
        })
    return rows


def _suggest_channel(channels: list[dict], kind: str) -> int | None:
    if not channels:
        return None
    if kind == 'gfp':
        terms = ('gfp', 'green', 'fitc', '488', 'egfp', 'fluorescein')
    else:
        terms = ('dic', 'brightfield', 'bright field', 'transmitted', 'transmission', 'dia', 'phase')
    for row in channels:
        name = str(row.get('name', '')).lower()
        if any(term in name for term in terms):
            return int(row['index'])
    return None


def _position_records(experiment: Any) -> list[dict]:
    rows = []
    for loop in experiment or []:
        loop_type = str(getattr(loop, 'type', loop.__class__.__name__))
        if 'XYPosLoop' not in loop_type:
            continue
        params = getattr(loop, 'parameters', None)
        points = getattr(params, 'points', []) if params is not None else []
        for i, point in enumerate(points):
            stage = getattr(point, 'stagePositionUm', None)
            if hasattr(stage, 'x'):
                xyz = [getattr(stage, 'x', None), getattr(stage, 'y', None), getattr(stage, 'z', None)]
            elif isinstance(stage, (list, tuple)):
                xyz = list(stage) + [None, None, None]
            else:
                xyz = [None, None, None]
            rows.append({
                'position_index': i,
                'name': str(getattr(point, 'name', '') or ''),
                'stage_x_um': xyz[0],
                'stage_y_um': xyz[1],
                'stage_z_um': xyz[2],
                'pfs_offset': getattr(point, 'pfsOffset', None),
            })
        break
    return rows


def _scale_uint8(arr: np.ndarray, significant_bits: int | None = None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    work = arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        if significant_bits and significant_bits > 0:
            denom = float((1 << int(significant_bits)) - 1)
        else:
            info = np.iinfo(arr.dtype)
            if info.min < 0:
                work = work - float(info.min)
                denom = float(info.max - info.min)
            else:
                denom = float(info.max) if info.max else 1.0
        return np.clip(work * (255.0 / max(denom, 1.0)), 0, 255).astype(np.uint8)
    finite = work[np.isfinite(work)]
    if not finite.size:
        return np.zeros(work.shape, dtype=np.uint8)
    if float(np.nanmin(finite)) >= 0.0 and float(np.nanmax(finite)) <= 1.0:
        work = work * 255.0
    return np.clip(work, 0, 255).astype(np.uint8)


class ND2LargeImageReader:
    """Native, lazy Nikon ND2 reader compatible with the whole-array engine.

    `series_index` is intentionally interpreted as the Nikon XY-position index so
    it can plug into the existing whole-array processing interface without a file
    conversion step.

    ND2's native delayed chunks are complete image frames in Y/X. Consequently a
    subregion request avoids materialising the whole ND2 dataset, but may still
    require decoding one complete underlying frame. The metadata probe reports an
    estimated frame size so unusually large stitched frames can be identified.
    """

    def __init__(self, uri: str, source_type: str = 'auto', series_index: int = 0, level: int = 0):
        self.uri = str(uri).strip()
        self.source_type_requested = source_type
        self.position_index = max(0, int(series_index))
        self.level = int(level)
        self._fs = None
        self._fs_path = None
        self._handle = None
        self._nd2 = None
        self._lazy = None
        self.source_info = {}
        self.axes: list[str] = []
        self.shape: tuple[int, ...] = ()
        self.dtype = None
        self.format = ND2_SOURCE_LABEL
        self.is_bigtiff = False
        self.is_ome = False
        self.significant_bits: int | None = None
        self.channels: list[dict] = []
        self.positions: list[dict] = []
        self._open()

    def _open(self):
        try:
            fs, path = fsspec.core.url_to_fs(self.uri)
            self._fs, self._fs_path = fs, path
            try:
                self.source_info = _normalise_info(fs.info(path))
            except Exception:
                self.source_info = {}
            protocol = _protocol(fs)
            kwargs = {}
            if protocol in {'http', 'https'}:
                kwargs = {'block_size': 8 * 1024 * 1024, 'cache_type': 'readahead'}
            self._handle = fs.open(path, 'rb', **kwargs)
            self._nd2 = nd2.ND2File(self._handle)
            sizes = dict(self._nd2.sizes)
            self.axes = [str(k).upper() for k in sizes]
            self.shape = tuple(int(v) for v in sizes.values())
            self.dtype = np.dtype(self._nd2.dtype)
            attrs = self._nd2.attributes
            self.significant_bits = getattr(attrs, 'bitsPerComponentSignificant', None)
            self.channels = _channel_records(self._nd2.metadata)
            self.positions = _position_records(self._nd2.experiment)
            if self.position_index >= self.position_count:
                raise IndexError(
                    f'ND2 contains {self.position_count} XY position(s); position {self.position_index} is unavailable.'
                )
            # ND2 creates chunks of one coordinate frame and a complete Y/X frame.
            # Keeping this lazy prevents materialising the entire multi-GB dataset.
            self._lazy = self._nd2.to_dask(wrapper=True, copy=True)
        except Exception:
            self.close()
            raise

    @property
    def width(self) -> int:
        return int(dict(zip(self.axes, self.shape))['X'])

    @property
    def height(self) -> int:
        return int(dict(zip(self.axes, self.shape))['Y'])

    @property
    def channel_count(self) -> int:
        sizes = dict(zip(self.axes, self.shape))
        if 'C' in sizes:
            return int(sizes['C'])
        if 'S' in sizes:
            return int(sizes['S'])
        return max(1, len(self.channels))

    @property
    def position_count(self) -> int:
        sizes = dict(zip(self.axes, self.shape))
        if 'P' in sizes:
            return int(sizes['P'])
        return max(1, len(self.positions))

    @property
    def voxel_size_um(self) -> dict:
        try:
            v = self._nd2.voxel_size()
            return {'x': float(v.x), 'y': float(v.y), 'z': float(v.z)}
        except Exception:
            return {'x': None, 'y': None, 'z': None}

    @property
    def estimated_native_frame_bytes(self) -> int:
        # nd2's delayed reader chunks complete native frames in Y/X and all
        # frame components/channels held in the raw frame shape.
        components = max(1, self.channel_count)
        sizes = dict(zip(self.axes, self.shape))
        if 'S' in sizes and 'C' in sizes:
            components *= int(sizes['S'])
        return int(self.width * self.height * components * self.dtype.itemsize)

    def metadata(self) -> dict:
        try:
            attrs = _safe_dataclass(self._nd2.attributes)
        except Exception:
            attrs = {}
        meta = {
            'uri': self.uri,
            'format': ND2_SOURCE_LABEL,
            'is_bigtiff': False,
            'is_ome': False,
            'nd2_version': list(getattr(self._nd2, 'version', ()) or ()),
            'nd2_is_legacy': bool(getattr(self._nd2, 'is_legacy', False)),
            'shape': list(self.shape),
            'axes': self.axes,
            'sizes': dict(zip(self.axes, self.shape)),
            'dtype': str(self.dtype),
            'significant_bits': self.significant_bits,
            'width_px': self.width,
            'height_px': self.height,
            'channel_count': self.channel_count,
            'channel_metadata': self.channels,
            'channel_names': [c['name'] for c in self.channels],
            'suggested_gfp_channel': _suggest_channel(self.channels, 'gfp'),
            'suggested_dic_channel': _suggest_channel(self.channels, 'dic'),
            'position_count': self.position_count,
            'selected_position_index': self.position_index,
            'xy_positions': self.positions,
            'voxel_size_um': self.voxel_size_um,
            'estimated_native_frame_bytes': self.estimated_native_frame_bytes,
            'estimated_native_frame_mib': self.estimated_native_frame_bytes / (1024 ** 2),
            'native_read_granularity': (
                'Lazy ND2 access is frame-based: the full ND2 dataset is not loaded, but a cropped request may decode '
                'one complete underlying Y/X frame before the requested region is returned.'
            ),
            'series_index': self.position_index,
            'position_index': self.position_index,
            'pyramid_level': 0,
            'core_attributes': attrs,
            **self.source_info,
        }
        if self.estimated_native_frame_bytes > 1024 ** 3:
            meta['frame_memory_warning'] = (
                'One native ND2 frame is estimated to exceed 1 GiB. For cloud analysis, a one-time ND2→OME-Zarr '
                'conversion is likely safer and more efficient than repeatedly decoding that frame.'
            )
        meta['reference_fingerprint_sha256'] = _fingerprint(meta)
        return meta

    def _selector(self, x0: int, y0: int, x1: int, y1: int, channel: int,
                  z_index: int, t_index: int):
        selector = []
        remaining = []
        for axis, size in zip(self.axes, self.shape):
            if axis == 'Y':
                selector.append(slice(y0, y1)); remaining.append('Y')
            elif axis == 'X':
                selector.append(slice(x0, x1)); remaining.append('X')
            elif axis == 'P':
                selector.append(min(self.position_index, int(size)-1))
            elif axis == 'T':
                selector.append(min(max(0, int(t_index)), int(size)-1))
            elif axis == 'Z':
                selector.append(min(max(0, int(z_index)), int(size)-1))
            elif axis == 'C':
                if channel < 0 or channel >= int(size):
                    return None, []
                selector.append(int(channel))
            elif axis == 'S':
                # If S is the only channel-like dimension, use the requested
                # index. If C is also present, S is an RGB component and we use
                # the first component for scalar channel analysis.
                if 'C' not in self.axes:
                    if channel < 0 or channel >= int(size):
                        return None, []
                    selector.append(int(channel))
                else:
                    selector.append(0)
            else:
                selector.append(0)
        return tuple(selector), remaining

    def read_channel_region(self, x0: int, y0: int, width: int, height: int, channel: int = 0,
                            z_index: int = 0, t_index: int = 0) -> np.ndarray:
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(self.width, x0 + max(1, int(width)))
        y1 = min(self.height, y0 + max(1, int(height)))
        selector, remaining = self._selector(x0, y0, x1, y1, int(channel), z_index, t_index)
        if selector is None:
            return np.zeros((y1-y0, x1-x0), dtype=np.uint8)
        region = self._lazy[selector]
        if hasattr(region, 'compute'):
            region = region.compute(scheduler='synchronous')
        arr = np.asarray(region).squeeze()
        if arr.ndim != 2:
            raise RuntimeError(f'ND2 region returned shape {arr.shape}; expected one 2D channel plane.')
        if remaining == ['X', 'Y']:
            arr = arr.T
        return _scale_uint8(arr, self.significant_bits)

    def read_rgb_region(self, x0: int, y0: int, width: int, height: int,
                        red_channel: int = 0, green_channel: int = 1, blue_channel: int = 2,
                        z_index: int = 0, t_index: int = 0) -> np.ndarray:
        h = min(self.height, max(0, int(y0)) + int(height)) - max(0, int(y0))
        w = min(self.width, max(0, int(x0)) + int(width)) - max(0, int(x0))
        channels = []
        for ch in (red_channel, green_channel, blue_channel):
            if ch is None or int(ch) < 0 or int(ch) >= self.channel_count:
                channels.append(np.zeros((h, w), dtype=np.uint8))
            else:
                channels.append(self.read_channel_region(x0, y0, width, height, int(ch), z_index, t_index))
        return np.stack(channels, axis=-1)

    def close(self):
        if self._nd2 is not None:
            try:
                self._nd2.close()
            except Exception:
                pass
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
        self._lazy = None
        self._nd2 = None
        self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class DispatchLargeImageReader:
    """Factory-compatible reader that adds ND2 without changing TIFF/Zarr code."""

    def __new__(cls, uri: str, source_type: str = 'auto', series_index: int = 0, level: int = 0):
        requested = str(source_type or 'auto').strip().lower()
        clean_uri = str(uri).lower().split('?', 1)[0]
        if requested in {'nikon nd2', 'nd2'} or clean_uri.endswith('.nd2'):
            return ND2LargeImageReader(uri, source_type=source_type, series_index=series_index, level=level)
        return TiffZarrLargeImageReader(uri, source_type=source_type, series_index=series_index, level=level)


def install_nd2_dispatch() -> None:
    """Install ND2 support into the existing whole-array engine at runtime."""
    import large_data_core
    large_data_core.LargeImageReader = DispatchLargeImageReader


def probe_nd2_source(uri: str, position_index: int = 0) -> dict:
    with ND2LargeImageReader(uri, source_type=ND2_SOURCE_LABEL, series_index=position_index) as reader:
        return reader.metadata()
