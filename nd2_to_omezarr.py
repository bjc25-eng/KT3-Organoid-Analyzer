from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import nd2
import numpy as np
import zarr
from numcodecs import Blosc


def _channel_names(metadata, count: int) -> list[str]:
    rows = []
    items = getattr(metadata, 'channels', None)
    if items is None and isinstance(metadata, dict):
        items = metadata.get('channels', [])
    for i, item in enumerate(items or []):
        ch = getattr(item, 'channel', item)
        if isinstance(ch, dict):
            name = ch.get('name')
        else:
            name = getattr(ch, 'name', None)
        rows.append(str(name or f'Channel {i}'))
    while len(rows) < count:
        rows.append(f'Channel {len(rows)}')
    return rows[:count]


def _voxel_size_um(f) -> tuple[float, float, float]:
    try:
        v = f.voxel_size()
        return float(v.x), float(v.y), float(v.z)
    except Exception:
        return 1.0, 1.0, 1.0


class NativeND2ChannelAccessor:
    """Small-region accessor for the current uncompressed Nikon ND2 layout.

    The `nd2` public Dask view can expose a huge whole-frame chunk.  For an
    uncompressed ND2 the underlying modern reader can memory-map the raw frame,
    which lets us slice small regions without materialising the whole image.
    This class normalises common internal frame layouts into channel/Y/X reads.
    """

    def __init__(self, f: nd2.ND2File):
        self.f = f
        self.sizes = {str(k).upper(): int(v) for k, v in dict(f.sizes).items()}
        self.channels = int(self.sizes.get('C', 1))
        self.height = int(self.sizes['Y'])
        self.width = int(self.sizes['X'])
        compression = getattr(f.attributes, 'compressionType', None)
        if compression not in (None, '', 'none', 'None'):
            raise RuntimeError(
                f'ND2 compression is {compression!r}. This low-memory converter is intended for uncompressed ND2 files.'
            )

        self.raw0 = f._rdr.read_frame(0)  # intentionally uses the memory-mapped native reader
        self.raw0_squeezed = np.squeeze(self.raw0)
        self.mode = self._detect_mode()

    def _detect_mode(self) -> str:
        a = self.raw0_squeezed
        if a.ndim == 2 and a.shape == (self.height, self.width):
            return 'separate_frames'
        if a.ndim == 3:
            if a.shape[0] == self.channels and a.shape[1:] == (self.height, self.width):
                return 'cyx'
            if a.shape[-1] == self.channels and a.shape[:2] == (self.height, self.width):
                return 'yxc'
        raise RuntimeError(
            'Unsupported native ND2 frame layout. '
            f'f.shape={tuple(self.f.shape)}, sizes={self.sizes}, native_frame_shape={tuple(self.raw0.shape)}, '
            f'squeezed_shape={tuple(a.shape)}.'
        )

    def describe(self) -> dict:
        return {
            'nd2_shape': list(self.f.shape),
            'sizes': self.sizes,
            'native_frame_shape': list(self.raw0.shape),
            'native_frame_squeezed_shape': list(self.raw0_squeezed.shape),
            'access_mode': self.mode,
        }

    def read(self, channel: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        channel = int(channel)
        if channel < 0 or channel >= self.channels:
            raise IndexError(channel)
        if self.mode == 'cyx':
            view = self.raw0_squeezed[channel, y0:y1, x0:x1]
        elif self.mode == 'yxc':
            view = self.raw0_squeezed[y0:y1, x0:x1, channel]
        else:
            frame = self.raw0_squeezed if channel == 0 else np.squeeze(self.f._rdr.read_frame(channel))
            if frame.shape != (self.height, self.width):
                raise RuntimeError(f'Channel {channel} frame has unexpected shape {frame.shape}.')
            view = frame[y0:y1, x0:x1]
        return np.asarray(view)


def convert(source: Path, output: Path, chunk: int = 1024, clevel: int = 3) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')

    started = time.time()
    with nd2.ND2File(source) as f:
        accessor = NativeND2ChannelAccessor(f)
        c = accessor.channels
        y = accessor.height
        x = accessor.width
        dtype = np.dtype(f.dtype)
        names = _channel_names(f.metadata, c)
        vx, vy, vz = _voxel_size_um(f)

        root = zarr.open_group(str(output), mode='w')
        compressor = Blosc(cname='zstd', clevel=int(clevel), shuffle=Blosc.BITSHUFFLE)
        arr = root.create_dataset(
            '0',
            shape=(c, y, x),
            chunks=(1, min(int(chunk), y), min(int(chunk), x)),
            dtype=dtype,
            compressor=compressor,
            overwrite=True,
        )

        root.attrs['multiscales'] = [{
            'version': '0.4',
            'name': source.stem,
            'axes': [
                {'name': 'c', 'type': 'channel'},
                {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
            ],
            'datasets': [{
                'path': '0',
                'coordinateTransformations': [{
                    'type': 'scale',
                    'scale': [1.0, float(vy), float(vx)],
                }],
            }],
        }]
        root.attrs['omero'] = {
            'name': source.stem,
            'channels': [
                {
                    'label': name,
                    'active': True,
                    'coefficient': 1.0,
                    'color': '00FF00' if 'gfp' in name.lower() or 'green' in name.lower() else 'FFFFFF',
                    'window': {'start': 0.0, 'end': float((1 << int(getattr(f.attributes, 'bitsPerComponentSignificant', 16) or 16)) - 1), 'min': 0.0, 'max': float(np.iinfo(dtype).max)},
                }
                for name in names
            ],
        }
        root.attrs['kt3_source'] = {
            'original_filename': source.name,
            'original_size_bytes': source.stat().st_size,
            'nd2_version': list(f.version),
            'channel_names': names,
            'voxel_size_um': {'x': vx, 'y': vy, 'z': vz},
            'native_access': accessor.describe(),
            'conversion': {
                'chunk_shape_cyx': [1, min(int(chunk), y), min(int(chunk), x)],
                'compressor': 'Blosc zstd bitshuffle',
                'compression_level': int(clevel),
                'preserves_source_dtype': True,
            },
        }

        total_chunks = c * math.ceil(y / chunk) * math.ceil(x / chunk)
        done = 0
        for ch in range(c):
            for y0 in range(0, y, chunk):
                y1 = min(y, y0 + chunk)
                for x0 in range(0, x, chunk):
                    x1 = min(x, x0 + chunk)
                    block = accessor.read(ch, y0, y1, x0, x1)
                    if block.dtype != dtype:
                        block = block.astype(dtype, copy=False)
                    arr[ch, y0:y1, x0:x1] = block
                    done += 1
                    if done == 1 or done % 25 == 0 or done == total_chunks:
                        elapsed = max(0.001, time.time() - started)
                        rate = done / elapsed
                        remaining = (total_chunks - done) / rate if rate > 0 else 0
                        print(
                            f'\rOME-Zarr: {done}/{total_chunks} chunks ({100.0*done/total_chunks:5.1f}%) '
                            f'elapsed={elapsed/60:.1f} min eta={remaining/60:.1f} min',
                            end='', flush=True,
                        )
        print()

    summary = {
        'source': str(source),
        'output': str(output),
        'source_size_bytes': source.stat().st_size,
        'output_size_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        'shape_cyx': [c, y, x],
        'dtype': str(dtype),
        'channel_names': names,
        'pixel_size_um': {'x': vx, 'y': vy},
        'chunk_shape_cyx': [1, min(int(chunk), y), min(int(chunk), x)],
        'elapsed_seconds': time.time() - started,
    }
    (output / 'conversion_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert an uncompressed, memory-mappable Nikon ND2 to chunked OME-Zarr without loading the full image into RAM.')
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--chunk', type=int, default=1024, help='Y/X chunk edge in pixels (default: 1024)')
    parser.add_argument('--clevel', type=int, default=3, choices=range(0, 10), help='Blosc zstd compression level (default: 3)')
    args = parser.parse_args()

    summary = convert(args.source, args.output, args.chunk, args.clevel)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
