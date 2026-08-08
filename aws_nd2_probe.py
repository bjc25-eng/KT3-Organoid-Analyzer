from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from urllib.parse import urlparse

import nd2
import requests


def _safe(value):
    if is_dataclass(value):
        try:
            return asdict(value)
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _channels(metadata) -> list[dict]:
    rows = []
    items = getattr(metadata, 'channels', None)
    if items is None and isinstance(metadata, dict):
        items = metadata.get('channels', [])
    for fallback, item in enumerate(items or []):
        ch = getattr(item, 'channel', item)
        if isinstance(ch, dict):
            index = ch.get('index', fallback)
            name = ch.get('name', f'Channel {fallback}')
            excitation = ch.get('excitationLambdaNm')
            emission = ch.get('emissionLambdaNm')
        else:
            index = getattr(ch, 'index', fallback)
            name = getattr(ch, 'name', f'Channel {fallback}')
            excitation = getattr(ch, 'excitationLambdaNm', None)
            emission = getattr(ch, 'emissionLambdaNm', None)
        rows.append({
            'index': int(index) if index is not None else fallback,
            'name': str(name or f'Channel {fallback}'),
            'excitation_nm': None if excitation is None else float(excitation),
            'emission_nm': None if emission is None else float(emission),
        })
    return rows


def _suggest(channels: list[dict], kind: str) -> int | None:
    terms = (
        ('gfp', 'green', 'fitc', '488', 'egfp', 'fluorescein')
        if kind == 'gfp'
        else ('dic', 'brightfield', 'bright field', 'transmitted', 'transmission', 'dia', 'phase')
    )
    for row in channels:
        name = str(row.get('name', '')).lower()
        if any(term in name for term in terms):
            return int(row['index'])
    return None


def _positions(experiment) -> list[dict]:
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
            })
        break
    return rows


def _download_http(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length') or 0)
        free = shutil.disk_usage(destination.parent).free
        if total and free < total + (512 * 1024 * 1024):
            raise OSError(
                f'Not enough free scratch space: need about {(total + 512 * 1024 * 1024) / 1e9:.2f} GB, '
                f'have {free / 1e9:.2f} GB in {destination.parent}'
            )
        done = 0
        with destination.open('wb') as fh:
            for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\rDownloading ND2: {100.0 * done / total:5.1f}% ({done / 1e9:.2f}/{total / 1e9:.2f} GB)', end='', file=sys.stderr)
                else:
                    print(f'\rDownloading ND2: {done / 1e9:.2f} GB', end='', file=sys.stderr)
    print(file=sys.stderr)


def _download_s3(uri: str, destination: Path) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError('boto3 is required for s3:// URIs. Install it with: pip install boto3') from exc
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    if not bucket or not key:
        raise ValueError(f'Invalid S3 URI: {uri}')
    s3 = boto3.client('s3')
    head = s3.head_object(Bucket=bucket, Key=key)
    total = int(head.get('ContentLength') or 0)
    free = shutil.disk_usage(destination.parent).free
    if total and free < total + (512 * 1024 * 1024):
        raise OSError(
            f'Not enough free scratch space: need about {(total + 512 * 1024 * 1024) / 1e9:.2f} GB, '
            f'have {free / 1e9:.2f} GB in {destination.parent}'
        )
    done = 0

    def progress(n):
        nonlocal done
        done += int(n)
        if total:
            print(f'\rDownloading ND2: {100.0 * done / total:5.1f}% ({done / 1e9:.2f}/{total / 1e9:.2f} GB)', end='', file=sys.stderr)

    s3.download_file(bucket, key, str(destination), Callback=progress)
    print(file=sys.stderr)


def _materialise_source(source: str, scratch: Path) -> tuple[Path, bool]:
    source = source.strip()
    if source.startswith(('http://', 'https://')):
        dest = scratch / 'source.nd2'
        _download_http(source, dest)
        return dest, True
    if source.startswith('s3://'):
        dest = scratch / 'source.nd2'
        _download_s3(source, dest)
        return dest, True
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path, False


def probe_local_nd2(path: Path, position_index: int = 0) -> dict:
    with nd2.ND2File(path) as f:
        sizes = {str(k).upper(): int(v) for k, v in dict(f.sizes).items()}
        channels = _channels(f.metadata)
        positions = _positions(f.experiment)
        attrs = f.attributes
        try:
            voxel = f.voxel_size()
            voxel_um = {'x': float(voxel.x), 'y': float(voxel.y), 'z': float(voxel.z)}
        except Exception:
            voxel_um = {'x': None, 'y': None, 'z': None}
        dtype = f.dtype
        channel_count = int(sizes.get('C', max(1, len(channels))))
        position_count = int(sizes.get('P', max(1, len(positions))))
        if position_index < 0 or position_index >= position_count:
            raise IndexError(f'ND2 contains {position_count} XY position(s); position {position_index} is unavailable.')
        frame_bytes = int(sizes.get('X', 0) * sizes.get('Y', 0) * max(1, channel_count) * dtype.itemsize)
        return {
            'source_file': path.name,
            'source_file_size_bytes': path.stat().st_size,
            'nd2_version': list(f.version),
            'nd2_is_legacy': bool(f.is_legacy),
            'shape': list(f.shape),
            'sizes': sizes,
            'dtype': str(dtype),
            'significant_bits': getattr(attrs, 'bitsPerComponentSignificant', None),
            'width_px': int(sizes.get('X', 0)),
            'height_px': int(sizes.get('Y', 0)),
            'channel_count': channel_count,
            'channel_metadata': channels,
            'channel_names': [c['name'] for c in channels],
            'suggested_gfp_channel': _suggest(channels, 'gfp'),
            'suggested_dic_channel': _suggest(channels, 'dic'),
            'position_count': position_count,
            'selected_position_index': int(position_index),
            'xy_positions': positions,
            'voxel_size_um': voxel_um,
            'estimated_native_frame_bytes': frame_bytes,
            'estimated_native_frame_mib': frame_bytes / (1024 ** 2),
            'core_attributes': _safe(attrs),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description='Download/materialise one Nikon ND2 source and print a metadata probe as JSON.')
    parser.add_argument('source', help='Presigned https:// URL, s3://bucket/key URI, or local ND2 path')
    parser.add_argument('--position', type=int, default=0, help='XY position index to inspect (default: 0)')
    parser.add_argument('--output', type=Path, default=Path('nd2_probe.json'), help='JSON output path')
    parser.add_argument('--scratch-dir', type=Path, default=None, help='Optional scratch directory with enough free space for the ND2')
    parser.add_argument('--keep-source', action='store_true', help='Keep a downloaded source instead of deleting it after the probe')
    args = parser.parse_args()

    scratch_owner = args.scratch_dir is None
    # Avoid /tmp: small cloud instances can mount it as a limited tmpfs even
    # when the main EBS volume has ample free space.  Use a unique directory
    # on the user's home filesystem by default.
    scratch = (
        Path(tempfile.mkdtemp(prefix='nd2_probe_', dir=str(Path.home())))
        if scratch_owner
        else args.scratch_dir.expanduser().resolve()
    )
    scratch.mkdir(parents=True, exist_ok=True)
    local_path = None
    downloaded = False
    try:
        local_path, downloaded = _materialise_source(args.source, scratch)
        print(f'Local ND2: {local_path} ({local_path.stat().st_size / 1e9:.2f} GB)', file=sys.stderr)
        result = probe_local_nd2(local_path, args.position)
        result['original_source'] = 's3/object URL' if args.source.startswith(('http://', 'https://', 's3://')) else str(local_path)
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
        print(json.dumps(result, indent=2, default=str))
        print(f'\nSaved: {args.output.resolve()}', file=sys.stderr)
        return 0
    finally:
        if downloaded and local_path is not None and local_path.exists() and not args.keep_source:
            try:
                local_path.unlink()
            except OSError:
                pass
        if scratch_owner:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
