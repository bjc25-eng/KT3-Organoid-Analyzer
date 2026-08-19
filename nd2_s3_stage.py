from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import boto3
from botocore.exceptions import ClientError


def get_s3_client(
    region_name: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
):
    """Create an S3 client, falling back to the normal AWS credential chain.

    On EC2 this means an instance profile can be used without storing static keys.
    Streamlit secrets can still be supplied explicitly by the caller.
    """
    kwargs = {}
    if region_name:
        kwargs['region_name'] = region_name
    if access_key_id:
        kwargs['aws_access_key_id'] = access_key_id
    if secret_access_key:
        kwargs['aws_secret_access_key'] = secret_access_key
    if session_token:
        kwargs['aws_session_token'] = session_token
    return boto3.client('s3', **kwargs)


def list_nd2_objects(client, bucket: str, prefix: str = '') -> list[dict]:
    """List ND2 objects under an S3 prefix without downloading them."""
    bucket = str(bucket or '').strip()
    prefix = str(prefix or '').lstrip('/')
    if not bucket:
        return []

    rows: list[dict] = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get('Contents', []) or []:
            key = str(item.get('Key', ''))
            if not key.lower().endswith('.nd2'):
                continue
            rows.append({
                'key': key,
                'size': int(item.get('Size', 0) or 0),
                'etag': str(item.get('ETag', '') or '').strip('"'),
                'last_modified': item.get('LastModified'),
            })
    rows.sort(key=lambda r: r['key'])
    return rows


def _object_token(bucket: str, key: str, etag: str = '') -> str:
    raw = f'{bucket}\n{key}\n{etag}'.encode('utf-8')
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_paths(cache_root: str | Path, bucket: str, key: str, etag: str = '') -> tuple[Path, Path]:
    """Return deterministic local ND2 and work/checkpoint paths for an S3 object."""
    root = Path(cache_root).expanduser().resolve()
    token = _object_token(bucket, key, etag)
    stem = Path(key).name
    object_root = root / token
    return object_root / 'input' / stem, object_root / 'work'


def stage_s3_nd2(
    client,
    bucket: str,
    key: str,
    cache_root: str | Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Download one S3 ND2 once to local persistent storage.

    A completed file is reused when its size matches the current S3 object. A
    partial download is written to ``.part`` and never mistaken for a valid ND2.
    """
    head = client.head_object(Bucket=bucket, Key=key)
    size = int(head.get('ContentLength', 0) or 0)
    etag = str(head.get('ETag', '') or '').strip('"')
    local_path, work_root = cache_paths(cache_root, bucket, key, etag)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size == size:
        if progress_callback:
            progress_callback(size, size)
        return {
            'local_path': str(local_path),
            'work_root': str(work_root),
            'size': size,
            'etag': etag,
            'reused': True,
            's3_uri': f's3://{bucket}/{key}',
        }

    part_path = local_path.with_suffix(local_path.suffix + '.part')
    if part_path.exists():
        part_path.unlink()

    transferred = 0

    def _progress(delta: int):
        nonlocal transferred
        transferred += int(delta)
        if progress_callback:
            progress_callback(min(transferred, size), size)

    try:
        client.download_file(bucket, key, str(part_path), Callback=_progress)
        if part_path.stat().st_size != size:
            raise IOError(
                f'S3 download size mismatch for {key}: expected {size} bytes, got {part_path.stat().st_size}.'
            )
        os.replace(part_path, local_path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    return {
        'local_path': str(local_path),
        'work_root': str(work_root),
        'size': size,
        'etag': etag,
        'reused': False,
        's3_uri': f's3://{bucket}/{key}',
    }


def upload_tree(client, local_root: str | Path, bucket: str, prefix: str) -> dict:
    """Upload all files below a local directory to an S3 result prefix."""
    root = Path(local_root)
    if not root.exists():
        raise FileNotFoundError(root)
    prefix = str(prefix or '').strip('/')
    uploaded = 0
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        key = f'{prefix}/{rel}' if prefix else rel
        client.upload_file(str(path), bucket, key)
        uploaded += 1
    return {'bucket': bucket, 'prefix': prefix, 'uploaded_files': uploaded}
