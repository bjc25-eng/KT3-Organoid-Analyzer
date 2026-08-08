from __future__ import annotations

"""S3-aware façade around the validated KT3 analysis core.

The scientific segmentation and measurement logic remains in ``analysis_core.py``.
This module adds AWS S3 I/O without duplicating or changing that logic.
"""

import io
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import boto3

from analysis_core import (
    APP_TITLE,
    BRIGHTFIELD_MODE,
    GFP_MODE,
    PSC_ABSENT,
    PSC_PRESENT,
    Settings,
    build_settings_from_widgets,
    process,
    process_experiment,
    zip_bytes,
)

SUPPORTED_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


class LocalUpload:
    """Small adapter that makes a local/S3-downloaded file look like Streamlit UploadedFile."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name = self.path.name

    def getbuffer(self):
        return memoryview(self.path.read_bytes())


def get_s3_client(
    region_name: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
):
    """Create an S3 client using explicit values or the normal AWS credential chain."""
    kwargs = {'region_name': region_name or os.getenv('AWS_DEFAULT_REGION', 'eu-west-2')}
    if access_key_id and secret_access_key:
        kwargs['aws_access_key_id'] = access_key_id
        kwargs['aws_secret_access_key'] = secret_access_key
        if session_token:
            kwargs['aws_session_token'] = session_token
    return boto3.client('s3', **kwargs)


def list_s3_images(client, bucket: str, prefix: str = '') -> list[str]:
    """List supported microscopy image keys below an S3 prefix."""
    keys: list[str] = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get('Contents', []):
            key = item.get('Key', '')
            if Path(key).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                keys.append(key)
    return sorted(keys)


def download_s3_images(client, bucket: str, keys: Iterable[str], destination: str | Path) -> list[Path]:
    """Download selected S3 microscopy images and return local file paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    used: set[str] = set()
    for i, key in enumerate(keys, 1):
        name = Path(key).name
        if name in used:
            name = f'{i:03d}_{name}'
        used.add(name)
        path = destination / name
        client.download_file(bucket, key, str(path))
        paths.append(path)
    return paths


def process_s3_batch(client, bucket: str, keys: list[str], settings: Settings, cols: int = 5):
    """Download an S3 batch and run the unchanged validated analysis core."""
    import tempfile

    root = Path(tempfile.mkdtemp(prefix='kt3_s3_input_'))
    paths = download_s3_images(client, bucket, keys, root)
    uploads = [LocalUpload(p) for p in paths]
    return process(uploads, settings, int(cols))


def upload_directory_to_s3(client, local_dir: str | Path, bucket: str, prefix: str) -> int:
    """Upload every file in an analysis result directory to S3."""
    local_dir = Path(local_dir)
    prefix = prefix.strip('/')
    count = 0
    for path in local_dir.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f'{prefix}/{rel}' if prefix else rel
        client.upload_file(str(path), bucket, key)
        count += 1
    return count


def upload_bytes_to_s3(client, payload: bytes, bucket: str, key: str, content_type: str = 'application/zip') -> None:
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def make_run_id(label: str = '') -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    label = re.sub(r'[^A-Za-z0-9._-]+', '_', str(label).strip()).strip('._-')[:80]
    return f'{stamp}_{label}' if label else stamp


def result_prefix(base_prefix: str, label: str = '') -> str:
    base = str(base_prefix).strip('/')
    run = make_run_id(label)
    return f'{base}/{run}' if base else run


def presigned_download_url(client, bucket: str, key: str, expires_seconds: int = 3600) -> str:
    return client.generate_presigned_url(
        'get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=int(expires_seconds)
    )


def archive_results_to_s3(
    client,
    out_dir: str | Path,
    bucket: str,
    base_prefix: str = 'results/',
    run_label: str = '',
    zip_name: str = 'KT3_PDO_PSC_analysis_results.zip',
) -> dict:
    """Persist the full result tree plus one ZIP in S3 and return their locations."""
    out_dir = Path(out_dir)
    prefix = result_prefix(base_prefix, run_label)
    file_count = upload_directory_to_s3(client, out_dir, bucket, prefix)
    payload = zip_bytes(out_dir)
    zip_key = f'{prefix}/{zip_name}'
    upload_bytes_to_s3(client, payload, bucket, zip_key)
    return {
        'bucket': bucket,
        'prefix': prefix,
        'zip_key': zip_key,
        'uploaded_result_files': file_count,
        'presigned_url': presigned_download_url(client, bucket, zip_key, 3600),
    }
