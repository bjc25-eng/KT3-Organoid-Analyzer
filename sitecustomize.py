"""Compatibility patch for reading private S3 OME-Zarr stores on Streamlit.

The app already constructs an authenticated boto3 client from Streamlit Secrets.
This module makes s3_omezarr reuse that exact client instead of creating a second
s3fs/aiobotocore credential stack. Scientific analysis code is unchanged.
"""

from __future__ import annotations

from collections.abc import MutableMapping


class Boto3ZarrStore(MutableMapping):
    """Read-only Zarr-v2 mapping backed by an existing boto3 S3 client."""

    def __init__(self, client, bucket: str, prefix: str):
        self.client = client
        self.bucket = str(bucket)
        self.prefix = str(prefix).strip('/')

    def _key(self, key: str) -> str:
        key = str(key).lstrip('/')
        return f"{self.prefix}/{key}" if self.prefix else key

    def __getitem__(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return obj['Body'].read()

    def __setitem__(self, key: str, value) -> None:
        raise TypeError('Boto3ZarrStore is read-only')

    def __delitem__(self, key: str) -> None:
        raise TypeError('Boto3ZarrStore is read-only')

    def __iter__(self):
        prefix = self.prefix.rstrip('/') + '/' if self.prefix else ''
        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get('Contents', []):
                full = item.get('Key', '')
                if full.startswith(prefix):
                    yield full[len(prefix):]

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def __contains__(self, key: object) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(str(key)))
            return True
        except Exception:
            return False


def _patched_open_s3_group(client, bucket: str, prefix: str, region: str):
    import zarr
    store = Boto3ZarrStore(client, bucket, prefix)
    return zarr.open_group(store=store, mode='r')


try:
    import s3_omezarr
    s3_omezarr._open_s3_group = _patched_open_s3_group
except Exception:
    # Do not prevent Streamlit startup if the optional OME-Zarr module itself
    # fails to import; the app will then display its normal analysis error.
    pass
