from __future__ import annotations

"""QC wrapper for the existing whole-array OME-Zarr S3 processor.

The OME-Zarr implementation already masks GFP signal to the inner 0.86× well
radius before segmentation, so outside-well fluorescence is excluded at source.
This wrapper additionally replaces the old intensity-peak PDO splitter with the
same conservative shape-confirmed splitter used by the individual-image route.
"""

import s3_omezarr as _base
from pdo_qc import segment_pdos_conservative


def list_s3_omezarr_datasets(client, bucket: str, prefix: str = '') -> list[str]:
    return _base.list_s3_omezarr_datasets(client, bucket, prefix)


def process_s3_omezarr(*args, **kwargs):
    # s3_omezarr imported segment_pdos into its module namespace. Rebinding that
    # symbol here means the existing tiled/streaming implementation is preserved
    # while duplicate/over-segmentation is fixed without duplicating its S3 code.
    _base.segment_pdos = segment_pdos_conservative
    return _base.process_s3_omezarr(*args, **kwargs)
