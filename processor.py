from __future__ import annotations

"""S3-aware facade around the validated KT3 analysis core.

The scientific segmentation and measurement logic remains in ``analysis_core.py``.
This module adds AWS S3 I/O plus one-PDO-per-crop exports. For GFP datasets it
also applies the conservative PDO duplicate/outside-well QC implemented in
``pdo_qc.py`` before final exports are generated.
"""

import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import boto3
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from analysis_core import (
    APP_TITLE,
    BRIGHTFIELD_MODE,
    GFP_MODE,
    PSC_ABSENT,
    PSC_PRESENT,
    Settings,
    build_settings_from_widgets,
    process as core_process,
    process_experiment,
    zip_bytes,
)
from pdo_qc import rebuild_quantitative_outputs

SUPPORTED_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


class LocalUpload:
    """Make a local/S3-downloaded file look like Streamlit UploadedFile."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name = self.path.name

    def getbuffer(self):
        return memoryview(self.path.read_bytes())


def _font(size: int = 18, bold: bool = False):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _fixed_crop(image: Image.Image, cx: float, cy: float, size: int = 256) -> Image.Image:
    half = size // 2
    left = int(round(cx)) - half
    top = int(round(cy)) - half
    right, bottom = left + size, top + size
    canvas = Image.new('RGB', (size, size), 'black')
    sl, st = max(0, left), max(0, top)
    sr, sb = min(image.width, right), min(image.height, bottom)
    if sr > sl and sb > st:
        patch = image.crop((sl, st, sr, sb))
        canvas.paste(patch, (sl - left, st - top))
    return canvas


def _label_pdo_crop(crop: Image.Image, row: pd.Series) -> Image.Image:
    title = _font(18, True)
    body = _font(15, False)
    psc = row.get('PSC_like_focus_count_in_well', None)
    psc_text = 'not analysed' if pd.isna(psc) else str(int(psc))
    well = str(row['well_index'])
    n = int(row['PDO_number_in_well'])
    total = int(row['PDO_count_in_well'])
    ecd = float(row['equivalent_circular_diameter_um'])
    qc = str(row.get('membership_status', 'accepted'))
    lines = [
        f"Image {int(row['image_series']):02d} | Well {well} | PDO {n}/{total}",
        f"Equivalent circular diameter: {ecd:.1f} µm",
        f"PSC-like foci in well: {psc_text} | QC: {qc}",
    ]
    header = 92
    out = Image.new('RGB', (crop.width, crop.height + header), 'white')
    out.paste(crop, (0, header))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, out.width, header), fill='black')
    y = 7
    for i, text in enumerate(lines):
        f = title if i == 0 else body
        d.text((10, y), text, font=f, fill='white')
        y += 30 if i == 0 else 25
    return out


def _contact_sheet(paths: list[Path], out_path: Path, columns: int = 5, tile_width: int = 260, gap: int = 6):
    if not paths:
        return
    images = []
    for path in paths:
        im = Image.open(path).convert('RGB')
        scale = tile_width / im.width
        images.append(im.resize((tile_width, int(round(im.height * scale))), Image.Resampling.LANCZOS))
    cell_h = max(im.height for im in images)
    rows = math.ceil(len(images) / columns)
    sheet = Image.new('RGB', (columns * tile_width + (columns + 1) * gap,
                              rows * cell_h + (rows + 1) * gap), 'white')
    for i, im in enumerate(images):
        row, col = divmod(i, columns)
        sheet.paste(im, (gap + col * (tile_width + gap), gap + row * (cell_h + gap)))
    sheet.save(out_path, dpi=(300, 300))


def add_pdo_centred_exports(out_dir: str | Path, crop_size_px: int = 256, contact_columns: int = 5) -> pd.DataFrame:
    """Create exactly one raw/labelled crop per final accepted/retained PDO."""
    out_dir = Path(out_dir)
    pdo_csv = out_dir / 'csv' / 'PDO_raw_data.csv'
    well_csv = out_dir / 'csv' / 'well_raw_data.csv'
    if not pdo_csv.exists() or not well_csv.exists():
        return pd.DataFrame()

    pdo = pd.read_csv(pdo_csv)
    wells = pd.read_csv(well_csv)

    raw_dir = out_dir / 'pdo_centred_raw_crops'
    labelled_dir = out_dir / 'pdo_centred_labelled_crops'
    for d in [raw_dir, labelled_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(exist_ok=True)

    if pdo.empty:
        pdo.to_csv(out_dir / 'csv' / 'PDO_centred_raw_data.csv', index=False)
        contact = out_dir / 'figures' / 'PDO_centred_contact_sheet_compact.png'
        if contact.exists():
            contact.unlink()
        return pdo

    lookup = wells[[
        'image_series', 'source_image', 'well_index', 'well_col_index', 'well_row_index',
        'well_centre_x_px', 'well_centre_y_px', 'well_radius_px', 'um_per_pixel'
    ]].drop_duplicates(['image_series', 'well_index'])
    pdo = pdo.merge(lookup, on=['image_series', 'well_index'], how='left')

    labelled_paths: list[Path] = []
    raw_names, labelled_names = [], []
    image_cache: dict[tuple[int, str], Image.Image] = {}

    for _, row in pdo.iterrows():
        series = int(row['image_series'])
        source = str(row['source_image'])
        key = (series, source)
        if key not in image_cache:
            raw_path = out_dir / 'raw_images' / f'series_{series:02d}__{source}'
            image_cache[key] = Image.open(raw_path).convert('RGB')
        image = image_cache[key]
        crop = _fixed_crop(image, float(row['centroid_x_px']), float(row['centroid_y_px']), int(crop_size_px))
        well_slug = str(row['well_index']).replace(',', '_')
        pno = int(row['PDO_number_in_well'])
        stem = f'series_{series:02d}_well_{well_slug}_PDO_{pno:02d}'
        rp = raw_dir / f'{stem}.png'
        lp = labelled_dir / f'{stem}_labelled.png'
        crop.save(rp, dpi=(300, 300))
        _label_pdo_crop(crop, row).save(lp, dpi=(300, 300))
        raw_names.append(rp.name)
        labelled_names.append(lp.name)
        labelled_paths.append(lp)

    pdo['pdo_centred_raw_crop'] = raw_names
    pdo['pdo_centred_labelled_crop'] = labelled_names
    pdo['projected_area_um2'] = pdo['projected_area_px2'].astype(float) * (pdo['um_per_pixel'].astype(float) ** 2)
    pdo['distance_to_well_centre_px'] = ((pdo['centroid_x_px'] - pdo['well_centre_x_px']) ** 2 +
                                         (pdo['centroid_y_px'] - pdo['well_centre_y_px']) ** 2) ** 0.5
    pdo.to_csv(out_dir / 'csv' / 'PDO_centred_raw_data.csv', index=False)

    contact = out_dir / 'figures' / 'PDO_centred_contact_sheet_compact.png'
    if contact.exists():
        contact.unlink()
    _contact_sheet(labelled_paths, contact, columns=int(contact_columns))
    return pdo


def process(files, settings: Settings, cols: int = 5, create_pdo_centred: bool = True, crop_size_px: int = 256):
    """Run core analysis, then apply conservative GFP QC before final exports.

    Fixes applied for GFP data:
    - prevents intensity-only over-segmentation/duplicate PDOs;
    - rejects clear outside-microwell objects and flags ambiguous edge objects;
    - regenerates corrected well/PDO tables, figures and crops;
    - creates one PDO-centred crop only from the corrected PDO table.
    """
    root, out, summary, image_summary = core_process(files, settings, int(cols))

    if settings.organoid_mode == GFP_MODE:
        summary, image_summary = rebuild_quantitative_outputs(out, settings, cols=int(cols))

    if create_pdo_centred:
        add_pdo_centred_exports(out, crop_size_px=int(crop_size_px), contact_columns=int(cols))
    return root, out, summary, image_summary


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
    keys: list[str] = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get('Contents', []):
            key = item.get('Key', '')
            if Path(key).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                keys.append(key)
    return sorted(keys)


def download_s3_images(client, bucket: str, keys: Iterable[str], destination: str | Path) -> list[Path]:
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


def process_s3_batch(
    client,
    bucket: str,
    keys: list[str],
    settings: Settings,
    cols: int = 5,
    create_pdo_centred: bool = True,
    crop_size_px: int = 256,
):
    root = Path(tempfile.mkdtemp(prefix='kt3_s3_input_'))
    paths = download_s3_images(client, bucket, keys, root)
    uploads = [LocalUpload(p) for p in paths]
    return process(uploads, settings, int(cols), create_pdo_centred=create_pdo_centred,
                   crop_size_px=int(crop_size_px))


def upload_directory_to_s3(client, local_dir: str | Path, bucket: str, prefix: str) -> int:
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
