from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

import large_data_core as ldc
from analysis_core import GFP_MODE, Settings


def _rgb_source(path: Path, width=180, height=160):
    rgb = np.full((height, width, 3), 35, dtype=np.uint8)
    cv2.circle(rgb, (90, 80), 12, (35, 235, 35), -1)
    cv2.circle(rgb, (103, 80), 3, (245, 35, 35), -1)
    tifffile.imwrite(path, rgb, photometric='rgb', tile=(32, 32), metadata={'axes': 'YXS'})
    return rgb


def _source(path: Path):
    return {
        'source_uri': str(path),
        'source_type': 'TIFF',
        'series_index': 0,
        'pyramid_level': 0,
        'experiment_id': 'EXP_A',
        'device_id': 'ARRAY_1',
        'biological_replicate_id': 'R1',
        'pdo_model': 'PDO_A',
        'condition_index': 1,
        'condition': 'Vehicle',
        'organoid_mode': GFP_MODE,
        'rfp_psc_present': True,
        'drug_or_therapeutic': 'Vehicle',
        'concentration': 0.0,
        'concentration_unit': 'nM',
        'timepoint_index': 1,
        'timepoint': 'Day 0',
        'elapsed_time': 0.0,
        'time_unit': 'days',
        'field_id': 'F01',
        'source_sha256': '',
        'compute_full_sha256': False,
    }


def test_large_reader_reads_only_requested_region(tmp_path):
    path = tmp_path/'source.tif'
    rgb = _rgb_source(path)
    with ldc.LargeImageReader(str(path), source_type='TIFF') as reader:
        assert reader.width == rgb.shape[1]
        assert reader.height == rgb.shape[0]
        assert reader.channel_count == 3
        tile = reader.read_rgb_region(70, 60, 40, 35, 0, 1, 2)
        assert tile.shape == (35, 40, 3)
        assert np.array_equal(tile, rgb[60:95, 70:110])
        meta = reader.metadata()
        assert meta['reference_fingerprint_sha256']
        assert len(meta['reference_fingerprint_sha256']) == 64
        assert meta['width_px'] == rgb.shape[1]
        assert meta['height_px'] == rgb.shape[0]


def test_streaming_sha256_matches_file_content(tmp_path):
    path = tmp_path/'source.tif'
    _rgb_source(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ldc.compute_streaming_sha256(str(path), chunk_bytes=1024) == expected


def test_scan_wells_resume_skips_completed_tiles(monkeypatch, tmp_path):
    class FakeReader:
        width = 1024
        height = 512

    calls = []

    def fake_read(reader, x0, y0, w, h, config, organoid_mode):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[0, 0, 0] = min(255, x0 // 4)
        return arr, arr, arr

    def fake_detect(arr, settings):
        calls.append(1)
        # One local well in the centre of every expanded read. The core ownership
        # rule plus deduplication leaves one detection per core tile.
        h, w = arr.shape[:2]
        return np.array([[w//2, h//2, 25]], dtype=int)

    monkeypatch.setattr(ldc, '_read_analysis_region', fake_read)
    monkeypatch.setattr(ldc, 'detect_wells', fake_detect)
    settings = Settings(well_rmax=40)
    wells, checkpoint = ldc.scan_wells_tiled(
        FakeReader(), settings, {}, tmp_path, 'fingerprint', GFP_MODE, tile_size=512
    )
    first_calls = len(calls)
    assert checkpoint.exists()
    assert first_calls == 2

    calls.clear()
    wells2, _ = ldc.scan_wells_tiled(
        FakeReader(), settings, {}, tmp_path, 'fingerprint', GFP_MODE, tile_size=512
    )
    assert len(calls) == 0
    assert np.array_equal(wells, wells2)


def test_large_source_analysis_preserves_full_resolution_coordinates(monkeypatch, tmp_path):
    path = tmp_path/'source.tif'
    _rgb_source(path)

    def fixed_scan(reader, settings, config, work_dir, source_fingerprint, organoid_mode, tile_size, progress_callback=None):
        cp = Path(work_dir)/'tile_scan_checkpoint.json'
        cp.write_text(json.dumps({'source_fingerprint': source_fingerprint, 'completed_tiles': ['0_0'], 'wells': [[90,80,25]]}))
        return np.array([[90, 80, 25]], dtype=int), cp

    monkeypatch.setattr(ldc, 'scan_wells_tiled', fixed_scan)
    settings = Settings(green_low=20, green_high=50, pdo_min_area=20, psc_peak_threshold=5, psc_red_minimum=20)
    result = ldc.analyse_large_source(
        _source(path), settings,
        {'red_channel':0, 'green_channel':1, 'blue_channel':2, 'brightfield_channel':-1, 'well_detection_channel':-1},
        tmp_path/'work', tile_size=512, standard_crop_size=256
    )
    assert result['complete'] is True
    wells = pd.read_csv(tmp_path/'work'/'well_observations_partial.csv')
    assert len(wells) == 1
    assert int(wells.iloc[0]['well_centre_x_px_fullres']) == 90
    assert int(wells.iloc[0]['well_centre_y_px_fullres']) == 80
    assert wells.iloc[0]['trajectory_id'] == 'EXP_A__ARRAY_1__L01__F01__W1_1'
    crop_path = tmp_path/'work'/wells.iloc[0]['standard_rgb_crop']
    assert crop_path.exists()
    assert Image.open(crop_path).size == (256, 256)
    pdo = pd.read_csv(tmp_path/'work'/'pdo_observations_partial.csv')
    assert len(pdo) >= 1
    assert abs(float(pdo.iloc[0]['centroid_x_px_fullres']) - 90) < 4
    assert abs(float(pdo.iloc[0]['centroid_y_px_fullres']) - 80) < 4


def test_large_experiment_ml_export_references_raw_source_without_copy(monkeypatch, tmp_path):
    path = tmp_path/'source.tif'
    _rgb_source(path)

    def fixed_scan(reader, settings, config, work_dir, source_fingerprint, organoid_mode, tile_size, progress_callback=None):
        cp = Path(work_dir)/'tile_scan_checkpoint.json'
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({'source_fingerprint': source_fingerprint, 'completed_tiles': ['0_0'], 'wells': [[90,80,25]]}))
        return np.array([[90,80,25]], dtype=int), cp

    monkeypatch.setattr(ldc, 'scan_wells_tiled', fixed_scan)
    settings = Settings(green_low=20, green_high=50, pdo_min_area=20)
    root, out, manifest, wdf, pdf, pscdf, ldf, status, ml = ldc.process_large_experiment(
        [_source(path)], settings,
        {'red_channel':0, 'green_channel':1, 'blue_channel':2, 'brightfield_channel':-1, 'well_detection_channel':-1},
        tile_size=512, standard_crop_size=256, make_ml_export=True
    )
    assert status['all_complete'] is True
    assert ml is not None and Path(ml).exists()
    raw_manifest = pd.read_csv(Path(ml)/'tables'/'raw_source_manifest.csv')
    assert raw_manifest.iloc[0]['source_uri'] == str(path)
    assert bool(raw_manifest.iloc[0]['raw_image_copied_into_export']) is False
    assert len(str(raw_manifest.iloc[0]['reference_fingerprint_sha256'])) == 64
    assert not any(p.name == path.name for p in Path(ml).rglob('*') if p.is_file())
    assert (Path(ml)/'assets'/'well_crops_256').exists()
    assert len(ldf) == 1
    assert ldf.iloc[0]['trajectory_id'] == 'EXP_A__ARRAY_1__L01__F01__W1_1'


def test_resume_bundle_roundtrip(tmp_path):
    root = tmp_path/'run'
    (root/'sources'/'A').mkdir(parents=True)
    (root/'sources'/'A'/'well_analysis_checkpoint.json').write_text('{"completed_wells":["x"]}')
    payload = ldc.make_resume_bundle(root)
    restored = ldc.restore_resume_bundle(payload)
    assert (restored/'sources'/'A'/'well_analysis_checkpoint.json').exists()
