from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis_core import GFP_MODE, Settings, zip_bytes
from nd2_omezarr import probe_omezarr
from omezarr_nd2_bridge import calibrated_scan_settings, infer_channel_indices

st.set_page_config(page_title='Background Converted ND2', page_icon='⚙️', layout='wide')
st.title('Background Converted ND2 Analysis')
st.caption(
    'Launch the converted whole-array analysis as a detached EC2 process. Once started, '
    'the job continues even if this browser disconnects or Streamlit restarts.'
)

cache_root = st.text_input('Persistent ND2 cache root', '/home/ec2-user/kt3_nd2_cache')
root = Path(cache_root).expanduser()
markers = sorted(root.glob('*/converted/*.conversion.json')) if root.exists() else []
entries = []
for marker in markers:
    for zarr_path in sorted(marker.parent.glob('*.ome.zarr')):
        entries.append((f'{zarr_path.name} — {zarr_path.parent.parent.name}', zarr_path))

if not entries:
    st.warning('No validated converted OME-Zarr dataset was found.')
    st.stop()

label = st.selectbox('Validated converted dataset', [q[0] for q in entries])
_, zarr_path = entries[[q[0] for q in entries].index(label)]
meta = probe_omezarr(zarr_path)
channels = infer_channel_indices(meta)

a, b, c, d = st.columns(4)
a.metric('Width', f"{int(meta['width_px']):,} px")
b.metric('Height', f"{int(meta['height_px']):,} px")
c.metric('Channels', int(meta['channel_count']))
d.metric('Chunk decode', str(meta.get('chunk_decode_test', 'unknown')))

st.subheader('1. Experiment metadata')
a, b, c, d = st.columns(4)
with a:
    experiment_id = st.text_input('Experiment ID', 'K3T_RMC6236')
with b:
    device_id = st.text_input('Array / device ID', 'Array_001')
with c:
    replicate_id = st.text_input('Biological replicate ID', 'Replicate_1')
with d:
    pdo_model = st.text_input('PDO model / patient / line', 'K3T')

a, b, c, d = st.columns(4)
with a:
    condition = st.text_input('Condition', 'DMSO')
with b:
    field_id = st.text_input('Field ID', 'F01')
with c:
    timepoint = st.text_input('Time point', 'Day 0')
with d:
    elapsed = st.number_input('Elapsed days', min_value=0.0, value=0.0)

st.subheader('2. Analysis settings')
a, b, c, d = st.columns(4)
with a:
    well_diameter = st.number_input('Microwell diameter (µm)', min_value=1.0, value=100.0, step=1.0)
with b:
    tile_size = st.selectbox('Well-scan tile size (px)', [1024, 1536, 2048, 3072], index=2)
with c:
    standard_crop = st.selectbox('ML crop size (px)', [128, 224, 256, 320, 512], index=2)
with d:
    hough_p2 = st.number_input('Well detection sensitivity', 1.0, 100.0, 27.0, 1.0)

split = st.checkbox('Split touching PDOs', True)
exclude_ambiguous = st.checkbox('Exclude ambiguous wall-touching PDO candidates', False)

settings = Settings(
    well_diameter_um=float(well_diameter),
    well_rmin=23,
    well_rmax=40,
    well_spacing=54,
    hough_p2=float(hough_p2),
    green_low=30.0,
    green_high=45.0,
    pdo_min_area=20,
    split_pdos=bool(split),
    pdo_peak_distance=18,
    psc_peak_threshold=9.0,
    psc_red_minimum=12.0,
    psc_peak_distance=4,
    histogram_bins=12,
    organoid_mode=GFP_MODE,
    rfp_psc_present=True,
    brightfield_contrast_threshold=10.0,
    brightfield_min_area=80,
)
settings.exclude_ambiguous_edge_candidates = bool(exclude_ambiguous)
settings, expected_radius_px, umpp = calibrated_scan_settings(settings, meta)
st.caption(
    f'{umpp:.6f} µm/px → expected well radius {expected_radius_px:.2f} px; '
    f'Hough radius {settings.well_rmin}–{settings.well_rmax} px; tile size {int(tile_size)} px.'
)

source = {
    'experiment_id': experiment_id,
    'device_id': device_id,
    'biological_replicate_id': replicate_id,
    'pdo_model': pdo_model,
    'condition_index': 1,
    'condition': condition,
    'organoid_mode': GFP_MODE,
    'rfp_psc_present': True,
    'drug_or_therapeutic': '',
    'concentration': 0.0,
    'concentration_unit': '',
    'timepoint_index': 1,
    'timepoint': timepoint,
    'elapsed_time': float(elapsed),
    'time_unit': 'days',
    'field_id': field_id,
    'source_uri': str(zarr_path),
    'source_type': 'OME-Zarr',
    'series_index': 0,
    'pyramid_level': 0,
    'source_sha256': '',
    'compute_full_sha256': False,
}
channel_config = {
    'red_channel': int(channels['rfp']),
    'green_channel': int(channels['gfp']),
    'blue_channel': -1,
    'brightfield_channel': int(channels['dic']),
    'well_detection_channel': int(channels['dic']),
    'z_index': 0,
    'internal_t_index': 0,
}

work_root = zarr_path.parent.parent / 'omezarr_analysis_work'
job_root = work_root / 'background_job'
config_path = job_root / 'job_config.json'
status_path = job_root / 'job_status.json'
log_path = job_root / 'worker.log'
pid_path = job_root / 'worker.pid'
worker_script = Path(__file__).resolve().parents[1] / 'converted_nd2_background_worker.py'


def read_status() -> dict:
    if not status_path.exists():
        return {'state': 'not_started'}
    try:
        return json.loads(status_path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'state': 'unknown', 'message': f'Could not read job status: {exc!s}'}


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


st.subheader('3. Persistent background job')
st.caption(f'Checkpoint root: {work_root}')
status = read_status()
state = str(status.get('state', 'not_started'))
known_pid = status.get('pid')
alive = pid_alive(known_pid)

if state == 'running' and not alive:
    st.warning(
        'The previous worker is no longer running, but its analysis checkpoints are intact. '
        'Press the button below once to start a new detached worker; it will resume from those checkpoints.'
    )
elif state == 'running' and alive:
    done = int(status.get('done', 0) or 0)
    total = int(status.get('total', 0) or 0)
    frac = min(1.0, max(0.0, done / max(1, total))) if total else 0.0
    st.success(f'Background worker is running (PID {known_pid}). You may close this browser.')
    st.progress(frac, text=f"{status.get('message', 'Running')} {done:,} / {total:,}" if total else status.get('message', 'Running'))
elif state == 'completed':
    st.success(
        f"Completed: {int(status.get('well_count', 0)):,} wells, "
        f"{int(status.get('pdo_count', 0)):,} PDO observations, "
        f"{int(status.get('psc_focus_count', 0)):,} PSC/RFP foci."
    )
elif state == 'failed':
    st.error(f"Previous worker failed: {status.get('message', 'unknown error')}")
else:
    st.info('No background worker is currently running.')

start_label = 'Resume in detached background worker' if state in {'running', 'failed'} else 'Start detached whole-array analysis'
start_disabled = bool(state == 'running' and alive)
if st.button(start_label, type='primary', use_container_width=True, disabled=start_disabled):
    job_root.mkdir(parents=True, exist_ok=True)
    config = {
        'source': source,
        'settings': vars(settings),
        'channel_config': channel_config,
        'tile_size': int(tile_size),
        'standard_crop_size': int(standard_crop),
        'work_root': str(work_root),
        'status_path': str(status_path),
    }
    config_path.write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    log_handle = open(log_path, 'ab', buffering=0)
    proc = subprocess.Popen(
        [sys.executable, str(worker_script), str(config_path)],
        cwd=str(worker_script.parent),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    pid_path.write_text(str(proc.pid), encoding='utf-8')
    st.success(
        f'Background analysis started as PID {proc.pid}. It will continue independently of this browser and Streamlit session.'
    )
    st.rerun()

if st.button('Refresh job status', use_container_width=True):
    st.rerun()

if log_path.exists():
    with st.expander('Background worker log'):
        try:
            lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
            st.code('\n'.join(lines[-80:]) or 'Log is currently empty.')
        except Exception as exc:
            st.caption(f'Could not read worker log: {exc!s}')

status = read_status()
if status.get('state') == 'completed':
    out = Path(str(status.get('output_dir', '')))
    ml_path = Path(str(status.get('ml_path'))) if status.get('ml_path') else None
    if out.exists():
        st.download_button(
            'Download analysis results ZIP',
            zip_bytes(out),
            file_name='converted_nd2_analysis_results.zip',
            mime='application/zip',
            use_container_width=True,
        )
        wells_csv = out / 'csv' / 'large_well_observations.csv'
        if wells_csv.exists():
            try:
                wdf = pd.read_csv(wells_csv)
                st.markdown('**Well observations preview**')
                st.dataframe(wdf.head(50), hide_index=True, use_container_width=True)
            except Exception:
                pass
    if ml_path is not None and ml_path.exists():
        st.download_button(
            'Download ML/QC export ZIP',
            zip_bytes(ml_path),
            file_name='converted_nd2_ml_qc_export.zip',
            mime='application/zip',
            use_container_width=True,
        )
