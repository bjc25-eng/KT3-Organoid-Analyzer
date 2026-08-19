from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analysis_core import GFP_MODE, build_settings_from_widgets, zip_bytes
from nd2_omezarr import probe_omezarr
from omezarr_nd2_bridge import (
    calibrated_scan_settings,
    infer_channel_indices,
    process_converted_nd2_omezarr_qc,
)

st.set_page_config(page_title='Analyze Converted ND2', page_icon='🔬', layout='wide')
st.title('Analyze Converted ND2 OME-Zarr')
st.caption(
    'Run the validated microwell/PDO final-QC pipeline on the chunked OME-Zarr copy, '
    'using the physical calibration and GFP/RFP/DIC channel metadata preserved from the Nikon ND2.'
)

cache_root = st.text_input('Persistent ND2 cache root', '/home/ec2-user/kt3_nd2_cache')
root = Path(cache_root).expanduser()
markers = sorted(root.glob('*/converted/*.conversion.json')) if root.exists() else []

if not markers:
    st.warning('No validated ND2 → OME-Zarr conversion markers were found. Complete conversion first.')
    st.stop()

entries = []
for marker in markers:
    candidates = sorted(marker.parent.glob('*.ome.zarr'))
    for zarr_path in candidates:
        entries.append((f'{zarr_path.name} — {zarr_path.parent.parent.name}', zarr_path, marker))

if not entries:
    st.warning('A conversion marker exists, but no converted .ome.zarr dataset was found beside it.')
    st.stop()

label = st.selectbox('Validated converted dataset', [q[0] for q in entries])
_, zarr_path, marker_path = entries[[q[0] for q in entries].index(label)]

try:
    meta = probe_omezarr(zarr_path)
except Exception as exc:
    st.error(f'Could not open converted OME-Zarr: {type(exc).__name__}: {exc!s}')
    st.stop()

channels = infer_channel_indices(meta)
voxel = meta.get('voxel_size_um') or {}
st.success(f'Ready: {zarr_path}')
a, b, c, d = st.columns(4)
a.metric('Width', f"{int(meta['width_px']):,} px")
b.metric('Height', f"{int(meta['height_px']):,} px")
c.metric('Channels', int(meta['channel_count']))
d.metric('Chunk decode', str(meta.get('chunk_decode_test', 'unknown')))
st.caption(
    f"Physical pixel size: X={voxel.get('x')} µm, Y={voxel.get('y')} µm · "
    f"chunks={meta.get('level0_chunks')}"
)
if meta.get('channel_metadata'):
    st.dataframe(pd.DataFrame(meta['channel_metadata']), hide_index=True, use_container_width=True)

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

st.subheader('2. Channel mapping')
st.info('Detected from converted metadata. Check these once before the first run.')
a, b, c = st.columns(3)
with a:
    gfp_channel = st.number_input('GFP PDO channel', min_value=0, value=int(channels['gfp']), step=1)
with b:
    rfp_channel = st.number_input('RFP PSC/stromal channel', min_value=0, value=int(channels['rfp']), step=1)
with c:
    dic_channel = st.number_input('DIC / well channel', min_value=0, value=int(channels['dic']), step=1)

st.subheader('3. Detection and final-QC settings')
a, b, c = st.columns(3)
with a:
    well_diameter = st.number_input('Microwell diameter (µm)', min_value=1.0, value=100.0, step=1.0)
with b:
    tile_size = st.selectbox('Well-scan tile size (px)', [1024, 1536, 2048, 3072], index=0)
with c:
    standard_crop = st.selectbox('ML crop size (px)', [128, 224, 256, 320, 512], index=2)

split = st.checkbox('Split touching PDOs', True)
exclude_ambiguous = st.checkbox(
    'Exclude ambiguous wall-touching PDO candidates',
    False,
    help='For the first validation run, leave this off so ambiguous candidates are retained but flagged.',
)

with st.expander('Advanced detection thresholds'):
    st.caption('Physical calibration overrides well radius and spacing. The remaining thresholds stay active.')
    rmin = st.number_input('Legacy minimum well radius (overridden)', 5, 1000, 23, step=1)
    rmax = st.number_input('Legacy maximum well radius (overridden)', 6, 2000, 40, step=1)
    spacing = st.number_input('Legacy minimum well spacing (overridden)', 10, 5000, 54, step=1)
    hp2 = st.number_input('Well detection sensitivity', 1.0, 100.0, 27.0, 1.0)
    gl = st.number_input('GFP PDO low threshold', 0.0, 255.0, 30.0, 1.0)
    gh = st.number_input('GFP PDO high threshold', 0.0, 255.0, 45.0, 1.0)
    amin = st.number_input('Minimum GFP PDO area (px²)', 1, 1000000, 20, step=1)
    pdist = st.number_input('PDO split peak distance (px)', 1, 1000, 18, step=1)
    pt = st.number_input('PSC focus threshold', 0.0, 255.0, 9.0, 0.5)
    prm = st.number_input('PSC red-minus-blue minimum', 0.0, 255.0, 12.0, 0.5)
    ppd = st.number_input('PSC focus minimum spacing (px)', 1, 1000, 4, step=1)

settings = build_settings_from_widgets(
    well_diameter, rmin, rmax, spacing, hp2, gl, gh, amin, split, pdist,
    pt, prm, ppd, 12, 10.0, 80,
    organoid_mode=GFP_MODE, rfp_psc_present=True,
)
settings.exclude_ambiguous_edge_candidates = bool(exclude_ambiguous)
settings, expected_radius_px, umpp = calibrated_scan_settings(settings, meta)
st.caption(
    f'Physical scan geometry: {umpp:.6f} µm/px → expected 100-µm-well radius '
    f'{expected_radius_px:.2f} px; Hough radius {settings.well_rmin}–{settings.well_rmax} px; '
    f'minimum centre spacing {settings.well_spacing} px.'
)

channel_config = {
    'red_channel': int(rfp_channel),
    'green_channel': int(gfp_channel),
    'blue_channel': -1,
    'brightfield_channel': int(dic_channel),
    'well_detection_channel': int(dic_channel),
    'z_index': 0,
    'internal_t_index': 0,
}

work_root = zarr_path.parent.parent / 'omezarr_analysis_work'
st.subheader('4. Run / resume')
st.caption(f'Persistent checkpoint root: {work_root}')

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

run = st.button('Run / resume converted whole-array analysis', type='primary', use_container_width=True)
if run:
    progress_bar = st.progress(0, text='Opening chunked OME-Zarr…')
    status = st.empty()

    def progress(done, total, phase):
        frac = int(min(99, max(1, 100 * done / max(1, total))))
        label = 'Scanning DIC tiles for microwells' if phase == 'well_scan' else 'Analysing GFP PDOs + RFP PSCs with final QC'
        progress_bar.progress(frac, text=f'{label}: {done:,} / {total:,}')
        status.caption('Persistent checkpoints are being written after each tile/well.')

    try:
        root_out, out, manifest, wdf, pdf, pscdf, tracking, run_status, ml_path = process_converted_nd2_omezarr_qc(
            [source],
            settings,
            channel_config,
            tile_size=int(tile_size),
            standard_crop_size=int(standard_crop),
            work_root=str(work_root),
            progress_callback=progress,
            make_ml_export=True,
        )
        progress_bar.progress(100, text='Whole-array analysis complete.')
        status.empty()
        st.success(
            f"Complete: {len(wdf):,} wells, {len(pdf):,} PDO observations, "
            f"{len(pscdf):,} PSC/RFP foci."
        )
        st.session_state['converted_nd2_results_zip'] = zip_bytes(out)
        st.session_state['converted_nd2_ml_zip'] = zip_bytes(ml_path) if ml_path is not None else None
        st.dataframe(wdf.head(50), use_container_width=True, hide_index=True)
        if not pdf.empty:
            st.markdown('**PDO observations preview**')
            st.dataframe(pdf.head(50), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f'Analysis failed: {type(exc).__name__}: {exc!s}')

if st.session_state.get('converted_nd2_results_zip'):
    st.download_button(
        'Download analysis results ZIP',
        st.session_state['converted_nd2_results_zip'],
        file_name='converted_nd2_analysis_results.zip',
        mime='application/zip',
        use_container_width=True,
    )
if st.session_state.get('converted_nd2_ml_zip'):
    st.download_button(
        'Download ML/QC export ZIP',
        st.session_state['converted_nd2_ml_zip'],
        file_name='converted_nd2_ml_qc_export.zip',
        mime='application/zip',
        use_container_width=True,
    )
