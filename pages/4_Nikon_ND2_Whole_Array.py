from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analysis_core import BRIGHTFIELD_MODE, GFP_MODE, PSC_ABSENT, build_settings_from_widgets, zip_bytes
from large_data_core import make_resume_bundle, restore_resume_bundle
from nd2_large_source import ND2_SOURCE_LABEL, install_nd2_dispatch, probe_nd2_source
from nd2_qc import process_large_experiment_qc

install_nd2_dispatch()

st.set_page_config(page_title='Nikon ND2 Whole-Array Imaging', page_icon='🟢', layout='wide')
st.title('Nikon ND2 Whole-Array Imaging')
st.caption(
    'Probe and analyse Nikon .nd2 microscopy data natively. The original ND2 is retained as the source; no PNG/TIFF conversion is required before analysis.'
)

with st.expander('How ND2 mode works', expanded=True):
    st.markdown('''
- The app reads Nikon `.nd2` metadata first: dimensions, channels, bit depth, physical pixel size, time/Z structure and XY positions.
- ND2 access is **lazy**: the complete multi-gigabyte dataset is not loaded into RAM.
- Nikon ND2 is natively frame-oriented. A crop request may still require decoding one complete underlying Y/X frame. The probe therefore reports **estimated native frame memory**.
- If the file contains many normal microscope XY positions, each position can be analysed independently and efficiently.
- If a single stitched ND2 frame itself is extremely large, the probe will flag it. In that case the recommended cloud workflow is a one-time conversion to chunked OME-Zarr before repeated analysis.
- For your DIC + GFP data, use **GFP-labelled PDOs**, turn PSC/RFP analysis off, set the GFP channel to the fluorescence channel, and set the dedicated well-detection channel to DIC.
- GFP PDOs use the same final QC logic as the validated OME-Zarr route: conservative shape-supported splitting, full-object segmentation, physical well-radius membership, segmented-object overlap, DIC wall evidence, ambiguous-edge classification and DIC microwell-validity QC.
''')

st.warning(
    'On the hosted app, a multi-gigabyte ND2 should normally be stored in S3 or another range-readable/private object store and supplied via a time-limited URL. Do not upload a 3.5 GB ND2 through the Streamlit uploader.'
)

st.subheader('1. Probe one ND2 file')
probe_uri = st.text_input('ND2 source URI / path', '', placeholder='https://.../experiment.nd2 or local path on the compute worker')
probe_position = st.number_input('XY position to inspect', min_value=0, value=0, step=1)

if st.button('Probe ND2 metadata', disabled=not probe_uri.strip()):
    try:
        meta = probe_nd2_source(probe_uri.strip(), int(probe_position))
        st.session_state['nd2_probe_meta'] = meta
        st.success(
            f"ND2 detected — {meta['width_px']:,} × {meta['height_px']:,} px per position, "
            f"{meta['channel_count']} channel(s), {meta['position_count']} XY position(s)."
        )
    except Exception as exc:
        st.error(f'Could not probe ND2: {exc}')

meta = st.session_state.get('nd2_probe_meta')
if meta:
    a, b, c, d = st.columns(4)
    a.metric('XY positions', int(meta.get('position_count', 1)))
    b.metric('Channels', int(meta.get('channel_count', 1)))
    bpp = meta.get('significant_bits') or meta.get('dtype', '')
    c.metric('Bit depth / dtype', str(bpp))
    d.metric('Native frame', f"{float(meta.get('estimated_native_frame_mib', 0)):.1f} MiB")

    channels = pd.DataFrame(meta.get('channel_metadata', []))
    if len(channels):
        st.markdown('**Detected channels**')
        st.dataframe(channels, use_container_width=True, hide_index=True)
    voxel = meta.get('voxel_size_um', {}) or {}
    st.caption(
        f"Physical pixel size: X={voxel.get('x')} µm, Y={voxel.get('y')} µm. "
        f"Suggested GFP channel: {meta.get('suggested_gfp_channel')}; suggested DIC/brightfield channel: {meta.get('suggested_dic_channel')}."
    )
    if meta.get('frame_memory_warning'):
        st.warning(meta['frame_memory_warning'])
    with st.expander('Full ND2 metadata probe'):
        st.json(meta)

st.divider()
st.subheader('2. Configure one ND2 analysis / benchmark source')
a, b, c, d = st.columns(4)
with a:
    experiment_id = st.text_input('Experiment ID', 'ND2_Benchmark_001')
with b:
    device_id = st.text_input('Array / device ID', 'Array_001')
with c:
    replicate_id = st.text_input('Biological replicate ID', 'Replicate_1')
with d:
    pdo_model = st.text_input('PDO model / patient / line', '')

a, b, c, d = st.columns(4)
with a:
    condition = st.text_input('Condition', 'Benchmark')
with b:
    field_id = st.text_input('Field ID', 'F01')
with c:
    timepoint = st.text_input('Time point', 'Day 0')
with d:
    elapsed = st.number_input('Elapsed days', min_value=0.0, value=0.0)

st.markdown('**ND2 source and native dimensions**')
a, b, c = st.columns(3)
with a:
    source_uri = st.text_input('ND2 source for analysis', probe_uri if probe_uri else '', key='nd2_analysis_uri')
with b:
    position_index = st.number_input('ND2 XY position index', min_value=0, value=int(probe_position), step=1, key='nd2_pos')
with c:
    internal_t = st.number_input('Internal ND2 time index', min_value=0, value=0, step=1)

st.markdown('**DIC + GFP channel mapping**')
suggested_gfp = int(meta['suggested_gfp_channel']) if meta and meta.get('suggested_gfp_channel') is not None else 1
suggested_dic = int(meta['suggested_dic_channel']) if meta and meta.get('suggested_dic_channel') is not None else 0
a, b, c = st.columns(3)
with a:
    gfp_channel = st.number_input('GFP channel index', min_value=0, value=max(0, suggested_gfp), step=1)
with b:
    dic_channel = st.number_input('DIC / well-detection channel index', min_value=0, value=max(0, suggested_dic), step=1)
with c:
    z_index = st.number_input('Z plane index', min_value=0, value=0, step=1)

st.info('PSC/RFP analysis is disabled in this ND2 workflow because the current source type is DIC + GFP only.')

st.subheader('3. Detection, QC and memory settings')
a, b, c = st.columns(3)
with a:
    well_diameter = st.number_input('Microwell diameter (µm)', min_value=1.0, value=100.0, step=1.0)
with b:
    tile_size = st.selectbox('Well-scan tile size (px)', [1024, 1536, 2048, 3072, 4096], index=2)
with c:
    standard_crop = st.selectbox('ML crop size (px)', [128, 224, 256, 320, 512], index=2)

split = st.checkbox('Split touching PDOs', True)
exclude_ambiguous = st.checkbox(
    'Exclude ambiguous wall-touching PDO candidates',
    False,
    help='Leave off for the first validation run. Ambiguous candidates will be retained but flagged in PDO_candidate_QC.csv.',
)
st.caption(
    'Final-QC mode uses the physical pixel size stored in the ND2 metadata. It will stop with an explicit error if valid X/Y calibration is unavailable rather than estimating µm/px from detected Hough circles.'
)

with st.expander('Advanced detection thresholds'):
    rmin = st.number_input('Minimum well radius (px)', 5, 1000, 23, step=1)
    rmax = st.number_input('Maximum well radius (px)', 6, 2000, 40, step=1)
    spacing = st.number_input('Minimum well spacing (px)', 10, 5000, 54, step=1)
    hp2 = st.number_input('Well detection sensitivity', 1.0, 100.0, 27.0, 1.0)
    gl = st.number_input('GFP PDO low threshold', 0.0, 255.0, 30.0, 1.0)
    gh = st.number_input('GFP PDO high threshold', 0.0, 255.0, 45.0, 1.0)
    amin = st.number_input('Minimum GFP PDO area (px²)', 1, 1000000, 20, step=1)
    pdist = st.number_input('PDO split peak distance (px)', 1, 1000, 18, step=1)

settings = build_settings_from_widgets(
    well_diameter, rmin, rmax, spacing, hp2, gl, gh, amin, split, pdist,
    9.0, 12.0, 4, 12, 10.0, 80,
    organoid_mode=GFP_MODE, rfp_psc_present=False
)
settings.exclude_ambiguous_edge_candidates = bool(exclude_ambiguous)

# We map DIC to the dedicated well-detection channel and GFP to green. Red/blue
# are absent so -1 gives a clean black background for the GFP composite.
channel_config = {
    'red_channel': -1,
    'green_channel': int(gfp_channel),
    'blue_channel': -1,
    'brightfield_channel': int(dic_channel),
    'well_detection_channel': int(dic_channel),
    'z_index': int(z_index),
    'internal_t_index': int(internal_t),
}

st.subheader('4. Resume / checkpointing')
resume_file = st.file_uploader('Optional previous ND2 resume bundle (.zip)', type=['zip'], accept_multiple_files=False)
if resume_file is not None and st.session_state.get('nd2_resume_name') != resume_file.name:
    try:
        restored = restore_resume_bundle(resume_file.getvalue())
        st.session_state['nd2_work_root'] = str(restored)
        st.session_state['nd2_resume_name'] = resume_file.name
        st.success('Resume bundle restored. Matching tile/well detection work will be reused. Per-well measurements from an older QC schema are automatically recomputed.')
    except Exception as exc:
        st.error(f'Could not restore resume bundle: {exc}')

source = {
    'experiment_id': experiment_id,
    'device_id': device_id,
    'biological_replicate_id': replicate_id,
    'pdo_model': pdo_model,
    'condition_index': 1,
    'condition': condition,
    'organoid_mode': GFP_MODE,
    'rfp_psc_present': False,
    'drug_or_therapeutic': '',
    'concentration': 0.0,
    'concentration_unit': '',
    'timepoint_index': 1,
    'timepoint': timepoint,
    'elapsed_time': float(elapsed),
    'time_unit': 'days',
    'field_id': field_id,
    'source_uri': source_uri.strip(),
    'source_type': ND2_SOURCE_LABEL,
    # Existing whole-array engine calls this `series_index`; the ND2 dispatch
    # intentionally interprets it as Nikon XY-position index.
    'series_index': int(position_index),
    'pyramid_level': 0,
    'source_sha256': '',
    'compute_full_sha256': False,
}

run = st.button('Run / resume native ND2 analysis', type='primary', use_container_width=True, disabled=not source_uri.strip())
if run:
    progress_bar = st.progress(0, text='Opening ND2 lazily…')
    status = st.empty()

    def progress(done, total, phase):
        frac = int(min(99, max(1, 100 * done / max(1, total))))
        label = 'Scanning DIC for microwells' if phase == 'well_scan' else 'Analysing GFP-positive PDOs with final QC'
        progress_bar.progress(frac, text=f'{label}: {done:,} / {total:,}')
        status.caption('Incremental checkpoints are being written throughout the run.')

    try:
        root, out, manifest, wdf, pdf, pscdf, tracking, run_status, ml_path = process_large_experiment_qc(
            [source], settings, channel_config,
            tile_size=int(tile_size), standard_crop_size=int(standard_crop),
            work_root=st.session_state.get('nd2_work_root'), progress_callback=progress,
            make_ml_export=True,
        )
        st.session_state['nd2_work_root'] = str(root)
        st.session_state['nd2_results_zip'] = zip_bytes(out)
        st.session_state['nd2_resume_zip'] = make_resume_bundle(root)
        st.session_state['nd2_ml_zip'] = zip_bytes(ml_path) if ml_path is not None else None
        st.session_state['nd2_manifest'] = manifest.to_dict('records')
        st.session_state['nd2_run_status'] = run_status
        progress_bar.progress(100, text='ND2 final-QC analysis pass complete.')
        status.empty()
        if run_status.get('all_complete'):
            st.success('Native ND2 analysis completed with final PDO/well QC.')
        else:
            st.warning('The run contains an incomplete/error source. Download the resume bundle before troubleshooting or restarting.')
    except Exception as exc:
        progress_bar.empty(); status.empty()
        st.error(f'ND2 analysis stopped: {exc}')
        root = st.session_state.get('nd2_work_root')
        if root and Path(root).exists():
            try:
                st.session_state['nd2_resume_zip'] = make_resume_bundle(root)
            except Exception:
                pass

if st.session_state.get('nd2_run_status'):
    st.divider()
    st.subheader('ND2 outputs')
    run_status = st.session_state['nd2_run_status']
    a, b, c, d = st.columns(4)
    a.metric('Outside-well rejected', int(run_status.get('qc_rejected_outside_well_candidates', 0)))
    b.metric('Ambiguous candidates', int(run_status.get('qc_ambiguous_PDO_candidates', 0)))
    c.metric('False-well candidates rejected', int(run_status.get('qc_rejected_false_well_candidates', 0)))
    d.metric('False detected wells', int(run_status.get('qc_rejected_false_wells', 0)))

    a, b, c = st.columns(3)
    with a:
        if st.session_state.get('nd2_results_zip'):
            st.download_button('Download results ZIP', st.session_state['nd2_results_zip'], 'ND2_whole_array_results.zip', 'application/zip', use_container_width=True)
    with b:
        if st.session_state.get('nd2_ml_zip'):
            st.download_button('Download ML export ZIP', st.session_state['nd2_ml_zip'], 'ND2_ML_virtual_model_export.zip', 'application/zip', type='primary', use_container_width=True)
    with c:
        if st.session_state.get('nd2_resume_zip'):
            st.download_button('Download resume bundle', st.session_state['nd2_resume_zip'], 'ND2_resume_bundle.zip', 'application/zip', use_container_width=True)
    with st.expander('Raw ND2 source manifest / provenance', expanded=True):
        st.dataframe(pd.DataFrame(st.session_state.get('nd2_manifest', [])), use_container_width=True, hide_index=True)

st.caption(
    'Native ND2 remains the archival source. Validate one representative ~3 GB DIC+GFP ND2 first; after QC review, apply the unchanged settings to the remaining arrays.'
)
