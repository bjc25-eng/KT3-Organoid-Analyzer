from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from analysis_core import (
    BRIGHTFIELD_MODE,
    GFP_MODE,
    PSC_ABSENT,
    PSC_PRESENT,
    build_settings_from_widgets,
    zip_bytes,
)
from large_data_core import (
    DEFAULT_STANDARD_CROP_SIZE,
    DEFAULT_TILE_SIZE,
    LargeSourceError,
    make_resume_bundle,
    probe_large_source,
    process_large_experiment,
    restore_resume_bundle,
)

st.set_page_config(page_title='Large Dataset / Whole-Array Imaging', page_icon='🧩', layout='wide')
st.title('Large Dataset / Whole-Array Imaging')
st.caption(
    'Analyse multi-gigabyte OME-TIFF, BigTIFF and OME-Zarr sources by reading small regions instead of loading the entire image into RAM.'
)

with st.expander('How large-source mode works', expanded=False):
    st.markdown('''
- **Do not upload the 3–5 GB source through Streamlit.** Enter a file reference instead: normally a public/signed HTTPS URL or an accessible OME-Zarr location.
- The analyzer scans the source in **overlapping tiles** for microwells, then reads only a small region around each well for PDO/PSC analysis.
- All reported `*_fullres` coordinates are source-image pixel coordinates, not tile coordinates.
- The ML export contains **standardized per-well crops and masks**, but does **not duplicate the giant raw file**.
- `reference_fingerprint_sha256` fingerprints the source URI + remote metadata + image metadata. It is not presented as a content hash. If you already have a file SHA-256, enter it; optional full-file hashing can also be requested for TIFF, but that streams the entire source and can be slow/expensive.
- Checkpoints are written after each scanned tile and each analysed well. A downloadable **resume bundle** can be re-uploaded later. The free hosted Streamlit filesystem itself is not durable across a hard server restart, so keeping the resume bundle is important for very long runs.
''')

st.warning(
    'Large TIFFs served over HTTP need a server that supports byte-range requests. Private datasets should use a time-limited signed URL rather than placing credentials in the app.'
)

# ------------------------- experiment identity -------------------------
st.subheader('1. Experiment identity')
a, b, c, d = st.columns(4)
with a:
    experiment_id = st.text_input('Experiment ID', 'Experiment_001', key='large_exp')
with b:
    device_id = st.text_input('Array / device ID', 'Array_001', key='large_device')
with c:
    replicate_id = st.text_input('Biological replicate ID', 'Replicate_1', key='large_rep')
with d:
    pdo_model = st.text_input('PDO model / patient / line', '', key='large_model')

# ------------------------- layout metadata -------------------------
st.subheader('2. Conditions and time points')
a, b, c = st.columns(3)
with a:
    n_conditions = st.selectbox('Number of lanes / conditions', list(range(1, 13)), index=5, key='large_ncond')
with b:
    n_timepoints = st.selectbox('Number of imaging time points', list(range(1, 13)), index=3, key='large_ntp')
with c:
    time_unit = st.selectbox('Time unit', ['hours', 'days'], index=1, key='large_timeunit')

condition_meta = []
with st.expander('Lane / condition metadata', expanded=True):
    for i in range(n_conditions):
        st.markdown(f'**Lane {i+1}**')
        a, b, c = st.columns([1.3, 1.0, 1.0])
        with a:
            name = st.text_input('Condition name', f'Condition {i+1}', key=f'large_name_{i}')
        with b:
            omode = st.selectbox('Organoid detection', [GFP_MODE, BRIGHTFIELD_MODE], key=f'large_omode_{i}')
        with c:
            pmode = st.selectbox('RFP stromal cells', [PSC_PRESENT, PSC_ABSENT], key=f'large_pmode_{i}')
        d, e, f = st.columns([1.2, 0.8, 0.8])
        with d:
            drug = st.text_input('Drug / therapeutic', '', key=f'large_drug_{i}')
        with e:
            conc = st.number_input('Concentration', min_value=0.0, value=0.0, format='%.6g', key=f'large_conc_{i}')
        with f:
            unit = st.selectbox('Unit', ['nM', 'µM', 'ng/mL', 'µg/mL', 'other'], key=f'large_unit_{i}')
        condition_meta.append({
            'condition_index': i+1,
            'condition': name,
            'organoid_mode': omode,
            'rfp_psc_present': pmode == PSC_PRESENT,
            'drug_or_therapeutic': drug,
            'concentration': float(conc),
            'concentration_unit': unit,
        })
        if i < n_conditions - 1:
            st.divider()

time_meta = []
with st.expander('Time-point metadata', expanded=True):
    cols = st.columns(min(4, n_timepoints))
    for t in range(n_timepoints):
        with cols[t % len(cols)]:
            label = st.text_input(f'Time point {t+1}', f'Day {t}', key=f'large_tp_{t}')
            elapsed = st.number_input(f'Elapsed {time_unit}', min_value=0.0, value=float(t), key=f'large_elapsed_{t}')
            time_meta.append({'timepoint_index': t+1, 'timepoint': label, 'elapsed_time': float(elapsed)})

# ------------------------- source references -------------------------
st.subheader('3. Large-image source references')
st.caption(
    'Choose how many whole-array fields exist per condition × time point. The table below then creates one row per large source. '
    'Rows with a blank `source_uri` are ignored.'
)
fields_per_cell = st.selectbox('Whole-array fields per condition × time point', list(range(1, 9)), index=0, key='large_fields')

rows = []
for cond in condition_meta:
    for tp in time_meta:
        for field_i in range(1, fields_per_cell+1):
            rows.append({
                'condition_index': cond['condition_index'],
                'condition': cond['condition'],
                'timepoint_index': tp['timepoint_index'],
                'timepoint': tp['timepoint'],
                'field_id': f'F{field_i:02d}',
                'source_uri': '',
                'source_type': 'auto',
                'series_index': 0,
                'pyramid_level': 0,
                'source_sha256': '',
                'compute_full_sha256': False,
            })

source_template = pd.DataFrame(rows)
source_editor = st.data_editor(
    source_template,
    use_container_width=True,
    hide_index=True,
    disabled=['condition_index', 'condition', 'timepoint_index', 'timepoint'],
    column_config={
        'source_uri': st.column_config.TextColumn('Source URI / path', width='large'),
        'source_type': st.column_config.SelectboxColumn('Type', options=['auto', 'OME-TIFF', 'BigTIFF', 'TIFF', 'OME-Zarr']),
        'series_index': st.column_config.NumberColumn('Series', min_value=0, step=1),
        'pyramid_level': st.column_config.NumberColumn('Level', min_value=0, step=1),
        'source_sha256': st.column_config.TextColumn('Known SHA-256 (optional)', width='medium'),
        'compute_full_sha256': st.column_config.CheckboxColumn('Stream full SHA-256'),
    },
    key='large_source_editor'
)

source_rows = source_editor[source_editor['source_uri'].astype(str).str.strip() != ''].copy()
st.info(f'{len(source_rows)} large source reference(s) configured.')

if len(source_rows):
    first = source_rows.iloc[0]
    if st.button('Probe first configured source', use_container_width=False):
        try:
            meta = probe_large_source(
                str(first['source_uri']), str(first['source_type']), int(first['series_index']), int(first['pyramid_level'])
            )
            st.success(f"Detected {meta['format']} — {meta['width_px']:,} × {meta['height_px']:,} px, {meta['channel_count']} channel(s).")
            st.json(meta)
        except Exception as exc:
            st.error(f'Could not probe source: {exc}')

# ------------------------- channel map -------------------------
st.subheader('4. Channel mapping')
st.caption('Channel indices are zero-based. Use −1 where a dedicated channel is not available.')
a, b, c, d = st.columns(4)
with a:
    red_ch = st.number_input('RFP / red channel index', min_value=-1, value=0, step=1, key='large_red')
with b:
    green_ch = st.number_input('GFP / green channel index', min_value=-1, value=1, step=1, key='large_green')
with c:
    blue_ch = st.number_input('Blue channel index', min_value=-1, value=2, step=1, key='large_blue')
with d:
    brightfield_ch = st.number_input('Brightfield channel index', min_value=-1, value=-1, step=1, key='large_bf')
a, b, c = st.columns(3)
with a:
    well_detection_ch = st.number_input('Dedicated well-detection channel', min_value=-1, value=-1, step=1, key='large_well_ch')
with b:
    z_index = st.number_input('Z plane index', min_value=0, value=0, step=1, key='large_z')
with c:
    internal_t_index = st.number_input('Internal T index', min_value=0, value=0, step=1, key='large_internal_t')

# ------------------------- analysis settings -------------------------
st.subheader('5. Tiled analysis settings')
a, b, c = st.columns(3)
with a:
    well_diameter = st.number_input('Microwell diameter (µm)', min_value=1.0, value=100.0, step=1.0, key='large_well_um')
with b:
    tile_size = st.selectbox('Tile size (px)', [1024, 1536, 2048, 3072, 4096], index=2, key='large_tile')
with c:
    standard_crop = st.selectbox('ML crop size (px)', [128, 224, 256, 320, 512], index=2, key='large_crop')

split = st.checkbox('Split touching PDOs', True, key='large_split')
with st.expander('Advanced detection thresholds'):
    rmin = st.number_input('Minimum well radius (px)', 5, 1000, 23, step=1, key='large_rmin')
    rmax = st.number_input('Maximum well radius (px)', 6, 2000, 40, step=1, key='large_rmax')
    spacing = st.number_input('Minimum well spacing (px)', 10, 5000, 54, step=1, key='large_spacing')
    hp2 = st.number_input('Well detection sensitivity', 1.0, 100.0, 27.0, 1.0, key='large_hp2')
    gl = st.number_input('GFP PDO low threshold', 0.0, 255.0, 30.0, 1.0, key='large_gl')
    gh = st.number_input('GFP PDO high threshold', 0.0, 255.0, 45.0, 1.0, key='large_gh')
    amin = st.number_input('Minimum GFP PDO area (px²)', 1, 1000000, 20, step=1, key='large_amin')
    pdist = st.number_input('PDO split peak distance (px)', 1, 1000, 18, step=1, key='large_pdist')
    bf_contrast = st.number_input('Unlabelled PDO contrast threshold', 0.5, 100.0, 10.0, 0.5, key='large_bfcon')
    bf_min_area = st.number_input('Unlabelled PDO minimum area (px²)', 5, 1000000, 80, step=5, key='large_bfarea')
    pt = st.number_input('PSC focus threshold', 0.0, 255.0, 9.0, 0.5, key='large_pt')
    prm = st.number_input('PSC red-minus-blue minimum', 0.0, 255.0, 12.0, 0.5, key='large_prm')
    ppd = st.number_input('PSC focus minimum spacing (px)', 1, 1000, 4, step=1, key='large_ppd')

settings = build_settings_from_widgets(
    well_diameter, rmin, rmax, spacing, hp2, gl, gh, amin, split, pdist,
    pt, prm, ppd, 12, bf_contrast, bf_min_area
)
channel_config = {
    'red_channel': int(red_ch), 'green_channel': int(green_ch), 'blue_channel': int(blue_ch),
    'brightfield_channel': int(brightfield_ch), 'well_detection_channel': int(well_detection_ch),
    'z_index': int(z_index), 'internal_t_index': int(internal_t_index),
}

# ------------------------- resume -------------------------
st.subheader('6. Resume / checkpointing')
resume_file = st.file_uploader(
    'Optional previous large-analysis resume bundle (.zip)', type=['zip'], accept_multiple_files=False,
    key='large_resume_upload'
)
if resume_file is not None and st.session_state.get('large_resume_upload_name') != resume_file.name:
    try:
        restored = restore_resume_bundle(resume_file.getvalue())
        st.session_state['large_work_root'] = str(restored)
        st.session_state['large_resume_upload_name'] = resume_file.name
        st.success('Resume bundle restored. Completed tiles/wells will be skipped when the source fingerprints match.')
    except Exception as exc:
        st.error(f'Could not restore resume bundle: {exc}')

make_ml = st.checkbox('Create ML / Virtual Model Export', value=True, key='large_ml')

# assemble sources with metadata
sources = []
condition_lookup = {c['condition_index']: c for c in condition_meta}
time_lookup = {t['timepoint_index']: t for t in time_meta}
for _, row in source_rows.iterrows():
    ci, ti = int(row['condition_index']), int(row['timepoint_index'])
    cond, tp = condition_lookup[ci], time_lookup[ti]
    sources.append({
        **cond, **tp,
        'experiment_id': experiment_id,
        'device_id': device_id,
        'biological_replicate_id': replicate_id,
        'pdo_model': pdo_model,
        'time_unit': time_unit,
        'field_id': str(row['field_id']),
        'source_uri': str(row['source_uri']).strip(),
        'source_type': str(row['source_type']),
        'series_index': int(row['series_index']),
        'pyramid_level': int(row['pyramid_level']),
        'source_sha256': str(row.get('source_sha256', '') or '').strip(),
        'compute_full_sha256': bool(row.get('compute_full_sha256', False)),
    })

run = st.button('Run / resume whole-array analysis', type='primary', use_container_width=True, disabled=not sources)
if run:
    overall = st.progress(0, text='Preparing large-source analysis…')
    status_box = st.empty()

    def progress(done, total, phase):
        frac = int(min(99, max(1, 100 * done / max(1, total))))
        label = 'Scanning tiles for microwells' if phase == 'well_scan' else 'Analysing detected microwells'
        overall.progress(frac, text=f'{label}: {done:,} / {total:,}')
        status_box.caption('Checkpoints are being written incrementally to the working dataset.')

    try:
        work_root = st.session_state.get('large_work_root')
        root, out, source_manifest, wdf, pdf, pscdf, tracking, run_status, ml_path = process_large_experiment(
            sources, settings, channel_config,
            tile_size=int(tile_size), standard_crop_size=int(standard_crop),
            work_root=work_root, progress_callback=progress, make_ml_export=make_ml
        )
        st.session_state['large_work_root'] = str(root)
        st.session_state['large_out'] = str(out)
        st.session_state['large_results_zip'] = zip_bytes(out)
        st.session_state['large_resume_zip'] = make_resume_bundle(root)
        st.session_state['large_source_manifest'] = source_manifest.to_dict('records')
        st.session_state['large_run_status'] = run_status
        if ml_path is not None:
            st.session_state['large_ml_zip'] = zip_bytes(ml_path)
        overall.progress(100, text='Whole-array analysis pass complete.')
        status_box.empty()
        if run_status['all_complete']:
            st.success('All configured large sources completed.')
        else:
            st.warning('The pass finished with partial/error sources. Download the resume bundle before leaving this page, correct any source problem, and run again.')
    except Exception as exc:
        overall.empty()
        status_box.empty()
        st.error(f'Large-source analysis stopped: {exc}')
        work_root = st.session_state.get('large_work_root')
        if work_root and Path(work_root).exists():
            try:
                st.session_state['large_resume_zip'] = make_resume_bundle(work_root)
            except Exception:
                pass

if 'large_run_status' in st.session_state:
    st.divider()
    st.subheader('Large-dataset results')
    status = st.session_state['large_run_status']
    a, b, c, d = st.columns(4)
    a.metric('Sources', status.get('sources_total', 0))
    b.metric('Complete', status.get('sources_complete', 0))
    c.metric('Partial', status.get('sources_partial', 0))
    d.metric('Errors', status.get('sources_error', 0))

    cols = st.columns(3)
    with cols[0]:
        if 'large_results_zip' in st.session_state:
            st.download_button(
                'Download results ZIP', st.session_state['large_results_zip'],
                'whole_array_analysis_results.zip', 'application/zip', use_container_width=True
            )
    with cols[1]:
        if 'large_ml_zip' in st.session_state:
            st.download_button(
                'Download ML export ZIP', st.session_state['large_ml_zip'],
                'whole_array_ML_virtual_model_export.zip', 'application/zip', type='primary', use_container_width=True
            )
    with cols[2]:
        if 'large_resume_zip' in st.session_state:
            st.download_button(
                'Download resume bundle', st.session_state['large_resume_zip'],
                'whole_array_resume_bundle.zip', 'application/zip', use_container_width=True
            )

    if 'large_source_manifest' in st.session_state:
        with st.expander('Raw-source manifest and provenance', expanded=True):
            st.dataframe(pd.DataFrame(st.session_state['large_source_manifest']), use_container_width=True, hide_index=True)

st.caption(
    'Large-source mode is designed for region-readable formats and storage. If a 3–5 GB TIFF is only on your local workstation, '
    'it must first be placed somewhere this hosted app can read by range request, or the same processing engine can be run locally against the file path.'
)
