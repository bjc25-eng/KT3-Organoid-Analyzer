from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analysis_core import GFP_MODE, build_settings_from_widgets, zip_bytes
from nd2_large_source import ND2_SOURCE_LABEL, install_nd2_dispatch, probe_nd2_source
from nd2_physical_scan import install_nd2_physical_well_scan
from nd2_qc import process_large_experiment_qc
from nd2_s3_stage import get_s3_client, list_nd2_objects, stage_s3_nd2, upload_tree

install_nd2_dispatch()
install_nd2_physical_well_scan()

st.set_page_config(page_title='Nikon ND2 Whole-Array Imaging', page_icon='🟢', layout='wide')
st.title('Nikon ND2 Whole-Array Imaging')
st.caption(
    'Large-file workflow: select an ND2 already in S3, stage it once onto the compute worker, '
    'then probe and run/resume without a multi-GB browser upload.'
)


def secret(name: str, default: str = '') -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def format_gib(size: int) -> str:
    return f'{float(size) / (1024 ** 3):.2f} GiB'


region = secret('AWS_DEFAULT_REGION', 'eu-west-2')
default_bucket = secret('S3_BUCKET', '')
default_input_prefix = secret('S3_INPUT_PREFIX', 'uploads/')
default_output_prefix = secret('S3_OUTPUT_PREFIX', 'results/')
default_cache_root = secret('ND2_CACHE_ROOT', '/tmp/kt3_nd2_cache')

s3 = get_s3_client(
    region_name=region or None,
    access_key_id=secret('AWS_ACCESS_KEY_ID') or None,
    secret_access_key=secret('AWS_SECRET_ACCESS_KEY') or None,
    session_token=secret('AWS_SESSION_TOKEN') or None,
)

with st.expander('How the large-ND2 mode works', expanded=True):
    st.markdown('''
1. Put the original Nikon `.nd2` in the configured S3 input prefix.
2. Select it below and press **Stage & probe selected ND2**.
3. The file is copied once to local compute storage. A complete cached copy is reused on later runs.
4. The ND2 metadata is probed before analysis: channels, XY positions, physical pixel size and native frame memory.
5. **Run / resume** uses a deterministic work directory tied to that S3 object, so checkpoints are reused automatically.
6. Results can be written back to S3 automatically; browser ZIP downloads remain optional convenience outputs.

The scientific final-QC logic is unchanged: physically calibrated microwell detection, conservative shape-supported PDO splitting, full-object segmentation, physical well-radius membership, DIC wall evidence, ambiguous-edge classification and DIC microwell-validity QC.
''')

if default_cache_root.startswith('/tmp'):
    st.info(
        'ND2_CACHE_ROOT currently points to temporary storage. This is suitable for testing, but for true automatic resume '
        'across worker restarts set ND2_CACHE_ROOT to persistent storage such as an EC2 EBS mount (for example /data/kt3_nd2_cache).'
    )

st.subheader('1. Select and stage an ND2 from S3')
a, b = st.columns([2, 3])
with a:
    bucket = st.text_input('S3 bucket', default_bucket)
with b:
    input_prefix = st.text_input('ND2 input prefix', default_input_prefix)

if st.button('Refresh ND2 list', use_container_width=True):
    st.session_state.pop('nd2_s3_objects', None)

if 'nd2_s3_objects' not in st.session_state:
    try:
        st.session_state['nd2_s3_objects'] = list_nd2_objects(s3, bucket, input_prefix) if bucket else []
    except Exception as exc:
        st.session_state['nd2_s3_objects'] = []
        st.error(f'Could not list ND2 files in S3: {exc}')

objects = st.session_state.get('nd2_s3_objects', [])
if objects:
    labels = [f"{row['key']}  ({format_gib(row['size'])})" for row in objects]
    selected_label = st.selectbox('ND2 file', labels)
    selected = objects[labels.index(selected_label)]
    st.caption(f"Selected: s3://{bucket}/{selected['key']} · {format_gib(selected['size'])}")
else:
    selected = None
    st.warning('No .nd2 objects were found under this bucket/prefix.')

cache_root = st.text_input(
    'Persistent ND2 cache / checkpoint root on compute worker',
    default_cache_root,
    help='Use an EBS-backed path on EC2 for resume across application/worker restarts.',
)

probe_position = st.number_input('XY position to inspect', min_value=0, value=0, step=1)

stage_probe = st.button(
    'Stage & probe selected ND2',
    type='primary',
    use_container_width=True,
    disabled=selected is None or not bucket.strip() or not cache_root.strip(),
)

if stage_probe and selected is not None:
    stage_bar = st.progress(0, text='Checking S3 object and local cache…')
    stage_text = st.empty()

    def stage_progress(done: int, total: int):
        pct = int(min(100, 100 * done / max(1, total)))
        stage_bar.progress(pct, text=f'Staging ND2: {format_gib(done)} / {format_gib(total)}')

    try:
        staged = stage_s3_nd2(
            s3,
            bucket.strip(),
            selected['key'],
            cache_root.strip(),
            progress_callback=stage_progress,
        )
        st.session_state['nd2_staged'] = staged
        st.session_state['nd2_work_root'] = staged['work_root']
        stage_text.caption('Cached copy reused.' if staged['reused'] else 'S3 copy staged successfully.')

        meta = probe_nd2_source(staged['local_path'], int(probe_position))
        st.session_state['nd2_probe_meta'] = meta
        stage_bar.progress(100, text='ND2 staged and metadata probe complete.')
        st.success(
            f"ND2 detected — {meta['width_px']:,} × {meta['height_px']:,} px per position, "
            f"{meta['channel_count']} channel(s), {meta['position_count']} XY position(s)."
        )
    except Exception as exc:
        stage_bar.empty()
        stage_text.empty()
        st.error(f'Could not stage/probe ND2: {exc}')

staged = st.session_state.get('nd2_staged')
meta = st.session_state.get('nd2_probe_meta')

if staged:
    st.success(
        f"Ready on compute worker: {staged['local_path']} · checkpoint root: {staged['work_root']}"
    )

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
st.subheader('2. Configure analysis')
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

st.markdown('**Native ND2 indices**')
a, b, c = st.columns(3)
with a:
    position_index = st.number_input('ND2 XY position index', min_value=0, value=int(probe_position), step=1, key='nd2_pos')
with b:
    internal_t = st.number_input('Internal ND2 time index', min_value=0, value=0, step=1)
with c:
    z_index = st.number_input('Z plane index', min_value=0, value=0, step=1)

st.markdown('**DIC + GFP channel mapping**')
suggested_gfp = int(meta['suggested_gfp_channel']) if meta and meta.get('suggested_gfp_channel') is not None else 1
suggested_dic = int(meta['suggested_dic_channel']) if meta and meta.get('suggested_dic_channel') is not None else 0
a, b = st.columns(2)
with a:
    gfp_channel = st.number_input('GFP channel index', min_value=0, value=max(0, suggested_gfp), step=1)
with b:
    dic_channel = st.number_input('DIC / well-detection channel index', min_value=0, value=max(0, suggested_dic), step=1)

st.info('PSC/RFP analysis remains disabled in this ND2 workflow because the current source type is DIC + GFP only.')

st.subheader('3. Detection and QC settings')
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
    help='Leave off for the first validation run. Ambiguous candidates are retained but flagged in PDO_candidate_QC.csv.',
)

with st.expander('Advanced detection thresholds'):
    st.caption(
        'Radius/spacing fields are legacy UI values. Native ND2 final-QC derives the Hough radius range and spacing '
        'from physical ND2 calibration; sensitivity and GFP thresholds remain active.'
    )
    rmin = st.number_input('Legacy minimum well radius (px; overridden in ND2 final-QC)', 5, 1000, 23, step=1)
    rmax = st.number_input('Legacy maximum well radius (px; overridden in ND2 final-QC)', 6, 2000, 40, step=1)
    spacing = st.number_input('Legacy minimum well spacing (px; overridden in ND2 final-QC)', 10, 5000, 54, step=1)
    hp2 = st.number_input('Well detection sensitivity', 1.0, 100.0, 27.0, 1.0)
    gl = st.number_input('GFP PDO low threshold', 0.0, 255.0, 30.0, 1.0)
    gh = st.number_input('GFP PDO high threshold', 0.0, 255.0, 45.0, 1.0)
    amin = st.number_input('Minimum GFP PDO area (px²)', 1, 1000000, 20, step=1)
    pdist = st.number_input('PDO split peak distance (px)', 1, 1000, 18, step=1)

settings = build_settings_from_widgets(
    well_diameter, rmin, rmax, spacing, hp2, gl, gh, amin, split, pdist,
    9.0, 12.0, 4, 12, 10.0, 80,
    organoid_mode=GFP_MODE, rfp_psc_present=False,
)
settings.exclude_ambiguous_edge_candidates = bool(exclude_ambiguous)

channel_config = {
    'red_channel': -1,
    'green_channel': int(gfp_channel),
    'blue_channel': -1,
    'brightfield_channel': int(dic_channel),
    'well_detection_channel': int(dic_channel),
    'z_index': int(z_index),
    'internal_t_index': int(internal_t),
}

st.subheader('4. Automatic checkpointing and S3 outputs')
if staged:
    st.caption(
        f"Checkpoint directory: {staged['work_root']}. Pressing Run / resume again for this staged object reuses compatible checkpoints automatically."
    )
else:
    st.caption('Stage an ND2 first. No resume ZIP upload is required.')

output_prefix = st.text_input(
    'S3 results prefix',
    f"{default_output_prefix.rstrip('/')}/nd2/{experiment_id}" if default_output_prefix else f"results/nd2/{experiment_id}",
)
auto_upload = st.checkbox('Automatically save completed result tree back to S3', True)

source_uri = staged['local_path'] if staged else ''
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
    'source_uri': source_uri,
    'source_type': ND2_SOURCE_LABEL,
    'series_index': int(position_index),
    'pyramid_level': 0,
    'source_sha256': '',
    'compute_full_sha256': False,
}

run = st.button(
    'Run / resume native ND2 analysis',
    type='primary',
    use_container_width=True,
    disabled=not bool(source_uri),
)

if run:
    progress_bar = st.progress(0, text='Opening staged ND2 lazily…')
    status = st.empty()

    def progress(done, total, phase):
        frac = int(min(99, max(1, 100 * done / max(1, total))))
        label = 'Scanning DIC for microwells' if phase == 'well_scan' else 'Analysing GFP-positive PDOs with final QC'
        progress_bar.progress(frac, text=f'{label}: {done:,} / {total:,}')
        status.caption('Incremental checkpoints are being written to the persistent work directory.')

    try:
        root, out, manifest, wdf, pdf, pscdf, tracking, run_status, ml_path = process_large_experiment_qc(
            [source],
            settings,
            channel_config,
            tile_size=int(tile_size),
            standard_crop_size=int(standard_crop),
            work_root=staged['work_root'],
            progress_callback=progress,
            make_ml_export=True,
        )
        st.session_state['nd2_work_root'] = str(root)
        st.session_state['nd2_results_zip'] = zip_bytes(out)
        st.session_state['nd2_ml_zip'] = zip_bytes(ml_path) if ml_path is not None else None
        st.session_state['nd2_manifest'] = manifest.to_dict('records')
        st.session_state['nd2_run_status'] = run_status
        st.session_state['nd2_output_dir'] = str(out)

        if auto_upload and bucket.strip():
            upload_record = upload_tree(s3, out, bucket.strip(), output_prefix.strip())
            st.session_state['nd2_s3_upload'] = upload_record

        progress_bar.progress(100, text='ND2 final-QC analysis pass complete.')
        status.empty()
        if run_status.get('all_complete'):
            st.success('Native ND2 analysis completed with final PDO/well QC.')
        else:
            st.warning('The run contains an incomplete/error source. Fix the problem and press Run / resume; existing checkpoints remain in place.')
    except Exception as exc:
        progress_bar.empty()
        status.empty()
        st.error(f'ND2 analysis stopped: {exc}')
        if staged:
            st.info(f"Checkpoint data remain at {staged['work_root']}. Press Run / resume after correcting the problem.")

if st.session_state.get('nd2_run_status'):
    st.divider()
    st.subheader('ND2 outputs')
    run_status = st.session_state['nd2_run_status']
    a, b, c, d = st.columns(4)
    a.metric('Outside-well rejected', int(run_status.get('qc_rejected_outside_well_candidates', 0)))
    b.metric('Ambiguous candidates', int(run_status.get('qc_ambiguous_PDO_candidates', 0)))
    c.metric('False-well candidates rejected', int(run_status.get('qc_rejected_false_well_candidates', 0)))
    d.metric('False detected wells', int(run_status.get('qc_rejected_false_wells', 0)))

    upload_record = st.session_state.get('nd2_s3_upload')
    if upload_record:
        st.success(
            f"Saved {upload_record['uploaded_files']} result files to "
            f"s3://{upload_record['bucket']}/{upload_record['prefix']}/"
        )

    a, b = st.columns(2)
    with a:
        if st.session_state.get('nd2_results_zip'):
            st.download_button(
                'Download results ZIP',
                st.session_state['nd2_results_zip'],
                'ND2_whole_array_results.zip',
                'application/zip',
                use_container_width=True,
            )
    with b:
        if st.session_state.get('nd2_ml_zip'):
            st.download_button(
                'Download ML export ZIP',
                st.session_state['nd2_ml_zip'],
                'ND2_ML_virtual_model_export.zip',
                'application/zip',
                type='primary',
                use_container_width=True,
            )

    with st.expander('Raw ND2 source manifest / provenance', expanded=True):
        st.dataframe(pd.DataFrame(st.session_state.get('nd2_manifest', [])), use_container_width=True, hide_index=True)

st.caption(
    'Recommended validation: run one representative ~3 GB DIC+GFP ND2 first, review the well/PDO QC outputs, '
    'then apply the unchanged settings to the remaining files.'
)
