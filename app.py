from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from processor import (
    APP_TITLE,
    BRIGHTFIELD_MODE,
    GFP_MODE,
    PSC_ABSENT,
    PSC_PRESENT,
    archive_results_to_s3,
    build_settings_from_widgets,
    get_s3_client,
    list_s3_images,
    process,
    process_s3_batch,
    zip_bytes,
)


def secret(name: str, default: str = '') -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def s3_client(region: str):
    return get_s3_client(
        region_name=region,
        access_key_id=secret('AWS_ACCESS_KEY_ID') or None,
        secret_access_key=secret('AWS_SECRET_ACCESS_KEY') or None,
        session_token=secret('AWS_SESSION_TOKEN') or None,
    )


def settings_sidebar():
    st.sidebar.header('Analysis settings')
    well = st.sidebar.number_input('Microwell diameter (µm)', 1.0, 1000.0, 100.0, 1.0)
    split = st.sidebar.checkbox('Split touching PDOs', True)
    bins = st.sidebar.slider('Histogram bins', 5, 30, 12)
    cols = st.sidebar.slider('Contact-sheet columns', 3, 10, 5)
    create_pdo_centred = st.sidebar.checkbox('Create one crop per PDO', True)
    crop_size = st.sidebar.slider('PDO-centred crop size (px)', 128, 512, 256, 16)
    with st.sidebar.expander('Advanced thresholds'):
        rmin = st.number_input('Minimum well radius (px)', 5, 200, 23, step=1)
        rmax = st.number_input('Maximum well radius (px)', 6, 300, 40, step=1)
        spacing = st.number_input('Minimum well spacing (px)', 10, 500, 54, step=1)
        hp2 = st.number_input('Well detection sensitivity', 1.0, 100.0, 27.0, 1.0)
        gl = st.number_input('GFP PDO low threshold', 0.0, 255.0, 30.0, 1.0)
        gh = st.number_input('GFP PDO high threshold', 0.0, 255.0, 45.0, 1.0)
        amin = st.number_input('Minimum GFP PDO area (px²)', 1, 100000, 20, step=1)
        pdist = st.number_input('PDO split peak distance (px)', 1, 100, 18, step=1)
        bf_contrast = st.number_input('Unlabelled PDO contrast threshold', 0.5, 100.0, 10.0, 0.5)
        bf_min_area = st.number_input('Unlabelled PDO minimum area (px²)', 5, 100000, 80, step=5)
        pt = st.number_input('PSC focus threshold', 0.0, 255.0, 9.0, 0.5)
        prm = st.number_input('PSC red-minus-blue minimum', 0.0, 255.0, 12.0, 0.5)
        ppd = st.number_input('PSC focus minimum spacing (px)', 1, 100, 4, step=1)
    settings = build_settings_from_widgets(
        well, rmin, rmax, spacing, hp2, gl, gh, amin, split, pdist,
        pt, prm, ppd, bins, bf_contrast, bf_min_area
    )
    return settings, int(cols), bool(create_pdo_centred), int(crop_size)


def show_results(out: Path, summary: pd.DataFrame, image_summary: pd.DataFrame, s3_record: dict | None = None):
    st.divider()
    st.subheader('Results')
    if len(summary):
        row = summary.iloc[0]
        a, b, c, d, e = st.columns(5)
        a.metric('Images', int(row.get('images_processed', 0)))
        b.metric('Visible wells', int(row.get('fully_visible_wells', 0)))
        c.metric('PDO wells', int(row.get('PDO_containing_wells', 0)))
        d.metric('PDOs', int(row.get('PDO_count', 0)))
        m = row.get('mean_PDO_diameter_um', None)
        e.metric('Mean PDO diameter', f'{float(m):.1f} µm' if m is not None and pd.notna(m) else '—')

    st.warning(
        'Automated outputs require visual QC. Keep the manually QC\'d thesis dataset separate from a fresh automated rerun if their PDO counts differ.'
    )

    payload = zip_bytes(out)
    st.download_button(
        'Download complete results ZIP', payload, 'KT3_PDO_PSC_analysis_results.zip',
        'application/zip', type='primary', use_container_width=True
    )

    if s3_record:
        st.success(
            f"Saved to s3://{s3_record['bucket']}/{s3_record['prefix']}/ "
            f"({s3_record['uploaded_result_files']} result files plus ZIP)."
        )
        st.link_button('Download ZIP directly from S3 (valid 1 hour)', s3_record['presigned_url'], use_container_width=True)

    figs = [
        ('PDO size distribution', out/'figures'/'PDO_size_distribution.png'),
        ('PSC frequency across PDOs', out/'figures'/'PSC_count_frequency_across_PDOs.png'),
        ('PDO-centred contact sheet', out/'figures'/'PDO_centred_contact_sheet_compact.png'),
        ('PDO-well contact sheet', out/'figures'/'PDO_well_contact_sheet_compact.png'),
    ]
    figs = [(label, p) for label, p in figs if p.exists()]
    for start in range(0, len(figs), 2):
        cc = st.columns(min(2, len(figs)-start))
        for col, (label, p) in zip(cc, figs[start:start+2]):
            col.image(str(p), caption=label, use_container_width=True)

    with st.expander('Image-by-image summary'):
        st.dataframe(image_summary, use_container_width=True, hide_index=True)

    pdo_csv = out/'csv'/'PDO_centred_raw_data.csv'
    if pdo_csv.exists():
        pdo_df = pd.read_csv(pdo_csv)
        with st.expander('PDO-level table'):
            st.dataframe(pdo_df, use_container_width=True, hide_index=True)
            st.caption(f'One row per automatically detected PDO: {len(pdo_df)} rows.')

    overlays = sorted((out/'indexed_large_images').glob('*.png'))
    if overlays:
        with st.expander('Indexed large-image QC overlays'):
            for start in range(0, min(12, len(overlays)), 3):
                cc = st.columns(3)
                for col, p in zip(cc, overlays[start:start+3]):
                    col.image(str(p), caption=p.stem, use_container_width=True)


st.set_page_config(page_title=APP_TITLE, page_icon='🔬', layout='wide')
st.title(APP_TITLE)
st.caption('S3-backed single/batch PDO + PSC analysis. The longitudinal multi-condition workflow remains available from the Streamlit page menu.')

with st.expander('Measurement notes'):
    st.markdown('''
- PDO size is a **2D equivalent circular diameter** from segmented projected area, not a true 3D organoid diameter.
- PSC counts are **PSC-like red fluorescent foci**, not definitive individual-cell counts.
- The new PDO-centred export generates **one crop per automatically detected PDO**, so multiple PDOs in one well produce multiple crops.
- Indexed overlays and labelled crops should be visually reviewed before thesis/publication use.
''')

base_settings, cols, create_pdo_centred, crop_size = settings_sidebar()

st.subheader('Assay channels')
a, b = st.columns(2)
with a:
    organoid_mode = st.selectbox('Are the organoids GFP-labelled?', [GFP_MODE, BRIGHTFIELD_MODE])
with b:
    psc_mode = st.selectbox('Are RFP-labelled PSC/stromal cells present?', [PSC_PRESENT, PSC_ABSENT])
settings = replace(base_settings, organoid_mode=organoid_mode, rfp_psc_present=psc_mode == PSC_PRESENT)

st.subheader('Image source')
source = st.radio('Choose input source', ['AWS S3', 'Browser upload'], horizontal=True)

bucket_default = secret('S3_BUCKET')
region_default = secret('AWS_DEFAULT_REGION', 'eu-west-2')
input_prefix_default = secret('S3_INPUT_PREFIX', 'uploads/')
output_prefix_default = secret('S3_OUTPUT_PREFIX', 'results/')

selected_keys = []
uploaded = []
bucket = bucket_default
region = region_default
output_prefix = output_prefix_default
save_to_s3 = False
run_label = ''

if source == 'AWS S3':
    c1, c2 = st.columns([2, 1])
    bucket = c1.text_input('S3 bucket', bucket_default, placeholder='your-bucket-name')
    region = c2.text_input('AWS region', region_default)
    input_prefix = st.text_input('Input prefix', input_prefix_default)

    if st.button('List S3 images', disabled=not bool(bucket)):
        try:
            client = s3_client(region)
            st.session_state['s3_keys'] = list_s3_images(client, bucket, input_prefix)
        except (NoCredentialsError, ClientError, BotoCoreError) as exc:
            st.error(f'Could not read S3: {exc}')

    keys = st.session_state.get('s3_keys', [])
    if keys:
        st.caption(f'Found {len(keys)} supported image(s).')
        selected_keys = st.multiselect('Images to analyse', keys, default=keys)
    else:
        st.info('Enter the bucket/prefix and click “List S3 images”.')

    st.markdown('#### Result storage')
    output_prefix = st.text_input('Output prefix', output_prefix_default)
    run_label = st.text_input('Optional run label', '', placeholder='e.g. KT3_day7_repeat1')
    save_to_s3 = st.checkbox('Save complete results back to S3', True)
else:
    uploaded = st.file_uploader(
        'Upload microscopy images', type=['png','jpg','jpeg','tif','tiff','bmp'],
        accept_multiple_files=True
    ) or []
    if bucket_default:
        with st.expander('Optional S3 result archive'):
            save_to_s3 = st.checkbox('Save result package to S3', False)
            bucket = st.text_input('Result bucket', bucket_default)
            region = st.text_input('Result region', region_default)
            output_prefix = st.text_input('Result prefix', output_prefix_default)
            run_label = st.text_input('Optional run label', '')

has_input = bool(selected_keys) if source == 'AWS S3' else bool(uploaded)
run = st.button('Run analysis', type='primary', use_container_width=True, disabled=not has_input)

if run:
    bar = st.progress(5, text='Preparing analysis…')
    try:
        client = None
        if source == 'AWS S3':
            bar.progress(12, text='Downloading selected microscopy images from S3…')
            client = s3_client(region)
            _, out, summary, image_summary = process_s3_batch(
                client, bucket, selected_keys, settings, cols,
                create_pdo_centred=create_pdo_centred, crop_size_px=crop_size
            )
        else:
            bar.progress(12, text='Processing uploaded microscopy images…')
            _, out, summary, image_summary = process(
                uploaded, settings, cols,
                create_pdo_centred=create_pdo_centred, crop_size_px=crop_size
            )

        s3_record = None
        if save_to_s3:
            if not bucket:
                raise RuntimeError('S3 archiving is enabled but no bucket was supplied.')
            if client is None:
                client = s3_client(region)
            bar.progress(88, text='Saving complete result package to S3…')
            s3_record = archive_results_to_s3(
                client, out, bucket, output_prefix, run_label,
                zip_name='KT3_PDO_PSC_analysis_results.zip'
            )

        bar.progress(100, text='Complete')
        st.session_state['out'] = str(out)
        st.session_state['summary'] = summary.to_dict('records')
        st.session_state['image_summary'] = image_summary.to_dict('records')
        st.session_state['s3_record'] = s3_record
        st.success('Analysis complete.')
    except Exception as exc:
        bar.empty()
        st.error(f'Analysis stopped: {exc}')
        st.info('For S3 errors, check Streamlit Secrets and IAM access. For image-analysis errors, review the well radius and detection thresholds.')

if 'out' in st.session_state:
    out = Path(st.session_state['out'])
    if out.exists():
        show_results(
            out,
            pd.DataFrame(st.session_state.get('summary', [])),
            pd.DataFrame(st.session_state.get('image_summary', [])),
            st.session_state.get('s3_record')
        )
