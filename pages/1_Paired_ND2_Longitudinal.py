from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis_core import build_settings_from_widgets, zip_bytes
from advanced_analysis import ANALYSIS_GROWTH, ANALYSIS_WATERFALL, ANALYSIS_HEATMAP, ANALYSIS_DISTRIBUTIONS, ANALYSIS_PSC
from nd2_longitudinal import probe_pair, process_paired_nd2_longitudinal
from trajectory_safe_analysis import run_selected_analyses

st.set_page_config(page_title='Paired ND2 Longitudinal Growth', page_icon='🧫', layout='wide')
st.title('Paired ND2 Longitudinal Growth')
st.caption('Automatically register different Friday/Monday crops, match the same physical microwells, and quantify PDO growth.')

with st.expander('How Friday and Monday are matched', expanded=True):
    st.markdown('''
- Supply one ND2 source for baseline and one for follow-up.
- The same Nikon XY position index is treated as a **field pair**, but the two images do **not** need to have the same crop.
- Within each field pair the app registers the **DIC image content** with a robust affine transform, allowing translation, rotation and modest scale difference.
- Detected microwell centres are then matched one-to-one after registration. This avoids assuming that a local label such as `well 4,8` refers to the same physical well on both days.
- Only field pairs and microwell matches that pass automatic registration QC enter the growth analysis.
- Growth is calculated from **total PDO projected area per matched physical microwell**, so segmentation into one versus several PDO objects does not break the well-level trajectory.
- Registration overlays and CSV audit tables are included in the results ZIP.
- Use S3/presigned HTTPS URLs for multi-GB ND2 files rather than browser upload.
''')

st.subheader('1. ND2 sources')
c1, c2 = st.columns(2)
with c1:
    friday_uri = st.text_input('Baseline ND2 URI', placeholder='https://.../Friday.nd2 or s3://bucket/Friday.nd2')
with c2:
    monday_uri = st.text_input('Follow-up ND2 URI', placeholder='https://.../Monday.nd2 or s3://bucket/Monday.nd2')

if st.button('Probe both ND2 files', disabled=not (friday_uri.strip() and monday_uri.strip())):
    try:
        pair = probe_pair(friday_uri.strip(), monday_uri.strip())
        st.session_state['paired_nd2_probe'] = pair
        st.success('Both ND2 files were opened successfully.')
    except Exception as exc:
        st.error(f'Could not probe the ND2 pair: {exc}')

pair = st.session_state.get('paired_nd2_probe')
if pair:
    fmeta, mmeta = pair['friday'], pair['monday']
    a, b, c, d = st.columns(4)
    a.metric('Baseline XY positions', int(fmeta.get('position_count', 1)))
    b.metric('Follow-up XY positions', int(mmeta.get('position_count', 1)))
    c.metric('Position pairs available', int(pair['matching_position_count']))
    d.metric('Same frame dimensions', 'Yes' if pair['same_dimensions'] else 'No')
    if not pair['same_position_count']:
        st.warning('The ND2 files contain different numbers of XY positions. Only explicitly selected common indices will be analysed.')
    if not pair['same_dimensions']:
        st.info('Different frame dimensions are allowed: DIC registration will determine the overlapping physical microwells.')

    st.markdown('**Detected channels**')
    cf, cm = st.columns(2)
    with cf:
        st.caption('Baseline')
        st.dataframe(pd.DataFrame(fmeta.get('channel_metadata', [])), hide_index=True, use_container_width=True)
    with cm:
        st.caption('Follow-up')
        st.dataframe(pd.DataFrame(mmeta.get('channel_metadata', [])), hide_index=True, use_container_width=True)

    max_pos = max(0, int(pair['matching_position_count']) - 1)
    default_positions = list(range(min(12, max_pos + 1)))
    positions = st.multiselect(
        'XY position pairs to analyse',
        options=list(range(max_pos + 1)),
        default=default_positions,
        format_func=lambda x: f'Position {x} / Field {x+1:02d}',
    )

    st.subheader('2. Channels and timing')
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        gfp_channel = st.number_input('GFP channel index', min_value=0, value=int(fmeta.get('suggested_gfp_channel') or 0), step=1)
    with q2:
        dic_channel = st.number_input('DIC channel index', min_value=0, value=int(fmeta.get('suggested_dic_channel') or 1), step=1)
    with q3:
        use_rfp = st.checkbox('RFP PSC channel present', value=True)
        rfp_channel = st.number_input('RFP channel index', min_value=0, value=2, step=1, disabled=not use_rfp)
    with q4:
        elapsed_days = st.number_input('Elapsed days', min_value=0.0001, value=3.0, step=0.5)

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        experiment_id = st.text_input('Experiment ID', 'KT3_PSC_growth')
    with e2:
        device_id = st.text_input('Array / device ID', 'Array_001')
    with e3:
        replicate_id = st.text_input('Biological replicate ID', 'Replicate_1')
    with e4:
        condition_name = st.text_input('Condition', 'KT3 + PSC')

    l1, l2 = st.columns(2)
    with l1:
        friday_label = st.text_input('Baseline label', 'Friday / Day 0')
    with l2:
        monday_label = st.text_input('Follow-up label', 'Monday / Day 3')

    st.subheader('3. Analysis settings')
    st.caption('Physical well geometry and PDO area calibration are derived from native ND2 metadata. The values below control segmentation.')
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        well = st.number_input('Microwell diameter (µm)', 1.0, 1000.0, 100.0, 1.0)
    with s2:
        gl = st.number_input('GFP low threshold', 0.0, 255.0, 30.0, 1.0)
    with s3:
        gh = st.number_input('GFP high threshold', 0.0, 255.0, 45.0, 1.0)
    with s4:
        amin = st.number_input('Minimum GFP PDO area (px²)', 1, 100000, 20, step=1)

    split = st.checkbox('Split touching PDOs', True)
    make_ml = st.checkbox('Create ML / Virtual Model Export package', True)
    cols = st.slider('Contact-sheet columns', 3, 10, 5)
    settings = build_settings_from_widgets(
        well, 23, 40, 54, 27.0, gl, gh, amin, split, 18,
        9.0, 12.0, 4, 12, 10.0, 80
    )

    run = st.button('Register fields and calculate longitudinal growth', type='primary', use_container_width=True, disabled=not bool(positions))
    if run:
        meta = {
            'experiment_id': experiment_id,
            'device_id': device_id,
            'biological_replicate_id': replicate_id,
            'pdo_model': 'KT3',
            'time_unit': 'days',
        }
        bar = st.progress(5, text='Reading selected ND2 positions…')
        try:
            root, out, summary, tracking, ml_path = process_paired_nd2_longitudinal(
                friday_uri.strip(), monday_uri.strip(), [int(x) for x in positions],
                settings, int(cols), meta, condition_name, friday_label, monday_label,
                float(elapsed_days), int(gfp_channel), int(dic_channel),
                int(rfp_channel) if use_rfp else None, make_ml_export=make_ml,
            )
            bar.progress(72, text='DIC registration complete; calculating matched-well growth…')
            adv = Path(tempfile.mkdtemp(prefix='kt3_paired_nd2_advanced_'))
            selected = [ANALYSIS_GROWTH, ANALYSIS_WATERFALL, ANALYSIS_HEATMAP, ANALYSIS_DISTRIBUTIONS]
            if use_rfp:
                selected.append(ANALYSIS_PSC)
            manifest = run_selected_analyses(tracking, adv, selected, dose_unit='nM', time_unit='days')
            bar.progress(100, text='Complete')
            st.session_state['paired_nd2_long_zip'] = zip_bytes(out)
            st.session_state['paired_nd2_adv_zip'] = zip_bytes(adv)
            st.session_state['paired_nd2_tracking'] = tracking.to_dict('records')
            st.session_state['paired_nd2_summary'] = summary.to_dict('records')
            st.session_state['paired_nd2_registration'] = pd.read_csv(Path(out)/'csv'/'field_registration_summary.csv').to_dict('records')
            st.session_state['paired_nd2_growth'] = pd.read_csv(Path(out)/'csv'/'matched_well_growth.csv').to_dict('records')
            st.session_state['paired_nd2_adv_manifest'] = manifest.to_dict('records')
            st.session_state['paired_nd2_adv_path'] = str(adv)
            if ml_path is not None:
                st.session_state['paired_nd2_ml_zip'] = zip_bytes(ml_path)
            st.success('Registration and paired longitudinal analysis complete.')
        except Exception as exc:
            bar.empty()
            st.error(f'Analysis stopped: {exc}')

if 'paired_nd2_long_zip' in st.session_state:
    st.divider()
    st.subheader('Results')
    a, b, c = st.columns(3)
    with a:
        st.download_button('Download registered longitudinal results ZIP', st.session_state['paired_nd2_long_zip'], 'KT3_PSC_registered_ND2_longitudinal_results.zip', 'application/zip', use_container_width=True)
    with b:
        st.download_button('Download growth-analysis ZIP', st.session_state['paired_nd2_adv_zip'], 'KT3_PSC_registered_ND2_growth_analysis.zip', 'application/zip', type='primary', use_container_width=True)
    with c:
        if 'paired_nd2_ml_zip' in st.session_state:
            st.download_button('Download ML export ZIP', st.session_state['paired_nd2_ml_zip'], 'KT3_PSC_registered_ND2_ML_export.zip', 'application/zip', use_container_width=True)

    tracking = pd.DataFrame(st.session_state['paired_nd2_tracking'])
    summary = pd.DataFrame(st.session_state['paired_nd2_summary'])
    registration = pd.DataFrame(st.session_state.get('paired_nd2_registration', []))
    growth = pd.DataFrame(st.session_state.get('paired_nd2_growth', []))
    accepted_fields = int(registration['registration_status'].eq('accepted').sum()) if 'registration_status' in registration.columns else 0
    st.caption(f'{accepted_fields} field pair(s) passed registration QC; {tracking["trajectory_id"].nunique() if "trajectory_id" in tracking.columns else 0:,} physical microwell trajectories were matched.')

    with st.expander('Field registration QC', expanded=True):
        st.dataframe(registration, use_container_width=True, hide_index=True)
    with st.expander('Matched-well growth table', expanded=True):
        st.dataframe(growth, use_container_width=True, hide_index=True)
    with st.expander('Registered longitudinal tracking'):
        st.dataframe(tracking, use_container_width=True, hide_index=True)
    with st.expander('Condition/time summary'):
        st.dataframe(summary, use_container_width=True, hide_index=True)

    adv = Path(st.session_state['paired_nd2_adv_path'])
    pngs = sorted(adv.rglob('*.png'))
    if pngs:
        st.subheader('Growth figures')
        for start in range(0, len(pngs), 2):
            cc = st.columns(min(2, len(pngs) - start))
            for col, p in zip(cc, pngs[start:start+2]):
                col.image(str(p), caption=p.stem.replace('_', ' '), use_container_width=True)

    st.warning('Inspect the registration overlays in the longitudinal results ZIP before thesis/publication use. Rejected field pairs are not included in growth calculations. The 12 fields are sampling units from the same array, not independent biological replicates.')
