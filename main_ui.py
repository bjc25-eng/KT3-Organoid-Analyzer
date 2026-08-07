from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis_core import (
    APP_TITLE, GFP_MODE, BRIGHTFIELD_MODE, PSC_PRESENT, PSC_ABSENT,
    build_settings_from_widgets, process, process_experiment, zip_bytes
)


def _analysis_sidebar():
    st.sidebar.header('Analysis settings')
    well = st.sidebar.number_input('Microwell diameter (µm)', 1.0, 1000.0, 100.0, 1.0)
    split = st.sidebar.checkbox('Split touching PDOs', True)
    bins = st.sidebar.slider('Histogram bins', 5, 30, 12)
    cols = st.sidebar.slider('Contact-sheet columns', 3, 10, 5)
    with st.sidebar.expander('Advanced thresholds'):
        rmin = st.number_input('Minimum well radius (px)', 5, 200, 23, step=1, key='rmin')
        rmax = st.number_input('Maximum well radius (px)', 6, 300, 40, step=1, key='rmax')
        spacing = st.number_input('Minimum well spacing (px)', 10, 500, 54, step=1, key='spacing')
        hp2 = st.number_input('Well detection sensitivity', 1.0, 100.0, 27.0, 1.0, key='hp2')
        gl = st.number_input('GFP PDO low threshold', 0.0, 255.0, 30.0, 1.0, key='gl')
        gh = st.number_input('GFP PDO high threshold', 0.0, 255.0, 45.0, 1.0, key='gh')
        amin = st.number_input('Minimum GFP PDO area (px²)', 1, 100000, 20, step=1, key='amin')
        pdist = st.number_input('PDO split peak distance (px)', 1, 100, 18, step=1, key='pdist')
        bf_contrast = st.number_input('Unlabelled PDO contrast threshold', 0.5, 100.0, 10.0, 0.5, key='bfcontrast')
        bf_min_area = st.number_input('Unlabelled PDO minimum area (px²)', 5, 100000, 80, step=5, key='bfarea')
        pt = st.number_input('PSC focus threshold', 0.0, 255.0, 9.0, 0.5, key='pt')
        prm = st.number_input('PSC red-minus-blue minimum', 0.0, 255.0, 12.0, 0.5, key='prm')
        ppd = st.number_input('PSC focus minimum spacing (px)', 1, 100, 4, step=1, key='ppd')
    settings = build_settings_from_widgets(
        well, rmin, rmax, spacing, hp2, gl, gh, amin, split, pdist,
        pt, prm, ppd, bins, bf_contrast, bf_min_area
    )
    return settings, cols


def _render_longitudinal(base_settings, cols):
    st.subheader('Experiment identity')
    st.caption('These fields become stable metadata in the machine-learning export.')
    a, b, c, d = st.columns(4)
    with a:
        experiment_id = st.text_input('Experiment ID', 'Experiment_001')
    with b:
        device_id = st.text_input('Array / device ID', 'Array_001')
    with c:
        replicate_id = st.text_input('Biological replicate ID', 'Replicate_1')
    with d:
        pdo_model = st.text_input('PDO model / patient / line', '')

    st.subheader('Experiment layout')
    a, b, c = st.columns(3)
    with a:
        n_conditions = st.selectbox('Number of lanes / conditions', list(range(1, 13)), index=5)
    with b:
        n_timepoints = st.selectbox('Number of imaging time points', list(range(1, 13)), index=3)
    with c:
        time_unit = st.selectbox('Time unit', ['hours', 'days'], index=1)

    timepoints, elapsed = [], []
    with st.expander('Time-point labels and elapsed time', expanded=True):
        for start in range(0, n_timepoints, 4):
            inds = list(range(start, min(start+4, n_timepoints)))
            cc = st.columns(len(inds))
            for col, t in zip(cc, inds):
                with col:
                    timepoints.append(st.text_input(f'Time point {t+1}', f'Day {t}', key=f'time_{t}'))
                    elapsed.append(st.number_input(f'Elapsed {time_unit}', min_value=0.0, value=float(t), key=f'elapsed_{t}'))

    conditions = []
    with st.expander('Lane / condition setup', expanded=True):
        for i in range(n_conditions):
            st.markdown(f'**Lane {i+1}**')
            a, b, c = st.columns([1.2, 1.0, 1.0])
            with a:
                name = st.text_input('Condition name', f'Condition {i+1}', key=f'name_{i}')
            with b:
                omode = st.selectbox('Organoid detection', [GFP_MODE, BRIGHTFIELD_MODE], key=f'omode_{i}')
            with c:
                pmode = st.selectbox('RFP stromal cells', [PSC_PRESENT, PSC_ABSENT], key=f'pmode_{i}')
            d, e, f = st.columns([1.2, 0.8, 0.8])
            with d:
                drug = st.text_input('Drug / therapeutic', '', key=f'drug_{i}')
            with e:
                conc = st.number_input('Concentration', min_value=0.0, value=0.0, format='%.6g', key=f'conc_{i}')
            with f:
                unit = st.selectbox('Unit', ['nM', 'µM', 'ng/mL', 'µg/mL', 'other'], key=f'unit_{i}')
            conditions.append({
                'condition_index': i+1, 'condition': name,
                'organoid_mode': omode, 'rfp_psc_present': pmode == PSC_PRESENT,
                'drug_or_therapeutic': drug, 'concentration': float(conc),
                'concentration_unit': unit,
            })
            if i < n_conditions-1:
                st.divider()

    st.subheader('Upload condition × time-point images')
    st.caption('Each upload box may contain several microscope fields. Field identity is retained in the stable trajectory ID.')
    entries = []
    for cond in conditions:
        with st.expander(f"Lane {cond['condition_index']}: {cond['condition']}", expanded=(cond['condition_index'] == 1)):
            for start in range(0, n_timepoints, 4):
                inds = list(range(start, min(start+4, n_timepoints)))
                cc = st.columns(len(inds))
                for col, t in zip(cc, inds):
                    with col:
                        files = st.file_uploader(
                            timepoints[t], type=['png','jpg','jpeg','tif','tiff','bmp'],
                            accept_multiple_files=True, key=f"upload_{cond['condition_index']}_{t}"
                        )
                        entries.append({
                            **cond, 'timepoint_index': t+1, 'timepoint': timepoints[t],
                            'elapsed_time': float(elapsed[t]), 'time_unit': time_unit, 'files': files
                        })

    filled = sum(bool(e['files']) for e in entries)
    st.info(f'{filled} of {n_conditions*n_timepoints} condition × time-point upload boxes contain images.')

    st.subheader('Machine Learning / Virtual Model Export')
    make_ml = st.checkbox('Create ML / Virtual Model Export package', value=True)
    with st.expander('What is included'):
        st.markdown('''
- original uploaded microscopy images
- automated **well masks**, **PDO semantic masks**, and **PSC-like focus point masks**
- stable `image_uid`, `trajectory_id`, `well_observation_id`, PDO IDs and PSC-focus IDs
- experiment, device/array, biological-replicate and PDO-model metadata
- drug/therapeutic, concentration, units and elapsed time
- longitudinal well features, PDO-object features and PSC-focus coordinates/scores
- QC flags marking automated, not-yet-manually-reviewed outputs
- schema/manifest files and SHA-256 hashes for exported assets
''')

    run = st.button('Run longitudinal experiment analysis', type='primary', use_container_width=True, disabled=filled == 0)
    if run:
        metadata = {
            'experiment_id': experiment_id,
            'device_id': device_id,
            'biological_replicate_id': replicate_id,
            'pdo_model': pdo_model,
            'time_unit': time_unit,
        }
        bar = st.progress(5, text='Starting longitudinal analysis…')
        try:
            root, out, summary, tracking, ml_path = process_experiment(
                entries, base_settings, int(cols), metadata, make_ml_export=make_ml
            )
            bar.progress(100, text='Complete')
            st.session_state['long_zip'] = zip_bytes(out)
            st.session_state['long_out'] = str(out)
            st.session_state['long_summary'] = summary.to_dict('records')
            st.session_state['long_tracking'] = tracking.to_dict('records')
            if ml_path is not None:
                st.session_state['ml_zip'] = zip_bytes(ml_path)
                st.session_state['ml_path'] = str(ml_path)
            st.success('Longitudinal experiment analysis complete.')
        except Exception as exc:
            bar.empty()
            st.error(f'Analysis stopped: {exc}')

    if 'long_zip' not in st.session_state:
        return

    out = Path(st.session_state['long_out'])
    summary = pd.DataFrame(st.session_state['long_summary'])
    tracking = pd.DataFrame(st.session_state['long_tracking'])
    st.divider()
    st.subheader('Longitudinal results')
    a, b = st.columns(2)
    with a:
        st.download_button('Download complete longitudinal results ZIP', st.session_state['long_zip'], 'PDO_PSC_longitudinal_experiment_results.zip', 'application/zip', use_container_width=True)
    with b:
        if 'ml_zip' in st.session_state:
            st.download_button('Download ML / Virtual Model Export ZIP', st.session_state['ml_zip'], 'PDO_PSC_ML_virtual_model_export.zip', 'application/zip', type='primary', use_container_width=True)

    fdir = out/'figures'
    figs = [
        ('Mean PDO diameter by condition','condition_comparison_mean_PDO_diameter.png'),
        ('PDO-containing wells by condition','condition_comparison_PDO_occupancy.png'),
        ('PDO count by condition','condition_comparison_PDO_count.png'),
        ('PSC-like foci by condition','condition_comparison_PSC_foci.png'),
    ]
    available = [(label, fdir/name) for label, name in figs if (fdir/name).exists()]
    for start in range(0, len(available), 2):
        cc = st.columns(min(2, len(available)-start))
        for col, (label, path) in zip(cc, available[start:start+2]):
            col.image(str(path), caption=label, use_container_width=True)

    with st.expander('Condition × time-point summary', expanded=True):
        cols_show = [c for c in [
            'condition','drug_or_therapeutic','concentration','concentration_unit','timepoint','elapsed_time',
            'organoid_detection_mode','RFP_PSC_stromal_cells_present','fully_visible_wells',
            'PDO_containing_wells','PDO_containing_well_percentage','PDO_count','mean_PDO_diameter_um',
            'mean_PSC_foci_in_PDO_wells'
        ] if c in summary.columns]
        st.dataframe(summary[cols_show], use_container_width=True, hide_index=True)

    with st.expander('Well-by-well longitudinal tracking'):
        st.caption('`trajectory_id` remains constant across time and includes experiment, array, lane, field and well x,y.')
        st.dataframe(tracking, use_container_width=True, hide_index=True)


def _render_single(base_settings, cols):
    st.subheader('Single / batch analysis')
    a, b = st.columns(2)
    with a:
        omode = st.selectbox('Are the organoids GFP-labelled?', [GFP_MODE, BRIGHTFIELD_MODE])
    with b:
        pmode = st.selectbox('Are RFP-labelled PSC/stromal cells present?', [PSC_PRESENT, PSC_ABSENT])
    files = st.file_uploader('Upload large microscopy images', type=['png','jpg','jpeg','tif','tiff','bmp'], accept_multiple_files=True, key='single_files')
    run = st.button('Run analysis', type='primary', use_container_width=True, disabled=not files)
    if run:
        s = replace(base_settings, organoid_mode=omode, rfp_psc_present=pmode == PSC_PRESENT)
        try:
            _, out, summary, image_summary = process(files, s, int(cols))
            st.session_state['zip'] = zip_bytes(out)
            st.session_state['out'] = str(out)
            st.session_state['summary'] = summary.to_dict('records')
            st.session_state['idf'] = image_summary.to_dict('records')
            st.success('Analysis complete.')
        except Exception as exc:
            st.error(f'Analysis stopped: {exc}')
    if 'zip' in st.session_state:
        st.download_button('Download complete results ZIP', st.session_state['zip'], 'KT3_PDO_PSC_analysis_results.zip', 'application/zip', type='primary', use_container_width=True)


def render_app():
    st.set_page_config(page_title=APP_TITLE, page_icon='🔬', layout='wide')
    st.title(APP_TITLE)
    st.caption('Analyze PDOs and stromal cells longitudinally and export structured datasets for high-content analysis, machine learning and virtual-model development.')
    with st.expander('Measurement, tracking and export notes'):
        st.markdown('''
- PDO size is a **2D equivalent circular diameter** from projected segmented area.
- PSC counts are **PSC-like red fluorescent foci**, not definitive individual-cell counts.
- Longitudinal tracking includes **field of view** as well as well x,y to prevent field collisions.
- ML masks are automated and explicitly marked as **not manually reviewed**.
''')
    workflow = st.sidebar.selectbox('Workflow', ['Longitudinal multi-condition experiment','Single / batch analysis'], index=0)
    settings, cols = _analysis_sidebar()
    if workflow == 'Longitudinal multi-condition experiment':
        _render_longitudinal(settings, cols)
    else:
        _render_single(settings, cols)
