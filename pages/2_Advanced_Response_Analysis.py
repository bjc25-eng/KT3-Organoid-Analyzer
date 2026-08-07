from pathlib import Path
import io
import tempfile
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from advanced_analysis import (
    ANALYSIS_OPTIONS,
    ANALYSIS_GROWTH,
    ANALYSIS_CLASSIFICATION,
    ANALYSIS_PSC,
    run_selected_analyses,
)

st.set_page_config(page_title='Advanced Response Analysis', page_icon='📊', layout='wide')
st.title('Advanced Response Analysis')
st.caption('Choose which high-content longitudinal analyses to run. The underlying microwell-level data are retained rather than reduced to a single bulk value.')

with st.expander('How this page uses the data', expanded=False):
    st.markdown('''
- The primary longitudinal response variable is **total projected PDO area per microwell**, normalised to that same well's first measured time point.
- This avoids assuming that a segmented PDO object remains one-to-one identifiable if it splits, merges or changes shape over time.
- Equivalent circular diameter remains available in the raw output, but area is used for growth-response calculations.
- RFP stromal-cell analysis uses automated **PSC-like red fluorescent foci**, not definitive individual-cell counts.
- Well-level measurements are high-content observations, **not independent biological replicates**. Statistical inference across experiments should retain the biological replicate/device structure.
''')

# Use the most recent longitudinal run when available. A CSV upload fallback lets
# users reopen an exported experiment later without rerunning image segmentation.
tracking = None
source_label = None
if 'long_tracking' in st.session_state and st.session_state['long_tracking']:
    tracking = pd.DataFrame(st.session_state['long_tracking'])
    source_label = 'current longitudinal experiment in this Streamlit session'

st.subheader('1. Load longitudinal tracking data')
if tracking is not None:
    st.success(f'Using the {source_label}.')
    use_upload = st.checkbox('Use an exported tracking CSV instead', False)
else:
    use_upload = True
    st.info('Run a longitudinal experiment on the main page first, or upload the exported `well_longitudinal_tracking.csv` file here.')

uploaded = None
if use_upload:
    uploaded = st.file_uploader('Upload well_longitudinal_tracking.csv', type=['csv'])
    if uploaded is not None:
        tracking = pd.read_csv(uploaded)
        source_label = uploaded.name

if tracking is None or not len(tracking):
    st.stop()

required = {'condition_index', 'condition', 'timepoint_index', 'timepoint', 'well_index', 'total_PDO_projected_area_um2'}
missing = required - set(tracking.columns)
if missing:
    st.error('The tracking table is missing required columns: ' + ', '.join(sorted(missing)))
    st.stop()

st.caption(f'Loaded **{len(tracking):,} well × time observations** from {source_label}.')

# Build editable metadata tables. This is deliberately explicit: concentration
# and elapsed time are experimental metadata and must never be guessed from names.
st.subheader('2. Experimental metadata')

cond_meta = tracking[['condition_index', 'condition']].drop_duplicates().sort_values('condition_index').copy()
if 'concentration' in tracking.columns:
    existing_dose = tracking.groupby('condition_index')['concentration'].first()
    cond_meta['concentration'] = cond_meta['condition_index'].map(existing_dose)
else:
    cond_meta['concentration'] = np.nan
cond_meta['concentration'] = pd.to_numeric(cond_meta['concentration'], errors='coerce')

st.markdown('**Condition metadata**')
st.caption('Enter a numeric drug/therapeutic concentration only when the experiment is a concentration series. Leave it blank for non-dose conditions.')
cond_meta_edit = st.data_editor(
    cond_meta,
    hide_index=True,
    use_container_width=True,
    disabled=['condition_index', 'condition'],
    column_config={
        'condition_index': st.column_config.NumberColumn('Lane'),
        'condition': st.column_config.TextColumn('Condition'),
        'concentration': st.column_config.NumberColumn('Concentration', min_value=0.0, format='%.6g'),
    },
    key='advanced_condition_metadata'
)

time_meta = tracking[['timepoint_index', 'timepoint']].drop_duplicates().sort_values('timepoint_index').copy()
if 'elapsed_time' in tracking.columns:
    existing_time = tracking.groupby('timepoint_index')['elapsed_time'].first()
    time_meta['elapsed_time'] = time_meta['timepoint_index'].map(existing_time)
else:
    time_meta['elapsed_time'] = time_meta['timepoint_index'].astype(float) - float(time_meta['timepoint_index'].min())

st.markdown('**Time metadata**')
st.caption('Enter the actual elapsed time for each imaging point. These values are used for growth-rate calculations.')
time_meta_edit = st.data_editor(
    time_meta,
    hide_index=True,
    use_container_width=True,
    disabled=['timepoint_index', 'timepoint'],
    column_config={
        'timepoint_index': st.column_config.NumberColumn('Order'),
        'timepoint': st.column_config.TextColumn('Time point'),
        'elapsed_time': st.column_config.NumberColumn('Elapsed time', min_value=0.0, format='%.6g'),
    },
    key='advanced_time_metadata'
)

u1, u2 = st.columns(2)
with u1:
    dose_unit = st.text_input('Concentration unit', 'nM')
with u2:
    time_unit = st.text_input('Elapsed-time unit', 'days')

# Attach user-supplied metadata to every well/time row.
working = tracking.drop(columns=[c for c in ['concentration', 'elapsed_time'] if c in tracking.columns], errors='ignore')
working = working.merge(cond_meta_edit[['condition_index', 'concentration']], on='condition_index', how='left')
working = working.merge(time_meta_edit[['timepoint_index', 'elapsed_time']], on='timepoint_index', how='left')

st.subheader('3. Select analyses')
selected = st.multiselect(
    'Analysis modules',
    ANALYSIS_OPTIONS,
    default=ANALYSIS_OPTIONS,
    help='Select any combination. Only the selected modules are generated and included in the results ZIP.'
)

with st.expander('What each module outputs', expanded=False):
    st.markdown('''
**Dose-response / growth-rate curve**  
Tracks baseline-normalised total PDO area through time for every condition. If numeric concentrations are supplied, it also produces an endpoint concentration-response plot and attempts a 4-parameter logistic fit only when the data contain enough distinct concentrations. The fitted midpoint is labelled as an **imaging-response midpoint**, not automatically a GI50/IC50.

**Individual PDO growth waterfall**  
One bar per tracked microwell, sorted from greatest regression/inhibition to greatest growth. This exposes the resistant tail that a bulk average can hide.

**Well × time response heatmap**  
Rows are fixed microwell indices and columns are imaging time points. Wells are sorted by their final response, making persistent growth, early inhibition and regrowth patterns visible.

**Full response distributions for every condition**  
Outputs empirical cumulative distribution curves through time and a final-condition violin plot, preserving population heterogeneity rather than reporting only means.

**Resistant/responding population analysis**  
Classifies tracked wells using **thresholds supplied by the user**. No biological cut-off is assumed by the software.

**PSC-associated drug-response analysis**  
For conditions containing RFP stromal cells, relates PSC-like focus burden to the baseline-normalised PDO response and reports exploratory Spearman associations with Benjamini-Hochberg FDR correction across the condition/time tests.
''')

responder_max = resistant_min = None
if ANALYSIS_CLASSIFICATION in selected:
    st.markdown('**Responder / resistant classification thresholds**')
    st.warning('These thresholds are experimental definitions. The app will not choose them automatically.')
    t1, t2 = st.columns(2)
    with t1:
        responder_max = st.number_input(
            'Responder: final PDO area change ≤ (%)',
            value=-20.0, step=5.0,
            help='Example value shown for interface convenience only; replace it with your experimentally justified definition.'
        )
    with t2:
        resistant_min = st.number_input(
            'Resistant: final PDO area change ≥ (%)',
            value=20.0, step=5.0,
            help='Example value shown for interface convenience only; replace it with your experimentally justified definition.'
        )
    if responder_max >= resistant_min:
        st.error('The responder threshold must be lower than the resistant threshold.')

if ANALYSIS_GROWTH in selected:
    dose_count = pd.to_numeric(cond_meta_edit['concentration'], errors='coerce').notna().sum()
    if dose_count == 0:
        st.info('No concentrations are currently entered. The growth-rate/trajectory analysis will still run, but the dose-response fit will be skipped.')

if ANALYSIS_PSC in selected and 'RFP_PSC_stromal_cells_present' in working.columns:
    if not working['RFP_PSC_stromal_cells_present'].fillna(False).astype(bool).any():
        st.info('No condition is marked as containing RFP PSC/stromal cells. The PSC-association module will be skipped.')

can_run = bool(selected) and not (ANALYSIS_CLASSIFICATION in selected and responder_max >= resistant_min)
run = st.button('Run selected advanced analyses', type='primary', use_container_width=True, disabled=not can_run)

if run:
    root = Path(tempfile.mkdtemp(prefix='kt3_advanced_'))
    out = root/'advanced_analysis'
    with st.spinner('Running selected analyses…'):
        manifest = run_selected_analyses(
            working,
            out,
            selected,
            dose_unit=dose_unit,
            time_unit=time_unit,
            responder_max_pct=responder_max,
            resistant_min_pct=resistant_min,
        )
        working.to_csv(out/'analysis_input_with_metadata.csv', index=False)
        cond_meta_edit.to_csv(out/'condition_metadata.csv', index=False)
        time_meta_edit.to_csv(out/'timepoint_metadata.csv', index=False)

        bio = io.BytesIO()
        with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in out.rglob('*'):
                if p.is_file():
                    zf.write(p, p.relative_to(out))
        st.session_state['advanced_zip'] = bio.getvalue()
        st.session_state['advanced_out'] = str(out)
        st.session_state['advanced_manifest'] = manifest.to_dict('records')
    st.success('Selected analyses complete.')

if 'advanced_manifest' in st.session_state:
    out = Path(st.session_state['advanced_out'])
    manifest = pd.DataFrame(st.session_state['advanced_manifest'])
    st.divider()
    st.subheader('Analysis results')
    st.dataframe(manifest, use_container_width=True, hide_index=True)
    st.download_button(
        'Download selected analysis results ZIP',
        st.session_state['advanced_zip'],
        'PDO_PSC_advanced_response_analysis.zip',
        'application/zip',
        type='primary',
        use_container_width=True,
    )

    pngs = sorted(out.rglob('*.png'))
    if pngs:
        st.subheader('Generated figures')
        for start in range(0, len(pngs), 2):
            cols = st.columns(min(2, len(pngs)-start))
            for col, p in zip(cols, pngs[start:start+2]):
                col.image(str(p), caption=p.stem.replace('_', ' '), use_container_width=True)

    with st.expander('Generated CSV files'):
        csvs = sorted(out.rglob('*.csv'))
        st.write([str(p.relative_to(out)) for p in csvs])
