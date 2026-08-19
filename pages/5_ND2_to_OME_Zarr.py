from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from nd2_large_source import probe_nd2_source
from nd2_omezarr import convert_nd2_to_omezarr, probe_omezarr


st.set_page_config(page_title='ND2 → OME-Zarr', page_icon='🧱', layout='wide')
st.title('ND2 → OME-Zarr Conversion')
st.caption(
    'Convert a staged Nikon ND2 once into a chunked OME-Zarr representation for efficient whole-array analysis.'
)

cache_root = st.text_input(
    'Persistent ND2 cache root',
    '/home/ec2-user/kt3_nd2_cache',
    help='This should be the same EBS-backed cache root used on the Nikon ND2 Whole Array page.',
)

root = Path(cache_root).expanduser()
nd2_files = sorted(root.glob('*/input/*.nd2')) if root.exists() else []

if not nd2_files:
    st.warning('No staged ND2 files were found under this cache root. Stage the ND2 first on the Nikon ND2 Whole Array page.')
    st.stop()

labels = [f'{p.name}  —  {p.parent.parent.name}' for p in nd2_files]
selected_label = st.selectbox('Staged ND2', labels)
selected = nd2_files[labels.index(selected_label)]
st.caption(str(selected))

st.subheader('1. Probe source')
position = st.number_input('ND2 XY position / series', min_value=0, value=0, step=1)

if st.button('Probe staged ND2', use_container_width=True):
    try:
        meta = probe_nd2_source(str(selected), int(position))
        st.session_state['convert_nd2_meta'] = meta
        st.success(
            f"ND2 detected — {meta['width_px']:,} × {meta['height_px']:,} px, "
            f"{meta['channel_count']} channels."
        )
    except Exception as exc:
        st.error(f'Could not probe ND2: {type(exc).__name__}: {exc!s}')

meta = st.session_state.get('convert_nd2_meta')
if meta:
    a, b, c, d = st.columns(4)
    a.metric('Width', f"{int(meta['width_px']):,} px")
    b.metric('Height', f"{int(meta['height_px']):,} px")
    c.metric('Channels', int(meta['channel_count']))
    d.metric('Native frame', f"{float(meta.get('estimated_native_frame_mib', 0)):.1f} MiB")

    channels = pd.DataFrame(meta.get('channel_metadata', []))
    if not channels.empty:
        st.dataframe(channels, use_container_width=True, hide_index=True)
    voxel = meta.get('voxel_size_um') or {}
    st.caption(f"ND2 pixel size: X={voxel.get('x')} µm, Y={voxel.get('y')} µm")

st.subheader('2. Convert once to chunked OME-Zarr')
st.info(
    'For this large stitched ND2 the conversion uses one worker and 1024-pixel tiles to keep memory use conservative. '
    'The original ND2 is not modified or deleted.'
)

overwrite = st.checkbox('Overwrite an existing converted copy', False)
convert = st.button(
    'Convert staged ND2 → OME-Zarr',
    type='primary',
    use_container_width=True,
    disabled=meta is None,
)

if convert and meta is not None:
    status = st.empty()
    log_box = st.empty()
    recent: list[str] = []

    def on_line(line: str):
        recent.append(line)
        del recent[:-12]
        status.caption('bioformats2raw conversion is running…')
        log_box.code('\n'.join(recent), language='text')

    try:
        result = convert_nd2_to_omezarr(
            selected,
            meta,
            series_index=int(position),
            max_workers=1,
            tile_size=1024,
            resolutions=1,
            overwrite=overwrite,
            line_callback=on_line,
        )
        st.session_state['converted_omezarr'] = result
        status.empty()
        st.success('OME-Zarr conversion and validation completed successfully.')
    except Exception as exc:
        status.empty()
        st.error(f'Conversion failed: {type(exc).__name__}: {exc!s}')

result = st.session_state.get('converted_omezarr')
if result:
    converted_meta = result['metadata']
    st.subheader('3. Validated converted dataset')
    st.success(f"Ready: {result['output_path']}")
    a, b, c, d = st.columns(4)
    a.metric('Width', f"{int(converted_meta['width_px']):,} px")
    b.metric('Height', f"{int(converted_meta['height_px']):,} px")
    c.metric('Channels', int(converted_meta['channel_count']))
    d.metric('Pyramid levels', int(converted_meta['level_count']))

    zchannels = pd.DataFrame(converted_meta.get('channel_metadata', []))
    if not zchannels.empty:
        st.markdown('**OME-Zarr channels**')
        st.dataframe(zchannels, use_container_width=True, hide_index=True)

    voxel = converted_meta.get('voxel_size_um') or {}
    st.caption(
        f"OME-Zarr pixel size: X={voxel.get('x')} µm, Y={voxel.get('y')} µm · "
        f"level-0 chunks: {converted_meta.get('level0_chunks')}"
    )

    validation = result.get('validation') or {}
    if validation.get('warnings'):
        for warning in validation['warnings']:
            st.warning(warning)
    st.success('Dimensions, channel count and physical X/Y calibration match the original ND2.')

    with st.expander('Full OME-Zarr metadata'):
        st.json(probe_omezarr(result['output_path']))
