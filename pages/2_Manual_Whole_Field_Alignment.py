from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from manual_field_alignment import (
    ManualTransform,
    affine_day7_to_day10,
    alpha_overlay,
    dataframe_csv_bytes,
    draw_markers,
    mutual_nearest_well_matches,
    pdo_pattern_overlay,
    png_bytes,
    transform_points,
    warp_day7_to_day10,
)

st.set_page_config(page_title="Manual Whole-Field Alignment", page_icon="🧭", layout="wide")
st.title("Manual Whole-Field Alignment")
st.caption(
    "Visually align the whole Day 7 field onto the larger Day 10 field, then map only the wells you are confident are physically the same."
)

st.info(
    "This page does not assume that well indices match between days. Day 7 is transformed into Day 10 coordinates using scale, rotation and x/y translation chosen by you."
)


def _load_result_zip(uploaded) -> dict:
    if uploaded is None:
        return {}
    raw = uploaded.getvalue()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = set(zf.namelist())
    required = {
        "csv/image_summary.csv",
        "csv/well_raw_data.csv",
        "csv/PDO_raw_data.csv",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Results ZIP is missing: {', '.join(missing)}")

    image_summary = pd.read_csv(zf.open("csv/image_summary.csv"))
    wells = pd.read_csv(zf.open("csv/well_raw_data.csv"))
    pdos = pd.read_csv(zf.open("csv/PDO_raw_data.csv"))

    raw_images: dict[int, np.ndarray] = {}
    for _, r in image_summary.iterrows():
        series = int(r["image_series"])
        export = str(r["raw_image_export"])
        path = f"raw_images/{export}"
        if path not in names:
            continue
        raw_images[series] = np.asarray(Image.open(zf.open(path)).convert("RGB"))

    return {
        "image_summary": image_summary,
        "wells": wells,
        "pdos": pdos,
        "raw_images": raw_images,
    }


@st.cache_data(show_spinner=False)
def _load_cached(raw: bytes) -> dict:
    class _U:
        def __init__(self, b):
            self._b = b
        def getvalue(self):
            return self._b
    return _load_result_zip(_U(raw))


st.subheader("1. Upload the two completed analysis ZIPs")
c1, c2 = st.columns(2)
with c1:
    day7_upload = st.file_uploader("Day 7 results ZIP", type=["zip"], key="manual_align_day7")
with c2:
    day10_upload = st.file_uploader("Day 10 results ZIP", type=["zip"], key="manual_align_day10")

if not (day7_upload and day10_upload):
    st.stop()

try:
    day7 = _load_cached(day7_upload.getvalue())
    day10 = _load_cached(day10_upload.getvalue())
except Exception as exc:
    st.error(f"Could not open the result ZIPs: {exc}")
    st.stop()

s7 = day7["image_summary"].copy()
s10 = day10["image_summary"].copy()

st.subheader("2. Choose the field pair")

left, right = st.columns(2)
with left:
    d7_series = st.selectbox(
        "Day 7 field",
        options=[int(x) for x in s7["image_series"].tolist()],
        format_func=lambda x: f"Series {x:02d} — {s7.loc[s7.image_series.eq(x), 'source_image'].iloc[0]}",
    )

suggested_source = str(s7.loc[s7.image_series.eq(d7_series), "source_image"].iloc[0])
matching_d10 = s10.loc[s10["source_image"].astype(str).eq(suggested_source), "image_series"].tolist()
default_d10 = int(matching_d10[0]) if matching_d10 else int(s10.iloc[min(len(s10)-1, max(0, d7_series-1))]["image_series"])

with right:
    d10_options = [int(x) for x in s10["image_series"].tolist()]
    d10_series = st.selectbox(
        "Day 10 field",
        options=d10_options,
        index=d10_options.index(default_d10) if default_d10 in d10_options else 0,
        format_func=lambda x: f"Series {x:02d} — {s10.loc[s10.image_series.eq(x), 'source_image'].iloc[0]}",
    )

img7 = day7["raw_images"].get(int(d7_series))
img10 = day10["raw_images"].get(int(d10_series))
if img7 is None or img10 is None:
    st.error("The selected field is missing its raw image export in one of the ZIPs.")
    st.stop()

w7 = day7["wells"].loc[day7["wells"]["image_series"].eq(int(d7_series))].copy().reset_index(drop=True)
w10 = day10["wells"].loc[day10["wells"]["image_series"].eq(int(d10_series))].copy().reset_index(drop=True)
p7 = day7["pdos"].loc[day7["pdos"]["image_series"].eq(int(d7_series))].copy()
p10 = day10["pdos"].loc[day10["pdos"]["image_series"].eq(int(d10_series))].copy()

um7 = float(s7.loc[s7.image_series.eq(d7_series), "um_per_pixel"].iloc[0])
um10 = float(s10.loc[s10.image_series.eq(d10_series), "um_per_pixel"].iloc[0])
physical_scale = um7 / um10 if um10 > 0 else 1.0

st.caption(
    f"Physical calibration suggests Day 7 should start near scale **{physical_scale:.4f}×** in Day 10 pixel coordinates. You can override it visually."
)

state_key = f"manual_transform_{d7_series}_{d10_series}"
if state_key not in st.session_state:
    st.session_state[state_key] = {
        "scale": float(physical_scale),
        "rotation": 0.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "opacity": 0.5,
    }

st.subheader("3. Slide Day 7 over Day 10")
st.caption(
    "Use the whole pattern of GFP-positive organoids and the microwell array. Day 7 is the movable layer; Day 10 stays fixed."
)

ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
with ctrl1:
    scale = st.number_input(
        "Scale Day 7",
        min_value=0.2,
        max_value=2.0,
        value=float(st.session_state[state_key]["scale"]),
        step=0.002,
        format="%.4f",
        key=f"scale_{state_key}",
    )
with ctrl2:
    rotation = st.number_input(
        "Rotation (°)",
        min_value=-15.0,
        max_value=15.0,
        value=float(st.session_state[state_key]["rotation"]),
        step=0.05,
        format="%.2f",
        key=f"rot_{state_key}",
    )
with ctrl3:
    shift_x = st.number_input(
        "Shift X (Day 10 px)",
        min_value=-2000.0,
        max_value=2000.0,
        value=float(st.session_state[state_key]["shift_x"]),
        step=2.0,
        key=f"sx_{state_key}",
    )
with ctrl4:
    shift_y = st.number_input(
        "Shift Y (Day 10 px)",
        min_value=-2000.0,
        max_value=2000.0,
        value=float(st.session_state[state_key]["shift_y"]),
        step=2.0,
        key=f"sy_{state_key}",
    )
with ctrl5:
    opacity = st.slider(
        "Day 7 opacity",
        0.0,
        1.0,
        float(st.session_state[state_key]["opacity"]),
        0.05,
        key=f"opacity_{state_key}",
    )

st.session_state[state_key] = {
    "scale": scale,
    "rotation": rotation,
    "shift_x": shift_x,
    "shift_y": shift_y,
    "opacity": opacity,
}

if st.button("Reset to physical scale / centred", use_container_width=False):
    st.session_state[state_key] = {
        "scale": float(physical_scale),
        "rotation": 0.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "opacity": 0.5,
    }
    st.rerun()

transform = ManualTransform(scale, rotation, shift_x, shift_y)
matrix = affine_day7_to_day10(img7.shape, img10.shape, transform)
warped7, footprint = warp_day7_to_day10(img7, img10, matrix)
alpha = alpha_overlay(img10, warped7, footprint, opacity)
pattern = pdo_pattern_overlay(img10, warped7, footprint)

p7xy = p7[["centroid_x_px", "centroid_y_px"]].to_numpy(float) if not p7.empty else np.empty((0, 2))
p10xy = p10[["centroid_x_px", "centroid_y_px"]].to_numpy(float) if not p10.empty else np.empty((0, 2))
p7t = transform_points(p7xy, matrix)
marker_view = draw_markers(alpha, p7t, p10xy, radius=7)

mode = st.radio(
    "Alignment view",
    ["Alpha overlay", "PDO pattern: Day 7 magenta / Day 10 green", "Overlay + detected PDO markers"],
    horizontal=True,
)
if mode == "Alpha overlay":
    shown = alpha
elif mode.startswith("PDO pattern"):
    shown = pattern
else:
    shown = marker_view

st.image(shown, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Day 7 PDOs in selected field", len(p7))
m2.metric("Day 10 PDOs in selected field", len(p10))
m3.metric("Day 7 wells", len(w7))
m4.metric("Day 10 wells", len(w10))

st.subheader("4. Save this alignment and map physical wells")
match_table = mutual_nearest_well_matches(w7, w10, matrix)

if match_table.empty:
    st.warning("No mutual nearest well matches were found for this transform.")
    st.stop()

max_err_default = min(20.0, max(3.0, float(match_table["match_error_px"].quantile(0.5)) * 1.5))
max_error_px = st.slider(
    "Only call a well pair confident when geometric error is ≤ (Day 10 px)",
    min_value=1.0,
    max_value=40.0,
    value=float(max_err_default),
    step=0.5,
)
match_table["confident_geometry"] = match_table["match_error_px"] <= max_error_px

match_table.insert(0, "day7_series", int(d7_series))
match_table.insert(1, "day10_series", int(d10_series))
match_table.insert(2, "day7_source_image", str(s7.loc[s7.image_series.eq(d7_series), "source_image"].iloc[0]))
match_table.insert(3, "day10_source_image", str(s10.loc[s10.image_series.eq(d10_series), "source_image"].iloc[0]))
match_table["manual_scale"] = float(scale)
match_table["manual_rotation_deg"] = float(rotation)
match_table["manual_shift_x_px"] = float(shift_x)
match_table["manual_shift_y_px"] = float(shift_y)

confident = match_table.loc[match_table["confident_geometry"]].copy()

q1, q2, q3 = st.columns(3)
q1.metric("Mutual nearest well pairs", len(match_table))
q2.metric("Within chosen error", len(confident))
q3.metric("Median error", f"{match_table['match_error_px'].median():.1f} px")

st.dataframe(
    match_table[
        [
            "day7_well_index", "day10_well_index", "match_error_px", "match_error_um",
            "day7_PDO_count", "day10_PDO_count", "day7_PSC_count", "day10_PSC_count",
            "confident_geometry",
        ]
    ],
    hide_index=True,
    use_container_width=True,
)

transform_record = {
    "day7_series": int(d7_series),
    "day10_series": int(d10_series),
    "day7_source_image": str(s7.loc[s7.image_series.eq(d7_series), "source_image"].iloc[0]),
    "day10_source_image": str(s10.loc[s10.image_series.eq(d10_series), "source_image"].iloc[0]),
    "manual_scale": float(scale),
    "manual_rotation_deg": float(rotation),
    "manual_shift_x_px": float(shift_x),
    "manual_shift_y_px": float(shift_y),
    "matrix_2x3": matrix.tolist(),
    "confidence_error_cutoff_px": float(max_error_px),
}

b1, b2, b3 = st.columns(3)
with b1:
    st.download_button(
        "Download aligned overlay PNG",
        png_bytes(shown),
        f"Day7_series_{d7_series:02d}_onto_Day10_series_{d10_series:02d}_overlay.png",
        "image/png",
        use_container_width=True,
    )
with b2:
    st.download_button(
        "Download confident matched wells CSV",
        dataframe_csv_bytes(confident),
        f"Day7_series_{d7_series:02d}_Day10_series_{d10_series:02d}_confident_well_map.csv",
        "text/csv",
        use_container_width=True,
        type="primary",
    )
with b3:
    st.download_button(
        "Download transform JSON",
        json.dumps(transform_record, indent=2).encode("utf-8"),
        f"Day7_series_{d7_series:02d}_Day10_series_{d10_series:02d}_manual_transform.json",
        "application/json",
        use_container_width=True,
    )

st.warning(
    "Use the organoid pattern to decide whether the whole-field alignment is correct before accepting the well map. The geometric cutoff only filters the already manually aligned field; it does not prove biological identity on its own."
)
