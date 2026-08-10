import py_compile
from pathlib import Path

import numpy as np
import pandas as pd

from nd2_longitudinal import _balanced_rgb, _recalibrate_tracking
from trajectory_safe_analysis import make_trajectory_safe


def test_balanced_rgb_keeps_dic_common_and_separates_fluorescence():
    dic = np.full((8, 8), 40, dtype=np.uint8)
    gfp = np.zeros((8, 8), dtype=np.uint8)
    rfp = np.zeros((8, 8), dtype=np.uint8)
    gfp[2, 3] = 100
    rfp[5, 6] = 80
    rgb = _balanced_rgb(dic, gfp, rfp)
    assert rgb.shape == (8, 8, 3)
    assert tuple(rgb[0, 0]) == (40, 40, 40)
    assert int(rgb[2, 3, 1]) > int(rgb[2, 3, 0])
    assert int(rgb[5, 6, 0]) > int(rgb[5, 6, 2])


def test_trajectory_safe_prevents_field_well_collisions():
    df = pd.DataFrame({
        'condition_index': [1, 1, 1, 1],
        'condition': ['KT3 + PSC'] * 4,
        'timepoint_index': [1, 2, 1, 2],
        'timepoint': ['Friday', 'Monday', 'Friday', 'Monday'],
        'well_index': ['4,8', '4,8', '4,8', '4,8'],
        'trajectory_id': ['Field01__W4_8', 'Field01__W4_8', 'Field07__W4_8', 'Field07__W4_8'],
        'total_PDO_projected_area_um2': [100.0, 150.0, 200.0, 220.0],
        'elapsed_time': [0.0, 3.0, 0.0, 3.0],
    })
    safe = make_trajectory_safe(df)
    assert set(safe['well_index']) == {'Field01__W4_8', 'Field07__W4_8'}
    assert set(safe['local_well_index']) == {'4,8'}


def test_recalibration_uses_native_nd2_scale_and_recomputes_growth():
    # Hough-derived physical values below were calculated with 0.5 um/px.
    # Native ND2 metadata says baseline is 0.4 and follow-up is 0.8 um/px.
    df = pd.DataFrame({
        'trajectory_id': ['F01__W1_1', 'F01__W1_1'],
        'timepoint_index': [1, 2],
        'um_per_pixel': [0.5, 0.5],
        'total_PDO_projected_area_um2': [25.0, 25.0],  # 100 px2 at both times
        'mean_PDO_diameter_um': [5.0, 5.0],
        'max_PDO_diameter_um': [5.0, 5.0],
        'baseline_total_PDO_area_um2': [25.0, 25.0],
        'relative_total_PDO_area_vs_baseline': [1.0, 1.0],
    })
    out = _recalibrate_tracking(df, {1: 0.4, 2: 0.8})
    assert np.isclose(out.loc[0, 'total_PDO_projected_area_um2'], 16.0)
    assert np.isclose(out.loc[1, 'total_PDO_projected_area_um2'], 64.0)
    assert np.isclose(out.loc[0, 'um_per_pixel'], 0.4)
    assert np.isclose(out.loc[1, 'um_per_pixel'], 0.8)
    assert np.isclose(out.loc[1, 'relative_total_PDO_area_vs_baseline'], 4.0)


def test_paired_nd2_streamlit_page_compiles():
    page = Path(__file__).resolve().parents[1] / 'pages' / '1_Paired_ND2_Longitudinal.py'
    py_compile.compile(str(page), doraise=True)
