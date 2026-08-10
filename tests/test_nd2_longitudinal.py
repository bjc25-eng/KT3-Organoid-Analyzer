import numpy as np
import pandas as pd

from nd2_longitudinal import _balanced_rgb
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
