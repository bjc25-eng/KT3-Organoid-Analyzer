import io
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import analysis_core as core


class FakeUpload:
    def __init__(self, name, array):
        self.name = name
        bio = io.BytesIO()
        Image.fromarray(array.astype(np.uint8)).save(bio, format='PNG')
        self._data = bio.getvalue()

    def getbuffer(self):
        return memoryview(self._data)


def synthetic_image():
    rgb = np.full((170, 170, 3), 40, dtype=np.uint8)
    cv2.circle(rgb, (45, 45), 11, (40, 235, 40), -1)
    cv2.circle(rgb, (58, 45), 3, (245, 40, 40), -1)
    return rgb


def fixed_wells():
    return np.array([[45,45,25],[115,45,25],[45,115,25],[115,115,25]], dtype=int)


def test_process_exports_raw_images_and_masks(monkeypatch):
    monkeypatch.setattr(core, 'detect_wells', lambda rgb, s: fixed_wells())
    upload = FakeUpload('test (series 01).png', synthetic_image())
    _, out, _, _ = core.process([upload], core.Settings(), 4)
    assert len(list((out/'raw_images').glob('*'))) == 1
    assert (out/'segmentation_masks'/'series_01__well_mask.png').exists()
    assert (out/'segmentation_masks'/'series_01__pdo_semantic_mask.png').exists()
    assert (out/'segmentation_masks'/'series_01__psc_focus_point_mask.png').exists()
    assert (out/'csv'/'PSC_focus_raw_data.csv').exists()
    assert (out/'csv'/'mask_manifest.csv').exists()


def test_process_experiment_creates_stable_ids_and_ml_package(monkeypatch):
    monkeypatch.setattr(core, 'detect_wells', lambda rgb, s: fixed_wells())
    img = synthetic_image()
    entries = [
        {'condition_index':1,'condition':'Vehicle','organoid_mode':core.GFP_MODE,'rfp_psc_present':True,
         'drug_or_therapeutic':'MRTX1133','concentration':0.0,'concentration_unit':'nM',
         'timepoint_index':1,'timepoint':'Day 0','elapsed_time':0.0,'time_unit':'days',
         'files':[FakeUpload('field (series 01).png', img)]},
        {'condition_index':1,'condition':'Vehicle','organoid_mode':core.GFP_MODE,'rfp_psc_present':True,
         'drug_or_therapeutic':'MRTX1133','concentration':0.0,'concentration_unit':'nM',
         'timepoint_index':2,'timepoint':'Day 2','elapsed_time':2.0,'time_unit':'days',
         'files':[FakeUpload('field (series 01).png', img)]},
    ]
    meta = {'experiment_id':'EXP_A','device_id':'ARRAY_7','biological_replicate_id':'BIO_1','pdo_model':'PDO_X','time_unit':'days'}
    _, out, _, tracking, ml = core.process_experiment(entries, core.Settings(), 4, meta, make_ml_export=True)
    assert ml is not None and ml.exists()
    assert (ml/'README.md').exists()
    assert (ml/'schema.json').exists()
    assert (ml/'dataset_manifest.json').exists()
    for name in ['experiment_metadata.csv','condition_metadata.csv','timepoint_metadata.csv','well_observations.csv',
                 'pdo_observations.csv','psc_focus_observations.csv','longitudinal_trajectories.csv','qc_flags.csv','asset_manifest.csv']:
        assert (ml/'tables'/name).exists(), name
    assert len(list((ml/'assets'/'raw_images').rglob('*.*'))) >= 2
    assert len(list((ml/'assets'/'masks').rglob('*.png'))) >= 6
    assert tracking['trajectory_id'].notna().all()
    first = tracking[tracking.timepoint_index == 1].set_index('well_index')['trajectory_id']
    second = tracking[tracking.timepoint_index == 2].set_index('well_index')['trajectory_id']
    common = first.index.intersection(second.index)
    assert len(common) > 0
    assert (first.loc[common].values == second.loc[common].values).all()
    assert tracking['well_observation_id'].nunique() == len(tracking)


def test_field_of_view_is_part_of_trajectory_id():
    df = pd.DataFrame([
        {'image_series':1,'well_index':'3,4','well_col_index':3,'well_row_index':4},
        {'image_series':2,'well_index':'3,4','well_col_index':3,'well_row_index':4},
    ])
    meta = {'experiment_id':'E','device_id':'A','biological_replicate_id':'R','condition_index':1,
            'condition':'C','timepoint_index':1,'timepoint':'D0','elapsed_time':0,'time_unit':'days',
            'drug_or_therapeutic':'','concentration':0,'concentration_unit':'nM',
            'organoid_detection_mode':core.GFP_MODE,'GFP_labelled_organoids':True,'RFP_PSC_stromal_cells_present':False}
    out = core._make_ids(df, meta)
    assert out.loc[0,'trajectory_id'] != out.loc[1,'trajectory_id']


def test_ml_export_zip_contains_schema(monkeypatch):
    monkeypatch.setattr(core, 'detect_wells', lambda rgb, s: fixed_wells())
    entry = {'condition_index':1,'condition':'Control','organoid_mode':core.GFP_MODE,'rfp_psc_present':False,
             'drug_or_therapeutic':'','concentration':0.0,'concentration_unit':'nM',
             'timepoint_index':1,'timepoint':'Day 0','elapsed_time':0.0,'time_unit':'days',
             'files':[FakeUpload('field (series 01).png', synthetic_image())]}
    _, _, _, _, ml = core.process_experiment([entry], core.Settings(rfp_psc_present=False), 4,
                                              {'experiment_id':'E1','device_id':'A1','biological_replicate_id':'R1'}, True)
    payload = core.zip_bytes(ml)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
    assert 'schema.json' in names
    assert 'dataset_manifest.json' in names
    assert 'tables/longitudinal_trajectories.csv' in names
