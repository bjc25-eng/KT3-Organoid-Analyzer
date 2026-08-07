import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import app


class FakeUpload:
    def __init__(self, name: str, array: np.ndarray):
        self.name = name
        bio = io.BytesIO()
        Image.fromarray(array.astype(np.uint8)).save(bio, format="PNG")
        self._data = bio.getvalue()

    def getbuffer(self):
        return memoryview(self._data)


def synthetic_rgb(size=180):
    arr = np.full((size, size, 3), 40, dtype=np.uint8)
    return arr


def fixed_wells():
    return np.array(
        [
            [45, 45, 25],
            [115, 45, 25],
            [45, 115, 25],
            [115, 115, 25],
        ],
        dtype=int,
    )


def test_settings_defaults_are_valid():
    s = app.Settings()
    assert s.well_diameter_um > 0
    assert s.well_rmin < s.well_rmax
    assert s.well_spacing > 0
    assert s.green_low < s.green_high
    assert s.pdo_min_area > 0
    assert s.histogram_bins >= 5


def test_series_parsing_and_natural_sorting():
    assert app.infer_series("KT3 day 7005 (series 09).png", 1) == 9
    assert app.infer_series("unknown.png", 7) == 7
    paths = [Path("series 10.png"), Path("series 2.png"), Path("series 1.png")]
    names = [p.name for p in sorted(paths, key=app.natural_key)]
    assert names == ["series 1.png", "series 2.png", "series 10.png"]


def test_detect_wells_coerces_radius_values_to_python_ints(monkeypatch):
    captured = {}

    def fake_hough(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(app.cv2, "HoughCircles", fake_hough)
    s = app.Settings()
    s.well_rmin = 23.0
    s.well_rmax = 40.0
    s.well_spacing = 54.0
    result = app.detect_wells(synthetic_rgb(128), s)

    assert result.shape == (0, 3)
    assert isinstance(captured["minRadius"], int)
    assert isinstance(captured["maxRadius"], int)
    assert captured["minRadius"] == 23
    assert captured["maxRadius"] == 40
    assert captured["minDist"] == 54.0


def test_detect_wells_deduplicates_nearby_hough_hits(monkeypatch):
    fake = np.array([[[50.0, 50.0, 28.0], [55.0, 55.0, 29.0], [120.0, 50.0, 28.0]]], dtype=np.float32)
    monkeypatch.setattr(app.cv2, "HoughCircles", lambda *a, **k: fake)
    wells = app.detect_wells(synthetic_rgb(180), app.Settings())
    assert wells.dtype.kind in "iu"
    assert len(wells) == 2


def test_cluster_and_grid_index():
    xs = app.cluster([10, 11, 69, 70, 130], tol=3)
    ys = app.cluster([20, 21, 80, 81], tol=3)
    assert len(xs) == 3
    assert len(ys) == 2
    assert app.grid_index(71, 79, xs, ys) == (2, 2)


def test_green_excess_prefers_green_signal():
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[15:25, 15:25, 1] = 200
    g = app.green_excess(rgb)
    assert g[20, 20] > 100
    assert g[0, 0] < 1


def test_segment_pdos_returns_empty_without_green_signal():
    green = np.zeros((100, 100), dtype=np.float32)
    assert app.segment_pdos(green, app.Settings()) == []


def test_segment_pdos_detects_single_object():
    green = np.zeros((100, 100), dtype=np.float32)
    cv2.circle(green, (50, 50), 12, 100.0, -1)
    s = app.Settings(green_low=20, green_high=50, pdo_min_area=20, split_pdos=False)
    objs = app.segment_pdos(green, s)
    assert len(objs) == 1
    assert objs[0]["area"] > 300
    assert abs(objs[0]["x"] - 50) < 2
    assert abs(objs[0]["y"] - 50) < 2


def touching_green_field():
    yy, xx = np.mgrid[0:120, 0:120]
    g1 = 120 * np.exp(-((xx - 48) ** 2 + (yy - 60) ** 2) / (2 * 9.0**2))
    g2 = 120 * np.exp(-((xx - 72) ** 2 + (yy - 60) ** 2) / (2 * 9.0**2))
    return (g1 + g2).astype(np.float32)


def test_touching_pdo_split_can_separate_two_peaks():
    green = touching_green_field()
    s = app.Settings(green_low=20, green_high=55, pdo_min_area=20, split_pdos=True, pdo_peak_distance=10)
    objs = app.segment_pdos(green, s)
    assert len(objs) >= 2


def test_touching_pdo_split_can_be_disabled():
    green = touching_green_field()
    s = app.Settings(green_low=20, green_high=55, pdo_min_area=20, split_pdos=False, pdo_peak_distance=10)
    objs = app.segment_pdos(green, s)
    assert len(objs) == 1


def test_psc_detection_finds_bright_red_focus():
    rgb = synthetic_rgb(120)
    cv2.circle(rgb, (68, 60), 3, (255, 40, 40), -1)
    s = app.Settings(psc_peak_threshold=5.0, psc_red_minimum=20.0, psc_peak_distance=3)
    foci = app.detect_psc(rgb, 60, 60, 30, s)
    assert len(foci) >= 1
    assert any(abs(x - 68) <= 3 and abs(y - 60) <= 3 for x, y, _ in foci)


def test_psc_detection_does_not_invent_focus_in_uniform_image():
    rgb = synthetic_rgb(120)
    foci = app.detect_psc(rgb, 60, 60, 30, app.Settings())
    assert foci == []


def test_crop_square_handles_image_edges():
    rgb = synthetic_rgb(80)
    crop = app.crop_square(rgb, 8, 8, 20, scale=2)
    assert crop.mode == "RGB"
    assert crop.width > 0
    assert crop.height > 0
    assert crop.width == crop.height


def test_labelled_crop_has_header_and_preserves_measurements():
    crop = Image.new("RGB", (240, 240), "gray")
    out = app.labelled_crop(crop, 5, "13,4", 2, 3, [39.08, 32.20])
    assert out.height > crop.height
    assert out.width >= crop.width
    header_pixel = out.getpixel((5, 5))
    assert max(header_pixel) < 30


def test_indexed_overlay_returns_valid_image():
    rgb = synthetic_rgb(120)
    wells = [{"x": 60, "y": 60, "r": 25, "well": "1,1"}]
    out = app.indexed_overlay(rgb, wells)
    assert out.size == (120, 120)
    assert out.mode == "RGB"


def test_make_contact_sheet(tmp_path):
    paths = []
    for i in range(7):
        p = tmp_path / f"crop_{i}.png"
        Image.new("RGB", (120, 150), "white").save(p)
        paths.append(p)
    out = tmp_path / "contact.png"
    app.make_contact(paths, out, cols=3, gap=4)
    assert out.exists()
    im = Image.open(out)
    assert im.width > 0 and im.height > 0


def test_zip_bytes_contains_nested_outputs(tmp_path):
    (tmp_path / "csv").mkdir()
    (tmp_path / "csv" / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "figure.png").write_bytes(b"png")
    payload = app.zip_bytes(tmp_path)
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        names = set(zf.namelist())
    assert "csv/a.csv" in names
    assert "figure.png" in names


def synthetic_pipeline_image(with_pdo=True, with_psc=True):
    rgb = synthetic_rgb(170)
    if with_pdo:
        cv2.circle(rgb, (45, 45), 11, (40, 235, 40), -1)
    if with_psc:
        cv2.circle(rgb, (58, 45), 3, (245, 40, 40), -1)
    return rgb


def test_process_end_to_end_with_pdo_and_psc(monkeypatch):
    monkeypatch.setattr(app, "detect_wells", lambda rgb, s: fixed_wells())
    upload = FakeUpload("KT3 test (series 01).png", synthetic_pipeline_image(True, True))
    root, out, summary, image_summary = app.process([upload], app.Settings(), 4)

    assert int(summary.iloc[0]["images_processed"]) == 1
    assert int(summary.iloc[0]["fully_visible_wells"]) == 4
    assert int(summary.iloc[0]["PDO_count"]) >= 1
    assert (out / "csv" / "well_raw_data.csv").exists()
    assert (out / "csv" / "PDO_raw_data.csv").exists()
    assert (out / "csv" / "overall_summary.csv").exists()
    assert (out / "figures" / "PDO_size_distribution.png").exists()
    assert (out / "figures" / "PSC_count_frequency_across_PDOs.png").exists()
    assert (out / "figures" / "PDO_well_contact_sheet_compact.png").exists()
    assert len(list((out / "labelled_crops").glob("*.png"))) >= 1
    assert len(list((out / "raw_crops").glob("*.png"))) >= 1
    assert len(list((out / "indexed_large_images").glob("*.png"))) == 1

    pdo = pd.read_csv(out / "csv" / "PDO_raw_data.csv")
    assert len(pdo) >= 1
    assert (pdo["equivalent_circular_diameter_um"] > 0).all()
    assert (pdo["PSC_like_focus_count_in_well"] >= 0).all()

    z = app.zip_bytes(out)
    with zipfile.ZipFile(io.BytesIO(z), "r") as zf:
        names = set(zf.namelist())
    assert "csv/overall_summary.csv" in names
    assert "figures/PDO_size_distribution.png" in names


def test_process_handles_zero_pdo_image(monkeypatch):
    monkeypatch.setattr(app, "detect_wells", lambda rgb, s: fixed_wells())
    upload = FakeUpload("blank (series 02).png", synthetic_pipeline_image(False, False))
    root, out, summary, image_summary = app.process([upload], app.Settings(), 4)
    assert int(summary.iloc[0]["PDO_count"]) == 0
    assert int(summary.iloc[0]["PDO_containing_wells"]) == 0
    assert (out / "csv" / "overall_summary.csv").exists()
    assert not (out / "figures" / "PDO_size_distribution.png").exists()


def test_process_raises_helpful_error_if_no_wells(monkeypatch):
    monkeypatch.setattr(app, "detect_wells", lambda rgb, s: np.empty((0, 3), dtype=int))
    upload = FakeUpload("bad.png", synthetic_rgb(120))
    with pytest.raises(RuntimeError, match="No fully visible wells detected"):
        app.process([upload], app.Settings(), 4)


def test_streamlit_app_smoke_loads_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert len(at.exception) == 0
    assert any("KT3 PDO + PSC Microwell Analyzer" in t.value for t in at.title)
