from types import SimpleNamespace

import pytest

from analysis_core import Settings
from nd2_physical_scan import physical_scan_settings


class FakeReader:
    def metadata(self):
        return {"voxel_size_um": {"x": 0.5, "y": 0.5}}


def test_physical_scan_uses_validated_radius_factors():
    settings = Settings(well_diameter_um=100.0, well_rmin=23, well_rmax=40, well_spacing=54)
    derived, expected_radius, umpp = physical_scan_settings(FakeReader(), settings)
    assert umpp == pytest.approx(0.5)
    assert expected_radius == pytest.approx(100.0)
    assert derived.well_rmin == 80
    assert derived.well_rmax == 120
    assert derived.well_spacing == 150
    # Do not mutate the user/settings object used elsewhere in the analysis.
    assert settings.well_rmin == 23
    assert settings.well_rmax == 40
    assert settings.well_spacing == 54
