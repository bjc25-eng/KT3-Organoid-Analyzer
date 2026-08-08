from types import SimpleNamespace

import numpy as np

import nd2_large_source as nls


def test_nd2_channel_metadata_and_suggestions():
    channels = [
        SimpleNamespace(channel=SimpleNamespace(index=0, name='DIC', emissionLambdaNm=None, excitationLambdaNm=None, color=None)),
        SimpleNamespace(channel=SimpleNamespace(index=1, name='GFP 488', emissionLambdaNm=525.0, excitationLambdaNm=488.0, color=None)),
    ]
    rows = nls._channel_records(SimpleNamespace(channels=channels))
    assert [r['name'] for r in rows] == ['DIC', 'GFP 488']
    assert nls._suggest_channel(rows, 'dic') == 0
    assert nls._suggest_channel(rows, 'gfp') == 1


def test_nd2_position_metadata_extracts_xy_stage_coordinates():
    points = [
        SimpleNamespace(name='P1', stagePositionUm=SimpleNamespace(x=100.0, y=200.0, z=5.0), pfsOffset=1.5),
        SimpleNamespace(name='P2', stagePositionUm=SimpleNamespace(x=300.0, y=400.0, z=6.0), pfsOffset=1.7),
    ]
    loop = SimpleNamespace(type='XYPosLoop', parameters=SimpleNamespace(points=points))
    rows = nls._position_records([loop])
    assert len(rows) == 2
    assert rows[0]['position_index'] == 0
    assert rows[0]['stage_x_um'] == 100.0
    assert rows[1]['stage_y_um'] == 400.0


def test_nd2_uint16_scaling_uses_significant_bit_depth():
    raw = np.array([[0, 2048, 4095]], dtype=np.uint16)
    out = nls._scale_uint8(raw, significant_bits=12)
    assert out.dtype == np.uint8
    assert int(out[0, 0]) == 0
    assert 126 <= int(out[0, 1]) <= 128
    assert int(out[0, 2]) == 255


def test_nd2_source_label_is_exposed():
    assert nls.ND2_SOURCE_LABEL == 'Nikon ND2'
