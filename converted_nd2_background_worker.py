from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from analysis_core import Settings
from omezarr_nd2_bridge import process_converted_nd2_omezarr_qc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    parser = argparse.ArgumentParser(description='Detached converted-ND2 whole-array analysis worker')
    parser.add_argument('config', help='Path to background job configuration JSON')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_json(config_path)
    status_path = Path(cfg['status_path']).expanduser().resolve()

    status = {
        'state': 'running',
        'pid': os.getpid(),
        'started_at': _now(),
        'updated_at': _now(),
        'phase': 'starting',
        'done': 0,
        'total': 0,
        'message': 'Opening converted OME-Zarr and existing checkpoints.',
        'config_path': str(config_path),
        'work_root': str(cfg['work_root']),
    }
    _atomic_json(status_path, status)

    settings = Settings()
    for key, value in dict(cfg['settings']).items():
        setattr(settings, key, value)

    def progress(done, total, phase):
        status.update({
            'state': 'running',
            'updated_at': _now(),
            'phase': str(phase),
            'done': int(done),
            'total': int(total),
            'message': (
                'Scanning DIC tiles for microwells.'
                if str(phase) == 'well_scan'
                else 'Analysing GFP PDOs and RFP/PSC signal with final QC.'
            ),
        })
        _atomic_json(status_path, status)

    try:
        result = process_converted_nd2_omezarr_qc(
            [dict(cfg['source'])],
            settings,
            dict(cfg['channel_config']),
            tile_size=int(cfg['tile_size']),
            standard_crop_size=int(cfg['standard_crop_size']),
            work_root=str(cfg['work_root']),
            progress_callback=progress,
            make_ml_export=True,
        )
        root_out, out, manifest, wdf, pdf, pscdf, tracking, run_status, ml_path = result
        status.update({
            'state': 'completed',
            'updated_at': _now(),
            'phase': 'complete',
            'done': int(status.get('total') or status.get('done') or 1),
            'total': int(status.get('total') or status.get('done') or 1),
            'message': 'Whole-array analysis completed successfully.',
            'root_out': str(root_out),
            'output_dir': str(out),
            'ml_path': None if ml_path is None else str(ml_path),
            'well_count': int(len(wdf)),
            'pdo_count': int(len(pdf)),
            'psc_focus_count': int(len(pscdf)),
        })
        _atomic_json(status_path, status)
        return 0
    except Exception as exc:
        status.update({
            'state': 'failed',
            'updated_at': _now(),
            'phase': 'failed',
            'message': f'{type(exc).__name__}: {exc!s}',
            'traceback': traceback.format_exc(),
        })
        _atomic_json(status_path, status)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
