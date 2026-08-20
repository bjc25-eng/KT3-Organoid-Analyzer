from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nd2_large_source import ND2LargeImageReader
from nd2_omezarr import convert_nd2_to_omezarr, probe_omezarr
from nd2_s3_stage import get_s3_client, list_nd2_objects, stage_s3_nd2, upload_tree


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    import re
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text).strip()).strip('_')
    return s or 'condition'


def _channel_index(meta: dict, terms: tuple[str, ...], fallback: int) -> int:
    for row in meta.get('channel_metadata') or []:
        name = str(row.get('name', '')).lower()
        if any(term in name for term in terms):
            return int(row.get('index', fallback))
    return int(fallback)


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\n[' + _now() + '] RUN ' + ' '.join(command) + '\n')
        log.flush()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f'Command exited with status {code}: {command}')


def _combine_results(result_root: Path) -> None:
    summary_rows = []
    well_tables = []
    pdo_tables = []
    for condition_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        summary_path = condition_dir / 'benchmark' / 'benchmark_summary.json'
        well_path = condition_dir / 'benchmark' / 'well_measurements.csv'
        pdo_path = condition_dir / 'benchmark' / 'pdo_measurements.csv'
        if summary_path.exists():
            row = json.loads(summary_path.read_text(encoding='utf-8'))
            row['condition_id'] = condition_dir.name
            summary_rows.append(row)
        if well_path.exists():
            df = pd.read_csv(well_path)
            df.insert(0, 'condition_id', condition_dir.name)
            well_tables.append(df)
        if pdo_path.exists():
            df = pd.read_csv(pdo_path)
            df.insert(0, 'condition_id', condition_dir.name)
            pdo_tables.append(df)

    if summary_rows:
        flat = []
        for row in summary_rows:
            q = dict(row)
            for key in list(q):
                if isinstance(q[key], (dict, list)):
                    q[key] = json.dumps(q[key], default=str)
            flat.append(q)
        pd.DataFrame(flat).to_csv(result_root / 'all_conditions_summary.csv', index=False)
    if well_tables:
        pd.concat(well_tables, ignore_index=True).to_csv(
            result_root / 'all_conditions_well_measurements.csv', index=False
        )
    if pdo_tables:
        pd.concat(pdo_tables, ignore_index=True).to_csv(
            result_root / 'all_conditions_pdo_measurements.csv', index=False
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            'Unattended sequential S3 ND2 -> OME-Zarr -> low-memory full-array PDO analysis. '
            'Each condition is independent and completed conditions are skipped on restart.'
        )
    )
    ap.add_argument('--bucket', required=True)
    ap.add_argument('--prefix', default='')
    ap.add_argument('--region', default='eu-west-2')
    ap.add_argument('--cache-root', type=Path, default=Path('/home/ec2-user/kt3_nd2_cache'))
    ap.add_argument('--result-root', type=Path, default=Path('/home/ec2-user/kt3_batch_results'))
    ap.add_argument('--results-s3-prefix', default='', help='Optional prefix in the same bucket for completed outputs.')
    ap.add_argument('--well-diameter-um', type=float, default=100.0)
    ap.add_argument('--tile', type=int, default=4096)
    ap.add_argument('--hough-p2', type=float, default=27.0)
    ap.add_argument('--green-low', type=float, default=30.0)
    ap.add_argument('--green-high', type=float, default=45.0)
    ap.add_argument('--pdo-min-area', type=int, default=20)
    ap.add_argument('--max-conditions', type=int, default=0, help='0 means process every ND2 found.')
    ap.add_argument('--cleanup-local-after-upload', action='store_true')
    args = ap.parse_args()

    cache_root = args.cache_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    status_path = result_root / 'batch_status.json'

    client = get_s3_client(region_name=args.region)
    objects = list_nd2_objects(client, args.bucket, args.prefix)
    if args.max_conditions > 0:
        objects = objects[: int(args.max_conditions)]
    if not objects:
        raise RuntimeError(f'No .nd2 objects found under s3://{args.bucket}/{args.prefix}')

    batch_status = {
        'state': 'running',
        'started_at': _now(),
        'updated_at': _now(),
        'bucket': args.bucket,
        'prefix': args.prefix,
        'condition_count': len(objects),
        'conditions': {},
    }
    _write_status(status_path, batch_status)

    repo_root = Path(__file__).resolve().parent
    failures = 0

    for index, obj in enumerate(objects, 1):
        key = str(obj['key'])
        condition = _slug(Path(key).stem)
        condition_dir = result_root / condition
        benchmark_dir = condition_dir / 'benchmark'
        log_path = condition_dir / 'run.log'
        completion = benchmark_dir / 'benchmark_summary.json'

        if completion.exists():
            batch_status['conditions'][condition] = {
                'state': 'completed', 's3_key': key, 'skipped_existing': True,
                'summary': str(completion),
            }
            batch_status['updated_at'] = _now()
            _write_status(status_path, batch_status)
            print(f'[{index}/{len(objects)}] {condition}: already complete; skipping.', flush=True)
            continue

        state = {
            'state': 'running',
            's3_key': key,
            'index': index,
            'total': len(objects),
            'started_at': _now(),
            'phase': 'stage',
        }
        batch_status['conditions'][condition] = state
        batch_status['updated_at'] = _now()
        _write_status(status_path, batch_status)

        try:
            print(f'[{index}/{len(objects)}] {condition}: staging {key}', flush=True)
            staged = stage_s3_nd2(client, args.bucket, key, cache_root)
            local_nd2 = Path(staged['local_path'])
            state.update({'phase': 'probe', 'local_nd2': str(local_nd2)})
            _write_status(status_path, batch_status)

            with ND2LargeImageReader(str(local_nd2), series_index=0, level=0) as reader:
                nd2_meta = reader.metadata()

            state['phase'] = 'convert'
            _write_status(status_path, batch_status)
            conversion = convert_nd2_to_omezarr(
                local_nd2,
                nd2_meta,
                series_index=0,
                max_workers=1,
                tile_size=1024,
                resolutions=1,
                overwrite=False,
                line_callback=lambda line: print(f'[{condition}] {line}', flush=True),
            )
            zarr_path = Path(conversion['output_path']).resolve()
            zmeta = probe_omezarr(zarr_path)
            gfp = _channel_index(zmeta, ('gfp', 'green', 'fitc', '488'), 0)
            dic = _channel_index(zmeta, ('dic', 'brightfield', 'bright field', 'transmitted', 'phase'), max(0, int(zmeta['channel_count']) - 1))

            state.update({
                'phase': 'analyse',
                'omezarr': str(zarr_path),
                'gfp_channel': int(gfp),
                'dic_channel': int(dic),
            })
            _write_status(status_path, batch_status)

            benchmark_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(repo_root / 'aws_full_array_benchmark.py'),
                str(zarr_path),
                '--output-dir', str(benchmark_dir),
                '--tile', str(int(args.tile)),
                '--gfp-channel', str(int(gfp)),
                '--dic-channel', str(int(dic)),
                '--well-diameter-um', str(float(args.well_diameter_um)),
                '--hough-p2', str(float(args.hough_p2)),
                '--green-low', str(float(args.green_low)),
                '--green-high', str(float(args.green_high)),
                '--pdo-min-area', str(int(args.pdo_min_area)),
            ]
            _run_logged(command, log_path)

            state.update({'phase': 'complete', 'state': 'completed', 'completed_at': _now()})

            if args.results_s3_prefix:
                s3_prefix = '/'.join(q.strip('/') for q in [args.results_s3_prefix, condition] if q.strip('/'))
                state['phase'] = 'upload'
                _write_status(status_path, batch_status)
                uploaded = upload_tree(client, condition_dir, args.bucket, s3_prefix)
                state['uploaded'] = uploaded
                state['phase'] = 'complete'
                state['state'] = 'completed'

                if args.cleanup_local_after_upload:
                    # The original ND2 remains safe in S3. Only the local cache for this
                    # object is removed after results have successfully uploaded.
                    object_root = local_nd2.parent.parent
                    if object_root.exists():
                        shutil.rmtree(object_root)
                    state['local_cache_removed_after_upload'] = True

            print(f'[{index}/{len(objects)}] {condition}: COMPLETE', flush=True)
        except Exception as exc:
            failures += 1
            state.update({
                'state': 'failed',
                'phase': 'failed',
                'failed_at': _now(),
                'error': f'{type(exc).__name__}: {exc!s}',
            })
            print(f'[{index}/{len(objects)}] {condition}: FAILED: {type(exc).__name__}: {exc!s}', flush=True)
            # Continue with the next condition so one failure never blocks all six.
        finally:
            batch_status['updated_at'] = _now()
            _write_status(status_path, batch_status)
            _combine_results(result_root)

    batch_status['state'] = 'completed_with_failures' if failures else 'completed'
    batch_status['failure_count'] = failures
    batch_status['completed_at'] = _now()
    batch_status['updated_at'] = _now()
    _write_status(status_path, batch_status)
    _combine_results(result_root)

    print(json.dumps(batch_status, indent=2, default=str))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
