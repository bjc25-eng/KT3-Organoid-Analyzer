from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


GFP_TERMS = ('gfp', 'egfp', 'green', 'fitc', 'fluorescein', '488')
DIC_TERMS = ('dic', 'brightfield', 'bright field', 'bright-field', 'transmitted',
             'transmission', 'diascopic', 'dia', 'phase')
FINAL_WELL_NAME = 'well_measurements.csv'
FINAL_PDO_NAME = 'pdo_measurements.csv'
FINAL_SUMMARY_NAME = 'condition_summary.json'
PIPELINE_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    import re
    value = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text).strip()).strip('_')
    return value or 'condition'


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def discover_conditions(objects: Iterable[dict]) -> list[dict]:
    """Return sorted S3 rows with stable, collision-free condition IDs."""
    rows = [dict(row) for row in objects]
    rows.sort(key=lambda row: str(row.get('key', '')))
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get('key', ''))
        base = _slug(Path(key).stem)
        counts[base] = counts.get(base, 0) + 1
        row.update(condition_name=Path(key).stem, condition_id=base)
    for row in rows:
        base = str(row['condition_id'])
        if counts[base] > 1:
            row['condition_id'] = f"{base}_{_short_hash(str(row['key']))}"
    return rows


def _normalise_name(value: object) -> str:
    return ' '.join(str(value or '').lower().replace('_', ' ').replace('-', ' ').split())


def _metadata_matches(meta: dict, terms: tuple[str, ...], kind: str) -> list[dict]:
    matches = []
    count = int(meta.get('channel_count', 0) or 0)
    for position, raw in enumerate(meta.get('channel_metadata') or []):
        row = dict(raw or {})
        if any(term in _normalise_name(row.get('name')) for term in terms):
            index = int(row.get('index', position))
            if index < 0 or (count and index >= count):
                raise RuntimeError(f'{kind} channel metadata index {index} is outside 0..{count - 1}.')
            matches.append({'index': index, 'name': str(row.get('name', ''))})
    return matches


def resolve_channel_mapping(zarr_meta: dict, nd2_meta: dict | None = None) -> dict:
    """Resolve GFP and DIC using OME/ND2 labels; never assume channel order."""
    nd2_meta = nd2_meta or {}
    resolved: dict[str, int] = {}
    evidence: dict[str, dict] = {}
    for kind, terms in (('gfp', GFP_TERMS), ('dic', DIC_TERMS)):
        zm = _metadata_matches(zarr_meta, terms, kind.upper())
        nm = _metadata_matches(nd2_meta, terms, kind.upper()) if nd2_meta else []
        if len(zm) > 1 or len(nm) > 1:
            raise RuntimeError(f'Ambiguous {kind.upper()} channel metadata: OME-Zarr={zm}, ND2={nm}.')
        if zm and nm and zm[0]['index'] != nm[0]['index']:
            raise RuntimeError(f'{kind.upper()} index disagrees: OME-Zarr={zm[0]}, ND2={nm[0]}.')
        # Analysis reads the converted array, so its own OME metadata must name
        # the channel. ND2 labels are corroborating evidence, never an order
        # fallback for an unlabeled conversion.
        match = zm[0] if zm else None
        if match is None:
            zn = [str(r.get('name', '')) for r in zarr_meta.get('channel_metadata') or []]
            nn = [str(r.get('name', '')) for r in nd2_meta.get('channel_metadata') or []]
            raise RuntimeError(f'Cannot verify {kind.upper()} channel from metadata; OME-Zarr={zn}, ND2={nn}.')
        resolved[kind] = int(match['index'])
        evidence[kind] = {'index': int(match['index']),
                          'omezarr_match': zm[0] if zm else None,
                          'nd2_match': nm[0] if nm else None}
    if resolved['gfp'] == resolved['dic']:
        raise RuntimeError('Metadata resolved GFP and DIC to the same channel.')
    return {'gfp_channel': resolved['gfp'], 'dic_channel': resolved['dic'], 'evidence': evidence}


def _rss_mib() -> float | None:
    try:
        resident = int(Path('/proc/self/statm').read_text(encoding='ascii').split()[1])
        return resident * int(os.sysconf('SC_PAGE_SIZE')) / (1024 ** 2)
    except Exception:
        try:
            import psutil  # type: ignore
            return float(psutil.Process().memory_info().rss) / (1024 ** 2)
        except Exception:
            return None


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)


class ConditionLogger:
    def __init__(self, condition: str, path: Path):
        self.condition, self.path = condition, path
        self.started = time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, phase: str, message: str, **extra: object) -> None:
        rss = _rss_mib()
        payload = {'timestamp': _now(), 'condition': self.condition, 'phase': phase,
                   'elapsed_seconds': round(time.monotonic() - self.started, 3),
                   'pid': os.getpid(), 'rss_mib': None if rss is None else round(rss, 1),
                   'message': message, **extra}
        line = json.dumps(payload, default=str)
        print(line, flush=True)
        with self.path.open('a', encoding='utf-8') as log:
            log.write(line + '\n')

    def output(self, phase: str, line: str, **extra: object) -> None:
        self.event(phase, 'process_output', output=line.rstrip(), **extra)


def _run_logged(command: list[str], logger: ConditionLogger, phase: str) -> None:
    logger.event(phase, 'command_start', command=command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        logger.output(phase, line, child_pid=process.pid)
    code = process.wait()
    logger.event(phase, 'command_end', child_pid=process.pid, exit_code=code)
    if code:
        raise RuntimeError(f'Command exited with status {code}: {command}')


def build_analysis_commands(python: str, repo_root: Path, zarr_path: Path,
                            condition_dir: Path, *, tile: int, gfp_channel: int,
                            dic_channel: int, well_diameter_um: float, hough_p2: float,
                            green_low: float, green_high: float,
                            pdo_min_area: int) -> dict[str, list[str]]:
    benchmark, refined, hex_qc = (condition_dir / name for name in ('benchmark', 'refined', 'hex_qc'))
    common = ['--tile', str(int(tile)), '--gfp-channel', str(int(gfp_channel)),
              '--well-diameter-um', str(float(well_diameter_um)),
              '--green-low', str(float(green_low)), '--green-high', str(float(green_high)),
              '--pdo-min-area', str(int(pdo_min_area))]
    return {
        'benchmark': [python, str(repo_root / 'aws_full_array_benchmark.py'), str(zarr_path),
                      '--output-dir', str(benchmark), '--dic-channel', str(int(dic_channel)),
                      '--hough-p2', str(float(hough_p2)), *common],
        'refine': [python, str(repo_root / 'aws_refine_lattice.py'), str(zarr_path),
                   str(benchmark / 'wells_raw.csv'), '--output-dir', str(refined), *common],
        'hex_qc': [python, str(repo_root / 'aws_extract_hex_array_component.py'), str(refined),
                   '--output-dir', str(hex_qc)],
    }


def _signature(payload: dict) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _phase_valid(condition_dir: Path, phase: str, signature: str, required: list[Path]) -> bool:
    marker = condition_dir / f'.phase_{phase}.json'
    try:
        data = _read_json(marker)
        return (all(path.is_file() for path in required) and data.get('state') == 'completed'
                and data.get('signature') == signature)
    except Exception:
        return False


def _mark_phase(condition_dir: Path, phase: str, signature: str, command: list[str]) -> None:
    _atomic_json(condition_dir / f'.phase_{phase}.json',
                 {'state': 'completed', 'phase': phase, 'signature': signature,
                  'completed_at': _now(), 'command': command})


def _completion_valid(path: Path, source: dict, signature: str) -> bool:
    if not (path.is_file() and (path.parent / FINAL_WELL_NAME).is_file()
            and (path.parent / FINAL_PDO_NAME).is_file()):
        return False
    try:
        row = _read_json(path)
        identity = row.get('source_object') or {}
        return (row.get('completion_status') == 'completed'
                and identity.get('key') == source.get('key')
                and str(identity.get('etag', '')) == str(source.get('etag', ''))
                and int(identity.get('size', 0)) == int(source.get('size', 0))
                and row.get('analysis_signature') == signature
                and (path.parent / FINAL_WELL_NAME).is_file()
                and (path.parent / FINAL_PDO_NAME).is_file())
    except Exception:
        return False


def _atomic_write_csv(path: Path, rows: list[dict], required_fields: tuple[str, ...] = ()) -> None:
    fields, seen = list(required_fields), set(required_fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        if fields:
            writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _csv_rows(path: Path, condition: str) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as handle:
        return [{'condition_id': condition, **row} for row in csv.DictReader(handle)]


def combine_results(result_root: Path) -> None:
    summaries, wells, pdos = [], [], []
    result_root.mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in result_root.iterdir() if p.is_dir()):
        summary = folder / FINAL_SUMMARY_NAME
        try:
            payload = _read_json(summary)
            if payload.get('completion_status') != 'completed':
                continue
            condition = str(payload.get('condition_id') or folder.name)
            summaries.append({k: json.dumps(v, sort_keys=True, default=str)
                              if isinstance(v, (dict, list)) else v for k, v in payload.items()})
            wells.extend(_csv_rows(folder / FINAL_WELL_NAME, condition))
            pdos.extend(_csv_rows(folder / FINAL_PDO_NAME, condition))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    _atomic_write_csv(result_root / 'all_conditions_summary.csv', summaries,
                      ('condition_id', 'completion_status'))
    _atomic_write_csv(result_root / 'all_conditions_well_measurements.csv', wells,
                      ('condition_id', 'well_id'))
    _atomic_write_csv(result_root / 'all_conditions_pdo_measurements.csv', pdos,
                      ('condition_id', 'well_id'))


def _analysis_config(args: argparse.Namespace, source: dict) -> dict:
    return {'pipeline_version': PIPELINE_VERSION,
            'source': {'key': source['key'], 'etag': source.get('etag', ''),
                       'size': int(source.get('size', 0))},
            'well_diameter_um': float(args.well_diameter_um), 'tile': int(args.tile),
            'hough_p2': float(args.hough_p2), 'green_low': float(args.green_low),
            'green_high': float(args.green_high), 'pdo_min_area': int(args.pdo_min_area),
            'series_index': int(args.series_index)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Headless sequential, restartable S3 ND2 batch analysis.')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--region', default='eu-west-2')
    parser.add_argument('--cache-root', type=Path, default=Path('/home/ec2-user/kt3_nd2_cache'))
    parser.add_argument('--result-root', type=Path, default=Path('/home/ec2-user/kt3_batch_results'))
    parser.add_argument('--results-s3-prefix', default='')
    parser.add_argument('--bioformats2raw', type=Path, default=None)
    parser.add_argument('--series-index', type=int, default=0)
    parser.add_argument('--well-diameter-um', type=float, default=100.0)
    parser.add_argument('--tile', type=int, default=2048)
    parser.add_argument('--hough-p2', type=float, default=27.0)
    parser.add_argument('--green-low', type=float, default=30.0)
    parser.add_argument('--green-high', type=float, default=45.0)
    parser.add_argument('--pdo-min-area', type=int, default=20)
    parser.add_argument('--expected-conditions', type=int, default=0)
    parser.add_argument('--max-conditions', type=int, default=0)
    parser.add_argument('--cleanup-local-after-upload', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def _load_status(path: Path, args: argparse.Namespace, count: int) -> dict:
    try:
        prior = _read_json(path)
    except Exception:
        prior = {}
    return {**prior, 'state': 'running', 'started_at': prior.get('started_at') or _now(),
            'run_started_at': _now(), 'updated_at': _now(), 'pid': os.getpid(),
            'bucket': args.bucket, 'prefix': args.prefix, 'condition_count': count,
            'conditions': prior.get('conditions') if isinstance(prior.get('conditions'), dict) else {}}


def _upload_aggregates(client, root: Path, bucket: str, prefix: str) -> None:
    for name in ('batch_status.json', 'all_conditions_summary.csv',
                 'all_conditions_well_measurements.csv', 'all_conditions_pdo_measurements.csv'):
        path = root / name
        if path.is_file():
            client.upload_file(str(path), bucket, f'{prefix.strip("/")}/{name}')


def run_batch(args: argparse.Namespace, *, client=None, list_objects: Callable | None = None,
              stage_object: Callable | None = None, nd2_probe: Callable | None = None,
              converter: Callable | None = None, zarr_probe: Callable | None = None,
              command_runner: Callable = _run_logged) -> int:
    from nd2_s3_stage import get_s3_client, list_nd2_objects, stage_s3_nd2, upload_tree
    list_objects, stage_object = list_objects or list_nd2_objects, stage_object or stage_s3_nd2
    client = client or get_s3_client(region_name=args.region)
    objects = discover_conditions(list_objects(client, args.bucket, args.prefix))
    if not objects:
        raise RuntimeError(f'No .nd2 objects found under s3://{args.bucket}/{args.prefix}')
    if args.expected_conditions > 0 and len(objects) != args.expected_conditions:
        raise RuntimeError(f'Expected {args.expected_conditions} ND2 conditions but discovered {len(objects)}.')
    if args.max_conditions > 0:
        objects = objects[:args.max_conditions]
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'bucket': args.bucket, 'prefix': args.prefix,
                          'condition_count': len(objects), 'objects': [
                              {'condition_name': r['condition_name'], 'condition_id': r['condition_id'],
                               's3_uri': f"s3://{args.bucket}/{r['key']}", 'key': r['key'],
                               'size_bytes': int(r.get('size', 0)),
                               'size_gib': round(int(r.get('size', 0)) / 1024 ** 3, 3),
                               'etag': r.get('etag', '')} for r in objects]}, indent=2, default=str))
        return 0

    if nd2_probe is None:
        from nd2_large_source import ND2LargeImageReader
        def nd2_probe(path: Path, series: int) -> dict:
            with ND2LargeImageReader(str(path), series_index=series, level=0) as reader:
                return reader.metadata()
    if converter is None or zarr_probe is None:
        from nd2_omezarr import convert_nd2_to_omezarr, probe_omezarr
        converter, zarr_probe = converter or convert_nd2_to_omezarr, zarr_probe or probe_omezarr

    cache_root, result_root = args.cache_root.resolve(), args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    status_path = result_root / 'batch_status.json'
    status = _load_status(status_path, args, len(objects)); _atomic_json(status_path, status)
    repo_root = Path(__file__).resolve().parent
    for index, source in enumerate(objects, 1):
        condition, folder = str(source['condition_id']), result_root / str(source['condition_id'])
        signature = _signature(_analysis_config(args, source))
        logger = ConditionLogger(condition, folder / 'run.log')
        state = {**status['conditions'].get(condition, {}), 'state': 'running', 'phase': 'initialise',
                 'condition': condition, 'condition_name': source['condition_name'],
                 's3_key': source['key'], 'size_bytes': int(source.get('size', 0)),
                 'etag': source.get('etag', ''), 'index': index, 'total': len(objects),
                 'run_started_at': _now(), 'pid': os.getpid()}
        for stale_key in ('error', 'traceback', 'failed_at', 'failed_phase',
                          'combined_output_error', 'aggregate_upload_error'):
            state.pop(stale_key, None)
        status['conditions'][condition] = state
        try:
            if _completion_valid(folder / FINAL_SUMMARY_NAME, source, signature):
                state.update(state='completed', phase='complete', skipped_existing=True)
                logger.event('complete', 'completed_condition_reused')
            else:
                folder.mkdir(parents=True, exist_ok=True)
                state.update(phase='stage', skipped_existing=False); _atomic_json(status_path, status)
                logger.event('stage', 'phase_start', s3_key=source['key'])
                progress = {'last': -10}
                def on_progress(done: int, total: int) -> None:
                    percent = int(100 * done / total) if total else 0
                    if percent >= progress['last'] + 10 or done == total:
                        progress['last'] = percent
                        logger.event('stage', 'download_progress', bytes_done=done,
                                     bytes_total=total, percent=percent)
                staged = stage_object(client, args.bucket, source['key'], cache_root,
                                      progress_callback=on_progress)
                local_nd2 = Path(staged['local_path'])
                state.update(local_nd2=str(local_nd2), staging_reused=bool(staged.get('reused')))
                logger.event('stage', 'phase_complete', reused=bool(staged.get('reused')))

                state['phase'] = 'probe_nd2'; _atomic_json(status_path, status)
                nd2_meta = nd2_probe(local_nd2, args.series_index)
                logger.event('probe_nd2', 'phase_complete', channels=nd2_meta.get('channel_metadata'),
                             voxel_size_um=nd2_meta.get('voxel_size_um'))
                state['phase'] = 'convert'; _atomic_json(status_path, status)
                conversion = converter(local_nd2, nd2_meta, series_index=args.series_index,
                                       executable=args.bioformats2raw, max_workers=1, tile_size=1024,
                                       resolutions=1, overwrite=False,
                                       line_callback=lambda line: logger.output('convert', line))
                zarr_path = Path(conversion['output_path']).resolve()
                zmeta = zarr_probe(zarr_path)
                validation = conversion.get('validation') or {}
                if validation and not validation.get('ok', False):
                    raise RuntimeError(f'OME-Zarr validation failed: {validation}')
                mapping = resolve_channel_mapping(zmeta, nd2_meta)
                state.update(omezarr=str(zarr_path), conversion_reused=bool(conversion.get('reused')),
                             gfp_channel=mapping['gfp_channel'], dic_channel=mapping['dic_channel'],
                             channel_mapping_evidence=mapping['evidence'],
                             voxel_size_um=zmeta.get('voxel_size_um'))
                logger.event('convert', 'phase_complete', channel_mapping=mapping,
                             voxel_size_um=zmeta.get('voxel_size_um'))

                commands = build_analysis_commands(sys.executable, repo_root, zarr_path, folder,
                    tile=args.tile, gfp_channel=mapping['gfp_channel'], dic_channel=mapping['dic_channel'],
                    well_diameter_um=args.well_diameter_um, hough_p2=args.hough_p2,
                    green_low=args.green_low, green_high=args.green_high,
                    pdo_min_area=args.pdo_min_area)
                required = {
                    'benchmark': [folder/'benchmark'/n for n in ('benchmark_summary.json','wells_raw.csv','well_measurements.csv','pdo_measurements.csv')],
                    'refine': [folder/'refined'/n for n in ('refined_summary.json','well_measurements.csv','pdo_measurements.csv')],
                    'hex_qc': [folder/'hex_qc'/n for n in ('hex_array_summary.json','hex_array_well_measurements.csv','hex_array_pdo_measurements.csv')]}
                upstream = _signature({'base': signature, 'mapping': mapping,
                                       'omezarr': {'shape': zmeta.get('shape'), 'axes': zmeta.get('axes'),
                                                   'voxel_size_um': zmeta.get('voxel_size_um')}})
                for phase in ('benchmark', 'refine', 'hex_qc'):
                    phase_sig = _signature({'upstream': upstream, 'phase': phase, 'command': commands[phase]})
                    state['phase'] = phase; _atomic_json(status_path, status)
                    if _phase_valid(folder, phase, phase_sig, required[phase]):
                        logger.event(phase, 'phase_reused')
                    else:
                        command_runner(commands[phase], logger, phase)
                        missing = [str(path) for path in required[phase] if not path.is_file()]
                        if missing:
                            raise RuntimeError(f'{phase} completed without required outputs: {missing}')
                        _mark_phase(folder, phase, phase_sig, commands[phase])
                    upstream = phase_sig

                _atomic_copy(folder/'hex_qc'/'hex_array_well_measurements.csv', folder/FINAL_WELL_NAME)
                _atomic_copy(folder/'hex_qc'/'hex_array_pdo_measurements.csv', folder/FINAL_PDO_NAME)
                final = {'completion_status': 'completed', 'condition_id': condition,
                         'condition_name': source['condition_name'],
                         'source_object': {'bucket': args.bucket, 'key': source['key'],
                                           'size': int(source.get('size', 0)), 'etag': source.get('etag', '')},
                         'analysis_signature': signature, 'completed_at': _now(),
                         'channel_mapping': mapping, 'omezarr_validation': validation,
                         'pixel_size_um': zmeta.get('voxel_size_um'),
                         'scientific_settings': {'well_diameter_um': args.well_diameter_um,
                           'hough_p2': args.hough_p2, 'green_low': args.green_low,
                           'green_high': args.green_high, 'pdo_min_area': args.pdo_min_area},
                         'benchmark': _read_json(folder/'benchmark'/'benchmark_summary.json'),
                         'lattice_qc': _read_json(folder/'refined'/'refined_summary.json'),
                         'whole_array_qc': _read_json(folder/'hex_qc'/'hex_array_summary.json'),
                         'output_files': [FINAL_WELL_NAME, FINAL_PDO_NAME, FINAL_SUMMARY_NAME]}
                _atomic_json(folder/FINAL_SUMMARY_NAME, final)
                state.update(state='completed', phase='complete', completed_at=_now(),
                             summary=str(folder/FINAL_SUMMARY_NAME))
                logger.event('complete', 'condition_complete')

            if args.results_s3_prefix:
                state['phase'] = 'upload'; _atomic_json(status_path, status)
                prefix = '/'.join(x.strip('/') for x in (args.results_s3_prefix, condition) if x.strip('/'))
                state['uploaded'] = upload_tree(client, folder, args.bucket, prefix)
                state.update(phase='complete', state='completed', uploaded_at=_now())
                if args.cleanup_local_after_upload and state.get('local_nd2'):
                    object_root = Path(state['local_nd2']).parent.parent.resolve()
                    if cache_root not in object_root.parents:
                        raise RuntimeError(f'Refusing cleanup outside cache root: {object_root}')
                    shutil.rmtree(object_root); state['local_cache_removed_after_upload'] = True
        except Exception as exc:
            tb = traceback.format_exc()
            state.update(state='failed', failed_phase=state.get('phase'), phase='failed',
                         failed_at=_now(), error=f'{type(exc).__name__}: {exc}', traceback=tb)
            logger.event('failed', 'condition_failed', error=state['error'], traceback=tb)
        finally:
            state['elapsed_seconds_this_run'] = round(time.monotonic() - logger.started, 3)
            state['rss_mib'] = _rss_mib(); status['updated_at'] = _now()
            _atomic_json(status_path, status)
            try:
                combine_results(result_root)
            except Exception as exc:
                state['combined_output_error'] = f'{type(exc).__name__}: {exc}'
                logger.event('combine', 'combined_output_failed', error=state['combined_output_error'])
                _atomic_json(status_path, status)
            else:
                state.pop('combined_output_error', None)
            if args.results_s3_prefix:
                try:
                    _upload_aggregates(client, result_root, args.bucket, args.results_s3_prefix)
                except Exception as exc:
                    state['aggregate_upload_error'] = f'{type(exc).__name__}: {exc}'
                    logger.event('upload', 'aggregate_upload_failed', error=state['aggregate_upload_error'])
                    _atomic_json(status_path, status)
                else:
                    state.pop('aggregate_upload_error', None)

    failures = sum((status['conditions'].get(str(r['condition_id'])) or {}).get('state') == 'failed'
                   for r in objects)
    status.update(state='completed_with_failures' if failures else 'completed',
                  failure_count=failures, completed_at=_now(), updated_at=_now())
    _atomic_json(status_path, status)
    try:
        combine_results(result_root)
    except Exception as exc:
        status['combined_output_error'] = f'{type(exc).__name__}: {exc}'
        status['state'] = 'completed_with_failures'
        _atomic_json(status_path, status)
    if args.results_s3_prefix:
        try:
            _upload_aggregates(client, result_root, args.bucket, args.results_s3_prefix)
        except Exception as exc:
            status['aggregate_upload_error'] = f'{type(exc).__name__}: {exc}'
            status['updated_at'] = _now()
            _atomic_json(status_path, status)
    operational_errors = bool(status.get('combined_output_error') or status.get('aggregate_upload_error'))
    operational_errors = operational_errors or any(
        row.get('combined_output_error') or row.get('aggregate_upload_error')
        for row in status['conditions'].values()
    )
    if operational_errors and not failures:
        status['state'] = 'completed_with_failures'
        status['updated_at'] = _now()
        _atomic_json(status_path, status)
    print(json.dumps(status, indent=2, default=str))
    return 1 if failures or operational_errors else 0


def main(argv: list[str] | None = None) -> int:
    return run_batch(build_parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
