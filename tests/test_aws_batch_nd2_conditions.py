from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import aws_batch_nd2_conditions as batch
from omezarr_cyx import SingletonTZCYX


def _objects():
    return [
        {'key': 'inputs/condition_b.nd2', 'size': 222, 'etag': 'etag-b'},
        {'key': 'inputs/condition_a.nd2', 'size': 111, 'etag': 'etag-a'},
    ]


def _nd2_meta():
    return {'channel_count': 3, 'channel_metadata': [
        {'index': 0, 'name': 'GFP 488'}, {'index': 1, 'name': 'RFP 561'},
        {'index': 2, 'name': 'DIC'}],
        'voxel_size_um': {'x': 0.65, 'y': 0.65}}


def _zarr_meta():
    return {'channel_count': 3, 'shape': [3, 32, 48], 'axes': ['C', 'Y', 'X'],
            'channel_metadata': [
                {'index': 0, 'name': 'GFP 488'}, {'index': 1, 'name': 'RFP 561'},
                {'index': 2, 'name': 'DIC transmitted'}],
            'voxel_size_um': {'x': 0.65, 'y': 0.65}}


def _write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)


def _successful_runner(command, _logger, phase):
    output = Path(command[command.index('--output-dir') + 1]); output.mkdir(parents=True, exist_ok=True)
    if phase == 'benchmark':
        (output/'benchmark_summary.json').write_text('{"raw_well_candidates": 12}')
        _write_csv(output/'wells_raw.csv', ['x_px_fullres','y_px_fullres','radius_px'], [[10,10,5]])
        _write_csv(output/'well_measurements.csv', ['well_id','PDO_count'], [[1,1]])
        _write_csv(output/'pdo_measurements.csv', ['well_id','projected_area_um2'], [[1,25]])
    elif phase == 'refine':
        (output/'refined_summary.json').write_text('{"pitch_px":20,"pixel_size_um":{"x":0.65,"y":0.65}}')
        _write_csv(output/'well_measurements.csv', ['well_id','PDO_count'], [[1,1]])
        _write_csv(output/'pdo_measurements.csv', ['well_id','projected_area_um2'], [[1,25]])
    else:
        (output/'hex_array_summary.json').write_text('{"largest_component_wells":1}')
        _write_csv(output/'hex_array_well_measurements.csv', ['well_id','PDO_count'], [[1,1]])
        _write_csv(output/'hex_array_pdo_measurements.csv', ['well_id','projected_area_um2'], [[1,25]])


class BatchND2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fake_stage = types.ModuleType('nd2_s3_stage')
        fake_stage.get_s3_client = lambda **_kwargs: object()
        fake_stage.list_nd2_objects = lambda *_args: []
        fake_stage.stage_s3_nd2 = lambda *_args, **_kwargs: None
        fake_stage.upload_tree = lambda *_args, **_kwargs: {'uploaded_files': 0}
        self.modules = patch.dict(sys.modules, {'nd2_s3_stage': fake_stage})
        self.modules.start()

    def tearDown(self):
        self.modules.stop(); self.temp.cleanup()

    def args(self, *extra: str):
        return batch.build_parser().parse_args([
            '--bucket','test-bucket','--prefix','inputs/',
            '--cache-root',str(self.root/'cache'),'--result-root',str(self.root/'results'),*extra])

    def pipeline_kwargs(self):
        def stage(_client, _bucket, key, cache_root, progress_callback=None):
            path = Path(cache_root)/Path(key).name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'mock')
            if progress_callback: progress_callback(4,4)
            return {'local_path':str(path),'reused':False}
        return {'client':object(),'list_objects':lambda *_:_objects(),'stage_object':stage,
                'nd2_probe':lambda *_:_nd2_meta(),
                'converter':lambda path,*_a,**_k:{'output_path':str(Path(path).with_suffix('.ome.zarr')),
                    'reused':False,'validation':{'ok':True,'errors':[],'warnings':[]}},
                'zarr_probe':lambda *_:_zarr_meta()}

    def test_condition_discovery_is_sorted_and_collision_safe(self):
        rows = batch.discover_conditions([
            {'key':'z/sample.nd2'},{'key':'a/sample.nd2'},{'key':'a/unique name.nd2'}])
        self.assertEqual([r['key'] for r in rows], ['a/sample.nd2','a/unique name.nd2','z/sample.nd2'])
        self.assertEqual(rows[1]['condition_id'], 'unique_name')
        self.assertNotEqual(rows[0]['condition_id'], rows[2]['condition_id'])

    def test_dry_run_lists_exact_objects_without_staging(self):
        args = self.args('--dry-run','--expected-conditions','2'); called = {'stage':False}
        def no_stage(*_a,**_k): called['stage']=True; self.fail('dry-run staged data')
        output = io.StringIO()
        with redirect_stdout(output):
            code = batch.run_batch(args,client=object(),list_objects=lambda *_:_objects(),stage_object=no_stage)
        payload = json.loads(output.getvalue())
        self.assertEqual(code,0); self.assertFalse(called['stage'])
        self.assertEqual([r['key'] for r in payload['objects']],
                         ['inputs/condition_a.nd2','inputs/condition_b.nd2'])
        self.assertEqual(payload['objects'][0]['size_bytes'],111)
        self.assertFalse(args.result_root.exists())

    def test_completed_condition_is_skipped_on_restart(self):
        args=self.args('--max-conditions','1'); source=batch.discover_conditions(_objects())[0]
        folder=args.result_root/source['condition_id']; folder.mkdir(parents=True)
        signature=batch._signature(batch._analysis_config(args,source))
        (folder/batch.FINAL_SUMMARY_NAME).write_text(json.dumps({
            'completion_status':'completed','condition_id':source['condition_id'],
            'source_object':{'key':source['key'],'etag':source['etag'],'size':source['size']},
            'analysis_signature':signature}))
        _write_csv(folder/batch.FINAL_WELL_NAME,['well_id'],[[1]])
        _write_csv(folder/batch.FINAL_PDO_NAME,['well_id'],[[1]])
        with redirect_stdout(io.StringIO()):
            code=batch.run_batch(args,client=object(),list_objects=lambda *_:_objects(),
                stage_object=lambda *_a,**_k:self.fail('completed condition was staged'),
                nd2_probe=lambda *_:self.fail('completed condition was probed'),
                converter=lambda *_a,**_k:self.fail('completed condition was converted'),
                zarr_probe=lambda *_:self.fail('completed condition opened OME-Zarr'))
        status=json.loads((args.result_root/'batch_status.json').read_text())
        self.assertEqual(code,0)
        self.assertTrue(status['conditions'][source['condition_id']]['skipped_existing'])

    def test_failure_continues_and_combined_csvs_are_written(self):
        args=self.args(); kwargs=self.pipeline_kwargs(); failed={'done':False}
        def runner(command,logger,phase):
            if logger.condition=='condition_a' and not failed['done']:
                failed['done']=True; raise RuntimeError('synthetic condition failure')
            _successful_runner(command,logger,phase)
        with redirect_stdout(io.StringIO()):
            code=batch.run_batch(args,command_runner=runner,**kwargs)
        status=json.loads((args.result_root/'batch_status.json').read_text())
        self.assertEqual(code,1)
        self.assertEqual(status['conditions']['condition_a']['state'],'failed')
        self.assertEqual(status['conditions']['condition_b']['state'],'completed')
        self.assertIn('Traceback',status['conditions']['condition_a']['traceback'])
        with (args.result_root/'all_conditions_well_measurements.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(rows,[{'condition_id':'condition_b','well_id':'1','PDO_count':'1'}])
        for name in ('all_conditions_summary.csv','all_conditions_pdo_measurements.csv'):
            self.assertTrue((args.result_root/name).is_file())

    def test_completed_phases_are_reused_after_interrupted_finalisation(self):
        args=self.args('--max-conditions','1'); kwargs=self.pipeline_kwargs()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(batch.run_batch(args,command_runner=_successful_runner,**kwargs),0)
        condition=batch.discover_conditions(_objects())[0]['condition_id']
        folder=args.result_root/condition
        (folder/batch.FINAL_SUMMARY_NAME).unlink()
        (folder/batch.FINAL_WELL_NAME).unlink()
        (folder/batch.FINAL_PDO_NAME).unlink()
        def no_analysis(*_args,**_kwargs):
            self.fail('a completed phase was rerun')
        with redirect_stdout(io.StringIO()):
            code=batch.run_batch(args,command_runner=no_analysis,**kwargs)
        self.assertEqual(code,0)
        self.assertTrue((folder/batch.FINAL_SUMMARY_NAME).is_file())

    def test_channel_mapping_uses_mock_omezarr_metadata_with_rfp(self):
        meta=_zarr_meta(); mapping=batch.resolve_channel_mapping(meta,_nd2_meta())
        self.assertEqual(meta['voxel_size_um'],{'x':0.65,'y':0.65})
        self.assertEqual((mapping['gfp_channel'],mapping['dic_channel']),(0,2))
        with self.assertRaisesRegex(RuntimeError,'Cannot verify GFP'):
            batch.resolve_channel_mapping({'channel_count':2,'channel_metadata':[
                {'index':0,'name':'Channel 0'},{'index':1,'name':'Channel 1'}]})

    def test_combined_pdo_csv_keeps_headers_when_no_pdos_exist(self):
        result=self.root/'empty-pdo-results'; folder=result/'condition_zero'; folder.mkdir(parents=True)
        (folder/batch.FINAL_SUMMARY_NAME).write_text(json.dumps({
            'completion_status':'completed','condition_id':'condition_zero'}))
        _write_csv(folder/batch.FINAL_WELL_NAME,['well_id','PDO_count'],[[1,0]])
        _write_csv(folder/batch.FINAL_PDO_NAME,['well_id','projected_area_um2'],[])
        batch.combine_results(result)
        header=(result/'all_conditions_pdo_measurements.csv').read_text().splitlines()[0]
        self.assertEqual(header,'condition_id,well_id')

    def test_command_construction(self):
        commands=batch.build_analysis_commands('/venv/bin/python',Path('/repo'),Path('/cache/a.zarr'),
            self.root,tile=2048,gfp_channel=2,dic_channel=0,well_diameter_um=100,
            hough_p2=27,green_low=30,green_high=45,pdo_min_area=20)
        self.assertEqual(list(commands),['benchmark','refine','hex_qc'])
        self.assertEqual(commands['benchmark'][0],'/venv/bin/python')
        self.assertEqual(Path(commands['benchmark'][1]).name,'aws_full_array_benchmark.py')
        self.assertEqual(commands['benchmark'][commands['benchmark'].index('--dic-channel')+1],'0')
        self.assertEqual(Path(commands['refine'][1]).name,'aws_refine_lattice.py')
        self.assertEqual(Path(commands['hex_qc'][1]).name,'aws_extract_hex_array_component.py')

    def test_singleton_time_axis_is_indexed_before_tile_read(self):
        class FakeArray:
            shape=(1,3,81698,8219)
            def __init__(self): self.selector=None
            def __getitem__(self,selector):
                self.selector=selector
                return np.zeros((20,30),dtype=np.uint16)
        array=FakeArray(); planes=SingletonTZCYX(array,['T','C','Y','X'])
        tile=planes.read(2,slice(100,120),slice(200,230))
        self.assertEqual(planes.shape_cyx,(3,81698,8219))
        self.assertEqual(array.selector,(0,2,slice(100,120),slice(200,230)))
        self.assertEqual(tile.shape,(20,30))

    def test_singleton_time_and_z_axes_are_supported(self):
        class FakeArray:
            shape=(1,3,1,100,80)
            def __init__(self): self.selector=None
            def __getitem__(self,selector):
                self.selector=selector
                return np.zeros((5,7),dtype=np.uint16)
        array=FakeArray(); planes=SingletonTZCYX(array,['T','C','Z','Y','X'])
        planes.read(1,slice(10,15),slice(20,27))
        self.assertEqual(array.selector,(0,1,0,slice(10,15),slice(20,27)))

    def test_non_singleton_time_or_z_axis_is_rejected(self):
        class FakeArray:
            shape=(2,3,10,10)
        with self.assertRaisesRegex(RuntimeError,'T axis has size 2'):
            SingletonTZCYX(FakeArray(),['T','C','Y','X'])


if __name__ == '__main__':
    unittest.main()
