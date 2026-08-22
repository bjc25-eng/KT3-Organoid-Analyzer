from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import zarr
from PIL import Image, ImageDraw

import aws_export_pdo_positive_crops as crop_base
import aws_pdo_full_component_qc as component_qc
from nd2_omezarr import probe_omezarr
from omezarr_cyx import SingletonTZCYX


REVIEW_VERSION = 1
OUTPUT_DIRECTORY = 'final_containment_boundary_review'
AUTHORITATIVE_CSV = 'all_conditions_pdo_full_component_measurements.csv'
REVIEW_STATUS = 'visual_review_only_no_exclusion_rule'
MAX_BIN_ROWS = 10
MAX_MANY_TO_ONE_ROWS = 10
EXPECTED_MAXIMUM_DIAGNOSTICS = 62

BIN_DEFINITIONS = (
    ('closest_below_0p40', None, 0.40),
    ('bin_0p40_to_0p45', 0.40, 0.45),
    ('bin_0p45_to_0p50', 0.45, 0.50),
    ('bin_0p50_to_0p55', 0.50, 0.55),
    ('bin_0p55_to_0p65', 0.55, 0.65),
)

REPLAY_FIELDS = (
    'sampling_reasons', 'sampling_primary_bin', 'mask_replay_status',
    'mask_replay_failure_reason', 'component_hash_verification_status',
    'existing_unmasked_component_id', 'regenerated_unmasked_component_id',
    'reproduced_clipped_component_id', 'replayed_crop_bounds',
    'verified_mask_file', 'labelled_diagnostic', 'export_status', 'error',
)
MANIFEST_FIELDS = (*component_qc.PDO_FIELDS, *REPLAY_FIELDS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _atomic_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def _well_id(value: object) -> str:
    return crop_base._normalise_well_id(value)


def _number(row: dict, field: str) -> float:
    value = crop_base._number(row, field)
    if not math.isfinite(value):
        raise RuntimeError(f"Authoritative field '{field}' must be finite for review selection.")
    return value


def row_key(row: dict) -> tuple[str, str, int]:
    return (str(row['condition_id']).strip(), _well_id(row['well_id']),
            int(round(_number(row, 'pdo_number_in_well'))))


def review_sort_key(row: dict) -> tuple:
    condition, well, pdo = row_key(row)
    well_order = (0, int(well)) if well.isdigit() else (1, well)
    return (_number(row, 'full_component_fraction_inside_final_well'),
            condition, well_order, pdo)


def _condition_balanced_closest_below(rows: list[dict], limit: int) -> list[dict]:
    queues: dict[str, list[dict]] = {}
    for row in rows:
        queues.setdefault(str(row['condition_id']), []).append(row)
    for values in queues.values():
        values.sort(key=lambda row: (-_number(row, 'full_component_fraction_inside_final_well'),
                                     row_key(row)))
    output = []
    conditions = sorted(queues)
    while len(output) < limit:
        progressed = False
        for condition in conditions:
            if queues[condition] and len(output) < limit:
                output.append(queues[condition].pop(0)); progressed = True
        if not progressed:
            break
    return output


def _condition_balanced_interval(rows: list[dict], lower: float, upper: float,
                                 limit: int) -> list[dict]:
    unused = list(rows); selected = []; counts: dict[str, int] = {}
    targets = [lower + (index + 0.5) * (upper-lower) / limit for index in range(limit)]
    for target in targets:
        if not unused:
            break
        available_conditions = {str(row['condition_id']) for row in unused}
        minimum_count = min(counts.get(condition, 0) for condition in available_conditions)
        allowed = {condition for condition in available_conditions
                   if counts.get(condition, 0) == minimum_count}
        candidate = min(
            (row for row in unused if str(row['condition_id']) in allowed),
            key=lambda row: (abs(_number(row, 'full_component_fraction_inside_final_well')-target),
                             row_key(row)),
        )
        unused.remove(candidate); selected.append(candidate)
        condition = str(candidate['condition_id']); counts[condition] = counts.get(condition, 0) + 1
    return selected


def _many_to_one_representatives(rows: list[dict], limit: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        component_id = str(row.get('unmasked_component_id', '')).strip()
        if component_id and _truthy(row.get('many_production_PDOs_to_one_unmasked_component')):
            grouped.setdefault(component_id, []).append(row)
    representatives = []
    for component_id, values in grouped.items():
        representative = min(
            values,
            key=lambda row: (abs(_number(row, 'full_component_fraction_inside_final_well')-0.50),
                             row_key(row)),
        )
        fraction = _number(representative, 'full_component_fraction_inside_final_well')
        representatives.append((0 if 0.40 <= fraction < 0.60 else 1,
                                abs(fraction-0.50), component_id, representative))
    candidates = [item[3] for item in sorted(representatives, key=lambda item: item[:3])]
    selected = []; counts: dict[str, int] = {}
    while candidates and len(selected) < limit:
        conditions = {str(row['condition_id']) for row in candidates}
        minimum_count = min(counts.get(condition, 0) for condition in conditions)
        allowed = {condition for condition in conditions if counts.get(condition, 0) == minimum_count}
        index = next(index for index, row in enumerate(candidates)
                     if str(row['condition_id']) in allowed)
        row = candidates.pop(index); selected.append(row)
        condition = str(row['condition_id']); counts[condition] = counts.get(condition, 0) + 1
    return selected


def select_review_rows(rows: list[dict]) -> list[dict]:
    seen_source = set()
    for row in rows:
        key = row_key(row)
        if key in seen_source:
            raise RuntimeError(f'Duplicate authoritative PDO identity: {key}.')
        seen_source.add(key)
        if row.get('containment_qc_status') != component_qc.QC_STATUS:
            raise RuntimeError(f'PDO {key} lacks completed diagnostic containment provenance.')
    selected: dict[tuple[str, str, int], dict] = {}
    reasons: dict[tuple[str, str, int], set[str]] = {}

    def add(values: list[dict], reason: str) -> None:
        for source in values:
            key = row_key(source); selected[key] = source
            reasons.setdefault(key, set()).add(reason)

    below = [row for row in rows
             if _number(row, 'full_component_fraction_inside_final_well') < 0.40]
    add(_condition_balanced_closest_below(below, MAX_BIN_ROWS), 'closest_below_0p40')
    for name, lower, upper in BIN_DEFINITIONS[1:]:
        eligible = [row for row in rows if lower <= _number(
            row, 'full_component_fraction_inside_final_well') < upper]
        add(_condition_balanced_interval(eligible, lower, upper, MAX_BIN_ROWS), name)
    add(_many_to_one_representatives(rows, MAX_MANY_TO_ONE_ROWS),
        'many_to_one_component_review')

    known = []
    for well_id in ('606', '624'):
        eligible = [row for row in rows
                    if str(row['condition_id']) == component_qc.DMSO_CONDITION
                    and _well_id(row['well_id']) == well_id]
        if not eligible:
            raise RuntimeError(f'Mandatory DMSO known visual failure well {well_id} is missing.')
        known.append(min(eligible, key=review_sort_key))
    add(known, 'known_visual_failure_mandatory')

    output = []
    bin_order = {name: index for index, (name, _, _) in enumerate(BIN_DEFINITIONS)}
    bin_order.update(many_to_one_component_review=5, known_visual_failure_mandatory=6)
    for key, source in selected.items():
        row = dict(source)
        row['sampling_reasons'] = ';'.join(sorted(reasons[key], key=lambda value: bin_order[value]))
        row['sampling_primary_bin'] = min(reasons[key], key=lambda value: bin_order[value])
        output.append(row)
    output.sort(key=review_sort_key)
    if len(output) > EXPECTED_MAXIMUM_DIAGNOSTICS:
        raise RuntimeError(f'Review selection exceeded maximum {EXPECTED_MAXIMUM_DIAGNOSTICS}: {len(output)}.')
    return output


def validate_authoritative_row(row: dict) -> None:
    required = (
        'full_component_fraction_inside_final_well',
        'full_component_fraction_outside_final_well', 'full_component_area_px2',
        'full_component_fraction_inside_production_0p86r', 'unmasked_component_id',
        'production_PDO_projected_area_px2', 'production_PDO_centroid_x_px_fullres',
        'production_PDO_centroid_y_px_fullres',
    )
    for field in required:
        if field not in row or (field != 'unmasked_component_id' and not math.isfinite(crop_base._number(row, field))):
            raise RuntimeError(f'Selected authoritative row {row_key(row)} lacks trusted {field}.')
    if not str(row['unmasked_component_id']).strip():
        raise RuntimeError(f'Selected authoritative row {row_key(row)} lacks component identity.')


def replay_verified_masks(row: dict, well: dict, pdo: dict,
                          planes: SingletonTZCYX, width: int, height: int,
                          maximum: float, minimum_area: int) -> tuple[dict, object, object, tuple, np.ndarray]:
    result = {
        'mask_replay_status': 'failed', 'mask_replay_failure_reason': 'other',
        'component_hash_verification_status': 'not_verified',
        'existing_unmasked_component_id': str(row['unmasked_component_id']),
        'regenerated_unmasked_component_id': '', 'reproduced_clipped_component_id': '',
        'replayed_crop_bounds': '',
    }
    reproduced = component_qc.reproduce_production_components(
        planes, well, width, height, maximum, minimum_area)
    clipped, _ = component_qc.match_production_row(
        pdo, reproduced, math.sqrt(_number(row, 'pixel_size_x_um')*_number(row, 'pixel_size_y_um')))
    if clipped is None:
        result['mask_replay_failure_reason'] = 'production_component_reproduction_failed'
        return result, None, None, (), np.empty((0, 0), np.uint8)
    result['reproduced_clipped_component_id'] = (
        f"{_well_id(row['well_id'])}:PDO{int(_number(row, 'pdo_number_in_well'))}:label{clipped.label_id}")
    full, match = component_qc.recover_unmasked_component(
        planes, clipped, well, width, height, maximum, minimum_area)
    result['replayed_crop_bounds'] = match.get('actual_source_clipped_crop_bounds', '')
    if full is None:
        result['mask_replay_failure_reason'] = (
            'crop_extent_incomplete' if match.get('unmasked_component_match_failure_reason') == 'crop_extent_incomplete'
            else str(match.get('unmasked_component_match_failure_reason') or 'other'))
        return result, clipped, None, (), np.empty((0, 0), np.uint8)
    regenerated, bbox, mask = component_qc.component_hash(str(row['condition_id']), full)
    result['regenerated_unmasked_component_id'] = regenerated
    if regenerated != str(row['unmasked_component_id']):
        result.update(mask_replay_failure_reason='mask_replay_hash_mismatch',
                      component_hash_verification_status='hash_mismatch')
        return result, clipped, full, bbox, mask
    result.update(mask_replay_status='verified', mask_replay_failure_reason='',
                  component_hash_verification_status='exact_hash_match')
    return result, clipped, full, bbox, mask


def _mask_in_bounds(component, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds; mask = np.zeros((y1-y0, x1-x0), np.uint8)
    coords = component.coords_yx_fullres
    yy, xx = coords[:, 0]-y0, coords[:, 1]-x0
    valid = (yy>=0)&(yy<mask.shape[0])&(xx>=0)&(xx<mask.shape[1])
    mask[yy[valid],xx[valid]]=255; return mask


def _dashed_circle(array: np.ndarray, centre: tuple[int, int], radius: int) -> None:
    for start in range(0, 360, 20):
        cv2.ellipse(array, centre, (radius, radius), 0, start, min(start+11, 360),
                    (255, 255, 0), 1, lineType=cv2.LINE_AA)


def render_diagnostic(row: dict, well: dict, clipped, full, dic: np.ndarray,
                      gfp: np.ndarray, bounds: tuple[int, int, int, int],
                      display_range: tuple[float, float], panel_size: int) -> Image.Image:
    x0,y0,_,_=bounds; dic8=crop_base._u8_local(dic); gfp8=crop_base._u8_range(gfp,*display_range)
    clipped_mask=_mask_in_bounds(clipped,bounds); full_mask=_mask_in_bounds(full,bounds)
    dic_rgb=np.stack([dic8]*3,axis=-1); gfp_rgb=np.zeros_like(dic_rgb); gfp_rgb[...,1]=gfp8
    mask_rgb=np.zeros_like(dic_rgb); mask_rgb[...,1]=full_mask
    composite=dic_rgb.copy(); composite[...,1]=np.maximum(composite[...,1],gfp8)
    panels=[dic_rgb,gfp_rgb,mask_rgb,composite]
    wx=_number(well,'x_px_fullres')-x0; wy=_number(well,'y_px_fullres')-y0; radius=_number(well,'radius_px')
    clip_contours,_=cv2.findContours(clipped_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    full_contours,_=cv2.findContours(full_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for panel in panels:
        cv2.circle(panel,(round(wx),round(wy)),round(radius),(255,255,0),2)
        _dashed_circle(panel,(round(wx),round(wy)),round(component_qc.PRODUCTION_INTERIOR_FRACTION*radius))
        cv2.drawContours(panel,clip_contours,-1,(0,255,255),2)
        cv2.drawContours(panel,full_contours,-1,(255,0,255),2)
        px=round(_number(row,'production_PDO_centroid_x_px_fullres')-x0)
        py=round(_number(row,'production_PDO_centroid_y_px_fullres')-y0)
        cv2.drawMarker(panel,(px,py),(255,255,255),cv2.MARKER_CROSS,9,2)
    lines=(
        f"{row['condition_name']} | {row['dose']} | well {row['well_id']} | PDO {int(_number(row,'pdo_number_in_well'))}",
        f"EXISTING full-component fraction inside: {_number(row,'full_component_fraction_inside_final_well'):.6f} | outside: {_number(row,'full_component_fraction_outside_final_well'):.6f}",
        f"EXISTING full-component area: {_number(row,'full_component_area_px2'):.0f} px² | Production PDO area: {_number(row,'production_PDO_projected_area_px2'):.0f} px²",
        f"Many-to-one: {row['many_production_PDOs_to_one_unmasked_component']} | Sampling: {row['sampling_reasons']}",
        f"Mask replay: {row['mask_replay_status']} | Component identity: {row['component_hash_verification_status']}",
    )
    gap=8; title_h=25; width=4*panel_size+5*gap; _,font=crop_base._fonts(); wrapped=[]
    for line in lines: wrapped.extend(textwrap.wrap(line,width=max(80,width//9),break_long_words=False) or [''])
    header=12+23*len(wrapped); canvas=Image.new('RGB',(width,header+title_h+panel_size+2*gap),'white'); draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,width,header),fill='black'); yy=6
    for line in wrapped: draw.text((10,yy),line,fill='white',font=font); yy+=23
    for index,(panel,title) in enumerate(zip(panels,('DIC','GFP','GFP mask','Composite'))):
        xx=gap+index*(panel_size+gap); py=header+gap; draw.rectangle((xx,py,xx+panel_size,py+title_h),fill='black'); draw.text((xx+6,py+3),title,fill='white',font=font)
        image=Image.fromarray(panel).resize((panel_size,panel_size),Image.Resampling.LANCZOS); canvas.paste(image,(xx,py+title_h)); image.close()
    return canvas


def _unique_index(rows: list[dict], source: str) -> dict[tuple[str, int], dict]:
    result = {}
    for row in rows:
        key=(_well_id(row['well_id']),int(round(_number(row,'pdo_number_in_well'))))
        if key in result: raise RuntimeError(f'Duplicate {source} identity {key}.')
        result[key]=row
    return result


def export_review(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
                  open_group: Callable = zarr.open_group) -> dict:
    root=args.result_root.expanduser().resolve(); authoritative=root/AUTHORITATIVE_CSV
    if not authoritative.is_file(): raise FileNotFoundError(f'Missing authoritative completed measurements: {authoritative}')
    selected=select_review_rows(_read_csv(authoritative)); output=root/OUTPUT_DIRECTORY; output.mkdir(parents=True,exist_ok=True)
    batch_path=root/'batch_status.json'; batch=_read_json(batch_path) if batch_path.is_file() else {}
    manifest=[]
    by_condition: dict[str,list[dict]]={}
    for row in selected: validate_authoritative_row(row); by_condition.setdefault(str(row['condition_id']),[]).append(row)
    for condition_id in sorted(by_condition):
        folder=root/condition_id; summary=_read_json(folder/'condition_summary.json')
        minimum=component_qc.validate_scientific_settings(summary)
        zarr_path=crop_base.resolve_omezarr(condition_id,summary,batch,args.cache_root)
        metadata=probe(zarr_path); crop_base.validate_omezarr(metadata,summary,args.expected_pixel_size_um)
        group=open_group(str(zarr_path),mode='r'); array=group[metadata['level0_array_path']]; planes=SingletonTZCYX(array,metadata['axes']); _,height,width=planes.shape_cyx
        maximum,_=component_qc.quantitative_window_end(group,component_qc.CHANNELS['gfp'],array.dtype)
        display=crop_base._metadata_window(metadata,component_qc.CHANNELS['gfp']) or (0.0,maximum)
        wells={_well_id(row['well_id']):row for row in _read_csv(folder/'well_measurements.csv')}
        pdos=_unique_index(_read_csv(folder/'pdo_measurements.csv'),'production PDO')
        for source in by_condition[condition_id]:
            row=dict(source); key=(_well_id(row['well_id']),int(round(_number(row,'pdo_number_in_well'))))
            try:
                well=wells[key[0]]; pdo=pdos[key]
                replay,clipped,full,bbox,mask=replay_verified_masks(row,well,pdo,planes,width,height,maximum,minimum); row.update(replay)
                if row['mask_replay_status']!='verified':
                    row.update(verified_mask_file='',labelled_diagnostic='',export_status='failed_verification',error=row['mask_replay_failure_reason']); manifest.append(row); _atomic_csv(output/'diagnostic_manifest.csv',sorted(manifest,key=review_sort_key)); continue
                stem=f"{condition_id}__well_{key[0]}__PDO_{key[1]:02d}"
                mask_path=output/'verified_masks'/f'{stem}.npz'; mask_path.parent.mkdir(parents=True,exist_ok=True)
                _,clipped_bbox,clipped_mask=component_qc.component_hash(condition_id,clipped)
                temporary=mask_path.with_suffix('.npz.tmp')
                with temporary.open('wb') as handle:
                    np.savez_compressed(
                        handle,
                        full_component_mask=mask.astype(np.uint8),
                        full_component_bbox_xyxy_exclusive=np.asarray(bbox,dtype=np.int64),
                        clipped_production_component_mask=clipped_mask.astype(np.uint8),
                        clipped_component_bbox_xyxy_exclusive=np.asarray(clipped_bbox,dtype=np.int64),
                        unmasked_component_id=np.asarray(row['regenerated_unmasked_component_id']),
                    )
                os.replace(temporary,mask_path)
                wx,wy=_number(well,'x_px_fullres'),_number(well,'y_px_fullres')
                radius=_number(well,'radius_px')
                component_half=max(wx-bbox[0],bbox[2]-wx,wy-bbox[1],bbox[3]-wy)+component_qc.CROP_EDGE_GUARD_PX
                half=max(1,int(math.ceil(2.0*radius)),int(math.ceil(component_half)))
                half=min(half,component_qc.MAX_CROP_HALF_WIDTH_PX)
                requested=component_qc._requested_bounds(wx,wy,half); bounds=component_qc._actual_bounds(requested,width,height)
                x0,y0,x1,y1=bounds; dic=planes.read(component_qc.CHANNELS['dic'],slice(y0,y1),slice(x0,x1)); gfp=planes.read(component_qc.CHANNELS['gfp'],slice(y0,y1),slice(x0,x1))
                row.update(verified_mask_file=str(mask_path),mask_replay_status='verified',component_hash_verification_status='exact_hash_match')
                image=render_diagnostic(row,well,clipped,full,dic,gfp,bounds,display,args.panel_size)
                image_path=output/'labelled_diagnostics'/f'{stem}.png'; image_path.parent.mkdir(parents=True,exist_ok=True); image.save(image_path,dpi=(300,300)); image.close()
                row.update(labelled_diagnostic=str(image_path),export_status='completed',error='')
            except Exception as exc:
                row.update(mask_replay_status=row.get('mask_replay_status','failed'),mask_replay_failure_reason=row.get('mask_replay_failure_reason','other'),component_hash_verification_status=row.get('component_hash_verification_status','not_verified'),verified_mask_file='',labelled_diagnostic='',export_status='failed',error=f'{type(exc).__name__}: {exc}')
            manifest.append(row); manifest.sort(key=review_sort_key); _atomic_csv(output/'diagnostic_manifest.csv',manifest)
    completed=[row for row in manifest if row.get('export_status')=='completed' and Path(row['labelled_diagnostic']).is_file()]
    completed.sort(key=review_sort_key)
    sheets=crop_base._contact_sheets([Path(row['labelled_diagnostic']) for row in completed],output/'contact_sheets',args.contact_sheet_size)
    expected={row_key(row) for row in selected}; actual={row_key(row) for row in completed}; success=expected==actual
    summary={
        'review_version':REVIEW_VERSION,'completion_status':'completed' if success else 'failed_qc','completed_at':_now(),
        'authoritative_quantitative_source':str(authoritative),'authoritative_values_recalculated':False,
        'selection_counts':{name:sum(name in row['sampling_reasons'].split(';') for row in selected) for name,_,_ in BIN_DEFINITIONS}
                           | {'many_to_one_component_review':sum('many_to_one_component_review' in row['sampling_reasons'].split(';') for row in selected),'known_visual_failure_mandatory':sum('known_visual_failure_mandatory' in row['sampling_reasons'].split(';') for row in selected)},
        'selected_unique_PDO_rows':len(selected),'maximum_diagnostics':EXPECTED_MAXIMUM_DIAGNOSTICS,
        'verified_diagnostics':len(completed),'diagnostic_identity_set_verified':success,
        'global_sort':'increasing full_component_fraction_inside_final_well, then condition_id, well_id, pdo_number_in_well',
        'mask_replay_scope':'selected_rows_only_presentation_and_visual_validation',
        'hash_verification':'exact equality to existing unmasked_component_id required',
        'review_status':REVIEW_STATUS,'exclusion_rule':None,'contact_sheets':sheets,
        'outputs':['diagnostic_manifest.csv','labelled_diagnostics','contact_sheets','boundary_review_summary.json','verified_masks'],
    }
    _atomic_json(output/'boundary_review_summary.json',summary); return summary


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description='Final visual containment-boundary review from existing measurements plus selected-row mask-only replay.')
    parser.add_argument('--result-root',type=Path,required=True); parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--expected-pixel-size-um',type=float,default=component_qc.EXPECTED_PIXEL_SIZE_UM)
    parser.add_argument('--panel-size',type=int,default=384); parser.add_argument('--contact-sheet-size',type=int,default=12)
    return parser


def run(args: argparse.Namespace, *, probe: Callable = probe_omezarr,
        open_group: Callable = zarr.open_group) -> int:
    try:
        result=export_review(args,probe=probe,open_group=open_group)
        print(f"Final containment boundary review {result['completion_status']}: {result['verified_diagnostics']}/{result['selected_unique_PDO_rows']} hash-verified diagnostics; no exclusion rule applied.",flush=True)
        return 0 if result['completion_status']=='completed' else 1
    except Exception as exc:
        output=args.result_root.expanduser().resolve()/OUTPUT_DIRECTORY
        _atomic_json(output/'boundary_review_summary.json',{'review_version':REVIEW_VERSION,'completion_status':'failed','failed_at':_now(),'review_status':REVIEW_STATUS,'exclusion_rule':None,'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
        print(f'Final containment boundary review FAILED: {type(exc).__name__}: {exc}',flush=True); return 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__=='__main__':
    raise SystemExit(main())
