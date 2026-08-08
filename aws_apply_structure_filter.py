from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

DEFAULT_CUTOFF = 16.332088470458984
REVIEWED_TRUE_IDS = {1586,2469,3033,4062,4861,5911,6333,7005,7792,8942,10372,11994,2605,5251,6588,9230,11845,13133}
REVIEWED_FALSE_IDS = {1,1287,3932,7920,10541,14419}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)


def _u8_local(arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, low_pct)); hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a-lo) * (255.0/(hi-lo)), 0, 255).astype(np.uint8)


def _score_patch(dic_raw: np.ndarray, cx: float, cy: float, r: float) -> float:
    dic = _u8_local(dic_raw)
    yy, xx = np.indices(dic.shape, dtype=np.float32)
    rr = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    wall = (rr >= 0.78*r) & (rr <= 1.08*r)
    outer = (rr >= 1.15*r) & (rr <= 1.38*r)
    if not np.any(wall) or not np.any(outer):
        return float('nan')
    f = dic.astype(np.float32)
    return float(np.mean(f[outer]) - np.mean(f[wall]))


def _crop_overlay(arr, row: dict, dic_channel: int, gfp_channel: int, gfp_max: float):
    x=int(float(row['x_px_fullres'])); y=int(float(row['y_px_fullres'])); r=float(row['radius_px'])
    half=int(max(110, round(r*1.55)))
    h=int(arr.shape[1]); w=int(arr.shape[2])
    x0=max(0,x-half); x1=min(w,x+half+1); y0=max(0,y-half); y1=min(h,y+half+1)
    draw=np.asarray(arr[dic_channel,y0:y1,x0:x1]); graw=np.asarray(arr[gfp_channel,y0:y1,x0:x1])
    dic=_u8_local(draw)
    gfp=np.clip(graw.astype(np.float32)*(255.0/max(gfp_max,1.0)),0,255).astype(np.uint8)
    rgb=np.stack([dic,dic,dic],axis=-1); rgb[...,1]=np.maximum(rgb[...,1],gfp)
    return Image.fromarray(rgb)


def _window_end(root, channel: int, dtype) -> float:
    try:
        row=root.attrs.get('omero',{}).get('channels',[])[channel]
        end=float(row.get('window',{}).get('end'))
        if end>0: return end
    except Exception:
        pass
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype,np.integer) else 1.0


def _make_borderline_sheet(rows: list[dict], arr, outpath: Path, dic_channel: int, gfp_channel: int, gfp_max: float, cutoff: float):
    below=sorted([r for r in rows if float(r['wall_darkness_vs_outer']) < cutoff], key=lambda r: cutoff-float(r['wall_darkness_vs_outer']))[:6]
    above=sorted([r for r in rows if float(r['wall_darkness_vs_outer']) >= cutoff], key=lambda r: float(r['wall_darkness_vs_outer'])-cutoff)[:6]
    selected=below+above
    cols, rows_n = 4, 3
    cell_w, image_h, label_h = 360, 300, 52
    sheet=Image.new('RGB',(cols*cell_w,rows_n*(image_h+label_h)+46),'white')
    draw=ImageDraw.Draw(sheet); font=ImageFont.load_default()
    draw.text((10,14),f'Borderline DIC structural QC | cutoff = {cutoff:.3f}',fill='black',font=font)
    for i,row in enumerate(selected):
        crop=_crop_overlay(arr,row,dic_channel,gfp_channel,gfp_max).resize((image_h,image_h),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(cell_w,image_h+label_h),'white'); canvas.paste(crop,((cell_w-image_h)//2,label_h))
        cd=ImageDraw.Draw(canvas)
        score=float(row['wall_darkness_vs_outer']); wid=int(float(row['well_id']))
        status='KEEP' if score>=cutoff else 'REJECT'
        cd.text((8,10),f'Well {wid} | score {score:.3f} | {status}',fill='black',font=font)
        sx=(i%cols)*cell_w; sy=46+(i//cols)*(image_h+label_h); sheet.paste(canvas,(sx,sy))
    outpath.parent.mkdir(parents=True,exist_ok=True); sheet.save(outpath)


def main() -> int:
    ap=argparse.ArgumentParser(description='Apply provisional DIC wall-darkness structural filter to lattice-accepted wells and generate borderline QC.')
    ap.add_argument('source',type=Path)
    ap.add_argument('results_dir',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('structure_filtered_results'))
    ap.add_argument('--dic-channel',type=int,default=1)
    ap.add_argument('--gfp-channel',type=int,default=0)
    ap.add_argument('--cutoff',type=float,default=DEFAULT_CUTOFF)
    args=ap.parse_args()

    src=args.source.expanduser().resolve(); results=args.results_dir.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    wells=_read_csv(results/'well_measurements.csv'); pdos=_read_csv(results/'pdo_measurements.csv')
    root=zarr.open_group(str(src),mode='r'); arr=root['0']; h=int(arr.shape[1]); w=int(arr.shape[2]); gfp_max=_window_end(root,args.gfp_channel,arr.dtype)

    scored=[]
    for i,row in enumerate(wells,1):
        x=int(float(row['x_px_fullres'])); y=int(float(row['y_px_fullres'])); r=float(row['radius_px'])
        half=int(max(110,round(r*1.55))); x0=max(0,x-half); x1=min(w,x+half+1); y0=max(0,y-half); y1=min(h,y+half+1)
        raw=np.asarray(arr[args.dic_channel,y0:y1,x0:x1])
        score=_score_patch(raw,float(x-x0),float(y-y0),r)
        q=dict(row); q['wall_darkness_vs_outer']=score; q['structure_keep']=bool(np.isfinite(score) and score>=args.cutoff); scored.append(q)
        if i==1 or i%2000==0 or i==len(wells): print(f'Structural scoring: {i}/{len(wells)}',flush=True)

    kept=[r for r in scored if r['structure_keep']]
    kept_ids={int(float(r['well_id'])) for r in kept}
    kept_pdos=[r for r in pdos if int(float(r['well_id'])) in kept_ids]

    fields=list(scored[0].keys())
    _write_csv(out/'all_lattice_wells_with_structure_score.csv',scored,fields)
    _write_csv(out/'final_filtered_well_measurements.csv',kept,fields)
    if pdos:
        _write_csv(out/'final_filtered_pdo_measurements.csv',kept_pdos,list(pdos[0].keys()))
    else:
        _write_csv(out/'final_filtered_pdo_measurements.csv',[],['well_id'])

    _make_borderline_sheet(scored,arr,out/'borderline_structure_qc_4x3.png',args.dic_channel,args.gfp_channel,gfp_max,args.cutoff)

    reviewed_true_available=sorted(REVIEWED_TRUE_IDS & {int(float(r['well_id'])) for r in scored})
    reviewed_false_available=sorted(REVIEWED_FALSE_IDS & {int(float(r['well_id'])) for r in scored})
    score_by_id={int(float(r['well_id'])):float(r['wall_darkness_vs_outer']) for r in scored}
    summary={
        'provisional_cutoff_wall_darkness_vs_outer':float(args.cutoff),
        'cutoff_basis':'midpoint between reviewed true-well minimum 16.466827392578125 and reviewed false-accepted maximum 16.197349548339844',
        'lattice_accepted_input_wells':len(scored),'structure_filtered_wells':len(kept),'structure_rejected_wells':len(scored)-len(kept),
        'pdo_positive_wells_after_filter':sum(str(r.get('PDO_present','')).lower() in {'true','1'} for r in kept),
        'pdo_objects_after_filter':len(kept_pdos),
        'reviewed_true_ids_available':reviewed_true_available,
        'reviewed_true_passed':sum(score_by_id[i]>=args.cutoff for i in reviewed_true_available),
        'reviewed_false_ids_available':reviewed_false_available,
        'reviewed_false_rejected':sum(score_by_id[i]<args.cutoff for i in reviewed_false_available),
        'borderline_qc_file':'borderline_structure_qc_4x3.png',
        'qc_status':'provisional cutoff; visually inspect borderline QC before production use'
    }
    (out/'structure_filter_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
