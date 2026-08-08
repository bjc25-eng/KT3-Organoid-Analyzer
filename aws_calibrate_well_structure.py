from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import zarr


DEFAULT_TRUE_WELL_IDS = [
    # Reviewed GFP-positive sheet
    1586, 2469, 3033, 4062, 4861, 5911, 6333, 7005, 7792, 8942, 10372, 11994,
    # Reviewed GFP-negative sheet that visibly contains a genuine microwell
    2605, 5251, 6588, 9230, 11845, 13133,
]
DEFAULT_FALSE_WELL_IDS = [
    # Reviewed GFP-negative sheet entries that do not visibly contain a microwell
    1, 1287, 3932, 7920, 10541, 14419,
]


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


def _load_wells(path: Path) -> dict[int, dict]:
    out = {}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            out[int(float(row['well_id']))] = row
    return out


def _parse_ids(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return list(default)
    return [int(q.strip()) for q in text.split(',') if q.strip()]


def _score_candidate(arr, row: dict, dic_channel: int) -> dict:
    x=int(float(row['x_px_fullres'])); y=int(float(row['y_px_fullres'])); r=float(row['radius_px'])
    half=int(max(110, round(r*1.55)))
    h=int(arr.shape[1]); w=int(arr.shape[2])
    x0=max(0,x-half); x1=min(w,x+half+1); y0=max(0,y-half); y1=min(h,y+half+1)
    raw=np.asarray(arr[int(dic_channel),y0:y1,x0:x1])
    dic=_u8_local(raw)

    cy=float(y-y0); cx=float(x-x0)
    yy,xx=np.indices(dic.shape,dtype=np.float32)
    rr=np.sqrt((xx-cx)**2+(yy-cy)**2)

    # Structural bands are expressed relative to the Hough-estimated radius.
    inner=(rr >= 0.35*r) & (rr <= 0.65*r)
    wall=(rr >= 0.78*r) & (rr <= 1.08*r)
    outer=(rr >= 1.15*r) & (rr <= 1.38*r)

    f=dic.astype(np.float32)
    gx=cv2.Sobel(f,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(f,cv2.CV_32F,0,1,ksize=3)
    grad=np.hypot(gx,gy)

    wall_mean=float(np.mean(f[wall])) if np.any(wall) else np.nan
    inner_mean=float(np.mean(f[inner])) if np.any(inner) else np.nan
    outer_mean=float(np.mean(f[outer])) if np.any(outer) else np.nan
    wall_grad_mean=float(np.mean(grad[wall])) if np.any(wall) else np.nan
    inner_grad_mean=float(np.mean(grad[inner])) if np.any(inner) else np.nan
    outer_grad_mean=float(np.mean(grad[outer])) if np.any(outer) else np.nan

    # Angular edge coverage: for each angle, take the strongest gradient in the
    # expected wall band. The threshold is derived from that candidate patch's
    # interior/background gradient distribution, not a fixed global number.
    angles=np.linspace(0,2*np.pi,96,endpoint=False)
    wall_peaks=[]
    for a in angles:
        vals=[]
        for frac in np.linspace(0.78,1.08,13):
            px=int(round(cx + frac*r*np.cos(a))); py=int(round(cy + frac*r*np.sin(a)))
            if 0 <= px < grad.shape[1] and 0 <= py < grad.shape[0]:
                vals.append(float(grad[py,px]))
        wall_peaks.append(max(vals) if vals else 0.0)
    background_grad=np.concatenate([grad[inner].ravel(),grad[outer].ravel()])
    reference=float(np.percentile(background_grad,90)) if background_grad.size else 0.0
    angular_edge_coverage=float(np.mean(np.asarray(wall_peaks) > reference)) if wall_peaks else 0.0

    return {
        'well_id':int(float(row['well_id'])),
        'x_px_fullres':x,'y_px_fullres':y,'radius_px':r,
        'inner_mean_dic_u8':inner_mean,'wall_mean_dic_u8':wall_mean,'outer_mean_dic_u8':outer_mean,
        'wall_darkness_vs_inner':inner_mean-wall_mean,
        'wall_darkness_vs_outer':outer_mean-wall_mean,
        'inner_grad_mean':inner_grad_mean,'wall_grad_mean':wall_grad_mean,'outer_grad_mean':outer_grad_mean,
        'wall_gradient_ratio_vs_background':wall_grad_mean/max(1e-6,0.5*(inner_grad_mean+outer_grad_mean)),
        'angular_edge_coverage':angular_edge_coverage,
        'angular_reference_gradient_p90':reference,
    }


def _group_summary(rows: list[dict], label: str) -> dict:
    metrics=['wall_darkness_vs_inner','wall_darkness_vs_outer','wall_gradient_ratio_vs_background','angular_edge_coverage']
    out={'label':label,'n':len(rows)}
    for m in metrics:
        vals=np.asarray([float(r[m]) for r in rows if np.isfinite(float(r[m]))],dtype=float)
        out[m]={
            'min':float(np.min(vals)) if len(vals) else None,
            'median':float(np.median(vals)) if len(vals) else None,
            'max':float(np.max(vals)) if len(vals) else None,
        }
    return out


def main() -> int:
    ap=argparse.ArgumentParser(description='Measure DIC wall/ring structure in manually reviewed true and false accepted well candidates; does not apply a cutoff.')
    ap.add_argument('source',type=Path)
    ap.add_argument('well_measurements_csv',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('well_structure_calibration'))
    ap.add_argument('--dic-channel',type=int,default=1)
    ap.add_argument('--true-well-ids',default=None,help='Comma-separated reviewed true well IDs; defaults to the current QC-sheet review set.')
    ap.add_argument('--false-well-ids',default=None,help='Comma-separated reviewed false accepted IDs; defaults to the current QC-sheet review set.')
    args=ap.parse_args()

    src=args.source.expanduser().resolve(); wells_csv=args.well_measurements_csv.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    wells=_load_wells(wells_csv)
    true_ids=_parse_ids(args.true_well_ids,DEFAULT_TRUE_WELL_IDS)
    false_ids=_parse_ids(args.false_well_ids,DEFAULT_FALSE_WELL_IDS)
    missing=[i for i in true_ids+false_ids if i not in wells]
    if missing:
        raise RuntimeError(f'Reviewed well IDs are missing from the measurement table: {missing}')

    root=zarr.open_group(str(src),mode='r'); arr=root['0']
    rows=[]
    for label,ids in [('true_well',true_ids),('false_accepted',false_ids)]:
        for wid in ids:
            score=_score_candidate(arr,wells[wid],args.dic_channel); score['manual_review_label']=label; rows.append(score)

    fields=list(rows[0].keys())
    with (out/'reviewed_well_structure_scores.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.DictWriter(fh,fieldnames=fields); wr.writeheader(); wr.writerows(rows)

    true_rows=[r for r in rows if r['manual_review_label']=='true_well']
    false_rows=[r for r in rows if r['manual_review_label']=='false_accepted']
    summary={
        'manual_review_basis':'IDs taken from the uploaded whole-array QC contact sheets; no structural cutoff is applied by this script.',
        'true_well_ids':true_ids,'false_accepted_ids':false_ids,
        'true_well_summary':_group_summary(true_rows,'true_well'),
        'false_accepted_summary':_group_summary(false_rows,'false_accepted'),
        'next_step':'Inspect whether one or more measured DIC structural metrics cleanly separate the reviewed groups before defining a production cutoff.'
    }
    (out/'well_structure_calibration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

    print('REVIEWED WELL STRUCTURE SCORES')
    print('well_id\tlabel\tdark_inner\tdark_outer\tgrad_ratio\tedge_coverage')
    for r in rows:
        print(f"{r['well_id']}\t{r['manual_review_label']}\t{r['wall_darkness_vs_inner']:.3f}\t{r['wall_darkness_vs_outer']:.3f}\t{r['wall_gradient_ratio_vs_background']:.3f}\t{r['angular_edge_coverage']:.3f}")
    print('\nGROUP SUMMARY')
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
