from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)


def _load_pitch(hex_dir: Path) -> float:
    d = json.loads((hex_dir / 'hex_array_summary.json').read_text(encoding='utf-8'))
    return float(d['pitch_px'])


def _pca_axes(points: np.ndarray):
    centre = np.mean(points, axis=0)
    vals, vecs = np.linalg.eigh(np.cov((points-centre).T))
    long_axis = vecs[:, np.argmax(vals)]
    if long_axis[1] < 0:
        long_axis = -long_axis
    short_axis = np.asarray([-long_axis[1], long_axis[0]])
    return centre, long_axis, short_axis


def _robust_bounds(points: np.ndarray, pitch: float):
    centre, long_axis, short_axis = _pca_axes(points)
    d = points-centre
    l = d @ long_axis
    s = d @ short_axis

    # Estimate straight side walls from local minima/maxima along the long axis.
    bins = np.linspace(np.percentile(l, 1.0), np.percentile(l, 99.0), 60)
    lefts, rights = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        q = s[(l >= a) & (l < b)]
        if len(q) >= 30:
            lefts.append(float(np.percentile(q, 2)))
            rights.append(float(np.percentile(q, 98)))
    if len(lefts) < 10:
        raise RuntimeError('Could not estimate robust straight array boundaries.')

    # Conservative side bounds: recover ragged edge detections, but avoid large
    # excursions produced by capture-well regions or background artefacts.
    s_lo = float(np.median(lefts))
    s_hi = float(np.median(rights))

    # Top/bottom ends are taken from the dense component itself, not the full image.
    l_lo = float(np.percentile(l, 0.5))
    l_hi = float(np.percentile(l, 99.5))

    # Small tolerance around the detected array boundary. This is intentionally
    # below one full pitch so optional capture-well regions stay outside.
    side_margin = 0.55 * pitch
    end_margin = 0.40 * pitch
    return centre, long_axis, short_axis, l_lo-end_margin, l_hi+end_margin, s_lo-side_margin, s_hi+side_margin


def _coords(points: np.ndarray, centre, long_axis, short_axis):
    d = points-centre
    return d @ long_axis, d @ short_axis


def _inside(l, s, llo, lhi, slo, shi):
    return (l >= llo) & (l <= lhi) & (s >= slo) & (s <= shi)


def _u8_local(arr: np.ndarray):
    a=np.asarray(arr,dtype=np.float32)
    finite=a[np.isfinite(a)]
    if finite.size==0: return np.zeros(a.shape,dtype=np.uint8)
    lo=float(np.percentile(finite,0.5)); hi=float(np.percentile(finite,99.5))
    if hi<=lo: return np.zeros(a.shape,dtype=np.uint8)
    return np.clip((a-lo)*(255.0/(hi-lo)),0,255).astype(np.uint8)


def _make_recovered_sheet(source: Path, rows: list[dict], outpath: Path, dic_channel: int, n: int = 12):
    if not rows:
        Image.new('RGB',(900,300),'white').save(outpath)
        return
    root=zarr.open_group(str(source),mode='r'); arr=root['0']; h=int(arr.shape[1]); w=int(arr.shape[2])
    rows=sorted(rows,key=lambda r: float(r['y_px_fullres']))
    if len(rows)>n:
        idx=np.linspace(0,len(rows)-1,n)
        rows=[rows[int(round(i))] for i in idx]
    cols=4; rows_n=3; cell_w=350; image_h=285; label_h=48
    sheet=Image.new('RGB',(cols*cell_w,rows_n*(image_h+label_h)+42),'white')
    draw=ImageDraw.Draw(sheet); font=ImageFont.load_default(); draw.text((10,12),'QC: recovered detected wells inside straight array ROI',fill='black',font=font)
    for i,row in enumerate(rows):
        x=int(float(row['x_px_fullres'])); y=int(float(row['y_px_fullres'])); r=max(40,int(float(row['radius_px'])))
        half=max(115,int(round(r*1.6))); x0=max(0,x-half); x1=min(w,x+half+1); y0=max(0,y-half); y1=min(h,y+half+1)
        dic=_u8_local(np.asarray(arr[dic_channel,y0:y1,x0:x1]))
        rgb=np.stack([dic,dic,dic],axis=-1)
        crop=Image.fromarray(rgb).resize((image_h,image_h),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(cell_w,image_h+label_h),'white'); canvas.paste(crop,((cell_w-image_h)//2,label_h))
        cd=ImageDraw.Draw(canvas); wid=int(float(row['well_id'])); pdo=int(float(row.get('PDO_count',0)))
        cd.text((8,10),f'Well {wid} | PDO count {pdo}',fill='black',font=font)
        sx=(i%cols)*cell_w; sy=42+(i//cols)*(image_h+label_h); sheet.paste(canvas,(sx,sy))
    sheet.save(outpath)


def main() -> int:
    ap=argparse.ArgumentParser(description='Recover only real, already-detected lattice-consistent edge wells inside a robust straight array ROI. No synthetic/predicted wells are created.')
    ap.add_argument('source', type=Path)
    ap.add_argument('refined_results_dir', type=Path)
    ap.add_argument('hex_results_dir', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('straight_roi_recovery'))
    ap.add_argument('--dic-channel', type=int, default=1)
    ap.add_argument('--max-overview-height', type=int, default=4200)
    args=ap.parse_args()

    source=args.source.expanduser().resolve(); refined=args.refined_results_dir.expanduser().resolve(); hexdir=args.hex_results_dir.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    all_wells=_read_csv(refined/'well_measurements.csv')
    pdos=_read_csv(refined/'pdo_measurements.csv')
    seed_wells=_read_csv(hexdir/'hex_array_well_measurements.csv')
    pitch=_load_pitch(hexdir)

    seed_ids={int(float(r['well_id'])) for r in seed_wells}
    seed_pts=np.asarray([(float(r['x_px_fullres']),float(r['y_px_fullres'])) for r in seed_wells],dtype=float)
    all_pts=np.asarray([(float(r['x_px_fullres']),float(r['y_px_fullres'])) for r in all_wells],dtype=float)

    centre,long_axis,short_axis,llo,lhi,slo,shi=_robust_bounds(seed_pts,pitch)
    l_all,s_all=_coords(all_pts,centre,long_axis,short_axis)
    inside_mask=_inside(l_all,s_all,llo,lhi,slo,shi)

    final_wells=[]; recovered=[]; outside=[]
    for row,is_inside,lcoord,scoord in zip(all_wells,inside_mask,l_all,s_all):
        q=dict(row); q['roi_long_coord_px']=float(lcoord); q['roi_short_coord_px']=float(scoord); q['inside_straight_array_roi']=bool(is_inside)
        wid=int(float(row['well_id']))
        if is_inside:
            final_wells.append(q)
            if wid not in seed_ids:
                q['recovery_status']='recovered_detected_edge_well'; recovered.append(q)
            else:
                q['recovery_status']='original_main_component'
        else:
            q['recovery_status']='outside_array_roi'; outside.append(q)

    final_ids={int(float(r['well_id'])) for r in final_wells}
    final_pdos=[r for r in pdos if int(float(r['well_id'])) in final_ids]
    recovered_ids={int(float(r['well_id'])) for r in recovered}
    recovered_pdos=[r for r in pdos if int(float(r['well_id'])) in recovered_ids]

    wf=list(final_wells[0].keys())
    _write_csv(out/'final_detected_wells_inside_straight_roi.csv',final_wells,wf)
    _write_csv(out/'recovered_detected_edge_wells.csv',recovered,wf)
    _write_csv(out/'excluded_detected_wells_outside_roi.csv',outside,wf)
    if pdos:
        pf=list(pdos[0].keys()); _write_csv(out/'final_pdo_measurements_inside_straight_roi.csv',final_pdos,pf); _write_csv(out/'recovered_edge_pdo_measurements.csv',recovered_pdos,pf)

    # Overview overlay.
    root=zarr.open_group(str(source),mode='r'); arr=root['0']; _,h,w=map(int,arr.shape)
    step=max(1,int(math.ceil(h/max(500,int(args.max_overview_height)))))
    dic=_u8_local(np.asarray(arr[int(args.dic_channel),::step,::step]))
    fig_h=14; fig_w=max(3.8,fig_h*(w/h))
    plt.figure(figsize=(fig_w,fig_h)); plt.imshow(dic,cmap='gray',origin='upper',extent=[0,w,h,0],interpolation='nearest')
    if outside:
        op=np.asarray([(float(r['x_px_fullres']),float(r['y_px_fullres'])) for r in outside]); plt.scatter(op[:,0],op[:,1],s=4,c='deepskyblue',alpha=0.45,label='Excluded outside straight ROI')
    sp=np.asarray([(float(r['x_px_fullres']),float(r['y_px_fullres'])) for r in seed_wells]); plt.scatter(sp[:,0],sp[:,1],s=2.5,c='orange',alpha=0.38,label='Original main-component wells')
    if recovered:
        rp=np.asarray([(float(r['x_px_fullres']),float(r['y_px_fullres'])) for r in recovered]); plt.scatter(rp[:,0],rp[:,1],s=13,facecolors='none',edgecolors='magenta',linewidths=0.9,label='Recovered real detected edge wells')
    corners=[]
    for ll,ss in [(llo,slo),(llo,shi),(lhi,shi),(lhi,slo),(llo,slo)]: corners.append(centre+ll*long_axis+ss*short_axis)
    cp=np.asarray(corners); plt.plot(cp[:,0],cp[:,1],c='red',linewidth=1.2,label='Straight analysis ROI')
    plt.xlim(0,w); plt.ylim(h,0); plt.xlabel('Full-resolution x (px)'); plt.ylabel('Full-resolution y (px)'); plt.title('Straight ROI recovery QC — detected wells only'); plt.legend(loc='best',fontsize=6,markerscale=1.5); plt.tight_layout(); plt.savefig(out/'straight_roi_recovered_wells_overlay.png',dpi=220); plt.close()

    _make_recovered_sheet(source,recovered,out/'recovered_edge_wells_QC_4x3.png',args.dic_channel)

    summary={
        'pitch_px':pitch,
        'input_refined_lattice_wells':len(all_wells),
        'original_main_component_wells':len(seed_wells),
        'final_detected_wells_inside_straight_roi':len(final_wells),
        'recovered_real_detected_edge_wells':len(recovered),
        'excluded_detected_wells_outside_roi':len(outside),
        'pdo_positive_wells_final':sum(str(r.get('PDO_present','')).lower() in {'true','1'} for r in final_wells),
        'pdo_objects_final':len(final_pdos),
        'pdo_positive_recovered_edge_wells':sum(str(r.get('PDO_present','')).lower() in {'true','1'} for r in recovered),
        'pdo_objects_in_recovered_edge_wells':len(recovered_pdos),
        'method':'Straight ROI estimated from dominant connected component; reinclude only existing Hough-detected wells that already passed the local lattice filter. No predicted/synthetic wells are created.',
        'qc_status':'Review recovered_edge_wells_QC_4x3.png before production use.',
        'overlay':'straight_roi_recovered_wells_overlay.png',
        'recovered_qc_sheet':'recovered_edge_wells_QC_4x3.png'
    }
    (out/'straight_roi_recovery_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
