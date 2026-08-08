from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _u8_local(arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    a=np.asarray(arr,dtype=np.float32)
    finite=a[np.isfinite(a)]
    if finite.size==0: return np.zeros(a.shape,dtype=np.uint8)
    lo=float(np.percentile(finite,low_pct)); hi=float(np.percentile(finite,high_pct))
    if hi<=lo: lo=float(np.min(finite)); hi=float(np.max(finite))
    if hi<=lo: return np.zeros(a.shape,dtype=np.uint8)
    return np.clip((a-lo)*(255.0/(hi-lo)),0,255).astype(np.uint8)


def _window_end(root, channel: int, dtype) -> float:
    try:
        row=root.attrs.get('omero',{}).get('channels',[])[channel]
        end=float(row.get('window',{}).get('end'))
        if end>0: return end
    except Exception:
        pass
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype,np.integer) else 1.0


def _crop(arr,row,dic_channel,gfp_channel,gfp_max):
    x=int(float(row['x_px_fullres'])); y=int(float(row['y_px_fullres'])); r=float(row['radius_px'])
    half=int(max(120,round(r*1.7)))
    h=int(arr.shape[1]); w=int(arr.shape[2])
    x0=max(0,x-half); x1=min(w,x+half+1); y0=max(0,y-half); y1=min(h,y+half+1)
    dic=_u8_local(np.asarray(arr[dic_channel,y0:y1,x0:x1]))
    graw=np.asarray(arr[gfp_channel,y0:y1,x0:x1])
    gfp=np.clip(graw.astype(np.float32)*(255.0/max(gfp_max,1.0)),0,255).astype(np.uint8)
    rgb=np.stack([dic,dic,dic],axis=-1); rgb[...,1]=np.maximum(rgb[...,1],gfp)
    return Image.fromarray(rgb)


def main() -> int:
    ap=argparse.ArgumentParser(description='QC all PDO-positive wells rejected by provisional DIC structural filtering.')
    ap.add_argument('source',type=Path)
    ap.add_argument('refined_results_dir',type=Path)
    ap.add_argument('structure_results_dir',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('rejected_positive_qc'))
    ap.add_argument('--dic-channel',type=int,default=1)
    ap.add_argument('--gfp-channel',type=int,default=0)
    args=ap.parse_args()

    src=args.source.expanduser().resolve(); refined=args.refined_results_dir.expanduser().resolve(); structured=args.structure_results_dir.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    wells=_read_csv(refined/'well_measurements.csv')
    scored=_read_csv(structured/'all_lattice_wells_with_structure_score.csv')
    score_by_id={int(float(r['well_id'])):r for r in scored}

    rejected_positive=[]
    for row in wells:
        wid=int(float(row['well_id']))
        pdo_present=str(row.get('PDO_present','')).lower() in {'true','1'}
        s=score_by_id.get(wid)
        if pdo_present and s is not None and str(s.get('structure_keep','')).lower() not in {'true','1'}:
            q=dict(row); q['wall_darkness_vs_outer']=s.get('wall_darkness_vs_outer'); rejected_positive.append(q)

    root=zarr.open_group(str(src),mode='r'); arr=root['0']; gfp_max=_window_end(root,args.gfp_channel,arr.dtype)
    cols=3; n=max(1,len(rejected_positive)); rows_n=int(np.ceil(n/cols)); cell_w=390; image_h=330; label_h=60
    sheet=Image.new('RGB',(cols*cell_w,rows_n*(image_h+label_h)+50),'white')
    draw=ImageDraw.Draw(sheet); font=ImageFont.load_default(); draw.text((10,15),'QC: PDO-positive wells rejected by provisional DIC structure filter',fill='black',font=font)
    for i,row in enumerate(rejected_positive):
        crop=_crop(arr,row,args.dic_channel,args.gfp_channel,gfp_max).resize((image_h,image_h),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(cell_w,image_h+label_h),'white'); canvas.paste(crop,((cell_w-image_h)//2,label_h))
        cd=ImageDraw.Draw(canvas); wid=int(float(row['well_id'])); score=float(row['wall_darkness_vs_outer']); pcount=int(float(row.get('PDO_count',0)))
        cd.text((8,10),f'Well {wid} | PDO {pcount} | score {score:.3f} | REJECT',fill='black',font=font)
        sx=(i%cols)*cell_w; sy=50+(i//cols)*(image_h+label_h); sheet.paste(canvas,(sx,sy))
    sheet.save(out/'rejected_PDO_positive_QC.png')

    summary={
        'rejected_pdo_positive_count':len(rejected_positive),
        'well_ids':[int(float(r['well_id'])) for r in rejected_positive],
        'scores':[float(r['wall_darkness_vs_outer']) for r in rejected_positive],
        'qc_file':'rejected_PDO_positive_QC.png',
        'interpretation':'If these are genuine microwells/PDOs, the provisional single-metric structural cutoff is unsafe and should not be used in production.'
    }
    (out/'rejected_positive_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
