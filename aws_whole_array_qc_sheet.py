from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont


def _read_scale(root) -> tuple[float, float]:
    try:
        ms = root.attrs['multiscales'][0]
        scale = ms['datasets'][0]['coordinateTransformations'][0]['scale']
        return float(scale[-1]), float(scale[-2])
    except Exception:
        return 1.0, 1.0


def _window_end(root, channel: int, dtype) -> float:
    try:
        row = root.attrs.get('omero', {}).get('channels', [])[channel]
        end = float(row.get('window', {}).get('end'))
        if end > 0:
            return end
    except Exception:
        pass
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def _u8_absolute(arr: np.ndarray, maximum: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return np.clip(a * (255.0 / max(float(maximum), 1.0)), 0, 255).astype(np.uint8)


def _u8_local(arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        lo = float(np.min(finite)); hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _spread_sample(rows: list[dict], n: int, y_key: str = 'y_px_fullres') -> list[dict]:
    if len(rows) <= n:
        return rows
    rows = sorted(rows, key=lambda r: float(r[y_key]))
    idx = np.linspace(0, len(rows)-1, n)
    return [rows[int(round(i))] for i in idx]


def _crop_overlay(arr, x: int, y: int, radius: int, gfp_channel: int, dic_channel: int, gfp_max: float):
    h = int(arr.shape[1]); w = int(arr.shape[2])
    half = int(max(110, round(radius * 1.55)))
    x0=max(0,x-half); x1=min(w,x+half+1); y0=max(0,y-half); y1=min(h,y+half+1)
    graw=np.asarray(arr[gfp_channel,y0:y1,x0:x1])
    draw=np.asarray(arr[dic_channel,y0:y1,x0:x1])
    gfp=_u8_absolute(graw,gfp_max)
    dic=_u8_local(draw)
    rgb=np.stack([dic,dic,dic],axis=-1)
    rgb[...,1]=np.maximum(rgb[...,1],gfp)
    return Image.fromarray(rgb), x0, y0


def _make_sheet(title: str, rows: list[dict], outpath: Path, arr, gfp_channel: int, dic_channel: int,
                gfp_max: float, pdo_by_well: dict[int,list[dict]] | None = None, rejected: bool = False):
    cols, rows_n = 4, 3
    cell_w, image_h, label_h = 360, 320, 42
    cell_h=image_h+label_h
    sheet=Image.new('RGB',(cols*cell_w,rows_n*cell_h+46),'white')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.load_default()
    draw.text((10,14),title,fill='black',font=font)

    for i,row in enumerate(rows[:12]):
        x=int(float(row['x_px_fullres'])); y=int(float(row['y_px_fullres'])); r=int(float(row['radius_px']))
        crop,x0,y0=_crop_overlay(arr,x,y,r,gfp_channel,dic_channel,gfp_max)
        crop=crop.resize((image_h,image_h),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(cell_w,cell_h),'white')
        canvas.paste(crop,((cell_w-image_h)//2,label_h))
        cd=ImageDraw.Draw(canvas)

        if rejected:
            sx=image_h/crop.width if crop.width else 1.0
            # Crop has already been resized; candidate should be centered except at image edges.
            cx=(cell_w-image_h)//2 + image_h/2
            cy=label_h + image_h/2
            rr=max(8,int(round(r*(image_h/(2*max(110,round(r*1.55))+1)))))
            cd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=(255,0,255),width=3)
            label=f"Rejected cand. {row.get('candidate_id','?')}"
        else:
            wid=int(float(row['well_id']))
            pcount=int(float(row.get('PDO_count',0)))
            label=f"Well {wid} | PDO count {pcount}"
            for p in (pdo_by_well or {}).get(wid,[]):
                px=float(p['centroid_x_px_fullres']); py=float(p['centroid_y_px_fullres']); area=float(p['projected_area_px2'])
                # Map source coordinates into the displayed crop.
                half=max(110,round(r*1.55)); src_w=2*half+1
                scale=image_h/src_w
                dx=(cell_w-image_h)//2 + (px-(x-half))*scale
                dy=label_h + (py-(y-half))*scale
                pr=max(4,math.sqrt(area/math.pi)*scale)
                cd.ellipse((dx-pr,dy-pr,dx+pr,dy+pr),outline=(0,255,255),width=3)
        cd.text((8,12),label,fill='black',font=font)
        sx=(i%cols)*cell_w; sy=46+(i//cols)*cell_h
        sheet.paste(canvas,(sx,sy))
    outpath.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(outpath)


def main() -> int:
    ap=argparse.ArgumentParser(description='Create 4x3 whole-array QC contact sheets from refined lattice benchmark outputs.')
    ap.add_argument('source',type=Path)
    ap.add_argument('results_dir',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('whole_array_qc'))
    ap.add_argument('--gfp-channel',type=int,default=0)
    ap.add_argument('--dic-channel',type=int,default=1)
    args=ap.parse_args()

    src=args.source.expanduser().resolve(); results=args.results_dir.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    root=zarr.open_group(str(src),mode='r'); arr=root['0']; gfp_max=_window_end(root,int(args.gfp_channel),arr.dtype)

    wells=_read_csv(results/'well_measurements.csv')
    pdos=_read_csv(results/'pdo_measurements.csv')
    audit=_read_csv(results/'lattice_audit.csv')
    pdo_by_well={}
    for p in pdos:
        pdo_by_well.setdefault(int(float(p['well_id'])),[]).append(p)

    positive=[r for r in wells if str(r.get('PDO_present','')).lower() in {'true','1'}]
    negative=[r for r in wells if str(r.get('PDO_present','')).lower() not in {'true','1'}]
    rejected=[r for r in audit if str(r.get('accepted','')).lower() not in {'true','1'}]

    pos_sample=_spread_sample(positive,12)
    neg_sample=_spread_sample(negative,12)
    rej_sample=_spread_sample(rejected,12)

    _make_sheet('Whole-array QC: GFP-positive accepted wells',pos_sample,out/'qc_positive_4x3.png',arr,args.gfp_channel,args.dic_channel,gfp_max,pdo_by_well,False)
    _make_sheet('Whole-array QC: GFP-negative accepted wells',neg_sample,out/'qc_negative_4x3.png',arr,args.gfp_channel,args.dic_channel,gfp_max,pdo_by_well,False)
    _make_sheet('Whole-array QC: rejected off-grid candidates',rej_sample,out/'qc_rejected_4x3.png',arr,args.gfp_channel,args.dic_channel,gfp_max,None,True)

    summary={
        'accepted_wells':len(wells),'pdo_positive_wells':len(positive),'pdo_negative_wells':len(negative),
        'rejected_candidates':len(rejected),'positive_sample_n':len(pos_sample),'negative_sample_n':len(neg_sample),
        'rejected_sample_n':len(rej_sample),'outputs':['qc_positive_4x3.png','qc_negative_4x3.png','qc_rejected_4x3.png']
    }
    (out/'qc_sheet_summary.json').write_text(__import__('json').dumps(summary,indent=2),encoding='utf-8')
    print(__import__('json').dumps(summary,indent=2))
    print(f'QC sheets written to: {out}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
