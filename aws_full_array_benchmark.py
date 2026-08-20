from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy.ndimage import gaussian_filter

from analysis_core import Settings, segment_pdos
from nd2_omezarr import probe_omezarr


def _window_end(root, channel: int, dtype) -> float:
    try:
        row = root.attrs.get('omero', {}).get('channels', [])[channel]
        end = float(row.get('window', {}).get('end'))
        if end > 0:
            return end
    except Exception:
        pass
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def _channel_labels(root, count: int) -> list[str]:
    try:
        rows = root.attrs.get('omero', {}).get('channels', [])
        labels = [str(row.get('label', f'Channel {i}')) for i, row in enumerate(rows)]
    except Exception:
        labels = []
    while len(labels) < count:
        labels.append(f'Channel {len(labels)}')
    return labels[:count]


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
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _dedupe(wells: list[tuple[int,int,int]], distance_px: float) -> list[tuple[int,int,int]]:
    kept: list[tuple[int,int,int]] = []
    d2 = float(distance_px) ** 2
    for x, y, r in sorted(wells, key=lambda q: (q[1], q[0])):
        if all((x-a)**2 + (y-b)**2 > d2 for a,b,_ in kept):
            kept.append((int(x), int(y), int(r)))
    return kept


def _cluster_centres(values: list[int], tol: float, min_support: int = 2) -> list[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - float(np.mean(groups[-1]))) <= float(tol):
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups if len(g) >= int(min_support)]


def _grid_filter(wells: list[tuple[int,int,int]], expected_radius: float) -> tuple[list[tuple[int,int,int]], dict]:
    if not wells:
        return [], {'x_grid_lines': 0, 'y_grid_lines': 0, 'rejected_off_grid': 0}
    # Centre jitter is much smaller than the physical well radius; clusters with
    # at least two supporting detections remove isolated gap false-positives.
    cluster_tol = max(6.0, 0.32 * float(expected_radius))
    residual_tol = max(8.0, 0.38 * float(expected_radius))
    xs = _cluster_centres([w[0] for w in wells], cluster_tol, min_support=2)
    ys = _cluster_centres([w[1] for w in wells], cluster_tol, min_support=2)
    if not xs or not ys:
        return wells, {'x_grid_lines': len(xs), 'y_grid_lines': len(ys), 'rejected_off_grid': 0, 'grid_filter_applied': False}
    kept=[]
    for x,y,r in wells:
        dx=min(abs(float(x)-g) for g in xs)
        dy=min(abs(float(y)-g) for g in ys)
        if dx <= residual_tol and dy <= residual_tol:
            kept.append((x,y,r))
    return kept, {
        'x_grid_lines': len(xs), 'y_grid_lines': len(ys),
        'cluster_tolerance_px': cluster_tol, 'residual_tolerance_px': residual_tol,
        'rejected_off_grid': len(wells)-len(kept), 'grid_filter_applied': True,
    }


def _detect_wells_tile(dic_u8: np.ndarray, expected_radius: float, p2: float) -> np.ndarray:
    rmin=max(2,int(round(expected_radius*0.80)))
    rmax=max(rmin+2,int(round(expected_radius*1.20)))
    min_dist=max(2,int(round(expected_radius*1.5)))
    blur=cv2.GaussianBlur(dic_u8,(7,7),1.5)
    circles=cv2.HoughCircles(
        blur,cv2.HOUGH_GRADIENT,dp=1.15,minDist=float(min_dist),
        param1=75.0,param2=float(p2),minRadius=rmin,maxRadius=rmax,
    )
    return np.empty((0,3),dtype=int) if circles is None else np.round(circles[0]).astype(int)


def _tiles(width:int,height:int,tile:int):
    for y0 in range(0,height,tile):
        for x0 in range(0,width,tile):
            yield x0,y0,min(tile,width-x0),min(tile,height-y0)


def main() -> int:
    ap=argparse.ArgumentParser(description='Low-memory full-array benchmark for the validated DIC+GFP KT3 OME-Zarr workflow.')
    ap.add_argument('source',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('full_array_benchmark'))
    ap.add_argument('--tile',type=int,default=4096)
    ap.add_argument('--gfp-channel',type=int,default=0)
    ap.add_argument('--dic-channel',type=int,default=1)
    ap.add_argument('--well-diameter-um',type=float,default=100.0)
    ap.add_argument('--hough-p2',type=float,default=27.0)
    ap.add_argument('--green-low',type=float,default=30.0)
    ap.add_argument('--green-high',type=float,default=45.0)
    ap.add_argument('--pdo-min-area',type=int,default=20)
    args=ap.parse_args()

    started=time.time()
    src=args.source.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    meta=probe_omezarr(src)
    root=zarr.open_group(str(src),mode='r'); arr=root[meta['level0_array_path']]
    if meta['axes'] != ['C','Y','X'] or arr.ndim != 3:
        raise RuntimeError(f"Expected C,Y,X OME-Zarr; got axes={meta['axes']} shape={arr.shape}")
    c,h,w=map(int,arr.shape); labels=_channel_labels(root,c)
    voxel=meta.get('voxel_size_um') or {}
    px_x=float(voxel.get('x',0) or 0); px_y=float(voxel.get('y',0) or 0)
    if px_x <= 0 or px_y <= 0:
        raise RuntimeError('OME-Zarr metadata does not contain valid physical X/Y pixel calibration.')
    if not (0 <= int(args.gfp_channel) < c and 0 <= int(args.dic_channel) < c):
        raise RuntimeError(f'Channel mapping GFP={args.gfp_channel}, DIC={args.dic_channel} is invalid for {c} channels.')
    px_um=(px_x+px_y)/2.0
    expected_radius=float(args.well_diameter_um)/(2.0*px_um)
    tile=max(1024,int(args.tile)); overlap=int(math.ceil(expected_radius*3.0))
    tile_list=list(_tiles(w,h,tile))

    # PASS 1: structural DIC only. Local contrast stretching is deliberately
    # restricted to geometry detection; fluorescence remains absolutely scaled.
    raw_wells=[]; t_scan=time.time()
    for ti,(cx0,cy0,cw,ch) in enumerate(tile_list,1):
        rx0=max(0,cx0-overlap); ry0=max(0,cy0-overlap)
        rx1=min(w,cx0+cw+overlap); ry1=min(h,cy0+ch+overlap)
        dic_raw=np.asarray(arr[int(args.dic_channel),ry0:ry1,rx0:rx1])
        dic=_u8_local(dic_raw)
        local=_detect_wells_tile(dic,expected_radius,args.hough_p2)
        for lx,ly,r in local:
            gx,gy=int(rx0+lx),int(ry0+ly)
            if not (cx0 <= gx < cx0+cw and cy0 <= gy < cy0+ch):
                continue
            if gx-r < 2 or gx+r >= w-2 or gy-r < 2 or gy+r >= h-2:
                continue
            raw_wells.append((gx,gy,int(r)))
        if ti == 1 or ti % 5 == 0 or ti == len(tile_list):
            print(f'Well scan: {ti}/{len(tile_list)} tiles; candidates={len(raw_wells)}',flush=True)

    raw_wells=_dedupe(raw_wells,max(12.0,0.30*expected_radius))
    filtered,grid_info=_grid_filter(raw_wells,expected_radius)
    filtered=_dedupe(filtered,max(12.0,0.30*expected_radius))
    scan_seconds=time.time()-t_scan

    with (out/'wells_raw.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.writer(fh); wr.writerow(['x_px_fullres','y_px_fullres','radius_px']); wr.writerows(raw_wells)
    with (out/'wells_grid_filtered.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.writer(fh); wr.writerow(['well_id','x_px_fullres','y_px_fullres','radius_px'])
        for i,(x,y,r) in enumerate(filtered,1): wr.writerow([i,x,y,r])

    # Group accepted wells by core tile so each GFP tile is decompressed once.
    by_tile={i:[] for i in range(len(tile_list))}
    for wi,(x,y,r) in enumerate(filtered,1):
        tx=min(w//tile, x//tile); ty=min(h//tile, y//tile)
        cols=math.ceil(w/tile); idx=int(ty*cols+tx)
        by_tile.setdefault(idx,[]).append((wi,x,y,r))

    settings=Settings(
        well_diameter_um=float(args.well_diameter_um),green_low=float(args.green_low),
        green_high=float(args.green_high),pdo_min_area=int(args.pdo_min_area),
        rfp_psc_present=False,split_pdos=False,
    )
    gfp_max=_window_end(root,int(args.gfp_channel),arr.dtype)
    pdo_rows=[]; well_rows=[]; t_pdo=time.time()
    processed=0
    for ti,(cx0,cy0,cw,ch) in enumerate(tile_list):
        wells_here=by_tile.get(ti,[])
        if not wells_here:
            continue
        rx0=max(0,cx0-overlap); ry0=max(0,cy0-overlap)
        rx1=min(w,cx0+cw+overlap); ry1=min(h,cy0+ch+overlap)
        graw=np.asarray(arr[int(args.gfp_channel),ry0:ry1,rx0:rx1])
        gfp=_u8_absolute(graw,gfp_max)
        for wi,wx,wy,wr in wells_here:
            lx,ly=int(wx-rx0),int(wy-ry0)
            cr=int(math.ceil(wr*0.95)); xa=max(0,lx-cr); xb=min(gfp.shape[1],lx+cr+1); ya=max(0,ly-cr); yb=min(gfp.shape[0],ly+cr+1)
            sub=gfp[ya:yb,xa:xb].astype(np.float32)
            yy,xx=np.ogrid[:sub.shape[0],:sub.shape[1]]; scx=lx-xa; scy=ly-ya
            interior=(xx-scx)**2+(yy-scy)**2 <= (0.86*wr)**2
            green=gaussian_filter(np.where(interior,sub,0.0),0.8)
            objs=segment_pdos(green,settings)
            kept=[]
            for obj in objs:
                ox=float(obj['x']); oy=float(obj['y'])
                if (ox-scx)**2+(oy-scy)**2 <= (0.86*wr)**2:
                    kept.append(obj)
            total_area_px=sum(float(o['area']) for o in kept)
            well_rows.append({
                'well_id':wi,'x_px_fullres':wx,'y_px_fullres':wy,'radius_px':wr,
                'PDO_count':len(kept),'PDO_present':bool(kept),
                'total_PDO_projected_area_px2':total_area_px,
                'total_PDO_projected_area_um2':total_area_px*(px_um**2),
            })
            for pi,obj in enumerate(kept,1):
                area=float(obj['area']); gx=rx0+xa+float(obj['x']); gy=ry0+ya+float(obj['y'])
                pdo_rows.append({
                    'well_id':wi,'pdo_number_in_well':pi,'centroid_x_px_fullres':gx,'centroid_y_px_fullres':gy,
                    'projected_area_px2':area,'projected_area_um2':area*(px_um**2),
                    'equivalent_circular_diameter_um':2.0*math.sqrt(area/math.pi)*px_um,
                })
            processed += 1
        if processed and (processed % 500 < len(wells_here) or processed == len(filtered)):
            print(f'PDO analysis: {processed}/{len(filtered)} wells; PDOs={len(pdo_rows)}',flush=True)

    pdo_seconds=time.time()-t_pdo
    fields_w=['well_id','x_px_fullres','y_px_fullres','radius_px','PDO_count','PDO_present','total_PDO_projected_area_px2','total_PDO_projected_area_um2']
    with (out/'well_measurements.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.DictWriter(fh,fieldnames=fields_w); wr.writeheader(); wr.writerows(well_rows)
    fields_p=['well_id','pdo_number_in_well','centroid_x_px_fullres','centroid_y_px_fullres','projected_area_px2','projected_area_um2','equivalent_circular_diameter_um']
    with (out/'pdo_measurements.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.DictWriter(fh,fieldnames=fields_p); wr.writeheader(); wr.writerows(pdo_rows)

    elapsed=time.time()-started
    summary={
        'source':str(src),'shape_cyx':[c,h,w],'channel_labels':labels,'dtype':str(arr.dtype),
        'pixel_size_um':{'x':px_x,'y':px_y},'well_diameter_um':float(args.well_diameter_um),
        'gfp_channel':int(args.gfp_channel),'dic_channel':int(args.dic_channel),'hough_p2':float(args.hough_p2),
        'expected_well_radius_px':expected_radius,'tile_size_px':tile,'tile_count':len(tile_list),
        'raw_well_candidates':len(raw_wells),'grid_filtered_wells':len(filtered),**grid_info,
        'pdo_segmentation_mode':'conservative_contiguous_components_no_watershed_split',
        'green_low_uint8':float(args.green_low),'green_high_uint8':float(args.green_high),
        'pdo_positive_wells':sum(bool(r['PDO_present']) for r in well_rows),'detected_pdos':len(pdo_rows),
        'well_scan_seconds':scan_seconds,'pdo_analysis_seconds':pdo_seconds,'total_elapsed_seconds':elapsed,
        'outputs':['wells_raw.csv','wells_grid_filtered.csv','well_measurements.csv','pdo_measurements.csv'],
        'qc_status':'automated_full_array_benchmark_not_manually_reviewed',
    }
    (out/'benchmark_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
