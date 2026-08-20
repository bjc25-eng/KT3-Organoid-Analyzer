from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

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


def _u8_absolute(arr: np.ndarray, maximum: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return np.clip(a * (255.0 / max(float(maximum), 1.0)), 0, 255).astype(np.uint8)


def _load_wells(path: Path) -> list[tuple[int,int,int]]:
    rows=[]
    with path.open(newline='',encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rows.append((int(float(row['x_px_fullres'])), int(float(row['y_px_fullres'])), int(float(row['radius_px']))))
    return rows


def _infer_pitch(points: np.ndarray, expected_radius: float) -> tuple[float, list[float]]:
    if len(points) < 10:
        raise RuntimeError('Too few well candidates to infer lattice pitch.')
    tree=cKDTree(points)
    k=min(12,len(points))
    dists,_=tree.query(points,k=k)
    diameter=2.0*float(expected_radius)
    low=1.03*diameter
    high=2.20*diameter
    candidates=[]
    for row in np.atleast_2d(dists):
        valid=[float(d) for d in row[1:] if np.isfinite(d) and low <= float(d) <= high]
        if valid:
            candidates.append(min(valid))
    if len(candidates) < max(20, int(0.1*len(points))):
        raise RuntimeError(f'Could not infer a stable lattice pitch; only {len(candidates)} candidate neighbour distances.')
    pitch=float(np.median(candidates))
    return pitch,candidates


def _lattice_filter(wells: list[tuple[int,int,int]], pitch: float, image_w: int, image_h: int) -> tuple[list[tuple[int,int,int]], list[dict], dict]:
    pts=np.asarray([(x,y) for x,y,_ in wells],dtype=float)
    tree=cKDTree(pts)
    low=0.78*float(pitch); high=1.22*float(pitch)
    accepted=[]; audit=[]
    for i,(x,y,r) in enumerate(wells):
        neighbours=tree.query_ball_point([x,y],r=high)
        direct=[]
        for j in neighbours:
            if j == i:
                continue
            d=float(np.linalg.norm(pts[j]-pts[i]))
            if low <= d <= high:
                direct.append(d)
        # Genuine interior lattice sites have several direct neighbours.  Keep
        # sites with >=2. Near full-image boundaries allow one direct neighbour
        # because neighbouring wells may lie beyond the acquisition bounds.
        near_image_edge=(x < 1.3*pitch or y < 1.3*pitch or image_w-x < 1.3*pitch or image_h-y < 1.3*pitch)
        keep=len(direct) >= 2 or (near_image_edge and len(direct) >= 1)
        audit.append({
            'candidate_id':i+1,'x_px_fullres':x,'y_px_fullres':y,'radius_px':r,
            'direct_lattice_neighbour_count':len(direct),
            'median_direct_neighbour_distance_px':float(np.median(direct)) if direct else np.nan,
            'near_image_edge':bool(near_image_edge),'accepted':bool(keep),
        })
        if keep:
            accepted.append((x,y,r))
    info={
        'pitch_px':float(pitch),'annulus_low_px':low,'annulus_high_px':high,
        'raw_candidates':len(wells),'accepted_wells':len(accepted),'rejected_wells':len(wells)-len(accepted),
        'acceptance_rule':'>=2 neighbours within 0.78-1.22 pitch; image-edge candidates may have >=1',
    }
    return accepted,audit,info


def _tiles(width:int,height:int,tile:int):
    for y0 in range(0,height,tile):
        for x0 in range(0,width,tile):
            yield x0,y0,min(tile,width-x0),min(tile,height-y0)


def _write_dicts(path:Path,rows:list[dict],fields:list[str]):
    with path.open('w',newline='',encoding='utf-8') as fh:
        wr=csv.DictWriter(fh,fieldnames=fields); wr.writeheader(); wr.writerows(rows)


def main() -> int:
    ap=argparse.ArgumentParser(description='Refine raw full-array well detections with local lattice consistency and rerun only the fast GFP PDO pass.')
    ap.add_argument('source',type=Path)
    ap.add_argument('raw_wells_csv',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('refined_lattice_benchmark'))
    ap.add_argument('--tile',type=int,default=4096)
    ap.add_argument('--gfp-channel',type=int,default=0)
    ap.add_argument('--well-diameter-um',type=float,default=100.0)
    ap.add_argument('--green-low',type=float,default=30.0)
    ap.add_argument('--green-high',type=float,default=45.0)
    ap.add_argument('--pdo-min-area',type=int,default=20)
    args=ap.parse_args()

    started=time.time(); src=args.source.expanduser().resolve(); raw_csv=args.raw_wells_csv.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    meta=probe_omezarr(src)
    root=zarr.open_group(str(src),mode='r'); arr=root[meta['level0_array_path']]
    if meta['axes'] != ['C','Y','X'] or arr.ndim != 3:
        raise RuntimeError(f"Expected C,Y,X OME-Zarr; got axes={meta['axes']} shape={arr.shape}")
    c,h,w=map(int,arr.shape)
    voxel=meta.get('voxel_size_um') or {}
    px_x=float(voxel.get('x',0) or 0); px_y=float(voxel.get('y',0) or 0)
    if px_x <= 0 or px_y <= 0:
        raise RuntimeError('OME-Zarr metadata does not contain valid physical X/Y pixel calibration.')
    if not 0 <= int(args.gfp_channel) < c:
        raise RuntimeError(f'GFP channel {args.gfp_channel} is invalid for {c} channels.')
    px_um=(px_x+px_y)/2.0
    expected_radius=float(args.well_diameter_um)/(2.0*px_um)
    wells=_load_wells(raw_csv)
    pts=np.asarray([(x,y) for x,y,_ in wells],dtype=float)
    pitch,pitch_samples=_infer_pitch(pts,expected_radius)
    accepted,audit,info=_lattice_filter(wells,pitch,w,h)
    info['pitch_um']=float(pitch*px_um)
    info['pitch_sample_count']=len(pitch_samples)
    info['pitch_sample_p10_px']=float(np.percentile(pitch_samples,10))
    info['pitch_sample_p90_px']=float(np.percentile(pitch_samples,90))
    info['expected_well_radius_px']=expected_radius

    _write_dicts(out/'lattice_audit.csv',audit,['candidate_id','x_px_fullres','y_px_fullres','radius_px','direct_lattice_neighbour_count','median_direct_neighbour_distance_px','near_image_edge','accepted'])
    with (out/'wells_lattice_filtered.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.writer(fh); wr.writerow(['well_id','x_px_fullres','y_px_fullres','radius_px'])
        for i,(x,y,r) in enumerate(accepted,1): wr.writerow([i,x,y,r])

    tile=max(1024,int(args.tile)); overlap=int(math.ceil(expected_radius*3.0)); tile_list=list(_tiles(w,h,tile)); cols=math.ceil(w/tile)
    by_tile={i:[] for i in range(len(tile_list))}
    for wi,(x,y,r) in enumerate(accepted,1):
        tx=min(cols-1,x//tile); ty=min(math.ceil(h/tile)-1,y//tile); idx=int(ty*cols+tx)
        by_tile[idx].append((wi,x,y,r))

    settings=Settings(well_diameter_um=float(args.well_diameter_um),green_low=float(args.green_low),green_high=float(args.green_high),pdo_min_area=int(args.pdo_min_area),rfp_psc_present=False,split_pdos=False)
    gfp_max=_window_end(root,int(args.gfp_channel),arr.dtype)
    well_rows=[]; pdo_rows=[]; t0=time.time(); processed=0
    for ti,(cx0,cy0,cw,ch) in enumerate(tile_list):
        wells_here=by_tile.get(ti,[])
        if not wells_here:
            continue
        rx0=max(0,cx0-overlap); ry0=max(0,cy0-overlap); rx1=min(w,cx0+cw+overlap); ry1=min(h,cy0+ch+overlap)
        graw=np.asarray(arr[int(args.gfp_channel),ry0:ry1,rx0:rx1]); gfp=_u8_absolute(graw,gfp_max)
        for wi,wx,wy,wr in wells_here:
            lx,ly=int(wx-rx0),int(wy-ry0); cr=int(math.ceil(wr*0.95)); xa=max(0,lx-cr); xb=min(gfp.shape[1],lx+cr+1); ya=max(0,ly-cr); yb=min(gfp.shape[0],ly+cr+1)
            sub=gfp[ya:yb,xa:xb].astype(np.float32); yy,xx=np.ogrid[:sub.shape[0],:sub.shape[1]]; scx=lx-xa; scy=ly-ya
            interior=(xx-scx)**2+(yy-scy)**2 <= (0.86*wr)**2
            green=gaussian_filter(np.where(interior,sub,0.0),0.8); objs=segment_pdos(green,settings)
            kept=[]
            for obj in objs:
                ox=float(obj['x']); oy=float(obj['y'])
                if (ox-scx)**2+(oy-scy)**2 <= (0.86*wr)**2:
                    kept.append(obj)
            total_area=sum(float(o['area']) for o in kept)
            well_rows.append({'well_id':wi,'x_px_fullres':wx,'y_px_fullres':wy,'radius_px':wr,'PDO_count':len(kept),'PDO_present':bool(kept),'total_PDO_projected_area_px2':total_area,'total_PDO_projected_area_um2':total_area*(px_um**2)})
            for pi,obj in enumerate(kept,1):
                area=float(obj['area']); gx=rx0+xa+float(obj['x']); gy=ry0+ya+float(obj['y'])
                pdo_rows.append({'well_id':wi,'pdo_number_in_well':pi,'centroid_x_px_fullres':gx,'centroid_y_px_fullres':gy,'projected_area_px2':area,'projected_area_um2':area*(px_um**2),'equivalent_circular_diameter_um':2.0*math.sqrt(area/math.pi)*px_um})
            processed+=1
        if processed and (processed % 1000 < len(wells_here) or processed == len(accepted)):
            print(f'Refined PDO analysis: {processed}/{len(accepted)} wells; PDOs={len(pdo_rows)}',flush=True)

    pdo_seconds=time.time()-t0
    _write_dicts(out/'well_measurements.csv',well_rows,['well_id','x_px_fullres','y_px_fullres','radius_px','PDO_count','PDO_present','total_PDO_projected_area_px2','total_PDO_projected_area_um2'])
    _write_dicts(out/'pdo_measurements.csv',pdo_rows,['well_id','pdo_number_in_well','centroid_x_px_fullres','centroid_y_px_fullres','projected_area_px2','projected_area_um2','equivalent_circular_diameter_um'])

    summary={**info,'pixel_size_um':{'x':px_x,'y':px_y},'gfp_channel':int(args.gfp_channel),'well_diameter_um':float(args.well_diameter_um),'green_low_uint8':float(args.green_low),'green_high_uint8':float(args.green_high),'pdo_min_area_px':int(args.pdo_min_area),'pdo_positive_wells':sum(bool(r['PDO_present']) for r in well_rows),'detected_pdos':len(pdo_rows),'pdo_analysis_seconds':pdo_seconds,'total_refinement_seconds':time.time()-started,'pdo_segmentation_mode':'conservative_contiguous_components_no_watershed_split','qc_status':'automated_lattice_refinement_not_manually_reviewed'}
    (out/'refined_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
