from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import zarr
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from analysis_core import Settings, segment_pdos


def _read_scale(root) -> tuple[float, float]:
    try:
        ms = root.attrs['multiscales'][0]
        scale = ms['datasets'][0]['coordinateTransformations'][0]['scale']
        return float(scale[-1]), float(scale[-2])
    except Exception:
        return 1.0, 1.0


def _channel_labels(root, count: int) -> list[str]:
    try:
        rows = root.attrs.get('omero', {}).get('channels', [])
        labels = [str(row.get('label', f'Channel {i}')) for i, row in enumerate(rows)]
    except Exception:
        labels = []
    while len(labels) < count:
        labels.append(f'Channel {len(labels)}')
    return labels[:count]


def _window_end(root, channel: int, dtype) -> float:
    try:
        row = root.attrs.get('omero', {}).get('channels', [])[channel]
        value = float(row.get('window', {}).get('end'))
        if value > 0:
            return value
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
    lo = float(np.percentile(finite, low_pct)); hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a-lo) * (255.0/(hi-lo)), 0, 255).astype(np.uint8)


def _detect_wells(dic: np.ndarray, expected_radius: float, p2: float) -> np.ndarray:
    rmin = max(2, int(round(expected_radius * 0.80)))
    rmax = max(rmin+2, int(round(expected_radius * 1.20)))
    min_dist = max(2, int(round(expected_radius * 1.5)))
    blur = cv2.GaussianBlur(dic, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.15, minDist=float(min_dist),
        param1=75.0, param2=float(p2), minRadius=rmin, maxRadius=rmax,
    )
    if circles is None:
        return np.empty((0,3), dtype=int)
    return np.round(circles[0]).astype(int)


def main() -> int:
    ap = argparse.ArgumentParser(description='QC PDO segmentation on one OME-Zarr tile using DIC wells and the separate GFP channel.')
    ap.add_argument('source', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('pdo_qc'))
    ap.add_argument('--tile', type=int, default=4096)
    ap.add_argument('--gfp-channel', type=int, default=0)
    ap.add_argument('--dic-channel', type=int, default=1)
    ap.add_argument('--well-diameter-um', type=float, default=100.0)
    ap.add_argument('--hough-p2', type=float, default=27.0)
    ap.add_argument('--green-low', type=float, default=30.0)
    ap.add_argument('--green-high', type=float, default=45.0)
    ap.add_argument('--pdo-min-area', type=int, default=20)
    args = ap.parse_args()

    src = args.source.expanduser().resolve()
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(src), mode='r')
    arr = root['0']
    if arr.ndim != 3:
        raise RuntimeError(f'Expected C,Y,X OME-Zarr; got {arr.shape}')
    c,h,w = map(int, arr.shape)
    labels = _channel_labels(root,c)
    px_x, px_y = _read_scale(root); px_um=(px_x+px_y)/2.0
    expected_radius = float(args.well_diameter_um)/(2.0*px_um)

    tile=min(int(args.tile),h,w)
    y0=max(0,h//2-tile//2); x0=max(0,w//2-tile//2)
    y1=y0+tile; x1=x0+tile
    graw=np.asarray(arr[int(args.gfp_channel),y0:y1,x0:x1])
    draw=np.asarray(arr[int(args.dic_channel),y0:y1,x0:x1])
    gfp=_u8_absolute(graw,_window_end(root,int(args.gfp_channel),arr.dtype))
    dic=_u8_local(draw)
    wells=_detect_wells(dic,expected_radius,args.hough_p2)

    settings=Settings(
        well_diameter_um=float(args.well_diameter_um),
        green_low=float(args.green_low), green_high=float(args.green_high),
        pdo_min_area=int(args.pdo_min_area), rfp_psc_present=False,
    )

    rows=[]
    total_pdos=0
    overlay=np.stack([dic,dic,dic],axis=-1)
    overlay[...,1]=np.maximum(overlay[...,1],gfp)
    marked=Image.fromarray(overlay)
    pen=ImageDraw.Draw(marked)

    for wi,(wx,wy,wr) in enumerate(wells,1):
        # High-contrast QC well outline: magenta, not used in scientific outputs.
        pen.ellipse((wx-wr,wy-wr,wx+wr,wy+wr), outline=(255,0,255), width=3)
        cr=int(math.ceil(wr*0.95))
        xa=max(0,wx-cr); xb=min(tile,wx+cr+1)
        ya=max(0,wy-cr); yb=min(tile,wy+cr+1)
        sub=gfp[ya:yb,xa:xb].astype(np.float32)
        yy,xx=np.ogrid[:sub.shape[0],:sub.shape[1]]
        cx=wx-xa; cy=wy-ya
        mask=(xx-cx)**2+(yy-cy)**2 <= (0.86*wr)**2
        green=gaussian_filter(np.where(mask,sub,0.0),0.8)
        objects=segment_pdos(green,settings)
        # Retain only candidate centroids in the interior mask.
        kept=[]
        for obj in objects:
            ox=float(obj['x']); oy=float(obj['y'])
            if (ox-cx)**2+(oy-cy)**2 <= (0.86*wr)**2:
                kept.append(obj)
        total_pdos += len(kept)
        for pi,obj in enumerate(kept,1):
            gx=xa+float(obj['x']); gy=ya+float(obj['y'])
            area_px=float(obj['area'])
            deq_um=2.0*math.sqrt(area_px/math.pi)*px_um
            radius=max(4, int(round(math.sqrt(area_px/math.pi))))
            pen.ellipse((gx-radius,gy-radius,gx+radius,gy+radius), outline=(0,255,255), width=3)
            rows.append({
                'well_number':wi,'pdo_number_in_well':pi,
                'well_x_fullres_px':x0+int(wx),'well_y_fullres_px':y0+int(wy),
                'pdo_x_fullres_px':x0+gx,'pdo_y_fullres_px':y0+gy,
                'projected_area_px2':area_px,'equivalent_circular_diameter_um':deq_um,
            })

    marked.save(out/'central_PDO_QC_overlay.png')
    Image.fromarray(gfp).save(out/'central_GFP_scaled.png')
    Image.fromarray(dic).save(out/'central_DIC_normalized.png')
    with (out/'pdo_qc_objects.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['well_number'])
        writer.writeheader(); writer.writerows(rows)

    summary={
        'source':str(src),'shape_cyx':[c,h,w],'channel_labels':labels,
        'preview_origin_fullres_px':{'x':x0,'y':y0},'preview_size_px':tile,
        'pixel_size_um':{'x':px_x,'y':px_y},'expected_well_radius_px':expected_radius,
        'detected_wells':int(len(wells)),'detected_pdos':int(total_pdos),
        'pdo_positive_wells':int(len({r['well_number'] for r in rows})),
        'green_low_uint8':float(args.green_low),'green_high_uint8':float(args.green_high),
        'gfp_raw_min':int(graw.min()),'gfp_raw_max':int(graw.max()),
        'qc_overlay_legend':'magenta = detected microwell; cyan = automated PDO equivalent-area circle',
        'note':'Automated PDO calls are QC only until visually reviewed against GFP signal.'
    }
    (out/'pdo_qc_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'QC outputs: {out}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
