from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import zarr
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from analysis_core import Settings, segment_pdos
from aws_full_array_benchmark import (
    _channel_labels,
    _detect_wells_tile,
    _dedupe,
    _read_scale,
    _tiles,
    _u8_absolute,
    _u8_local,
    _window_end,
)

OME_ZARR_RE = re.compile(r'(.+?\.ome\.zarr)(?:/|$)', re.I)


def list_s3_omezarr_datasets(client, bucket: str, prefix: str = '') -> list[str]:
    roots: set[str] = set()
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get('Contents', []):
            key = item.get('Key', '')
            m = OME_ZARR_RE.search(key)
            if m:
                roots.add(m.group(1).rstrip('/') + '/')
    return sorted(roots)


def _open_s3_group(client, bucket: str, prefix: str, region: str):
    creds = client._request_signer._credentials
    if hasattr(creds, 'get_frozen_credentials'):
        creds = creds.get_frozen_credentials()
    fs = s3fs.S3FileSystem(
        key=getattr(creds, 'access_key', None),
        secret=getattr(creds, 'secret_key', None),
        token=getattr(creds, 'token', None),
        client_kwargs={'region_name': region},
    )
    store = s3fs.S3Map(root=f"{bucket}/{prefix.rstrip('/')}", s3=fs, check=False)
    return zarr.open_group(store=store, mode='r')


def _largest_hex_component(wells: list[tuple[int, int, int]]) -> tuple[list[tuple[int, int, int]], float]:
    if len(wells) < 2:
        return wells, float('nan')
    pts = np.asarray([(x, y) for x, y, _ in wells], dtype=float)
    tree = cKDTree(pts)
    distances, _ = tree.query(pts, k=min(7, len(pts)))
    nn = distances[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if not len(nn):
        return wells, float('nan')
    pitch = float(np.median(nn))
    low, high = 0.78 * pitch, 1.22 * pitch
    pairs = tree.query_pairs(r=high, output_type='set')
    adjacency = [[] for _ in range(len(wells))]
    for i, j in pairs:
        d = float(np.linalg.norm(pts[i] - pts[j]))
        if low <= d <= high:
            adjacency[i].append(j)
            adjacency[j].append(i)
    seen = np.zeros(len(wells), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(wells)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in adjacency[i]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        components.append(comp)
    components.sort(key=len, reverse=True)
    keep = components[0] if components else list(range(len(wells)))
    return [wells[i] for i in keep], pitch


def _assign_indices(wdf: pd.DataFrame, pitch: float) -> pd.DataFrame:
    if wdf.empty:
        return wdf
    out = wdf.copy()
    tol = max(8.0, 0.35 * pitch) if np.isfinite(pitch) else 20.0
    ys = sorted(float(v) for v in out['well_centre_y_px'])
    groups = [[ys[0]]]
    for y in ys[1:]:
        if abs(y - float(np.mean(groups[-1]))) <= tol:
            groups[-1].append(y)
        else:
            groups.append([y])
    centres = np.asarray([float(np.mean(g)) for g in groups])
    out['well_row_index'] = [int(np.argmin(abs(centres - float(y)))) + 1 for y in out['well_centre_y_px']]
    out['well_col_index'] = 0
    for _, ids in out.groupby('well_row_index').groups.items():
        for col, idx in enumerate(out.loc[list(ids)].sort_values('well_centre_x_px').index, 1):
            out.at[idx, 'well_col_index'] = col
    out['well_index'] = [f'{int(c)},{int(r)}' for c, r in zip(out['well_col_index'], out['well_row_index'])]
    return out


def _composite(dic_raw, gfp_raw, gfp_max: float) -> Image.Image:
    dic = _u8_local(dic_raw)
    gfp = _u8_absolute(gfp_raw, gfp_max)
    rgb = np.stack([dic, dic, dic], axis=-1)
    rgb[..., 1] = np.maximum(rgb[..., 1], gfp)
    return Image.fromarray(rgb)


def _font(size=16, bold=False):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _labelled_crop(crop: Image.Image, row: pd.Series) -> Image.Image:
    lines = [
        f"Well {row['well_index']} | PDO {int(row['pdo_number_in_well'])}/{int(row['pdo_count_in_well'])}",
        f"Equivalent circular diameter: {float(row['equivalent_circular_diameter_um']):.1f} µm",
    ]
    header = 62
    out = Image.new('RGB', (crop.width, crop.height + header), 'white')
    out.paste(crop, (0, header))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, out.width, header), fill='black')
    d.text((8, 7), lines[0], fill='white', font=_font(16, True))
    d.text((8, 34), lines[1], fill='white', font=_font(14, False))
    return out


def _contact_sheet(paths: list[Path], out: Path, cols: int = 5, max_items: int = 200):
    paths = paths[:max_items]
    if not paths:
        return
    ims = [Image.open(p).convert('RGB') for p in paths]
    tw = 260
    thumbs = []
    for im in ims:
        s = tw / im.width
        thumbs.append(im.resize((tw, int(round(im.height*s))), Image.Resampling.LANCZOS))
    h = max(im.height for im in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    gap = 6
    sheet = Image.new('RGB', (cols*tw+(cols+1)*gap, rows*h+(rows+1)*gap), 'white')
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(im, (gap+c*(tw+gap), gap+r*(h+gap)))
    sheet.save(out, dpi=(300, 300))


def process_s3_omezarr(
    client,
    bucket: str,
    dataset_prefix: str,
    settings: Settings,
    region: str = 'eu-west-2',
    cols: int = 5,
    tile_size: int = 4096,
    gfp_channel: int = 0,
    dic_channel: int = 1,
    crop_size_px: int = 256,
    create_pdo_centred: bool = True,
):
    root = _open_s3_group(client, bucket, dataset_prefix, region)
    if '0' not in root:
        raise RuntimeError('OME-Zarr dataset does not contain a level-0 array named 0.')
    arr = root['0']
    if arr.ndim != 3:
        raise RuntimeError(f'Expected C,Y,X OME-Zarr, got {arr.shape}.')
    c, height, width = map(int, arr.shape)
    if max(gfp_channel, dic_channel) >= c:
        raise RuntimeError(f'Dataset has {c} channels; selected indices are not available.')

    px_x, px_y = _read_scale(root)
    px_um = (px_x + px_y) / 2.0
    if not np.isfinite(px_um) or px_um <= 0:
        raise RuntimeError('OME-Zarr metadata does not contain a valid pixel size.')
    expected_radius = float(settings.well_diameter_um) / (2.0 * px_um)
    tile = max(1024, int(tile_size))
    overlap = int(math.ceil(expected_radius * 3.0))
    tiles = list(_tiles(width, height, tile))

    work = Path(tempfile.mkdtemp(prefix='kt3_omezarr_'))
    out = work/'results'
    for d in [out/'csv', out/'figures', out/'pdo_centred_raw_crops', out/'pdo_centred_labelled_crops', out/'indexed_large_images']:
        d.mkdir(parents=True, exist_ok=True)

    raw_wells = []
    for cx0, cy0, cw, ch in tiles:
        x0=max(0,cx0-overlap); y0=max(0,cy0-overlap)
        x1=min(width,cx0+cw+overlap); y1=min(height,cy0+ch+overlap)
        dic = _u8_local(np.asarray(arr[dic_channel, y0:y1, x0:x1]))
        local = _detect_wells_tile(dic, expected_radius, float(settings.hough_p2))
        for lx, ly, r in local:
            gx, gy = int(x0+lx), int(y0+ly)
            if cx0 <= gx < cx0+cw and cy0 <= gy < cy0+ch:
                if gx-r >= 2 and gx+r < width-2 and gy-r >= 2 and gy+r < height-2:
                    raw_wells.append((gx, gy, int(r)))
    raw_wells = _dedupe(raw_wells, max(12.0, 0.30*expected_radius))
    wells, pitch = _largest_hex_component(raw_wells)
    if not wells:
        raise RuntimeError('No dominant microwell array was detected.')

    tile_cols = math.ceil(width/tile)
    tile_rows = math.ceil(height/tile)
    by_tile: dict[int, list[tuple[int,int,int,int]]] = {}
    for wid, (x, y, r) in enumerate(wells, 1):
        tx=min(tile_cols-1,x//tile); ty=min(tile_rows-1,y//tile)
        by_tile.setdefault(int(ty*tile_cols+tx), []).append((wid,x,y,r))

    gfp_max = _window_end(root, gfp_channel, arr.dtype)
    s = replace(settings, rfp_psc_present=False)
    well_rows, pdo_rows = [], []
    for ti, (cx0, cy0, cw, ch) in enumerate(tiles):
        here = by_tile.get(ti, [])
        if not here:
            continue
        x0=max(0,cx0-overlap); y0=max(0,cy0-overlap)
        x1=min(width,cx0+cw+overlap); y1=min(height,cy0+ch+overlap)
        gfp = _u8_absolute(np.asarray(arr[gfp_channel, y0:y1, x0:x1]), gfp_max)
        for wid, wx, wy, wr in here:
            lx, ly = wx-x0, wy-y0
            cr=int(math.ceil(wr*0.95))
            xa=max(0,lx-cr); xb=min(gfp.shape[1],lx+cr+1)
            ya=max(0,ly-cr); yb=min(gfp.shape[0],ly+cr+1)
            sub=gfp[ya:yb,xa:xb].astype(np.float32)
            yy,xx=np.ogrid[:sub.shape[0],:sub.shape[1]]
            scx,scy=lx-xa,ly-ya
            mask=(xx-scx)**2+(yy-scy)**2 <= (0.86*wr)**2
            signal=gaussian_filter(np.where(mask,sub,0.0),0.8)
            objs=segment_pdos(signal,s)
            kept=[o for o in objs if (float(o['x'])-scx)**2+(float(o['y'])-scy)**2 <= (0.86*wr)**2]
            well_rows.append({
                'well_id':wid,'well_centre_x_px':wx,'well_centre_y_px':wy,'well_radius_px':wr,
                'um_per_pixel':px_um,'PDO_count':len(kept),'PSC_like_focus_count':np.nan,
                'qc_status':'automated_dominant_hex_array_not_manually_reviewed',
            })
            for n,o in enumerate(kept,1):
                area=float(o['area'])
                pdo_rows.append({
                    'well_id':wid,'pdo_number_in_well':n,'pdo_count_in_well':len(kept),
                    'centroid_x_px':x0+xa+float(o['x']),'centroid_y_px':y0+ya+float(o['y']),
                    'projected_area_px2':area,'projected_area_um2':area*(px_um**2),
                    'equivalent_circular_diameter_um':2*math.sqrt(area/math.pi)*px_um,
                    'PSC_like_focus_count_in_well':np.nan,
                    'qc_status':'automated_dominant_hex_array_not_manually_reviewed',
                })

    wdf=_assign_indices(pd.DataFrame(well_rows),pitch)
    pdf=pd.DataFrame(pdo_rows)
    if not pdf.empty:
        pdf=pdf.merge(wdf[['well_id','well_index','well_col_index','well_row_index']],on='well_id',how='left')
        pdf['PDO_number_in_well']=pdf['pdo_number_in_well']
        pdf['PDO_count_in_well']=pdf['pdo_count_in_well']

    labelled=[]
    if create_pdo_centred and not pdf.empty:
        half=int(crop_size_px)//2
        for _,row in pdf.iterrows():
            cx,cy=float(row.centroid_x_px),float(row.centroid_y_px)
            x0=max(0,int(round(cx))-half); y0=max(0,int(round(cy))-half)
            x1=min(width,x0+int(crop_size_px)); y1=min(height,y0+int(crop_size_px))
            x0=max(0,x1-int(crop_size_px)); y0=max(0,y1-int(crop_size_px))
            crop=_composite(np.asarray(arr[dic_channel,y0:y1,x0:x1]),np.asarray(arr[gfp_channel,y0:y1,x0:x1]),gfp_max)
            if crop.size!=(int(crop_size_px),int(crop_size_px)):
                canvas=Image.new('RGB',(int(crop_size_px),int(crop_size_px)),'black'); canvas.paste(crop,(0,0)); crop=canvas
            stem=f"well_{str(row.well_index).replace(',','_')}_PDO_{int(row.pdo_number_in_well):02d}"
            rp=out/'pdo_centred_raw_crops'/f'{stem}.png'; lp=out/'pdo_centred_labelled_crops'/f'{stem}_labelled.png'
            crop.save(rp,dpi=(300,300)); _labelled_crop(crop,row).save(lp,dpi=(300,300)); labelled.append(lp)
        _contact_sheet(labelled,out/'figures'/'PDO_centred_contact_sheet_compact.png',int(cols))

    wdf.to_csv(out/'csv'/'well_raw_data.csv',index=False)
    pdf.to_csv(out/'csv'/'PDO_raw_data.csv',index=False)
    pdf.to_csv(out/'csv'/'PDO_centred_raw_data.csv',index=False)

    if not pdf.empty:
        fig,ax=plt.subplots(figsize=(6.2,4.6)); ax.hist(pdf.equivalent_circular_diameter_um,bins=int(settings.histogram_bins),edgecolor='black')
        ax.set(xlabel='PDO equivalent circular diameter (µm)',ylabel='Number of PDOs'); fig.tight_layout(); fig.savefig(out/'figures'/'PDO_size_distribution.png',dpi=300); plt.close(fig)
    freq=wdf.PDO_count.value_counts().sort_index().rename_axis('PDO_count').reset_index(name='well_count')
    freq.to_csv(out/'csv'/'PDO_count_frequency_across_wells.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.2,4.6)); ax.bar(freq.PDO_count.astype(str),freq.well_count,edgecolor='black')
    ax.set(xlabel='PDO count per accepted microwell',ylabel='Number of microwells'); fig.tight_layout(); fig.savefig(out/'figures'/'PDO_count_per_well_distribution.png',dpi=300); plt.close(fig)

    summary=pd.DataFrame([{
        'source_dataset':dataset_prefix,'images_processed':1,'fully_visible_wells':len(wdf),
        'PDO_containing_wells':int((wdf.PDO_count>0).sum()),'PDO_count':len(pdf),
        'mean_PDO_diameter_um':float(pdf.equivalent_circular_diameter_um.mean()) if len(pdf) else np.nan,
        'median_PDO_diameter_um':float(pdf.equivalent_circular_diameter_um.median()) if len(pdf) else np.nan,
        'SD_PDO_diameter_um':float(pdf.equivalent_circular_diameter_um.std(ddof=1)) if len(pdf)>1 else np.nan,
        'pixel_size_um':px_um,'inferred_hex_pitch_px':pitch,'raw_well_candidates':len(raw_wells),
        'dominant_hex_array_wells':len(wdf),'channel_labels':'; '.join(_channel_labels(root,c)),
        'GFP_channel_index':gfp_channel,'DIC_channel_index':dic_channel,
        'qc_status':'automated_dominant_hex_array_not_manually_reviewed',
    }])
    summary.to_csv(out/'csv'/'overall_summary.csv',index=False)
    idf=pd.DataFrame([{'image_series':1,'source_image':dataset_prefix,'fully_visible_wells':len(wdf),'PDO_containing_wells':int((wdf.PDO_count>0).sum()),'PDO_count':len(pdf),'um_per_pixel':px_um}])
    idf.to_csv(out/'csv'/'image_summary.csv',index=False)
    (out/'analysis_metadata.json').write_text(json.dumps({
        'bucket':bucket,'dataset_prefix':dataset_prefix,'shape_cyx':[c,height,width],
        'pixel_size_um':px_um,'well_diameter_um':settings.well_diameter_um,'inferred_hex_pitch_px':pitch,
        'gfp_channel_index':gfp_channel,'dic_channel_index':dic_channel,
        'channel_labels':_channel_labels(root,c),
        'method_note':'Whole-array OME-Zarr streamed from S3. Dominant connected component at 0.78-1.22x inferred nearest-neighbour pitch retained. PSC/RFP foci are not analysed in this route.'
    },indent=2),encoding='utf-8')
    return work,out,summary,idf
