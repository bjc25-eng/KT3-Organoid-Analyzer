from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial import cKDTree


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


def _estimate_hex_angle(points: np.ndarray, pitch: float) -> float:
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=1.22*pitch, output_type='set')
    angles = []
    for i, j in pairs:
        v = points[j] - points[i]
        d = float(np.linalg.norm(v))
        if 0.78*pitch <= d <= 1.22*pitch:
            angles.append(math.atan2(float(v[1]), float(v[0])))
    if not angles:
        raise RuntimeError('Could not infer lattice orientation from neighbour vectors.')
    a = np.asarray(angles, dtype=float)
    theta = math.atan2(float(np.mean(np.sin(6*a))), float(np.mean(np.cos(6*a)))) / 6.0
    return theta


def _fit_affine_lattice(points: np.ndarray, pitch: float, theta: float):
    b1 = pitch*np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
    b2 = pitch*np.asarray([math.cos(theta+math.pi/3), math.sin(theta+math.pi/3)], dtype=float)
    origin = np.median(points, axis=0)
    keep = np.ones(len(points), dtype=bool)
    uv_int = None
    for _ in range(5):
        B = np.column_stack([b1, b2])
        uv = (np.linalg.inv(B) @ (points-origin).T).T
        uv_int = np.rint(uv).astype(int)
        X = np.column_stack([np.ones(np.sum(keep)), uv_int[keep,0], uv_int[keep,1]])
        coef_x, *_ = np.linalg.lstsq(X, points[keep,0], rcond=None)
        coef_y, *_ = np.linalg.lstsq(X, points[keep,1], rcond=None)
        origin = np.asarray([coef_x[0], coef_y[0]])
        b1 = np.asarray([coef_x[1], coef_y[1]])
        b2 = np.asarray([coef_x[2], coef_y[2]])
        pred = origin + uv_int[:,0,None]*b1 + uv_int[:,1,None]*b2
        resid = np.linalg.norm(points-pred, axis=1)
        new_keep = resid <= 0.32*pitch
        if np.array_equal(new_keep, keep):
            break
        if np.sum(new_keep) < max(50, 0.5*len(points)):
            break
        keep = new_keep
    pred = origin + uv_int[:,0,None]*b1 + uv_int[:,1,None]*b2
    resid = np.linalg.norm(points-pred, axis=1)
    return origin, b1, b2, uv_int, resid, keep


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
    bins = np.linspace(np.percentile(l,1), np.percentile(l,99), 45)
    lefts=[]; rights=[]
    for a,b in zip(bins[:-1], bins[1:]):
        q=s[(l>=a)&(l<b)]
        if len(q) >= 20:
            lefts.append(float(np.min(q))); rights.append(float(np.max(q)))
    if len(lefts) < 8:
        raise RuntimeError('Could not estimate stable straight array side boundaries.')
    s_lo=float(np.percentile(lefts,10))
    s_hi=float(np.percentile(rights,90))
    l_lo=float(np.percentile(l,0.5))
    l_hi=float(np.percentile(l,99.5))
    # Half-pitch tolerance includes boundary-centred wells without reaching into
    # the optional large capture-well zones.
    margin=0.45*pitch
    return centre,long_axis,short_axis,l_lo-margin,l_hi+margin,s_lo-margin,s_hi+margin


def _inside_rect(p: np.ndarray, centre, long_axis, short_axis, llo,lhi,slo,shi):
    d=p-centre
    l=float(d@long_axis); s=float(d@short_axis)
    return llo<=l<=lhi and slo<=s<=shi, l, s


def _u8_local(arr: np.ndarray):
    a=np.asarray(arr,dtype=np.float32); finite=a[np.isfinite(a)]
    if finite.size==0: return np.zeros(a.shape,dtype=np.uint8)
    lo=float(np.percentile(finite,0.5)); hi=float(np.percentile(finite,99.5))
    if hi<=lo: return np.zeros(a.shape,dtype=np.uint8)
    return np.clip((a-lo)*(255.0/(hi-lo)),0,255).astype(np.uint8)


def main() -> int:
    ap=argparse.ArgumentParser(description='Fit the full regular hexagonal microwell lattice, reconstruct straight array edges, and QC candidates before changing final counts.')
    ap.add_argument('source', type=Path)
    ap.add_argument('refined_results_dir', type=Path)
    ap.add_argument('hex_results_dir', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('hex_lattice_reconstruction'))
    ap.add_argument('--dic-channel', type=int, default=1)
    ap.add_argument('--max-overview-height', type=int, default=4200)
    args=ap.parse_args()

    source=args.source.expanduser().resolve(); refined=args.refined_results_dir.expanduser().resolve(); hexdir=args.hex_results_dir.expanduser().resolve(); out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    seed_rows=_read_csv(hexdir/'hex_array_well_measurements.csv')
    all_rows=_read_csv(refined/'well_measurements.csv')
    excluded_rows=_read_csv(hexdir/'excluded_nonarray_wells.csv')
    pitch=_load_pitch(hexdir)

    seed=np.asarray([(float(r['x_px_fullres']),float(r['y_px_fullres'])) for r in seed_rows],dtype=float)
    allpts=np.asarray([(float(r['x_px_fullres']),float(r['y_px_fullres'])) for r in all_rows],dtype=float)
    seed_ids={int(float(r['well_id'])) for r in seed_rows}
    all_ids=np.asarray([int(float(r['well_id'])) for r in all_rows],dtype=int)

    theta=_estimate_hex_angle(seed,pitch)
    origin,b1,b2,uv_seed,resid,fit_keep=_fit_affine_lattice(seed,pitch,theta)
    inlier_seed=seed[fit_keep]
    centre,long_axis,short_axis,llo,lhi,slo,shi=_robust_bounds(inlier_seed,pitch)

    umin=int(np.min(uv_seed[:,0]))-3; umax=int(np.max(uv_seed[:,0]))+3
    vmin=int(np.min(uv_seed[:,1]))-3; vmax=int(np.max(uv_seed[:,1]))+3
    tree_all=cKDTree(allpts)
    match_tol=0.34*pitch

    predicted=[]; recovered=[]; missing=[]; observed_seed=[]
    seen_matched_ids=set()
    for u in range(umin,umax+1):
        for v in range(vmin,vmax+1):
            p=origin+u*b1+v*b2
            inside,lcoord,scoord=_inside_rect(p,centre,long_axis,short_axis,llo,lhi,slo,shi)
            if not inside:
                continue
            dist,idx=tree_all.query(p,k=1)
            matched_id=int(all_ids[int(idx)]) if np.isfinite(dist) and dist<=match_tol else None
            row={'grid_u':u,'grid_v':v,'predicted_x_px_fullres':float(p[0]),'predicted_y_px_fullres':float(p[1]),'long_coord_px':lcoord,'short_coord_px':scoord,'nearest_detected_distance_px':float(dist) if np.isfinite(dist) else None,'matched_well_id':matched_id}
            if matched_id is not None:
                seen_matched_ids.add(matched_id)
                if matched_id in seed_ids:
                    row['status']='observed_main_component'; observed_seed.append(row)
                else:
                    row['status']='recovered_detected_edge_candidate'; recovered.append(row)
            else:
                row['status']='predicted_missing_site'; missing.append(row)
            predicted.append(row)

    # Detected wells that remain outside the reconstructed straight array ROI.
    outside=[]
    for r,p,wid in zip(all_rows,allpts,all_ids):
        inside,lcoord,scoord=_inside_rect(p,centre,long_axis,short_axis,llo,lhi,slo,shi)
        if not inside:
            q=dict(r); q['long_coord_px']=lcoord; q['short_coord_px']=scoord; outside.append(q)

    fields=['grid_u','grid_v','predicted_x_px_fullres','predicted_y_px_fullres','long_coord_px','short_coord_px','nearest_detected_distance_px','matched_well_id','status']
    _write_csv(out/'reconstructed_grid_sites.csv',predicted,fields)
    _write_csv(out/'recovered_detected_edge_candidates.csv',recovered,fields)
    _write_csv(out/'predicted_missing_sites.csv',missing,fields)
    if outside:
        _write_csv(out/'detected_candidates_outside_reconstructed_array.csv',outside,list(outside[0].keys()))

    root=zarr.open_group(str(source),mode='r'); arr=root['0']; _,h,w=map(int,arr.shape)
    step=max(1,int(math.ceil(h/max(500,int(args.max_overview_height)))))
    dic=_u8_local(np.asarray(arr[int(args.dic_channel),::step,::step]))

    fig_h=14; fig_w=max(3.8,fig_h*(w/h))
    plt.figure(figsize=(fig_w,fig_h))
    plt.imshow(dic,cmap='gray',origin='upper',extent=[0,w,h,0],interpolation='nearest')
    if outside:
        op=np.asarray([(float(r['x_px_fullres']),float(r['y_px_fullres'])) for r in outside]); plt.scatter(op[:,0],op[:,1],s=4,c='deepskyblue',alpha=0.45,label='Detected outside reconstructed array')
    plt.scatter(seed[:,0],seed[:,1],s=2.5,c='orange',alpha=0.35,label='Current main-component wells')
    if recovered:
        rp=np.asarray([(r['predicted_x_px_fullres'],r['predicted_y_px_fullres']) for r in recovered]); plt.scatter(rp[:,0],rp[:,1],s=13,facecolors='none',edgecolors='magenta',linewidths=0.8,label='Recovered detected edge candidates')
    if missing:
        mp=np.asarray([(r['predicted_x_px_fullres'],r['predicted_y_px_fullres']) for r in missing]); plt.scatter(mp[:,0],mp[:,1],s=10,marker='+',c='lime',alpha=0.7,label='Predicted sites with no Hough detection')

    corners=[]
    for ll,ss in [(llo,slo),(llo,shi),(lhi,shi),(lhi,slo),(llo,slo)]:
        corners.append(centre+ll*long_axis+ss*short_axis)
    cp=np.asarray(corners); plt.plot(cp[:,0],cp[:,1],c='red',linewidth=1.1,label='Reconstructed straight array boundary')
    plt.xlim(0,w); plt.ylim(h,0); plt.xlabel('Full-resolution x (px)'); plt.ylabel('Full-resolution y (px)'); plt.title('Hexagonal lattice reconstruction QC'); plt.legend(loc='best',fontsize=6,markerscale=1.5); plt.tight_layout(); plt.savefig(out/'hex_lattice_reconstruction_overlay.png',dpi=220); plt.close()

    summary={
        'pitch_input_px':pitch,
        'fitted_basis_vector_1_px':[float(x) for x in b1],
        'fitted_basis_vector_2_px':[float(x) for x in b2],
        'fitted_basis_lengths_px':[float(np.linalg.norm(b1)),float(np.linalg.norm(b2))],
        'fitted_basis_angle_deg':float(np.degrees(np.arccos(np.clip(np.dot(b1,b2)/(np.linalg.norm(b1)*np.linalg.norm(b2)),-1,1)))),
        'seed_main_component_wells':len(seed_rows),
        'lattice_fit_inliers':int(np.sum(fit_keep)),
        'median_seed_fit_residual_px':float(np.median(resid[fit_keep])),
        'reconstructed_sites_inside_straight_boundary':len(predicted),
        'observed_main_component_sites_matched':len(observed_seed),
        'recovered_detected_edge_candidates':len(recovered),
        'predicted_missing_sites_without_hough_detection':len(missing),
        'detected_candidates_outside_reconstructed_array':len(outside),
        'qc_status':'QC only: recovered and predicted sites are not added to final biological counts until visually reviewed.',
        'overlay':'hex_lattice_reconstruction_overlay.png'
    }
    (out/'hex_lattice_reconstruction_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
