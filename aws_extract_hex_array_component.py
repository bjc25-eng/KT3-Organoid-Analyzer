from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)


def _load_pitch(results_dir: Path) -> float:
    p = results_dir / 'refined_summary.json'
    if not p.exists():
        raise RuntimeError(f'Missing {p}')
    d = json.loads(p.read_text(encoding='utf-8'))
    pitch = float(d['pitch_px'])
    if not np.isfinite(pitch) or pitch <= 0:
        raise RuntimeError(f'Invalid pitch_px in {p}: {pitch}')
    return pitch


def _components(points: np.ndarray, pitch: float) -> tuple[list[list[int]], list[int]]:
    low = 0.78 * pitch
    high = 1.22 * pitch
    tree = cKDTree(points)
    adjacency = [[] for _ in range(len(points))]
    pairs = tree.query_pairs(r=high, output_type='set')
    for i, j in pairs:
        d = float(np.linalg.norm(points[i] - points[j]))
        if low <= d <= high:
            adjacency[i].append(j)
            adjacency[j].append(i)

    seen = np.zeros(len(points), dtype=bool)
    comps: list[list[int]] = []
    degrees = [len(a) for a in adjacency]
    for start in range(len(points)):
        if seen[start]:
            continue
        q = deque([start]); seen[start] = True; comp = []
        while q:
            i = q.popleft(); comp.append(i)
            for j in adjacency[i]:
                if not seen[j]:
                    seen[j] = True; q.append(j)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps, degrees


def main() -> int:
    ap = argparse.ArgumentParser(description='Keep only the dominant connected 100-um hexagonal microwell lattice, excluding large capture-well zones and other disconnected regions.')
    ap.add_argument('refined_results_dir', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('hex_array_only'))
    args = ap.parse_args()

    src = args.refined_results_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)

    wells = _read_csv(src / 'well_measurements.csv')
    pdos = _read_csv(src / 'pdo_measurements.csv')
    pitch = _load_pitch(src)

    if not wells:
        raise RuntimeError('No lattice-accepted wells found.')

    pts = np.asarray([(float(r['x_px_fullres']), float(r['y_px_fullres'])) for r in wells], dtype=float)
    comps, degrees = _components(pts, pitch)
    if not comps:
        raise RuntimeError('No connected components found.')

    main_idx = set(comps[0])
    main_wells = []
    excluded_wells = []
    for i, row in enumerate(wells):
        q = dict(row)
        q['lattice_degree'] = degrees[i]
        q['hex_array_member'] = i in main_idx
        (main_wells if i in main_idx else excluded_wells).append(q)

    main_ids = {int(float(r['well_id'])) for r in main_wells}
    main_pdos = [r for r in pdos if int(float(r['well_id'])) in main_ids]
    excluded_pdos = [r for r in pdos if int(float(r['well_id'])) not in main_ids]

    wf = list(main_wells[0].keys())
    _write_csv(out / 'hex_array_well_measurements.csv', main_wells, wf)
    _write_csv(out / 'excluded_nonarray_wells.csv', excluded_wells, wf)
    if pdos:
        pf = list(pdos[0].keys())
        _write_csv(out / 'hex_array_pdo_measurements.csv', main_pdos, pf)
        _write_csv(out / 'excluded_nonarray_pdo_measurements.csv', excluded_pdos, pf)
    else:
        _write_csv(out / 'hex_array_pdo_measurements.csv', [], ['well_id'])
        _write_csv(out / 'excluded_nonarray_pdo_measurements.csv', [], ['well_id'])

    main_pts = pts[list(main_idx)]
    bounds = {
        'x_min_px': float(np.min(main_pts[:,0])), 'x_max_px': float(np.max(main_pts[:,0])),
        'y_min_px': float(np.min(main_pts[:,1])), 'y_max_px': float(np.max(main_pts[:,1])),
    }

    component_sizes = [len(c) for c in comps[:20]]
    summary = {
        'pitch_px': pitch,
        'input_lattice_accepted_wells': len(wells),
        'connected_component_count': len(comps),
        'largest_component_wells': len(main_wells),
        'excluded_nonarray_wells': len(excluded_wells),
        'largest_component_fraction': len(main_wells) / len(wells),
        'largest_component_bounds_fullres_px': bounds,
        'pdo_positive_wells_in_hex_array': sum(str(r.get('PDO_present','')).lower() in {'true','1'} for r in main_wells),
        'pdo_objects_in_hex_array': len(main_pdos),
        'pdo_objects_excluded_outside_hex_array': len(excluded_pdos),
        'largest_component_sizes_top20': component_sizes,
        'selection_rule': 'Keep only the largest connected component under neighbour distances 0.78-1.22 times the inferred 100-um-well lattice pitch.',
        'scientific_scope': 'Large circular bead-capture wells and disconnected top/bottom device regions are excluded from the 100-um hexagonal microwell analysis.'
    }
    (out / 'hex_array_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # Coordinate-only QC map; does not load the microscopy image.
    plt.figure(figsize=(6, 12))
    if excluded_wells:
        ep = np.asarray([(float(r['x_px_fullres']), float(r['y_px_fullres'])) for r in excluded_wells])
        plt.scatter(ep[:,0], ep[:,1], s=2, alpha=0.35, label='Excluded non-array candidates')
    plt.scatter(main_pts[:,0], main_pts[:,1], s=2, alpha=0.5, label='Dominant hexagonal array')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Full-resolution x (px)')
    plt.ylabel('Full-resolution y (px)')
    plt.title('Dominant hexagonal microwell array component')
    plt.legend(loc='best', markerscale=4)
    plt.tight_layout()
    plt.savefig(out / 'hex_array_component_map.png', dpi=220)
    plt.close()

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
