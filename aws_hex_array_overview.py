from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


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


def main() -> int:
    ap = argparse.ArgumentParser(description='Overlay accepted hex-array wells and excluded candidates on a downsampled whole-image DIC overview.')
    ap.add_argument('source', type=Path)
    ap.add_argument('hex_results_dir', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('hex_array_overview'))
    ap.add_argument('--dic-channel', type=int, default=1)
    ap.add_argument('--max-overview-height', type=int, default=4200)
    args = ap.parse_args()

    source = args.source.expanduser().resolve()
    results = args.hex_results_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)

    accepted = _read_csv(results / 'hex_array_well_measurements.csv')
    excluded = _read_csv(results / 'excluded_nonarray_wells.csv')

    root = zarr.open_group(str(source), mode='r')
    arr = root['0']
    if arr.ndim != 3:
        raise RuntimeError(f'Expected C,Y,X OME-Zarr; got {arr.shape}')
    _, h, w = map(int, arr.shape)

    step = max(1, int(math.ceil(h / max(500, int(args.max_overview_height)))))
    dic_raw = np.asarray(arr[int(args.dic_channel), ::step, ::step])
    dic = _u8_local(dic_raw)

    fig_h = 14
    fig_w = max(3.5, fig_h * (w / h))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(dic, cmap='gray', origin='upper', extent=[0, w, h, 0], interpolation='nearest')

    if excluded:
        ep = np.asarray([(float(r['x_px_fullres']), float(r['y_px_fullres'])) for r in excluded], dtype=float)
        plt.scatter(ep[:,0], ep[:,1], s=4, alpha=0.55, c='deepskyblue', label='Excluded / outside dominant hex array')
    if accepted:
        apoints = np.asarray([(float(r['x_px_fullres']), float(r['y_px_fullres'])) for r in accepted], dtype=float)
        plt.scatter(apoints[:,0], apoints[:,1], s=3, alpha=0.55, c='orange', label='Included 100 µm hex-array wells')

    plt.xlim(0, w); plt.ylim(h, 0)
    plt.xlabel('Full-resolution x (px)')
    plt.ylabel('Full-resolution y (px)')
    plt.title('Whole-image DIC QC: included hexagonal array vs excluded regions')
    plt.legend(loc='best', fontsize=7, markerscale=2)
    plt.tight_layout()
    out_png = out / 'whole_image_DIC_hex_array_overlay.png'
    plt.savefig(out_png, dpi=220)
    plt.close()

    summary = {
        'source_shape_cyx': [int(v) for v in arr.shape],
        'overview_downsample_step': step,
        'included_hex_array_wells': len(accepted),
        'excluded_candidates': len(excluded),
        'output': out_png.name,
        'qc_question': 'Orange points should fall only on the regular 100 µm hexagonal microwell array. Large circular bead-capture wells and other non-analysis regions should remain blue/excluded.'
    }
    (out / 'whole_image_DIC_overlay_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
