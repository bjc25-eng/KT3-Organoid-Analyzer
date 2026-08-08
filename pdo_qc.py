from __future__ import annotations

"""Conservative PDO split validation and microwell-membership QC.

This module is deliberately stricter than the original GFP segmentation in two
places:

1. Fluorescence intensity peaks alone can no longer create additional PDOs.
   A connected GFP component is split only when its *shape* supports multiple
   objects in the distance transform and the resulting pieces pass area and
   geometry checks.
2. Every GFP candidate is assigned to its nearest detected microwell exactly
   once and receives a microwell-membership QC classification. Clearly
   outside-wall objects are rejected; ambiguous wall-touching objects are
   retained by default but flagged for visual review.

The thresholds below were chosen conservatively for the KT3 microwell images.
They are QC rules, not biological ground truth, and the automated output still
requires visual review before thesis/publication use.
"""

import math
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

SHAPE_PEAK_MIN_DISTANCE_PX = 5
SHAPE_PEAK_THRESHOLD_FRACTION = 0.45
SHAPE_PEAK_THRESHOLD_MIN_PX = 2.5
SPLIT_MIN_PIECE_AREA_PX = 60
SPLIT_MAX_OBJECTS_PER_COMPONENT = 6
PEAK_SEPARATION_OVER_RADII_SUM_MIN = 0.75
TWO_PIECE_AREA_RATIO_MIN = 0.20
MULTI_PIECE_AREA_RATIO_MIN = 0.08

WALL_SECTOR_HALF_WIDTH_DEG = 20
WALL_BEFORE_FRACTION_THRESHOLD = 0.35
OUTSIDE_COMPONENT_INSIDE_FRACTION = 0.75
AMBIGUOUS_COMPONENT_INSIDE_FRACTION = 0.88
AMBIGUOUS_CENTROID_FRACTION = 0.65


def _setting(settings, name: str, default):
    return getattr(settings, name, default)


def _green_excess(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return gaussian_filter(g - np.maximum(r, b), 0.8)


def segment_pdos_conservative(green: np.ndarray, settings) -> list[dict]:
    """Segment GFP PDOs with shape-confirmed touching-object splitting."""
    labs = label(green > float(settings.green_low))
    min_area = max(1, int(settings.pdo_min_area))
    green_high = float(settings.green_high)
    split_enabled = bool(settings.split_pdos)

    peak_min_distance = max(2, int(_setting(settings, "pdo_shape_peak_min_distance", SHAPE_PEAK_MIN_DISTANCE_PX)))
    peak_fraction = float(_setting(settings, "pdo_shape_peak_threshold_fraction", SHAPE_PEAK_THRESHOLD_FRACTION))
    peak_floor = float(_setting(settings, "pdo_shape_peak_threshold_min", SHAPE_PEAK_THRESHOLD_MIN_PX))
    split_min_area = max(min_area, int(_setting(settings, "pdo_split_min_piece_area", SPLIT_MIN_PIECE_AREA_PX)))
    max_objects = max(2, int(_setting(settings, "pdo_split_max_objects_per_component", SPLIT_MAX_OBJECTS_PER_COMPONENT)))

    out: list[dict] = []

    def append_unsplit(reg, parent_id: int):
        cy, cx = reg.centroid
        out.append({
            "x": float(cx),
            "y": float(cy),
            "area": float(reg.area),
            "coords_yx": reg.coords.astype(np.int32, copy=True),
            "parent_component_id": int(parent_id),
            "split_method": "unsplit",
            "split_confidence": "single_component",
        })

    for parent_id, reg in enumerate(regionprops(labs, intensity_image=green), start=1):
        if reg.area < min_area or reg.intensity_max < green_high:
            continue
        if not split_enabled:
            append_unsplit(reg, parent_id)
            continue

        y0, x0, y1, x1 = reg.bbox
        mask = labs[y0:y1, x0:x1] == reg.label
        sub = green[y0:y1, x0:x1]
        dist = distance_transform_edt(mask)
        dmax = float(dist.max())
        if dmax <= 0:
            append_unsplit(reg, parent_id)
            continue

        peaks = peak_local_max(
            dist,
            min_distance=peak_min_distance,
            threshold_abs=max(peak_floor, peak_fraction * dmax),
            labels=mask.astype(np.uint8),
            exclude_border=False,
        )
        if len(peaks) < 2:
            append_unsplit(reg, parent_id)
            continue
        if len(peaks) > max_objects:
            strengths = np.asarray([dist[tuple(p)] for p in peaks], dtype=float)
            keep = np.argsort(strengths)[::-1][:max_objects]
            peaks = peaks[keep]

        peak_radii = np.asarray([float(dist[tuple(p)]) for p in peaks], dtype=float)
        geometry_ok = True
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                sep = float(np.linalg.norm(peaks[i].astype(float) - peaks[j].astype(float)))
                denom = peak_radii[i] + peak_radii[j]
                if denom <= 0 or sep / denom < PEAK_SEPARATION_OVER_RADII_SUM_MIN:
                    geometry_ok = False
                    break
            if not geometry_ok:
                break
        if not geometry_ok:
            append_unsplit(reg, parent_id)
            continue

        markers = np.zeros_like(dist, dtype=np.int32)
        for marker_id, (py, px) in enumerate(peaks, start=1):
            markers[int(py), int(px)] = marker_id
        ws = watershed(-dist, markers=markers, mask=mask)
        pieces = [
            p for p in regionprops(ws, intensity_image=sub)
            if p.area >= split_min_area and p.intensity_max >= green_high
        ]
        if len(pieces) < 2:
            append_unsplit(reg, parent_id)
            continue

        areas = sorted(float(p.area) for p in pieces)
        area_ratio = areas[0] / areas[-1] if areas[-1] else 0.0
        required_ratio = TWO_PIECE_AREA_RATIO_MIN if len(pieces) == 2 else MULTI_PIECE_AREA_RATIO_MIN
        if area_ratio < required_ratio:
            append_unsplit(reg, parent_id)
            continue

        for piece in pieces:
            cy, cx = piece.centroid
            coords = piece.coords.astype(np.int32, copy=True)
            coords[:, 0] += int(y0)
            coords[:, 1] += int(x0)
            out.append({
                "x": float(x0 + cx),
                "y": float(y0 + cy),
                "area": float(piece.area),
                "coords_yx": coords,
                "parent_component_id": int(parent_id),
                "split_method": "distance_transform_watershed",
                "split_confidence": "shape_supported",
            })
    return out


def _sample_nearest(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(xs).astype(int), 0, image.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, image.shape[0] - 1)
    return image[yi, xi]


def wall_before_centroid_fraction(
    rgb: np.ndarray,
    well_x: float,
    well_y: float,
    well_r: float,
    pdo_x: float,
    pdo_y: float,
    sector_half_width_deg: float = WALL_SECTOR_HALF_WIDTH_DEG,
) -> float:
    """Estimate whether a dark microwell wall lies between well centre and PDO."""
    dx, dy = float(pdo_x - well_x), float(pdo_y - well_y)
    centroid_distance = math.hypot(dx, dy)
    if well_r <= 0 or centroid_distance <= 0.40 * well_r:
        return 0.0

    rb = (rgb[..., 0].astype(np.float32) + rgb[..., 2].astype(np.float32)) / 2.0
    theta = math.atan2(dy, dx)
    ray_offsets = np.arange(-float(sector_half_width_deg), float(sector_half_width_deg) + 0.1, 4.0)
    band_offsets = np.deg2rad(np.asarray([-6.0, -3.0, 0.0, 3.0, 6.0]))
    radii = np.linspace(0.45 * well_r, 1.25 * well_r, 81)
    before = []

    for offset_deg in ray_offsets:
        base = theta + math.radians(float(offset_deg))
        radial_values = []
        for rr in radii:
            ang = base + band_offsets
            xs = well_x + rr * np.cos(ang)
            ys = well_y + rr * np.sin(ang)
            radial_values.append(float(np.median(_sample_nearest(rb, xs, ys))))
        profile = gaussian_filter1d(np.asarray(radial_values, dtype=float), 1.2)
        wall_r = float(radii[int(np.argmin(profile))])
        before.append(wall_r + max(1.0, 0.03 * well_r) < centroid_distance)

    return float(np.mean(before)) if before else 0.0


def assess_pdo_well_membership(
    rgb: np.ndarray,
    obj: dict,
    well_x: float,
    well_y: float,
    well_r: float,
    settings,
) -> dict:
    """Classify a segmented PDO candidate as accepted, ambiguous or outside."""
    dx = float(obj["x"]) - float(well_x)
    dy = float(obj["y"]) - float(well_y)
    centroid_distance = math.hypot(dx, dy)
    centroid_fraction = centroid_distance / float(well_r) if well_r else float("inf")

    coords = np.asarray(obj.get("coords_yx", []), dtype=float)
    if coords.size:
        inside = ((coords[:, 1] - float(well_x)) ** 2 + (coords[:, 0] - float(well_y)) ** 2) <= float(well_r) ** 2
        inside_fraction = float(np.mean(inside))
    else:
        inside_fraction = float("nan")

    wall_fraction = wall_before_centroid_fraction(
        rgb, float(well_x), float(well_y), float(well_r),
        float(obj["x"]), float(obj["y"]),
        float(_setting(settings, "wall_profile_sector_half_width_deg", WALL_SECTOR_HALF_WIDTH_DEG)),
    )

    outside_inside_threshold = float(_setting(settings, "outside_component_inside_fraction", OUTSIDE_COMPONENT_INSIDE_FRACTION))
    ambiguous_inside_threshold = float(_setting(settings, "ambiguous_component_inside_fraction", AMBIGUOUS_COMPONENT_INSIDE_FRACTION))
    ambiguous_centroid = float(_setting(settings, "ambiguous_centroid_fraction", AMBIGUOUS_CENTROID_FRACTION))
    wall_threshold = float(_setting(settings, "wall_before_fraction_threshold", WALL_BEFORE_FRACTION_THRESHOLD))

    hard_far_outside = centroid_fraction > 1.05 and inside_fraction < 0.50
    multi_sign_outside = (
        centroid_fraction > 0.60
        and inside_fraction < outside_inside_threshold
        and wall_fraction >= wall_threshold
    )

    if hard_far_outside or multi_sign_outside:
        status = "rejected_outside_well"
        reason = "centroid/shape/wall evidence indicates the GFP object is outside the microwell interior"
    elif (
        (centroid_fraction > ambiguous_centroid and inside_fraction < ambiguous_inside_threshold)
        or (centroid_fraction > 0.75 and wall_fraction >= 0.25)
    ):
        status = "ambiguous_wall_touching"
        reason = "edge/wall-touching candidate retained for visual QC"
    else:
        status = "accepted"
        reason = "candidate consistent with detected microwell interior"

    return {
        "membership_status": status,
        "membership_reason": reason,
        "centroid_distance_fraction_of_well_radius": float(centroid_fraction),
        "component_inside_detected_well_fraction": float(inside_fraction),
        "wall_before_centroid_fraction": float(wall_fraction),
    }


def _font(size=16, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _fixed_crop(rgb: np.ndarray, cx: float, cy: float, size: int = 220) -> Image.Image:
    image = Image.fromarray(rgb).convert("RGB")
    half = size // 2
    left = int(round(cx)) - half
    top = int(round(cy)) - half
    canvas = Image.new("RGB", (size, size), "black")
    sl, st = max(0, left), max(0, top)
    sr, sb = min(image.width, left + size), min(image.height, top + size)
    if sr > sl and sb > st:
        canvas.paste(image.crop((sl, st, sr, sb)), (sl - left, st - top))
    return canvas


def _qc_label(crop: Image.Image, status: str, series: int, well: str, reason: str) -> Image.Image:
    header = 78
    out = Image.new("RGB", (crop.width, crop.height + header), "white")
    out.paste(crop, (0, header))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, out.width, header), fill="black")
    d.text((8, 6), f"Image {series:02d} | Well {well}", fill="white", font=_font(16, True))
    d.text((8, 31), status.replace("_", " "), fill="white", font=_font(14, True))
    d.text((8, 53), reason[:42], fill="white", font=_font(11, False))
    return out


def _make_contact(paths: list[Path], out: Path, cols: int = 5, gap: int = 5):
    if not paths:
        return
    ims = [Image.open(p).convert("RGB") for p in paths]
    tw = min(620, max(i.width for i in ims))
    thumbs = []
    for im in ims:
        sc = tw / im.width
        thumbs.append(im.resize((tw, int(round(im.height * sc))), Image.Resampling.LANCZOS))
    ch = max(i.height for i in thumbs)
    rows = math.ceil(len(thumbs) / max(1, int(cols)))
    sheet = Image.new("RGB", (int(cols) * tw + (int(cols) + 1) * gap, rows * ch + (rows + 1) * gap), "white")
    for i, im in enumerate(thumbs):
        rr, cc = divmod(i, int(cols))
        sheet.paste(im, (gap + cc * (tw + gap), gap + rr * (ch + gap)))
    sheet.save(out, dpi=(300, 300))


def _rebuild_figures(out_dir: Path, pdf: pd.DataFrame, wdf: pd.DataFrame, settings):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for name in ["PDO_size_distribution.png", "PSC_count_frequency_across_PDOs.png", "PDO_count_per_well_distribution.png"]:
        p = fig_dir / name
        if p.exists():
            p.unlink()

    if not pdf.empty:
        d = pdf["equivalent_circular_diameter_um"].astype(float)
        mean = float(d.mean())
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        ax.hist(d, bins=int(settings.histogram_bins), edgecolor="black")
        ax.axvline(mean, ls="--", label=f"Mean = {mean:.1f} µm")
        ax.axvline(mean - sd, ls=":", label=f"±1 SD = {sd:.1f} µm")
        ax.axvline(mean + sd, ls=":")
        ax.set(xlabel="PDO equivalent circular diameter (µm)", ylabel="Number of PDOs")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "PDO_size_distribution.png", dpi=300)
        plt.close(fig)

        if bool(getattr(settings, "rfp_psc_present", False)):
            fr = (
                pdf["PSC_like_focus_count_in_well"].dropna().astype(int)
                .value_counts().sort_index().rename_axis("PSC_like_focus_count")
                .reset_index(name="PDO_count")
            )
            if not fr.empty:
                fr["percentage_of_PDOs"] = 100 * fr.PDO_count / len(pdf)
                fr.to_csv(out_dir / "csv" / "PSC_count_frequency_across_PDOs.csv", index=False)
                fig, ax = plt.subplots(figsize=(6.2, 4.6))
                bars = ax.bar(fr.PSC_like_focus_count, fr.PDO_count, edgecolor="black")
                for b, c, pct in zip(bars, fr.PDO_count, fr.percentage_of_PDOs):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                            f"{int(c)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
                ax.set(xlabel="PSC-like fluorescent foci in the same well", ylabel="Number of PDOs")
                ax.set_xticks(fr.PSC_like_focus_count)
                fig.tight_layout()
                fig.savefig(fig_dir / "PSC_count_frequency_across_PDOs.png", dpi=300)
                plt.close(fig)

    counts = wdf["PDO_count"].astype(int).value_counts().sort_index()
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        ax.bar(counts.index.astype(int), counts.values, edgecolor="black")
        ax.set(xlabel="PDOs per detected microwell", ylabel="Number of wells")
        ax.set_xticks(counts.index.astype(int))
        fig.tight_layout()
        fig.savefig(fig_dir / "PDO_count_per_well_distribution.png", dpi=300)
        plt.close(fig)


def _rebuild_well_crops(out_dir: Path, wdf: pd.DataFrame, pdf: pd.DataFrame, cols: int):
    from analysis_core import crop_square, labelled_crop

    raw_dir = out_dir / "raw_crops"
    lab_dir = out_dir / "labelled_crops"
    for d in [raw_dir, lab_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    labelled_paths: list[Path] = []
    image_cache: dict[tuple[int, str], np.ndarray] = {}
    for _, well_row in wdf[wdf["PDO_count"].astype(int) > 0].iterrows():
        series = int(well_row["image_series"])
        source = str(well_row["source_image"])
        key = (series, source)
        if key not in image_cache:
            raw_path = out_dir / "raw_images" / f"series_{series:02d}__{source}"
            image_cache[key] = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
        rgb = image_cache[key]
        subset = pdf[(pdf["image_series"] == series) & (pdf["well_index"].astype(str) == str(well_row["well_index"]))]
        sizes = subset["equivalent_circular_diameter_um"].astype(float).tolist()
        psc_n = well_row.get("PSC_like_focus_count", np.nan)
        crop = crop_square(rgb, int(well_row["well_centre_x_px"]), int(well_row["well_centre_y_px"]), int(well_row["well_radius_px"]))
        base = f"series_{series:02d}_well_{int(well_row['well_col_index'])}_{int(well_row['well_row_index'])}"
        rp = raw_dir / f"{base}.png"
        lp = lab_dir / f"{base}_labelled.png"
        crop.save(rp, dpi=(300, 300))
        labelled_crop(crop, series, str(well_row["well_index"]), len(sizes), psc_n, sizes).save(lp, dpi=(300, 300))
        labelled_paths.append(lp)

    contact = out_dir / "figures" / "PDO_well_contact_sheet_compact.png"
    if contact.exists():
        contact.unlink()
    if labelled_paths:
        _make_contact(labelled_paths, contact, cols=max(1, int(cols)), gap=5)


def rebuild_quantitative_outputs(out_dir: str | Path, settings, cols: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild GFP quantitative outputs using conservative split + well QC."""
    out_dir = Path(out_dir)
    csv_dir = out_dir / "csv"
    well_csv = csv_dir / "well_raw_data.csv"
    image_csv = csv_dir / "image_summary.csv"
    summary_csv = csv_dir / "overall_summary.csv"

    if not well_csv.exists():
        return pd.read_csv(summary_csv), pd.read_csv(image_csv)

    from analysis_core import GFP_MODE
    if getattr(settings, "organoid_mode", GFP_MODE) != GFP_MODE:
        return pd.read_csv(summary_csv), pd.read_csv(image_csv)

    wdf = pd.read_csv(well_csv)
    old_idf = pd.read_csv(image_csv) if image_csv.exists() else pd.DataFrame()
    if wdf.empty:
        return pd.read_csv(summary_csv), old_idf

    candidate_qc_rows: list[dict] = []
    pdo_rows: list[dict] = []
    accepted_by_well: dict[tuple[int, str], list[dict]] = {}
    ambiguous_by_well: dict[tuple[int, str], int] = {}
    rejected_by_well: dict[tuple[int, str], int] = {}

    qc_crop_dir = out_dir / "QC_rejected_or_ambiguous_candidates"
    if qc_crop_dir.exists():
        shutil.rmtree(qc_crop_dir)
    qc_crop_dir.mkdir(parents=True, exist_ok=True)
    qc_crop_paths: list[Path] = []

    exclude_ambiguous = bool(_setting(settings, "exclude_ambiguous_edge_candidates", False))

    for series, group in wdf.groupby("image_series", sort=True):
        series = int(series)
        source = str(group.iloc[0]["source_image"])
        raw_path = out_dir / "raw_images" / f"series_{series:02d}__{source}"
        rgb = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
        objects = segment_pdos_conservative(_green_excess(rgb), settings)

        centres = group[["well_centre_x_px", "well_centre_y_px"]].to_numpy(dtype=float)
        radii = group["well_radius_px"].to_numpy(dtype=float)
        group_rows = list(group.iterrows())

        for candidate_number, obj in enumerate(objects, start=1):
            p = np.asarray([float(obj["x"]), float(obj["y"])])
            distances = np.linalg.norm(centres - p[None, :], axis=1)
            nearest_pos = int(np.argmin(distances))
            _, well_row = group_rows[nearest_pos]
            wx = float(well_row["well_centre_x_px"])
            wy = float(well_row["well_centre_y_px"])
            wr = float(radii[nearest_pos])
            well_index = str(well_row["well_index"])

            membership = assess_pdo_well_membership(rgb, obj, wx, wy, wr, settings)
            status = membership["membership_status"]
            if float(distances[nearest_pos]) > 1.20 * wr:
                status = "rejected_outside_well"
                membership["membership_status"] = status
                membership["membership_reason"] = "candidate centroid is too far from the nearest detected microwell"

            included = status == "accepted" or (status == "ambiguous_wall_touching" and not exclude_ambiguous)
            candidate_qc_rows.append({
                "image_series": series,
                "source_image": source,
                "candidate_number": candidate_number,
                "nearest_well_index": well_index,
                "candidate_centroid_x_px": float(obj["x"]),
                "candidate_centroid_y_px": float(obj["y"]),
                "candidate_area_px2": float(obj["area"]),
                "parent_component_id": int(obj["parent_component_id"]),
                "split_method": obj["split_method"],
                "split_confidence": obj["split_confidence"],
                **membership,
                "included_in_quantitative_output": bool(included),
            })

            key = (series, well_index)
            if status == "ambiguous_wall_touching":
                ambiguous_by_well[key] = ambiguous_by_well.get(key, 0) + 1
            elif status == "rejected_outside_well":
                rejected_by_well[key] = rejected_by_well.get(key, 0) + 1

            if status != "accepted":
                crop = _fixed_crop(rgb, float(obj["x"]), float(obj["y"]), 220)
                fname = f"series_{series:02d}_well_{well_index.replace(',', '_')}_cand_{candidate_number:03d}_{status}.png"
                qp = qc_crop_dir / fname
                _qc_label(crop, status, series, well_index, membership["membership_reason"]).save(qp, dpi=(300, 300))
                qc_crop_paths.append(qp)

            if included:
                enriched = dict(obj)
                enriched.update(membership)
                accepted_by_well.setdefault(key, []).append(enriched)

    for key, objects in accepted_by_well.items():
        series, well_index = key
        well_row = wdf[(wdf["image_series"] == series) & (wdf["well_index"].astype(str) == well_index)].iloc[0]
        objects = sorted(objects, key=lambda o: (float(o["x"]), float(o["y"])))
        umpp = float(well_row["um_per_pixel"])
        psc_n = well_row.get("PSC_like_focus_count", np.nan)
        total = len(objects)
        for n, obj in enumerate(objects, start=1):
            size = 2.0 * math.sqrt(float(obj["area"]) / math.pi) * umpp
            pdo_rows.append({
                "image_series": int(series),
                "well_index": well_index,
                "organoid_detection_mode": getattr(settings, "organoid_mode", "GFP-labelled (green fluorescence)"),
                "GFP_labelled_organoids": True,
                "RFP_PSC_stromal_cells_present": bool(getattr(settings, "rfp_psc_present", False)),
                "PDO_number_in_well": int(n),
                "PDO_count_in_well": int(total),
                "centroid_x_px": float(obj["x"]),
                "centroid_y_px": float(obj["y"]),
                "projected_area_px2": float(obj["area"]),
                "equivalent_circular_diameter_um": float(size),
                "PSC_like_focus_count_in_well": psc_n,
                "split_method": obj["split_method"],
                "split_confidence": obj["split_confidence"],
                "membership_status": obj["membership_status"],
                "membership_reason": obj["membership_reason"],
                "centroid_distance_fraction_of_well_radius": obj["centroid_distance_fraction_of_well_radius"],
                "component_inside_detected_well_fraction": obj["component_inside_detected_well_fraction"],
                "wall_before_centroid_fraction": obj["wall_before_centroid_fraction"],
                "qc_status": "automated_pending_visual_review",
            })

    pdf = pd.DataFrame(pdo_rows)
    if pdf.empty:
        pdf = pd.DataFrame(columns=[
            "image_series", "well_index", "organoid_detection_mode", "GFP_labelled_organoids",
            "RFP_PSC_stromal_cells_present", "PDO_number_in_well", "PDO_count_in_well",
            "centroid_x_px", "centroid_y_px", "projected_area_px2",
            "equivalent_circular_diameter_um", "PSC_like_focus_count_in_well",
            "split_method", "split_confidence", "membership_status", "membership_reason",
            "centroid_distance_fraction_of_well_radius", "component_inside_detected_well_fraction",
            "wall_before_centroid_fraction", "qc_status",
        ])

    for idx, row in wdf.iterrows():
        key = (int(row["image_series"]), str(row["well_index"]))
        subset = pdf[(pdf["image_series"] == key[0]) & (pdf["well_index"].astype(str) == key[1])]
        sizes = subset["equivalent_circular_diameter_um"].astype(float).tolist() if not subset.empty else []
        wdf.at[idx, "PDO_count"] = int(len(sizes))
        wdf.at[idx, "PDO_sizes_um"] = "; ".join(f"{v:.4f}" for v in sizes)
        wdf.at[idx, "qc_multiple_pdos_in_well"] = bool(len(sizes) > 1)
        wdf.at[idx, "qc_no_pdo_detected"] = bool(len(sizes) == 0)
        wdf.at[idx, "qc_ambiguous_PDO_candidates_in_well"] = int(ambiguous_by_well.get(key, 0))
        wdf.at[idx, "qc_rejected_PDO_candidates_near_well"] = int(rejected_by_well.get(key, 0))

    candidate_qc = pd.DataFrame(candidate_qc_rows)
    candidate_qc.to_csv(csv_dir / "PDO_candidate_QC.csv", index=False)
    pdf.to_csv(csv_dir / "PDO_raw_data.csv", index=False)
    wdf.to_csv(csv_dir / "well_raw_data.csv", index=False)

    image_rows = []
    for series, wells in wdf.groupby("image_series", sort=True):
        series = int(series)
        base = {}
        if not old_idf.empty and (old_idf["image_series"] == series).any():
            base = old_idf.loc[old_idf["image_series"] == series].iloc[0].to_dict()
        image_pdf = pdf[pdf["image_series"] == series]
        base.update({
            "image_series": series,
            "fully_visible_wells": int(len(wells)),
            "PDO_containing_wells": int((wells["PDO_count"].astype(int) > 0).sum()),
            "PDO_count": int(len(image_pdf)),
            "qc_ambiguous_PDO_candidates": int((candidate_qc["image_series"].eq(series) & candidate_qc["membership_status"].eq("ambiguous_wall_touching")).sum()) if not candidate_qc.empty else 0,
            "qc_rejected_outside_well_candidates": int((candidate_qc["image_series"].eq(series) & candidate_qc["membership_status"].eq("rejected_outside_well")).sum()) if not candidate_qc.empty else 0,
            "qc_status": "automated_pending_visual_review",
        })
        image_rows.append(base)
    idf = pd.DataFrame(image_rows)
    idf.to_csv(image_csv, index=False)

    _rebuild_figures(out_dir, pdf, wdf, settings)
    _rebuild_well_crops(out_dir, wdf, pdf, cols=int(cols))

    qc_contact = out_dir / "figures" / "PDO_candidate_QC_contact_sheet.png"
    if qc_contact.exists():
        qc_contact.unlink()
    if qc_crop_paths:
        _make_contact(qc_crop_paths[:200], qc_contact, cols=max(1, int(cols)), gap=5)

    summary_base = {
        "organoid_detection_mode": getattr(settings, "organoid_mode", "GFP-labelled (green fluorescence)"),
        "GFP_labelled_organoids": True,
        "RFP_PSC_stromal_cells_present": bool(getattr(settings, "rfp_psc_present", False)),
        "images_processed": int(len(idf)),
        "fully_visible_wells": int(len(wdf)),
        "PDO_containing_wells": int((wdf["PDO_count"].astype(int) > 0).sum()),
        "PDO_count": int(len(pdf)),
        "qc_ambiguous_PDO_candidates": int((candidate_qc["membership_status"] == "ambiguous_wall_touching").sum()) if not candidate_qc.empty else 0,
        "qc_rejected_outside_well_candidates": int((candidate_qc["membership_status"] == "rejected_outside_well").sum()) if not candidate_qc.empty else 0,
        "qc_ambiguous_candidates_excluded": bool(exclude_ambiguous),
        "qc_status": "automated_pending_visual_review",
    }
    if not pdf.empty:
        d = pdf["equivalent_circular_diameter_um"].astype(float)
        summary_base.update({
            "mean_PDO_diameter_um": float(d.mean()),
            "median_PDO_diameter_um": float(d.median()),
            "SD_PDO_diameter_um": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "min_PDO_diameter_um": float(d.min()),
            "max_PDO_diameter_um": float(d.max()),
        })
    if bool(getattr(settings, "rfp_psc_present", False)):
        summary_base["PSC_like_foci_all_detected_wells"] = int(pd.to_numeric(wdf["PSC_like_focus_count"], errors="coerce").fillna(0).sum())
        summary_base["PSC_like_foci_in_PDO_wells"] = int(pd.to_numeric(wdf.loc[wdf["PDO_count"].astype(int) > 0, "PSC_like_focus_count"], errors="coerce").fillna(0).sum())
    else:
        summary_base["PSC_like_foci_all_detected_wells"] = None
        summary_base["PSC_like_foci_in_PDO_wells"] = None

    summary = pd.DataFrame([summary_base])
    summary.to_csv(summary_csv, index=False)
    return summary, idf
