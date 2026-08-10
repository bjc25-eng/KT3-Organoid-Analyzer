from __future__ import annotations

"""QC-aware whole-array OME-Zarr S3 processor.

This wrapper keeps the validated tiled S3/OME-Zarr reader and well detector from
``s3_omezarr`` but changes GFP PDO handling so segmentation is performed on the
complete local GFP object before microwell-membership QC. Candidates are then
classified with the same shape + DIC-wall logic used by the individual-image
route.

Clear outside-well candidates are rejected. Ambiguous wall-touching candidates
are retained by default unless ``exclude_ambiguous_edge_candidates`` is enabled.
A candidate-level audit table is always written to ``csv/PDO_candidate_QC.csv``.
"""

import json
import math
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

import s3_omezarr as _base
from pdo_qc import assess_pdo_well_membership, segment_pdos_conservative


def list_s3_omezarr_datasets(client, bucket: str, prefix: str = "") -> list[str]:
    return _base.list_s3_omezarr_datasets(client, bucket, prefix)


def process_s3_omezarr(
    client,
    bucket: str,
    dataset_prefix: str,
    settings,
    region: str = "eu-west-2",
    cols: int = 5,
    tile_size: int = 4096,
    gfp_channel: int = 0,
    dic_channel: int = 1,
    crop_size_px: int = 256,
    create_pdo_centred: bool = True,
):
    root = _base._open_s3_group(client, bucket, dataset_prefix, region)
    if "0" not in root:
        raise RuntimeError("OME-Zarr dataset does not contain a level-0 array named 0.")

    arr = root["0"]
    if arr.ndim != 3:
        raise RuntimeError(f"Expected C,Y,X OME-Zarr, got {arr.shape}.")

    c, height, width = map(int, arr.shape)
    if max(gfp_channel, dic_channel) >= c:
        raise RuntimeError(
            f"Dataset has {c} channels; selected indices are not available."
        )

    px_x, px_y = _base._read_scale(root)
    px_um = (px_x + px_y) / 2.0
    if not np.isfinite(px_um) or px_um <= 0:
        raise RuntimeError("OME-Zarr metadata does not contain a valid pixel size.")

    expected_radius = float(settings.well_diameter_um) / (2.0 * px_um)
    tile = max(1024, int(tile_size))
    overlap = int(math.ceil(expected_radius * 3.0))
    tiles = list(_base._tiles(width, height, tile))

    work = Path(tempfile.mkdtemp(prefix="kt3_omezarr_"))
    out = work / "results"
    for d in [
        out / "csv",
        out / "figures",
        out / "pdo_centred_raw_crops",
        out / "pdo_centred_labelled_crops",
        out / "indexed_large_images",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Pass 1: detect fully visible microwells from DIC and retain the dominant
    # connected hexagonal component.
    raw_wells = []
    for cx0, cy0, cw, ch in tiles:
        x0 = max(0, cx0 - overlap)
        y0 = max(0, cy0 - overlap)
        x1 = min(width, cx0 + cw + overlap)
        y1 = min(height, cy0 + ch + overlap)
        dic = _base._u8_local(np.asarray(arr[dic_channel, y0:y1, x0:x1]))
        local = _base._detect_wells_tile(
            dic, expected_radius, float(settings.hough_p2)
        )
        for lx, ly, r in local:
            gx, gy = int(x0 + lx), int(y0 + ly)
            if cx0 <= gx < cx0 + cw and cy0 <= gy < cy0 + ch:
                if (
                    gx - r >= 2
                    and gx + r < width - 2
                    and gy - r >= 2
                    and gy + r < height - 2
                ):
                    raw_wells.append((gx, gy, int(r)))

    raw_wells = _base._dedupe(raw_wells, max(12.0, 0.30 * expected_radius))
    wells, pitch = _base._largest_hex_component(raw_wells)
    if not wells:
        raise RuntimeError("No dominant microwell array was detected.")

    tile_cols = math.ceil(width / tile)
    by_tile: dict[int, list[tuple[int, int, int, int]]] = {}
    for wid, (x, y, r) in enumerate(wells, 1):
        tx = min(tile_cols - 1, x // tile)
        ty = min(math.ceil(height / tile) - 1, y // tile)
        by_tile.setdefault(int(ty * tile_cols + tx), []).append((wid, x, y, r))

    gfp_max = _base._window_end(root, gfp_channel, arr.dtype)
    exclude_ambiguous = bool(
        getattr(settings, "exclude_ambiguous_edge_candidates", False)
    )

    well_rows: list[dict] = []
    pdo_rows: list[dict] = []
    candidate_qc_rows: list[dict] = []

    # Pass 2: segment complete GFP candidates locally, then classify membership
    # using the physical 100-um well calibration plus DIC-wall evidence.
    for ti, (cx0, cy0, cw, ch) in enumerate(tiles):
        here = by_tile.get(ti, [])
        if not here:
            continue

        x0 = max(0, cx0 - overlap)
        y0 = max(0, cy0 - overlap)
        x1 = min(width, cx0 + cw + overlap)
        y1 = min(height, cy0 + ch + overlap)

        gfp = _base._u8_absolute(
            np.asarray(arr[gfp_channel, y0:y1, x0:x1]), gfp_max
        )
        dic = _base._u8_local(
            np.asarray(arr[dic_channel, y0:y1, x0:x1])
        )

        for wid, wx, wy, wr in here:
            lx, ly = int(wx - x0), int(wy - y0)

            # Membership uses the calibrated physical well radius, not the Hough
            # radius. This prevents an oversized detected circle from expanding
            # the region in which a PDO can be considered "inside".
            membership_radius = float(expected_radius)

            # DIC wall profiling samples out to 1.25 x radius.
            cr = int(
                math.ceil(max(float(wr), 1.30 * membership_radius))
            ) + 3
            xa = max(0, lx - cr)
            xb = min(gfp.shape[1], lx + cr + 1)
            ya = max(0, ly - cr)
            yb = min(gfp.shape[0], ly + cr + 1)

            sub = gfp[ya:yb, xa:xb].astype(np.float32)
            dic_sub = dic[ya:yb, xa:xb]
            scx, scy = float(lx - xa), float(ly - ya)

            # Do not mask to the well before segmentation: the whole segmented
            # component is required to measure how much lies inside the well.
            signal = gaussian_filter(sub, 0.8)
            objs = segment_pdos_conservative(signal, settings)
            qc_rgb = np.stack([dic_sub, dic_sub, dic_sub], axis=-1).astype(np.uint8)

            kept = []
            local_qc = []
            for candidate_number, obj in enumerate(objs, start=1):
                membership = assess_pdo_well_membership(
                    qc_rgb,
                    obj,
                    well_x=scx,
                    well_y=scy,
                    well_r=membership_radius,
                    settings=settings,
                )
                status = membership["membership_status"]

                centroid_distance = math.hypot(
                    float(obj["x"]) - scx,
                    float(obj["y"]) - scy,
                )
                if centroid_distance > 1.20 * membership_radius:
                    status = "rejected_outside_well"
                    membership["membership_status"] = status
                    membership["membership_reason"] = (
                        "candidate centroid is too far from the assigned microwell"
                    )

                included = status == "accepted" or (
                    status == "ambiguous_wall_touching" and not exclude_ambiguous
                )

                qc_row = {
                    "well_id": wid,
                    "candidate_number": candidate_number,
                    "candidate_centroid_x_px": x0 + xa + float(obj["x"]),
                    "candidate_centroid_y_px": y0 + ya + float(obj["y"]),
                    "candidate_area_px2": float(obj["area"]),
                    "detected_well_radius_px": float(wr),
                    "membership_reference_radius_px": membership_radius,
                    "parent_component_id": obj.get("parent_component_id", np.nan),
                    "split_method": obj.get("split_method", ""),
                    "split_confidence": obj.get("split_confidence", ""),
                    **membership,
                    "included_in_quantitative_output": bool(included),
                }
                candidate_qc_rows.append(qc_row)
                local_qc.append(qc_row)

                if included:
                    enriched = dict(obj)
                    enriched.update(membership)
                    kept.append(enriched)

            well_rows.append(
                {
                    "well_id": wid,
                    "well_centre_x_px": wx,
                    "well_centre_y_px": wy,
                    "well_radius_px": wr,
                    "membership_reference_radius_px": membership_radius,
                    "um_per_pixel": px_um,
                    "PDO_count": len(kept),
                    "PSC_like_focus_count": np.nan,
                    "qc_ambiguous_PDO_candidates_in_well": sum(
                        r["membership_status"] == "ambiguous_wall_touching"
                        for r in local_qc
                    ),
                    "qc_rejected_PDO_candidates_near_well": sum(
                        r["membership_status"] == "rejected_outside_well"
                        for r in local_qc
                    ),
                    "qc_status": "automated_membership_qc_not_manually_reviewed",
                }
            )

            for n, obj in enumerate(
                sorted(kept, key=lambda o: (float(o["x"]), float(o["y"]))), 1
            ):
                area = float(obj["area"])
                pdo_rows.append(
                    {
                        "well_id": wid,
                        "pdo_number_in_well": n,
                        "pdo_count_in_well": len(kept),
                        "centroid_x_px": x0 + xa + float(obj["x"]),
                        "centroid_y_px": y0 + ya + float(obj["y"]),
                        "projected_area_px2": area,
                        "projected_area_um2": area * (px_um ** 2),
                        "equivalent_circular_diameter_um": (
                            2 * math.sqrt(area / math.pi) * px_um
                        ),
                        "PSC_like_focus_count_in_well": np.nan,
                        "split_method": obj.get("split_method", ""),
                        "split_confidence": obj.get("split_confidence", ""),
                        "membership_status": obj["membership_status"],
                        "membership_reason": obj["membership_reason"],
                        "centroid_distance_fraction_of_well_radius": obj[
                            "centroid_distance_fraction_of_well_radius"
                        ],
                        "component_inside_detected_well_fraction": obj[
                            "component_inside_detected_well_fraction"
                        ],
                        "wall_before_centroid_fraction": obj[
                            "wall_before_centroid_fraction"
                        ],
                        "qc_status": "automated_membership_qc_not_manually_reviewed",
                    }
                )

    wdf = _base._assign_indices(pd.DataFrame(well_rows), pitch)
    pdf = pd.DataFrame(pdo_rows)
    qdf = pd.DataFrame(candidate_qc_rows)

    if not pdf.empty:
        pdf = pdf.merge(
            wdf[["well_id", "well_index", "well_col_index", "well_row_index"]],
            on="well_id",
            how="left",
        )
        pdf["PDO_number_in_well"] = pdf["pdo_number_in_well"]
        pdf["PDO_count_in_well"] = pdf["pdo_count_in_well"]

    if not qdf.empty:
        qdf = qdf.merge(
            wdf[["well_id", "well_index", "well_col_index", "well_row_index"]],
            on="well_id",
            how="left",
        )

    labelled = []
    if create_pdo_centred and not pdf.empty:
        half = int(crop_size_px) // 2
        for _, row in pdf.iterrows():
            cx, cy = float(row.centroid_x_px), float(row.centroid_y_px)
            xx0 = max(0, int(round(cx)) - half)
            yy0 = max(0, int(round(cy)) - half)
            xx1 = min(width, xx0 + int(crop_size_px))
            yy1 = min(height, yy0 + int(crop_size_px))
            xx0 = max(0, xx1 - int(crop_size_px))
            yy0 = max(0, yy1 - int(crop_size_px))
            crop = _base._composite(
                np.asarray(arr[dic_channel, yy0:yy1, xx0:xx1]),
                np.asarray(arr[gfp_channel, yy0:yy1, xx0:xx1]),
                gfp_max,
            )
            if crop.size != (int(crop_size_px), int(crop_size_px)):
                canvas = Image.new(
                    "RGB", (int(crop_size_px), int(crop_size_px)), "black"
                )
                canvas.paste(crop, (0, 0))
                crop = canvas
            stem = (
                f"well_{str(row.well_index).replace(',', '_')}"
                f"_PDO_{int(row.pdo_number_in_well):02d}"
            )
            rp = out / "pdo_centred_raw_crops" / f"{stem}.png"
            lp = out / "pdo_centred_labelled_crops" / f"{stem}_labelled.png"
            crop.save(rp, dpi=(300, 300))
            _base._labelled_crop(crop, row).save(lp, dpi=(300, 300))
            labelled.append(lp)

        _base._contact_sheet(
            labelled,
            out / "figures" / "PDO_centred_contact_sheet_compact.png",
            int(cols),
        )

    wdf.to_csv(out / "csv" / "well_raw_data.csv", index=False)
    pdf.to_csv(out / "csv" / "PDO_raw_data.csv", index=False)
    pdf.to_csv(out / "csv" / "PDO_centred_raw_data.csv", index=False)
    qdf.to_csv(out / "csv" / "PDO_candidate_QC.csv", index=False)

    if not pdf.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        ax.hist(
            pdf.equivalent_circular_diameter_um,
            bins=int(settings.histogram_bins),
            edgecolor="black",
        )
        ax.set(
            xlabel="PDO equivalent circular diameter (µm)",
            ylabel="Number of PDOs",
        )
        fig.tight_layout()
        fig.savefig(out / "figures" / "PDO_size_distribution.png", dpi=300)
        plt.close(fig)

    freq = (
        wdf.PDO_count.value_counts()
        .sort_index()
        .rename_axis("PDO_count")
        .reset_index(name="well_count")
    )
    freq.to_csv(out / "csv" / "PDO_count_frequency_across_wells.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(freq.PDO_count.astype(str), freq.well_count, edgecolor="black")
    ax.set(
        xlabel="PDO count per accepted microwell",
        ylabel="Number of microwells",
    )
    fig.tight_layout()
    fig.savefig(out / "figures" / "PDO_count_per_well_distribution.png", dpi=300)
    plt.close(fig)

    rejected_n = (
        int((qdf.membership_status == "rejected_outside_well").sum())
        if not qdf.empty
        else 0
    )
    ambiguous_n = (
        int((qdf.membership_status == "ambiguous_wall_touching").sum())
        if not qdf.empty
        else 0
    )

    summary = pd.DataFrame(
        [
            {
                "source_dataset": dataset_prefix,
                "images_processed": 1,
                "fully_visible_wells": len(wdf),
                "PDO_containing_wells": int((wdf.PDO_count > 0).sum()),
                "PDO_count": len(pdf),
                "mean_PDO_diameter_um": (
                    float(pdf.equivalent_circular_diameter_um.mean())
                    if len(pdf)
                    else np.nan
                ),
                "median_PDO_diameter_um": (
                    float(pdf.equivalent_circular_diameter_um.median())
                    if len(pdf)
                    else np.nan
                ),
                "SD_PDO_diameter_um": (
                    float(pdf.equivalent_circular_diameter_um.std(ddof=1))
                    if len(pdf) > 1
                    else np.nan
                ),
                "pixel_size_um": px_um,
                "inferred_hex_pitch_px": pitch,
                "raw_well_candidates": len(raw_wells),
                "dominant_hex_array_wells": len(wdf),
                "channel_labels": "; ".join(_base._channel_labels(root, c)),
                "GFP_channel_index": gfp_channel,
                "DIC_channel_index": dic_channel,
                "qc_rejected_outside_well_candidates": rejected_n,
                "qc_ambiguous_PDO_candidates": ambiguous_n,
                "qc_status": "automated_membership_qc_not_manually_reviewed",
            }
        ]
    )
    summary.to_csv(out / "csv" / "overall_summary.csv", index=False)

    idf = pd.DataFrame(
        [
            {
                "image_series": 1,
                "source_image": dataset_prefix,
                "fully_visible_wells": len(wdf),
                "PDO_containing_wells": int((wdf.PDO_count > 0).sum()),
                "PDO_count": len(pdf),
                "um_per_pixel": px_um,
                "qc_rejected_outside_well_candidates": rejected_n,
                "qc_ambiguous_PDO_candidates": ambiguous_n,
            }
        ]
    )
    idf.to_csv(out / "csv" / "image_summary.csv", index=False)

    (out / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "bucket": bucket,
                "dataset_prefix": dataset_prefix,
                "shape_cyx": [c, height, width],
                "pixel_size_um": px_um,
                "well_diameter_um": settings.well_diameter_um,
                "membership_reference_radius_px": expected_radius,
                "inferred_hex_pitch_px": pitch,
                "gfp_channel_index": gfp_channel,
                "dic_channel_index": dic_channel,
                "channel_labels": _base._channel_labels(root, c),
                "outside_well_candidates_rejected": rejected_n,
                "ambiguous_wall_touching_candidates": ambiguous_n,
                "exclude_ambiguous_edge_candidates": exclude_ambiguous,
                "method_note": (
                    "Whole-array OME-Zarr streamed from S3. GFP candidates are "
                    "segmented before well masking, then classified using complete "
                    "segmented-shape overlap with the calibrated physical well radius "
                    "and DIC microwell-wall evidence. Clear outside-well objects are "
                    "rejected; ambiguous wall-touching candidates are retained unless "
                    "explicitly excluded. Conservative shape-confirmed splitting is "
                    "preserved. PSC/RFP foci are not analysed in this route."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return work, out, summary, idf
