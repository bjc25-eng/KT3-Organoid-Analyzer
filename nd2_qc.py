from __future__ import annotations

"""QC-aware native ND2 whole-array processing.

This module ports the validated OME-Zarr PDO QC rules into the lazy native ND2
workflow while preserving the existing ND2 reader, tile scan, checkpoint/resume
behaviour and ML export. GFP candidates are segmented as complete objects, then
classified against the physically calibrated 100-um microwell radius using
segmented-shape overlap, DIC wall evidence and the calibrated microwell-validity
score.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import large_data_core as ldc
from pdo_qc import assess_pdo_well_membership, segment_pdos_conservative
from well_qc import WELL_VALIDITY_SCORE_THRESHOLD, assess_microwell_boundary

ND2_QC_SCHEMA_VERSION = "1.0"


def _physical_pixel_size(metadata: dict, settings) -> tuple[float, float]:
    """Return ND2 µm/px and physical microwell radius in pixels.

    Native ND2 metadata is the calibration source. We deliberately do not infer
    pixel size from Hough radii because that would let well-detection errors alter
    the physical membership boundary and PDO size measurements.
    """
    voxel = metadata.get("voxel_size_um") or {}
    values = []
    for key in ("x", "y"):
        try:
            value = float(voxel.get(key))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0:
            values.append(value)
    if len(values) != 2:
        raise ldc.LargeSourceError(
            "ND2 metadata does not contain valid X/Y physical pixel size. "
            "The final PDO membership QC requires physical calibration and will "
            "not substitute a Hough-radius estimate."
        )
    um_per_pixel = float(np.mean(values))
    membership_radius_px = float(settings.well_diameter_um) / (2.0 * um_per_pixel)
    return um_per_pixel, membership_radius_px


def classify_nd2_gfp_candidates(
    pdo_rgb: np.ndarray,
    dic_rgb: np.ndarray,
    well_x: float,
    well_y: float,
    membership_radius_px: float,
    detected_radius_px: float,
    settings,
) -> tuple[list[dict], list[dict], dict]:
    """Segment and QC GFP candidates for one ND2 microwell crop."""
    signal = ldc.green_excess(pdo_rgb)
    candidates = segment_pdos_conservative(signal, settings)
    exclude_ambiguous = bool(
        getattr(settings, "exclude_ambiguous_edge_candidates", False)
    )

    if candidates:
        well_validity = assess_microwell_boundary(
            dic_rgb,
            well_x=float(well_x),
            well_y=float(well_y),
            well_r=float(membership_radius_px),
            settings=settings,
        )
    else:
        well_validity = {
            "well_validity_status": "not_evaluated_no_candidate",
            "well_validity_reason": "well validity is evaluated only for candidate-bearing wells",
            "well_wall_evidence_score": float("nan"),
            "well_wall_evidence_threshold": float(
                getattr(settings, "well_validity_score_threshold", WELL_VALIDITY_SCORE_THRESHOLD)
            ),
        }

    kept: list[dict] = []
    qc_rows: list[dict] = []
    for candidate_number, obj in enumerate(candidates, start=1):
        membership = assess_pdo_well_membership(
            dic_rgb,
            obj,
            well_x=float(well_x),
            well_y=float(well_y),
            well_r=float(membership_radius_px),
            settings=settings,
        )
        membership.update(well_validity)
        status = str(membership["membership_status"])

        if well_validity["well_validity_status"] == "rejected_false_well":
            status = "rejected_false_well"
            membership["membership_status"] = status
            membership["membership_reason"] = well_validity["well_validity_reason"]
        else:
            centroid_distance = math.hypot(
                float(obj["x"]) - float(well_x),
                float(obj["y"]) - float(well_y),
            )
            if centroid_distance > 1.20 * float(membership_radius_px):
                status = "rejected_outside_well"
                membership["membership_status"] = status
                membership["membership_reason"] = (
                    "candidate centroid is too far from the assigned microwell"
                )

        included = status == "accepted" or (
            status == "ambiguous_wall_touching" and not exclude_ambiguous
        )
        qc_rows.append(
            {
                "candidate_number": int(candidate_number),
                "candidate_centroid_x_px": float(obj["x"]),
                "candidate_centroid_y_px": float(obj["y"]),
                "candidate_area_px2": float(obj["area"]),
                "detected_well_radius_px": float(detected_radius_px),
                "membership_reference_radius_px": float(membership_radius_px),
                "parent_component_id": obj.get("parent_component_id", np.nan),
                "split_method": obj.get("split_method", ""),
                "split_confidence": obj.get("split_confidence", ""),
                **membership,
                "included_in_quantitative_output": bool(included),
            }
        )
        if included:
            enriched = dict(obj)
            enriched.update(membership)
            kept.append(enriched)

    return kept, qc_rows, well_validity


def _write_candidate_qc(work_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(work_dir / "PDO_candidate_QC_partial.csv", index=False)


def _gfp_training_masks(shape: tuple[int, int], cx: int, cy: int,
                        membership_radius_px: float, assigned: list[dict]):
    h, w = map(int, shape)
    well_mask = np.zeros((h, w), dtype=np.uint8)
    pdo_mask = np.zeros((h, w), dtype=np.uint8)
    psc_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(
        well_mask,
        (int(round(cx)), int(round(cy))),
        max(1, int(round(membership_radius_px))),
        255,
        -1,
    )
    for obj in assigned:
        coords = np.asarray(obj.get("coords_yx", []), dtype=int)
        if coords.size:
            valid = (
                (coords[:, 0] >= 0) & (coords[:, 0] < h)
                & (coords[:, 1] >= 0) & (coords[:, 1] < w)
            )
            coords = coords[valid]
            pdo_mask[coords[:, 0], coords[:, 1]] = 255
    return well_mask, pdo_mask, psc_mask


def analyse_large_source_qc(source: dict, settings, config: dict, work_dir: Path,
                            tile_size: int = ldc.DEFAULT_TILE_SIZE,
                            standard_crop_size: int = ldc.DEFAULT_STANDARD_CROP_SIZE,
                            progress_callback=None) -> dict:
    """Analyse one native ND2 source with the validated final GFP QC rules."""
    work_dir.mkdir(parents=True, exist_ok=True)
    uri = str(source["source_uri"])
    with ldc.LargeImageReader(
        uri,
        source_type=source.get("source_type", "auto"),
        series_index=int(source.get("series_index", 0)),
        level=int(source.get("pyramid_level", 0)),
    ) as reader:
        metadata = reader.metadata()
        fingerprint = metadata["reference_fingerprint_sha256"]
        if source.get("source_sha256"):
            metadata["source_sha256"] = str(source["source_sha256"])
        if bool(source.get("compute_full_sha256", False)) and reader.format != "OME-Zarr":
            metadata["source_sha256"] = ldc.compute_streaming_sha256(uri)

        umpp, membership_radius_px = _physical_pixel_size(metadata, settings)
        metadata["analysis_um_per_pixel"] = umpp
        metadata["membership_reference_radius_px"] = membership_radius_px
        metadata["nd2_qc_schema_version"] = ND2_QC_SCHEMA_VERSION
        ldc._atomic_json(work_dir / "source_metadata.json", metadata)

        if (
            reader.channel_count == 1
            and source["organoid_mode"] == ldc.GFP_MODE
            and source.get("rfp_psc_present")
        ):
            raise ldc.LargeSourceError(
                "A single-channel source cannot quantify both GFP-labelled PDOs and RFP PSCs. "
                "Use a multichannel source or disable RFP PSC analysis."
            )

        circles, _ = ldc.scan_wells_tiled(
            reader,
            settings,
            config,
            work_dir,
            fingerprint,
            source["organoid_mode"],
            tile_size,
            progress_callback,
        )
        if len(circles) == 0:
            raise ldc.LargeSourceError("No fully visible microwells were detected in this source.")

        xs, ys = ldc.cluster(circles[:, 0]), ldc.cluster(circles[:, 1])
        analysis_checkpoint_path = work_dir / "well_analysis_checkpoint.json"
        state = ldc._read_json(analysis_checkpoint_path, default={}) or {}
        compatible_resume = (
            state.get("source_fingerprint") == fingerprint
            and state.get("nd2_qc_schema_version") == ND2_QC_SCHEMA_VERSION
        )
        if compatible_resume:
            completed = set(state.get("completed_wells", []))
            wells_rows = ldc._read_partial(work_dir / "well_observations_partial.csv")
            pdo_rows = ldc._read_partial(work_dir / "pdo_observations_partial.csv")
            psc_rows = ldc._read_partial(work_dir / "psc_observations_partial.csv")
            candidate_qc_rows = ldc._read_partial(work_dir / "PDO_candidate_QC_partial.csv")
        else:
            # Tile/well detection checkpoints remain reusable, but per-well
            # measurements from an older QC schema must be recomputed.
            completed = set()
            wells_rows, pdo_rows, psc_rows, candidate_qc_rows = [], [], [], []
            state = {
                "source_fingerprint": fingerprint,
                "completed_wells": [],
                "nd2_qc_schema_version": ND2_QC_SCHEMA_VERSION,
            }

        exp = ldc.stable_token(source.get("experiment_id", "Experiment_001"), "Experiment_001")
        dev = ldc.stable_token(source.get("device_id", "Array_001"), "Array_001")
        lane = int(source["condition_index"])
        tp = int(source["timepoint_index"])
        field_id = ldc.stable_token(
            source.get("field_id", f"F{int(source.get('field_index', 1)):02d}"), "F01"
        )
        source_uid = f"{exp}__{dev}__L{lane:02d}__T{tp:02d}__{field_id}"

        for i, (x, y, r) in enumerate(circles):
            col, row = ldc.grid_index(int(x), int(y), xs, ys)
            well_index = f"{col},{row}"
            trajectory_id = f"{exp}__{dev}__L{lane:02d}__{field_id}__W{col}_{row}"
            obs_id = f"{trajectory_id}__T{tp:02d}"
            if obs_id in completed:
                if progress_callback:
                    progress_callback(i + 1, len(circles), "well_analysis")
                continue

            crop_r = max(
                int(math.ceil(float(r) * 1.75)),
                int(math.ceil(1.35 * membership_radius_px)) + 4,
                int(settings.well_rmax) + 8,
            )
            x0 = max(0, int(x) - crop_r)
            y0 = max(0, int(y) - crop_r)
            x1 = min(reader.width, int(x) + crop_r)
            y1 = min(reader.height, int(y) + crop_r)
            rgb, pdo_rgb, well_rgb = ldc._read_analysis_region(
                reader, x0, y0, x1 - x0, y1 - y0, config, source["organoid_mode"]
            )
            cx, cy = int(x) - x0, int(y) - y0

            local_qc: list[dict] = []
            well_validity = {
                "well_validity_status": "not_applicable_brightfield",
                "well_validity_reason": "GFP final-QC route not used",
                "well_wall_evidence_score": np.nan,
                "well_wall_evidence_threshold": np.nan,
            }
            if source["organoid_mode"] == ldc.GFP_MODE:
                assigned, raw_qc, well_validity = classify_nd2_gfp_candidates(
                    pdo_rgb,
                    well_rgb,
                    cx,
                    cy,
                    membership_radius_px,
                    float(r),
                    settings,
                )
                for q in raw_qc:
                    enriched_q = {
                        "source_uid": source_uid,
                        "trajectory_id": trajectory_id,
                        "well_observation_id": obs_id,
                        "well_index": well_index,
                        "well_col_index": int(col),
                        "well_row_index": int(row),
                        **q,
                    }
                    enriched_q["candidate_centroid_x_px_fullres"] = x0 + float(
                        q["candidate_centroid_x_px"]
                    )
                    enriched_q["candidate_centroid_y_px_fullres"] = y0 + float(
                        q["candidate_centroid_y_px"]
                    )
                    candidate_qc_rows.append(enriched_q)
                    local_qc.append(enriched_q)
            else:
                assigned = ldc.segment_unlabelled_pdos_in_well(
                    pdo_rgb, cx, cy, int(r), settings
                )

            if source.get("rfp_psc_present"):
                foci = ldc.detect_psc(rgb, cx, cy, int(r), settings)
            else:
                foci = []
            psc_n = len(foci) if source.get("rfp_psc_present") else None
            sizes = [
                2.0 * math.sqrt(float(o["area"]) / math.pi) * umpp for o in assigned
            ]

            focus_rows_local = []
            for focus_n, (fx, fy, score) in enumerate(foci, 1):
                gx, gy = x0 + int(fx), y0 + int(fy)
                focus_id = f"{obs_id}__PSCFOCUS{focus_n:03d}"
                psc_rows.append(
                    {
                        "source_uid": source_uid,
                        "trajectory_id": trajectory_id,
                        "well_observation_id": obs_id,
                        "psc_focus_id": focus_id,
                        "focus_number_in_well": focus_n,
                        "focus_x_px_fullres": gx,
                        "focus_y_px_fullres": gy,
                        "focus_score": float(score),
                        "condition_index": lane,
                        "condition": source["condition"],
                        "timepoint_index": tp,
                        "timepoint": source["timepoint"],
                        "elapsed_time": source.get("elapsed_time", np.nan),
                        "qc_status": "automated_not_manually_reviewed",
                    }
                )
                focus_rows_local.append({"focus_x_px": int(fx), "focus_y_px": int(fy)})

            for pdo_n, (obj, size_um) in enumerate(zip(assigned, sizes), 1):
                row_pdo = {
                    "source_uid": source_uid,
                    "trajectory_id": trajectory_id,
                    "well_observation_id": obs_id,
                    "pdo_observation_id": f"{obs_id}__PDO{pdo_n:02d}",
                    "PDO_number_in_well": pdo_n,
                    "PDO_count_in_well": len(assigned),
                    "centroid_x_px_fullres": x0 + float(obj["x"]),
                    "centroid_y_px_fullres": y0 + float(obj["y"]),
                    "projected_area_px2": float(obj["area"]),
                    "projected_area_um2": float(obj["area"]) * umpp ** 2,
                    "equivalent_circular_diameter_um": float(size_um),
                    "PSC_like_focus_count_in_well": psc_n,
                    "condition_index": lane,
                    "condition": source["condition"],
                    "timepoint_index": tp,
                    "timepoint": source["timepoint"],
                    "elapsed_time": source.get("elapsed_time", np.nan),
                    "split_method": obj.get("split_method", ""),
                    "split_confidence": obj.get("split_confidence", ""),
                    "membership_status": obj.get("membership_status", ""),
                    "membership_reason": obj.get("membership_reason", ""),
                    "centroid_distance_fraction_of_well_radius": obj.get(
                        "centroid_distance_fraction_of_well_radius", np.nan
                    ),
                    "component_inside_detected_well_fraction": obj.get(
                        "component_inside_detected_well_fraction", np.nan
                    ),
                    "wall_before_centroid_fraction": obj.get(
                        "wall_before_centroid_fraction", np.nan
                    ),
                    "well_validity_status": obj.get("well_validity_status", ""),
                    "well_validity_reason": obj.get("well_validity_reason", ""),
                    "well_wall_evidence_score": obj.get("well_wall_evidence_score", np.nan),
                    "well_wall_evidence_threshold": obj.get(
                        "well_wall_evidence_threshold", np.nan
                    ),
                    "qc_status": "automated_membership_and_well_validity_qc_not_manually_reviewed",
                }
                pdo_rows.append(row_pdo)

            crop_dir = work_dir / "ml_crops"
            full_dir = work_dir / "fullres_crops"
            mask_dir = work_dir / "ml_masks"
            full_dir.mkdir(parents=True, exist_ok=True)
            raw_name = f"{obs_id}__fullres.png"
            ldc.Image.fromarray(rgb).save(full_dir / raw_name)
            std_name = f"{obs_id}__rgb_256.png"
            ldc._save_standard_crop(rgb, crop_dir / std_name, standard_crop_size)
            if (
                source["organoid_mode"] == ldc.BRIGHTFIELD_MODE
                and int(config.get("brightfield_channel", -1)) >= 0
            ):
                bf = pdo_rgb[..., 0]
                ldc._save_standard_crop(
                    np.stack([bf, bf, bf], axis=-1),
                    crop_dir / f"{obs_id}__brightfield_256.png",
                    standard_crop_size,
                )

            if source["organoid_mode"] == ldc.GFP_MODE:
                well_mask, pdo_mask, psc_mask = _gfp_training_masks(
                    rgb.shape[:2], cx, cy, membership_radius_px, assigned
                )
            else:
                one_circle = np.asarray([[cx, cy, int(r)]], dtype=int)
                well_mask, pdo_mask, psc_mask = ldc.make_training_masks(
                    pdo_rgb, one_circle, focus_rows_local, settings
                )
            ldc._save_standard_mask(
                well_mask, mask_dir / f"{obs_id}__well_mask_256.png", standard_crop_size
            )
            ldc._save_standard_mask(
                pdo_mask, mask_dir / f"{obs_id}__pdo_mask_256.png", standard_crop_size
            )
            ldc._save_standard_mask(
                psc_mask, mask_dir / f"{obs_id}__psc_focus_mask_256.png", standard_crop_size
            )

            ambiguous_n = sum(
                q.get("membership_status") == "ambiguous_wall_touching" for q in local_qc
            )
            outside_n = sum(
                q.get("membership_status") == "rejected_outside_well" for q in local_qc
            )
            false_well_n = sum(
                q.get("membership_status") == "rejected_false_well" for q in local_qc
            )
            wells_rows.append(
                {
                    "source_uid": source_uid,
                    "source_uri": uri,
                    "source_format": metadata["format"],
                    "experiment_id": source.get("experiment_id", "Experiment_001"),
                    "device_id": source.get("device_id", "Array_001"),
                    "biological_replicate_id": source.get(
                        "biological_replicate_id", "Replicate_1"
                    ),
                    "pdo_model": source.get("pdo_model", ""),
                    "condition_index": lane,
                    "condition": source["condition"],
                    "field_id": field_id,
                    "timepoint_index": tp,
                    "timepoint": source["timepoint"],
                    "elapsed_time": source.get("elapsed_time", np.nan),
                    "time_unit": source.get("time_unit", "days"),
                    "drug_or_therapeutic": source.get("drug_or_therapeutic", ""),
                    "concentration": source.get("concentration", np.nan),
                    "concentration_unit": source.get("concentration_unit", ""),
                    "organoid_detection_mode": source["organoid_mode"],
                    "GFP_labelled_organoids": bool(source["organoid_mode"] == ldc.GFP_MODE),
                    "RFP_PSC_stromal_cells_present": bool(source.get("rfp_psc_present")),
                    "well_index": well_index,
                    "well_col_index": col,
                    "well_row_index": row,
                    "well_centre_x_px_fullres": int(x),
                    "well_centre_y_px_fullres": int(y),
                    "well_radius_px": int(r),
                    "membership_reference_radius_px": float(membership_radius_px),
                    "crop_x0_px_fullres": x0,
                    "crop_y0_px_fullres": y0,
                    "crop_width_px": int(x1 - x0),
                    "crop_height_px": int(y1 - y0),
                    "um_per_pixel": umpp,
                    "PDO_count": len(assigned),
                    "PSC_like_focus_count": psc_n,
                    "trajectory_id": trajectory_id,
                    "well_observation_id": obs_id,
                    "standard_rgb_crop": str((crop_dir / std_name).relative_to(work_dir)),
                    "fullres_rgb_crop": str((full_dir / raw_name).relative_to(work_dir)),
                    "qc_status": "automated_membership_and_well_validity_qc_not_manually_reviewed",
                    "qc_fully_visible_well": True,
                    "qc_multiple_pdos_in_well": bool(len(assigned) > 1),
                    "qc_no_pdo_detected": bool(len(assigned) == 0),
                    "qc_ambiguous_PDO_candidates_in_well": int(ambiguous_n),
                    "qc_rejected_PDO_candidates_near_well": int(outside_n),
                    "qc_rejected_false_well_candidates": int(false_well_n),
                    **well_validity,
                    "qc_brightfield_detection_requires_visual_review": bool(
                        source["organoid_mode"] == ldc.BRIGHTFIELD_MODE
                    ),
                }
            )

            completed.add(obs_id)
            state = {
                "source_fingerprint": fingerprint,
                "completed_wells": sorted(completed),
                "source_uid": source_uid,
                "nd2_qc_schema_version": ND2_QC_SCHEMA_VERSION,
            }
            ldc._atomic_json(analysis_checkpoint_path, state)
            if (i + 1) % 20 == 0 or i == len(circles) - 1:
                ldc._write_incremental_tables(work_dir, wells_rows, pdo_rows, psc_rows)
                _write_candidate_qc(work_dir, candidate_qc_rows)
            if progress_callback:
                progress_callback(i + 1, len(circles), "well_analysis")

        ldc._write_incremental_tables(work_dir, wells_rows, pdo_rows, psc_rows)
        _write_candidate_qc(work_dir, candidate_qc_rows)
        pd.DataFrame(
            circles, columns=["x_px_fullres", "y_px_fullres", "radius_px"]
        ).to_csv(work_dir / "detected_wells_fullres.csv", index=False)
        return {
            "source_uid": source_uid,
            "metadata": metadata,
            "well_count": len(wells_rows),
            "pdo_count": len(pdo_rows),
            "psc_focus_count": len(psc_rows),
            "work_dir": str(work_dir),
            "complete": len(completed) >= len(circles),
        }


def process_large_experiment_qc(*args, **kwargs):
    """Run/resume large ND2 analysis and add candidate-level final QC outputs."""
    original = ldc.analyse_large_source
    ldc.analyse_large_source = analyse_large_source_qc
    try:
        result = ldc.process_large_experiment(*args, **kwargs)
    finally:
        ldc.analyse_large_source = original

    root, out, source_manifest, wdf, pdf, pscdf, ldf, run_status, ml_path = result
    source_dirs = list((Path(root) / "sources").glob("*"))
    qdf = ldc._concat_csv([p / "PDO_candidate_QC_partial.csv" for p in source_dirs])
    qdf.to_csv(Path(out) / "csv" / "PDO_candidate_QC.csv", index=False)

    if not qdf.empty:
        outside_n = int(qdf["membership_status"].eq("rejected_outside_well").sum())
        ambiguous_n = int(qdf["membership_status"].eq("ambiguous_wall_touching").sum())
        false_candidate_n = int(qdf["membership_status"].eq("rejected_false_well").sum())
        false_well_n = int(
            qdf.loc[qdf["membership_status"].eq("rejected_false_well"), "well_observation_id"].nunique()
        )
    else:
        outside_n = ambiguous_n = false_candidate_n = false_well_n = 0

    run_status.update(
        {
            "nd2_qc_schema_version": ND2_QC_SCHEMA_VERSION,
            "qc_rejected_outside_well_candidates": outside_n,
            "qc_ambiguous_PDO_candidates": ambiguous_n,
            "qc_rejected_false_well_candidates": false_candidate_n,
            "qc_rejected_false_wells": false_well_n,
        }
    )
    ldc._atomic_json(Path(out) / "run_status.json", run_status)

    summary = pd.DataFrame(
        [
            {
                "sources_total": int(run_status.get("sources_total", 0)),
                "fully_visible_wells": int(len(wdf)),
                "PDO_containing_wells": int((pd.to_numeric(wdf.get("PDO_count", 0), errors="coerce").fillna(0) > 0).sum()) if not wdf.empty else 0,
                "PDO_count": int(len(pdf)),
                "qc_rejected_outside_well_candidates": outside_n,
                "qc_ambiguous_PDO_candidates": ambiguous_n,
                "qc_rejected_false_well_candidates": false_candidate_n,
                "qc_rejected_false_wells": false_well_n,
                "qc_schema_version": ND2_QC_SCHEMA_VERSION,
                "qc_status": "automated_membership_and_well_validity_qc_not_manually_reviewed",
            }
        ]
    )
    summary.to_csv(Path(out) / "csv" / "ND2_QC_summary.csv", index=False)

    if ml_path is not None:
        tables = Path(ml_path) / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        qdf.to_csv(tables / "PDO_candidate_QC.csv", index=False)

    config_path = Path(root) / "run_configuration.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg["nd2_final_qc"] = {
            "schema_version": ND2_QC_SCHEMA_VERSION,
            "exclude_ambiguous_edge_candidates": bool(
                getattr(args[1] if len(args) > 1 else kwargs.get("settings"),
                        "exclude_ambiguous_edge_candidates", False)
            ),
            "well_validity_score_threshold": float(WELL_VALIDITY_SCORE_THRESHOLD),
            "physical_pixel_size_source": "native ND2 X/Y voxel_size_um metadata",
        }
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return root, out, source_manifest, wdf, pdf, pscdf, ldf, run_status, ml_path
