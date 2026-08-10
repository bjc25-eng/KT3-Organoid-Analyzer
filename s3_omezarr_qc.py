from __future__ import annotations

"""Whole-array OME-Zarr QC with PDO membership and DIC well-validity veto."""

import json

import numpy as np
import pandas as pd

import s3_omezarr_qc_base as _core
from pdo_qc import assess_pdo_well_membership as _assess_membership
from well_qc import assess_microwell_boundary


def list_s3_omezarr_datasets(client, bucket: str, prefix: str = "") -> list[str]:
    return _core.list_s3_omezarr_datasets(client, bucket, prefix)


def _assess_membership_and_well_validity(
    rgb,
    obj,
    well_x: float,
    well_y: float,
    well_r: float,
    settings,
):
    membership = _assess_membership(
        rgb,
        obj,
        well_x=well_x,
        well_y=well_y,
        well_r=well_r,
        settings=settings,
    )
    validity = assess_microwell_boundary(
        rgb,
        well_x=well_x,
        well_y=well_y,
        well_r=well_r,
        settings=settings,
    )
    membership.update(validity)
    if validity["well_validity_status"] == "rejected_false_well":
        membership["membership_status"] = "rejected_false_well"
        membership["membership_reason"] = validity["well_validity_reason"]
    return membership


def process_s3_omezarr(*args, **kwargs):
    # The preserved membership-QC core resolves this function from module globals,
    # so replacing it here adds the well-validity veto without duplicating the
    # validated S3 reading, tiling, segmentation, crop and plotting code.
    _core.assess_pdo_well_membership = _assess_membership_and_well_validity
    work, out, summary, idf = _core.process_s3_omezarr(*args, **kwargs)

    qpath = out / "csv" / "PDO_candidate_QC.csv"
    wpath = out / "csv" / "well_raw_data.csv"
    qdf = pd.read_csv(qpath) if qpath.exists() else pd.DataFrame()

    if not qdf.empty and "well_validity_status" in qdf.columns:
        false_mask = qdf["well_validity_status"].eq("rejected_false_well")
        false_candidate_n = int(false_mask.sum())
        false_well_n = int(qdf.loc[false_mask, "well_id"].nunique())
    else:
        false_candidate_n = 0
        false_well_n = 0

    summary["qc_rejected_false_well_candidates"] = false_candidate_n
    summary["qc_rejected_false_wells"] = false_well_n
    summary["qc_status"] = (
        "automated_membership_and_well_validity_qc_not_manually_reviewed"
    )
    summary.to_csv(out / "csv" / "overall_summary.csv", index=False)

    idf["qc_rejected_false_well_candidates"] = false_candidate_n
    idf["qc_rejected_false_wells"] = false_well_n
    idf["qc_status"] = (
        "automated_membership_and_well_validity_qc_not_manually_reviewed"
    )
    idf.to_csv(out / "csv" / "image_summary.csv", index=False)

    # Add the validity score to candidate-bearing well rows for auditability.
    # Empty wells are deliberately not vetoed: the calibration set is the fully
    # reviewed PDO-positive population, so changing the 11,635-well denominator
    # would go beyond the evidence used to calibrate this QC rule.
    if wpath.exists() and not qdf.empty and "well_wall_evidence_score" in qdf.columns:
        wdf = pd.read_csv(wpath)
        vcols = [
            "well_id",
            "well_validity_status",
            "well_validity_reason",
            "well_wall_evidence_score",
            "well_wall_evidence_threshold",
        ]
        validity = qdf[vcols].drop_duplicates(subset=["well_id"])
        wdf = wdf.drop(
            columns=[c for c in vcols[1:] if c in wdf.columns],
            errors="ignore",
        ).merge(validity, on="well_id", how="left")
        wdf.to_csv(wpath, index=False)

    metadata_path = out / "analysis_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "false_well_candidates_rejected": false_candidate_n,
                "false_wells_rejected": false_well_n,
                "well_validity_method": (
                    "Candidate-bearing detected wells are checked for a "
                    "circumferential DIC wall. The normalized wall-depth cutoff "
                    "was calibrated from the manually reviewed KT3_day_10005 run."
                ),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return work, out, summary, idf
