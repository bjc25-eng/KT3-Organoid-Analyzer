#!/usr/bin/env python3
"""Export a deterministic Stage-1 manual PSC annotation pilot.

Scientific scope:
- Selects 300 final wells from final_analysis_qc/final_well_measurements.csv:
  50 per RMC6236 dose, split 25 PDO-positive / 25 PDO-negative.
- Within each dose/status stratum, samples across the full range of the existing
  background-corrected RFP signal (Q1-Q4) as evenly as availability permits.
- Reads only already validated final well coordinates and OME-Zarr pixels.
- Creates blinded DIC / GFP(PDO) / RFP(PSC-associated) / Composite crops.
- Does NOT detect wells, segment PDOs, quantify RFP, count PSCs, or modify any
  upstream output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import zarr
from PIL import Image, ImageDraw

import aws_export_pdo_positive_crops as crop_base
import aws_export_final_pdo_rfp_crops as presentation
from omezarr_cyx import SingletonTZCYX

CHANNELS = dict(presentation.CHANNELS)
SEED = 20260822
TARGET_PER_STATUS = 25
CONDITION_ORDER = [
    "K3T_PSC_RMC6236_Lane_1_DMSO",
    "K3T_PSC_RMC6236_5nm_Lane_2",
    "K3T_PSC_RMC6236_25nm_Lane_3",
    "K3T_PSC_RMC6236_50nm_Lane_1",
    "K3T_PSC_RMC6236_100nm_Lane_5",
    "K3T_PSC_RMC6236_150nm_Lane_6",
]
MANUAL_FIELDS = (
    "manual_PSC_present",
    "manual_PSC_count_exact",
    "manual_PSC_count_bin",
    "manual_confidence",
    "manual_ambiguous",
    "manual_image_quality",
    "manual_notes",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_well(value: object) -> str:
    return crop_base._normalise_well_id(value)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def finite_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def build_manifest(final_wells: list[dict]) -> list[dict]:
    """Reproduce the frozen 300-well Stage-1 pilot deterministically."""
    indices_by_condition: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, row in enumerate(final_wells):
        value = finite_or_none(row.get("RFP_background_corrected_mean"))
        if value is not None:
            indices_by_condition[row["condition_id"]].append((value, index))

    quartile_by_index: dict[int, int] = {}
    for condition_id, pairs in indices_by_condition.items():
        pairs.sort(key=lambda item: (item[0], item[1]))
        n = len(pairs)
        if n == 0:
            continue
        for rank, (_, index) in enumerate(pairs):
            quartile_by_index[index] = min(4, rank * 4 // n + 1)

    rng = random.Random(SEED)
    selected: list[tuple[int, dict, int]] = []

    for condition_id in CONDITION_ORDER:
        if not any(row["condition_id"] == condition_id for row in final_wells):
            raise RuntimeError(f"Missing final wells for condition {condition_id}.")

        for pdo_present in (True, False):
            pools: dict[int, list[tuple[int, dict]]] = {}
            for quartile in (1, 2, 3, 4):
                pool = [
                    (index, final_wells[index])
                    for index in range(len(final_wells))
                    if final_wells[index]["condition_id"] == condition_id
                    and truthy(final_wells[index]["PDO_present"]) == pdo_present
                    and quartile_by_index.get(index) == quartile
                ]
                rng.shuffle(pool)
                pools[quartile] = pool

            requested = {1: 6, 2: 6, 3: 6, 4: 7}
            chosen = {
                q: pools[q][: min(requested[q], len(pools[q]))]
                for q in (1, 2, 3, 4)
            }
            remaining = {
                q: pools[q][len(chosen[q]) :]
                for q in (1, 2, 3, 4)
            }

            shortfall = TARGET_PER_STATUS - sum(len(rows) for rows in chosen.values())
            while shortfall:
                available = [q for q in (1, 2, 3, 4) if remaining[q]]
                if not available:
                    raise RuntimeError(
                        f"Insufficient eligible wells for {condition_id}, "
                        f"PDO_present={pdo_present}."
                    )
                q = max(available, key=lambda item: len(remaining[item]))
                chosen[q].append(remaining[q].pop())
                shortfall -= 1

            for q in (1, 2, 3, 4):
                for index, row in chosen[q]:
                    selected.append((index, row, q))

    rng.shuffle(selected)
    if len(selected) != 300:
        raise RuntimeError(f"Stage-1 design selected {len(selected)} wells, expected 300.")

    manifest: list[dict] = []
    for sample_order, (_, row, quartile) in enumerate(selected, start=1):
        sample_id = f"M{sample_order:03d}"
        output = {
            "sample_order": sample_order,
            "sample_id": sample_id,
            "image_filename": f"{sample_id}.png",
            "condition_id": row["condition_id"],
            "condition_name": row["condition_name"],
            "dose_nM": row["dose_nM"],
            "well_id": norm_well(row["well_id"]),
            "PDO_present_final": row["PDO_present"],
            "PDO_count_final": row["PDO_count"],
            "RFP_quartile_within_dose": f"Q{quartile}",
            "RFP_background_corrected_mean": row["RFP_background_corrected_mean"],
            "RFP_background_qc": row["background_qc"],
            "x_px_fullres": row["x_px_fullres"],
            "y_px_fullres": row["y_px_fullres"],
            "radius_px": row["radius_px"],
        }
        for field in MANUAL_FIELDS:
            output[field] = ""
        manifest.append(output)

    return manifest


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def labelled_crop(
    sample_id: str,
    images: dict[str, Image.Image],
    well: dict,
    left: int,
    top: int,
    panel_size: int,
) -> Image.Image:
    gap, title_height, header_height = 8, 28, 92
    width = 2 * panel_size + 3 * gap
    height = header_height + 2 * (panel_size + title_height) + 3 * gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font = crop_base._fonts()

    draw.rectangle((0, 0, width, header_height), fill="black")
    draw.text(
        (12, 8),
        f"{sample_id} | BLINDED MANUAL PSC ANNOTATION",
        fill="white",
        font=title_font,
    )
    draw.text(
        (12, 39),
        "Count PSCs whose cell body lies inside the yellow well boundary.",
        fill="white",
        font=body_font,
    )
    draw.text(
        (12, 63),
        "Green = GFP PDO | Red = RFP PSC-associated signal",
        fill="white",
        font=body_font,
    )

    panel_defs = (
        ("dic", "DIC"),
        ("gfp", "GFP (PDO)"),
        ("rfp", "RFP (PSC-associated)"),
        ("composite", "Composite"),
    )
    for index, (kind, label) in enumerate(panel_defs):
        column, row_index = index % 2, index // 2
        x = gap + column * (panel_size + gap)
        y = header_height + gap + row_index * (panel_size + title_height + gap)

        draw.rectangle((x, y, x + panel_size, y + title_height), fill="black")
        draw.text((x + 7, y + 5), label, fill="white", font=body_font)

        panel = crop_base._overlay_panel(
            images[kind],
            panel_size=panel_size,
            well_x=float(well["x_px_fullres"]),
            well_y=float(well["y_px_fullres"]),
            well_radius=float(well["radius_px"]),
            left=left,
            top=top,
            pdos=[],
            pixel_size_um=1.0,
        )
        canvas.paste(panel, (x, y + title_height))
        panel.close()

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crop-radius-scale", type=float, default=1.25)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--contact-sheet-size", type=int, default=6)
    args = parser.parse_args()

    final_well_path = args.result_root / "final_analysis_qc" / "final_well_measurements.csv"
    final_wells = read_csv(final_well_path)
    manifest = build_manifest(final_wells)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "Stage1_PSC_manual_annotation_manifest.csv"
    write_manifest(manifest_path, manifest)

    well_map = {
        (row["condition_id"], norm_well(row["well_id"])): row
        for row in final_wells
    }

    batch_path = args.result_root / "batch_status.json"
    batch_status = read_json(batch_path) if batch_path.is_file() else {}

    labelled_dir = args.output_root / "labelled_crops"
    labelled_dir.mkdir(parents=True, exist_ok=True)

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        by_condition[row["condition_id"]].append(row)

    exported: list[dict] = []

    for condition_id in presentation.CONDITIONS:
        selected = by_condition.get(condition_id, [])
        if not selected:
            continue

        condition_folder = args.result_root / condition_id
        condition_summary = read_json(condition_folder / "condition_summary.json")
        zarr_path = crop_base.resolve_omezarr(
            condition_id, condition_summary, batch_status, args.cache_root
        )
        metadata = presentation.probe_omezarr(zarr_path)
        crop_base.validate_omezarr(
            metadata, condition_summary, presentation.EXPECTED_PIXEL_SIZE_UM
        )

        root = zarr.open_group(str(zarr_path), mode="r")
        array = root[metadata["level0_array_path"]]
        planes = SingletonTZCYX(array, metadata["axes"])
        channels, height, width = planes.shape_cyx
        if channels != 3:
            raise RuntimeError(
                f"{condition_id}: expected 3 channels, found {channels}."
            )

        ranges = crop_base.display_ranges(
            metadata, planes, width, height, 2048, 4
        )

        for item in selected:
            key = (condition_id, norm_well(item["well_id"]))
            well = well_map.get(key)
            if well is None:
                raise RuntimeError(f"Selected well missing from final table: {key}.")

            x = float(well["x_px_fullres"])
            y = float(well["y_px_fullres"])
            radius = float(well["radius_px"])
            half = max(1, int(round(radius * args.crop_radius_scale)))

            arrays: dict[str, object] = {}
            left = top = 0
            for kind in ("dic", "gfp", "rfp"):
                arrays[kind], left, top = crop_base._read_padded(
                    planes, CHANNELS[kind], x, y, half, width, height
                )

            images = crop_base._raw_images(
                arrays["dic"], arrays["gfp"], arrays["rfp"], ranges
            )
            image = labelled_crop(
                item["sample_id"], images, well, left, top, args.panel_size
            )
            output_path = labelled_dir / item["image_filename"]
            image.save(output_path, dpi=(300, 300))
            image.close()
            for raw_image in images.values():
                raw_image.close()

            output_row = dict(item)
            output_row.update(
                {
                    "labelled_crop": str(output_path),
                    "export_status": "completed",
                    "export_error": "",
                }
            )
            exported.append(output_row)

    exported.sort(key=lambda row: int(row["sample_order"]))
    if len(exported) != len(manifest):
        raise RuntimeError(
            f"Only {len(exported)}/{len(manifest)} Stage-1 crops exported."
        )

    fields = list(manifest[0].keys()) + [
        "labelled_crop",
        "export_status",
        "export_error",
    ]
    with (args.output_root / "image_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)

    completed_paths = [Path(row["labelled_crop"]) for row in exported]
    contact_sheets = crop_base._contact_sheets(
        completed_paths,
        args.output_root / "contact_sheets",
        args.contact_sheet_size,
    )

    pdf_path = args.output_root / "Stage1_PSC_manual_annotation_review.pdf"
    pages = [Image.open(path).convert("RGB") for path in contact_sheets]
    if pages:
        pages[0].save(
            pdf_path,
            save_all=True,
            append_images=pages[1:],
            resolution=150.0,
        )
        for page in pages:
            page.close()

    summary = {
        "selected_wells": len(manifest),
        "exported_crops": len(completed_paths),
        "manifest": str(manifest_path),
        "pdf": str(pdf_path),
        "scientific_scope": "presentation-only; no PSC counting or segmentation",
        "green_channel": "GFP PDO",
        "red_channel": "RFP PSC-associated signal",
        "sampling_seed": SEED,
        "sampling_design": "50 wells/dose = 25 final PDO-positive + 25 final PDO-negative; spans Q1-Q4 RFP",
    }
    (args.output_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        f"Stage-1 PSC manual annotation export completed: "
        f"{len(completed_paths)}/{len(manifest)} blinded wells exported."
    )


if __name__ == "__main__":
    main()
