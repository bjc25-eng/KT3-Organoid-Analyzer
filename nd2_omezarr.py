from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import zarr


DEFAULT_BIOFORMATS2RAW = Path(
    "/home/ec2-user/tools/bioformats2raw/build/install/bioformats2raw/bin/bioformats2raw"
)


def _axis_name(axis) -> str:
    if isinstance(axis, dict):
        return str(axis.get("name", "")).upper()
    return str(axis).upper()


def _unit_to_um(value: float, unit: str | None) -> float | None:
    unit_norm = str(unit or "").strip().lower().replace("µ", "u").replace("μ", "u")
    factors = {
        "um": 1.0,
        "micrometer": 1.0,
        "micrometre": 1.0,
        "nm": 1e-3,
        "nanometer": 1e-3,
        "nanometre": 1e-3,
        "mm": 1e3,
        "millimeter": 1e3,
        "millimetre": 1e3,
        "m": 1e6,
        "meter": 1e6,
        "metre": 1e6,
    }
    factor = factors.get(unit_norm)
    return None if factor is None else float(value) * factor


def system_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except Exception:
        return 0


def find_bioformats2raw(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("BIOFORMATS2RAW")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    on_path = shutil.which("bioformats2raw")
    if on_path:
        candidates.append(Path(on_path))
    candidates.append(DEFAULT_BIOFORMATS2RAW)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise FileNotFoundError(
        "bioformats2raw was not found. Expected the EC2 build at "
        f"{DEFAULT_BIOFORMATS2RAW} or set BIOFORMATS2RAW."
    )


def conversion_paths(input_nd2: str | Path) -> tuple[Path, Path, Path]:
    src = Path(input_nd2).expanduser().resolve()
    # Staged files live under <object-token>/input/<name>.nd2. Keep the converted
    # representation beside input/work so it survives Streamlit restarts.
    object_root = src.parent.parent
    converted_root = object_root / "converted"
    output = converted_root / f"{src.stem}.ome.zarr"
    marker = converted_root / f"{src.stem}.conversion.json"
    return output, marker, object_root / "bf2raw_tmp"


def probe_omezarr(path: str | Path) -> dict:
    root = zarr.open_group(str(Path(path).expanduser().resolve()), mode="r")
    attrs = dict(root.attrs)
    multiscales = attrs.get("multiscales") or []
    if not multiscales:
        raise RuntimeError("Converted Zarr has no OME-NGFF multiscales metadata.")
    ms = multiscales[0]
    datasets = ms.get("datasets") or []
    if not datasets:
        raise RuntimeError("Converted OME-Zarr contains no image datasets.")
    level0 = datasets[0]
    array_path = str(level0.get("path", "0"))
    if array_path not in root:
        raise RuntimeError(f"OME-Zarr level-0 array '{array_path}' is missing.")
    arr = root[array_path]

    axes_meta = ms.get("axes") or []
    axes = [_axis_name(a) for a in axes_meta]
    if len(axes) != arr.ndim:
        raise RuntimeError(
            f"OME-Zarr axes metadata {axes} does not match array shape {arr.shape}."
        )
    if "X" not in axes or "Y" not in axes:
        raise RuntimeError(f"OME-Zarr axes {axes} do not contain X and Y.")

    width = int(arr.shape[axes.index("X")])
    height = int(arr.shape[axes.index("Y")])
    channel_count = int(arr.shape[axes.index("C")]) if "C" in axes else 1

    transforms = level0.get("coordinateTransformations") or []
    scale = None
    for transform in transforms:
        if str(transform.get("type", "")).lower() == "scale":
            values = transform.get("scale")
            if isinstance(values, (list, tuple)) and len(values) == len(axes):
                scale = [float(v) for v in values]
                break

    voxel_size_um: dict[str, float] = {}
    if scale is not None:
        for spatial_axis in ("X", "Y"):
            idx = axes.index(spatial_axis)
            axis_meta = axes_meta[idx] if idx < len(axes_meta) and isinstance(axes_meta[idx], dict) else {}
            value_um = _unit_to_um(scale[idx], axis_meta.get("unit"))
            if value_um is not None:
                voxel_size_um[spatial_axis.lower()] = float(value_um)

    omero_channels = (attrs.get("omero") or {}).get("channels") or []
    channel_metadata = []
    for i in range(channel_count):
        row = omero_channels[i] if i < len(omero_channels) else {}
        channel_metadata.append(
            {
                "index": i,
                "name": str(row.get("label", f"Channel {i}")),
                "color": row.get("color"),
                "window": row.get("window"),
            }
        )

    return {
        "path": str(Path(path).expanduser().resolve()),
        "format": "OME-Zarr",
        "shape": [int(v) for v in arr.shape],
        "axes": axes,
        "dtype": str(arr.dtype),
        "width_px": width,
        "height_px": height,
        "channel_count": channel_count,
        "channel_metadata": channel_metadata,
        "voxel_size_um": voxel_size_um,
        "level_count": len(datasets),
        "level0_array_path": array_path,
        "level0_chunks": [int(v) for v in arr.chunks],
    }


def validate_against_nd2(zarr_meta: dict, nd2_meta: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    for key, label in (("width_px", "width"), ("height_px", "height"), ("channel_count", "channel count")):
        expected = int(nd2_meta.get(key, 0) or 0)
        actual = int(zarr_meta.get(key, 0) or 0)
        if expected and actual != expected:
            errors.append(f"{label} mismatch: ND2={expected}, OME-Zarr={actual}")

    nd2_voxel = nd2_meta.get("voxel_size_um") or {}
    zarr_voxel = zarr_meta.get("voxel_size_um") or {}
    for axis in ("x", "y"):
        expected = nd2_voxel.get(axis)
        actual = zarr_voxel.get(axis)
        try:
            expected_f = float(expected)
        except (TypeError, ValueError):
            expected_f = 0.0
        try:
            actual_f = float(actual)
        except (TypeError, ValueError):
            actual_f = 0.0
        if expected_f > 0 and actual_f <= 0:
            errors.append(f"OME-Zarr is missing physical {axis.upper()} pixel size.")
        elif expected_f > 0 and abs(actual_f - expected_f) > max(1e-6, expected_f * 1e-6):
            errors.append(
                f"physical {axis.upper()} pixel-size mismatch: ND2={expected_f} µm/px, "
                f"OME-Zarr={actual_f} µm/px"
            )

    nd2_names = [
        str(row.get("name", "")).strip().lower()
        for row in (nd2_meta.get("channel_metadata") or [])
    ]
    zarr_names = [
        str(row.get("name", "")).strip().lower()
        for row in (zarr_meta.get("channel_metadata") or [])
    ]
    if nd2_names and zarr_names and nd2_names != zarr_names:
        warnings.append(
            f"Channel labels differ (ND2={nd2_names}, OME-Zarr={zarr_names}); verify mapping before analysis."
        )

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def convert_nd2_to_omezarr(
    input_nd2: str | Path,
    nd2_meta: dict,
    *,
    series_index: int = 0,
    executable: str | Path | None = None,
    max_workers: int = 1,
    tile_size: int = 1024,
    resolutions: int = 1,
    overwrite: bool = False,
    line_callback: Callable[[str], None] | None = None,
) -> dict:
    src = Path(input_nd2).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    exe = find_bioformats2raw(executable)
    output, marker, tmp_dir = conversion_paths(src)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if output.exists() and marker.exists() and not overwrite:
        meta = probe_omezarr(output)
        validation = validate_against_nd2(meta, nd2_meta)
        if validation["ok"]:
            return {
                "output_path": str(output),
                "reused": True,
                "metadata": meta,
                "validation": validation,
                "command": [],
            }

    if output.exists():
        shutil.rmtree(output)
    if marker.exists():
        marker.unlink()

    command = [
        str(exe),
        str(src),
        str(output),
        "--series",
        str(int(series_index)),
        "--scale-format-string",
        "%2$d/",
        "--ngff-version",
        "0.4",
        "--resolutions",
        str(max(1, int(resolutions))),
        "--tile-width",
        str(max(256, int(tile_size))),
        "--tile-height",
        str(max(256, int(tile_size))),
        "--max-workers",
        str(max(1, int(max_workers))),
        "--memo-directory",
        str(tmp_dir),
        "--progress",
    ]

    env = os.environ.copy()
    total_memory = system_memory_bytes()
    # Keep headroom for the OS, Streamlit and native Bio-Formats libraries.
    if total_memory > 0:
        heap_mib = min(6144, max(1024, int((total_memory * 0.70) / (1024 ** 2))))
        existing = env.get("JAVA_OPTS", "").strip()
        java_opts = f"-Xmx{heap_mib}m -Djava.io.tmpdir={tmp_dir}"
        env["JAVA_OPTS"] = f"{existing} {java_opts}".strip()

    log_tail: list[str] = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if line:
            log_tail.append(line)
            log_tail = log_tail[-120:]
            if line_callback:
                line_callback(line)
    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(log_tail[-30:])
        raise RuntimeError(
            f"bioformats2raw exited with code {return_code}. Last output:\n{tail}"
        )

    meta = probe_omezarr(output)
    validation = validate_against_nd2(meta, nd2_meta)
    if not validation["ok"]:
        raise RuntimeError(
            "OME-Zarr conversion completed but validation failed: "
            + "; ".join(validation["errors"])
        )

    payload = {
        "input_nd2": str(src),
        "output_omezarr": str(output),
        "series_index": int(series_index),
        "command": command,
        "metadata": meta,
        "validation": validation,
    }
    marker.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {
        "output_path": str(output),
        "reused": False,
        "metadata": meta,
        "validation": validation,
        "command": command,
    }
