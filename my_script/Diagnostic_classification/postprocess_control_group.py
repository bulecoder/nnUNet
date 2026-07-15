#!/usr/bin/env python3
"""LCC postprocessing for the non-OSA control-group nnU-Net predictions.

This script does not run nnU-Net inference and does not calculate clinical
metrics. It only applies the same custom postprocessing used by the existing
OSA workflow:

* task 510: keep the largest physical airway connected component;
* task 511: for labels 1..7, keep the largest component centroid and redraw
  the historical nominal-6-mm landmark sphere.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/data1/xyh/data/control_group")
INFERENCE_ROOT = Path("/data1/xyh/data/control_group_output")
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"

EXPECTED_CASE_IDS = tuple(
    f"Case{index:03d}" for index in range(1, 44) if index not in {6, 29}
)
REFERENCE_PATTERN = re.compile(r"^(Case\d{3})_0000\.nii\.gz$")
PREDICTION_PATTERN = re.compile(r"^(Case\d{3})\.nii\.gz$")

LANDMARK_NAMES = {
    1: "AICV",
    2: "ANS",
    3: "BEP",
    4: "PNS",
    5: "TEE",
    6: "TEP",
    7: "TUV",
}

TASK_PATHS = {
    510: (
        INFERENCE_ROOT / "airway_510_raw",
        INFERENCE_ROOT / "airway_510_lcc",
    ),
    511: (
        INFERENCE_ROOT / "landmarks_511_raw",
        INFERENCE_ROOT / "landmarks_511_lcc",
    ),
}

SEGMENTATION_ALGORITHM = "largest_physical_connected_component_v1"
LANDMARK_ALGORITHM = "historical_mean_spacing_int_radius_v1"


class PostprocessError(RuntimeError):
    """A validation or postprocessing error that should stop the run."""


def image_header(path: Path) -> dict[str, object]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    try:
        reader.ReadImageInformation()
    except RuntimeError as error:
        raise PostprocessError(f"Cannot read NIfTI header: {path}: {error}") from error

    size = tuple(int(value) for value in reader.GetSize())
    spacing = tuple(float(value) for value in reader.GetSpacing())
    origin = tuple(float(value) for value in reader.GetOrigin())
    direction = tuple(float(value) for value in reader.GetDirection())
    if len(size) != 3 or any(value <= 0 for value in size):
        raise PostprocessError(f"Invalid 3D size {size}: {path}")
    if any((not np.isfinite(value)) or value <= 0 for value in spacing):
        raise PostprocessError(f"Invalid spacing {spacing}: {path}")
    if reader.GetNumberOfComponents() != 1:
        raise PostprocessError(f"Expected a scalar NIfTI: {path}")

    direction_matrix = np.asarray(direction, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction_matrix)):
        raise PostprocessError(f"Non-finite origin/direction: {path}")
    if abs(float(np.linalg.det(direction_matrix))) < 1e-6:
        raise PostprocessError(f"Singular direction matrix: {path}")
    return {
        "size": size,
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
    }


def geometry_from_image(image: sitk.Image) -> dict[str, object]:
    return {
        "size": tuple(int(value) for value in image.GetSize()),
        "spacing": tuple(float(value) for value in image.GetSpacing()),
        "origin": tuple(float(value) for value in image.GetOrigin()),
        "direction": tuple(float(value) for value in image.GetDirection()),
    }


def geometry_matches(
    left: dict[str, object], right: dict[str, object], tolerance: float = 1e-5
) -> bool:
    if left["size"] != right["size"]:
        return False
    return all(
        np.allclose(left[field], right[field], rtol=0.0, atol=tolerance)
        for field in ("spacing", "origin", "direction")
    )


def collect_cases(
    directory: Path,
    pattern: re.Pattern[str],
    expected_ids: Sequence[str],
) -> dict[str, Path]:
    if not directory.is_dir():
        raise PostprocessError(f"Directory does not exist: {directory}")

    cases: dict[str, Path] = {}
    invalid_nifti: list[str] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            case_id = match.group(1)
            if case_id in cases:
                raise PostprocessError(f"Duplicate case ID: {case_id}")
            cases[case_id] = path
        elif path.name.endswith(".nii.gz"):
            invalid_nifti.append(path.name)

    expected = set(expected_ids)
    actual = set(cases)
    problems: list[str] = []
    if expected - actual:
        problems.append("missing: " + ", ".join(sorted(expected - actual)))
    if actual - expected:
        problems.append("unexpected: " + ", ".join(sorted(actual - expected)))
    if invalid_nifti:
        problems.append("invalid NIfTI names: " + ", ".join(invalid_nifti))
    if problems:
        raise PostprocessError(
            f"Case-set validation failed for {directory}: " + " | ".join(problems)
        )
    return cases


def validate_output_path(output_dir: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    allowed_root = INFERENCE_ROOT.resolve()
    if output_dir == allowed_root or not output_dir.is_relative_to(allowed_root):
        raise PostprocessError(
            "Output must be a child directory inside the isolated control-group "
            f"output root {allowed_root}: {output_dir}"
        )
    existing = list(output_dir.glob("*.nii.gz")) if output_dir.is_dir() else []
    if existing and not overwrite:
        raise PostprocessError(
            f"Output already contains {len(existing)} NIfTI files: {output_dir}. "
            "Inspect them first; use --overwrite only for an intentional full rerun."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def label_values(image: sitk.Image) -> list[int]:
    array = sitk.GetArrayViewFromImage(image)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)):
            raise PostprocessError("Label image contains NaN or infinite values")
        if not np.allclose(array, np.rint(array), rtol=0.0, atol=1e-6):
            raise PostprocessError("Label image contains non-integer values")
    return sorted(int(value) for value in np.unique(array))


def airway_lcc(image: sitk.Image) -> tuple[sitk.Image, dict[str, object]]:
    values = label_values(image)
    if not set(values).issubset({0, 1}):
        raise PostprocessError(f"Task 510 contains unexpected labels: {values}")

    binary = sitk.BinaryThreshold(
        image, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0
    )
    binary = sitk.Cast(binary, sitk.sitkUInt8)
    components = sitk.ConnectedComponent(binary)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)
    component_labels = list(stats.GetLabels())

    if not component_labels:
        output = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        output.CopyInformation(image)
        return output, {
            "algorithm": SEGMENTATION_ALGORITHM,
            "raw_labels": values,
            "component_count": 0,
            "foreground_voxels_before": 0,
            "foreground_voxels_after": 0,
            "volume_before_cm3": 0.0,
            "volume_after_cm3": 0.0,
            "warning": "empty airway prediction",
        }

    largest = max(component_labels, key=lambda label: stats.GetPhysicalSize(label))
    output = sitk.BinaryThreshold(
        components,
        lowerThreshold=int(largest),
        upperThreshold=int(largest),
        insideValue=1,
        outsideValue=0,
    )
    output = sitk.Cast(output, sitk.sitkUInt8)
    output.CopyInformation(image)
    return output, {
        "algorithm": SEGMENTATION_ALGORITHM,
        "raw_labels": values,
        "component_count": len(component_labels),
        "foreground_voxels_before": int(
            sum(stats.GetNumberOfPixels(label) for label in component_labels)
        ),
        "foreground_voxels_after": int(stats.GetNumberOfPixels(largest)),
        "volume_before_cm3": round(
            sum(stats.GetPhysicalSize(label) for label in component_labels) / 1000.0,
            6,
        ),
        "volume_after_cm3": round(stats.GetPhysicalSize(largest) / 1000.0, 6),
        "warning": "",
    }


def draw_historical_landmark_sphere(
    array_zyx: np.ndarray,
    center_xyz: Sequence[float],
    radius_pixels_float: float,
    label_id: int,
) -> dict[str, object]:
    """Exactly reproduce the old mean-spacing/int-radius redraw behavior."""
    cx, cy, cz = (float(value) for value in center_xyz)
    size_z, size_y, size_x = array_zyx.shape
    radius = max(1, int(radius_pixels_float))

    raw_x_min, raw_x_max = int(cx - radius - 1), int(cx + radius + 2)
    raw_y_min, raw_y_max = int(cy - radius - 1), int(cy + radius + 2)
    raw_z_min, raw_z_max = int(cz - radius - 1), int(cz + radius + 2)
    x_min, x_max = max(0, raw_x_min), min(size_x, raw_x_max)
    y_min, y_max = max(0, raw_y_min), min(size_y, raw_y_max)
    z_min, z_max = max(0, raw_z_min), min(size_z, raw_z_max)
    clipped = (
        raw_x_min < 0
        or raw_y_min < 0
        or raw_z_min < 0
        or raw_x_max > size_x
        or raw_y_max > size_y
        or raw_z_max > size_z
    )

    z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
    sphere = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius**2
    roi = array_zyx[z_min:z_max, y_min:y_max, x_min:x_max]
    existing = roi[sphere]
    overwritten = int(np.count_nonzero((existing != 0) & (existing != label_id)))
    roi[sphere] = label_id
    return {
        "radius_pixels_float": float(radius_pixels_float),
        "radius_pixels_int": radius,
        "sphere_clipped": clipped,
        "overwritten_foreground_voxels": overwritten,
        "requested_sphere_voxels": int(np.count_nonzero(sphere)),
    }


def landmark_lcc(
    image: sitk.Image, nominal_radius_mm: float
) -> tuple[sitk.Image, dict[str, object], list[dict[str, object]]]:
    values = label_values(image)
    if not set(values).issubset(set(range(8))):
        raise PostprocessError(f"Task 511 contains unexpected labels: {values}")

    spacing = tuple(float(value) for value in image.GetSpacing())
    radius_pixels_float = nominal_radius_mm / float(np.mean(spacing))
    clean_array = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
    details: list[dict[str, object]] = []
    missing: list[str] = []
    total_components = 0

    for label_id, name in LANDMARK_NAMES.items():
        binary = sitk.BinaryThreshold(
            image,
            lowerThreshold=label_id,
            upperThreshold=label_id,
            insideValue=1,
            outsideValue=0,
        )
        components = sitk.ConnectedComponent(sitk.Cast(binary, sitk.sitkUInt8))
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(components)
        component_labels = list(stats.GetLabels())
        total_components += len(component_labels)

        base = {
            "label_id": label_id,
            "landmark": name,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "radius_pixels_float": radius_pixels_float,
            "radius_pixels_int": max(1, int(radius_pixels_float)),
        }
        if not component_labels:
            missing.append(name)
            details.append(
                {
                    **base,
                    "status": "MISSING",
                    "component_count": 0,
                    "largest_component_mm3": "",
                    "sphere_clipped": "",
                    "overwritten_foreground_voxels": "",
                    "requested_sphere_voxels": "",
                    "centroid_x_mm": "",
                    "centroid_y_mm": "",
                    "centroid_z_mm": "",
                    "redrawn_centroid_x_mm": "",
                    "redrawn_centroid_y_mm": "",
                    "redrawn_centroid_z_mm": "",
                    "centroid_shift_mm": "",
                    "redrawn_foreground_voxels": "",
                }
            )
            continue

        largest = max(component_labels, key=lambda label: stats.GetPhysicalSize(label))
        centroid = tuple(float(value) for value in stats.GetCentroid(largest))
        center_index = image.TransformPhysicalPointToContinuousIndex(centroid)
        draw_info = draw_historical_landmark_sphere(
            clean_array, center_index, radius_pixels_float, label_id
        )
        details.append(
            {
                **base,
                "status": "SUCCESS",
                "component_count": len(component_labels),
                "largest_component_mm3": round(stats.GetPhysicalSize(largest), 6),
                **draw_info,
                "centroid_x_mm": round(centroid[0], 6),
                "centroid_y_mm": round(centroid[1], 6),
                "centroid_z_mm": round(centroid[2], 6),
                "redrawn_centroid_x_mm": "",
                "redrawn_centroid_y_mm": "",
                "redrawn_centroid_z_mm": "",
                "centroid_shift_mm": "",
                "redrawn_foreground_voxels": "",
            }
        )

    output = sitk.GetImageFromArray(clean_array)
    output.CopyInformation(image)

    for detail in details:
        if detail["status"] != "SUCCESS":
            continue
        label_id = int(detail["label_id"])
        binary = sitk.BinaryThreshold(output, label_id, label_id, 1, 0)
        components = sitk.ConnectedComponent(sitk.Cast(binary, sitk.sitkUInt8))
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(components)
        component_labels = list(stats.GetLabels())
        if len(component_labels) != 1:
            raise PostprocessError(
                f"Redrawn {detail['landmark']} has {len(component_labels)} components"
            )
        component = component_labels[0]
        redrawn_centroid = np.asarray(stats.GetCentroid(component), dtype=float)
        source_centroid = np.asarray(
            [
                detail["centroid_x_mm"],
                detail["centroid_y_mm"],
                detail["centroid_z_mm"],
            ],
            dtype=float,
        )
        detail["redrawn_centroid_x_mm"] = round(float(redrawn_centroid[0]), 6)
        detail["redrawn_centroid_y_mm"] = round(float(redrawn_centroid[1]), 6)
        detail["redrawn_centroid_z_mm"] = round(float(redrawn_centroid[2]), 6)
        detail["centroid_shift_mm"] = round(
            float(np.linalg.norm(redrawn_centroid - source_centroid)), 6
        )
        detail["redrawn_foreground_voxels"] = int(stats.GetNumberOfPixels(component))

    clipped = [
        str(row["landmark"]) for row in details if row.get("sphere_clipped") is True
    ]
    overwritten = sum(
        int(row.get("overwritten_foreground_voxels") or 0) for row in details
    )
    shifts = [
        float(row["centroid_shift_mm"])
        for row in details
        if row.get("centroid_shift_mm") not in ("", None)
    ]
    warnings: list[str] = []
    if missing:
        warnings.append("missing landmarks: " + ",".join(missing))
    if clipped:
        warnings.append("spheres clipped at boundary: " + ",".join(clipped))
    if overwritten:
        warnings.append(f"sphere overlap overwrote {overwritten} foreground voxels")
    if shifts and max(shifts) > 0.5:
        warnings.append(f"maximum redrawn centroid shift is {max(shifts):.3f} mm")

    summary = {
        "algorithm": LANDMARK_ALGORITHM,
        "raw_labels": values,
        "component_count": total_components,
        "foreground_voxels_before": int(
            np.count_nonzero(sitk.GetArrayViewFromImage(image))
        ),
        "foreground_voxels_after": int(np.count_nonzero(clean_array)),
        "missing_landmarks": missing,
        "clipped_landmarks": clipped,
        "overwritten_foreground_voxels": overwritten,
        "max_centroid_shift_mm": max(shifts, default=0.0),
        "warning": " | ".join(warnings),
    }
    return output, summary, details


def write_image_atomic(image: sitk.Image, path: Path) -> None:
    temporary = path.with_name(path.name.removesuffix(".nii.gz") + ".tmp.nii.gz")
    if temporary.exists():
        temporary.unlink()
    sitk.WriteImage(image, str(temporary), True)
    if not geometry_matches(geometry_from_image(image), image_header(temporary)):
        temporary.unlink(missing_ok=True)
        raise PostprocessError(f"Geometry changed while writing: {path}")
    os.replace(temporary, path)


def write_csv_atomic(
    path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    input_dir = args.input_dir or TASK_PATHS[args.task][0]
    output_dir = args.output_dir or TASK_PATHS[args.task][1]
    reference_dir = args.reference_dir.resolve()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    report_dir = args.report_dir.resolve()

    if input_dir == output_dir:
        raise PostprocessError("Input and output directories must be different")
    if output_dir == reference_dir or output_dir == DATA_ROOT.resolve():
        raise PostprocessError("Output must not be the uploaded NIfTI reference directory")
    if not report_dir.is_relative_to(SCRIPT_DIR):
        raise PostprocessError("Reports must remain inside Diagnostic_classification")

    references = collect_cases(reference_dir, REFERENCE_PATTERN, EXPECTED_CASE_IDS)
    predictions = collect_cases(input_dir, PREDICTION_PATTERN, EXPECTED_CASE_IDS)
    validate_output_path(output_dir, args.overwrite)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = report_dir / f"postprocess_task{args.task}_run.json"
    manifest: dict[str, object] = {
        "status": "STARTED",
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "task": args.task,
        "algorithm": SEGMENTATION_ALGORITHM if args.task == 510 else LANDMARK_ALGORITHM,
        "nominal_radius_mm": args.radius_mm if args.task == 511 else None,
        "reference_dir": str(reference_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "case_ids": list(EXPECTED_CASE_IDS),
    }
    write_json_atomic(manifest_path, manifest)

    case_rows: list[dict[str, object]] = []
    landmark_rows: list[dict[str, object]] = []
    for index, case_id in enumerate(EXPECTED_CASE_IDS, start=1):
        reference_header = image_header(references[case_id])
        prediction_header = image_header(predictions[case_id])
        if not geometry_matches(reference_header, prediction_header):
            raise PostprocessError(f"Prediction geometry differs from input: {case_id}")

        prediction = sitk.ReadImage(str(predictions[case_id]))
        if args.task == 510:
            processed, summary = airway_lcc(prediction)
            missing: list[str] = []
        else:
            processed, summary, details = landmark_lcc(prediction, args.radius_mm)
            missing = list(summary["missing_landmarks"])
            landmark_rows.extend({"case_id": case_id, **row} for row in details)

        output_path = output_dir / f"{case_id}.nii.gz"
        write_image_atomic(processed, output_path)
        spacing = prediction.GetSpacing()
        case_rows.append(
            {
                "case_id": case_id,
                "task": args.task,
                "status": "SUCCESS_WITH_WARNING" if summary["warning"] else "SUCCESS",
                "algorithm": summary["algorithm"],
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "raw_labels": ";".join(str(value) for value in summary["raw_labels"]),
                "component_count": summary["component_count"],
                "foreground_voxels_before": summary["foreground_voxels_before"],
                "foreground_voxels_after": summary["foreground_voxels_after"],
                "volume_before_cm3": summary.get("volume_before_cm3", ""),
                "volume_after_cm3": summary.get("volume_after_cm3", ""),
                "missing_landmarks": ";".join(missing),
                "clipped_landmarks": ";".join(summary.get("clipped_landmarks", [])),
                "overwritten_foreground_voxels": summary.get(
                    "overwritten_foreground_voxels", ""
                ),
                "max_centroid_shift_mm": summary.get("max_centroid_shift_mm", ""),
                "warning": summary["warning"],
            }
        )
        print(f"[{index:02d}/41] {case_id} -> {output_path.name}")

    completed = collect_cases(output_dir, PREDICTION_PATTERN, EXPECTED_CASE_IDS)
    report_path = report_dir / f"postprocess_task{args.task}.csv"
    fields = (
        "case_id",
        "task",
        "status",
        "algorithm",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "raw_labels",
        "component_count",
        "foreground_voxels_before",
        "foreground_voxels_after",
        "volume_before_cm3",
        "volume_after_cm3",
        "missing_landmarks",
        "clipped_landmarks",
        "overwritten_foreground_voxels",
        "max_centroid_shift_mm",
        "warning",
    )
    write_csv_atomic(report_path, case_rows, fields)
    archived_report = report_dir / f"postprocess_task{args.task}_{run_id}.csv"
    shutil.copy2(report_path, archived_report)

    if args.task == 511:
        landmark_fields = (
            "case_id",
            "label_id",
            "landmark",
            "status",
            "component_count",
            "largest_component_mm3",
            "spacing_x",
            "spacing_y",
            "spacing_z",
            "radius_pixels_float",
            "radius_pixels_int",
            "sphere_clipped",
            "overwritten_foreground_voxels",
            "requested_sphere_voxels",
            "centroid_x_mm",
            "centroid_y_mm",
            "centroid_z_mm",
            "redrawn_centroid_x_mm",
            "redrawn_centroid_y_mm",
            "redrawn_centroid_z_mm",
            "centroid_shift_mm",
            "redrawn_foreground_voxels",
        )
        write_csv_atomic(
            report_dir / "landmark_centroids_task511.csv",
            landmark_rows,
            landmark_fields,
        )

    manifest.update(
        {
            "status": "COMPLETED",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "processed_count": len(completed),
            "latest_report": str(report_path),
            "archived_report": str(archived_report),
        }
    )
    write_json_atomic(manifest_path, manifest)
    print(f"Completed task {args.task}: {len(completed)} cases")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")
    return 0


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=(510, 511), required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Raw nnUNet prediction directory; defaults depend on --task.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="LCC output directory; defaults depend on --task.",
    )
    parser.add_argument("--reference-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--radius-mm",
        type=positive_float,
        default=6.0,
        help=(
            "Historical nominal landmark radius. The old pipeline divides this by "
            "mean spacing and truncates to an integer voxel radius."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        return run(parse_args())
    except PostprocessError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
