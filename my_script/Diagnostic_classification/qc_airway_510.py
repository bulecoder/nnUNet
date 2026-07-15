#!/usr/bin/env python3
"""Read-only QC for control-group task-510 airway predictions.

The script reads the raw five-fold ensemble masks and their reference CBCT
images, then writes CSV/Markdown/JSON reports. It never edits or rewrites a
NIfTI file.

The checks are screening checks, not a substitute for manual anatomical
review or comparison with ground truth. In particular, index-z plane areas
reported here are not the anatomically oriented CSAmin used downstream.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(
    "/data1/xyh/data/control_group_output/airway_510_raw"
)
DEFAULT_REFERENCE_DIR = Path("/data1/xyh/data/control_group")
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports" / "airway_510_qc"

EXPECTED_CASE_IDS = tuple(
    f"Case{index:03d}" for index in range(1, 44) if index not in {6, 29}
)
PREDICTION_PATTERN = re.compile(r"^(Case\d{3})\.nii\.gz$")

# Match the historical SimpleITK ConnectedComponent default used by the actual
# 510 LCC postprocessing. The complementary 26-connected background avoids
# treating a cavity that reaches the exterior only through an edge/corner as
# fully enclosed.
FOREGROUND_STRUCTURE = ndimage.generate_binary_structure(3, 1)  # 6-connected
BACKGROUND_STRUCTURE = ndimage.generate_binary_structure(3, 3)  # 26-connected

REPORT_FILENAMES = {
    "csv": "airway_510_qc_all_cases.csv",
    "markdown": "airway_510_qc_abnormal_cases.md",
    "json": "airway_510_qc_summary.json",
}

CSV_FIELDS = (
    "case_id",
    "status",
    "flags",
    "prediction_path",
    "reference_path",
    "geometry_matches_reference",
    "size_x",
    "size_y",
    "size_z",
    "spacing_x_mm",
    "spacing_y_mm",
    "spacing_z_mm",
    "voxel_volume_mm3",
    "labels",
    "foreground_voxels",
    "airway_volume_cm3",
    "foreground_component_count_6conn",
    "largest_component_fraction",
    "lcc_removal_volume_mm3",
    "lcc_removal_percent",
    "second_component_volume_mm3",
    "enclosed_hole_count_26conn_background",
    "enclosed_hole_total_volume_mm3",
    "largest_enclosed_hole_volume_mm3",
    "potential_volume_change_if_holes_filled_percent",
    "max_hole_area_on_one_index_z_plane_mm2",
    "foreground_bbox_voxels",
    "estimated_cropped_labeling_peak_mb",
    "foreground_index_z_first",
    "foreground_index_z_last",
    "foreground_index_z_span_slices",
    "internal_empty_index_z_slice_count",
    "internal_empty_index_z_slices",
    "local_index_z_area_collapse_count",
    "local_index_z_area_collapse_slices",
    "min_nonzero_index_z_plane_area_mm2",
    "max_adjacent_index_z_area_change_fraction",
    "touches_image_boundary",
    "error",
)

ERROR_FLAGS = {
    "MISSING_PREDICTION",
    "UNREADABLE_PREDICTION",
    "MISSING_REFERENCE",
    "UNREADABLE_REFERENCE",
    "NON_3D_OR_NON_SCALAR",
    "INVALID_REFERENCE_HEADER",
    "INVALID_GEOMETRY_METADATA",
    "INVALID_LABELS",
    "EMPTY_FOREGROUND",
    "GEOMETRY_MISMATCH",
}


class QcError(RuntimeError):
    """Fatal setup/reporting error."""


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("Expected a finite non-negative number")
    return parsed


def fraction_0_to_1(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("Expected a finite number strictly between 0 and 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--lcc-loss-warning-percent",
        type=nonnegative_float,
        default=1.0,
        help="Flag cases where LCC would remove at least this percentage (default: 1.0).",
    )
    parser.add_argument(
        "--hole-volume-warning-mm3",
        type=nonnegative_float,
        default=10.0,
        help="Flag enclosed-hole total volume at or above this value (default: 10 mm3).",
    )
    parser.add_argument(
        "--hole-plane-area-warning-mm2",
        type=nonnegative_float,
        default=5.0,
        help=(
            "Flag when holes occupy at least this area on one index-z plane "
            "(default: 5 mm2; this is not anatomical CSAmin)."
        ),
    )
    parser.add_argument(
        "--local-collapse-ratio",
        type=fraction_0_to_1,
        default=0.25,
        help=(
            "Flag an interior slice whose area is below this fraction of both "
            "adjacent slices (default: 0.25)."
        ),
    )
    parser.add_argument(
        "--min-neighbor-area-mm2",
        type=nonnegative_float,
        default=10.0,
        help="Minimum area required in both neighboring slices (default: 10 mm2).",
    )
    parser.add_argument(
        "--volume-outlier-iqr-multiplier",
        type=positive_float,
        default=1.5,
        help="Cohort volume outlier multiplier for the IQR rule (default: 1.5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing report files. NIfTI files are never modified.",
    )
    return parser.parse_args()


def blank_record(case_id: str, prediction_path: Path, reference_path: Path) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "PENDING",
        "flags": [],
        "prediction_path": str(prediction_path),
        "reference_path": str(reference_path),
        "geometry_matches_reference": "",
        "size_x": "",
        "size_y": "",
        "size_z": "",
        "spacing_x_mm": "",
        "spacing_y_mm": "",
        "spacing_z_mm": "",
        "voxel_volume_mm3": "",
        "labels": "",
        "foreground_voxels": "",
        "airway_volume_cm3": "",
        "foreground_component_count_6conn": "",
        "largest_component_fraction": "",
        "lcc_removal_volume_mm3": "",
        "lcc_removal_percent": "",
        "second_component_volume_mm3": "",
        "enclosed_hole_count_26conn_background": "",
        "enclosed_hole_total_volume_mm3": "",
        "largest_enclosed_hole_volume_mm3": "",
        "potential_volume_change_if_holes_filled_percent": "",
        "max_hole_area_on_one_index_z_plane_mm2": "",
        "foreground_bbox_voxels": "",
        "estimated_cropped_labeling_peak_mb": "",
        "foreground_index_z_first": "",
        "foreground_index_z_last": "",
        "foreground_index_z_span_slices": "",
        "internal_empty_index_z_slice_count": "",
        "internal_empty_index_z_slices": "",
        "local_index_z_area_collapse_count": "",
        "local_index_z_area_collapse_slices": "",
        "min_nonzero_index_z_plane_area_mm2": "",
        "max_adjacent_index_z_area_change_fraction": "",
        "touches_image_boundary": "",
        "error": "",
    }


def append_flag(record: dict[str, Any], flag: str) -> None:
    if flag not in record["flags"]:
        record["flags"].append(flag)


def read_header(path: Path) -> dict[str, Any]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return {
        "dimension": int(reader.GetDimension()),
        "components": int(reader.GetNumberOfComponents()),
        "size": tuple(int(value) for value in reader.GetSize()),
        "spacing": tuple(float(value) for value in reader.GetSpacing()),
        "origin": tuple(float(value) for value in reader.GetOrigin()),
        "direction": tuple(float(value) for value in reader.GetDirection()),
    }


def geometry_matches(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-5) -> bool:
    if left["size"] != right["size"]:
        return False
    for field in ("spacing", "origin", "direction"):
        if len(left[field]) != len(right[field]):
            return False
        if not np.allclose(left[field], right[field], rtol=0.0, atol=tolerance):
            return False
    return True


def valid_geometry_metadata(header: dict[str, Any]) -> bool:
    if header["dimension"] != 3 or header["components"] != 1:
        return False
    if len(header["size"]) != 3 or any(value <= 0 for value in header["size"]):
        return False
    if len(header["spacing"]) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in header["spacing"]
    ):
        return False
    if len(header["origin"]) != 3 or not np.all(np.isfinite(header["origin"])):
        return False
    if len(header["direction"]) != 9 or not np.all(np.isfinite(header["direction"])):
        return False
    direction = np.asarray(header["direction"], dtype=float).reshape(3, 3)
    return abs(float(np.linalg.det(direction))) >= 1e-6


def foreground_bbox(mask: np.ndarray) -> tuple[slice, slice, slice]:
    z_indices = np.flatnonzero(np.any(mask, axis=(1, 2)))
    y_indices = np.flatnonzero(np.any(mask, axis=(0, 2)))
    x_indices = np.flatnonzero(np.any(mask, axis=(0, 1)))
    return (
        slice(int(z_indices[0]), int(z_indices[-1]) + 1),
        slice(int(y_indices[0]), int(y_indices[-1]) + 1),
        slice(int(x_indices[0]), int(x_indices[-1]) + 1),
    )


def slice_profile_metrics(
    areas_mm2: np.ndarray,
    local_collapse_ratio: float,
    min_neighbor_area_mm2: float,
) -> dict[str, Any]:
    nonzero = np.flatnonzero(areas_mm2 > 0)
    first = int(nonzero[0])
    last = int(nonzero[-1])
    internal_empty = [
        int(index)
        for index in range(first + 1, last)
        if areas_mm2[index] == 0
    ]
    local_collapse = []
    for index in range(first + 1, last):
        current = float(areas_mm2[index])
        previous = float(areas_mm2[index - 1])
        following = float(areas_mm2[index + 1])
        if current == 0:
            continue
        if min(previous, following) < min_neighbor_area_mm2:
            continue
        if current < local_collapse_ratio * min(previous, following):
            local_collapse.append(int(index))

    pair_left = areas_mm2[first:last]
    pair_right = areas_mm2[first + 1 : last + 1]
    pair_max = np.maximum(pair_left, pair_right)
    valid = pair_max > 0
    if np.any(valid):
        adjacent_change = np.abs(pair_left[valid] - pair_right[valid]) / pair_max[valid]
        max_adjacent_change = float(np.max(adjacent_change))
    else:
        max_adjacent_change = 0.0

    return {
        "foreground_index_z_first": first,
        "foreground_index_z_last": last,
        "foreground_index_z_span_slices": last - first + 1,
        "internal_empty_index_z_slice_count": len(internal_empty),
        "internal_empty_index_z_slices": ";".join(map(str, internal_empty)),
        "local_index_z_area_collapse_count": len(local_collapse),
        "local_index_z_area_collapse_slices": ";".join(map(str, local_collapse)),
        "min_nonzero_index_z_plane_area_mm2": float(np.min(areas_mm2[nonzero])),
        "max_adjacent_index_z_area_change_fraction": max_adjacent_change,
    }


def component_and_hole_metrics(
    cropped_mask: np.ndarray,
    voxel_volume_mm3: float,
    index_z_plane_pixel_area_mm2: float,
) -> dict[str, Any]:
    component_labels, component_count = ndimage.label(
        cropped_mask, structure=FOREGROUND_STRUCTURE
    )
    component_sizes = np.bincount(component_labels.ravel())[1:]
    del component_labels
    component_sizes = np.sort(component_sizes)[::-1]
    foreground_voxels = int(np.sum(component_sizes))
    largest_voxels = int(component_sizes[0])
    second_voxels = int(component_sizes[1]) if component_count > 1 else 0
    removed_voxels = foreground_voxels - largest_voxels

    padded = np.pad(cropped_mask, pad_width=1, mode="constant", constant_values=False)
    background = np.logical_not(padded)
    del padded
    background_labels, _ = ndimage.label(background, structure=BACKGROUND_STRUCTURE)
    del background
    background_sizes = np.bincount(background_labels.ravel())
    external_label = int(background_labels[0, 0, 0])
    candidate_labels = np.flatnonzero(background_sizes > 0)
    enclosed_labels = candidate_labels[
        (candidate_labels != 0) & (candidate_labels != external_label)
    ]
    enclosed_sizes = background_sizes[enclosed_labels]

    max_hole_plane_area_mm2 = 0.0
    if enclosed_labels.size:
        enclosed_mask = np.isin(background_labels, enclosed_labels)
        enclosed_core = enclosed_mask[1:-1, 1:-1, 1:-1]
        hole_slice_counts = np.count_nonzero(enclosed_core, axis=(1, 2))
        max_hole_plane_area_mm2 = float(np.max(hole_slice_counts)) * index_z_plane_pixel_area_mm2
        del enclosed_mask, enclosed_core, hole_slice_counts
    del background_labels

    enclosed_voxels = int(np.sum(enclosed_sizes)) if enclosed_sizes.size else 0
    largest_hole_voxels = int(np.max(enclosed_sizes)) if enclosed_sizes.size else 0
    return {
        "foreground_voxels": foreground_voxels,
        "foreground_component_count_6conn": int(component_count),
        "largest_component_fraction": largest_voxels / foreground_voxels,
        "lcc_removal_volume_mm3": removed_voxels * voxel_volume_mm3,
        "lcc_removal_percent": 100.0 * removed_voxels / foreground_voxels,
        "second_component_volume_mm3": second_voxels * voxel_volume_mm3,
        "enclosed_hole_count_26conn_background": int(enclosed_labels.size),
        "enclosed_hole_total_volume_mm3": enclosed_voxels * voxel_volume_mm3,
        "largest_enclosed_hole_volume_mm3": largest_hole_voxels * voxel_volume_mm3,
        "potential_volume_change_if_holes_filled_percent": (
            100.0 * enclosed_voxels / foreground_voxels
            if foreground_voxels
            else 0.0
        ),
        "max_hole_area_on_one_index_z_plane_mm2": max_hole_plane_area_mm2,
    }


def analyze_case(
    case_id: str,
    prediction_path: Path,
    reference_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record = blank_record(case_id, prediction_path, reference_path)
    if not prediction_path.is_file():
        append_flag(record, "MISSING_PREDICTION")
        record["error"] = "Prediction file is missing"
        return record
    if not reference_path.is_file():
        append_flag(record, "MISSING_REFERENCE")
        record["error"] = "Reference image is missing"
        return record

    try:
        prediction_header = read_header(prediction_path)
    except Exception as error:  # continue so one corrupt case does not hide all others
        append_flag(record, "UNREADABLE_PREDICTION")
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    try:
        reference_header = read_header(reference_path)
    except Exception as error:
        append_flag(record, "UNREADABLE_REFERENCE")
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    if prediction_header["dimension"] != 3 or prediction_header["components"] != 1:
        append_flag(record, "NON_3D_OR_NON_SCALAR")
        record["error"] = "Prediction must be a scalar 3D NIfTI"
        return record
    if reference_header["dimension"] != 3 or reference_header["components"] != 1:
        append_flag(record, "INVALID_REFERENCE_HEADER")
        record["error"] = "Reference must be a scalar 3D NIfTI"
        return record
    if not valid_geometry_metadata(prediction_header) or not valid_geometry_metadata(
        reference_header
    ):
        append_flag(record, "INVALID_GEOMETRY_METADATA")
        record["error"] = "Invalid size, spacing, origin, or direction metadata"
        return record
    record["size_x"], record["size_y"], record["size_z"] = prediction_header[
        "size"
    ]
    (
        record["spacing_x_mm"],
        record["spacing_y_mm"],
        record["spacing_z_mm"],
    ) = prediction_header["spacing"]
    record["voxel_volume_mm3"] = float(np.prod(prediction_header["spacing"]))
    matches = geometry_matches(prediction_header, reference_header)
    record["geometry_matches_reference"] = matches
    if not matches:
        append_flag(record, "GEOMETRY_MISMATCH")

    try:
        prediction = sitk.ReadImage(str(prediction_path))
        array = sitk.GetArrayFromImage(prediction)
    except Exception as error:
        append_flag(record, "UNREADABLE_PREDICTION")
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    unique_values = np.unique(array)
    record["labels"] = ";".join(str(value) for value in unique_values.tolist())
    if not np.all(np.isfinite(unique_values)) or not set(unique_values.tolist()).issubset({0, 1}):
        append_flag(record, "INVALID_LABELS")
        record["error"] = "Expected only labels 0 and 1"
        return record

    mask = array == 1
    del array, prediction
    foreground_voxels = int(np.count_nonzero(mask))
    if foreground_voxels == 0:
        append_flag(record, "EMPTY_FOREGROUND")
        record["foreground_voxels"] = 0
        record["airway_volume_cm3"] = 0.0
        return record

    spacing_xyz = tuple(float(value) for value in prediction_header["spacing"])
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    index_z_plane_pixel_area_mm2 = spacing_xyz[0] * spacing_xyz[1]
    record["airway_volume_cm3"] = foreground_voxels * voxel_volume_mm3 / 1000.0

    record["touches_image_boundary"] = bool(
        np.any(mask[0])
        or np.any(mask[-1])
        or np.any(mask[:, 0, :])
        or np.any(mask[:, -1, :])
        or np.any(mask[:, :, 0])
        or np.any(mask[:, :, -1])
    )
    if record["touches_image_boundary"]:
        append_flag(record, "TOUCHES_IMAGE_BOUNDARY")

    index_z_areas_mm2 = (
        np.count_nonzero(mask, axis=(1, 2)).astype(np.float64)
        * index_z_plane_pixel_area_mm2
    )
    record.update(
        slice_profile_metrics(
            index_z_areas_mm2,
            args.local_collapse_ratio,
            args.min_neighbor_area_mm2,
        )
    )
    if record["internal_empty_index_z_slice_count"]:
        append_flag(record, "INTERNAL_EMPTY_INDEX_Z_SLICES")
    if record["local_index_z_area_collapse_count"]:
        append_flag(record, "LOCAL_INDEX_Z_AREA_COLLAPSE")

    bbox = foreground_bbox(mask)
    cropped_mask = np.ascontiguousarray(mask[bbox])
    record["foreground_bbox_voxels"] = int(cropped_mask.size)
    record["estimated_cropped_labeling_peak_mb"] = (
        8.0 * cropped_mask.size / (1024.0**2)
    )
    del mask
    record.update(
        component_and_hole_metrics(
            cropped_mask,
            voxel_volume_mm3,
            index_z_plane_pixel_area_mm2,
        )
    )
    del cropped_mask

    if (
        record["foreground_component_count_6conn"] > 1
        and record["lcc_removal_percent"] >= args.lcc_loss_warning_percent
    ):
        append_flag(record, "SIGNIFICANT_LCC_REMOVAL")
    if (
        record["enclosed_hole_count_26conn_background"] > 0
        and (
            record["enclosed_hole_total_volume_mm3"] >= args.hole_volume_warning_mm3
            or record["max_hole_area_on_one_index_z_plane_mm2"]
            >= args.hole_plane_area_warning_mm2
        )
    ):
        append_flag(record, "SIGNIFICANT_ENCLOSED_HOLES")
    return record


def add_cohort_volume_outlier_flags(
    records: list[dict[str, Any]], multiplier: float
) -> dict[str, float | None]:
    valid = [
        record
        for record in records
        if isinstance(record["airway_volume_cm3"], (int, float))
        and record["airway_volume_cm3"] > 0
        and not (set(record["flags"]) & ERROR_FLAGS)
        and "UNEXPECTED_CASE" not in record["flags"]
    ]
    if len(valid) < 4:
        return {"q1_cm3": None, "q3_cm3": None, "lower_cm3": None, "upper_cm3": None}
    volumes = np.asarray([record["airway_volume_cm3"] for record in valid], dtype=float)
    q1, q3 = np.percentile(volumes, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    for record in valid:
        if record["airway_volume_cm3"] < lower or record["airway_volume_cm3"] > upper:
            append_flag(record, "COHORT_VOLUME_IQR_OUTLIER")
    return {
        "q1_cm3": float(q1),
        "q3_cm3": float(q3),
        "lower_cm3": float(lower),
        "upper_cm3": float(upper),
    }


def finalize_statuses(records: list[dict[str, Any]]) -> None:
    for record in records:
        flags = set(record["flags"])
        if flags & ERROR_FLAGS:
            record["status"] = "ERROR"
        elif flags:
            record["status"] = "REVIEW"
        else:
            record["status"] = "OK"


def format_number(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value) if value != "" else "—"
    return f"{value:.{digits}f}"


def markdown_report(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    volume_limits: dict[str, float | None],
    generated_at: str,
) -> str:
    abnormal = [record for record in records if record["status"] != "OK"]
    lines = [
        "# 对照组 510 原始分割 QC 异常病例报告",
        "",
        f"生成时间：`{generated_at}`",
        "",
        f"输入目录：`{args.input_dir}`",
        "",
        f"参考图像目录：`{args.reference_dir}`",
        "",
        "> 本脚本只读取 NIfTI 并写出报告，不修改分割结果。所有阈值仅用于筛查，不能代替人工解剖学复核。",
        "",
        "## 汇总",
        "",
        f"- 预期病例数：{len(EXPECTED_CASE_IDS)}",
        f"- 报告病例数：{len(records)}",
        f"- 无警告：{sum(record['status'] == 'OK' for record in records)}",
        f"- 需要复核：{sum(record['status'] == 'REVIEW' for record in records)}",
        f"- 文件或标签错误：{sum(record['status'] == 'ERROR' for record in records)}",
        "",
        "## 筛查阈值",
        "",
        f"- LCC 潜在删除比例：≥ {args.lcc_loss_warning_percent:g}%",
        f"- 封闭空洞总体积：≥ {args.hole_volume_warning_mm3:g} mm³",
        f"- 单个 index-z 平面上的空洞面积：≥ {args.hole_plane_area_warning_mm2:g} mm²",
        f"- 局部面积骤降：当前层面积 < 相邻两层较小值的 {args.local_collapse_ratio:g} 倍，且相邻层均 ≥ {args.min_neighbor_area_mm2:g} mm²",
        f"- 队列容积离群：{args.volume_outlier_iqr_multiplier:g} × IQR",
        "",
    ]
    if volume_limits["lower_cm3"] is not None:
        lines.extend(
            [
                "队列容积筛查范围（仅统计离群，不是临床正常范围）：",
                "",
                f"- Q1–Q3：{volume_limits['q1_cm3']:.3f}–{volume_limits['q3_cm3']:.3f} cm³",
                f"- IQR筛查下限–上限：{volume_limits['lower_cm3']:.3f}–{volume_limits['upper_cm3']:.3f} cm³",
                "",
            ]
        )

    lines.extend(["## 异常病例", ""])
    if not abnormal:
        lines.extend(["未发现达到当前筛查阈值的病例。", ""])
    else:
        lines.extend(
            [
                "| 病例 | 状态 | 标记 | 容积 (cm³) | 连通域数 | LCC删除 (%) | 空洞数 | 空洞总体积 (mm³) | 最大单层空洞面积 (mm²) | 内部空层数 | 边界接触 |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for record in abnormal:
            flags = ", ".join(record["flags"]) or "—"
            lines.append(
                "| {case_id} | {status} | {flags} | {volume} | {components} | "
                "{lcc_loss} | {holes} | {hole_volume} | {hole_area} | {empty} | {boundary} |".format(
                    case_id=record["case_id"],
                    status=record["status"],
                    flags=flags,
                    volume=format_number(record["airway_volume_cm3"]),
                    components=format_number(record["foreground_component_count_6conn"], 0),
                    lcc_loss=format_number(record["lcc_removal_percent"]),
                    holes=format_number(record["enclosed_hole_count_26conn_background"], 0),
                    hole_volume=format_number(record["enclosed_hole_total_volume_mm3"]),
                    hole_area=format_number(record["max_hole_area_on_one_index_z_plane_mm2"]),
                    empty=format_number(record["internal_empty_index_z_slice_count"], 0),
                    boundary=record["touches_image_boundary"] if record["touches_image_boundary"] != "" else "—",
                )
            )
        lines.append("")

    lines.extend(
        [
            "## 解释限制",
            "",
            "- `SIGNIFICANT_LCC_REMOVAL` 表示后续LCC可能删除较多体积，应先确认气道是否被错误分成多段。",
            "- `SIGNIFICANT_ENCLOSED_HOLES` 量化的是26邻域背景定义下的三维封闭背景腔；不能据此自动填洞。",
            "- `INTERNAL_EMPTY_INDEX_Z_SLICES` 和 `LOCAL_INDEX_Z_AREA_COLLAPSE` 可能是断裂，也可能是真实狭窄，必须结合CBCT叠加图复核。",
            "- 单个 index-z 平面面积不是沿气道方向计算的 CSAmin，只用于发现可疑突变。",
            "- `COHORT_VOLUME_IQR_OUTLIER` 只表示相对本队列离群，不表示临床异常或模型错误。",
            "",
            f"全部逐病例数值见：`{REPORT_FILENAMES['csv']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def csv_content(records: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        row = dict(record)
        row["flags"] = ";".join(record["flags"])
        writer.writerow(row)
    return stream.getvalue()


def prepare_report_paths(report_dir: Path, overwrite: bool) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: report_dir / filename for key, filename in REPORT_FILENAMES.items()}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(f"  - {path}" for path in existing)
        raise QcError(
            "Report files already exist. Review them or rerun with --overwrite:\n" + joined
        )
    return paths


def run(args: argparse.Namespace) -> int:
    if not args.input_dir.is_dir():
        raise QcError(f"Input directory does not exist: {args.input_dir}")
    if not args.reference_dir.is_dir():
        raise QcError(f"Reference directory does not exist: {args.reference_dir}")

    prediction_files: dict[str, Path] = {}
    unexpected_files: list[str] = []
    for path in sorted(args.input_dir.iterdir()):
        if not path.is_file() or not path.name.endswith(".nii.gz"):
            continue
        match = PREDICTION_PATTERN.fullmatch(path.name)
        if match is None:
            unexpected_files.append(path.name)
            continue
        prediction_files[match.group(1)] = path

    case_ids = list(EXPECTED_CASE_IDS)
    extra_case_ids = sorted(set(prediction_files) - set(EXPECTED_CASE_IDS))
    case_ids.extend(extra_case_ids)
    records: list[dict[str, Any]] = []
    for case_id in case_ids:
        prediction_path = prediction_files.get(
            case_id, args.input_dir / f"{case_id}.nii.gz"
        )
        reference_path = args.reference_dir / f"{case_id}_0000.nii.gz"
        record = analyze_case(case_id, prediction_path, reference_path, args)
        if case_id in extra_case_ids:
            append_flag(record, "UNEXPECTED_CASE")
        records.append(record)

    volume_limits = add_cohort_volume_outlier_flags(
        records, args.volume_outlier_iqr_multiplier
    )
    finalize_statuses(records)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    paths = prepare_report_paths(args.report_dir, args.overwrite)
    summary = {
        "generated_at": generated_at,
        "read_only_nifti_qc": True,
        "input_dir": str(args.input_dir),
        "reference_dir": str(args.reference_dir),
        "expected_case_ids": list(EXPECTED_CASE_IDS),
        "found_prediction_count": len(prediction_files),
        "unexpected_nifti_filenames": unexpected_files,
        "extra_case_ids": extra_case_ids,
        "status_counts": {
            status: sum(record["status"] == status for record in records)
            for status in ("OK", "REVIEW", "ERROR")
        },
        "flag_counts": {
            flag: sum(flag in record["flags"] for record in records)
            for flag in sorted({flag for record in records for flag in record["flags"]})
        },
        "thresholds": {
            "lcc_loss_warning_percent": args.lcc_loss_warning_percent,
            "hole_volume_warning_mm3": args.hole_volume_warning_mm3,
            "hole_plane_area_warning_mm2": args.hole_plane_area_warning_mm2,
            "local_collapse_ratio": args.local_collapse_ratio,
            "min_neighbor_area_mm2": args.min_neighbor_area_mm2,
            "volume_outlier_iqr_multiplier": args.volume_outlier_iqr_multiplier,
        },
        "cohort_volume_iqr_limits": volume_limits,
        "reports": {key: str(path) for key, path in paths.items()},
    }

    atomic_write_text(paths["csv"], csv_content(records))
    atomic_write_text(
        paths["markdown"],
        markdown_report(records, args, volume_limits, generated_at),
    )
    atomic_write_text(
        paths["json"], json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    print("QC completed. No NIfTI files were modified.")
    print(f"All-case CSV: {paths['csv']}")
    print(f"Abnormal-case Markdown: {paths['markdown']}")
    print(f"Summary JSON: {paths['json']}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except QcError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
