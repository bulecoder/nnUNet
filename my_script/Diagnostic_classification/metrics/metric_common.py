"""Shared helpers for the control-group metric scripts."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk


EXPECTED_CASE_COUNT = 41
EXPECTED_CASE_IDS = tuple(
    f"Case{index:03d}" for index in range(1, 44) if index not in {6, 29}
)
CASE_FILE_RE = re.compile(r"^(Case\d{3})\.nii\.gz$")

AIRWAY_DIR = Path("/data1/xyh/data/control_group_output/airway_510_lcc")
LANDMARK_DIR = Path("/data1/xyh/data/control_group_output/landmarks_511_lcc")

REPORT_DIR = Path(
    "/data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/reports"
)
AIRWAY_QC_CSV = REPORT_DIR / "airway_510_lcc_qc/airway_510_qc_all_cases.csv"
LANDMARK_QC_CSV = REPORT_DIR / "postprocess_task511.csv"


LANDMARK_NAMES = {
    1: "AICV",
    2: "ANS",
    3: "BEP",
    4: "PNS",
    5: "TEE",
    6: "TEP",
    7: "TUV",
}


def list_case_files(
    directory: Union[str, os.PathLike], expected_count: int = EXPECTED_CASE_COUNT
) -> Dict[str, Path]:
    """Return sorted CaseXXX NIfTI files and reject partial/ambiguous inputs."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    cases: Dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = CASE_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        case_id = match.group(1)
        if case_id in cases:
            raise RuntimeError(f"Duplicate case ID in {root}: {case_id}")
        cases[case_id] = path

    cases = dict(sorted(cases.items()))
    if expected_count > 0 and len(cases) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} CaseXXX.nii.gz files in {root}, "
            f"but found {len(cases)}. Refusing to calculate a partial cohort."
        )
    if expected_count == EXPECTED_CASE_COUNT and set(cases) != set(EXPECTED_CASE_IDS):
        missing = sorted(set(EXPECTED_CASE_IDS) - set(cases))
        unexpected = sorted(set(cases) - set(EXPECTED_CASE_IDS))
        raise RuntimeError(
            f"The 41-file cohort in {root} does not match the expected control-group "
            f"case IDs. Missing: {missing}; unexpected: {unexpected}"
        )
    return cases


def require_same_case_set(
    first: Mapping[str, Path],
    second: Mapping[str, Path],
    first_name: str,
    second_name: str,
) -> None:
    first_ids = set(first)
    second_ids = set(second)
    if first_ids == second_ids:
        return
    only_first = sorted(first_ids - second_ids)
    only_second = sorted(second_ids - first_ids)
    raise RuntimeError(
        f"Case sets differ between {first_name} and {second_name}. "
        f"Only in {first_name}: {only_first}; only in {second_name}: {only_second}"
    )


def extract_landmark_centroids(
    image: sitk.Image, label_ids: Iterable[int]
) -> Dict[int, Optional[np.ndarray]]:
    """Extract label centroids in SimpleITK physical coordinates (x, y, z; mm)."""
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(image)
    centroids: Dict[int, Optional[np.ndarray]] = {}
    for label_id in label_ids:
        centroids[label_id] = (
            np.asarray(shape.GetCentroid(label_id), dtype=np.float64)
            if shape.HasLabel(label_id)
            else None
        )
    return centroids


def missing_landmark_names(
    centroids: Mapping[int, Optional[np.ndarray]], label_ids: Sequence[int]
) -> List[str]:
    return [LANDMARK_NAMES[label_id] for label_id in label_ids if centroids[label_id] is None]


def geometries_match(
    first: sitk.Image,
    second: sitk.Image,
    atol: float = 1e-5,
) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(first.GetDirection(), second.GetDirection(), atol=atol, rtol=0)
    )


def load_airway_qc_flags(path: Union[str, os.PathLike]) -> Dict[str, str]:
    """Read non-OK flags from the post-LCC airway QC table."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Airway QC CSV does not exist: {csv_path}")

    result: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                continue
            flags = (row.get("flags") or "").strip()
            status = (row.get("status") or "").strip()
            if flags:
                result[case_id] = flags
            elif status and status != "OK":
                result[case_id] = status
    return result


def load_landmark_qc_flags(path: Union[str, os.PathLike]) -> Dict[str, str]:
    """Read warnings emitted by Task 511 postprocessing."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Landmark QC CSV does not exist: {csv_path}")

    result: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                continue
            warning = (row.get("warning") or "").strip()
            status = (row.get("status") or "").strip()
            if warning:
                result[case_id] = warning
            elif status and status not in {"SUCCESS", "OK"}:
                result[case_id] = status
    return result


def combine_qc_flags(*flags: Optional[str]) -> str:
    unique: List[str] = []
    for flag in flags:
        if flag and flag not in unique:
            unique.append(flag)
    return " | ".join(unique)


def write_csv_atomic(
    dataframe: pd.DataFrame,
    output_path: Union[str, os.PathLike],
    float_format: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    dataframe.to_csv(
        temporary,
        index=False,
        float_format=float_format,
        encoding="utf-8-sig",
        na_rep="NA",
    )
    os.replace(temporary, output)
