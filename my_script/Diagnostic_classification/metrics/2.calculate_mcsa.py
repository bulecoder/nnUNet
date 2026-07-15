"""Calculate mCSA from Task 510 masks using the historical OSA workflow."""

from __future__ import annotations

import argparse
import gc
import io
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import SimpleITK as sitk
import trimesh
from skimage import measure
from tqdm import tqdm

from metric_common import (
    AIRWAY_DIR,
    AIRWAY_QC_CSV,
    EXPECTED_CASE_COUNT,
    LANDMARK_DIR,
    LANDMARK_QC_CSV,
    combine_qc_flags,
    extract_landmark_centroids,
    geometries_match,
    list_case_files,
    load_airway_qc_flags,
    load_landmark_qc_flags,
    missing_landmark_names,
    require_same_case_set,
    write_csv_atomic,
)


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "2.mcas.csv"
ANS_ID = 2
PNS_ID = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate minimum airway cross-sectional area with the historical "
            "NIfTI-to-smoothed-STL and ANS-PNS standardization method."
        )
    )
    parser.add_argument("--airway-dir", type=Path, default=AIRWAY_DIR)
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--airway-qc-csv", type=Path, default=AIRWAY_QC_CSV)
    parser.add_argument("--landmark-qc-csv", type=Path, default=LANDMARK_QC_CSV)
    parser.add_argument("--expected-case-count", type=int, default=EXPECTED_CASE_COUNT)
    parser.add_argument("--smoothing-iterations", type=int, default=10)
    parser.add_argument("--section-step-mm", type=float, default=0.5)
    parser.add_argument("--minimum-section-area-mm2", type=float, default=5.0)
    parser.add_argument("--moving-average-width", type=int, default=5)
    return parser.parse_args()


def nifti_to_historical_stl_mesh(
    image: sitk.Image, smoothing_iterations: int
) -> trimesh.Trimesh:
    """Reproduce the old marching-cubes, physical-coordinate and STL workflow."""
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(image)
    foreground_labels = {int(label) for label in shape.GetLabels()}
    unexpected = sorted(foreground_labels - {1})
    if unexpected:
        raise ValueError(f"unexpected airway labels: {unexpected}")
    if not shape.HasLabel(1):
        raise ValueError("empty airway segmentation")

    mask = sitk.GetArrayFromImage(image) > 0
    vertices_zyx, faces, _, _ = measure.marching_cubes(mask, level=0.5)

    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    vertices_xyz = vertices_zyx[:, ::-1].astype(np.float64, copy=False)
    # The old exporter allocated physical vertices with np.zeros_like(verts),
    # so coordinates were float32 before Laplacian smoothing. Preserve that.
    vertices_physical = (
        origin + (vertices_xyz * spacing) @ direction.T
    ).astype(vertices_zyx.dtype, copy=False)

    mesh = trimesh.Trimesh(vertices=vertices_physical, faces=faces)
    smoothed = trimesh.smoothing.filter_laplacian(
        mesh, iterations=smoothing_iterations
    )
    if smoothed is not None:
        mesh = smoothed

    # The historical workflow exported the smoothed mesh as STL and loaded it
    # again before sectioning. Keep that serialization step in memory.
    stl_bytes = mesh.export(file_type="stl")
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh


def calculate_historical_mcsa(
    mesh: trimesh.Trimesh,
    angle_rad: float,
    rotation_center: np.ndarray,
    section_step_mm: float,
    minimum_section_area_mm2: float,
    moving_average_width: int,
) -> float:
    transform = trimesh.transformations.rotation_matrix(
        -angle_rad, [1, 0, 0], rotation_center
    )
    mesh.apply_transform(transform)

    z_min, z_max = mesh.bounds[:, 2]
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        raise ValueError("invalid mesh z bounds")
    z_start = z_min + (z_max - z_min) * 0.1
    z_end = z_min + (z_max - z_min) * 0.9

    areas: List[float] = []
    for z_level in np.arange(z_start, z_end, section_step_mm):
        section_3d = mesh.section(
            plane_origin=[0, 0, float(z_level)], plane_normal=[0, 0, 1]
        )
        if section_3d is None:
            continue
        try:
            section_2d, _ = section_3d.to_2D()
            area = float(section_2d.area)
        except Exception:
            continue
        if area > minimum_section_area_mm2:
            areas.append(area)

    if not areas:
        raise ValueError("no valid airway cross-sections")

    area_values = np.asarray(areas, dtype=np.float64)
    # Preserve the old condition: smoothing is applied only when len > width.
    if len(area_values) > moving_average_width:
        kernel = np.ones(moving_average_width, dtype=np.float64) / moving_average_width
        area_values = np.convolve(area_values, kernel, mode="valid")
    return float(np.min(area_values))


def main() -> None:
    args = parse_args()
    if args.smoothing_iterations < 0:
        raise ValueError("--smoothing-iterations must be non-negative")
    if args.section_step_mm <= 0:
        raise ValueError("--section-step-mm must be positive")
    if args.minimum_section_area_mm2 < 0:
        raise ValueError("--minimum-section-area-mm2 must be non-negative")
    if args.moving_average_width < 1:
        raise ValueError("--moving-average-width must be at least 1")

    airway_cases = list_case_files(args.airway_dir, args.expected_case_count)
    landmark_cases = list_case_files(args.landmark_dir, args.expected_case_count)
    require_same_case_set(airway_cases, landmark_cases, "airway", "landmark")

    airway_qc = load_airway_qc_flags(args.airway_qc_csv)
    landmark_qc = load_landmark_qc_flags(args.landmark_qc_csv)
    rows = []

    for case_id in tqdm(airway_cases, desc="2.mCSA"):
        value = np.nan
        status = "Success"
        airway_image = None
        landmark_image = None
        mesh = None
        try:
            airway_image = sitk.ReadImage(str(airway_cases[case_id]))
            landmark_image = sitk.ReadImage(str(landmark_cases[case_id]))
            if not geometries_match(airway_image, landmark_image):
                raise ValueError("airway and landmark geometries do not match")

            points = extract_landmark_centroids(landmark_image, [ANS_ID, PNS_ID])
            missing = missing_landmark_names(points, [ANS_ID, PNS_ID])
            if missing:
                status = f"Missing: {','.join(missing)}"
            else:
                vector = points[ANS_ID] - points[PNS_ID]
                if np.linalg.norm(vector) == 0:
                    raise ValueError("ANS and PNS are identical")
                angle_rad = float(np.arctan2(vector[2], vector[1]))
                center = (points[ANS_ID] + points[PNS_ID]) / 2.0
                mesh = nifti_to_historical_stl_mesh(
                    airway_image, args.smoothing_iterations
                )
                value = calculate_historical_mcsa(
                    mesh=mesh,
                    angle_rad=angle_rad,
                    rotation_center=center,
                    section_step_mm=args.section_step_mm,
                    minimum_section_area_mm2=args.minimum_section_area_mm2,
                    moving_average_width=args.moving_average_width,
                )
        except Exception as exc:
            status = f"Error: {exc}"
        finally:
            mesh = None
            airway_image = None
            landmark_image = None
            gc.collect()

        rows.append(
            {
                "Case Name": case_id,
                "Pred_mCSA_mm2": value,
                "Status": status,
                "QC_Flags": combine_qc_flags(
                    airway_qc.get(case_id), landmark_qc.get(case_id)
                ),
            }
        )

    dataframe = pd.DataFrame(rows)
    write_csv_atomic(dataframe, args.output_csv, float_format="%.2f")
    valid = int(dataframe["Pred_mCSA_mm2"].notna().sum())
    print(f"mCSA completed: {valid}/{len(dataframe)} valid cases")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
