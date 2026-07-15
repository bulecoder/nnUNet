"""Calculate upper-airway volume from Task 510 postprocessed masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from metric_common import (
    AIRWAY_DIR,
    AIRWAY_QC_CSV,
    EXPECTED_CASE_COUNT,
    list_case_files,
    load_airway_qc_flags,
    write_csv_atomic,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "1.volumes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Task 510 airway volume in mm^3 and cm^3 (mL)."
    )
    parser.add_argument("--input-dir", type=Path, default=AIRWAY_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--airway-qc-csv", type=Path, default=AIRWAY_QC_CSV)
    parser.add_argument("--expected-case-count", type=int, default=EXPECTED_CASE_COUNT)
    return parser.parse_args()


def calculate_volume(path: Path) -> tuple[int, float, float]:
    image = sitk.ReadImage(str(path))
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(image)
    foreground_labels = {int(label) for label in shape.GetLabels()}
    unexpected = sorted(foreground_labels - {1})
    if unexpected:
        raise ValueError(f"unexpected foreground labels: {unexpected}")
    if not shape.HasLabel(1):
        raise ValueError("empty airway segmentation")

    voxel_count = int(shape.GetNumberOfPixels(1))
    voxel_volume_mm3 = float(np.prod(np.asarray(image.GetSpacing(), dtype=np.float64)))
    volume_mm3 = voxel_count * voxel_volume_mm3
    return voxel_count, volume_mm3, volume_mm3 / 1000.0


def main() -> None:
    args = parse_args()
    cases = list_case_files(args.input_dir, args.expected_case_count)
    qc_flags = load_airway_qc_flags(args.airway_qc_csv)

    rows = []
    for case_id, path in tqdm(cases.items(), desc="1.Volume"):
        voxel_count = np.nan
        volume_mm3 = np.nan
        volume_cm3 = np.nan
        status = "Success"
        try:
            voxel_count, volume_mm3, volume_cm3 = calculate_volume(path)
        except Exception as exc:
            status = f"Error: {exc}"
        rows.append(
            {
                "Case_ID": case_id,
                "Voxel_Count": voxel_count,
                "Volume_mm3": volume_mm3,
                "Volume_cm3_mL": volume_cm3,
                "Status": status,
                "QC_Flags": qc_flags.get(case_id, ""),
            }
        )

    dataframe = pd.DataFrame(rows)
    write_csv_atomic(dataframe, args.output_csv, float_format="%.4f")
    valid = int(dataframe["Volume_cm3_mL"].notna().sum())
    print(f"Volume completed: {valid}/{len(dataframe)} valid cases")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()

