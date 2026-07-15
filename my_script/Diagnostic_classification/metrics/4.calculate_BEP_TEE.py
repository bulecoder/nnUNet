"""Calculate the three-dimensional BEP-TEE distance in physical space."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from metric_common import (
    EXPECTED_CASE_COUNT,
    LANDMARK_DIR,
    LANDMARK_QC_CSV,
    extract_landmark_centroids,
    list_case_files,
    load_landmark_qc_flags,
    missing_landmark_names,
    write_csv_atomic,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "4.bep_tee_length.csv"
BEP_ID = 3
TEE_ID = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate BEP-TEE length (mm).")
    parser.add_argument("--input-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--landmark-qc-csv", type=Path, default=LANDMARK_QC_CSV)
    parser.add_argument("--expected-case-count", type=int, default=EXPECTED_CASE_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = list_case_files(args.input_dir, args.expected_case_count)
    qc_flags = load_landmark_qc_flags(args.landmark_qc_csv)

    rows = []
    for case_id, path in tqdm(cases.items(), desc="4.BEP-TEE"):
        value = np.nan
        status = "Success"
        try:
            image = sitk.ReadImage(str(path))
            points = extract_landmark_centroids(image, [BEP_ID, TEE_ID])
            missing = missing_landmark_names(points, [BEP_ID, TEE_ID])
            if missing:
                status = f"Missing: {','.join(missing)}"
            else:
                value = float(np.linalg.norm(points[BEP_ID] - points[TEE_ID]))
        except Exception as exc:
            status = f"Error: {exc}"
        rows.append(
            {
                "Patient_Name": case_id,
                "BEP_TEE_Length_mm": value,
                "Status": status,
                "QC_Flags": qc_flags.get(case_id, ""),
            }
        )

    dataframe = pd.DataFrame(rows)
    write_csv_atomic(dataframe, args.output_csv, float_format="%.3f")
    valid = int(dataframe["BEP_TEE_Length_mm"].notna().sum())
    print(f"BEP-TEE completed: {valid}/{len(dataframe)} valid cases")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()

