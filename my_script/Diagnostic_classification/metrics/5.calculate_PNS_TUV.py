"""Calculate PNS-TUV length and the historical soft-palate height."""

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
OUTPUT_CSV = SCRIPT_DIR / "5.pns_tuv_and_height.csv"
ANS_ID = 2
PNS_ID = 4
TUV_ID = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate PNS-TUV length and TUV-to-ANS-PNS-line height (mm)."
    )
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
    for case_id, path in tqdm(cases.items(), desc="5.PNS-TUV"):
        length_mm = np.nan
        height_mm = np.nan
        status = "Success"
        try:
            image = sitk.ReadImage(str(path))
            points = extract_landmark_centroids(image, [ANS_ID, PNS_ID, TUV_ID])
            missing_for_length = missing_landmark_names(points, [PNS_ID, TUV_ID])
            missing_for_height = missing_landmark_names(
                points, [ANS_ID, PNS_ID, TUV_ID]
            )

            if not missing_for_length:
                pns_to_tuv = points[TUV_ID] - points[PNS_ID]
                length_mm = float(np.linalg.norm(pns_to_tuv))

            if not missing_for_height:
                pns_to_ans = points[ANS_ID] - points[PNS_ID]
                pns_to_tuv = points[TUV_ID] - points[PNS_ID]
                base_length = float(np.linalg.norm(pns_to_ans))
                if base_length == 0:
                    raise ValueError("ANS and PNS are identical")
                # Preserve the old definition: distance from TUV to the
                # three-dimensional ANS-PNS line, not to an anatomical plane.
                height_mm = float(
                    np.linalg.norm(np.cross(pns_to_ans, pns_to_tuv)) / base_length
                )
            elif not missing_for_length:
                status = (
                    "Partial: height missing "
                    f"{','.join(missing_for_height)}"
                )
            else:
                status = f"Missing: {','.join(missing_for_length)}"
        except Exception as exc:
            status = f"Error: {exc}"
        rows.append(
            {
                "Patient_Name": case_id,
                "PNS_TUV_Length_mm": length_mm,
                "Soft_Palate_Height_mm": height_mm,
                "Status": status,
                "QC_Flags": qc_flags.get(case_id, ""),
            }
        )

    dataframe = pd.DataFrame(rows)
    write_csv_atomic(dataframe, args.output_csv, float_format="%.3f")
    valid_length = int(dataframe["PNS_TUV_Length_mm"].notna().sum())
    valid_height = int(dataframe["Soft_Palate_Height_mm"].notna().sum())
    print(
        f"PNS-TUV completed: length {valid_length}/{len(dataframe)}, "
        f"height {valid_height}/{len(dataframe)} valid cases"
    )
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
