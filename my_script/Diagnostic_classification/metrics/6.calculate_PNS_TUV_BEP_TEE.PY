"""Calculate the historical PNS-TUV-BEP-TEE two-triangle area."""

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
OUTPUT_CSV = SCRIPT_DIR / "6.pns_tuv_bep_tee_tissue_area.csv"
PNS_ID = 4
TUV_ID = 7
BEP_ID = 3
TEE_ID = 5
REQUIRED_IDS = [PNS_ID, TUV_ID, BEP_ID, TEE_ID]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the PNS-TUV-BEP-TEE two-triangle area (mm^2)."
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
    for case_id, path in tqdm(cases.items(), desc="6.Four-point area"):
        area_mm2 = np.nan
        status = "Success"
        try:
            image = sitk.ReadImage(str(path))
            points = extract_landmark_centroids(image, REQUIRED_IDS)
            missing = missing_landmark_names(points, REQUIRED_IDS)
            if missing:
                status = f"Missing: {','.join(missing)}"
            else:
                p_pns = points[PNS_ID]
                p_tuv = points[TUV_ID]
                p_bep = points[BEP_ID]
                p_tee = points[TEE_ID]

                # Preserve the old definition and diagonal: the sum of the
                # 3D triangle areas PNS-TUV-BEP and PNS-BEP-TEE.
                area1 = 0.5 * np.linalg.norm(
                    np.cross(p_tuv - p_pns, p_bep - p_pns)
                )
                area2 = 0.5 * np.linalg.norm(
                    np.cross(p_bep - p_pns, p_tee - p_pns)
                )
                area_mm2 = float(area1 + area2)
        except Exception as exc:
            status = f"Error: {exc}"
        rows.append(
            {
                "Patient_Name": case_id,
                "Pharyngeal_Area_mm2": area_mm2,
                "Status": status,
                "QC_Flags": qc_flags.get(case_id, ""),
            }
        )

    dataframe = pd.DataFrame(rows)
    write_csv_atomic(dataframe, args.output_csv, float_format="%.3f")
    valid = int(dataframe["Pharyngeal_Area_mm2"].notna().sum())
    print(f"Four-point area completed: {valid}/{len(dataframe)} valid cases")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()

