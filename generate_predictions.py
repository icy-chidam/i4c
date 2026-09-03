"""
Generate Phase-2 predictions.csv from the local data/test dataset.

Project layout expected:

phase2_repo/
├── data/
│   └── test/
│       ├── references/
│       │   ├── test_00000.png
│       │   └── ...
│       └── searches/
│           ├── test_00000.png
│           └── ...
├── register.py
├── weights/
│   ├── cnn_matcher_p2.npz
│   └── confidence_model_p2.joblib
└── generate_predictions.py

The generated temporary pairs file is compatible with register.py,
and register.py produces the official:

pair_id,x,y,theta,scale,found,score
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

REFERENCE_DIR = REPO_ROOT / "data" / "test" / "references"
SEARCH_DIR = REPO_ROOT / "data" / "test" / "searches"

TEMP_PAIRS = REPO_ROOT / "pairs_test.csv"
OUTPUT_CSV = REPO_ROOT / "predictions.csv"

REGISTER_SCRIPT = REPO_ROOT / "register.py"


# ------------------------------------------------------------
# Build temporary pairs.csv
# ------------------------------------------------------------

def build_pairs_csv() -> int:
    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Reference directory not found:\n{REFERENCE_DIR}"
        )

    if not SEARCH_DIR.exists():
        raise FileNotFoundError(
            f"Search directory not found:\n{SEARCH_DIR}"
        )

    reference_files = sorted(REFERENCE_DIR.glob("*.png"))

    if not reference_files:
        raise RuntimeError(
            f"No PNG reference images found in:\n{REFERENCE_DIR}"
        )

    rows = []

    for ref_path in reference_files:
        pair_id = ref_path.stem
        search_path = SEARCH_DIR / ref_path.name

        if not search_path.exists():
            print(
                f"[WARNING] Missing matching search image for {ref_path.name}",
                file=sys.stderr
            )
            continue

        # Absolute paths avoid any ambiguity in register.py
        rows.append({
            "pair_id": pair_id,
            "search_path": str(search_path.resolve()),
            "reference_path": str(ref_path.resolve()),
        })

    if not rows:
        raise RuntimeError(
            "No valid reference/search pairs were found."
        )

    with TEMP_PAIRS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "search_path", "reference_path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Created {TEMP_PAIRS}")
    print(f"[INFO] Number of pairs: {len(rows)}")

    return len(rows)


# ------------------------------------------------------------
# Run official register.py
# ------------------------------------------------------------

def run_register():
    command = [
        sys.executable,
        str(REGISTER_SCRIPT),
        "--input",
        str(TEMP_PAIRS),
        "--output",
        str(OUTPUT_CSV),
    ]

    print("\n[INFO] Running register.py:")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))
    print()

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"register.py failed with exit code {result.returncode}"
        )


# ------------------------------------------------------------
# Validate predictions.csv
# ------------------------------------------------------------

def validate_output(expected_count: int):
    if not OUTPUT_CSV.exists():
        raise RuntimeError(
            f"register.py did not create:\n{OUTPUT_CSV}"
        )

    with OUTPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_columns = [
        "pair_id",
        "x",
        "y",
        "theta",
        "scale",
        "found",
        "score",
    ]

    if reader.fieldnames != expected_columns:
        raise RuntimeError(
            "\nUnexpected CSV columns.\n"
            f"Expected: {expected_columns}\n"
            f"Got:      {reader.fieldnames}"
        )

    if len(rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} prediction rows, "
            f"but found {len(rows)}."
        )

    pair_ids = [row["pair_id"] for row in rows]

    if len(set(pair_ids)) != len(pair_ids):
        raise RuntimeError(
            "Duplicate pair_id values detected in predictions.csv."
        )

    for row in rows:
        found = row["found"]

        if found not in {"0", "1"}:
            raise RuntimeError(
                f"Invalid found value for {row['pair_id']}: {found}"
            )

        # Competition contract:
        # when found=0, pose values must be zero
        if found == "0":
            for field in ["x", "y", "theta", "scale"]:
                try:
                    value = float(row[field])
                except ValueError:
                    raise RuntimeError(
                        f"Invalid numeric value in {field} "
                        f"for {row['pair_id']}"
                    )

                if value != 0.0:
                    raise RuntimeError(
                        f"{row['pair_id']} has found=0 but "
                        f"{field}={value}; expected 0."
                    )

    print("\n[OK] predictions.csv validation passed.")
    print(f"[OK] Rows: {len(rows)}")
    print(f"[OK] Columns: {reader.fieldnames}")
    print(f"[OK] Output: {OUTPUT_CSV}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 70)
    print("Drift-Sense Phase-2 predictions.csv generator")
    print("=" * 70)

    count = build_pairs_csv()
    run_register()
    validate_output(count)

    print("\nDone.")
    print(f"\npredictions.csv:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()