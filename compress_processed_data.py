"""
Compresses all CSV files in Dataset/Processed_data/ to .csv.gz using gzip.
Original CSV files are kept intact.
"""

import gzip
import shutil
from pathlib import Path

PROCESSED_DATA_DIR = Path(__file__).parent / "Dataset" / "Processed_data"


def compress_csvs(directory: Path) -> None:
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    for csv_path in csv_files:
        gz_path = csv_path.with_suffix(".csv.gz")
        print(f"Compressing {csv_path.name} -> {gz_path.name} ...", end=" ")
        with csv_path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("done")


if __name__ == "__main__":
    compress_csvs(PROCESSED_DATA_DIR)
