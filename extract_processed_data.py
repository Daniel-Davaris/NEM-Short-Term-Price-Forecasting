"""
Extracts all .csv.gz files in Dataset/Processed_data/ back to .csv using gzip.
Original .csv.gz files are kept intact.
"""

import gzip
import shutil
from pathlib import Path

PROCESSED_DATA_DIR = Path.cwd() / "1_Dataset" / "Processed_data"


def extract_csvs(directory: Path) -> None:
    gz_files = sorted(directory.glob("*.csv.gz"))
    if not gz_files:
        print("No .csv.gz files found.")
        return

    for gz_path in gz_files:
        csv_path = gz_path.with_suffix("")  # Removes .gz, leaving .csv
        print(f"Extracting {gz_path.name} -> {csv_path.name} ...", end=" ")
        try:        
            with gzip.open(gz_path, "rb") as f_in, csv_path.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print("done")
        except OSError:
            print(f"Skipping {gz_path.name}: not a valid gzip file.")


if __name__ == "__main__":
    extract_csvs(PROCESSED_DATA_DIR)