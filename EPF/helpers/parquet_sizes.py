import os
from pathlib import Path

base = Path(__file__).parent
files = sorted(base.rglob("*.parquet"), key=lambda f: f.stat().st_size, reverse=True)

total_bytes = sum(f.stat().st_size for f in files)

print(f"{'Size (MB)':<12} {'File'}")
print("-" * 80)
for f in files:
    size_mb = round(f.stat().st_size / 1_048_576, 2)
    rel = f.relative_to(base)
    print(f"{size_mb:<12} {rel}")

print("-" * 80)
print(f"Total: {round(total_bytes / 1_073_741_824, 2)} GB across {len(files)} files")
