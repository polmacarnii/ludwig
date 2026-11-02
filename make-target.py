# called by prep-dataset.sh*
import numpy as np
from pathlib import Path

OG_DIR = Path("dataset/original")
FILTER_DIR = Path("dataset/filter")
TARGET_DIR = Path("dataset/target") # * : No need to make dir, already made at the point when its called

for og_path in sorted(OG_DIR.glob("*.raw")):
    filter_path = FILTER_DIR / og_path.name
    target_path = TARGET_DIR / og_path.name

    if not filter_path.exists():
        print(f"Skipping {filter_path} : missing filter")
        continue

    original = np.fromfile(og_path, dtype="<i2").astype(np.float32)
    filtered = np.fromfile(filter_path, dtype="<i2").astype(np.float32)

    if len(original) != len(filtered):
        print(f"Mismatched lenght on {og_path.name}")
        continue

    target = original - filtered
    target.tofile(target_path)
