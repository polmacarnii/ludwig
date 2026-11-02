import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

CHUNK = 2048 # This needs to be the same as lsd/config.h

def find_pairs(filter_dir, target_dir):
    # This is for matching every .raw file from filter_dir with its couple on target_dir
    filter_dir = str(filter_dir)
    target_dir = str(target_dir)

    pairs = []

    for path in sorted(glob.glob(f"{filter_dir}/*.raw")):
        name = os.path.basename(path)
        target = f"{target_dir}/{name}"

        with open(target, "rb"):
            pass

        pairs.append((path, target))

    return pairs


def split_pairs(pairs, val_fraction=0.1, seed=42):
    # Splits pairs into train and validation by song
    # Never chunk
    shuffled = list(pairs)

    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_fraction))

    return shuffled[val_count:], shuffled[:val_count]


class LudwigDataset(Dataset):
    # Yield (input, target) chunks of CHUNK samples, normalized into [-1, 1]

    def __init__(self, pairs, chunk=CHUNK):
        self.chunk = chunk
        self.items = []

        for filter_path, target_path in pairs:
            # Two bytes for int16 input sample, four bytes for float32 target sample
            # both hold the same number of samples.
            filter_bytes = os.path.getsize(filter_path)
            target_bytes = os.path.getsize(target_path)

            samples = min(filter_bytes // 2, target_bytes // 4)

            for start in range(0, samples - chunk + 1, chunk):
                self.items.append((filter_path, target_path, start))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        filter_path, target_path, start = self.items[index]

        # read only that window instead of the whole song, because a whole song would eat memory
        with open(filter_path, "rb") as f:
            f.seek(start * 2)
            raw = np.fromfile(f, dtype="<i2", count=self.chunk)

        with open(target_path, "rb") as f:
            f.seek(start * 4)
            target = np.fromfile(f, dtype="<f4", count=self.chunk)

        x = torch.from_numpy(raw.astype(np.float32) / 32768.0)
        y = torch.from_numpy(target.astype(np.float32) / 32768.0)

        return x.unsqueeze(0), y.unsqueeze(0)
