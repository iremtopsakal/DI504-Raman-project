import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from augment import apply_augmentation

"""
This file defines `RamanSpectraDataset` for loading Raman spectroscopy data
for deep learning.

1. Dataset Structure:
   - Assumes a directory structure where subfolders are named like '5', '6', etc., corresponding to
     concentrations in scientific notation (e.g., folder '5' -> concentration 1e-5)
   - Each folder contains `.txt` files with Raman spectra data in two columns: Raman shift (ignored)
     and intensity (used as input)

2. Label Encoding:
   - The label for each sample is the base-10 logarithm of the concentration, `log10(1e-5) = -5.0`

3. Augmentation Handling

4. Subset slection:
   - `subset="raw"`: Only loads raw entries (without augmentation)
   - `subset="aug"`: Only augmented duplicates (for use with offline augmentation)
   - `subset="all"`: Both raw and augmented entries.

5. Returns
6. Augmentation
"""


class RamanSpectraDataset(Dataset):
    def __init__(self, root_dir, augment=False, offline_aug=False, num_aug=2, subset="all"):
        assert subset in ["raw", "aug", "all"], "subset must be 'raw', 'aug', or 'all'"
        self.samples = []
        self.augment = augment
        self.offline_aug = offline_aug
        self.num_aug = num_aug
        self.subset = subset

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                conc = float(f"1e-{folder}")
                label = np.log10(conc)  # e.g., -5.0, -6.0
            except ValueError:
                continue

            for fname in os.listdir(folder_path):
                if fname.endswith('.txt') and "_aug" not in fname:
                    fpath = os.path.join(folder_path, fname)

                    if self.subset in ["raw", "all"]:
                        self.samples.append((fpath, label))

                    if self.offline_aug and self.subset in ["aug", "all"]:
                        for i in range(self.num_aug):
                            self.samples.append((fpath, label, True))  # Marked for augmentation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        if len(entry) == 3:
            fpath, label, to_augment = entry
        else:
            fpath, label = entry
            to_augment = False

        data = np.loadtxt(fpath)
        intensities = data[:, 1]

        intensities = intensities
        if self.augment or to_augment:
            intensities = apply_augmentation(intensities, force=to_augment)

        x = torch.tensor(intensities[np.newaxis, :], dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y
