import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
"""
This script provides data augmentation tools for Raman spectra.

1. Augmentation functions: 
    - shift, scale, and noise to simulate spectral variation.
2. apply_augmentation: 
    - Combines all augmentations with optional randomness.
3. generate_augmented_files: 
    - Saves new augmented .txt files when run directly.
4.  AugmentedWrapper:  
    - Dataset wrapper for augmentation during training.
"""

def shift_spectrum(intensities, max_shift=3):
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return intensities
    elif shift > 0:
        return np.pad(intensities[:-shift], (shift, 0), mode='edge')
    else:
        return np.pad(intensities[-shift:], (0, -shift), mode='edge')

def scale_intensity(intensities, scale_range=(0.95, 1.05)):
    scale = np.random.uniform(*scale_range)
    return intensities * scale

def add_noise(intensities, noise_level=0.005):
    noise = np.random.normal(0, noise_level, size=intensities.shape)
    return intensities + noise

def apply_augmentation(intensities, force=False, augment_prob=0.5):
    if force or np.random.rand() < augment_prob:
        intensities = shift_spectrum(intensities)
        intensities = scale_intensity(intensities)
        intensities = add_noise(intensities)
    return intensities

# === Offline Augmentation: Save New Files ===

def generate_augmented_files(input_root="Data/ASL baseline corrected merged", num_augmentations=2):
    for folder in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if fname.endswith('.txt') and "_aug" not in fname:
                original_path = os.path.join(folder_path, fname)
                try:
                    data = np.loadtxt(original_path)
                    x_values = data[:, 0]
                    y_values = data[:, 1]

                    for i in range(num_augmentations):
                        augmented_y = apply_augmentation(y_values)
                        augmented_data = np.column_stack((x_values, augmented_y))

                        out_name = fname.replace(".txt", f"_aug{i+1}.txt")
                        out_path = os.path.join(folder_path, out_name)
                        np.savetxt(out_path, augmented_data, fmt="%.6f")
                        print(f"Saved: {out_path}")
                except Exception as e:
                    print(f"Failed to process {original_path}: {e}")

# === Run offline augmentation only when executed directly ===

if __name__ == "__main__":
    generate_augmented_files()


class AugmentedWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, num_aug=1):
        self.base_dataset = base_dataset
        self.num_aug = num_aug

    def __len__(self):
        return len(self.base_dataset) * (self.num_aug + 1)

    def __getitem__(self, idx):
        base_len = len(self.base_dataset)
        base_idx = idx % base_len
        x, y = self.base_dataset[base_idx]

        if idx < base_len:
            return x, y  # Return original data
        else:
            x_aug = apply_augmentation(x.squeeze().numpy(), force=True)
            x_aug = (x_aug - x_aug.mean()) / (x_aug.std() + 1e-8)
            x = torch.tensor(x_aug[np.newaxis, :], dtype=torch.float32)
            return x, y