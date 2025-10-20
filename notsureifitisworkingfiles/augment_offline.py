import os
import numpy as np
from augment import apply_augmentation

input_root = "Data/ASL baseline corrected merged"
num_augmentations = 2  # how many augmented copies to make per original file

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
