import os
import numpy as np

# === Augmentation Functions ===

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

def apply_augmentation(intensities, augment_prob=0.5):
    """Used by dataset: apply live/random augmentation in RAM."""
    if np.random.rand() < augment_prob:
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