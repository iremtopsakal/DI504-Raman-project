import numpy as np

def shift_spectrum(intensities, max_shift=3):
    """Randomly shift the spectrum left/right by a few indices."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return intensities
    elif shift > 0:
        return np.pad(intensities[:-shift], (shift, 0), mode='edge')
    else:
        return np.pad(intensities[-shift:], (0, -shift), mode='edge')

def scale_intensity(intensities, scale_range=(0.95, 1.05)):
    """Randomly scale the entire spectrum intensity."""
    scale = np.random.uniform(*scale_range)
    return intensities * scale

def add_noise(intensities, noise_level=0.005):
    """Add small Gaussian noise."""
    noise = np.random.normal(0, noise_level, size=intensities.shape)
    return intensities + noise

def apply_augmentation(intensities, augment_prob=0.5):
    """Apply a random combination of augmentations with given probability."""
    if np.random.rand() < augment_prob:
        intensities = shift_spectrum(intensities)
        intensities = scale_intensity(intensities)
        intensities = add_noise(intensities)
    return intensities