import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from augment import apply_augmentation


# === Dataset Definition ===
class RamanSpectraDataset(Dataset):
    def __init__(self, root_dir, augment=False, offline_aug=False, num_aug=2):
        self.samples = []
        self.augment = augment
        self.offline_aug = offline_aug
        self.num_aug = num_aug

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                conc = float(f"1e-{folder}")
                label = np.log10(conc)  # e.g., -5.0, -6.0
            except ValueError:
                continue

            #for fname in os.listdir(folder_path):
                if fname.endswith('.txt'):
                    fpath = os.path.join(folder_path, fname)
                    self.samples.append((fpath, label))
            
            for fname in os.listdir(folder_path):
                if fname.endswith('.txt'):
                    fpath = os.path.join(folder_path, fname)
                    self.samples.append((fpath, label))

                if self.offline_aug:
                    for i in range(self.num_aug):
                        self.samples.append((fpath, label, True))  # Third item marks "needs augmentation")

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

        if self.augment or to_augment:
            intensities = apply_augmentation(intensities)

        # Normalize
        intensities = (intensities - intensities.mean()) / (intensities.std() + 1e-8)
        x = torch.tensor(intensities[np.newaxis, :], dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y
    
    #def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        data = np.loadtxt(fpath)
        intensities = data[:, 1]

        if self.augment:
            intensities = apply_augmentation(intensities)

        intensities = (intensities - intensities.mean()) / (intensities.std() + 1e-8)
        x = torch.tensor(intensities[np.newaxis, :], dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y




    #def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        data = np.loadtxt(fpath)
        intensities = data[:, 1]
        intensities = (intensities - intensities.mean()) / (intensities.std() + 1e-8)
        x = torch.tensor(intensities[np.newaxis, :], dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

# === Load & Split Dataset ===
def get_loaders(root_path, batch_size=32):
    dataset = RamanSpectraDataset(root_path)

    train_size = int(0.8 * len(dataset))
    val_size   = int(0.1 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader
