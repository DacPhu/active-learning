import os
from glob import glob

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    def __init__(self, root_dir, files=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        all_files = sorted(files if files else glob(os.path.join(root_dir, "*.h5")))
        self.files = [f for f in all_files if self._has_keys(f, ['image', 'label'])]

    def _has_keys(self, filepath, required_keys):
        try:
            with h5py.File(filepath, 'r') as f:
                return all(key in f for key in required_keys)
        except OSError as e:
            print(f"Error opening file {filepath}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error with file {filepath}: {e}")
            return False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_path = self.files[idx]
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][()]
            mask = f['label'][()]

        # Normalize the image to the range [0, 1]
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32)

        # Ensure that images and masks have a channel dimension (C, H, W)
        if image.ndim == 2:  # Grayscale image (H, W)
            image = image.unsqueeze(0)  # Add a channel dimension (1, H, W)
        if mask.ndim == 2:  # Grayscale mask (H, W)
            mask = mask.unsqueeze(0)  # Add a channel dimension (1, H, W)

        return image, mask


class ACDCSliceDataset(Dataset):
    def __init__(self, root_dir, files=None, target_size=(256, 256)):
        self.files = sorted(files if files else glob(os.path.join(root_dir, "*.h5")))
        self.target_size = target_size

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            image = torch.tensor(f["image"][:], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            mask = torch.tensor(f["label"][:], dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # Get current image and mask dimensions
        current_height, current_width = image.shape[1], image.shape[2]

        # Calculate padding amounts
        pad_height = self.target_size[0] - current_height
        pad_width = self.target_size[1] - current_width

        # Apply padding to the image and mask (padding is applied to the right and bottom)
        pad = (0, pad_width, 0, pad_height)  # (left, right, top, bottom)
        image = F.pad(image, pad, value=0)  # Padding the image
        mask = F.pad(mask, pad, value=0)  # Padding the mask

        return image, mask

    def __len__(self):
        return len(self.files)


def get_dataset_splits(dataset, initial_labeled=10):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    labeled = indices[:initial_labeled]
    unlabeled = indices[initial_labeled:]
    return labeled, unlabeled
