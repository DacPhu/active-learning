import os
from glob import glob

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from models.unet import UNet


# Dataset class for 2D slices
class ACDCSliceDataset(Dataset):
    def __init__(self, root_dir, files=None):
        self.files = sorted(files if files else glob(os.path.join(root_dir, "*.h5")))

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            image = torch.tensor(f["image"][:], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            mask = torch.tensor(f["label"][:], dtype=torch.float32).unsqueeze(0)    # (1, H, W)
        return image, mask

    def __len__(self):
        return len(self.files)

def compute_uncertainty(model, dataloader, device):
    model.eval()
    uncertainties = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            entropy = -outputs * torch.log(outputs + 1e-6) - (1 - outputs) * torch.log(1 - outputs + 1e-6)
            uncertainties.append(entropy.mean(dim=(1, 2, 3)).cpu())

    return torch.cat(uncertainties)

def train(model, dataloader, device):
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def get_dataset_splits(num_samples, initial_labeled=20):
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    labeled = indices[:initial_labeled]
    unlabeled = indices[initial_labeled:]
    return labeled.tolist(), unlabeled.tolist()

def active_learning(data_dir, rounds=5, query_size=10, initial_labeled=20, batch_size=4):
    all_h5_files = sorted(glob(os.path.join(data_dir, "*.h5")))
    print(f"Total number of files: {len(all_h5_files)}")

    dataset = ACDCSliceDataset(root_dir=data_dir, files=all_h5_files)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)

    labeled_idx, unlabeled_idx = get_dataset_splits(len(dataset), initial_labeled)

    for round in range(rounds):
        print(f"\n== Round {round + 1} ==")

        # Train on labeled data
        labeled_set = Subset(dataset, labeled_idx)
        train_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
        train(model, train_loader, device)

        # Compute uncertainties on unlabeled pool
        pool_set = Subset(dataset, unlabeled_idx)
        pool_loader = DataLoader(pool_set, batch_size=batch_size)
        uncertainties = compute_uncertainty(model, pool_loader, device)

        # Pick most uncertain samples
        top_uncertain = torch.topk(uncertainties, query_size).indices.numpy()
        query_indices = np.array(unlabeled_idx)[top_uncertain]

        # Update sets
        labeled_idx += query_indices.tolist()
        unlabeled_idx = np.delete(unlabeled_idx, top_uncertain).tolist()

        print(f"Labeled: {len(labeled_idx)}, Unlabeled: {len(unlabeled_idx)}")

    return model

def main():
    data_path = "assets/datasets/ACDC_preprocessed/ACDC_training_slices"
    model = active_learning(data_path, rounds=5, query_size=10, initial_labeled=20, batch_size=4)

if __name__ == '__main__':
    main()
