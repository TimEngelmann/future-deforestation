import torch
import torchvision
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import numpy as np
import sys

class DeforestationDataset(Dataset):
    def __init__(self, dataset, resolution=30, input_px=35, output_px=1, max_elements=None, root_path="",
                 mean=None, std=None, weighted_sampler=""):
        """
        Args:
            dataset: "train", "val", "test"
        """
        self.dataset = dataset
        self.year = 2014 if dataset == "test" else 2009
        self.past_horizons = [1,5,10]
        self.future_horizon = 1
        self.resolution = resolution
        self.input_px = input_px
        self.root_path = root_path
        self.mean = mean
        self.std = std

        self.data = torch.load(root_path + f"data/processed/{dataset}_layers.pt")
        if max_elements is not None:
            self.data = self.data[:max_elements]

        # helpers
        self.i = torch.tensor(int(input_px / 2))
        self.delta = self.data.shape[-1]
        self.mid = int(self.delta / 2)
        self.augmentation = int((self.delta - input_px) / 2)

        if weighted_sampler != "":
            weights = self.data[:, 1, self.mid, self.mid].clone()
            if weighted_sampler == "linear":
                self.weights = 1 - weights / weights.max() + 1
            if weighted_sampler == "exponential":
                self.weights = torch.exp(-2.5/weights.max() * weights)

        self.data[:,:5] = torch.log(self.data[:,:5] + 10)

        if mean is None or std is None:
            self.mean = self.data[:,:5].mean(axis=(0,2,3))
            self.std = self.data[:,:5].std(axis=(0,2,3))

        for i in range(5):
            self.data[:,i] = (self.data[:,i] - self.mean[i]) / self.std[i]
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        x = self.mid
        y = self.mid

        if self.dataset == "train":
            xi = np.arange(x-self.augmentation, x+self.augmentation+1)
            xvi, yvi = np.meshgrid(xi, xi)
            current = self.data[idx, -2, y-self.augmentation:y+self.augmentation+1, x-self.augmentation:x+self.augmentation+1]
            targets = self.data[idx, -1, y-self.augmentation:y+self.augmentation+1, x-self.augmentation:x+self.augmentation+1]
            target_points = np.vstack((xvi.flatten(), yvi.flatten(), targets.flatten(), current.flatten())).T
            target_points = target_points[((target_points[:,3] == 2) & (target_points[:,2] == 2) | (target_points[:,2] == 4))]

            # select random point from target points
            target_point = target_points[torch.randint(0, target_points.shape[0],(1,)).item()]
            x = int(target_point[0])
            y = int(target_point[1])
        
        features = self.data[idx, :-1, x-self.i:x+self.i+1, y-self.i:y+self.i+1]
        target = self.data[idx, -1, x, y].item()
        target = 1 if target == 4 else 0
        
        if self.dataset == "train":
            angles = [0,90,180,270]
            angle_idx = torch.randint(0, len(angles),(1,)).item()
            features = torchvision.transforms.functional.rotate(features, angles[angle_idx])

        return features.float(), torch.tensor(target).float()
    
if __name__ == "__main__":
    train_dataset = DeforestationDataset("train")
    features, target = train_dataset[20]