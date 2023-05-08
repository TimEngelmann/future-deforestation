import torch
import torchvision
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import colors


class DeforestationDataset(Dataset):
    def __init__(self, dataset, max_elements=None, root_path="", mean=None, std=None,
                 weighted_sampler="", transform=None, input_px=50):
        """
        Args:
            dataset: "train", "val", "test"
        """
        self.dataset = dataset
        self.root_path = root_path
        self.mean = mean
        self.std = std

        self.transform = transform
        if transform is None:
           self.transform = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(input_px)
            ])

        self.data = torch.load(root_path + f"data/processed/{dataset}_layers.pt")
        if max_elements is not None:
            self.data = self.data[:max_elements]

        if weighted_sampler != "":
            weights = self.data[:, 1, :, :].mean(axis=(1,2))
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
        sample = self.transform(self.data[idx])
        features = sample[:-1]
        target = (torch.count_nonzero(sample[-1] == 4) > 0)

        return features.float(), target.float(), sample[-1].float()


if __name__ == "__main__":
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomCrop(50)
    ])

    train_dataset = DeforestationDataset("val", transform=train_transforms)

    for i in range(100):
        sample, target, last_layer = train_dataset[i]

        if target == 1:
            # undo feature normalization
            for i in range(5):
                sample[i] = sample[i] * train_dataset.std[i] + train_dataset.mean[i]
            sample[:5] = torch.exp(sample[:5]) - 10

            sample = torch.cat((sample, last_layer.unsqueeze(0)), dim=0)

            titles = ["1year", "5year", "10year", "urban", "slope", "landuse", "pasture", "current", "future"]
            for idx, feature in enumerate(sample):
                if idx < 5:
                    mat = plt.imshow(feature)
                    cax = plt.colorbar(mat)
                elif idx == 5:
                    cmap = colors.ListedColormap(['#D5D5E5', '#129912', '#bbfcac', '#ffffb2', '#ea9999', '#0000ff'])
                    mat = plt.imshow(feature, cmap=cmap, vmin=-0.5, vmax=5.5, interpolation='nearest')
                    cax = plt.colorbar(mat, ticks=np.arange(0, 6))
                    cax.ax.set_yticklabels(['unknown', 'forest', 'natural', 'farming', 'urban', 'water'])
                elif idx == 6:
                    cmap = colors.ListedColormap(['#D5D5E5', '#930D03', '#FA9E4F', '#2466A7'])
                    mat = plt.imshow(feature, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')
                    cax = plt.colorbar(mat, ticks=np.arange(0, 4))
                    cax.ax.set_yticklabels(['unknown', 'severe', 'moderate', 'not. degraded'])
                else:
                    cmap = colors.ListedColormap(
                        ['#D5D5E5', '#FFFCB5', '#0F5018', '#409562', '#D90016', '#87FF0A', '#FD9407', '#191919'])
                    mat = plt.imshow(feature, cmap=cmap, vmin=-0.5, vmax=7.5, interpolation='nearest')
                    cax = plt.colorbar(mat, ticks=np.arange(0, 8))
                    cax.ax.set_yticklabels(['unknown', 'anthropic', 'primary', 'secondary', 'prim. deforestation', 'regrowth',
                                            'sec. deforestation', 'noise'])
                plt.title(titles[idx])
                plt.show()
            break