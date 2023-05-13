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
                 weighted_sampler="", input_px=50, task="pixel", rebalanced=False):
        """
        Args:
            dataset: "train", "val", "test"
        """
        self.dataset = dataset
        self.task = task
        self.input_px = input_px
        self.root_path = root_path
        self.mean = mean
        self.std = std

        transforms = []
        if self.dataset == "val":
            transforms.append(torchvision.transforms.CenterCrop(input_px))
        else:
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
            transforms.append(torchvision.transforms.RandomVerticalFlip())
            if task == "tile_regression" or task == "tile_classification":
                transforms.append(torchvision.transforms.RandomCrop(input_px))
        self.transform = torchvision.transforms.Compose(transforms)

        self.data = torch.load(root_path + f"data/processed/{dataset}_layers.pt")
        if max_elements is not None:
            self.data = self.data[:max_elements]

        if rebalanced:
            targets = self.data[:, -1, self.data.shape[2] // 2, self.data.shape[3] // 2]
            majority = self.data[targets == 2]
            minority = self.data[targets == 4]
            self.data = torch.cat([majority[:int(4*len(minority))], minority])
            self.data = self.data[torch.randperm(len(self.data))]

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

        if self.task == "pixel":
            i = self.input_px // 2
            mid = sample.shape[1] // 2

            x = mid
            y = mid

            if self.dataset == "train":
                # valid_pixels = (sample[-2] == 2) & ((sample[-1] == 4) | (sample[-1] == 2))
                valid_pixels = (sample[-1] == sample[-1, mid, mid])
                indices = valid_pixels.nonzero()
                indices = indices[(indices[:,0] > i) & (indices[:,1] > i)
                                   & (indices[:, 0] < sample.shape[1] - i) & (indices[:,1] < sample.shape[2] - i)]
                selected = np.random.randint(indices.shape[0])
                y = indices[selected, 0]
                x = indices[selected, 1]

            odd = self.input_px % 2
            sample = sample[:, y-i:y+i+odd, x-i:x+i+odd]
            target = sample[-1, i, i] == 4
        elif self.task == "tile_regression":
            target = torch.count_nonzero(sample[-1] == 4)
        else:
            target = (torch.count_nonzero(sample[-1] == 4) > 0)

        features = sample[:-1]

        return features.float(), target.float(), sample[-1].float()


if __name__ == "__main__":
    input_px_array = [50]
    nr_target = []
    for input_px in input_px_array:
        train_dataset = DeforestationDataset("train", task="tile_classification", input_px=input_px, rebalanced=False)

        class_0 = 0
        class_1 = 0
        for i in range(len(train_dataset)):
            sample, target, last_layer = train_dataset[i]
            if target == 0:
                class_0 += 1
            else:
                class_1 += 1

        print(class_0)
        print(class_1)
        nr_target.append(class_1/(class_0+class_1))

    plt.plot(input_px_array, nr_target)
    plt.show()

    '''
    for i in range(1000):
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
    '''

