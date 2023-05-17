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
            if task == "tile_regression" or task == "tile_classification" or task == "tile_segmentation":
                transforms.append(torchvision.transforms.RandomCrop(input_px))
        if task == "tile_segmentation":
            # pad to size divisible by 32 with -1
            if input_px % 32 != 0:
                pad = (32 - input_px % 32)//2
                transforms.append(torchvision.transforms.Pad(pad, fill=-1))
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
            weights = self.data[:, 1, :, :].float().mean(axis=(1,2))
            if weighted_sampler == "linear":
                self.weights = 1 - weights / weights.max() + 1
            if weighted_sampler == "exponential":
                self.weights = torch.exp(-2.5/weights.max() * weights)

        # self.data[:,:5] = torch.log(self.data[:,:5] + 10)

        if mean is None or std is None:
            self.mean = self.data[:,:5].float().mean(axis=(0,2,3))
            self.std = self.data[:,:5].float().std(axis=(0,2,3))

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
        else:
            target = sample[-1].clone()
            target[((target != 2) & (target != 4)) | (sample[-2] != 2)] = -1
            target[target == 2] = 0
            target[target == 4] = 1
            if self.task == "tile_regression":
                forest = torch.count_nonzero(target == 0)
                non_forest = torch.count_nonzero(target == 1)
                target = non_forest / (forest + non_forest)
            elif self.task == "tile_segmentation":
                target = target.unsqueeze(0)
            else:
                target = (torch.count_nonzero(target == 1) > 0)

        # target = target.int() if self.task == "tile_segmentation" else target.float()
        features = sample[:-1]

        return features.float(), target.float(), sample[-1].float()


if __name__ == "__main__":
    task = "pixel"
    input_px = 50
    train_dataset = DeforestationDataset("val", task=task, input_px=input_px, rebalanced=False)

    class_0 = 0
    class_1 = 0
    for i in range(len(train_dataset)):
        sample, target, last_layer = train_dataset[i]
        if task == "tile_segmentation":
            class_0 += torch.count_nonzero(target == 0)
            class_1 += torch.count_nonzero(target == 1)
        elif task == "tile_regression":
            class_1 += target
            class_0 += 1
        else:
            if target == 0:
                class_0 += 1
            else:
                class_1 += 1

    print(class_0)
    print(class_1)
    print(class_1/(class_0+class_1))

    for i in range(1000):
        sample, target, last_layer = train_dataset[i]

        if torch.count_nonzero(target == 1) > 0:
            # undo feature normalization
            for i in range(5):
                sample[i] = sample[i] * train_dataset.std[i] + train_dataset.mean[i]
            # sample[:5] = torch.exp(sample[:5]) - 10

            sample = torch.cat((sample, last_layer.unsqueeze(0)), dim=0)
            if train_dataset.task == "tile_segmentation":
                sample = torch.cat((sample, target), dim=0)
            else:
                segmentation_map = sample[-1].clone()
                segmentation_map[((segmentation_map != 2) & (segmentation_map != 4)) | (sample[-2] != 2)] = -1
                segmentation_map[segmentation_map == 2] = 0
                segmentation_map[segmentation_map == 4] = 1
                segmentation_map = segmentation_map.unsqueeze(0)
                sample = torch.cat((sample, segmentation_map), dim=0)

            green = "#115E05"
            red = "#F87F78"

            fig, axs = plt.subplots(1, 10, figsize=(54, 4))
            axs = axs.flatten()
            titles = ["1year", "5year", "10year", "urban", "slope", "landuse", "pasture", "current", "future", "target"]
            for idx, feature in enumerate(sample):
                if idx < 5:
                    mat = axs[idx].imshow(feature)
                    cax = plt.colorbar(mat, ax=axs[idx])
                elif idx == 5:
                    cmap = colors.ListedColormap(['#D5D5E5', '#129912', '#bbfcac', '#ffffb2', '#ea9999', '#0000ff'])
                    mat = axs[idx].imshow(feature, cmap=cmap, vmin=-0.5, vmax=5.5, interpolation='nearest')
                    cax = plt.colorbar(mat, ticks=np.arange(0, 6), ax=axs[idx])
                    cax.ax.set_yticklabels(['unknown', 'forest', 'natural', 'farming', 'urban', 'water'])
                elif idx == 6:
                    cmap = colors.ListedColormap(['#D5D5E5', '#930D03', '#FA9E4F', '#2466A7'])
                    mat = axs[idx].imshow(feature, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')
                    cax = plt.colorbar(mat, ticks=np.arange(0, 4), ax=axs[idx])
                    cax.ax.set_yticklabels(['unknown', 'severe', 'moderate', 'not. degraded'])
                elif idx == 9:
                    cmap = colors.ListedColormap(
                        ['#D5D5E5', green, red])
                    mat = axs[idx].imshow(feature, cmap=cmap, vmin=-1.5, vmax=1.5, interpolation='nearest')
                    cax = plt.colorbar(mat, ticks=np.arange(-1, 2), ax=axs[idx])
                    cax.ax.set_yticklabels(
                        ['unknown', 'primary', 'prim. defores.'])
                else:
                    cmap = colors.ListedColormap(
                        ['#D5D5E5', '#FFFCB5', green, '#409562', red, '#87FF0A', '#FD9407', '#191919'])
                    mat = axs[idx].imshow(feature, cmap=cmap, vmin=-0.5, vmax=7.5, interpolation='nearest')
                    cax = plt.colorbar(mat, ticks=np.arange(0, 8), ax=axs[idx])
                    cax.ax.set_yticklabels(['unknown', 'anthropic', 'primary', 'secondary', 'prim. defores.', 'regrowth',
                                            'sec. defores.', 'noise'])
                axs[idx].set_title(titles[idx])
                axs[idx].axis('off')
            plt.show()
            break

