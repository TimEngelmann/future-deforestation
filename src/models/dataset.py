import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset
class DeforestationDataset(Dataset):
    def __init__(self, dataset, max_elements=None, start_year=2006, nr_years_train=10, horizon=3, resolution=30, patch_size=35):
        """
        Args:
            dataset: "train", "val", "test"
        """
        self.dataset = dataset
        self.start_year = start_year
        self.nr_years_train = nr_years_train
        self.horizon = horizon
        self.resolution = resolution
        self.patch_size = patch_size
        
        # dataformat: [x_idx, y_idx, deforested, 2016, 2017, ..., 2021]
        data = torch.load(f'data/processed/{dataset}_data.pt')
        self.data = data[:max_elements] if max_elements is not None else data

        self.bio_data = {}
        path_bio_processed = f"data/processed/biomass/{resolution}m/"
        for year in np.arange(self.start_year, self.start_year + self.nr_years_train + 2 * horizon):
            self.bio_data[year] = torch.load(path_bio_processed + f"biomass_{year}.pt")
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.dataset == "test":
            # labels = self.data[idx, 3+self.horizon:]
            labels = self.data[idx, 3+self.horizon]
            years = np.arange(self.start_year + self.horizon, self.start_year + self.horizon + self.nr_years_train)
        else:
            # labels = self.data[idx, 3:3+self.horizon]
            labels = self.data[idx, 3]
            years = np.arange(self.start_year, self.start_year + self.nr_years_train)

        x_idx = self.data[idx, 0]
        y_idx = self.data[idx, 1]
        r = int(self.patch_size/2)

        features = []
        for year in years:
            window = self.bio_data[year][y_idx-r:y_idx+r, x_idx-r:x_idx+r]
            features.append(window)
        features = torch.stack(features)

        return features.float(), labels.float()