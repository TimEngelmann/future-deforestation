import numpy as np
import torch
import os
import rasterio
from torch.utils.data import Dataset

# function to load biomass data
def load_processed_biomass_data(year, shape=None, resolution=250, delta=None):
    path_bio = f"data/processed/biomass/amazonia/{resolution}m/" + f"biomass_{year}.tif"
    with rasterio.open(path_bio) as src:
        out_meta = src.meta
        if shape is not None:
            nodata = 255
            bio_data, out_transform = rasterio.mask.mask(src, shape, crop=True, nodata=nodata)
            bio_data = bio_data.squeeze()
            out_meta.update({"driver": "GTiff",
                 "height": bio_data.shape[0],
                 "width": bio_data.shape[1],
                 "transform": out_transform})
        else:
            bio_data = src.read(1)
    bio_data = torch.from_numpy(bio_data)
    if delta is not None:
        diff_x = int((bio_data.shape[1]%delta)/2 + delta)
        diff_y = int((bio_data.shape[0]%delta)/2 + delta)
        bio_data = torch.nn.functional.pad(bio_data, (diff_y, diff_y, diff_x, diff_x), value=255)
    return bio_data, out_meta

# function to load transition data
def load_processed_transition_data(year_start, year_end, shape=None, resolution=250, delta=None):
    path_bio = f"data/processed/biomass/amazonia/{resolution}m/" 
    path_bio = path_bio + f"transition_{year_start}_{year_end}.tif"
    with rasterio.open(path_bio) as src:
        out_meta = src.meta
        if shape is not None:
            nodata = 255
            transition_data, out_transform = rasterio.mask.mask(src, shape, crop=True, nodata=nodata)
            transition_data = transition_data.squeeze().astype(bool)
            out_meta.update({"driver": "GTiff",
                 "height": transition_data.shape[0],
                 "width": transition_data.shape[1],
                 "transform": out_transform})
        else:
            transition_data = src.read(1).astype(bool)
    transition_data = torch.from_numpy(transition_data)
    if delta is not None:
        diff_x = int((transition_data.shape[1]%delta)/2 + delta)
        diff_y = int((transition_data.shape[0]%delta)/2 + delta)
        transition_data = torch.nn.functional.pad(transition_data, (diff_y, diff_y, diff_x, diff_x), value=False)
    return transition_data, out_meta


class DeforestationDataset(Dataset):
    def __init__(self, dataset, resolution=250, input_px=400, output_px=40, delta=360, max_elements=None):
        """
        Args:
            dataset: "train", "val", "test"
        """
        self.dataset = dataset
        self.year = 2015 if dataset == "test" else 2010
        self.past_horizons = [1,5,10]
        self.future_horizon = 5
        self.resolution = resolution
        self.input_px = input_px
        self.output_px = output_px
        self.delta = delta
        
        # dataformat: [x_point, y_point, x_cluster, y_cluster]
        data = torch.load(f'data/processed/{dataset}_data.pt')
        self.data = data
        if max_elements is not None:
            if len(data) >= max_elements:
                self.data = data[:max_elements]

        feature_data = []
        bio_data, _ = load_processed_biomass_data(self.year, resolution=self.resolution, delta=self.delta)
        feature_data.append(bio_data)
    
        for past_horizon in self.past_horizons:
            transition_data, _ = load_processed_transition_data(self.year - past_horizon, self.year, resolution=self.resolution, delta=self.delta)
            feature_data.append(transition_data)
        self.feature_data = torch.stack(feature_data)

        target_data, _ = load_processed_transition_data(self.year, self.year+self.future_horizon, resolution=self.resolution, delta=self.delta)
        self.target_data = target_data
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx, 0]
        y = self.data[idx, 1]
        i = int(self.input_px/2)
        o = int(self.output_px/2)

        features = self.feature_data[:, y-i:y+i, x-i:x+i]
        target = torch.sum(self.target_data[y-o:y+o, x-o:x+o])/(self.output_px**2)

        return features.float(), target.float()
    
if __name__ == "__main__":
    train_dataset = DeforestationDataset("train")
    features, target = train_dataset[0]