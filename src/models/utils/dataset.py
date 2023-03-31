import numpy as np
import torch
import torchvision
import os
import rasterio

from torch.utils.data import Dataset

# function to load biomass data
def load_biomass_data(year, year_end=None, folder="processed", shape=None, resolution=250, delta=None):
    
    # determine the path to load from from
    path_bio = f"data/{folder}/biomass/amazonia/{resolution}m/" 
    path_bio = path_bio + f"biomass_{year}.tif" if year_end is None else path_bio + f"transition_{year}_{year_end}.tif"

    nodata = 255 if year_end is None else False

    with rasterio.open(path_bio) as src:
        out_meta = src.meta
        if shape is not None:
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
        bio_data = torch.nn.functional.pad(bio_data, (diff_y, diff_y, diff_x, diff_x), value=nodata)
    return bio_data, out_meta

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
        bio_data, _ = load_biomass_data(self.year, resolution=self.resolution, delta=self.delta)
        feature_data.append(bio_data)
    
        for past_horizon in self.past_horizons:
            transition_data, _ = load_biomass_data(self.year - past_horizon, self.year, resolution=self.resolution, delta=self.delta)
            feature_data.append(transition_data)
        self.feature_data = torch.stack(feature_data)

        target_data, _ = load_biomass_data(self.year, self.year+self.future_horizon, resolution=self.resolution, delta=self.delta)
        self.target_data = target_data

        targets_bin = self.data[:,-1]
        bin_count = torch.bincount(targets_bin)
        bin_weights = 1 / (bin_count/torch.sum(bin_count))
        sample_weights = [bin_weights[t] for t in targets_bin]
        self.weights = torch.tensor(sample_weights)

        # helpers
        self.i = torch.tensor(int(input_px/2))
        self.o = torch.tensor(int(output_px/2))
        self.width = target_data.shape[1]
        self.height = target_data.shape[0]
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        x = self.data[idx, 0]
        y = self.data[idx, 1]

        if self.dataset == "train":
            x = x + torch.randint(-self.o,self.o,(1,)).item()
            y = y + torch.randint(-self.o,self.o,(1,)).item()
            x = torch.maximum(x,self.i)
            x = torch.minimum(x,self.width - self.i)
            y = torch.maximum(y,self.i)
            y = torch.minimum(y,self.height - self.i)

        features = self.feature_data[:, y-self.i:y+self.i, x-self.i:x+self.i]
        
        if self.dataset == "train":
            angles = [0,90,180,270]
            angle_idx = torch.randint(0, len(angles),(1,)).item()
            features = torchvision.transforms.functional.rotate(features, angles[angle_idx])

        target = torch.sum(self.target_data[y-self.o:y+self.o, x-self.o:x+self.o])/(self.output_px**2)

        return features.float(), target.float()
    
if __name__ == "__main__":
    train_dataset = DeforestationDataset("train")
    features, target = train_dataset[0]