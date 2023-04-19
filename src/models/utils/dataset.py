import torch
import torchvision
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import numpy as np

class DeforestationDataset(Dataset):
    def __init__(self, dataset, resolution=30, input_px=35, output_px=1, delta=50, max_elements=None, root_path=""):
        """
        Args:
            dataset: "train", "val", "test"
        """
        self.dataset = dataset
        self.year = 2015 if dataset == "test" else 2010
        self.past_horizons = [1,5,10]
        self.future_horizon = 1
        self.resolution = resolution
        self.input_px = input_px
        self.root_path = root_path

        self.data = torch.load(root_path + f"data/processed/{dataset}_data.pt")
        
        # helpers
        self.augmentation = int((delta-input_px)/2)
        self.i = torch.tensor(int(input_px/2))
        
    def __len__(self):
        return self.data.shape[0]
    
    def get_target(self, x, y):
        path = self.root_path + f"data/processed/{30}m/deforestation/" + f"deforestation_{self.year + self.future_horizon}.tif"
        with rasterio.open(path) as src:
            # Define the window
            window = ((x, x+1), (y, y+1))
            data = src.read(window=window, boundless=True, fill_value=0, masked=False)
            data = data.squeeze().astype(np.uint8)
        
        if data == 4:
            return 1
        elif data == 2:
            return 0
        else:
            return -1
    
    def get_feature(self, x, y, year, data_type="landuse"):
        path = self.root_path + f"data/processed/{30}m/{data_type}/"  + f"{data_type}_{year}.tif"
        # Open the mosaic file
        with rasterio.open(path) as src:
            # Define the window
            window = Window(x - int(self.input_px/2), y - int(self.input_px/2), self.input_px, self.input_px)
            # Read the data from the window, ensure that it is filled with nodata
            fill_value = 255 if data_type == "landuse" else 0
            data = src.read(window=window, boundless=True, fill_value=fill_value)
            data = data.squeeze().astype(np.uint8)
        return torch.from_numpy(data)

    def aggregate_features(self, x, y):
        # load deforestation data of last ten years
        deforestation_prior = []
        for i in range(10):
            deforestation_prior.append(self.get_feature(x, y, self.year - i, data_type="deforestation") == 4)
        deforestation_prior = torch.stack(deforestation_prior)

        features = []
        # add deforestation of last year to features
        features.append(deforestation_prior[0].type(torch.bool))
        # add deforestation of last 5 years to features
        features.append(torch.sum(deforestation_prior[:5], axis=0, dtype=bool))
        # add deforestation of last 10 years to features
        features.append(torch.sum(deforestation_prior, axis=0, dtype=bool))
        # add landuse data of current year to features
        features.append(self.get_feature(x, y, self.year, data_type="landuse"))
        return torch.stack(features)

    def __getitem__(self, idx):

        x = self.data[idx, 0].item()
        y = self.data[idx, 1].item()
        target = 1 if self.data[idx, 2] == 4 else 0

        if self.dataset == "train":
            target = -1
            while target == -1:
                x = x + torch.randint(-self.augmentation,self.augmentation,(1,)).item()
                y = y + torch.randint(-self.augmentation,self.augmentation,(1,)).item()
                target = self.get_target(x, y)

        features = self.aggregate_features(x, y)
        
        if self.dataset == "train":
            angles = [0,90,180,270]
            angle_idx = torch.randint(0, len(angles),(1,)).item()
            features = torchvision.transforms.functional.rotate(features, angles[angle_idx])

        return features.float(), torch.tensor(target).float()
    
if __name__ == "__main__":
    train_dataset = DeforestationDataset("train")
    features, target = train_dataset[0]