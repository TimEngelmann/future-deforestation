import torch
import torchvision
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import numpy as np

def load_entire(year, data_type="landuse", dataset="train"):
    suffix = "_test" if dataset == "test" else ""
    path = f"data/processed/{30}m/{data_type}/"  + f"{data_type}_{year}{suffix}.tif"
    # Open the mosaic file
    with rasterio.open(path) as src:
        fill_value = 255 if data_type == "landuse" else 0
        data = src.read(fill_value=fill_value)
        data = data.squeeze()
        data = data.astype(bool) if data_type == "aggregated" else data.astype(np.uint8)
    return torch.from_numpy(data)

class DeforestationDataset(Dataset):
    def __init__(self, dataset, resolution=30, input_px=35, output_px=1, delta=50, max_elements=None, root_path=""):
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

        self.data = torch.load(root_path + f"data/processed/{30}m/{dataset}_data.pt")
        if max_elements is not None:
            self.data = self.data[:max_elements]
        
        # helpers
        self.augmentation = int((delta-input_px)/2)
        self.i = torch.tensor(int(input_px/2))

        # load features
        self.aggregated = []
        for i in self.past_horizons:
            self.aggregated.append(load_entire(i, data_type="aggregated"))
        self.aggregated = torch.stack(self.aggregated)

        self.deforestation_now = load_entire(self.year, data_type="deforestation")
        self.deforestation_future = load_entire(self.year + self.future_horizon, data_type="deforestation")
        
    def __len__(self):
        return self.data.shape[0]
    
    def get_targets(self, x, y):
        '''
        path = self.root_path + f"data/processed/{30}m/deforestation/" + f"deforestation_{self.year + self.future_horizon}.tif"
        with rasterio.open(path) as src:
            # Define the window
            window = Window(x-self.augmentation, y-self.augmentation, 2*self.augmentation+1, 2*self.augmentation+1)
            data = src.read(window=window, boundless=True, fill_value=0, masked=False)
            data = data.squeeze().astype(np.uint8)
        return torch.from_numpy(data)
        '''
        return self.deforestation_future[y-self.augmentation:y+self.augmentation+1, x-self.augmentation:x+self.augmentation+1]
    
    def get_feature(self, x, y, year, data_type="landuse"):
        suffix = "_test" if self.dataset == "test" else ""
        path = self.root_path + f"data/processed/{30}m/{data_type}/"  + f"{data_type}_{year}{suffix}.tif"
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
        '''
        features = []
        for i in self.past_horizons:
            features.append(self.get_feature(x, y, i, data_type="aggregated"))
        features.append(self.get_feature(x, y, self.year, data_type="deforestation"))
        return torch.stack(features)
        '''
        features = []
        features.append(self.aggregated[:, y-int(self.input_px/2):y+int(self.input_px/2)+1, x-int(self.input_px/2):x+int(self.input_px/2)+1])
        features.append(self.deforestation_now[y-int(self.input_px/2):y+int(self.input_px/2)+1, x-int(self.input_px/2):x+int(self.input_px/2)+1].unsqueeze(0))
        return torch.cat(features)

    def __getitem__(self, idx):

        x = self.data[idx, 0].item()
        y = self.data[idx, 1].item()
        target = self.data[idx, 2].item()

        if self.dataset == "train":
            targets = self.get_targets(x, y)
            xi = np.arange(0, targets.shape[1])
            yi = np.arange(0, targets.shape[0])
            xvi, yvi = np.meshgrid(xi, yi)
            target_points = np.vstack((xvi.flatten(), yvi.flatten(), targets.flatten())).T
            target_points = target_points[target_points[:,2] == target]

            # select random point from target points
            target_point = target_points[torch.randint(0, target_points.shape[0],(1,)).item()]
            x = target_point[0] - self.augmentation + x
            y = target_point[1] - self.augmentation + y
        
        target = 1 if target == 4 else 0
        features = self.aggregate_features(x, y)
        
        if self.dataset == "train":
            angles = [0,90,180,270]
            angle_idx = torch.randint(0, len(angles),(1,)).item()
            features = torchvision.transforms.functional.rotate(features, angles[angle_idx])

        return features.float(), torch.tensor(target).float()
    
if __name__ == "__main__":
    train_dataset = DeforestationDataset("train")
    features, target = train_dataset[20]