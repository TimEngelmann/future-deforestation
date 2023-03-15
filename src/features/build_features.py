import numpy as np
import os
import rasterio
import rasterio.mask
import torch
from sklearn.model_selection import train_test_split

# function to load biomass data
def load_biomass_data(year, shape, resolution=250):
    path_bio = f"data/raw/biomass/{resolution}m/" + f"mapbiomas-brazil-collection-70-acre-{year}.tif"
    with rasterio.open(path_bio) as src:
        if shape is not None:
            bio_data, out_transform = rasterio.mask.mask(src, shape, crop=True)
        else:
            bio_data = src.read(1)
        bio_data = np.squeeze(bio_data)
        out_meta = src.meta
    return bio_data

# function to transform labels to 1=forest, 2=non_forest, 0=unknown
def transform_to_labels(bio_data):
    class_dict = {1:1, 3:1, 4:1, 5:1,49:1, # forest
                10:2,11:2,12:2,32:2,29:2,13:2, 13:2, 50:2, # natural
                14:2,15:2,18:2,19:2,39:2,20:2,40:2,61:2,41:2,36:2,46:2,47:2,48:2,9:2,21:2, # farming
                22:2,23:2,24:2,30:2,25:2, # urban
                26:2,33:2,31:2, # water
                27:0,0:0} # unobserved
    bio_data_new = np.zeros_like(bio_data)
    for key, value in class_dict.items():
        bio_data_new[bio_data == key] = value
    return np.array(bio_data_new, dtype=np.int8)

def process_data(start_year, nr_years_train, horizon, patch_size=400):
    # parameters to load date
    start_train = start_year # start of training data
    start_target = start_year + nr_years_train  # model has to predict this and following horizon
    start_test = start_target + horizon # model won't see anything after this year
    overlap = 0

    # load all the different years
    years = np.arange(start_train, start_test+horizon)

    bio_data_dict = {}
    for year in years:
        bio_data = load_biomass_data(year, None, resolution=30)
        bio_data = transform_to_labels(bio_data)
        bio_data_dict[year] = bio_data

    # calculate deforestation
    deforestation = np.zeros_like(bio_data_dict[start_target - 1], dtype=bool)
    for year in np.arange(start_target, start_target+horizon):
        deforested = bio_data_dict[start_target - 1] - bio_data_dict[year]
        deforestation = deforestation | (deforested < 0)
    bio_data_dict[1111] = deforestation

    # pad image to shape, so that we can split into tiles
    site_tensor = np.array(list(bio_data_dict.values()))
    target_shape = np.array(site_tensor.shape)
    residual = np.array([target_shape[0], patch_size, patch_size]) - target_shape%patch_size
    target_shape = target_shape + residual
    reshaped_site_tensor = np.pad(site_tensor, (
            (0, 0), 
            (0, target_shape[1] - site_tensor.shape[1]), 
            (0, target_shape[2] - site_tensor.shape[2]))
        )

    # to torch and split in patches
    site_tensor = torch.from_numpy(reshaped_site_tensor)
    patches = site_tensor.unfold(1, patch_size, patch_size - overlap)
    patches = patches.unfold(2, patch_size, patch_size - overlap)
    patches = torch.moveaxis(patches, 0, -1)
    patches = torch.flatten(patches, 0, 1)

    # filter out empty and non changing tiles
    patches = patches[patches[:,:,:,0].sum(axis=1).sum(axis=1) > 0] # filter out empty patches
    patches = patches[patches[:,:,:,-1].sum(axis=1).sum(axis=1) > 0] # filter out non changing patches
    patches = patches[:,:,:,:-1] # drop deforestation layer

    torch.save(patches, 'data/interim/patches.pt')

def build_features(start_year, nr_years_train, horizon, patch_size):
    
    # process data if has not been done yet
    if not os.path.exists('data/interim/patches.pt'):
        process_data(start_year, nr_years_train, horizon, patch_size)
    
    patches = torch.load('data/interim/patches.pt')

    # split the dataset into train, val and test sets (val set is only valid of no overlap)
    train_features, val_features, train_labels, val_labels = train_test_split(patches[:,:,:,:nr_years_train], patches[:,:,:,nr_years_train:nr_years_train+horizon], test_size=0.2, random_state=42)
    test_features = patches[:,:,:,horizon:nr_years_train+horizon]
    test_labels = patches[:,:,:,-horizon:]

    # save features and labels
    torch.save(train_features, 'data/processed/train_features.pt')
    torch.save(train_labels, 'data/processed/train_labels.pt')
    torch.save(val_features, 'data/processed/val_features.pt')
    torch.save(val_labels, 'data/processed/val_labels.pt')
    torch.save(test_features, 'data/processed/test_features.pt')
    torch.save(test_labels, 'data/processed/test_labels.pt')

    


if __name__ == "__main__":
    start_year = 2006
    nr_years_train = 10
    horizon = 3
    patch_size = 512
    build_features(start_year, nr_years_train, horizon, patch_size)