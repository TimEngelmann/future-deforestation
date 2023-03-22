import numpy as np
import os
import rasterio
import rasterio.mask
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

torch.manual_seed(42)

# function to load biomass data
def load_biomass_data(year, shape, resolution=250):
    path_bio = f"data/raw/biomass/{resolution}m/" + f"mapbiomas-brazil-collection-70-acre-{year}.tif"
    with rasterio.open(path_bio) as src:
        if shape is not None:
            bio_data, _ = rasterio.mask.mask(src, shape, crop=True)
        else:
            bio_data = src.read(1)
        bio_data = np.squeeze(bio_data)
    return bio_data

# function to transform labels to 1=forest, 2=non_forest, 0=unknown
def transform_to_labels(bio_data):
    class_dict = {1:0, 3:0, 4:0, 5:0,49:0, # forest
                10:1,11:1,12:1,32:1,29:1,13:1, 13:1, 50:1, # natural
                14:1,15:1,18:1,19:1,39:1,20:1,40:1,61:1,41:1,36:1,46:1,47:1,48:1,9:1,21:1, # farming
                22:1,23:1,24:1,30:1,25:1, # urban
                26:2,33:2,31:2, # water
                27:255,0:255} # unobserved
    bio_data_new = np.zeros_like(bio_data)
    for key, value in class_dict.items():
        bio_data_new[bio_data == key] = value
    return np.array(bio_data_new, dtype=np.uint8)

def preprocess_data(path_bio_processed):
    all_years = np.arange(1985, 2022)

    bio_data = []
    for year in all_years:
        bio_data_year = load_biomass_data(year, None, resolution=resolution)
        bio_data_year = transform_to_labels(bio_data_year)
        bio_data.append(bio_data_year)
    bio_data = np.stack(bio_data)

    # replace unobserved with previously observed
    for i in np.arange(1, bio_data.shape[0]):
        previously_observed = (bio_data[i] == 255) & (bio_data[i-1] != 255)
        bio_data[i, previously_observed] = bio_data[i-1, previously_observed]

    # identify mask to changing labels
    mask = bio_data[0] == 255
    for i in np.arange(1, bio_data.shape[0]):
        mask = mask | ((bio_data[i-1] == 2) & (bio_data[i] == 0)) # mask pixels water -> forest
        mask = mask | ((bio_data[i-1] == 0) & (bio_data[i] == 2)) # mask pixels forest -> water

        if i > 1: # mask forest <-> not forest <-> forest
            mask = mask | ((bio_data[i-2] == 0) & (bio_data[i-1] != 0) & (bio_data[i] == 0)) 

        mask = mask | ((bio_data[i-1] == 1) & (bio_data[i] == 0))  # non-forest -> forest

    torch.save(torch.from_numpy(mask), path_bio_processed + f"biomass_mask.pt")
    
    '''
    # apply mask
    for i in np.arange(1, bio_data.shape[0]):
        bio_data[i, mask] = 255 # replace label of masked pixels
    bio_data[bio_data == 2] = 255 # replace water label with unobserved label
    '''

    # save
    for i, year in enumerate(all_years):
        torch.save(torch.from_numpy(bio_data[i]), path_bio_processed + f"biomass_{year}.pt")

def process_data(start_year, nr_years_train, horizon, resolution, max_patch_size):
    # preprocess data according to West 2020
    path_bio_processed = f"data/processed/biomass/{resolution}m/"
    if not os.path.exists(path_bio_processed + f"biomass_{1985}.pt"):
        preprocess_data(path_bio_processed)

    # parameters to load date
    start_train = start_year # start of training data
    start_target = start_year + nr_years_train  # model has to predict this and following horizon
    start_test = start_target + horizon # model won't see anything after this year

    # load all the different years
    years = np.arange(start_train, start_test+horizon)
        
    bio_data_dict = {}
    for year in years:
        bio_data = torch.load(path_bio_processed + f"biomass_{year}.pt").numpy()
        bio_data_dict[year] = bio_data

    # get coordinates
    x = np.arange(bio_data.shape[1])
    y = np.arange(bio_data.shape[0])
    xv, yv = np.meshgrid(x, y)

    # append val and test target
    data = [xv.flatten(), yv.flatten()]
    for year in np.arange(start_target, start_test+horizon):
        data.append(bio_data_dict[year].flatten())
    
    dtype = np.int16 if np.max(bio_data.shape) <= 32767 else np.int32
    data = np.array(data, dtype=dtype).T
    data = data[bio_data_dict[start_target - 1].flatten() == 0] # ensure forest cover in last input year
    data = data[np.max(data[:,2:],axis=1) <= 1] # filter out future non-observed points

    # enforce minimum distance to border
    r = int(max_patch_size/2)
    data = data[(data[:,0] > r) & (data[:,1] > r)]
    data = data[(data[:,0] < bio_data.shape[1] - r) & (data[:,1] < bio_data.shape[0] - r)]
                
    torch.save(torch.from_numpy(data), 'data/interim/data.pt')

def build_features(start_year, nr_years_train, horizon, resolution, max_patch_size):
    
    # process data if has not been done yet
    if not os.path.exists('data/interim/data.pt'):
        process_data(start_year, nr_years_train, horizon, resolution, max_patch_size)
    
    # data format: [x_idx, y_idx, 2020, 2021]
    data = torch.load('data/interim/data.pt')

    stratify_on = 2 # 2=deforestation in next year
    data_changing = data[data[:,stratify_on] == 1]
    data_non_changing = data[data[:,stratify_on] == 0]
    indices = torch.randperm(data_non_changing.shape[0])
    indices = indices[:int(data_changing.shape[0]*4)]

    data_subsample = torch.cat([
        data_changing, 
        data_non_changing[indices]
        ])
    
    train_data, val_data = train_test_split(data_subsample, test_size=0.2, random_state=42, stratify=data_subsample[:,stratify_on])
    test_data = data_subsample[data_subsample[:,-horizon-1] == 0] # ensure forest cover in last input year
    
    torch.save(train_data, 'data/processed/train_data.pt')
    torch.save(val_data, 'data/processed/val_data.pt')
    torch.save(test_data, 'data/processed/test_data.pt')
    
if __name__ == "__main__":
    start_year = 2010
    nr_years_train = 10
    horizon = 1
    resolution = 30
    max_patch_size = 35
    build_features(start_year, nr_years_train, horizon, resolution, max_patch_size)