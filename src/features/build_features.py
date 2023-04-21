import numpy as np
import rasterio
import rasterio.mask
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import train_test_split
import torch
import matplotlib as mpl
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from rasterio.merge import merge

# transform bio data
def transform_crs(dst_crs, resolution):

    input_paths = []
    output_paths = []

    # biomass data
    for year in [2010,2015]:
        input_paths.append(f"data/raw/biomass/amazonia/{resolution}m/" + f"mapbiomas-brazil-collection-70-amazonia-{year}.tif")
        output_paths.append(f"data/processed/{resolution}m/landuse/" + f"landuse_{year}.tif")

    for input_path, output_path in zip(input_paths, output_paths):
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open(output_path, 'w', **kwargs, compress="DEFLATE") as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
                    
# function to transform labels to 0=forest, 1=natural, 2=farming, 3=urban, 4=water, 255=unknown
def transform_landuse_to_labels(land_use):
    class_dict = {1:0, 3:0, 4:0, 5:0,49:0, # forest
                10:1,11:1,12:1,32:1,29:1,13:1, 13:1, 50:1, # natural
                14:2,15:2,18:2,19:2,39:2,20:2,40:2,61:2,41:2,36:2,46:2,47:2,48:2,9:2,21:2, # farming
                22:3,23:3,24:3,30:3,25:3, # urban
                26:4,33:4,31:4, # water
                27:255,0:255} # unobserved
    land_use_new = np.zeros_like(land_use).astype(np.uint8)
    for key, value in class_dict.items():
        land_use_new[land_use == key] = value
    return land_use_new

# function to load biomass data
def load_biomass_data(year=None,resolution=30):
    path_bio = f"../data/processed/{resolution}m/landuse/" + f"landuse_{year}.tif"
    with rasterio.open(path_bio) as src:
        out_meta = src.meta
        land_use = src.read(1)
    return land_use, out_meta

# deforestation data
def merge_mosaic(year=None, resolution=30, dst_crs="EPSG:6933"):
    
    path_bio = f"../data/raw/biomass/amazonia/{resolution}m/deforestation/" 
    path_bios = [path_bio + name for name in os.listdir(path_bio) if f"-{year}-" in name]
    
    src_files_to_mosaic = []
    for fp in path_bios:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    mosaic = (mosaic/100).astype(np.uint8)
    
    out_meta = src.meta.copy()
    
    # Get bounds of the merged mosaic
    out_height, out_width = mosaic.shape[1], mosaic.shape[2]
    out_ulx, out_uly = out_trans * (0, 0)
    out_lrx, out_lry = out_trans * (out_width, out_height)
    bounds = (out_ulx, out_lry, out_lrx, out_uly)

    out_meta.update({"driver": "GTiff", 
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans
                     })
    
    transform, width, height = calculate_default_transform(
            src.crs, dst_crs, out_meta['width'], out_meta['height'], *bounds)
    
    out_meta.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "dtype": "uint8",
        "compress": "DEFLATE"
    })
    
    path_out = f"../data/processed/{30}m/deforestation/"  + f"deforestation_{year}.tif"
    with rasterio.open(path_out, "w", **out_meta) as dest:
        reproject(
            source=mosaic[0],
            destination=rasterio.band(dest, 1),
            src_transform=out_trans,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)

def preprocess_data():
    # preprocess bio data
    transform_crs("EPSG:6933", 30)
    for year in [2010,2015]:
        land_use, out_meta = load_biomass_data(year)
        land_use = transform_landuse_to_labels(land_use)
        out_meta.update({"dtype": "uint8", "compress": "DEFLATE"})
        with rasterio.open(f"data/processed/{30}m/landuse/" + f"landuse_{year}.tif", "w", **out_meta) as dest:
            dest.write(np.expand_dims(land_use, axis=0))

    # preprocess deforestation data
    for year in range(1995, 2019):
        merge_mosaic(year)

def compute_raster(year, delta, input_px=35):
    path = f"data/processed/{30}m/deforestation/"  + f"deforestation_{year+1}.tif"
    # Open the mosaic file
    with rasterio.open(path) as src:
        data = src.read()
        data = data.squeeze()

    # only keep every delta pixel
    data = data[::delta,::delta]
    # create dataset with columns x, y, value
    # get x and y coordinates
    x = np.arange(0, data.shape[1]) * delta
    y = np.arange(0, data.shape[0]) * delta
    xv, yv = np.meshgrid(x, y)
    # create array with x, y, value
    data_points = np.vstack((xv.flatten(), yv.flatten(), data.flatten())).T

    # filter datapoints to only keep values that are 2 or 4
    data_points = data_points[(data_points[:,2] == 2) | (data_points[:,2] == 4)]

    # split datapoints into train and validation set
    train_data, val_data = train_test_split(data_points, test_size=0.2, random_state=42, stratify=data_points[:,-1])

    '''
    # downsample train data to have 20% labels 4
    train_data_deforested = train_data[train_data[:,-1] == 4]
    train_data_not_deforested = train_data[train_data[:,-1] == 2]
    train_data_not_deforested = train_data_not_deforested[np.random.choice(train_data_not_deforested.shape[0], train_data_deforested.shape[0] * 4, replace=False)]
    train_data = np.vstack((train_data_deforested, train_data_not_deforested))

    # downsample val data to have 20% labels 4
    val_data_deforested = val_data[val_data[:,-1] == 4]
    val_data_not_deforested = val_data[val_data[:,-1] == 2]
    val_data_not_deforested = val_data_not_deforested[np.random.choice(val_data_not_deforested.shape[0], val_data_deforested.shape[0] * 4, replace=False)]
    val_data_resampled = np.vstack((val_data_deforested, val_data_not_deforested))
    np.random.shuffle(val_data_resampled)
    '''

    # statistics of the dataset
    print("Deforestation percentage: ", np.count_nonzero(data_points[:,-1] == 4)/data_points.shape[0])
    print("# train points: ", train_data.shape)
    print("# val points: ", val_data.shape)
    print("Allowed augmentation: ", int((delta-input_px)/2))

    # plot data split
    # plot layers
    fig, axs = plt.subplots(figsize=(15,10))
    axs.scatter(val_data[:,0], val_data[:,1], s=0.001, c='red', alpha=1)
    axs.scatter(train_data[:,0], train_data[:,1], s=0.1, c='k', alpha=1)
    axs.set_aspect('equal', 'box')
    axs.set_title('Train and Validation Data Split')
    axs.legend(['Validation Data', 'Train Data'])
    plt.savefig('reports/figures/data_split/train_val_data_split.png')
    plt.close()

    return train_data, val_data

def aggregate_deforestation(dataset, past_horizons=[1,5,10]):
    year = 2014 if dataset == "test" else 2009
    path = f"data/processed/{30}m/deforestation/"  + f"deforestation_{year}.tif"
    with rasterio.open(path) as src:
        deforestation_aggregated = src.read(fill_value=0).squeeze() == 4

    for i in range(1,11):
        if i in past_horizons:
            suffix = "_test" if dataset == "test" else ""
            out_path = f"data/processed/{30}m/aggregated/"  + f"aggregated_{i}{suffix}.tif"
            out_meta = src.meta.copy()
            out_meta.update({"dtype": "uint8", "compress": "DEFLATE"}) # "nbits": 1
            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(np.expand_dims(deforestation_aggregated.astype(np.uint8), axis=0))
            if i == past_horizons[-1]:
                break

        path = f"data/processed/{30}m/deforestation/"  + f"deforestation_{year - i}.tif"
        with rasterio.open(path) as src:
            data = src.read(fill_value=0).squeeze() == 4
            deforestation_aggregated = deforestation_aggregated | data

def build_features(output_px=1, input_px=35, delta=50, resolution=30):

    # preprocess_data()

    train_data, val_data = compute_raster(2009, delta, input_px) # on next year

    # get height and width of the image
    with rasterio.open(f"data/processed/{30}m/landuse/" + f"landuse_2010.tif") as src:
        height = src.height
        width = src.width

    dtype = np.int16 if np.maximum(height, width) <= 32767 else np.int32
    torch.save(torch.from_numpy(train_data.astype(dtype)), f'data/processed//{30}m/train_data.pt')
    torch.save(torch.from_numpy(val_data.astype(dtype)), f'data/processed//{30}m/val_data.pt')
    # torch.save(torch.from_numpy(test_data.astype(dtype)), f'data/processed/{30}m/test_data.pt')

    # aggregate features
    # aggregate_deforestation("train")
    # aggregate_deforestation("test")
    

if __name__ == "__main__":
    output_px = 1
    input_px = 35
    delta = 50
    resolution = 30
    build_features(output_px, input_px, delta, resolution)