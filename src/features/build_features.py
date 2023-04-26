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
import cv2


# function to transform labels to 1=forest, 2=natural, 3=farming, 4=urban, 5=water, 0=unknown
def transform_landuse_to_labels(land_use):
    class_dict = {1:1, 3:1, 4:1, 5:1,49:1, # forest
                10:2,11:2,12:2,32:2,29:2, 13:2,50:2, # natural
                14:3,15:3,18:3,19:3,39:3,20:3,40:3,61:3,41:3,36:3,46:3,47:3,48:3,9:3,21:3, # farming
                22:4,23:4,24:4,30:4,25:4, # urban
                26:5,33:5,31:5, # water
                27:0,0:0} # unobserved
    land_use_new = np.zeros_like(land_use).astype(np.uint8)
    for key, value in class_dict.items():
        land_use_new[land_use == key] = value
    return land_use_new

# transform map biomass data
def preprocess_file(year, dst_crs, resolution, data_type="landuse"):
    if data_type == "deforestation":
        input_path = f"data/raw/biomass/amazonia/{resolution}m/deforestation/"
        input_paths = [input_path + name for name in os.listdir(input_path) if f"-{year}-" in name]

        src_files_to_mosaic = []
        for fp in input_paths:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        data, src_transform = merge(src_files_to_mosaic)
        data = (data / 100).astype(np.uint8)

        width = data.shape[2]
        height = data.shape[1]
        bounds = rasterio.transform.array_bounds(height, width, src_transform)

    else:
        file_suffix = "-pasture-quality" if data_type is "pasture" else ""
        input_path = f"data/raw/biomass/amazonia/{resolution}m/{data_type}/" + f"mapbiomas-brazil-collection-70{file_suffix}-amazonia-{year}.tif"

        src = rasterio.open(input_path)
        data = src.read(1)

        if data_type is "landuse":
            data = transform_landuse_to_labels(data)

        width = src.width
        height = src.height
        bounds = src.bounds
        src_transform = src.transform

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, width, height, *bounds, resolution=(resolution, resolution))

    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'dtype': "uint8",
        'compress': "DEFLATE"
    })

    output_path = f"data/processed/{data_type}/" + f"{data_type}_{year}.tif"
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        reproject(
            source=data,
            destination=rasterio.band(dst, 1),
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)

    # close all open src files
    if data_type == "deforestation":
        for src in src_files_to_mosaic:
            src.close()
    else:
        src.close()


# preprocess all data
def preprocess_data(dst_crs="EPSG:6933", resolution=30):

    # preprocess pasture and landuse data
    for year in [2014]:
        preprocess_file(year, dst_crs, resolution, "pasture")
        preprocess_file(year, dst_crs, resolution, "landuse")

    # preprocess deforestation data
    for year in range(1995, 2020):
        preprocess_file(year, dst_crs, 30, "deforestation")


# load window from raster
def load_window(year, window_size=5100, offset=[0,0], data_type="landuse"):
    path = f"../data/processed/{data_type}/" + f"{data_type}_{year}.tif"
    with rasterio.open(path) as src:
        fill_value = 0
        window = rasterio.windows.Window(offset[0], offset[1], window_size, window_size)
        data = src.read(window=window, fill_value=fill_value, boundless=True)
        data = data.squeeze().astype(np.uint8)
    return torch.from_numpy(data)

def load_slope_data(path_fabdem_tiles, window_size=5100, offset=[0,0]):
    bio_data_src = rasterio.open("data/processed/deforestation/deforestation_2009.tif")
    window = rasterio.windows.Window(offset[0], offset[1], window_size, window_size)
    window_bounds = rasterio.windows.bounds(window, bio_data_src.transform)

    # Identify the tiles that intersect with the spatial extent of the window
    tiles_to_load = []
    for path_fabdem in path_fabdem_tiles:
        slope_data_src = rasterio.open(path_fabdem)
        # calculate transformed bounds
        tile_bounds = rasterio.warp.transform_bounds(slope_data_src.crs, bio_data_src.crs, *slope_data_src.bounds)
        # check if tile_bounds has any overlap with window_bounds
        if not rasterio.coords.disjoint_bounds(tile_bounds, window_bounds):
            tiles_to_load.append(slope_data_src)
        else:
            slope_data_src.close()
    slope_data, slope_data_transform = merge(tiles_to_load)

    # Get bounds of the merged mosaic
    width = slope_data.shape[2]
    height = slope_data.shape[1]
    bounds = rasterio.transform.array_bounds(height, width, slope_data_transform)

    resolution = bio_data_src.transform[0]
    transform, width, height = calculate_default_transform(
        slope_data_src.crs, bio_data_src.crs, width, height, *bounds, resolution=(resolution, resolution))

    slope_data_new = np.zeros((height, width), np.float32)
    reproject(
        source=slope_data[0],
        destination=slope_data_new,
        src_transform=slope_data_transform,
        src_crs=slope_data_src.crs,
        dst_transform=transform,
        dst_crs=bio_data_src.crs,
        resampling=Resampling.bilinear)

    window_slope = rasterio.windows.from_bounds(*window_bounds, transform=transform)
    window_slope = [int(round(window_slope.col_off)), int(round(window_slope.row_off)), int(round(window_slope.width)), int(round(window_slope.height))]
    slope_data_new = slope_data_new[window_slope[1]:window_slope[1] + window_slope[3], window_slope[0]:window_slope[0] + window_slope[2]]
    slope_data_new = np.pad(slope_data_new, ((0, window_size - slope_data_new.shape[0]),
                                             (0, window_size - slope_data_new.shape[1])),
                                            mode="constant", constant_values=0)
    slope_data_new = slope_data_new.squeeze().astype(np.float16)
    return torch.from_numpy(slope_data_new)


# iterate trough raster loading features and target iteratively avoiding memory issues
def iterate_trough_raster(delta=55, window_multiple=500, overlap=10000, dataset="train"):
    past_horizons = [1, 5, 10]
    future_horizon = 1

    year = 2014 if dataset == "test" else 2009
    window_size_small = delta * window_multiple
    window_size_large = window_size_small + 2 * overlap

    with rasterio.open(f"data/processed/deforestation/" + f"deforestation_{year}.tif") as src:
        height = src.height
        width = src.width

    # get slope data paths
    path_fabdem_tiles = []
    for root, dirs, files in os.walk("data/raw/fabdem/"):
        for file in files:
            if file.endswith(".tif"):
                path_fabdem_tiles.append(os.path.join(root, file))

    data_points = []
    for window_x in range(0, int(width / window_size_small) + 1):
        for window_y in range(0, int(height / window_size_small) + 1):
            print(f"Loading window {window_x}, {window_y}")

            offset_large = [window_x * window_size_small - overlap, window_y * window_size_small - overlap]
            offset_small = [window_x * window_size_small, window_y * window_size_small]

            features = []

            # get past deforestation distances
            current_deforestation = load_window(year, window_size=window_size_large, offset=offset_large, data_type="deforestation")
            deforestation_aggregated = current_deforestation == 4
            for i in range(1, 11):
                if i in past_horizons:
                    deforestation_distance = cv2.distanceTransform(deforestation_aggregated is False, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
                    features.append(deforestation_distance[overlap:-overlap, overlap:-overlap].astype(torch.float16))
                    if i == past_horizons[-1]:
                        break
                data = load_window(year, window_size=window_size_large, offset=offset_large, data_type="deforestation") == 4
                deforestation_aggregated = deforestation_aggregated | data

            # load current landuse
            landuse = load_window(year, window_size=window_size_large, offset=offset_large, data_type="landuse")

            # get urban distance
            urban_distance = cv2.distanceTransform(landuse != 4, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE).astype(np.float16)
            features.append(urban_distance[overlap:-overlap, overlap:-overlap].astype(torch.float16))

            # get slope data
            features.append(load_slope_data(path_fabdem_tiles, window_size=window_size_small, offset=offset_small))

            # get class layers
            features.append(landuse[overlap:-overlap, overlap:-overlap])
            features.append(load_window(year, window_size=window_size_small, offset=offset_small, data_type="pasture"))
            features.append(current_deforestation[overlap:-overlap, overlap:-overlap])
            features.append(load_window(year+future_horizon, window_size=window_size_small, offset=offset_small, data_type="deforestation"))
            features = torch.stack(features)

            '''
            features = features.unfold(1, delta, delta).unfold(2, delta, delta)
            features = features.moveaxis(0, 2)

            train_indices = train_data[(train_data[:,0] < (offset[0]+window_size)) 
                                 & (train_data[:,1] < (offset[1]+window_size)) 
                                 & (train_data[:,0] > (offset[0]))
                                 & (train_data[:,1] > (offset[1]))]
    
            if len(train_indices) > 0:
                train_indices = ((train_indices[:,:2] - torch.tensor(offset)) / delta).type(torch.int)
                train_features.append(features[train_indices[:,1], train_indices[:,0], :, :, :])

            val_indices = val_data[(val_data[:, 0] < (offset[0] + window_size))
                                   & (val_data[:, 1] < (offset[1] + window_size))
                                   & (val_data[:, 0] > (offset[0]))
                                   & (val_data[:, 1] > (offset[1]))]

            if len(val_indices) > 0:
                val_indices = ((val_indices[:, :2] - torch.tensor(offset)) / delta).type(torch.int)
                val_features.append(features[val_indices[:, 1], val_indices[:, 0], :, :, :])

    # train_features = torch.cat(train_features, dim=0)
    val_features = torch.cat(val_features, dim=0)

    # torch.save(train_features, "../data/processed/30m/train_features.pt")
    torch.save(val_features, "../data/processed/30m/val_features.pt")
    '''


def compute_raster(year, delta, input_px=35):
    path_target = f"data/processed/{30}m/deforestation/"  + f"deforestation_{year+1}.tif"
    with rasterio.open(path_target) as src:
        target = src.read()
        target = target.squeeze().astype(np.uint8)
        target = np.pad(target, ((0, delta - target.shape[0] % delta), (0, delta - target.shape[1] % delta)), mode='constant')
        target = torch.from_numpy(target)

    path_current = f"data/processed/{30}m/deforestation/"  + f"deforestation_{year}.tif"
    with rasterio.open(path_current) as src:
        current = src.read()
        current = current.squeeze().astype(np.uint8)
        current = np.pad(current, ((0, delta - current.shape[0] % delta), (0, delta - current.shape[1] % delta)), mode='constant')
        current = torch.from_numpy(current)

    target = target[int(delta/2)::delta,int(delta/2)::delta]
    current = current[int(delta/2)::delta,int(delta/2)::delta]

    distance = 101
    path_deforestation = f"data/processed/{30}m/aggregated/"  + f"aggregated_{5}.tif"
    with rasterio.open(path_deforestation) as src:
        deforestation = src.read()
        deforestation = deforestation.squeeze().astype(bool)
        deforestation = np.pad(deforestation, ((int((distance-delta)/2), 0), (int((distance-delta)/2), 0)), mode='constant')
        deforestation = np.pad(deforestation, ((0, distance - deforestation.shape[0] % distance), (0, distance - deforestation.shape[1] % distance)), mode='constant')
        deforestation = torch.from_numpy(deforestation)

    # deforestation = torch.sum(deforestation.unfold(0, distance, delta).unfold(1, distance, delta), dim=(2,3)) # 5 year
    filters = torch.ones(1,1,distance,distance).type(torch.uint8)
    deforestation = torch.nn.functional.conv2d(deforestation.unsqueeze(0).type(torch.uint8), filters, stride=delta)[0]

    # create dataset with columns x, y, value
    # get x and y coordinates
    x = np.arange(0, target.shape[1]) * delta + int(delta/2)
    y = np.arange(0, target.shape[0]) * delta + int(delta/2)
    xv, yv = np.meshgrid(x, y)
    # create array with x, y, value
    data_points = np.vstack((xv.flatten(), yv.flatten(), target.flatten(), current.flatten(), deforestation.flatten())).T

    # filter datapoints to only keep values that are 2 or 4
    data_points = data_points[((data_points[:,3] == 2) & ((data_points[:,2] == 2) | (data_points[:,2] == 4)) & (data_points[:,4] > 0))]
    data_points = data_points[:,:-2]

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
    axs.scatter(val_data[:,0], val_data[:,1], s=0.01, c='red', alpha=1)
    axs.scatter(train_data[:,0], train_data[:,1], s=0.01, c='k', alpha=1)
    axs.set_aspect('equal', 'box')
    axs.set_title('Train and Validation Data Split')
    axs.legend(['Validation Data', 'Train Data'])
    axs.invert_yaxis()
    plt.savefig('reports/figures/data_split/train_val_data_split.png')
    plt.close()

    return train_data, val_data


def build_features(output_px=1, input_px=35, delta=50, resolution=30):

    # preprocess_data()

    # train_data, val_data = compute_raster(2009, delta, input_px) # on next year

    '''
    # get height and width of the image
    with rasterio.open(f"data/processed/{30}m/landuse/" + f"landuse_2010.tif") as src:
        height = src.height
        width = src.width

    dtype = np.int16 if np.maximum(height, width) <= 32767 else np.int32
    torch.save(torch.from_numpy(train_data.astype(dtype)), f'data/processed/{30}m/train_data.pt')
    torch.save(torch.from_numpy(val_data.astype(dtype)), f'data/processed/{30}m/val_data.pt')
    # torch.save(torch.from_numpy(test_data.astype(dtype)), f'data/processed/{30}m/test_data.pt')

    # aggregate features
    # aggregate_deforestation("train")
    # aggregate_deforestation("test")
    '''
    

if __name__ == "__main__":
    resolution = 30
    dst_crs = "EPSG:6933"
    preprocess_data(dst_crs, resolution)

    '''
    output_px = 1
    input_px = 35
    delta = 55
    build_features(output_px, input_px, delta, resolution)
    '''