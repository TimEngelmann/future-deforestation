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
import pandas as pd
import plotly.express as px


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
        file_suffix = "-pasture-quality" if data_type == "pasture" else ""
        input_path = f"data/raw/biomass/amazonia/{resolution}m/{data_type}/" + f"mapbiomas-brazil-collection-70{file_suffix}-amazonia-{year}.tif"

        src = rasterio.open(input_path)
        data = src.read(1)

        if data_type == "landuse":
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
    path = f"data/processed/{data_type}/" + f"{data_type}_{year}.tif"
    with rasterio.open(path) as src:
        fill_value = 0
        window = rasterio.windows.Window(offset[0], offset[1], window_size, window_size)
        data = src.read(window=window, fill_value=fill_value, boundless=True)
        data = data.squeeze().astype(np.uint8)
    return data

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

    if len(tiles_to_load) == 0:
        return np.zeros((window_size, window_size), np.float16)

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
    return slope_data_new


# iterate trough raster loading features and target iteratively avoiding memory issues
def iterate_trough_raster(delta=55, window_multiple=500, overlap=5000, dataset="train", filter_px=50):
    past_horizons = [1, 5, 10]
    future_horizon = 1

    mid = int(delta/2)

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

    # create tmp folder if doesnt exist
    if not os.path.exists("data/processed/tmp"):
        os.makedirs("data/processed/tmp")

    filtered_loss = 0
    px_of_interest = 0
    for window_x in range(0, int(width / window_size_small) + 1):
        for window_y in range(0, int(height / window_size_small) + 1):
    # for window_x in range(4, 6):
        # for window_y in range(4, 6):
            print(f"Loading window {window_x}, {window_y}")

            offset_large = [window_x * window_size_small - overlap, window_y * window_size_small - overlap]
            offset_small = [window_x * window_size_small, window_y * window_size_small]

            features = []

            # get past deforestation distances
            current_deforestation = load_window(year, window_size=window_size_large, offset=offset_large, data_type="deforestation")
            deforestation_aggregated = current_deforestation == 4
            for i in range(1, 11):
                if i in past_horizons:
                    if np.count_nonzero(deforestation_aggregated) == 0:
                        features.append(np.ones((window_size_small, window_size_small), dtype=np.float16)*window_size_small)
                    else:
                        deforestation_distance = cv2.distanceTransform((deforestation_aggregated == False).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
                        features.append(deforestation_distance[overlap:-overlap, overlap:-overlap].astype(np.float16))
                    if i == past_horizons[-1]:
                        break
                data = load_window(year-i, window_size=window_size_large, offset=offset_large, data_type="deforestation") == 4
                deforestation_aggregated = deforestation_aggregated | data

            # load current landuse
            landuse = load_window(year, window_size=window_size_large, offset=offset_large, data_type="landuse")

            # get urban distance
            if np.count_nonzero(landuse == 4) == 0:
                features.append(np.ones((window_size_small, window_size_small), dtype=np.float16)*window_size_small)
            else:
                urban_distance = cv2.distanceTransform((landuse != 4).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
                features.append(urban_distance[overlap:-overlap, overlap:-overlap].astype(np.float16))

            # get slope data
            features.append(load_slope_data(path_fabdem_tiles, window_size=window_size_small, offset=offset_small))

            # get class layers
            features.append(landuse[overlap:-overlap, overlap:-overlap])
            features.append(load_window(year, window_size=window_size_small, offset=offset_small, data_type="pasture"))
            features.append(current_deforestation[overlap:-overlap, overlap:-overlap])
            features.append(load_window(year+future_horizon, window_size=window_size_small, offset=offset_small, data_type="deforestation"))

            '''
            for idx, feature in enumerate(features):
                if idx < 5:
                    plt.imshow(feature[mid::delta, mid::delta])
                elif idx == 5:
                    cmap = colors.ListedColormap(['#D5D5E5', '#129912', '#bbfcac', '#ffffb2', '#ea9999', '#0000ff'])
                    plt.imshow(feature[mid::delta, mid::delta], cmap=cmap, vmin=-0.5, vmax=5.5, interpolation='nearest')
                elif idx == 6:
                    cmap = colors.ListedColormap(['#D5D5E5', '#930D03', '#FA9E4F', '#2466A7'])
                    plt.imshow(feature[mid::delta, mid::delta], cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')
                else:
                    cmap = colors.ListedColormap(['#D5D5E5', '#FFFCB5', '#0F5018', '#409562', '#D90016', '#87FF0A', '#FD9407', '#191919'])
                    plt.imshow(feature[mid::delta, mid::delta], cmap=cmap, vmin=-0.5, vmax=7.5, interpolation='nearest')

                plt.show()
            '''

            features = [torch.from_numpy(feature) for feature in features]
            features = torch.stack(features)

            features = features.unfold(1, delta, delta).unfold(2, delta, delta)
            features = features.moveaxis(0,2)

            target = features[:, :, -1, mid, mid]
            current = features[:, :, -2, mid, mid]
            distance = features[:, :, 1, mid, mid] # 0=1 year , 1=5 years, 2=10 years
            filter_points = ((target == 4) | (target == 2)) & (current == 2) & (distance <= filter_px)

            # keep track of filtering
            px_of_interest_batch = np.count_nonzero((target == 4) & (current == 2))
            px_of_interest_batch_kept = np.count_nonzero((target == 4) & (current == 2) & (distance <= filter_px))
            filtered_loss += px_of_interest_batch - px_of_interest_batch_kept
            px_of_interest += px_of_interest_batch

            # indices = filter_points.nonzero() * delta + mid + torch.tensor([offset_small[1], offset_small[0]])
            indices = (filter_points.nonzero() * delta + torch.tensor([offset_small[1], offset_small[0]])) / delta
            data_features = torch.cat([indices, features[filter_points][:, :, mid, mid]], dim=1) # y, x, features
            data_layers = features[filter_points]

            torch.save(data_features, f"data/processed/tmp/data_features_{window_x}_{window_y}.pt")
            torch.save(data_layers, f"data/processed/tmp/data_layers_{window_x}_{window_y}.pt")

    print("percentage of loss dropped: ", 1 - (filtered_loss / px_of_interest))


def plot_data_split(train_features, val_features, delta=55, file_name="data_split.png"):
    print("Deforestation percentage: ", np.count_nonzero(train_features[:, -1] == 4) / train_features.shape[0])
    print("# train points: ", train_features.shape[0])
    print("# val points: ", val_features.shape[0])

    val_x = val_features[:,1].astype(int) * delta + int(delta/2)
    val_y = val_features[:,0].astype(int) * delta + int(delta/2)
    train_x = train_features[:,1].astype(int) * delta + int(delta/2)
    train_y = train_features[:,0].astype(int) * delta + int(delta/2)

    fig, axs = plt.subplots(figsize=(15,10))
    axs.scatter(val_x, val_y, s=0.01, c='red', alpha=1)
    axs.scatter(train_x, train_y, s=0.01, c='k', alpha=1)
    axs.set_aspect('equal', 'box')
    axs.set_title('Train and Validation Data Split')
    axs.legend(['Validation Data', 'Train Data'])
    axs.invert_yaxis()
    plt.savefig(f'reports/figures/{file_name}.png')
    plt.close()


def split_data(data_features, data_layers, delta):
    # split datapoints into train and validation set
    train_features, val_features, train_layers, val_layers = train_test_split(data_features, data_layers,
                                                                              test_size=0.2, random_state=42,
                                                                              stratify=data_features[:, -1])
    plot_data_split(train_features, val_features, delta, file_name="data_split.png")
    return train_features, val_features, train_layers, val_layers


def build_features(dst_crs, resolution, delta, window_multiple, overlap, dataset, filter_px):

    # preprocess_data(dst_crs, resolution)
    # iterate_trough_raster(delta, window_multiple, overlap, dataset, filter_px)

    # load tmp data
    data_features = []
    data_layers = []
    files = os.listdir('data/processed/tmp')
    files.sort()
    for file in files:
        if file.endswith('.pt'):
            if file.startswith('data_features'):
                data_features.append(torch.load(f'data/processed/tmp/{file}'))
            if file.startswith('data_layers'):
                data_layers.append(torch.load(f'data/processed/tmp/{file}'))
                
    data_features = torch.cat(data_features, dim=0).numpy()
    data_layers = torch.cat(data_layers, dim=0).numpy()

    train_features, val_features, train_layers, val_layers = split_data(data_features, data_layers, delta)

    torch.save(torch.from_numpy(train_features), f'data/processed/train_features.pt')
    torch.save(torch.from_numpy(val_features), f'data/processed/val_features.pt')
    torch.save(torch.from_numpy(train_layers), f'data/processed/train_layers.pt')
    torch.save(torch.from_numpy(val_layers), f'data/processed/val_layers.pt')

if __name__ == "__main__":
    dst_crs = "EPSG:6933"
    resolution = 30
    delta = 55
    window_multiple = 200
    overlap = 2000
    dataset = "train"

    filter_px = 50

    build_features(dst_crs, resolution, delta, window_multiple, overlap, dataset, filter_px)