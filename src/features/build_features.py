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

def transform_crs(dst_crs, resolution):

    input_paths = []
    output_paths = []

    # biomass data
    for year in [2010,2015]:
        input_paths.append(f"data/raw/biomass/amazonia/{resolution}m/" + f"mapbiomas-brazil-collection-70-amazonia-{year}.tif")
        output_paths.append(f"data/interim/biomass/amazonia/{resolution}m/" + f"biomass_{year}.tif")

    # transition data
    for year_start, year_end in zip([2000,2005,2009,2010,2014,2015],[2010,2010,2010,2015,2015,2020]) :
        input_paths.append(f"data/raw/biomass/amazonia/{resolution}m/" + f"mapbiomas-brazil-collection-70-amazonia-{year_start}_{year_end}.tif")
        output_paths.append(f"data/interim/biomass/amazonia/{resolution}m/" + f"transition_{year_start}_{year_end}.tif")

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
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
                    
# function to load biomass data
def load_biomass_data(year=None, year_end=None, folder="interim", shape=None, resolution=250, delta=None, name=None):
    
    # determine the path to load from from
    path_bio = f"data/{folder}/biomass/amazonia/{resolution}m/" 
    if name is None:
        path_bio = path_bio + f"biomass_{year}.tif" if year_end is None else path_bio + f"transition_{year}_{year_end}.tif"
    else:
        path_bio = path_bio + name + ".tif"

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
                
# function to transform labels to 0=no-transition, 1=transition
def transform_transition_to_labels(transition_data):
    forest_labels = [1,3,4,5,49]
    deforestation_labels = [14,15,18,19,39,20,40,61,41,36,46,47,48,9,21,
                            22,23,24,30,25]
    transition_labels = []
    for forest_label in forest_labels:
        for deforestation_label in deforestation_labels:
            transition_labels.append(100*forest_label+deforestation_label)
    transition_data_new = np.zeros_like(transition_data, dtype=bool)
    for transition_label in transition_labels:
        transition_data_new[transition_data == transition_label] = True
    return transition_data_new

# function to transform labels to 0=forest, 1=natural, 2=farming, 3=urban, 4=water, 255=unknown
def transform_biodata_to_labels(bio_data):
    class_dict = {1:0, 3:0, 4:0, 5:0,49:0, # forest
                10:1,11:1,12:1,32:1,29:1,13:1, 13:1, 50:1, # natural
                14:2,15:2,18:2,19:2,39:2,20:2,40:2,61:2,41:2,36:2,46:2,47:2,48:2,9:2,21:2, # farming
                22:3,23:3,24:3,30:3,25:3, # urban
                26:4,33:4,31:4, # water
                27:255,0:255} # unobserved
    bio_data_new = np.zeros_like(bio_data)
    for key, value in class_dict.items():
        bio_data_new[bio_data == key] = value
    return bio_data_new

def preprocess_data(dst_crs, resolution):
    # will transform all data to target crs and save
    transform_crs(dst_crs, resolution)

    # preprocess bio data
    for year in [2010,2015]:
        bio_data, out_meta = load_biomass_data(year)
        bio_data = transform_biodata_to_labels(bio_data)
        with rasterio.open(f"data/interim/biomass/amazonia/{resolution}m/" + f"biomass_{year}.tif", "w", **out_meta) as dest:
            dest.write(np.expand_dims(bio_data, axis=0))

    # preprocess transition data
    for year_start, year_end in zip([2000,2005,2009,2010,2014,2015],[2010,2010,2010,2015,2015,2020]):
        transition_data, out_meta = load_biomass_data(year_start, year_end)
        transition_data = transform_transition_to_labels(transition_data)
        with rasterio.open(f"data/interim/biomass/amazonia/{resolution}m/" + f"transition_{year_start}_{year_end}.tif", "w", **out_meta) as dest:
            dest.write(np.expand_dims(transition_data, axis=0))

    # missing transition map
    transition_2010_2015, _ = load_biomass_data(2010, 2015)
    transition_2005_2010, out_meta = load_biomass_data(2010, 2015)
    transition_2005_2015 = (transition_2010_2015|transition_2005_2010)
    with rasterio.open(f"data/interim/biomass/amazonia/{resolution}m/" + f"transition_{2005}_{2015}.tif", "w", **out_meta) as dest:
            dest.write(np.expand_dims(transition_2005_2015, axis=0))  

def compute_raster(bio_data_2010, transition_2005_2010, delta):
    height = bio_data_2010.shape[0]
    width = bio_data_2010.shape[1]

    x = np.arange(0.5 * delta, width - 0.5 * delta, delta)
    y = np.arange(0.5 * delta, height - 0.5 * delta, delta)
    xv, yv = np.meshgrid(x, y)

    bbox = []
    for x, y in zip(xv.flatten(), yv.flatten()):
        window_bio = bio_data_2010[int(y-delta/2):int(y+delta/2), int(x-delta/2):int(x+delta/2)]
        forest_cover = np.count_nonzero(window_bio == 0)/delta**2
        empty_cover = np.count_nonzero(window_bio == 255)/delta**2

        # filter clusters that dont have any forest or are mostly empty
        if (forest_cover >= 0) & (empty_cover < 1):
            window_transition_prior = transition_2005_2010[int(y-delta/2):int(y+delta/2), int(x-delta/2):int(x+delta/2)]
            transition_prior = torch.sum(window_transition_prior)/delta**2
            bbox.append((x,y,transition_prior))

    xv, yv, transition_prior = zip(*bbox)

    bins = np.arange(0,np.percentile(transition_prior,99),0.01)
    transition_prior_bin = np.digitize(transition_prior, bins) - 1

    data_grid = np.vstack([xv, yv, transition_prior, transition_prior_bin]).T
    train_data, val_data = train_test_split(data_grid, test_size=0.1, random_state=42, stratify=data_grid[:,-1])

    delete_indices = []
    for point in val_data:
        for x_sign, y_sign in zip([+1,+1,+1,0,0,-1,-1,-1], [+1,0,-1,+1,-1,+1,0,-1]):
            indices = np.where((train_data[:,0] == point[0] - delta * x_sign) & (train_data[:,1] == point[1] - delta * y_sign))[0]
            if len(indices) > 0:
                delete_indices.append(indices[0])

    dropped_data = train_data[np.unique(delete_indices)]
    train_data = np.delete(train_data, delete_indices, axis=0)

    return train_data, val_data, dropped_data

# clusters -> filtered when empty, no forest -> no deforestation possible
# points -> output: filter when empty > 0.8, no-forest | input: filter when empty > 0.8, prior_loss <= 0.01
def filter_empty_points(x,y,bio_data, input_px, output_px):
    window_bio_out = bio_data[int(y-output_px/2):int(y+output_px/2), int(x-output_px/2):int(x+output_px/2)]
    forest_cover_out = np.count_nonzero(window_bio_out == 0)/output_px**2
    empty_cover_out = np.count_nonzero(window_bio_out == 255)/output_px**2
    if (forest_cover_out > 0) & (empty_cover_out < 0.8):
        window_bio_in = bio_data[int(y-input_px/2):int(y+input_px/2), int(x-input_px/2):int(x+input_px/2)]
        empty_cover_in = np.count_nonzero(window_bio_in == 255)/input_px**2
        if empty_cover_in < 0.8:
            return False
    return True

def filter_non_transitioning_points(x,y,transition_prior, input_px):
    window_transition_prior_in = transition_prior[int(y-input_px/2):int(y+input_px/2), int(x-input_px/2):int(x+input_px/2)]
    transition_prior_in = torch.sum(window_transition_prior_in)/input_px**2
    if transition_prior_in > 0.01:
        return False
    return True

def get_points_from_cluster(train_data, bio_data, transition_prior, transition_future, input_px, output_px, delta):
    train_data_points = []
    filtered_empty_px = 0
    filtered_non_transitioning_px = 0
    included_px = 0
    for train_data_cluster in train_data:
        x_cluster = train_data_cluster[0]
        y_cluster = train_data_cluster[1]
        x = np.arange(int(x_cluster-delta/2+output_px/2), int(x_cluster+delta/2+output_px/2), output_px)
        y = np.arange(int(y_cluster-delta/2+output_px/2), int(y_cluster+delta/2+output_px/2), output_px)
        xv, yv = np.meshgrid(x, y)

        for x, y in zip(xv.flatten(), yv.flatten()):

            window_transition_future = transition_future[int(y-output_px/2):int(y+output_px/2), int(x-output_px/2):int(x+output_px/2)]
            transition_future_px = torch.sum(window_transition_future).item()

            if filter_empty_points(x,y,bio_data, input_px, output_px):
                filtered_empty_px += transition_future_px
            elif filter_non_transitioning_points(x,y,transition_prior, input_px):
                filtered_non_transitioning_px += transition_future_px
            else:
                train_data_points.append((x,y,x_cluster,y_cluster, transition_future_px/output_px**2))
                included_px += transition_future_px
    return np.array(train_data_points), (filtered_empty_px, filtered_non_transitioning_px, included_px)

def report_data_statistics(train_filter_metrics, val_filter_metrics, dropped_filter_metrics, 
                           train_data_points, val_data_points, dropped_data_points, 
                           train_data, val_data, dropped_data,
                           transition_2010_2015, bio_data_2010, output_px):
    # print metrics
    filter_metrics = np.array([train_filter_metrics,val_filter_metrics,dropped_filter_metrics])
    print(np.sum(filter_metrics), torch.sum(transition_2010_2015).item())
    print("Disregarded Loss - empty input: {:.2f}%".format(np.sum(filter_metrics[:,0])/np.sum(filter_metrics) * 100))
    print("Disregarded Loss - non transitioning: {:.4f}%".format(np.sum(filter_metrics[:,1])/np.sum(filter_metrics)*100))
    print("train_data: {:.0f} | {:.2f}%".format(len(train_data_points), len(train_data_points)/(len(train_data_points)+len(val_data_points))))
    print("val_data: {:.0f} | {:.2f}%".format(len(val_data_points), len(val_data_points)/(len(train_data_points)+len(val_data_points))))

    # compute layers
    o = int(output_px/2)
    split_layer = np.ones_like(transition_2010_2015) * -1
    target_layer = np.ones(transition_2010_2015.shape) * -1
    for point in train_data_points:
        x = int(point[0])
        y = int(point[1])
        split_layer[y-o:y+o, x-o:x+o] = 0
        target_layer[y-o:y+o, x-o:x+o] = point[-1]
    for point in val_data_points:
        x = int(point[0])
        y = int(point[1])
        split_layer[y-o:y+o, x-o:x+o] = 1
        target_layer[y-o:y+o, x-o:x+o] = point[-1]
    for point in dropped_data_points:
        x = int(point[0])
        y = int(point[1])
        split_layer[y-o:y+o, x-o:x+o] = 2
        target_layer[y-o:y+o, x-o:x+o] = point[-1]

    # plot layers
    fig, axs = plt.subplots(figsize=(15,10))
    cmap = colors.ListedColormap(['white', 'grey'])
    axs.matshow(bio_data_2010 != 255,cmap=cmap,vmin = -.5, vmax = 1.5, alpha=0.5)
    cmap = colors.ListedColormap(['k','red', 'lightblue'])
    cmap.set_under('white')
    axs.matshow(split_layer,cmap=cmap,vmin = -.5, vmax = 2.5, alpha=0.2)
    axs.scatter(val_data[:,0], val_data[:,1], s=1, c='red', alpha=0.8)
    axs.scatter(train_data[:,0], train_data[:,1], s=1, c='k', alpha=0.8)
    axs.scatter(dropped_data[:,0], dropped_data[:,1], s=1, c='lightgrey', alpha=0.8)
    axs.set_aspect('equal', 'box')
    plt.savefig('reports/figures/data_split/train_val_data_split.png')
    plt.close()

    fig, axs = plt.subplots(figsize=(15,10))
    cmap = colors.ListedColormap(['white', 'grey'])
    axs.matshow(bio_data_2010 != 255,cmap=cmap,vmin = -.5, vmax = 1.5, alpha=0.5)
    cmap = mpl.colormaps['viridis']
    cmap.set_under('white', alpha=0)
    axs.matshow(target_layer, vmin = 0, vmax = 0.3, cmap=cmap)
    axs.set_aspect('equal', 'box')
    plt.savefig('reports/figures/data_split/train_val_data_target.png')
    plt.close()

    all_points = np.vstack([train_data_points, val_data_points, dropped_data_points])
    targets = all_points[:,-1]
    bins = np.arange(0,0.3,0.01)
    plt.hist(targets, alpha=0.5, bins=bins, color='blue')
    plt.vlines(targets.mean(), 0, 8000, color='blue', linestyles='--')
    plt.savefig('reports/figures/data_split/train_val_data_hist.png')
    plt.close()

def get_points_from_data(bio_data_2015, transition_prior, transition_future, input_px, output_px):
    x = np.arange(0, bio_data_2015.shape[1], output_px)
    y = np.arange(0, bio_data_2015.shape[0], output_px)
    xv, yv = np.meshgrid(x, y)
    data_points = np.vstack([xv.flatten(), yv.flatten()]).T

    train_data_points = []
    filtered_empty_px = 0
    filtered_non_transitioning_px = 0
    included_px = 0

    for point in data_points:
        x = point[0]
        y = point[1]
        
        window_transition_future = transition_future[int(y-output_px/2):int(y+output_px/2), int(x-output_px/2):int(x+output_px/2)]
        transition_future_px = torch.sum(window_transition_future).item()

        if filter_empty_points(x,y,bio_data_2015, input_px, output_px):
            filtered_empty_px += transition_future_px
        elif filter_non_transitioning_points(x,y,transition_prior, input_px):
            filtered_non_transitioning_px += transition_future_px
        else:
            train_data_points.append((x,y,0,0, transition_future_px/output_px**2))
            included_px += transition_future_px
    return np.array(train_data_points), (filtered_empty_px, filtered_non_transitioning_px, included_px)

def report_test_data_statistics(test_filter_metrics, transition_2015_2020, bio_data_2015, test_data_points, output_px):
    print(np.sum(test_filter_metrics), torch.sum(transition_2015_2020).item())
    print("Disregarded Loss - empty input: {:.2f}%".format(test_filter_metrics[0]/np.sum(test_filter_metrics) * 100))
    print("Disregarded Loss - non transitioning: {:.4f}%".format(test_filter_metrics[1]/np.sum(test_filter_metrics)*100))
    print("test_data: {:.0f}".format(len(test_data_points)))

    # compute layers
    o = int(output_px/2)
    test_split_layer = np.ones_like(transition_2015_2020) * -1
    test_target_layer = np.ones(transition_2015_2020.shape) * -1
    for point in test_data_points:
        x = int(point[0])
        y = int(point[1])
        test_split_layer[y-o:y+o, x-o:x+o] = 0
        test_target_layer[y-o:y+o, x-o:x+o] = point[-1]

    # plot target
    fig, axs = plt.subplots(figsize=(15,10))
    cmap = colors.ListedColormap(['white', 'grey'])
    axs.matshow(bio_data_2015 != 255,cmap=cmap,vmin = -.5, vmax = 1.5, alpha=0.5)
    cmap_2 = mpl.colormaps['viridis']
    cmap_2.set_under('white', alpha=0)
    axs.matshow(test_target_layer, vmin = 0, vmax = 0.3, cmap=cmap_2)
    axs.set_aspect('equal', 'box')
    plt.savefig('reports/figures/data_split/test_data_target.png')
    plt.close()

    # plot hist
    targets = test_data_points[:,-1]
    bins = np.arange(0,0.3,0.01)
    plt.hist(targets, alpha=0.5, bins=bins, color='blue')
    plt.vlines(targets.mean(), 0, 8000, color='blue', linestyles='--')
    plt.savefig('reports/figures/data_split/test_data_hist.png')
    plt.close()

def build_features(output_px=40, input_px=400, resolution=250):

    interim_path = f'data/interim/biomass/amazonia/{resolution}m/'
    if not os.path.exists(interim_path):
        os.makedirs(interim_path)
        dst_crs = 'EPSG:6933'
        preprocess_data(dst_crs, resolution)

    delta = input_px - output_px

    bio_data_2010, _ = load_biomass_data(2010, delta=delta)
    transition_2005_2010, _ = load_biomass_data(2005, 2010, delta=delta)
    transition_2010_2015, _ = load_biomass_data(2010, 2015, delta=delta)

    train_data, val_data, dropped_data = compute_raster(bio_data_2010, transition_2005_2010, delta)

    train_data_points, train_filter_metrics = get_points_from_cluster(train_data, bio_data_2010, transition_2005_2010, transition_2010_2015, input_px, output_px, delta)
    val_data_points, val_filter_metrics = get_points_from_cluster(val_data, bio_data_2010, transition_2005_2010, transition_2010_2015, input_px, output_px, delta)
    dropped_data_points, dropped_filter_metrics = get_points_from_cluster(dropped_data, bio_data_2010, transition_2005_2010, transition_2010_2015, input_px, output_px, delta)

    report_data_statistics(train_filter_metrics, val_filter_metrics, dropped_filter_metrics, 
                           train_data_points, val_data_points, dropped_data_points, 
                           train_data, val_data, dropped_data,
                           transition_2010_2015, bio_data_2010, output_px)

    bio_data_2015, _ = load_biomass_data(2015, None, delta=delta)
    transition_2010_2015, _ = load_biomass_data(2010, 2015, delta=delta)
    transition_2015_2020, _ = load_biomass_data(2015, 2020, delta=delta)

    test_data_points, test_filter_metrics = get_points_from_data(bio_data_2015, transition_2010_2015, transition_2015_2020,  input_px, output_px)

    report_test_data_statistics(test_filter_metrics, transition_2015_2020, bio_data_2015, test_data_points, output_px)

    print("99 Percentile Train Data: ", np.percentile(train_data_points[:,-1],99))
    bins = np.arange(0,np.percentile(train_data_points[:,-1],99),0.01)
    for data_points in [train_data_points, val_data_points, test_data_points]:
        targets_bin = np.digitize(data_points[:,-1], bins) - 1
        data_points[:,-1] = targets_bin

    # save data and biomass as pytorch files
    processed_path = f'data/processed/biomass/amazonia/{resolution}m/'
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    bio_data_folder = [bio_data_name.split(".")[0] for bio_data_name in os.listdir(interim_path) if ".tif" in bio_data_name]
    for bio_data_name in bio_data_folder:
        bio_data, _ = load_biomass_data(name=bio_data_name, delta=delta)
        torch.save(bio_data, f'data/processed/biomass/amazonia/{resolution}m/{bio_data_name}.pt')

    dtype = np.int16 if np.max(bio_data_2010.shape) <= 32767 else np.int32
    torch.save(torch.from_numpy(train_data_points.astype(dtype)), 'data/processed/train_data.pt')
    torch.save(torch.from_numpy(val_data_points.astype(dtype)), 'data/processed/val_data.pt')
    torch.save(torch.from_numpy(test_data_points.astype(dtype)), 'data/processed/test_data.pt')


if __name__ == "__main__":
    output_px = 40
    input_px = 400
    build_features(output_px, input_px)