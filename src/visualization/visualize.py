import torch
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import colors
import matplotlib as mpl

def histogram_plot(path, targets, predictions):
    # histogram plot
    plt.hist(targets, alpha=0.5, bins=np.arange(0,0.3,0.01), color='blue')
    plt.vlines(targets.mean(), 0, 6000, color='blue', linestyles='--')
    plt.hist(predictions, alpha=0.5, bins=np.arange(0,0.3,0.01), color='orange')
    plt.vlines(predictions.mean(), 0, 6000, color='orange', linestyles='--')
    plt.savefig(path + "hist.png")
    plt.close()

def layer_plot(path, target_layer, prediction_layer, window=None):

    if window is not None:
        target_layer_plot = target_layer[window[0]:window[1],window[2]:window[3]]
        prediction_layer_plot = prediction_layer[window[0]:window[1],window[2]:window[3]]
    else:
        target_layer_plot = target_layer
        prediction_layer_plot = prediction_layer

    cmap = mpl.colormaps['viridis']
    cmap.set_under('white')

    fig, axs = plt.subplots(1, 2, figsize=(15,10))
    axs = axs.flatten()
    axs[0].matshow(target_layer_plot, vmin = 0, vmax = 0.3, cmap=cmap)
    axs[1].matshow(prediction_layer_plot, vmin = 0, vmax = 0.3, cmap=cmap)
    
    if window is None:
        plt.savefig(path + "output.png")
    else:
        plt.savefig(path + "output_zoom.png")
    plt.close()

def visualize(model_nr):
    path = f"lightning_logs/version_{model_nr}/"

    # get targets
    output_px = 40
    o = int(output_px/2)
    data = torch.load(f'data/processed/test_data.pt')
    target_data = torch.load(f"data/processed/biomass/amazonia/250m/transition_2015_2020.pt")
    targets = []
    for point in data:
        x = point[0]
        y = point[1]
        target = torch.sum(target_data[y-o:y+o, x-o:x+o])/(output_px**2)
        targets.append(target)
    targets = torch.tensor(targets)

    # get predictions
    predictions = torch.load(path + "test_predictions.pt")

    # x_point, y_point, x_cluster, y_cluster, target | target | prediction
    data_frame = torch.column_stack([data, targets, predictions])

    target_layer = np.ones(target_data.shape) * -1
    prediction_layer = np.ones(target_data.shape) * -1
    for point in data_frame:
        x = int(point[0])
        y = int(point[1])
        target_layer[y-o:y+o, x-o:x+o] = point[-2]
        prediction_layer[y-o:y+o, x-o:x+o] = point[-1]

    histogram_plot(path, targets, predictions)

    window=[8000,10000,4000,6000]
    layer_plot(path, target_layer, prediction_layer, window)
    layer_plot(path, target_layer, prediction_layer)

if __name__ == "__main__":
    model_nr = 50
    visualize(model_nr)