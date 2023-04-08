import torch
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import colors
import matplotlib as mpl

from sklearn.metrics import recall_score, f1_score, precision_score, roc_auc_score

def histogram_plot(path, targets, predictions, dataset):
    # histogram plot
    n1, bins, patches = plt.hist(targets, alpha=0.5, bins=np.arange(0,0.3,0.01), color='blue')
    n2, bins, patches = plt.hist(predictions, alpha=0.5, bins=np.arange(0,0.3,0.01), color='orange')
    plt.vlines(targets.mean(), 0, int(np.maximum(np.max(n1), np.max(n2))), color='blue', linestyles='--')
    plt.vlines(predictions.mean(), 0, int(np.maximum(np.max(n1), np.max(n2))), color='orange', linestyles='--')
    plt.savefig(path + f"{dataset}_hist.png")
    plt.close()

def layer_plot(path, target_layer, prediction_layer, dataset, window=None):

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
        plt.savefig(path + f"{dataset}_output.png")
    else:
        plt.savefig(path + f"{dataset}_output_zoom.png")
    plt.close()

def calculate_classification_metrics(predictions, targets, path, dataset):
    thresholds = np.arange(0.01, 0.3, 0.01)
    metrics = []
    for threshold in thresholds:
        predictions_class = predictions > threshold
        targets_class = targets > threshold
        if (np.count_nonzero(targets_class == True) > 0) & (np.count_nonzero(targets_class == True) > 0):
            roc_auc = roc_auc_score(targets_class, predictions_class)
            precision = precision_score(targets_class, predictions_class, zero_division=0)
            recall = recall_score(targets_class, predictions_class)
            f1 = f1_score(targets_class, predictions_class)
        else:
            roc_auc = precision = recall = f1 = 0
        metrics.append([roc_auc, precision, recall, f1])

    roc_auc, precision, recall, f1 =zip(*metrics)
    plt.plot(thresholds, roc_auc)
    plt.plot(thresholds, precision)
    plt.plot(thresholds, recall)
    plt.plot(thresholds, f1)
    plt.legend(['AUC', 'Precision', 'Recall', 'F1'])
    plt.savefig(path + f"{dataset}_classification_metrics.png")
    plt.close()

def visualize(model_nr, dataset="val"):
    path = f"lightning_logs/version_{model_nr}/"

    # get targets
    output_px = 40
    o = int(output_px/2)
    data = torch.load(f'data/processed_transition/{dataset}_data.pt')
    if dataset == "test":
        target_data = torch.load(f"data/processed_transition/biomass/amazonia/250m/transition_2015_2020.pt")
    else:
        target_data = torch.load(f"data/processed_transition/biomass/amazonia/250m/transition_2010_2015.pt")
    targets = []
    for point in data:
        x = point[0]
        y = point[1]
        target = torch.sum(target_data[y-o:y+o, x-o:x+o])/(output_px**2)
        targets.append(target)
    targets = torch.tensor(targets)

    # get predictions
    predictions = torch.load(path + f"{dataset}_predictions.pt")

    histogram_plot(path, targets, predictions, dataset)

    calculate_classification_metrics(predictions, targets, path, dataset)

    # x_point, y_point, x_cluster, y_cluster, target | target | prediction
    data_frame = torch.column_stack([data, targets, predictions])

    target_layer = np.ones(target_data.shape) * -1
    prediction_layer = np.ones(target_data.shape) * -1
    for point in data_frame:
        x = int(point[0])
        y = int(point[1])
        target_layer[y-o:y+o, x-o:x+o] = point[-2]
        prediction_layer[y-o:y+o, x-o:x+o] = point[-1]

    window=[8000,10000,4000,6000]
    layer_plot(path, target_layer, prediction_layer, dataset, window)
    layer_plot(path, target_layer, prediction_layer, dataset)

if __name__ == "__main__":
    model_nr = 5
    visualize(model_nr, "val")