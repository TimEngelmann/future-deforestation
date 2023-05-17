import os
import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel
from torchmetrics import F1Score, Precision, Recall, AUROC, R2Score, MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt


def get_data_loaders(dataset, batch_size=64, num_workers=5, task="pixel", rebalanced_val=False, input_px=50):

    # for 55px width dataset
    mean = torch.tensor([98.2880, 25.6084, 21.9719, 1127.5090, 153.2372])
    std = torch.tensor([163.7474, 17.1673, 16.4770, 921.9048, 129.4803])

    # for 333px width dataset
    # mean = torch.tensor([113.0343, 47.2376, 39.1453, 1114.2917, 151.0670])
    # std = torch.tensor([160.0143, 42.5970, 39.4519, 907.6406, 127.7508])

    val_dataset = DeforestationDataset(dataset, mean=mean, std=std, input_px=input_px,
                                       task=task, rebalanced=rebalanced_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_loader, val_dataset

def plot_hist(targets, predictions, dataset, path, threshold=0.5, suffix=""):
    # plot distribution normalized
    green = "#115E05"
    red = "#F87F78"
    orange = "#FB9E0B"
    if threshold != 0:
        bins = np.linspace(0, 1, 20)
    else:
        bins = np.arange(0, 0.1, 0.001)
    target_bins = plt.hist(targets.float(), bins=bins, alpha=0.5, color='grey', label="Target")
    if threshold != 0:
        plt.hist(predictions[predictions < threshold], bins=bins, alpha=0.5, color=green, label="Prim. Forest")
        plt.hist(predictions[predictions >= threshold], bins=bins, alpha=0.5, color=red, label="Deforested")
    else:
        plt.hist(predictions, bins=bins, alpha=0.5, color=orange, label="Prediction")
    plt.legend()
    plt.ylabel("Count")
    if threshold == 0:
        plt.xlabel("Rate of deforestation")
        plt.ylim(0,10000)
        print(target_bins[0][0])
    else:
        plt.xlabel("Probability of deforestation")
    plt.savefig(path + f"{dataset}_hist{suffix}.png")
    plt.close()

def choose_thresholds(predictions, targets, dataset, path, suffix=""):
    # filter out -1 values
    predictions = predictions[targets != -1].flatten()
    targets = targets[targets != -1].flatten()

    # subsample to 10000
    if len(predictions) > 1000000:
        idx = np.random.choice(len(predictions), 100000, replace=False)
        predictions = predictions[idx]
        targets = targets[idx]

    precision, recall, thresholds = precision_recall_curve(targets, predictions)
    plt.figure()
    plt.plot(recall, precision)
    plt.axis("square")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(path + f"{dataset}_precision_recall_curve{suffix}.png")
    plt.close()

    plt.plot(thresholds, precision[:-1], label="precision")
    plt.plot(thresholds, recall[:-1], label="recall")
    precision_recall = precision[:-1] + recall[:-1]
    precision_recall[precision_recall == 0] = 1
    f1 = 2 * precision[:-1] * recall[:-1] / precision_recall
    plt.plot(thresholds, f1, label="f1")
    plt.legend()
    plt.savefig(path + f"{dataset}_precision_recall_threshold{suffix}.png")
    plt.close()

    threshold = float(thresholds[f1.argmax()])
    plot_hist(targets, predictions, dataset, path, threshold=threshold, suffix=suffix)

    return threshold


def predict_model(path, task="pixel", rebalanced_val=False, input_px=50, suffix="", predict=True):

    model_names = [name for name in os.listdir(path+"checkpoints/") if ".ckpt" in name]
    model_path = path+"checkpoints/"+model_names[-1]

    if predict:
        model = ForestModel.load_from_checkpoint(model_path)
        trainer = pl.Trainer(
            accelerator='mps',
            devices=1,
            logger=False
        )

    metrics = {}
    for dataset in ["val", "test"]:
        if predict:
            data_loader, data_dataset = get_data_loaders(dataset, task=task, rebalanced_val=rebalanced_val, input_px=input_px)
            predictions_dict = trainer.predict(model, data_loader)

            # concatenate predictions indicated by "preds" key
            predictions = torch.cat([pred["preds"] for pred in predictions_dict])
            torch.save(predictions, path+f"{dataset}_predictions{suffix}.pt")

            # concatenate targets indicated by "target" key
            targets = torch.cat([pred["target"] for pred in predictions_dict])
            torch.save(targets, path+f"{dataset}_targets{suffix}.pt")
        else:
            predictions = torch.load(path+f"{dataset}_predictions{suffix}.pt")
            targets = torch.load(path+f"{dataset}_targets{suffix}.pt")

        if task != "tile_regression":
            threshold = choose_thresholds(predictions, targets, dataset, path, suffix=suffix)

            if dataset == "val":
                # chosen_threshold = 0.5
                chosen_threshold = threshold
                print("Chosen threshold: ", chosen_threshold)
                precision = Precision(task="binary", threshold=chosen_threshold, ignore_index=-1)
                recall = Recall(task="binary", threshold=chosen_threshold, ignore_index=-1)
                f1 = F1Score(task="binary", threshold=chosen_threshold, ignore_index=-1)

            dataset_metrics = {
                f"{dataset}_precision": precision(predictions, targets).item(),
                f"{dataset}_recall": recall(predictions, targets).item(),
                f"{dataset}_f1": f1(predictions, targets).item()}
            metrics.update(dataset_metrics)
        else:
            plot_hist(targets, predictions, dataset, path, threshold=0, suffix=suffix)

            r2_score = R2Score()
            mean_squared_error = MeanSquaredError()
            mean_absolute_error = MeanAbsoluteError()
            dataset_metrics = {
                f"{dataset}_r2": r2_score(predictions, targets).item(),
                f"{dataset}_mse": mean_squared_error(predictions, targets).item(),
                f"{dataset}_mae": mean_absolute_error(predictions, targets).item()}
            metrics.update(dataset_metrics)

    # create new file called metrics.txt and save valid metrics
    with open(path + f"metrics{suffix}.txt", "w") as file:
        if task != "tile_regression":
            file.write("Chosen threshold: " + str(chosen_threshold) + "\n")
        file.write(str("----- Validation Metrics ----- \n"))
        file.write(str(metrics))

if __name__ == "__main__":
    task = "tile_regression"
    path = "models/version_16890429/"
    # path = "lightning_logs/version_16763565/"
    rebalanced_val = False
    input_px = 50
    predict_model(path, task, rebalanced_val, input_px, suffix="", predict=True)
