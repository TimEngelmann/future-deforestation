import os
import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel
from torchmetrics import F1Score, Precision, Recall, AUROC
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt


def get_data_loaders(batch_size=64, num_workers=5, task="pixel", rebalanced_val=False, input_px=50):
    mean  = torch.tensor([4.3101, 3.4019, 3.2393, 6.7158, 4.7855]) # precomputed
    std = torch.tensor([0.9210, 0.5348, 0.5516, 0.9364, 0.9808]) # precomputed
    val_dataset = DeforestationDataset("val", mean=mean, std=std, input_px=input_px,
                                       task=task, rebalanced=rebalanced_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_loader, val_dataset


def predict_model(model_nr, task="pixel", rebalanced_val=False, input_px=50):
    path = f"lightning_logs/version_{model_nr}/"

    val_loader, val_dataset = get_data_loaders(task=task, rebalanced_val=rebalanced_val, input_px=input_px)
    
    model_names = [name for name in os.listdir(path+"checkpoints/") if ".ckpt" in name]
    model_path = path+"checkpoints/"+model_names[-1]

    model = ForestModel.load_from_checkpoint(model_path)
    trainer = pl.Trainer(
        accelerator='mps', 
        devices=1,
        logger=False
    )

    val_predictions = trainer.predict(model, val_loader)

    # concatenate predictions indicated by "preds" key
    predictions = torch.cat([pred["preds"] for pred in val_predictions])
    torch.save(predictions, path + f"val_predictions.pt")

    # concatenate targets indicated by "target" key
    targets = torch.cat([pred["target"] for pred in val_predictions])
    torch.save(targets, path + f"val_targets.pt")

    val_predictions = torch.load(path + f"val_predictions.pt")
    val_targets = torch.load(path + f"val_targets.pt")

    # plot distribution normalized
    plt.hist(val_targets.float(), bins=20, alpha=0.5, color='lightgrey')
    plt.hist(val_predictions, bins=20, alpha=0.5, color='orange')
    plt.legend(["Target", "Prediction"])
    plt.xlabel("Probability of deforestation")
    plt.ylabel("Count")
    plt.savefig(path + "val_hist.png")
    plt.close()

    fpr, tpr, thresholds = roc_curve(val_targets, val_predictions)
    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))

    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    gmeanOpt = round(gmean[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)
    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

    plt.figure()
    plt.plot(fpr,tpr)
    # plt.scatter(fprOpt, tprOpt)
    plt.axis("square")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(path + "val_roc_curve.png")
    plt.close()

    precision, recall, thresholds = precision_recall_curve(val_targets, val_predictions)
    plt.figure()
    plt.plot(recall,precision)
    plt.axis("square")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(path + "val_precision_recall_curve.png")
    plt.close()

    precision = precision[:-10]
    recall = recall[:-10]
    thresholds = thresholds[:-10]

    plt.plot(thresholds, precision[:-1], label="precision")
    plt.plot(thresholds, recall[:-1], label="recall")
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    plt.plot(thresholds, f1, label="f1")
    plt.legend()
    plt.savefig(path + "val_precision_recall_threshold.png")
    plt.close()

    thresholdOpt = float(thresholds[f1.argmax()])
    print("Chosen threshold: ", thresholdOpt)
    precision = Precision(task="binary", threshold=thresholdOpt)
    recall = Recall(task="binary", threshold=thresholdOpt)
    f1 = F1Score(task="binary", threshold=thresholdOpt)
    auroc = AUROC(task="binary")

    valid_metrics = {
        "val_precision": precision(val_predictions, val_targets).item(),
        "val_recall": recall(val_predictions, val_targets).item(),
        "val_f1": f1(val_predictions, val_targets).item(),
        "val_auroc": auroc(val_predictions, val_targets).item()}

    print("----- Validation Metrics -----")
    print(valid_metrics)

    # create new file called metrics.txt and save valid metrics
    with open(path + "metrics.txt", "w") as file:
        file.write("Chosen threshold: " + str(thresholdOpt))
        file.write(str("----- Validation Metrics ----- \n"))
        file.write(str(valid_metrics))

if __name__ == "__main__":
    model_nr = "16521505" # 16508630
    task = "tile_classification"
    rebalanced_val = False
    input_px = 50
    predict_model(model_nr, task, rebalanced_val, input_px)
