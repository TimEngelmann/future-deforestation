import os
import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel
from torchmetrics import F1Score, Precision, Recall, AUROC
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt

def get_data_loaders(batch_size=64, num_workers=5, max_elements=None, do_augmentation=False):
    mean  = torch.tensor([4.3101, 3.4019, 3.2393, 6.7158, 4.7855]) # precomputed
    std = torch.tensor([0.9210, 0.5348, 0.5516, 0.9364, 0.9808]) # precomputed
    val_dataset = DeforestationDataset("val", max_elements=max_elements, mean=mean, std=std, do_augmentation=do_augmentation)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # test_dataset = DeforestationDataset("test", max_elements=max_elements)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_loader, val_dataset

def predict_model(model_nr, do_augmentation=False):
    path = f"lightning_logs/version_{model_nr}/"

    max_elements = 10000 if do_augmentation else None
    val_loader, val_dataset = get_data_loaders(max_elements=max_elements, do_augmentation=do_augmentation)
    
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
    suffix = "_augmented" if do_augmentation else ""
    torch.save(predictions, path + f"val_predictions{suffix}.pt")

    # concatenate targets indicated by "target" key
    targets = torch.cat([pred["target"] for pred in val_predictions])
    torch.save(targets, path + f"val_targets{suffix}.pt")

    # concatenate idx indicated by "idx" key
    idx = torch.cat([pred["idx"] for pred in val_predictions])
    torch.save(idx, path + f"val_idx{suffix}.pt")

    val_predictions = torch.load(path + f"val_predictions{suffix}.pt")
    val_targets = torch.load(path + f"val_targets{suffix}.pt")
    val_idx = torch.load(path + f"val_idx{suffix}.pt")

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

    '''
    test_predictions =  trainer.predict(model, test_loader)
    test_predictions = torch.cat(test_predictions)
    torch.save(test_predictions, path + "test_predictions.pt")
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print("----- Testing Metrics -----")
    print(test_metrics)
    '''

if __name__ == "__main__":
    model_nr = 13
    do_augmentation = True
    predict_model(model_nr, do_augmentation)
