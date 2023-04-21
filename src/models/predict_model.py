import os
import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel
from torchmetrics import F1Score, Precision, Recall, AUROC
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt

def get_data_loaders(batch_size=64, num_workers=5, max_elements=None):
    val_dataset = DeforestationDataset("val_resampled", max_elements=max_elements)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # test_dataset = DeforestationDataset("test", max_elements=max_elements)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_loader, None

def predict_model(model_nr):
    path = f"lightning_logs/version_{model_nr}/"

    '''
    val_loader, test_loader = get_data_loaders(max_elements=None)
    
    model_names = [name for name in os.listdir(path+"checkpoints/") if ".ckpt" in name]
    model_path = path+"checkpoints/"+model_names[-1]

    model = ForestModel.load_from_checkpoint(model_path)
    trainer = pl.Trainer(
        accelerator='mps', 
        devices=1,
        logger=False
    )

    val_predictions =  trainer.predict(model, val_loader)
    val_predictions = torch.cat(val_predictions)
    torch.save(val_predictions, path + "val_resampled_predictions.pt")
    # valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
    '''

    val_predictions = torch.load(path + "val_resampled_predictions.pt")
    val_targets = torch.load("data/processed/30m/val_resampled_data.pt")[:,-1] == 4

    # plot distribution normalized
    plt.hist(val_targets.float(), bins=20, alpha=0.5, color='lightgrey')
    plt.hist(val_predictions, bins=20, alpha=0.5, color='orange')
    plt.legend(["Target", "Prediction"])
    plt.xlabel("Probability of deforestation")
    plt.ylabel("Count")
    plt.savefig(path + "val_resampled_hist.png")
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
    plt.savefig(path + "val_resampled_roc_curve.png")
    plt.close()

    precision, recall, thresholds = precision_recall_curve(val_targets, val_predictions)
    plt.figure()
    plt.plot(recall,precision)
    plt.axis("square")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(path + "val_resampled_precision_recall_curve.png")
    plt.close()

    thresholds = np.arange(0.0, 1.0, 0.001)
    fscore = np.zeros(shape=(len(thresholds)))

    # Fit the model
    for index, elem in enumerate(thresholds):
        # Corrected probabilities
        val_predictions_class = val_predictions > elem
        # Calculate the f-score
        fscore[index] = f1_score(val_targets, val_predictions_class)

    plt.figure()
    plt.plot(thresholds,fscore)
    plt.axis("square")
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.savefig(path + "val_resampled_f1_score.png")

    thresholdOpt = thresholds[np.argmax(fscore)]
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
    model_nr = 8
    predict_model(model_nr)
