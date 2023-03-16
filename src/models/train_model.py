import numpy as np
import torch
from deforestation_dataset import DeforestationDataset

def get_data_loaders():
    # load datasets
    train_dataset = DeforestationDataset("train", max_elements=1000)
    val_dataset = DeforestationDataset("val", max_elements=1000)
    test_dataset = DeforestationDataset("test", max_elements=1000)

    # create the train, val and test dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10)

    return train_loader, val_loader, test_loader

def compile_model():
    input_dim=10
    hidden_dim=[128, 64, 64, 32]
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)]
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)]
    padding=[0, 0, 0, 0]
    dropout=0.2
    output_dim=3

    layers = []

    # convolutional blocks
    for i in range(len(hidden_dim)):
        layer_input_dim = input_dim if i == 0 else hidden_dim[i-1]
        layers.append(
            torch.nn.Conv2d(
            layer_input_dim,
            hidden_dim[i],
            kernel_size[i],
            stride=stride[i],
            padding=padding[i],
        ))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm2d(hidden_dim[i]))

    layers.append(torch.nn.MaxPool2d(2, 2, 0))
    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Flatten())

    lin_in = 288 # maybe find way to calculate this
    layers.append(torch.nn.Linear(lin_in, 100))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.BatchNorm1d(100))
    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(100, output_dim))

    model = torch.nn.Sequential(*layers)

    return model

import pytorch_lightning as pl
import torchmetrics

class ForestModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = compile_model()
        self.loss_fn = torch.nn.CrossEntropyLoss() # weights disregarded

        self.auroc = torchmetrics.AUROC(task="binary")
        self.prec = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")


    def forward(self, features):
        logits = self.model(features)
        return logits
    
    def shared_step(self, batch, stage):

        features = batch[0]
        target = batch[1]

        logits = self.forward(features)

        # logits_loss = logits.unsqueeze(1)
        # target_loss = target.unsqueeze(1)
        loss = self.loss_fn(logits[:,0], target[:,0])
        probs = logits.sigmoid()
        preds = (probs > 0.5).float()

        metrics_batch = {}
        for i in range(preds.shape[1]):
            metrics_batch[f"auc_{i}"] = self.auroc(probs[:,i], target[:,i])
            metrics_batch[f"precision_{i}"] = self.prec(preds[:,i], target[:,i])
            metrics_batch[f"recall_{i}"] = self.recall(preds[:,i], target[:,i])
            metrics_batch[f"f1_{i}"] = self.f1(preds[:,i], target[:,i])
        
        metrics_batch["loss"] = loss

        return metrics_batch

    def shared_epoch_end(self, outputs, stage):

        metrics_epoch = {}
        for key, value in outputs[0].items():
            metrics_epoch[f"{stage}_{key}"] = torch.stack([x[key] for x in outputs]).mean()

        self.log_dict(metrics_epoch, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

def train_model():
    train_loader, val_loader, test_loader = get_data_loaders()
    model = ForestModel()
    trainer = pl.Trainer(
        accelerator='mps', 
        devices=1,
        max_epochs=30,
        log_every_n_steps=5
    )

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    train_model()

