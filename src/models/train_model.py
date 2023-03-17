import numpy as np
import torch
import pytorch_lightning as pl
import torchmetrics
from dataset import DeforestationDataset

def get_data_loaders(batch_size=64, num_workers=5, max_elements=1000, train=True, val=True, test=True):
    
    train_loader = val_loader = test_loader = None

    if train:
        train_dataset = DeforestationDataset("train", max_elements=max_elements)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if val:
        val_dataset = DeforestationDataset("val", max_elements=max_elements)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if test:
        test_dataset = DeforestationDataset("test", max_elements=max_elements)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def compile_model():
    input_dim=10
    hidden_dim=[128, 64, 64, 32]
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)]
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)]
    padding=[0, 0, 0, 0]
    dropout=0.2
    output_dim=1

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

class ForestModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = compile_model()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        self.auroc = torchmetrics.AUROC(task="binary")
        self.prec = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    def forward(self, features):
        logits = self.model(features)
        return logits
    
    def shared_step(self, batch, stage):

        features = batch[0]
        target = batch[1]

        logits = self.forward(features)
        logits = logits.squeeze()

        loss = self.loss_fn(logits, target)

        probs = logits.sigmoid()
        preds = (probs > 0.5).float()

        metrics_batch = {}
        metrics_batch["loss"] = loss
        metrics_batch[f"auc"] = self.auroc(probs, target)
        metrics_batch[f"precision"] = self.prec(preds, target)
        metrics_batch[f"recall"] = self.recall(preds, target)
        metrics_batch[f"f1"] = self.f1(preds, target)

        '''
        for i in range(preds.shape[1]):
            metrics_batch[f"auc_{i}"] = self.auroc(probs[:,i], target[:,i])
            metrics_batch[f"precision_{i}"] = self.prec(preds[:,i], target[:,i])
            metrics_batch[f"recall_{i}"] = self.recall(preds[:,i], target[:,i])
            metrics_batch[f"f1_{i}"] = self.f1(preds[:,i], target[:,i])
        '''
        if stage == "train":
            self.training_step_outputs.append(metrics_batch)
        if stage == "valid":
            self.validation_step_outputs.append(metrics_batch)
        if stage == "test":
            self.test_step_outputs.append(metrics_batch)

        return loss

    def shared_epoch_end(self, outputs, stage):

        metrics_epoch = {}
        for key, value in outputs[0].items():
            metrics_epoch[f"{stage}_{key}"] = torch.stack([x[key] for x in outputs]).mean()

        self.log_dict(metrics_epoch, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        result = self.shared_epoch_end(outputs, "train")
        self.training_step_outputs.clear() 
        return result

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        result = self.shared_epoch_end(outputs, "valid")
        self.validation_step_outputs.clear() 
        return result

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        result = self.shared_epoch_end(outputs, "test")
        self.test_step_outputs.clear()
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

def train_model():
    train_loader, val_loader, _ = get_data_loaders(max_elements=1000, test=False)
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

