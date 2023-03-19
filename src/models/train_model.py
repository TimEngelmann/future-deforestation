import numpy as np
import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel

def get_data_loaders(batch_size=64, num_workers=5, max_elements=None):
    train_dataset = DeforestationDataset("train", max_elements=max_elements)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = DeforestationDataset("val", max_elements=max_elements)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def train_model():
    pl.seed_everything(42, workers=True)

    train_loader, val_loader = get_data_loaders(max_elements=5000)
    model = ForestModel()
    trainer = pl.Trainer(
        accelerator='mps', 
        devices=1,
        max_epochs=3,
        log_every_n_steps=5,
        deterministic=False
    )

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    train_model()

