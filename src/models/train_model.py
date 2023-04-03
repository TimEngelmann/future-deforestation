import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel
import os

def get_data_loaders(batch_size=64, num_workers=5, max_elements=None, output_px=40, input_px=400):
    train_dataset = DeforestationDataset("train", max_elements=max_elements, output_px=output_px, input_px=input_px)

    sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights, len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    val_dataset = DeforestationDataset("val", max_elements=max_elements, output_px=output_px, input_px=input_px)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def train_model(init_nr=None, max_epochs=30, output_px=40, input_px=400, accelerator='mps'):
    pl.seed_everything(42, workers=True)

    train_loader, val_loader = get_data_loaders(max_elements=None, output_px=output_px, input_px=input_px)
    model = ForestModel(input_px)
    if init_nr is not None:
        init_path = f"lightning_logs/version_{init_nr}/checkpoints/"
        checkpoints = [checkpoint for checkpoint in os.listdir(init_path) if ".ckpt" in checkpoint]
        if len(checkpoints) > 0:
            model = ForestModel.load_from_checkpoint(init_path + checkpoints[-1])

    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=1,
        max_epochs=max_epochs,
        log_every_n_steps=5,
        deterministic=False
    )

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    init_nr = None
    accelerator ='cuda'
    max_epochs = 100
    train_model(init_nr=init_nr, accelerator=accelerator, max_epochs=max_epochs)

