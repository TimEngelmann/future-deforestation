import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel
import os
import json
import sys
from pytorch_lightning.callbacks import TQDMProgressBar


def get_data_loaders(batch_size=64, num_workers=8, max_elements=None, output_px=1, input_px=35, root_path="",
                     weighted_sampler="linear", do_augmentation=False, do_rotation=False):
    train_dataset = DeforestationDataset("train", max_elements=max_elements, output_px=output_px, input_px=input_px,
                                         root_path=root_path, weighted_sampler=weighted_sampler,
                                         do_augmentation=do_augmentation, do_rotation=do_rotation)

    if weighted_sampler != "":
        sampler = torch.utils.data.WeightedRandomSampler(weights=train_dataset.weights, num_samples=len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = DeforestationDataset("val", max_elements=max_elements, output_px=output_px, input_px=input_px, root_path=root_path,
                                        mean=train_dataset.mean, std=train_dataset.std, do_augmentation=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def train_model(init_nr=None, 
                max_epochs=30, lr=0.0001,
                loss_fn="BCEWithLogitsLoss",
                architecture="VGG",
                output_px=1, input_px=55,
                accelerator='mps',
                root_path="",
                num_workers=8,
                dropout=0.0,
                weight_decay=0.0,
                weighted_sampler="",
                do_augmentation=False,
                do_rotation=False):
    pl.seed_everything(42, workers=True)

    train_loader, val_loader = get_data_loaders(max_elements=None, output_px=output_px, input_px=input_px,
                                                root_path=root_path, num_workers=num_workers,
                                                weighted_sampler=weighted_sampler,
                                                do_augmentation=do_augmentation, do_rotation=do_rotation)
    
    model = ForestModel(input_px, lr, loss_fn, architecture, dropout, weight_decay)
    if init_nr >= 0:
        init_path = f"lightning_logs/version_{init_nr}/checkpoints/"
        checkpoints = [checkpoint for checkpoint in os.listdir(init_path) if ".ckpt" in checkpoint]
        if len(checkpoints) > 0:
            model = ForestModel.load_from_checkpoint(init_path + checkpoints[-1])

    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=1,
        max_epochs=max_epochs,
        log_every_n_steps=10,
        deterministic=False,
        callbacks=[TQDMProgressBar(refresh_rate=100)]
    )

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    config_file = sys.argv[1]
    # config_file = "config"
    with open(f"configs/{config_file}.json", "r") as cfg:
        hyp = json.load(cfg)
    train_model(init_nr=hyp['init_nr'], 
                accelerator=hyp['accelerator'], 
                max_epochs=hyp['max_epochs'], 
                lr=hyp['lr'], 
                loss_fn=hyp['loss_fn'],
                architecture=hyp['architecture'],
                root_path=hyp['root_path'],
                num_workers=hyp['num_workers'],
                dropout=hyp['dropout'],
                weight_decay=hyp['weight_decay'],
                weighted_sampler=hyp['weighted_sampler'],
                do_augmentation=hyp['do_augmentation'],
                do_rotation=hyp['do_rotation'])

