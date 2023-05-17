import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel
import os
import json
import sys
from pytorch_lightning.callbacks import TQDMProgressBar
import torchvision


def get_data_loaders(batch_size=64, num_workers=8, max_elements=None, root_path="",
                     weighted_sampler="linear", input_px=50, task="pixel",
                     rebalanced_train=False, rebalanced_val=False):

    # mean = torch.tensor([98.2880,   25.6084,   21.9719, 1127.5090,  153.2372])
    # std = torch.tensor([163.7474,  17.1673,  16.4770, 921.9048, 129.4803])
    mean = std = None

    train_dataset = DeforestationDataset("train", max_elements=max_elements, root_path=root_path,
                                         weighted_sampler=weighted_sampler, input_px=input_px, task=task,
                                         rebalanced=rebalanced_train, mean=mean, std=std)

    if weighted_sampler != "":
        sampler = torch.utils.data.WeightedRandomSampler(weights=train_dataset.weights, num_samples=len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("Train mean:", train_dataset.mean, "std:", train_dataset.std)
    val_dataset = DeforestationDataset("val", max_elements=max_elements, root_path=root_path,
                                       mean=train_dataset.mean, std=train_dataset.std, input_px=input_px, task=task,
                                       rebalanced=rebalanced_val)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def train_model(init_nr=None, 
                max_epochs=30, lr=0.0001,
                loss_fn_weight=0,
                architecture="original",
                input_px=50,
                accelerator='mps',
                root_path="",
                num_workers=8,
                dropout=0.0,
                weight_decay=0.0,
                weighted_sampler="",
                task="pixel",
                loss_fn="BCE",
                rebalanced_train=False,
                rebalanced_val=False,
                alpha=None,
                gamma=None):
    pl.seed_everything(42, workers=True)

    train_loader, val_loader = get_data_loaders(batch_size=64, num_workers=num_workers,
                                                root_path=root_path, weighted_sampler=weighted_sampler,
                                                input_px=input_px, task=task,
                                                rebalanced_train=rebalanced_train, rebalanced_val=rebalanced_val)
    
    model = ForestModel(input_px, lr, loss_fn_weight, architecture, dropout, weight_decay,
                        task=task, alpha=alpha, gamma=gamma, loss_fn=loss_fn)
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
    # config_file = sys.argv[1]
    config_file = "config"
    with open(f"configs/{config_file}.json", "r") as cfg:
        hyp = json.load(cfg)
    train_model(init_nr=hyp['init_nr'], 
                accelerator=hyp['accelerator'], 
                max_epochs=hyp['max_epochs'], 
                lr=hyp['lr'],
                architecture=hyp['architecture'],
                root_path=hyp['root_path'],
                num_workers=hyp['num_workers'],
                weighted_sampler=hyp['weighted_sampler'],
                input_px=hyp['input_px'],
                task=hyp['task'],
                rebalanced_train=hyp['rebalanced_train'],
                rebalanced_val=hyp['rebalanced_val'],
                loss_fn=hyp['loss_fn'],
                loss_fn_weight=hyp['loss_fn_weight'],
                dropout=hyp['dropout'],
                alpha=hyp['alpha'],
                gamma=hyp['gamma'])

