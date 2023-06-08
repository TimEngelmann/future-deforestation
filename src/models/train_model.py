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

    # precalculated mean and std for the dataset
    # mean = torch.tensor([98.2880,   25.6084,   21.9719, 1127.5090,  153.2372])
    # std = torch.tensor([163.7474,  17.1673,  16.4770, 921.9048, 129.4803])
    mean = std = None  # will be calculated within the dataset

    # get the train dataset
    train_dataset = DeforestationDataset("train", max_elements=max_elements, root_path=root_path,
                                         weighted_sampler=weighted_sampler, input_px=input_px, task=task,
                                         rebalanced=rebalanced_train, mean=mean, std=std)

    # allows to sample pixels close to the deforestation line more often (not used for the final experiments)
    if weighted_sampler != "":
        sampler = torch.utils.data.WeightedRandomSampler(weights=train_dataset.weights, num_samples=len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers, sampler=sampler)
    else:
        # get regular train loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # get the validation dataset using train mean and std
    print("Train mean:", train_dataset.mean, "std:", train_dataset.std)
    val_dataset = DeforestationDataset("val", max_elements=max_elements, root_path=root_path,
                                       mean=train_dataset.mean, std=train_dataset.std, input_px=input_px, task=task,
                                       rebalanced=rebalanced_val)

    # get regular validation loader
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

    # get data loaders
    train_loader, val_loader = get_data_loaders(batch_size=64, num_workers=num_workers,
                                                root_path=root_path, weighted_sampler=weighted_sampler,
                                                input_px=input_px, task=task,
                                                rebalanced_train=rebalanced_train, rebalanced_val=rebalanced_val)

    # get model
    model = ForestModel(input_px, lr, loss_fn_weight, architecture, dropout, weight_decay,
                        task=task, alpha=alpha, gamma=gamma, loss_fn=loss_fn)

    # initialize model with checkpoint if init_nr is given
    if init_nr >= 0:
        init_path = f"lightning_logs/version_{init_nr}/checkpoints/"
        checkpoints = [checkpoint for checkpoint in os.listdir(init_path) if ".ckpt" in checkpoint]
        if len(checkpoints) > 0:
            model = ForestModel.load_from_checkpoint(init_path + checkpoints[-1])

    # initialize trainer
    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=1,
        max_epochs=max_epochs,
        log_every_n_steps=10,
        deterministic=False,
        callbacks=[TQDMProgressBar(refresh_rate=100)]
    )

    # train model
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    # config_file = sys.argv[1] # if executing with command line
    config_file = "config"  # if executing with pycharm
    with open(f"configs/{config_file}.json", "r") as cfg:
        hyp = json.load(cfg)
    train_model(init_nr=hyp['init_nr'],  # -1 for training from scratch
                accelerator=hyp['accelerator'],  # 'cuda' or 'mps' or 'cpu'
                max_epochs=hyp['max_epochs'],  # number of epochs
                lr=hyp['lr'],  # learning rate
                architecture=hyp['architecture'],  # 'original' or 'VGG' or 'RESNET' or 'UNet'
                root_path=hyp['root_path'],  # path to the data
                num_workers=hyp['num_workers'],  # number of workers for data loading
                weighted_sampler=hyp['weighted_sampler'],  # 'linear' or 'exponential' or ''
                input_px=hyp['input_px'],  # input size, 50px or 256px
                task=hyp['task'],  # 'pixel' or 'tile_segmentation' or 'tile_classification' or 'tile_regression'
                rebalanced_train=hyp['rebalanced_train'],  # Downsample training set
                rebalanced_val=hyp['rebalanced_val'],  # Downsample validation set
                loss_fn=hyp['loss_fn'],  # 'BCE' or 'FocalLoss' or 'DiceLoss' or 'MSELoss'
                loss_fn_weight=hyp['loss_fn_weight'],  # weight for the loss function
                dropout=hyp['dropout'],  # dropout rate for CNN
                alpha=hyp['alpha'],  # alpha for FocalLoss
                gamma=hyp['gamma'])  # gamma for FocalLoss

