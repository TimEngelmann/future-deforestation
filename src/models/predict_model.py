import os
import torch
import pytorch_lightning as pl
from utils.dataset import DeforestationDataset
from utils.model import ForestModel

def get_data_loaders(batch_size=64, num_workers=5, max_elements=None):
    val_dataset = DeforestationDataset("val", max_elements=max_elements)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = DeforestationDataset("test", max_elements=max_elements)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_loader, test_loader

def predict_model(model_nr):
    path = f"lightning_logs/version_{model_nr}/"

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
    torch.save(val_predictions, path + "val_predictions.pt")
    valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
    print("----- Validation Metrics -----")
    print(valid_metrics)

    test_predictions =  trainer.predict(model, test_loader)
    test_predictions = torch.cat(test_predictions)
    torch.save(test_predictions, path + "test_predictions.pt")
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print("----- Testing Metrics -----")
    print(test_metrics)

if __name__ == "__main__":
    model_nr = 6
    predict_model(model_nr)
